import math
from typing import Dict

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[: x.size(1), :])


class CrossModalAttentionEncoder(nn.Module):
    def __init__(self, d_model, nhead, d_ffn, num_layers, dropout=0.1):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        decoder_layer_1 = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ffn,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.decoder_1 = nn.TransformerDecoder(decoder_layer_1, num_layers=num_layers)
        decoder_layer_2 = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ffn,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.decoder_2 = nn.TransformerDecoder(decoder_layer_2, num_layers=num_layers)

    def _create_padding_mask(self, lengths, max_len):
        return torch.arange(max_len, device=lengths.device)[None, :] >= lengths[:, None]

    def _pool(self, x, lengths):
        mask = self._create_padding_mask(lengths, x.size(1)).unsqueeze(-1)
        x = x.masked_fill(mask, 0)
        return x.sum(dim=1) / lengths.unsqueeze(1).clamp(min=1)

    def forward(self, x_state, l_state, x_modality, l_modality):
        x_state_pe = self.pos_encoder(x_state)
        x_modality_pe = self.pos_encoder(x_modality)
        state_mask = self._create_padding_mask(l_state, x_state.size(1))
        modality_mask = self._create_padding_mask(l_modality, x_modality.size(1))
        enriched_state = self.decoder_1(
            tgt=x_state_pe,
            memory=x_modality_pe,
            tgt_key_padding_mask=state_mask,
            memory_key_padding_mask=modality_mask,
        )
        enriched_modality = self.decoder_2(
            tgt=x_modality_pe,
            memory=x_state_pe,
            tgt_key_padding_mask=modality_mask,
            memory_key_padding_mask=state_mask,
        )
        pooled_state = self._pool(enriched_state, l_state)
        pooled_modality = self._pool(enriched_modality, l_modality)
        return pooled_state, pooled_modality


class GatedFusionUnit(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.linear_candidate = nn.Linear(d_model * 2, d_model)
        self.linear_gate = nn.Linear(d_model * 2, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, current_state, new_info):
        combined = torch.cat([current_state, new_info], dim=-1)
        update_gate = torch.sigmoid(self.linear_gate(combined))
        candidate_state = torch.tanh(self.linear_candidate(combined))
        new_state = (1 - update_gate) * current_state + update_gate * candidate_state
        return self.norm(self.dropout(new_state))


class SentimentRegressionHead(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x):
        return self.regressor(x)


class HierarchicalFusionModel(nn.Module):
    def __init__(self, config: Dict, modality_info: Dict):
        super().__init__()
        self.config = config
        self.order = config["order"]
        hparams = config["model"]
        d_model = hparams["d_model"]

        self.projectors = nn.ModuleDict(
            {
                name: nn.Linear(modality_info[name].features.shape[-1], d_model)
                for name in modality_info
            }
        )
        self.fusion_encoders = nn.ModuleList(
            [
                CrossModalAttentionEncoder(
                    d_model,
                    hparams["num_heads"],
                    hparams["d_ffn"],
                    hparams["num_layers"],
                    hparams["dropout"],
                )
                for _ in range(len(self.order) - 1)
            ]
        )
        self.gfus = nn.ModuleList(
            [
                GatedFusionUnit(d_model, hparams["dropout"])
                for _ in range(len(self.order) - 1)
            ]
        )
        self.regressor = SentimentRegressionHead(d_model, hparams["dropout"])
        self.auxiliary_regressor = SentimentRegressionHead(d_model, hparams["dropout"])

    def forward(self, *batch):
        # Unpack batch according to the order defined in the collate_fn
        modalities_data = {
            name: (batch[2 * i], batch[2 * i + 1]) for i, name in enumerate(self.order)
        }
        projected = {
            name: self.projectors[name](data[0])
            for name, data in modalities_data.items()
        }

        current_state_seq = projected[self.order[0]]
        current_lengths = modalities_data[self.order[0]][1]
        intermediate_fusion_result = None

        for i in range(len(self.order) - 1):
            next_mod_name = self.order[i + 1]
            state_repr, new_info = self.fusion_encoders[i](
                current_state_seq,
                current_lengths,
                projected[next_mod_name],
                modalities_data[next_mod_name][1],
            )
            fused_state = self.gfus[i](state_repr, new_info)
            if i == 0:
                intermediate_fusion_result = fused_state

            current_state_seq = fused_state.unsqueeze(1)
            current_lengths = torch.ones(fused_state.size(0), device=fused_state.device)

        final_representation = fused_state
        return self.regressor(final_representation), self.auxiliary_regressor(
            intermediate_fusion_result
        )
