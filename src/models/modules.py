import math

import torch
import torch.nn


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
        self.regressor = nn.Linear(d_model, 1)

    def forward(self, x):
        return self.regressor(x)
