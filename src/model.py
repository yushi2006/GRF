import math
from enum import Enum
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__(); self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model); pe[:, 0::2] = torch.sin(position * div_term); pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(1), :]; return self.dropout(x)

# ==============================================================================
# 3. R-MulT (Recursive Multimodal Transformer) Architecture
# ==============================================================================
class BimodalTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, d_ffn=2048, dropout=0.1):
        super().__init__()
        self.cross_attn_a_to_b = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn_b_to_a = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.self_attn_a = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.self_attn_b = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ffn_a = nn.Sequential(nn.Linear(d_model, d_ffn), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_ffn, d_model))
        self.ffn_b = nn.Sequential(nn.Linear(d_model, d_ffn), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_ffn, d_model))
        self.norm_cross_a, self.norm_cross_b = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.norm_self_a, self.norm_self_b = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.norm_ffn_a, self.norm_ffn_b = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x_a, x_b, mask_a=None, mask_b=None):
        cross_a_out, _ = self.cross_attn_a_to_b(query=x_a, key=x_b, value=x_b, key_padding_mask=mask_b); x_a = self.norm_cross_a(x_a + self.dropout(cross_a_out))
        cross_b_out, _ = self.cross_attn_b_to_a(query=x_b, key=x_a, value=x_a, key_padding_mask=mask_a); x_b = self.norm_cross_b(x_b + self.dropout(cross_b_out))
        self_a_out, _ = self.self_attn_a(query=x_a, key=x_a, value=x_a, key_padding_mask=mask_a); x_a = self.norm_self_a(x_a + self.dropout(self_a_out))
        self_b_out, _ = self.self_attn_b(query=x_b, key=x_b, value=x_b, key_padding_mask=mask_b); x_b = self.norm_self_b(x_b + self.dropout(self_b_out))
        x_a = self.norm_ffn_a(x_a + self.dropout(self.ffn_a(x_a))); x_b = self.norm_ffn_b(x_b + self.dropout(self.ffn_b(x_b)))
        return x_a, x_b

class TransformerEncoder(nn.Module):
    def __init__(self, d_in_a, d_in_b, d_model, nhead, num_layers, d_ffn, dropout=0.1):
        super().__init__()
        self.proj_a = nn.Linear(d_in_a, d_model); self.proj_b = nn.Linear(d_in_b, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.encoder_stack = nn.ModuleList([BimodalTransformerEncoderLayer(d_model, nhead, d_ffn, dropout=0.2) for _ in range(num_layers)])
        self.fusion_layer = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.ReLU(), nn.LayerNorm(d_model))
    def _create_padding_mask(self, lengths, max_len): return torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) >= lengths.unsqueeze(1)
    def forward(self, x_a, lengths_a, x_b, lengths_b):
        mask_a = self._create_padding_mask(lengths_a, x_a.size(1)); mask_b = self._create_padding_mask(lengths_b, x_b.size(1))
        x_a = self.pos_encoder(self.proj_a(x_a)); x_b = self.pos_encoder(self.proj_b(x_b))
        for layer in self.encoder_stack: x_a, x_b = layer(x_a, x_b, mask_a=mask_a, mask_b=mask_b)
        x_a.masked_fill_(mask_a.unsqueeze(-1), 0); x_b.masked_fill_(mask_b.unsqueeze(-1), 0)
        pooled_a = x_a.sum(dim=1) / lengths_a.unsqueeze(1).clamp(min=1); pooled_b = x_b.sum(dim=1) / lengths_b.unsqueeze(1).clamp(min=1)
        return self.fusion_layer(torch.cat([pooled_a, pooled_b], dim=-1))

# ==============================================================================
# 3. MulT (Multimodal Transformer) Architecture
# ==============================================================================
class MultimodalTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, d_ffn, dropout=0.1):
        super().__init__()
        self.ca_t_to_a = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ca_t_to_v = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ca_a_to_t = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ca_a_to_v = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ca_v_to_t = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ca_v_to_a = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.proj_t = nn.Linear(d_model * 2, d_model); self.proj_a = nn.Linear(d_model * 2, d_model); self.proj_v = nn.Linear(d_model * 2, d_model)
        self.self_attn_t = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True); self.self_attn_a = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True); self.self_attn_v = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ffn_t = nn.Sequential(nn.Linear(d_model, d_ffn), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_ffn, d_model)); self.ffn_a = nn.Sequential(nn.Linear(d_model, d_ffn), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_ffn, d_model)); self.ffn_v = nn.Sequential(nn.Linear(d_model, d_ffn), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_ffn, d_model))
        self.norm1_t, self.norm1_a, self.norm1_v = nn.LayerNorm(d_model), nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.norm2_t, self.norm2_a, self.norm2_v = nn.LayerNorm(d_model), nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.norm3_t, self.norm3_a, self.norm3_v = nn.LayerNorm(d_model), nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x_t, x_a, x_v, mask_t, mask_a, mask_v):
        t_to_a, _ = self.ca_t_to_a(x_t, x_a, x_a, key_padding_mask=mask_a); t_to_v, _ = self.ca_t_to_v(x_t, x_v, x_v, key_padding_mask=mask_v)
        x_t = self.norm1_t(x_t + self.dropout(self.proj_t(torch.cat([t_to_a, t_to_v], dim=-1))))
        a_to_t, _ = self.ca_a_to_t(x_a, x_t, x_t, key_padding_mask=mask_t); a_to_v, _ = self.ca_a_to_v(x_a, x_v, x_v, key_padding_mask=mask_v)
        x_a = self.norm1_a(x_a + self.dropout(self.proj_a(torch.cat([a_to_t, a_to_v], dim=-1))))
        v_to_t, _ = self.ca_v_to_t(x_v, x_t, x_t, key_padding_mask=mask_t); v_to_a, _ = self.ca_v_to_a(x_v, x_a, x_a, key_padding_mask=mask_a)
        x_v = self.norm1_v(x_v + self.dropout(self.proj_v(torch.cat([v_to_t, v_to_a], dim=-1))))
        sa_t, _ = self.self_attn_t(x_t, x_t, x_t, key_padding_mask=mask_t); x_t = self.norm2_t(x_t + self.dropout(sa_t))
        sa_a, _ = self.self_attn_a(x_a, x_a, x_a, key_padding_mask=mask_a); x_a = self.norm2_a(x_a + self.dropout(sa_a))
        sa_v, _ = self.self_attn_v(x_v, x_v, x_v, key_padding_mask=mask_v); x_v = self.norm2_v(x_v + self.dropout(sa_v))
        x_t = self.norm3_t(x_t + self.dropout(self.ffn_t(x_t))); x_a = self.norm3_a(x_a + self.dropout(self.ffn_a(x_a))); x_v = self.norm3_v(x_v + self.dropout(self.ffn_v(x_v)))
        return x_t, x_a, x_v

class MULTModel(nn.Module):
    def __init__(self, d_in_t, d_in_a, d_in_v, d_model, nhead, num_layers, d_ffn, dropout=0.1):
        super().__init__()
        self.proj_t = nn.Linear(d_in_t, d_model); self.proj_a = nn.Linear(d_in_a, d_model); self.proj_v = nn.Linear(d_in_v, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.layers = nn.ModuleList([MultimodalTransformerLayer(d_model, nhead, d_ffn, dropout) for _ in range(num_layers)])
        self.fusion_layer = nn.Sequential(nn.Linear(d_model * 3, d_model), nn.ReLU(), nn.LayerNorm(d_model))
    def _create_padding_mask(self, lengths, max_len): return torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) >= lengths.unsqueeze(1)
    def forward(self, x_t, l_t, x_a, l_a, x_v, l_v):
        mask_t = self._create_padding_mask(l_t, x_t.size(1)); mask_a = self._create_padding_mask(l_a, x_a.size(1)); mask_v = self._create_padding_mask(l_v, x_v.size(1))
        x_t = self.pos_encoder(self.proj_t(x_t)); x_a = self.pos_encoder(self.proj_a(x_a)); x_v = self.pos_encoder(self.proj_v(x_v))
        for layer in self.layers: x_t, x_a, x_v = layer(x_t, x_a, x_v, mask_t, mask_a, mask_v)
        x_t.masked_fill_(mask_t.unsqueeze(-1), 0); x_a.masked_fill_(mask_a.unsqueeze(-1), 0); x_v.masked_fill_(mask_v.unsqueeze(-1), 0)
        pooled_t = x_t.sum(dim=1) / l_t.unsqueeze(1).clamp(min=1); pooled_a = x_a.sum(dim=1) / l_a.unsqueeze(1).clamp(min=1); pooled_v = x_v.sum(dim=1) / l_v.unsqueeze(1).clamp(min=1)
        return self.fusion_layer(torch.cat([pooled_t, pooled_a, pooled_v], dim=-1))

class SentimentClassifierHead(nn.Module):
    def __init__(self, d_model, num_classes, dropout=0.1):
        super().__init__(); self.classifier = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Dropout(0.3), nn.Linear(d_model // 2, num_classes))
    def forward(self, mutual_representation): return self.classifier(mutual_representation)
