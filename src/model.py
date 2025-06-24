import math
from enum import Enum
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TempConv(nn.Module):
    def __init__(self, kernel_size: int = 3, channels: int = 1):
        super(TempConv, self).__init__()
        self.conv1d = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(
                f"Input tensor must have 3 dimensions (batch, seq_len, channels), got {x.dim()}"
            )
        x = x.permute(0, 2, 1)
        x = self.conv1d(x)
        x = x.permute(0, 2, 1)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, feature_dim: int, seq_len: int, dropout: float = 0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(seq_len, feature_dim)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        if feature_dim % 2 != 0:
            raise ValueError(f"feature_dim must be even, got {feature_dim}")
        div_term = torch.exp(
            torch.arange(0, feature_dim, 2).float() * (-math.log(10000.0) / feature_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(
                f"Input tensor must have 3 dimensions (batch, seq, feature), got {x.dim()}"
            )
        if x.size(1) > self.pe.size(1):
            self.extend_pe(x.size(1))
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)

    def extend_pe(self, new_len: int):
        if new_len <= self.pe.size(1):
            return
        pe = torch.zeros(new_len, self.pe.size(2))
        position = torch.arange(0, new_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.pe.size(2), 2).float()
            * (-math.log(10000.0) / self.pe.size(2))
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0).to(self.pe.device)


class MultiHeadCrossModalAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super(MultiHeadCrossModalAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.kv_proj = nn.Linear(d_model, d_model * 2)
        self.out_proj = nn.Linear(d_model, d_model)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, context: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        B, T_kv, C = context.shape
        _, T_q, _ = query.shape

        Q = self.q_proj(query)
        KV = self.kv_proj(context).chunk(2, dim=-1)

        K, V = [
            t.view(B, T_kv, self.num_heads, self.head_dim).transpose(1, 2) for t in KV
        ]
        Q = Q.view(B, T_q, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = (Q @ K.transpose(-2, -1)) / self.scale
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_output = (attn_probs @ V).transpose(1, 2).contiguous().view(B, T_q, C)
        return self.out_proj(attn_output)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, num_modalities: int, d_ff: int):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class ModuleType(Enum):
    CrossAttention = 0
    FFN = 1


class ResidualBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        module_type: ModuleType = ModuleType.FFN,
        prenorm: bool = False,
        **kwargs,
    ):
        super(ResidualBlock, self).__init__()
        self.prenorm = prenorm
        self.norm = nn.LayerNorm(d_model)
        if module_type == ModuleType.FFN:
            num_modalities = kwargs["num_modalities"]
            d_ff = kwargs["d_ff"]
            self.module = FeedForward(d_model, num_modalities, d_ff)
        elif module_type == ModuleType.CrossAttention:
            num_heads = kwargs["num_heads"]
            self.module = MultiHeadCrossModalAttention(d_model, num_heads)

    def forward(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if isinstance(self.module, MultiHeadCrossModalAttention):
            if context is None:
                raise ValueError("Context tensor must be provided for CrossAttention.")
            out = self.module(context, x)
        else:
            out = self.module(x)

        if self.prenorm:
            out = x + self.norm(out)
        else:
            out = self.norm(x + out)
        return out


class FusionMode(Enum):
    BI = 0
    X2Y = 1
    Y2X = 2


class Fuser(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_modalities: int,
        num_heads: int = 8,
        device: str = "cuda",
        mode: FusionMode = FusionMode.BI,
        kernel_size: int = 3,
        max_seq_len: int = 1000,
        dropout: float = 0.3,
    ):
        super(Fuser, self).__init__()
        self.mode = mode
        self.device = device
        self.d_ff = 4 * d_model
        self.max_seq_len = max_seq_len

        self.temp_conv_x = TempConv(kernel_size=kernel_size, channels=d_model)
        self.temp_conv_y = TempConv(kernel_size=kernel_size, channels=d_model)
        self.pe_x = PositionalEncoding(
            feature_dim=d_model, seq_len=max_seq_len, dropout=dropout
        )
        self.pe_y = PositionalEncoding(
            feature_dim=d_model, seq_len=max_seq_len, dropout=dropout
        )
        if mode == FusionMode.BI:
            self.cross_x2y = ResidualBlock(
                d_model, ModuleType.CrossAttention, num_heads=num_heads
            )
            self.cross_y2x = ResidualBlock(
                d_model, ModuleType.CrossAttention, num_heads=num_heads
            )
        elif mode == FusionMode.X2Y:
            self.cross = ResidualBlock(
                d_model, ModuleType.CrossAttention, num_heads=num_heads
            )
        elif mode == FusionMode.Y2X:
            self.cross = ResidualBlock(
                d_model, ModuleType.CrossAttention, num_heads=num_heads
            )
        self.ffn = ResidualBlock(
            d_model, ModuleType.FFN, num_modalities=num_modalities, d_ff=self.d_ff
        )

    def _pad_or_truncate(self, x: torch.Tensor, max_len: int) -> torch.Tensor:
        """Pads or truncates a tensor to a specified sequence length."""
        current_len = x.size(1)
        if current_len == max_len:
            return x
        elif current_len > max_len:
            return x[:, :max_len, :]
        else:
            padding_size = max_len - current_len
            padding = (0, 0, 0, padding_size)
            return F.pad(x, padding, "constant", 0)

    def forward(self, x, y):
        x = self._pad_or_truncate(x, self.max_seq_len)
        y = self._pad_or_truncate(y, self.max_seq_len)

        x = x.to(self.device)
        y = y.to(self.device)

        x = self.temp_conv_x(x)
        y = self.temp_conv_y(y)

        x = self.pe_x(x)
        y = self.pe_y(y)

        if self.mode == FusionMode.BI:
            x2y = self.cross_x2y(x=y, context=x)
            y2x = self.cross_y2x(x=x, context=y)
            out = torch.cat([x2y, y2x], dim=1)
        elif self.mode == FusionMode.X2Y:
            out = self.cross(x=y, context=x)
        else:
            out = self.cross(x=x, context=y)

        return self.ffn(out)


class Classifier(nn.Module):
    def __init__(self, d_model: int, num_classes: int):
        super(Classifier, self).__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        B = x.size(0)
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = self.norm(x)
        return self.fc(x[:, 0])
