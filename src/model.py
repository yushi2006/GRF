import math
from enum import Enum
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def forward(self, x: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        Q = self.q_proj(Q)
        Q_T = Q.size(1)
        KV = self.kv_proj(x).chunk(2, dim=-1)

        K, V = [t.view(B, T, self.num_heads, self.head_dim).transpose(1, 2) for t in KV]

        Q = Q.view(B, Q_T, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = (Q @ K.transpose(-2, -1)) / self.scale
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_output = (attn_probs @ V).transpose(1, 2).contiguous().view(B, Q_T, C)

        return self.out_proj(attn_output)


# FeedForward Block
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
            out = self.module(x, context)
        else:
            out = self.module(x)

        if self.prenorm:
            out = x + self.norm(out)
        else:
            out = self.norm(x + out)

        return out


class Mode(Enum):
    BI = 0
    X2Y = 1
    Y2X = 2


class ModalityAwareFusion(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_modalities: int,
        num_heads: int,
        devices: List[str],
        mode: Mode = Mode.BI,
    ):
        super().__init__()
        self.mode = mode
        self.devices = devices
        self.d_ff = 4 * d_model

        if mode == Mode.BI:
            self.cross_x2y = ResidualBlock(
                d_model, ModuleType.CrossAttention, num_heads=num_heads
            )
            self.cross_y2x = ResidualBlock(
                d_model, ModuleType.CrossAttention, num_heads=num_heads
            )
        elif mode == Mode.X2Y:
            self.cross = ResidualBlock(
                d_model, ModuleType.CrossAttention, num_heads=num_heads
            )
        elif mode == Mode.Y2X:
            self.cross = ResidualBlock(
                d_model, ModuleType.CrossAttention, num_heads=num_heads
            )

        self.ffn = ResidualBlock(
            d_model, ModuleType.FFN, num_modalities=num_modalities, d_ff=self.d_ff
        )

    def forward(self, x, y):
        x = x.to(self.devices[0])
        y = y.to(self.devices[1])

        if self.mode == Mode.BI:
            x2y = self.cross_x2y(y, x)
            y2x = self.cross_y2x(x, y)
            out = torch.cat([x2y, y2x], dim=-1)
        elif self.mode == Mode.X2Y:
            out = self.cross(Q=y, KV=x)
        else:
            out = self.cross(Q=x, KV=y)

        return self.ffn(out)


class Classifier(nn.Module):
    def __init__(self, d_model: int, num_classes: int):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        B = x.size(0)
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = self.norm(x)
        return self.fc(x[:, 0])
