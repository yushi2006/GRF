import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

# Code adapted from the fairseq repo.


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    d_k = q.size(1)

    scores = torch.matmul(q, k.transpose(-2, -1))

    scaled_scores = scores / math.sqrt(d_k)

    if mask is not None:
        scaled_scores = scaled_scores.masked_fill(mask == 0, -1e9)

    attention_weights = F.softmax(scaled_scores, dim=-1)

    output = torch.matmul(attention_weights, v)

    return (output, attention_weights)


class MultiheadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super(MultiheadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = self.d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q_proj = self.W_q(q)
        k_proj = self.W_k(k)
        v_proj = self.W_v(v)

        q_split = self.split_heads(q_proj)
        k_split = self.split_heads(k_proj)
        v_split = self.split_heads(v_proj)

        attention_output, self.attention_weights = scaled_dot_product_attention(
            q_split, k_split, v_split, mask
        )

        concatenated_output = attention_output.transpose(1, 2).contiguous()

        batch_size, seq_len_q, _, _ = concatenated_output.size()
        concatenated_output = concatenated_output.view(
            batch_size, seq_len_q, self.d_model
        )

        final_output = self.W_o(concatenated_output)

        return final_output
