import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

# Self-Attention block
class MultiHeadSelfAttention(nn.Module):
    """
    Implements multi-head self-attention as described in the Transformer architecture.

    Args:
        d_model (int): The dimensionality of the model.
        num_heads (int): Number of attention heads.

    Example:
        attn = MultiHeadSelfAttention(d_model=512, num_heads=8)
        x = torch.rand(2, 10, 512)  # Batch size 2, sequence length 10
        output = attn(x)  # Output shape (2, 10, 512)
    """

    def __init__(self, d_model: int, num_heads: int):
        super(MultiHeadSelfAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.qkv_proj = nn.Linear(d_model, d_model * 3)  # Project input to Q, K, V
        self.out_proj = nn.Linear(d_model, d_model)  # Final projection layer
        self.scale = math.sqrt(self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes multi-head self-attention.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        B, T, C = x.shape  # Batch size, sequence length, embedding dim
        QKV = self.qkv_proj(x).chunk(3, dim=-1)  # Split into Q, K, V
        Q, K, V = [t.view(B, T, self.num_heads, self.head_dim).transpose(1, 2) for t in QKV]

        attn_scores = (Q @ K.transpose(-2, -1)) / self.scale  # Scaled dot-product attention
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_output = (attn_probs @ V).transpose(1, 2).contiguous().view(B, T, C)

        return self.out_proj(attn_output)

# Cross-Attention Block
class MultiHeadCrossModalAttention(nn.Module):
    """
    Implements multi-head cross-modal-attention.

    Args:
        d_model (int): Model dimensionality.
        num_heads (int): Number of attention heads.

    Example:
        cross_attn = MultiHeadCrossModalAttention(d_model=512, num_heads=8)
        x = torch.rand(2, 10, 512)  # Key-value input
        q = torch.rand(2, 5, 512)   # Query input
        output = cross_attn(x, q)   # Output shape (2, 5, 512)
    """
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
        """
        Computes multi-head cross-modal-attention.

        Args:
            x (torch.Tensor): Key-value input of shape (batch_size, seq_len_kv, d_model).
            q (torch.Tensor): Query input of shape (batch_size, seq_len_q, d_model).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len_q, d_model).
        """
        B, T, C = x.shape
        Q = self.q_proj(Q)
        Q_T = Q.size(1)
        KV = self.kv_proj(x).chunk(2, dim=-1)

        K, V = [t.view(B, T, self.num_heads, self.head_dim).transpose(1, 2) for t in KV]

        Q = Q.view(B, Q_T, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = (Q @ K.transpose(-2, -1)) / self.scale # (B, num_heads, Q_T, T)
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_output = (attn_probs @ V).transpose(1, 2).contiguous().view(B, Q_T, C) # (B, num_heads, Q_T, head_dim) => (B, Q_T, num_heads, head_dim) => (B, Q_T, C)

        return self.out_proj(attn_output)

# FeedForward Block
class FeedForward(nn.Module):
    """
    Implements a position-wise feed-forward network (FFN) used in Transformer blocks.

    Args:
        d_model (int): The dimensionality of the model.
        num_modalities (int): The number of modalities to fuse
        d_ff (int): The hidden layer size in the feed-forward network.

    Example:
        ffn = FeedForward(d_model=512, d_ff=2048)
        x = torch.rand(2, 10, 512)
        output = ffn(x)  # Output shape (2, 10, 512)
    """

    def __init__(self, d_model: int, num_modalities: int,  d_ff: int):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(num_modalities*d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the feed-forward network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: Output tensor of the same shape.
        """
        return self.fc2(F.gelu(self.fc1(x)))  # GELU activation for non-linearity


class ModalityAwareFusion(nn.Module):
    """
    Implements a Modality-Aware Fusion block combining self-attention, cross-modal attention,
    and feed-forward projection to merge two modalities in a multimodal-aware space.

    Args:
        d_model (int): Dimensionality of the model.
        num_modalities (int): The number of modalities to fuse.
        num_heads (int): Number of attention heads.
        devices (List[str]): List of devices to use for parallel computation (e.g., ["cuda:0", "cuda:1"]).

    Example:
        fusion = ModalityAwareFusion(d_model=512, num_heads=8, devices=["cuda:0", "cuda:1"])
        x = torch.rand(2, 10, 512).to("cuda:0")  # Modality 1
        y = torch.rand(2, 15, 512).to("cuda:1")  # Modality 2
        output = fusion(x, y)  # Output shape (2, min(T_x, T_y), 512)
    """
    def __init__(self, d_model: int, num_modalities: int, num_heads: int, devices: List[str]):
        super(ModalityAwareFusion, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = 4* d_model
        self.devices = devices

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.norm5 = nn.LayerNorm(d_model)
        
        self.self_attn_head1 = MultiHeadSelfAttention(d_model, num_heads)
        self.self_attn_head2 = MultiHeadSelfAttention(d_model, num_heads)
        
        self.cross_modal_attn1 = MultiHeadCrossModalAttention(d_model, num_heads)
        self.cross_modal_attn2 = MultiHeadCrossModalAttention(d_model, num_heads)
        
        self.ffn = FeedForward(d_model, num_modalities, self.d_ff)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Move each modality into another devices for parallel computation
        x = x.to(self.devices[0])
        y = y.to(self.devices[1])

        # Self attention Layers
        x = self.norm1(x +self.self_attn_head1(x)) # (B, T_A, d_model)
        y = self.norm2(y + self.self_attn_head2(y)) # (B, T_B, d_model)

        # Cross-Modal Layers
        x = self.norm3(x + self.cross_modal_attn1(x, y))
        y = self.norm4(y + self.cross_modal_attn2(y, x))

        # Feed-Forward Layers for projecting in a multimodal space
        xy = torch.cat([x, y], axis=-1)
        
        return xy + self.norm5(self.ffn(xy))

