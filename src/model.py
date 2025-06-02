import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        qkv = self.qkv_proj(x).chunk(3, dim=-1)  # Split into Q, K, V
        q, k, v = [t.view(B, T, self.num_heads, self.head_dim).transpose(1, 2) for t in qkv]

        attn_scores = (q @ k.transpose(-2, -1)) / self.scale  # Scaled dot-product attention
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_output = (attn_probs @ v).transpose(1, 2).contiguous().view(B, T, C)

        return self.out_proj(attn_output)

# Cross-Attention Block
class MultiCrossAttention(nn.Module):
    """
    Implements multi-head cross-attention.

    Args:
        d_model (int): Model dimensionality.
        num_heads (int): Number of attention heads.

    Example:
        cross_attn = MultiCrossAttention(d_model=512, num_heads=8)
        x = torch.rand(2, 10, 512)  # Key-value input
        q = torch.rand(2, 5, 512)   # Query input
        output = cross_attn(x, q)   # Output shape (2, 5, 512)
    """
    def __init__(self, d_model: int, num_heads: int):
        super(MultiCrossAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.kv_proj = nn.Linear(d_model, d_model * 2)
        self.out_proj = nn.Linear(d_model, d_model)
        self.scale = math.sqrt(self.head_dim)
    
    def forward(self, x: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """
        Computes multi-head cross-attention.

        Args:
            x (torch.Tensor): Key-value input of shape (batch_size, seq_len_kv, d_model).
            q (torch.Tensor): Query input of shape (batch_size, seq_len_q, d_model).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len_q, d_model).
        """
        B, T, C = x.shape
        kv = self.kv_proj(x).chunk(2, dim=-1)

        k, v = [t.view(B, T, self.num_heads, self.head_dim).transpose(1, 2) for t in kv]

        attn_scores = (q @ k.transpose(-2, -1)) / self.scale
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_output = (attn_probs @ v).transpose(1, 2).continguous().view(B, T, C)

        return self.out_proj(attn_output)

# FeedForward Block
class FeedForward(nn.Module):
    """
    Implements a position-wise feed-forward network (FFN) used in Transformer blocks.

    Args:
        d_model (int): The dimensionality of the model.
        d_ff (int): The hidden layer size in the feed-forward network.

    Example:
        ffn = FeedForward(d_model=512, d_ff=2048)
        x = torch.rand(2, 10, 512)
        output = ffn(x)  # Output shape (2, 10, 512)
    """

    def __init__(self, d_model: int, d_ff: int):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
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
    def __init__(self):
        super(ModalityAwareFusion, self).__init__()
