import sys
import os
import torch
import torch.nn as nn
import math
import unittest
from torch.testing import assert_close


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import MultiHeadSelfAttention

class TestMultiHeadSelfAttention(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)  # For reproducibility

    def test_output_shape(self):
        """Test that output shape matches input shape."""
        d_model = 64
        num_heads = 4
        attn = MultiHeadSelfAttention(d_model, num_heads)
        
        # Test various batch sizes and sequence lengths
        for batch_size in [1, 4]:
            for seq_len in [1, 10, 100]:
                x = torch.randn(batch_size, seq_len, d_model)
                output = attn(x)
                self.assertEqual(output.shape, (batch_size, seq_len, d_model))

    def test_attention_mechanism(self):
        """Test attention mechanism with fixed weights and known input."""
        d_model = 4
        num_heads = 2
        attn = MultiHeadSelfAttention(d_model, num_heads)
        
        # Override weights with known values
        attn.qkv_proj.weight.data.fill_(0.1)
        attn.qkv_proj.bias.data.fill_(0)
        attn.out_proj.weight.data.fill_(0.1)
        attn.out_proj.bias.data.fill_(0)
        
        # Input where all elements are 1
        x = torch.ones(1, 2, d_model)
        output = attn(x)
        
        # Manually compute expected output
        # After qkv_proj: each token becomes 0.1*4*1 = 0.4 repeated 12 times
        qkv = attn.qkv_proj(x).chunk(3, dim=-1)
        q, k, v = [t.view(1, 2, num_heads, d_model//num_heads) for t in qkv]
        
        # Scaled attention scores
        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_model//num_heads)
        # Should be [[[0.4*0.4*2 + ...] = 0.16*2 = 0.32 per element]] -> 0.32 in all positions
        # Softmax over sequence: becomes 0.5 for both tokens
        # Attention output: 0.5*(v1 + v2) = 0.5*(0.4+0.4) = 0.4 per head
        # Concatenated heads: [0.4, 0.4, 0.4, 0.4] per token
        # out_proj: 0.1*(0.4*4) = 0.16 per output element
        
        # Expected output: 0.16 for all elements
        expected = torch.full((1, 2, d_model), 0.16)
        assert_close(output, expected, atol=1e-5, rtol=1e-5)

    def test_attention_weights(self):
        """Test attention weights for identical and different keys/queries."""
        d_model = 8
        num_heads = 2
        attn = MultiHeadSelfAttention(d_model, num_heads)
        
        # Case 1: All tokens identical -> uniform attention
        x = torch.ones(1, 3, d_model)
        output = attn(x)
        qkv = attn.qkv_proj(x).chunk(3, dim=-1)
        q, k, v = [t.view(1, 3, num_heads, d_model//num_heads).transpose(1, 2) for t in qkv]
        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_model//num_heads)
        attn_probs = attn_scores.softmax(dim=-1)
        
        # All attention weights should be 1/3
        assert_close(attn_probs, torch.full_like(attn_probs, 1/3), atol=1e-5, rtol=1e-5)
        
        # Case 2: Different tokens
        x = torch.tensor([[[1.,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0]]]).float()
        output = attn(x)
        qkv = attn.qkv_proj(x).chunk(3, dim=-1)
        q, k, v = [t.view(1, 2, num_heads, d_model//num_heads).transpose(1, 2) for t in qkv]
        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_model//num_heads)
        attn_probs = attn_scores.softmax(dim=-1)
        
        # First token should attend more to itself than to second (unless initialized otherwise)
        # Since weights are random, we just check basic properties
        self.assertTrue(torch.all(attn_probs >= 0))
        self.assertTrue(torch.allclose(attn_probs.sum(dim=-1), torch.ones_like(attn_probs.sum(dim=-1))))

    def test_gradient_flow(self):
        """Test gradients propagate through the module."""
        d_model = 16
        num_heads = 4
        attn = MultiHeadSelfAttention(d_model, num_heads)
        x = torch.randn(2, 5, d_model, requires_grad=True)
        
        output = attn(x)
        loss = output.sum()
        loss.backward()
        
        # Check gradients on input and weights
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(attn.qkv_proj.weight.grad)
        self.assertIsNotNone(attn.out_proj.weight.grad)
        self.assertFalse(torch.isnan(x.grad).any())
        self.assertFalse(torch.isnan(attn.qkv_proj.weight.grad).any())

    def test_dtypes(self):
        """Test different floating point precisions."""
        d_model = 32
        num_heads = 8
        attn = MultiHeadSelfAttention(d_model, num_heads)
        
        for dtype in [torch.float32, torch.float16]:
            attn = attn.to(dtype)
            x = torch.randn(3, 10, d_model, dtype=dtype)
            output = attn(x)
            self.assertEqual(output.dtype, dtype)

if __name__ == '__main__':
    unittest.main()
