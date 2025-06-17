import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import unittest
from torch.testing import assert_close


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import MultiHeadCrossModalAttention

class TestMultiCrossModalAttention(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.d_model = 512
        self.num_heads = 8
        self.attn = MultiHeadCrossModalAttention(self.d_model, self.num_heads)
        
    def test_output_shape(self):
        """Test output shapes for different input sequences."""
        # Same sequence length
        x = torch.randn(2, 10, self.d_model)
        q = torch.randn(2, 10, self.d_model)
        output = self.attn(x, q)
        self.assertEqual(output.shape, (2, 10, self.d_model))
        
        # Different sequence lengths
        x = torch.randn(3, 15, self.d_model)
        q = torch.randn(3, 5, self.d_model)
        output = self.attn(x, q)
        self.assertEqual(output.shape, (3, 5, self.d_model))
        
        # Larger batch size
        x = torch.randn(5, 20, self.d_model)
        q = torch.randn(5, 8, self.d_model)
        output = self.attn(x, q)
        self.assertEqual(output.shape, (5, 8, self.d_model))
    
    def test_forward_pass(self):
        """Smoke test for forward pass with random inputs."""
        x = torch.randn(1, 7, self.d_model)
        q = torch.randn(1, 3, self.d_model)
        output = self.attn(x, q)
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())
    
    def test_gradient_flow(self):
        """Test gradient propagation through the module."""
        x = torch.randn(2, 12, self.d_model, requires_grad=True)
        q = torch.randn(2, 6, self.d_model, requires_grad=True)
        output = self.attn(x, q)
        
        # Create dummy loss and backprop
        loss = output.sum()
        loss.backward()
        
        # Check gradients
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(q.grad)
        self.assertFalse(torch.isnan(x.grad).any())
        self.assertFalse(torch.isnan(q.grad).any())
    
    def test_multi_head_effectiveness(self):
        """Test if multi-head attention produces non-trivial outputs."""
        # Create identical inputs
        x = torch.randn(1, 5, self.d_model)
        q = x.clone()  # Query same as key/value
        
        # First pass
        output1 = self.attn(x, q)
        
        # Perturb one head's weights slightly
        with torch.no_grad():
            self.attn.q_proj.weight[0] += 0.01
        output2 = self.attn(x, q)
        
        # Outputs should differ
        self.assertGreater(torch.abs(output1 - output2).max().item(), 1e-3)

if __name__ == "__main__":
    unittest.main()
