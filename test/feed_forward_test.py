import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import unittest
from torch.testing import assert_close


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import FeedForward

class TestFeedForward(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)  # For reproducibility

    def test_output_shape(self):
        """Test that output shape matches input shape."""
        d_model = 512
        d_ff = 2048
        ffn = FeedForward(d_model, d_ff)
        
        # Test various batch sizes and sequence lengths
        for batch_size in [1, 4]:
            for seq_len in [1, 10, 100]:
                x = torch.randn(batch_size, seq_len, d_model)
                output = ffn(x)
                self.assertEqual(output.shape, (batch_size, seq_len, d_model))

    def test_forward_pass(self):
        """Test forward pass with fixed weights and known input."""
        d_model = 2
        d_ff = 3
        ffn = FeedForward(d_model, d_ff)
        
        # Set weights and biases to known values
        with torch.no_grad():
            # fc1: input (2) -> output (3)
            ffn.fc1.weight.data = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            ffn.fc1.bias.data = torch.tensor([0.1, 0.2, 0.3])
            
            # fc2: input (3) -> output (2)
            ffn.fc2.weight.data = torch.tensor([[0.5, 0.6, 0.7], [0.8, 0.9, 1.0]])
            ffn.fc2.bias.data = torch.tensor([0.4, 0.5])
        
        # Single token input
        x = torch.tensor([[[1.0, 2.0]]])
        
        # Manual computation:
        # fc1 output = [1*1 + 2*2 + 0.1, 1*3 + 2*4 + 0.2, 1*5 + 2*6 + 0.3]
        #            = [1 + 4 + 0.1, 3 + 8 + 0.2, 5 + 12 + 0.3] = [5.1, 11.2, 17.3]
        # GELU activation (approximate with exact values for test)
        #   GELU(x) ≈ x for positive values in this range
        # fc2 output = [0.5*5.1 + 0.6*11.2 + 0.7*17.3 + 0.4, 
        #              0.8*5.1 + 0.9*11.2 + 1.0*17.3 + 0.5]
        #            = [2.55 + 6.72 + 12.11 + 0.4, 4.08 + 10.08 + 17.3 + 0.5]
        #            = [21.78, 31.96]
        
        expected = torch.tensor([[[21.78, 31.96]]])
        output = ffn(x)
        
        # Use high tolerance since we approximated GELU
        assert_close(output, expected, atol=0.1, rtol=0.01)

    def test_gelu_activation(self):
        """Test that GELU activation is applied correctly."""
        d_model = 4
        d_ff = 8
        ffn = FeedForward(d_model, d_ff)
        
        # Create input where some values will be negative after linear transform
        x = torch.ones(1, 1, d_model)
        
        # Force fc1 to produce negative values
        with torch.no_grad():
            ffn.fc1.weight.data.fill_(-1.0)
            ffn.fc1.bias.data.zero_()
        
        # Forward pass
        intermediate = ffn.fc1(x)
        output = ffn(x)
        
        # Verify GELU was applied (negative inputs should be suppressed)
        gelu_applied = F.gelu(intermediate)
        self.assertTrue(torch.allclose(ffn.fc1(x), intermediate))
        self.assertTrue(torch.allclose(F.gelu(intermediate), gelu_applied))
        self.assertTrue(torch.allclose(ffn.fc2(gelu_applied), output))

    def test_gradient_flow(self):
        """Test gradients propagate through the module."""
        d_model = 16
        d_ff = 32
        ffn = FeedForward(d_model, d_ff)
        x = torch.randn(2, 5, d_model, requires_grad=True)
        
        output = ffn(x)
        loss = output.sum()
        loss.backward()
        
        # Check gradients on input and weights
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(ffn.fc1.weight.grad)
        self.assertIsNotNone(ffn.fc2.weight.grad)
        self.assertFalse(torch.isnan(x.grad).any())
        self.assertFalse(torch.isnan(ffn.fc1.weight.grad).any())

    def test_dtypes(self):
        """Test different floating point precisions."""
        d_model = 32
        d_ff = 64
        ffn = FeedForward(d_model, d_ff)
        
        for dtype in [torch.float32, torch.float16]:
            ffn = ffn.to(dtype)
            x = torch.randn(3, 10, d_model, dtype=dtype)
            output = ffn(x)
            self.assertEqual(output.dtype, dtype)

    def test_parameter_updates(self):
        """Test that parameters are updated during training."""
        d_model = 8
        d_ff = 16
        ffn = FeedForward(d_model, d_ff)
        
        # Save initial parameters
        fc1_weight_before = ffn.fc1.weight.data.clone()
        fc2_weight_before = ffn.fc2.weight.data.clone()
        
        # Simple training step
        optimizer = torch.optim.SGD(ffn.parameters(), lr=0.1)
        x = torch.randn(1, 5, d_model)
        target = torch.randn(1, 5, d_model)
        
        optimizer.zero_grad()
        output = ffn(x)
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        
        # Verify parameters changed
        self.assertFalse(torch.allclose(fc1_weight_before, ffn.fc1.weight.data))
        self.assertFalse(torch.allclose(fc2_weight_before, ffn.fc2.weight.data))

if __name__ == '__main__':
    unittest.main()
