import unittest

import torch

# Import your modules here (TempConv, PositionalEncoding, etc.)
from src import (
    Classifier,
    FeedForward,
    Fuser,
    FusionMode,
    ModuleType,
    MultiHeadCrossModalAttention,
    PositionalEncoding,
    ResidualBlock,
    TempConv,
)


class TestTempConv(unittest.TestCase):
    def test_output_shape(self):
        batch_size, seq_len, channels = 32, 100, 64
        kernel_size = 3
        temp_conv = TempConv(kernel_size=kernel_size, channels=channels)
        x = torch.randn(batch_size, seq_len, channels)
        out = temp_conv(x)
        self.assertEqual(out.shape, (batch_size, seq_len, channels))

    def test_invalid_input_dim(self):
        temp_conv = TempConv()
        with self.assertRaises(ValueError):
            x = torch.randn(32, 100)
            temp_conv(x)


class TestPositionalEncoding(unittest.TestCase):
    def test_output_shape(self):
        feature_dim, seq_len, batch_size = 64, 100, 32
        pe = PositionalEncoding(feature_dim, seq_len, dropout=0.0)
        x = torch.randn(batch_size, seq_len, feature_dim)
        out = pe(x)
        self.assertEqual(out.shape, (batch_size, seq_len, feature_dim))

    def test_positional_values(self):
        feature_dim, seq_len = 64, 100
        pe = PositionalEncoding(feature_dim=feature_dim, seq_len=seq_len, dropout=0.0)
        x = torch.zeros(1, seq_len, feature_dim)
        out = pe(x)
        self.assertTrue(torch.allclose(out, pe.pe, rtol=1e-5, atol=1e-5))
        self.assertFalse(torch.allclose(out[0, 0], out[0, 1], rtol=1e-4, atol=1e-6))

    def test_extend_pe(self):
        feature_dim, initial_seq_len, new_seq_len = 64, 50, 100
        pe = PositionalEncoding(
            feature_dim=feature_dim, seq_len=initial_seq_len, dropout=0.0
        )
        x = torch.randn(1, new_seq_len, feature_dim)
        out = pe(x)
        self.assertEqual(out.shape, (1, new_seq_len, feature_dim))
        self.assertEqual(pe.pe.size(1), new_seq_len)

    def test_odd_feature_dim(self):
        with self.assertRaises(ValueError):
            PositionalEncoding(feature_dim=65, seq_len=100)


class TestMultiHeadCrossModalAttention(unittest.TestCase):
    def test_output_shape(self):
        d_model, num_heads, batch_size = 64, 8, 32
        seq_len_kv, seq_len_q = 100, 50

        mha = MultiHeadCrossModalAttention(d_model=d_model, num_heads=num_heads)
        context = torch.randn(batch_size, seq_len_kv, d_model)
        query = torch.randn(batch_size, seq_len_q, d_model)
        out = mha(context, query)

        self.assertEqual(out.shape, (batch_size, seq_len_q, d_model))


class TestFeedForward(unittest.TestCase):
    def test_output_shape(self):
        d_model, num_modalities, d_ff = 64, 2, 256
        batch_size, seq_len = 32, 100
        ff = FeedForward(d_model=d_model, num_modalities=num_modalities, d_ff=d_ff)
        x = torch.randn(batch_size, seq_len, d_model)
        out = ff(x)
        self.assertEqual(out.shape, (batch_size, seq_len, d_model))


class TestResidualBlock(unittest.TestCase):
    def test_output_shape_ffn(self):
        d_model, num_modalities, d_ff = 64, 2, 256
        batch_size, seq_len = 32, 100
        rb = ResidualBlock(
            d_model=d_model,
            module_type=ModuleType.FFN,
            num_modalities=num_modalities,
            d_ff=d_ff,
        )
        x = torch.randn(batch_size, seq_len, d_model)
        out = rb(x)
        self.assertEqual(out.shape, (batch_size, seq_len, d_model))

    def test_output_shape_cross_attention(self):
        d_model, num_heads, batch_size = 64, 8, 32
        seq_len_x, seq_len_context = 50, 100
        rb = ResidualBlock(
            d_model=d_model, module_type=ModuleType.CrossAttention, num_heads=num_heads
        )
        x = torch.randn(batch_size, seq_len_x, d_model)
        context = torch.randn(batch_size, seq_len_context, d_model)
        out = rb(x, context)
        self.assertEqual(out.shape, (batch_size, seq_len_x, d_model))


class TestFuser(unittest.TestCase):
    def test_output_shape_x2y(self):
        d_model, num_heads, batch_size = 64, 8, 32
        device, mode = "cpu", FusionMode.X2Y
        max_seq_len = 100
        fuser = Fuser(
            d_model=d_model,
            num_modalities=2,
            num_heads=num_heads,
            device=device,
            mode=mode,
            max_seq_len=max_seq_len,
        )
        seq_len_x, seq_len_y = 90, 50

        x = torch.randn(batch_size, seq_len_x, d_model)
        y = torch.randn(batch_size, seq_len_y, d_model)
        out = fuser(x, y)
        self.assertEqual(out.shape, (batch_size, max_seq_len, d_model))

    def test_output_shape_y2x(self):
        d_model, num_heads, batch_size = 64, 8, 32
        device, mode = "cpu", FusionMode.Y2X
        max_seq_len = 100
        fuser = Fuser(
            d_model=d_model,
            num_modalities=2,
            num_heads=num_heads,
            device=device,
            mode=mode,
            max_seq_len=max_seq_len,
        )
        seq_len_x, seq_len_y = 100, 50
        x = torch.randn(batch_size, seq_len_x, d_model)
        y = torch.randn(batch_size, seq_len_y, d_model)
        out = fuser(x, y)

        self.assertEqual(out.shape, (batch_size, max_seq_len, d_model))

    def test_output_shape_bi(self):
        d_model, num_heads, batch_size = 64, 8, 32
        device, mode = "cpu", FusionMode.BI
        max_seq_len = 100
        fuser = Fuser(
            d_model=d_model,
            num_modalities=2,
            num_heads=num_heads,
            device=device,
            mode=mode,
            max_seq_len=max_seq_len,
        )
        seq_len_x, seq_len_y = 100, 50
        x = torch.randn(batch_size, seq_len_x, d_model)
        y = torch.randn(batch_size, seq_len_y, d_model)

        out = fuser(x, y)
        self.assertEqual(out.shape, (batch_size, 2 * max_seq_len, d_model))

    def test_fuser_with_truncation(self):
        d_model, num_heads, batch_size = 64, 8, 32
        device, mode = "cpu", FusionMode.BI
        max_seq_len = 100
        fuser = Fuser(
            d_model=d_model,
            num_modalities=2,
            num_heads=num_heads,
            device=device,
            mode=mode,
            max_seq_len=max_seq_len,
        )
        seq_len_x, seq_len_y = 120, 80
        x = torch.randn(batch_size, seq_len_x, d_model)
        y = torch.randn(batch_size, seq_len_y, d_model)
        out = fuser(x, y)

        self.assertEqual(out.shape, (batch_size, 2 * max_seq_len, d_model))


class TestClassifier(unittest.TestCase):
    def test_output_shape(self):
        d_model, num_classes = 64, 10
        batch_size, seq_len = 32, 100
        classifier = Classifier(d_model=d_model, num_classes=num_classes)
        x = torch.randn(batch_size, seq_len, d_model)
        out = classifier(x)
        self.assertEqual(out.shape, (batch_size, num_classes))


if __name__ == "__main__":
    unittest.main()
