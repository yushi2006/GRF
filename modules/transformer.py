import torch
import torch.nn as nn

from modules import MultiheadAttention, PositionalEncoding


class TransformerLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        d_ffn: int = 2048,
        dropout: float = 0.1,
        activation="relu",
    ):
        super().__init__()

        self.cross_attn_a_to_b = MultiheadAttention(d_model, nhead)
        self.cross_attn_b_to_a = MultiheadAttention(d_model, nhead)
        self.self_attn_a = MultiheadAttention(d_model, nhead)
        self.self_attn_b = MultiheadAttention(d_model, nhead)

        self.ffn_a = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.ReLU() if activation == "relu" else nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ffn, d_model),
        )
        self.ffn_b = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.ReLU() if activation == "relu" else nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ffn, d_model),
        )
        self.norm_cross_a = nn.LayerNorm(d_model)
        self.norm_cross_b = nn.LayerNorm(d_model)
        self.norm_self_a = nn.LayerNorm(d_model)
        self.norm_self_b = nn.LayerNorm(d_model)
        self.norm_ffn_a = nn.LayerNorm(d_model)
        self.norm_ffn_b = nn.LayerNorm(d_model)
        self.dropout_cross_a = nn.Dropout(dropout)
        self.dropout_cross_b = nn.Dropout(dropout)
        self.dropout_self_a = nn.Dropout(dropout)
        self.dropout_self_b = nn.Dropout(dropout)
        self.dropout_ffn_a = nn.Dropout(dropout)
        self.dropout_ffn_b = nn.Dropout(dropout)

    def forward(self, x_a, x_b, mask_a=None, mask_b=None):
        _, cross_a_out = self.cross_attn_a_to_b(x=x_a, k=x_b, v=x_b, mask=mask_b)
        x_a = self.norm_cross_a(x_a + self.dropout_cross_a(cross_a_out))

        _, cross_b_out = self.cross_attn_b_to_a(x=x_b, k=x_a, v=x_a, mask=mask_a)
        x_b = self.norm_cross_b(x_b + self.dropout_cross_b(cross_b_out))

        _, self_a_out = self.self_attn_a(x=x_a, k=x_a, v=x_a, mask=mask_a)
        x_a = self.norm_self_a(x_a + self.dropout_self_a(self_a_out))

        _, self_b_out = self.self_attn_b(x=x_b, k=x_b, v=x_b, mask=mask_b)
        x_b = self.norm_self_b(x_b + self.dropout_self_b(self_b_out))

        x_a = self.norm_ffn_a(x_a + self.dropout_ffn_a(self.ffn_a(x_a)))
        x_b = self.norm_ffn_b(x_b + self.dropout_ffn_b(self.ffn_b(x_b)))

        return x_a, x_b


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        d_in_a,
        d_in_b,
        d_model,
        nhead,
        num_layers,
        d_ffn,
        num_classes,
        dropout=0.1,
    ):
        super().__init__()

        self.proj_a = nn.Linear(d_in_a, d_model)
        self.proj_b = nn.Linear(d_in_b, d_model)

        self.pos_encoder = PositionalEncoding(d_model, dropout)

        self.layers = nn.ModuleList(
            [
                TransformerLayer(d_model, nhead, d_ffn, dropout)
                for _ in range(num_layers)
            ]
        )

        self.fusion_layer = nn.Linear(d_model * 2, d_model)
        self.classifier = nn.Linear(d_model, num_classes)
        self.fusion_norm = nn.LayerNorm(d_model)

    def forward(self, x_a, x_b):
        x_a = self.proj_a(x_a)
        x_b = self.proj_b(x_b)

        x_a = self.pos_encoder(x_a)
        x_b = self.pos_encoder(x_b)

        for layer in self.layers:
            x_a, x_b = layer(x_a, x_b)

        pooled_a = x_a.mean(dim=1)
        pooled_b = x_b.mean(dim=1)

        fused = torch.cat([pooled_a, pooled_b], dim=-1)
        fused = self.fusion_norm(self.fusion_layer(fused))
        return self.classifier(fused)


if __name__ == "__main__":
    model = TransformerEncoder(
        d_in_a=80,
        d_in_b=768,
        d_model=128,
        nhead=4,
        num_layers=3,
        d_ffn=512,
        num_classes=5,
    )
    print(
        f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters."
    )

    dummy_a = torch.randn(4, 100, 80)
    dummy_b = torch.randn(4, 50, 768)

    output = model(dummy_a, dummy_b)
    print(f"Output shape: {output.shape}")
    assert output.shape == (4, 5)
    print("Forward pass successful.")
