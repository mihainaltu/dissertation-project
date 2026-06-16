# src/model.py

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv1d → BatchNorm → ReLU → MaxPool"""
    def __init__(self, in_ch, out_ch, kernel_size=7, pool_size=2, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size,
                      padding=kernel_size // 2, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(pool_size),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.block(x)


class PDLocalizationCNN(nn.Module):
    """
    1D-CNN for PD injection location classification.

    Input : (B, 2, 4000)  — 2-channel, 4000-sample waveform
    Output: (B, num_classes)  — logits
    """
    def __init__(self, num_classes=12, dropout=0.3):
        super().__init__()

        # ── Convolutional backbone ─────────────────────────────────────────
        # Input: (B, 2, 4000)
        self.features = nn.Sequential(
            ConvBlock(2,   32,  kernel_size=15, pool_size=2, dropout=0.1),  # → (B, 32, 2000)
            ConvBlock(32,  64,  kernel_size=11, pool_size=2, dropout=0.1),  # → (B, 64, 1000)
            ConvBlock(64,  128, kernel_size=7,  pool_size=2, dropout=0.2),  # → (B, 128, 500)
            ConvBlock(128, 256, kernel_size=5,  pool_size=2, dropout=0.2),  # → (B, 256, 250)
            ConvBlock(256, 256, kernel_size=3,  pool_size=2, dropout=0.2),  # → (B, 256, 125)
        )

        # ── Global pooling ─────────────────────────────────────────────────
        self.global_pool = nn.AdaptiveAvgPool1d(1)   # → (B, 256, 1)

        # ── Classifier head ────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Flatten(),                             # → (B, 256)
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),              # → (B, 12)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    model = PDLocalizationCNN(num_classes=12)
    x     = torch.randn(8, 2, 4000)
    out   = model(x)
    print(f"Input  shape : {x.shape}")
    print(f"Output shape : {out.shape}")
    print(f"Parameters   : {count_parameters(model):,}")
