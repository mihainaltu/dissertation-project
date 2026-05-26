# src/model_exp2.py

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
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


class CableShieldCNN(nn.Module):
    """
    Input : (B, 1, 1020)  — single channel, 1020 samples (~5µs)
    Output: (B, 2)
    """
    def __init__(self, dropout=0.4):
        super().__init__()

        # Input: (B, 1, 1020)
        self.features = nn.Sequential(
            ConvBlock(1,  32, kernel_size=15, pool_size=2, dropout=0.1),  # → (B, 32, 210)
            ConvBlock(32, 64, kernel_size=11, pool_size=2, dropout=0.2),  # → (B, 64, 105)
            ConvBlock(64, 128, kernel_size=7, pool_size=2, dropout=0.3),  # → (B, 128, 52)
            ConvBlock(128, 64, kernel_size=5, pool_size=2, dropout=0.3),  # → (B, 64, 26)
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)   # → (B, 64, 1)

        self.classifier = nn.Sequential(
            nn.Flatten(),                             # → (B, 64)
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    model = CableShieldCNN()
    x     = torch.randn(8, 1, 2100)
    out   = model(x)
    print(f"Input  shape : {x.shape}")
    print(f"Output shape : {out.shape}")
    print(f"Parameters   : {count_parameters(model):,}")