# src/model_multitask.py

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


class MultiTaskCNN(nn.Module):
    """
    Multi-task 1D-CNN for simultaneous:
      - Position classification (12 classes)
      - Voltage regression (1–10V)

    Input : (B, 2, 4000)
    Output: (B, 12) logits, (B, 1) voltage estimate
    """
    def __init__(self, num_positions=12, dropout=0.3):
        super().__init__()

        # ── Shared backbone ────────────────────────────────────────────────
        self.backbone = nn.Sequential(
            ConvBlock(2,   32,  kernel_size=15, pool_size=2, dropout=0.1),
            ConvBlock(32,  64,  kernel_size=11, pool_size=2, dropout=0.1),
            ConvBlock(64,  128, kernel_size=7,  pool_size=2, dropout=0.2),
            ConvBlock(128, 256, kernel_size=5,  pool_size=2, dropout=0.2),
            ConvBlock(256, 256, kernel_size=3,  pool_size=2, dropout=0.2),
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # → (B, 256, 1)

        # ── Position head (classification) ─────────────────────────────────
        self.position_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_positions),
        )

        # ── Voltage head (regression) ──────────────────────────────────────
        self.voltage_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),       # single continuous output
        )

    def forward(self, x):
        features = self.backbone(x)
        features = self.global_pool(features)
        pos_logits  = self.position_head(features)
        volt_output = self.voltage_head(features).squeeze(1)
        return pos_logits, volt_output


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    model = MultiTaskCNN()
    x     = torch.randn(8, 2, 4000)
    pos, volt = model(x)
    print(f"Input shape       : {x.shape}")
    print(f"Position output   : {pos.shape}")
    print(f"Voltage output    : {volt.shape}")
    print(f"Parameters        : {count_parameters(model):,}")