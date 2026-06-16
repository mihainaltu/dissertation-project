# src/transfer_learning.py

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / "exp1_localization"))

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json

from exp2_dataset import get_dataloaders_exp2
from model import PDLocalizationCNN, count_parameters


# ── Adapted model for transfer learning ───────────────────────────────────────

class TransferCNN(nn.Module):
    """
    Transfer learning model for cable shield classification.
    Takes the pretrained PD localization CNN backbone,
    adapts input to 1 channel, replaces head with binary classifier.
    """
    def __init__(self, pretrained_path, device, freeze_backbone=True):
        super().__init__()

        # Load pretrained model
        pretrained = PDLocalizationCNN(num_classes=12, dropout=0.3)
        pretrained.load_state_dict(
            torch.load(pretrained_path, map_location=device))

        # ── Adapt first conv layer: 2ch → 1ch ─────────────────────────────
        # Average the weights across input channels
        old_conv   = pretrained.features[0].block[0]  # first Conv1d
        new_conv   = nn.Conv1d(
            in_channels  = 1,                          # changed: 2 → 1
            out_channels = old_conv.out_channels,
            kernel_size  = old_conv.kernel_size,
            padding      = old_conv.padding,
            bias         = False
        )
        # Initialize new conv weights by averaging over the 2 input channels
        new_conv.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)
        pretrained.features[0].block[0] = new_conv

        # ── Extract backbone (everything except classifier head) ───────────
        self.backbone = nn.Sequential(
            pretrained.features,
            pretrained.global_pool,
        )

        # ── Freeze backbone if requested ───────────────────────────────────
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # ── New binary classification head ─────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x

    def unfreeze_backbone(self):
        """Call this for fine-tuning phase."""
        for param in self.backbone.parameters():
            param.requires_grad = True


# ── Config ────────────────────────────────────────────────────────────────────

CFG = {
    'root_dir'        : 'data/raw/sames-cable-data',
    'pretrained_path' : 'models/cnn_best.pt',
    'results_dir'     : 'results/transfer_learning',
    'models_dir'      : 'models',

    'batch_size'      : 16,
    'aug_factor'      : 10,
    'num_workers'     : 0,

    # Phase 1: train head only
    'epochs_head'     : 30,
    'lr_head'         : 1e-3,

    # Phase 2: fine-tune full model
    'epochs_finetune' : 20,
    'lr_finetune'     : 1e-4,

    'weight_decay'    : 1e-4,
    'device'          : 'cuda' if torch.cuda.is_available() else 'cpu',
}


# ── Training loop ─────────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, training=True):
    model.train() if training else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for signals, labels in tqdm(loader, leave=False,
                                    desc='Train' if training else 'Val  '):
            signals, labels = signals.to(device), labels.to(device)
            logits = model(signals)
            loss   = criterion(logits, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(labels)
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += len(labels)

    return total_loss / total, correct / total


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    Path(CFG['results_dir']).mkdir(parents=True, exist_ok=True)
    Path(CFG['models_dir']).mkdir(exist_ok=True)

    device = torch.device(CFG['device'])
    print(f"\nDevice: {device}")

    # Data
    train_loader, val_loader, test_loader, _ = get_dataloaders_exp2(
        root_dir    = CFG['root_dir'],
        batch_size  = CFG['batch_size'],
        aug_factor  = CFG['aug_factor'],
        num_workers = CFG['num_workers'],
    )

    # Model
    model = TransferCNN(
        pretrained_path = CFG['pretrained_path'],
        device          = device,
        freeze_backbone = True,
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total:,} total | {trainable:,} trainable (head only)")

    criterion = nn.CrossEntropyLoss()
    history   = {'train_loss': [], 'val_loss': [],
                  'train_acc':  [], 'val_acc':  [],
                  'phase': []}
    best_val  = 0.0

    # ── Phase 1: Train head only ───────────────────────────────────────────
    print(f"\n── Phase 1: Training head only ({CFG['epochs_head']} epochs) ──")
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()),
                     lr=CFG['lr_head'], weight_decay=CFG['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer,
                                   T_max=CFG['epochs_head'], eta_min=1e-5)

    for epoch in range(1, CFG['epochs_head'] + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion,
                                    optimizer, device, training=True)
        vl_loss, vl_acc = run_epoch(model, val_loader,   criterion,
                                    optimizer, device, training=False)
        scheduler.step()

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(vl_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(vl_acc)
        history['phase'].append(1)

        best_marker = ''
        if vl_acc > best_val:
            best_val = vl_acc
            torch.save(model.state_dict(),
                       f"{CFG['models_dir']}/cnn_transfer_best.pt")
            best_marker = ' ← best'

        print(f"[P1] Epoch {epoch:02d}/{CFG['epochs_head']}  |  "
              f"Train: {tr_acc*100:.2f}%  Val: {vl_acc*100:.2f}%"
              + best_marker)

    # ── Phase 2: Fine-tune full model ─────────────────────────────────────
    print(f"\n── Phase 2: Fine-tuning full model ({CFG['epochs_finetune']} epochs) ──")
    model.unfreeze_backbone()

    trainable2 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Now training {trainable2:,} parameters")

    optimizer2 = Adam(model.parameters(),
                      lr=CFG['lr_finetune'], weight_decay=CFG['weight_decay'])
    scheduler2 = CosineAnnealingLR(optimizer2,
                                    T_max=CFG['epochs_finetune'], eta_min=1e-6)

    for epoch in range(1, CFG['epochs_finetune'] + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion,
                                    optimizer2, device, training=True)
        vl_loss, vl_acc = run_epoch(model, val_loader,    criterion,
                                    optimizer2, device, training=False)
        scheduler2.step()

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(vl_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(vl_acc)
        history['phase'].append(2)

        best_marker = ''
        if vl_acc > best_val:
            best_val = vl_acc
            torch.save(model.state_dict(),
                       f"{CFG['models_dir']}/cnn_transfer_best.pt")
            best_marker = ' ← best'

        print(f"[P2] Epoch {epoch:02d}/{CFG['epochs_finetune']}  |  "
              f"Train: {tr_acc*100:.2f}%  Val: {vl_acc*100:.2f}%"
              + best_marker)

    # ── Test evaluation ───────────────────────────────────────────────────
    print(f"\nLoading best model (val acc: {best_val*100:.2f}%)...")
    model.load_state_dict(
        torch.load(f"{CFG['models_dir']}/cnn_transfer_best.pt",
                   map_location=device))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for signals, labels in tqdm(test_loader, desc='Testing'):
            signals = signals.to(device)
            preds   = model(signals).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    test_acc   = (all_preds == all_labels).mean()

    np.save(f"{CFG['results_dir']}/preds.npy",  all_preds)
    np.save(f"{CFG['results_dir']}/labels.npy", all_labels)

    with open(f"{CFG['results_dir']}/history.json", 'w') as f:
        json.dump(history, f)

    print(f"\n── Transfer Learning Results ─────────────────────────────")
    print(f"  RF from scratch (hand-crafted features) : 83.72%")
    print(f"  CNN from scratch (raw waveform)         : ~50.00% (failed)")
    print(f"  Transfer learning CNN                   : {test_acc*100:.2f}%")


if __name__ == '__main__':
    main()
