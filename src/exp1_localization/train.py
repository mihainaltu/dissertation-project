# src/train.py

import sys, os
sys.path.append(os.path.dirname(__file__))

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json

from dataset import get_dataloaders, POSITIONS
from model import PDLocalizationCNN, count_parameters


# ── Config ────────────────────────────────────────────────────────────────────

CFG = {
    'root_dir'    : 'data/raw/measurements',
    'results_dir' : 'results',
    'models_dir'  : 'models',

    'crop'        : (3500, 7500),
    'batch_size'  : 64,
    'num_workers' : 0,

    'num_classes' : 12,
    'dropout'     : 0.3,

    'epochs'      : 40,
    'lr'          : 1e-3,
    'weight_decay': 1e-4,

    'device'      : 'cuda' if torch.cuda.is_available() else 'cpu',
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
    Path(CFG['results_dir']).mkdir(exist_ok=True)
    Path(CFG['models_dir']).mkdir(exist_ok=True)

    device = torch.device(CFG['device'])
    print(f"\nDevice: {device}")

    # Data
    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        root_dir    = CFG['root_dir'],
        batch_size  = CFG['batch_size'],
        normalize   = True,
        crop        = CFG['crop'],
        num_workers = CFG['num_workers'],
    )

    # Model
    model = PDLocalizationCNN(
        num_classes = CFG['num_classes'],
        dropout     = CFG['dropout'],
    ).to(device)
    print(f"Parameters  : {count_parameters(model):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(),
                     lr=CFG['lr'], weight_decay=CFG['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer, T_max=CFG['epochs'], eta_min=1e-5)

    # Training
    history  = {'train_loss': [], 'val_loss': [],
                 'train_acc':  [], 'val_acc':  []}
    best_val = 0.0

    print(f"\nTraining for {CFG['epochs']} epochs...\n")

    for epoch in range(1, CFG['epochs'] + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion,
                                    optimizer, device, training=True)
        vl_loss, vl_acc = run_epoch(model, val_loader,   criterion,
                                    optimizer, device, training=False)
        scheduler.step()

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(vl_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(vl_acc)

        print(f"Epoch {epoch:02d}/{CFG['epochs']}  |  "
              f"Train loss: {tr_loss:.4f}  acc: {tr_acc*100:.2f}%  |  "
              f"Val   loss: {vl_loss:.4f}  acc: {vl_acc*100:.2f}%"
              + (" ← best" if vl_acc > best_val else ""))

        if vl_acc > best_val:
            best_val = vl_acc
            torch.save(model.state_dict(),
                       f"{CFG['models_dir']}/cnn_best.pt")

    # Save history
    with open(f"{CFG['results_dir']}/cnn_history.json", 'w') as f:
        json.dump(history, f)

    # ── Test evaluation ───────────────────────────────────────────────────
    print(f"\nLoading best model (val acc: {best_val*100:.2f}%)...")
    model.load_state_dict(torch.load(f"{CFG['models_dir']}/cnn_best.pt",
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
    print(f"\nTest accuracy: {test_acc*100:.2f}%")

    # Save predictions for plotting
    np.save(f"{CFG['results_dir']}/cnn_preds.npy",  all_preds)
    np.save(f"{CFG['results_dir']}/cnn_labels.npy", all_labels)

    print("\nDone! Run src/evaluate.py to generate plots.")


if __name__ == '__main__':
    main()