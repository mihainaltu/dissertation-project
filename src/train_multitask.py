# src/train_multitask.py

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

from dataset_multitask import get_dataloaders_multitask
from model_multitask import MultiTaskCNN, count_parameters


CFG = {
    'root_dir'    : 'data/raw/measurements',
    'results_dir' : 'results/multitask',
    'models_dir'  : 'models',

    'crop'        : (3500, 7500),
    'batch_size'  : 64,
    'num_workers' : 0,

    'num_positions': 12,
    'dropout'      : 0.3,

    'epochs'       : 40,
    'lr'           : 1e-3,
    'weight_decay' : 1e-4,

    # Loss weighting: L = L_position + lambda * L_voltage
    # Start with equal weighting, tune if needed
    'lambda_volt'  : 0.5,

    'device'       : 'cuda' if torch.cuda.is_available() else 'cpu',
}


def run_epoch(model, loader, criterion_pos, criterion_volt,
              optimizer, device, lam, training=True):
    model.train() if training else model.eval()

    total_loss = 0.0
    pos_correct, pos_total = 0, 0
    volt_mae_sum = 0.0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for signals, pos_labels, volt_labels in tqdm(
                loader, leave=False,
                desc='Train' if training else 'Val  '):

            signals, pos_labels, volt_labels = (
                signals.to(device),
                pos_labels.to(device),
                volt_labels.to(device)
            )

            pos_logits, volt_pred = model(signals)

            loss_pos  = criterion_pos(pos_logits, pos_labels)
            loss_volt = criterion_volt(volt_pred, volt_labels)
            loss      = loss_pos + lam * loss_volt

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss   += loss.item() * len(pos_labels)
            pos_correct  += (pos_logits.argmax(1) == pos_labels).sum().item()
            pos_total    += len(pos_labels)

            # Denormalize voltage for MAE: pred * 9 + 1
            volt_pred_v  = volt_pred.detach() * 9.0 + 1.0
            volt_true_v  = volt_labels.detach() * 9.0 + 1.0
            volt_mae_sum += (volt_pred_v - volt_true_v).abs().sum().item()

    n = pos_total
    return (total_loss / n,
            pos_correct / n,
            volt_mae_sum / n)


def main():
    Path(CFG['results_dir']).mkdir(parents=True, exist_ok=True)
    Path(CFG['models_dir']).mkdir(exist_ok=True)

    device = torch.device(CFG['device'])
    print(f"\nDevice: {device}")

    train_loader, val_loader, test_loader = get_dataloaders_multitask(
        root_dir    = CFG['root_dir'],
        batch_size  = CFG['batch_size'],
        crop        = CFG['crop'],
        num_workers = CFG['num_workers'],
    )

    model = MultiTaskCNN(
        num_positions = CFG['num_positions'],
        dropout       = CFG['dropout'],
    ).to(device)
    print(f"Parameters: {count_parameters(model):,}")

    criterion_pos  = nn.CrossEntropyLoss()
    criterion_volt = nn.MSELoss()
    optimizer      = Adam(model.parameters(),
                          lr=CFG['lr'],
                          weight_decay=CFG['weight_decay'])
    scheduler      = CosineAnnealingLR(optimizer,
                                        T_max=CFG['epochs'], eta_min=1e-5)

    history  = {'train_loss': [], 'val_loss': [],
                 'train_pos_acc': [], 'val_pos_acc': [],
                 'train_volt_mae': [], 'val_volt_mae': []}
    best_val_pos = 0.0

    print(f"\nTraining for {CFG['epochs']} epochs "
          f"(λ_volt={CFG['lambda_volt']})...\n")

    for epoch in range(1, CFG['epochs'] + 1):
        tr_loss, tr_pos, tr_mae = run_epoch(
            model, train_loader, criterion_pos, criterion_volt,
            optimizer, device, CFG['lambda_volt'], training=True)

        vl_loss, vl_pos, vl_mae = run_epoch(
            model, val_loader, criterion_pos, criterion_volt,
            optimizer, device, CFG['lambda_volt'], training=False)

        scheduler.step()

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(vl_loss)
        history['train_pos_acc'].append(tr_pos)
        history['val_pos_acc'].append(vl_pos)
        history['train_volt_mae'].append(tr_mae)
        history['val_volt_mae'].append(vl_mae)

        best_marker = ''
        if vl_pos > best_val_pos:
            best_val_pos = vl_pos
            torch.save(model.state_dict(),
                       f"{CFG['models_dir']}/cnn_multitask_best.pt")
            best_marker = ' ← best'

        print(f"Epoch {epoch:02d}/{CFG['epochs']}  |  "
              f"Pos acc: {tr_pos*100:.2f}% / {vl_pos*100:.2f}%  |  "
              f"Volt MAE: {tr_mae:.3f}V / {vl_mae:.3f}V"
              + best_marker)

    with open(f"{CFG['results_dir']}/history.json", 'w') as f:
        json.dump(history, f)

    # ── Test evaluation ───────────────────────────────────────────────────
    print(f"\nLoading best model (val pos acc: {best_val_pos*100:.2f}%)...")
    model.load_state_dict(
        torch.load(f"{CFG['models_dir']}/cnn_multitask_best.pt",
                   map_location=device))
    model.eval()

    all_pos_preds, all_pos_labels = [], []
    all_volt_preds, all_volt_labels = [], []

    with torch.no_grad():
        for signals, pos_labels, volt_labels in tqdm(
                test_loader, desc='Testing'):
            signals = signals.to(device)
            pos_logits, volt_pred = model(signals)

            all_pos_preds.extend(
                pos_logits.argmax(1).cpu().numpy())
            all_pos_labels.extend(pos_labels.numpy())

            # Denormalize
            all_volt_preds.extend(
                (volt_pred.cpu().numpy() * 9.0 + 1.0))
            all_volt_labels.extend(
                (volt_labels.numpy() * 9.0 + 1.0))

    all_pos_preds   = np.array(all_pos_preds)
    all_pos_labels  = np.array(all_pos_labels)
    all_volt_preds  = np.array(all_volt_preds)
    all_volt_labels = np.array(all_volt_labels)

    pos_acc  = (all_pos_preds == all_pos_labels).mean()
    volt_mae = np.abs(all_volt_preds - all_volt_labels).mean()
    volt_rmse = np.sqrt(((all_volt_preds - all_volt_labels)**2).mean())

    np.save(f"{CFG['results_dir']}/pos_preds.npy",   all_pos_preds)
    np.save(f"{CFG['results_dir']}/pos_labels.npy",  all_pos_labels)
    np.save(f"{CFG['results_dir']}/volt_preds.npy",  all_volt_preds)
    np.save(f"{CFG['results_dir']}/volt_labels.npy", all_volt_labels)

    print(f"\n── Multi-Task Results ────────────────────────────────────")
    print(f"  Position classification accuracy : {pos_acc*100:.2f}%")
    print(f"  Voltage regression MAE           : {volt_mae:.4f} V")
    print(f"  Voltage regression RMSE          : {volt_rmse:.4f} V")
    print(f"\n── Comparison with single-task CNN ───────────────────────")
    print(f"  Single-task position acc         : 99.15%")
    print(f"  Multi-task position acc          : {pos_acc*100:.2f}%")
    print(f"  Voltage MAE (random guess ~2.5V) : 2.500 V")
    print(f"  Voltage MAE (multi-task)         : {volt_mae:.4f} V")


if __name__ == '__main__':
    main()