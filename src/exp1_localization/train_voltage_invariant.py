# src/train_voltage_invariant.py

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

from exp1_dataset import get_dataloaders_voltage_split, POSITIONS
from model import PDLocalizationCNN, count_parameters


CFG = {
    'root_dir'    : 'data/raw/measurements',
    'results_dir' : 'results/voltage_invariant',
    'models_dir'  : 'models',

    'train_voltages' : ['1V','2V','3V','4V','5V','6V','7V','8V'],
    'test_voltages'  : ['9V','10V'],

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


def main():
    Path(CFG['results_dir']).mkdir(parents=True, exist_ok=True)
    Path(CFG['models_dir']).mkdir(exist_ok=True)

    device = torch.device(CFG['device'])
    print(f"\nDevice: {device}")

    train_loader, val_loader, test_loader, class_names = \
        get_dataloaders_voltage_split(
            root_dir       = CFG['root_dir'],
            train_voltages = CFG['train_voltages'],
            test_voltages  = CFG['test_voltages'],
            batch_size     = CFG['batch_size'],
            normalize      = True,
            crop           = CFG['crop'],
            num_workers    = CFG['num_workers'],
        )

    model = PDLocalizationCNN(
        num_classes = CFG['num_classes'],
        dropout     = CFG['dropout'],
    ).to(device)
    print(f"Parameters: {count_parameters(model):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(),
                     lr=CFG['lr'], weight_decay=CFG['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer,
                                   T_max=CFG['epochs'], eta_min=1e-5)

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

        best_marker = ''
        if vl_acc > best_val:
            best_val = vl_acc
            torch.save(model.state_dict(),
                       f"{CFG['models_dir']}/cnn_voltage_invariant_best.pt")
            best_marker = ' ← best'

        print(f"Epoch {epoch:02d}/{CFG['epochs']}  |  "
              f"Train loss: {tr_loss:.4f}  acc: {tr_acc*100:.2f}%  |  "
              f"Val loss: {vl_loss:.4f}  acc: {vl_acc*100:.2f}%"
              + best_marker)

    with open(f"{CFG['results_dir']}/history.json", 'w') as f:
        json.dump(history, f)

    # Test evaluation
    print(f"\nLoading best model (val acc: {best_val*100:.2f}%)...")
    model.load_state_dict(
        torch.load(f"{CFG['models_dir']}/cnn_voltage_invariant_best.pt",
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
    print(f"\nVoltage-invariant test accuracy: {test_acc*100:.2f}%")
    print(f"(Standard random-split accuracy was: 99.15%)")

    np.save(f"{CFG['results_dir']}/preds.npy",  all_preds)
    np.save(f"{CFG['results_dir']}/labels.npy", all_labels)

    # Also run RF baseline with same split for comparison
    print("\nRunning RF baseline with voltage-invariant split...")
    from exp1_dataset import build_file_list
    from features import extract_features, CROP
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import StandardScaler

    all_samples = build_file_list(CFG['root_dir'])

    # Rebuild voltage split for RF
    root = Path(CFG['root_dir'])
    rf_train, rf_test = [], []
    for pos_folder in sorted(root.iterdir()):
        if not pos_folder.is_dir():
            continue
        try:
            position = int(pos_folder.name)
        except ValueError:
            continue
        from exp1_dataset import POS_TO_LABEL
        if position not in POS_TO_LABEL:
            continue
        label = POS_TO_LABEL[position]
        for volt_folder in sorted(pos_folder.iterdir()):
            if not volt_folder.is_dir():
                continue
            for mat_file in sorted(volt_folder.glob('*.mat')):
                if volt_folder.name in CFG['train_voltages']:
                    rf_train.append((str(mat_file), label))
                elif volt_folder.name in CFG['test_voltages']:
                    rf_test.append((str(mat_file), label))

    from exp1_dataset import load_mat_file
    def build_X(samples):
        X, y = [], []
        for fp, lb in tqdm(samples, desc='Features', leave=False):
            sig = load_mat_file(fp)[:, CROP[0]:CROP[1]]
            X.append(extract_features(sig))
            y.append(lb)
        return np.array(X), np.array(y)

    X_train, y_train = build_X(rf_train)
    X_test,  y_test  = build_X(rf_test)

    scaler   = StandardScaler()
    X_train  = scaler.fit_transform(X_train)
    X_test   = scaler.transform(X_test)

    rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    rf_acc = accuracy_score(y_test, rf.predict(X_test))
    print(f"RF voltage-invariant test accuracy: {rf_acc*100:.2f}%")
    print(f"(RF random-split accuracy was: 99.80%)")

    print("\n── Final Comparison ─────────────────────────────")
    print(f"  Model          | Random Split | Voltage-Invariant")
    print(f"  Random Forest  |    99.80%    |   {rf_acc*100:.2f}%")
    print(f"  1D-CNN         |    99.15%    |   {test_acc*100:.2f}%")


if __name__ == '__main__':
    main()
