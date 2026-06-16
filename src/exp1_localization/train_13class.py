"""
train_13class.py — Retrain all classifiers with NonPD added as class 12.

What this does
--------------
1. Load Exp1 feature cache (results/baseline_v2/features_cache.npz)
2. Load NonPD signals from data/raw/nonpd/ and extract features
3. Merge into a 13-class dataset
4. Train RF + SVM (feature-based)
5. Retrain CNN with 13 output classes (padded NonPD signals)
6. Evaluate and compare against 12-class baselines
7. Save all results to results/baseline_13class/

Usage
-----
    # RF + SVM only (fast, CPU):
    python train_13class.py --data_dir data/raw/measurements \\
                            --nonpd_dir data/raw/nonpd \\
                            --mode features

    # CNN only (GPU recommended):
    python train_13class.py --data_dir data/raw/measurements \\
                            --nonpd_dir data/raw/nonpd \\
                            --mode cnn

    # Both:
    python train_13class.py --data_dir data/raw/measurements \\
                            --nonpd_dir data/raw/nonpd \\
                            --mode all
"""

import os
import argparse
import json
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.model_selection import StratifiedShuffleSplit

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ── local imports ─────────────────────────────────────────────────────────────
from features_v2 import feature_count, feature_names
from dataset_nonpd import (
    load_split, extract_features_batch, pad_batch, NONPD_LABEL
)

# ── constants ─────────────────────────────────────────────────────────────────
N_CLASSES    = 13
RANDOM_STATE = 42
LABEL_NAMES  = [
    '100m','200m','300m','500m','700m','900m',
    '1000m','1300m','1500m','1600m','1800m','1900m',
    'NonPD'
]

POS_TO_LABEL = {
    100:0,200:1,300:2,500:3,700:4,900:5,
    1000:6,1300:7,1500:8,1600:9,1800:10,1900:11
}


# ─────────────────────────────────────────────────────────────────────────────
# 1. Data assembly
# ─────────────────────────────────────────────────────────────────────────────

def load_exp1_cache(cache_path):
    c = np.load(cache_path)
    X, y, volts = c['X'], c['y'], c['volts']
    X = np.where(np.isfinite(X), X, 0.0)
    print(f'Exp1 cache loaded: X={X.shape}  classes={np.unique(y)}')
    return X, y, volts


def load_nonpd_features(nonpd_dir, cache_path=None):
    """Load or extract NonPD features, with optional caching."""
    if cache_path and Path(cache_path).exists():
        print(f'Loading NonPD feature cache from {cache_path}')
        c = np.load(cache_path)
        return c['X'], c['splits']

    print('Extracting NonPD features...')
    Xs, splits = [], []
    for split in ['train', 'val', 'test']:
        try:
            sigs = load_split(nonpd_dir, split)
        except FileNotFoundError as e:
            print(f'  [{split}] not found, skipping: {e}')
            continue
        X = extract_features_batch(sigs, desc=f'NonPD {split}')
        Xs.append(X)
        splits.extend([split] * len(X))
        print(f'  {split}: {len(X)} samples')

    X_all    = np.vstack(Xs)
    splits   = np.array(splits)
    if cache_path:
        np.savez(cache_path, X=X_all, splits=splits)
        print(f'Saved NonPD cache to {cache_path}')
    return X_all, splits


def build_13class_dataset(X_exp1, y_exp1, X_nonpd, max_nonpd=None):
    """
    Merge Exp1 (classes 0-11) with NonPD (class 12).
    Optionally subsample NonPD to avoid class imbalance.
    """
    rng = np.random.default_rng(RANDOM_STATE)

    if max_nonpd is not None and len(X_nonpd) > max_nonpd:
        idx = rng.choice(len(X_nonpd), max_nonpd, replace=False)
        X_nonpd = X_nonpd[idx]
        print(f'Subsampled NonPD to {max_nonpd} samples')

    y_nonpd = np.full(len(X_nonpd), NONPD_LABEL, dtype=np.int32)
    X_all   = np.vstack([X_exp1, X_nonpd])
    y_all   = np.concatenate([y_exp1, y_nonpd])
    print(f'13-class dataset: {len(X_all)} total  '
          f'(Exp1={len(X_exp1)}, NonPD={len(X_nonpd)})')
    return X_all, y_all


def stratified_split(X, y):
    sss  = StratifiedShuffleSplit(1, test_size=0.30, random_state=RANDOM_STATE)
    tr, tmp = next(sss.split(X, y))
    sss2 = StratifiedShuffleSplit(1, test_size=0.50, random_state=RANDOM_STATE)
    va, te = next(sss2.split(X[tmp], y[tmp]))
    return tr, tmp[va], tmp[te]


# ─────────────────────────────────────────────────────────────────────────────
# 2. RF + SVM
# ─────────────────────────────────────────────────────────────────────────────

def train_rf(X_tr, y_tr):
    clf = RandomForestClassifier(
        n_estimators=500, n_jobs=-1, random_state=RANDOM_STATE
    )
    clf.fit(X_tr, y_tr)
    return clf


def train_svm(X_tr, y_tr):
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('svm',    SVC(kernel='rbf', C=10, gamma='scale',
                       decision_function_shape='ovr',
                       random_state=RANDOM_STATE))
    ])
    pipe.fit(X_tr, y_tr)
    return pipe


def evaluate_clf(clf, X_te, y_te, name, out_dir):
    preds = clf.predict(X_te)
    acc   = accuracy_score(y_te, preds)
    print(f'\n{name}: {acc*100:.2f}%')
    print(classification_report(y_te, preds, target_names=LABEL_NAMES, digits=4))

    cm = confusion_matrix(y_te, preds)
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, ax=ax)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(f'{name} — 13-class ({acc*100:.2f}%)')
    plt.tight_layout()
    fig.savefig(out_dir / f'cm_{name.lower().replace(" ","_")}.png',
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Per-class accuracy
    per_cls = cm.diagonal() / cm.sum(axis=1)
    fig, ax = plt.subplots(figsize=(11, 4))
    colors  = ['green' if a >= 0.95 else 'orange' if a >= 0.85 else 'red'
               for a in per_cls]
    ax.bar(LABEL_NAMES, per_cls * 100, color=colors)
    ax.axhline(95, color='gray', linestyle='--', linewidth=0.8)
    ax.set_ylim(0, 105); ax.set_ylabel('Accuracy (%)')
    ax.set_title(f'{name} — Per-class accuracy')
    ax.set_xticklabels(LABEL_NAMES, rotation=45)
    for i, v in enumerate(per_cls):
        ax.text(i, v*100+1, f'{v*100:.1f}%', ha='center', fontsize=7)
    plt.tight_layout()
    fig.savefig(out_dir / f'perclass_{name.lower().replace(" ","_")}.png',
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    return acc, per_cls


# ─────────────────────────────────────────────────────────────────────────────
# 3. CNN  (13-class, modified from Exp1 architecture)
# ─────────────────────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=7, dropout=0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, padding=kernel//2),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
        )
    def forward(self, x): return self.block(x)


class PD13CNN(nn.Module):
    """
    Modified Exp1 CNN for 13-class detection + localization.
    Input:  (B, 2, 4000)  — channel 2 = zeros for NonPD signals
    Output: (B, 13)
    """
    def __init__(self, num_classes=13, dropout=0.3):
        super().__init__()
        self.convs = nn.Sequential(
            ConvBlock(2,   32, dropout=dropout),
            ConvBlock(32,  64, dropout=dropout),
            ConvBlock(64, 128, dropout=dropout),
            ConvBlock(128,256, dropout=dropout),
            ConvBlock(256,256, dropout=dropout),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )
    def forward(self, x):
        return self.head(self.pool(self.convs(x)))


class PD13Dataset(Dataset):
    """
    Combined dataset: Exp1 raw signals + NonPD padded signals.
    Returns (signal_2ch_4000, label).
    """
    def __init__(self, exp1_samples, nonpd_signals, normalize=True):
        """
        exp1_samples : list of (filepath, label) from Exp1
        nonpd_signals: np.ndarray (N, 400) NonPD signals
        """
        self.exp1_samples  = exp1_samples
        self.nonpd_padded  = pad_batch(nonpd_signals)   # (N, 2, 4000)
        self.normalize     = normalize
        self.n_exp1        = len(exp1_samples)
        self.n_nonpd       = len(nonpd_signals)

    def __len__(self):
        return self.n_exp1 + self.n_nonpd

    def __getitem__(self, idx):
        if idx < self.n_exp1:
            from scipy.io import loadmat
            fp, label = self.exp1_samples[idx]
            mat  = loadmat(fp, simplify_cells=False)
            data = mat['tpd']['Data'][0,0].astype(np.float32)
            sig  = data[:, 3500:7500]          # crop (2, 4000)
            if self.normalize:
                std = sig.std() + 1e-8
                sig = (sig - sig.mean()) / std
            return torch.tensor(sig), label
        else:
            i   = idx - self.n_exp1
            sig = self.nonpd_padded[i].copy()  # (2, 4000)
            if self.normalize:
                s = sig[0]   # only ch1 has data
                std = s.std() + 1e-8
                if std > 1e-8:
                    sig[0] = (s - s.mean()) / std
            return torch.tensor(sig), NONPD_LABEL


def build_exp1_samples(data_dir):
    """Rebuild file list from measurements dir."""
    from pathlib import Path
    samples = []
    root = Path(data_dir)
    for pos, label in POS_TO_LABEL.items():
        for volt in range(1, 11):
            volt_dir = root / str(pos) / f'{volt}V'
            if not volt_dir.exists():
                continue
            for f in volt_dir.glob('*.mat'):
                samples.append((str(f), label))
    return samples


def train_cnn_13(data_dir, nonpd_dir, out_dir, device,
                 epochs=50, batch_size=64, lr=1e-3,
                 max_nonpd=5000):
    print(f'\nTraining 13-class CNN on {device}')

    # Load data
    exp1_samples = build_exp1_samples(data_dir)
    rng = np.random.default_rng(RANDOM_STATE)
    rng.shuffle(exp1_samples)

    # Split Exp1
    n = len(exp1_samples)
    n_tr = int(0.70 * n); n_va = int(0.15 * n)
    tr_s  = exp1_samples[:n_tr]
    va_s  = exp1_samples[n_tr:n_tr+n_va]
    te_s  = exp1_samples[n_tr+n_va:]

    # NonPD signals — use predefined splits
    nonpd_tr = load_split(nonpd_dir, 'train')
    nonpd_va = load_split(nonpd_dir, 'val')
    nonpd_te = load_split(nonpd_dir, 'test')

    # Subsample NonPD train to avoid huge imbalance
    if max_nonpd and len(nonpd_tr) > max_nonpd:
        idx = rng.choice(len(nonpd_tr), max_nonpd, replace=False)
        nonpd_tr = nonpd_tr[idx]

    print(f'  Exp1 train/val/test: {len(tr_s)}/{len(va_s)}/{len(te_s)}')
    print(f'  NonPD train/val/test: {len(nonpd_tr)}/{len(nonpd_va)}/{len(nonpd_te)}')

    tr_ds = PD13Dataset(tr_s,  nonpd_tr)
    va_ds = PD13Dataset(va_s,  nonpd_va)
    te_ds = PD13Dataset(te_s,  nonpd_te)

    tr_dl = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,
                       num_workers=2, pin_memory=True)
    va_dl = DataLoader(va_ds, batch_size=batch_size, shuffle=False,
                       num_workers=2, pin_memory=True)
    te_dl = DataLoader(te_ds, batch_size=batch_size, shuffle=False,
                       num_workers=2, pin_memory=True)

    model = PD13CNN(num_classes=N_CLASSES).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit  = nn.CrossEntropyLoss()

    best_va_acc = 0.0
    history = {'train_acc':[], 'val_acc':[], 'train_loss':[], 'val_loss':[]}

    for epoch in range(1, epochs+1):
        # Train
        model.train()
        tr_loss = tr_correct = tr_total = 0
        for X, y in tqdm(tr_dl, desc=f'E{epoch}/{epochs} train', leave=False):
            X, y = X.to(device), y.to(device)
            opt.zero_grad()
            out  = model(X)
            loss = crit(out, y)
            loss.backward(); opt.step()
            tr_loss    += loss.item() * len(y)
            tr_correct += (out.argmax(1) == y).sum().item()
            tr_total   += len(y)
        sched.step()

        # Validate
        model.eval()
        va_loss = va_correct = va_total = 0
        with torch.no_grad():
            for X, y in va_dl:
                X, y = X.to(device), y.to(device)
                out  = model(X)
                va_loss    += crit(out, y).item() * len(y)
                va_correct += (out.argmax(1) == y).sum().item()
                va_total   += len(y)

        tr_acc = tr_correct / tr_total
        va_acc = va_correct / va_total
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(va_acc)
        history['train_loss'].append(tr_loss / tr_total)
        history['val_loss'].append(va_loss / va_total)

        if va_acc > best_va_acc:
            best_va_acc = va_acc
            torch.save(model.state_dict(),
                       out_dir / 'cnn_13class_best.pt')

        if epoch % 5 == 0 or epoch == 1:
            print(f'  E{epoch:3d}: train={tr_acc*100:.2f}%  val={va_acc*100:.2f}%')

    # Test
    model.load_state_dict(torch.load(out_dir / 'cnn_13class_best.pt',
                                     map_location=device))
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y in te_dl:
            X = X.to(device)
            all_preds.extend(model(X).argmax(1).cpu().numpy())
            all_labels.extend(y.numpy())

    preds  = np.array(all_preds)
    labels = np.array(all_labels)
    acc    = accuracy_score(labels, preds)
    print(f'\n  CNN 13-class test accuracy: {acc*100:.2f}%')
    print(classification_report(labels, preds,
                                 target_names=LABEL_NAMES, digits=4))

    # Save plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history['train_acc'], label='train')
    axes[0].plot(history['val_acc'],   label='val')
    axes[0].set_title('Accuracy'); axes[0].legend()
    axes[1].plot(history['train_loss'], label='train')
    axes[1].plot(history['val_loss'],   label='val')
    axes[1].set_title('Loss'); axes[1].legend()
    plt.tight_layout()
    fig.savefig(out_dir / 'cnn_13class_training.png', dpi=150)
    plt.close(fig)

    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, ax=ax)
    ax.set_title(f'CNN 13-class ({acc*100:.2f}%)')
    plt.tight_layout()
    fig.savefig(out_dir / 'cm_cnn_13class.png', dpi=150)
    plt.close(fig)

    json.dump(history, open(out_dir / 'cnn_13class_history.json', 'w'))
    return acc


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    if args.mode in ('features', 'all'):
        # ── Load Exp1 features ────────────────────────────────────────────
        print('\n[1] Loading Exp1 feature cache...')
        X_exp1, y_exp1, _ = load_exp1_cache(args.exp1_cache)

        # ── Load / extract NonPD features ─────────────────────────────────
        print('\n[2] Loading NonPD features...')
        nonpd_cache = out_dir / 'nonpd_features_cache.npz'
        X_nonpd, _ = load_nonpd_features(args.nonpd_dir,
                                          cache_path=str(nonpd_cache))

        # ── Build 13-class dataset ─────────────────────────────────────────
        print('\n[3] Building 13-class dataset...')
        X_all, y_all = build_13class_dataset(
            X_exp1, y_exp1, X_nonpd,
            max_nonpd=args.max_nonpd
        )
        tr_idx, va_idx, te_idx = stratified_split(X_all, y_all)
        X_tr, y_tr = X_all[tr_idx], y_all[tr_idx]
        X_te, y_te = X_all[te_idx], y_all[te_idx]

        # Class distribution
        print('\nClass distribution (test set):')
        unique, counts = np.unique(y_te, return_counts=True)
        for u, c in zip(unique, counts):
            print(f'  {LABEL_NAMES[u]:8s}: {c}')

        # ── Train RF ──────────────────────────────────────────────────────
        print('\n[4] Training RF (13-class)...')
        rf = train_rf(X_tr, y_tr)
        acc_rf, _ = evaluate_clf(rf, X_te, y_te, 'RF 13-class', out_dir)
        results['RF_13class'] = round(acc_rf * 100, 4)

        # ── Train SVM ─────────────────────────────────────────────────────
        print('\n[5] Training SVM (13-class)...')
        svm = train_svm(X_tr, y_tr)
        acc_svm, _ = evaluate_clf(svm, X_te, y_te, 'SVM 13-class', out_dir)
        results['SVM_13class'] = round(acc_svm * 100, 4)

        # ── Voltage-invariant RF ──────────────────────────────────────────
        # For this test we only evaluate on Exp1 samples (NonPD has no voltage)
        print('\n[6] Voltage-invariant RF (13-class, Exp1 only)...')
        # Reload with voltage info
        c      = np.load(args.exp1_cache)
        volts  = c['volts']
        X_nonpd_sub = X_nonpd[:args.max_nonpd] if args.max_nonpd else X_nonpd
        y_nonpd_sub = np.full(len(X_nonpd_sub), NONPD_LABEL, dtype=np.int32)
        v_nonpd     = np.full(len(X_nonpd_sub), 5, dtype=np.int32)  # neutral volt

        X_v = np.vstack([X_exp1, X_nonpd_sub])
        y_v = np.concatenate([y_exp1, y_nonpd_sub])
        v_v = np.concatenate([volts,  v_nonpd])

        tr_vi = np.where(v_v <= 8)[0]
        te_vi = np.where(v_v >= 9)[0]
        # For NonPD (volt=5) they land in train — fine

        rf_vi = train_rf(X_v[tr_vi], y_v[tr_vi])
        preds_vi = rf_vi.predict(X_v[te_vi])
        acc_vi   = accuracy_score(y_v[te_vi], preds_vi)
        print(f'  RF 13-class volt-invariant: {acc_vi*100:.2f}%')
        results['RF_13class_volt_invariant'] = round(acc_vi * 100, 4)

    if args.mode in ('cnn', 'all'):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'\n[CNN] Device: {device}')
        acc_cnn = train_cnn_13(
            data_dir   = args.data_dir,
            nonpd_dir  = args.nonpd_dir,
            out_dir    = out_dir,
            device     = device,
            epochs     = args.epochs,
            batch_size = args.batch_size,
            max_nonpd  = args.max_nonpd,
        )
        results['CNN_13class'] = round(acc_cnn * 100, 4)

    # ── Summary ───────────────────────────────────────────────────────────────
    prev = {
        'RF v2 top40 (12-class)':        99.92,
        'SVM v2 top40 (12-class)':       99.82,
        'RF v2 volt-invariant (12-class)':99.95,
    }
    all_res = {**prev, **results}

    print('\n' + '='*60)
    print('  RESULTS SUMMARY — 12-class vs 13-class')
    print('='*60)
    for k, v in all_res.items():
        marker = ' ◄ NEW' if k in results else ''
        print(f'  {k:<45s} {v:.2f}%{marker}')
    print('='*60)

    fig, ax = plt.subplots(figsize=(12, 5))
    names = list(all_res.keys())
    vals  = list(all_res.values())
    colors = ['#aaaaaa'] * len(prev) + ['steelblue'] * len(results)
    bars   = ax.bar(names, vals, color=colors)
    ax.set_ylim(min(vals) - 2, 101)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('12-class vs 13-class (with NonPD)')
    ax.set_xticklabels(names, rotation=35, ha='right', fontsize=9)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.1, f'{v:.2f}%',
                ha='center', fontsize=8)
    plt.tight_layout()
    fig.savefig(out_dir / 'comparison_13class.png', dpi=150)
    plt.close(fig)

    json.dump(results, open(out_dir / 'results_13class.json', 'w'), indent=2)
    print(f'\nAll outputs saved to: {out_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',   default='data/raw/measurements')
    parser.add_argument('--nonpd_dir',  default='data/raw/nonpd')
    parser.add_argument('--exp1_cache', default='results/baseline_v2/features_cache.npz')
    parser.add_argument('--out_dir',    default='results/baseline_13class')
    parser.add_argument('--mode',       default='all',
                        choices=['features','cnn','all'])
    parser.add_argument('--max_nonpd',  type=int, default=5000,
                        help='Max NonPD samples to use (avoids class imbalance)')
    parser.add_argument('--epochs',     type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()
    main(args)