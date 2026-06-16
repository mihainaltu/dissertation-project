# src/dataset_exp2.py

import os
import numpy as np
import scipy.io
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from pathlib import Path
from scipy.signal import hilbert


# ── Constants ─────────────────────────────────────────────────────────────────

FS       = 200e6       # 200 MHz
PRE      = 1000000     # pre-trigger samples
CROP_START = PRE - 20          # −0.1 µs
CROP_END   = PRE + 400        # +5 µs
CROP_LEN   = 420              # samples

LABEL_MAP = {
    'healthy': 1,   # -conn folders
    'damaged': 0,   # non-conn folders
}


# ── File loader ───────────────────────────────────────────────────────────────

def load_exp2_file(filepath, global_scale=1.0):
    """
    Load crop, compute Hilbert envelope, scale by global constant.
    NO per-sample normalization — preserves amplitude differences.
    """
    mat      = scipy.io.loadmat(filepath)
    tpd      = mat['tpd']
    data     = tpd['Data'][0, 0].flatten().astype(np.float32)
    pre      = int(tpd['PreSampleCount'][0, 0][0, 0])
    crop     = data[pre - 20 : pre + 400]
    envelope = np.abs(hilbert(crop)).astype(np.float32)
    return (envelope / global_scale).reshape(1, -1)   # (1, 420)


# ── File list builder ─────────────────────────────────────────────────────────

def build_file_list_exp2(root_dir):
    """
    Walk sames-cable-data folder tree.
    Folders ending in '-conn' → healthy (1)
    Folders without '-conn'   → damaged (0)
    Returns: list of (filepath, label) tuples
    """
    root    = Path(root_dir)
    samples = []

    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue

        name = folder.name
        if name.endswith('-conn'):
            label = 1   # healthy
        else:
            try:
                int(name)   # must be a numeric folder
                label = 0   # damaged
            except ValueError:
                continue    # skip unknown folders

        for mat_file in sorted(folder.glob('*.mat')):
            samples.append((str(mat_file), label))

    n_healthy = sum(1 for _, l in samples if l == 1)
    n_damaged = sum(1 for _, l in samples if l == 0)
    print(f"Found {len(samples)} files — Healthy: {n_healthy} | Damaged: {n_damaged}")
    return samples


# ── Augmentation ──────────────────────────────────────────────────────────────

def augment_signal(signal, jitter_max=50, noise_std=0.005):
    """
    Apply one random augmentation to a (1, N) signal.
    Options: jitter, time-reverse, add noise, combined jitter+noise
    """
    aug = np.random.choice(['jitter', 'reverse', 'noise', 'jitter+noise'])

    sig = signal.copy()

    if 'jitter' in aug:
        shift = np.random.randint(-jitter_max, jitter_max)
        sig   = np.roll(sig, shift, axis=1)

    if 'reverse' in aug:
        sig = sig[:, ::-1].copy()

    if 'noise' in aug:
        sig = sig + np.random.normal(0, noise_std, sig.shape).astype(np.float32)

    return sig

def compute_global_scale(samples):
    """
    Compute global scale = mean peak amplitude across training files.
    Call this on training set only, then pass scale to all datasets.
    """
    peaks = []
    for filepath, _ in samples:
        mat  = scipy.io.loadmat(filepath)
        tpd  = mat['tpd']
        data = tpd['Data'][0, 0].flatten().astype(np.float32)
        pre  = int(tpd['PreSampleCount'][0, 0][0, 0])
        crop = data[pre - 20 : pre + 400]
        peaks.append(np.abs(crop).max())
    return float(np.mean(peaks))


# ── Dataset ───────────────────────────────────────────────────────────────────

class CableDataset(Dataset):
    def __init__(self, samples, global_scale=1.0, augment=False, aug_factor=10):
        self.global_scale = global_scale
        self.augment      = augment

        if augment and aug_factor > 1:
            augmented = []
            for filepath, label in samples:
                augmented.append((filepath, label, False))
                for _ in range(aug_factor - 1):
                    augmented.append((filepath, label, True))
            self.samples = augmented
        else:
            self.samples = [(fp, lb, False) for fp, lb in samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filepath, label, do_aug = self.samples[idx]
        signal = load_exp2_file(filepath, self.global_scale)   # (1, 420)

        if do_aug:
            signal = augment_signal(signal)

        signal = torch.tensor(signal, dtype=torch.float32)
        label  = torch.tensor(label,  dtype=torch.long)
        return signal, label


# ── DataLoader factory ────────────────────────────────────────────────────────

def get_dataloaders_exp2(root_dir, batch_size=16, aug_factor=10,
                          val_size=0.15, test_size=0.15,
                          num_workers=0, random_state=42):

    all_samples = build_file_list_exp2(root_dir)
    labels      = [s[1] for s in all_samples]

    train_val, test = train_test_split(all_samples, test_size=test_size,
                                       stratify=labels, random_state=random_state)
    labels_tv  = [s[1] for s in train_val]
    val_ratio  = val_size / (1.0 - test_size)
    train, val = train_test_split(train_val, test_size=val_ratio,
                                  stratify=labels_tv, random_state=random_state)

    # Compute scale from training set only
    print("Computing global scale from training set...")
    global_scale = compute_global_scale(train)
    print(f"Global scale: {global_scale:.5f} V")

    train_ds = CableDataset(train, global_scale=global_scale,
                             augment=True,  aug_factor=aug_factor)
    val_ds   = CableDataset(val,   global_scale=global_scale, augment=False)
    test_ds  = CableDataset(test,  global_scale=global_scale, augment=False)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, global_scale


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else 'data/raw/sames-cable-data'

    train_loader, val_loader, test_loader = get_dataloaders_exp2(root)

    signals, labels = next(iter(train_loader))
    print(f"\nBatch signal shape : {signals.shape}")
    print(f"Batch label shape  : {labels.shape}")
    print(f"Labels in batch    : {labels.tolist()}")
    print(f"Signal range       : [{signals.min():.3f}, {signals.max():.3f}]")