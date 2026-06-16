# src/dataset.py

import os
import numpy as np
import scipy.io
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from pathlib import Path


# ─── Constants ────────────────────────────────────────────────────────────────

POSITIONS = [100, 200, 300, 500, 700, 900, 1000, 1300, 1500, 1600, 1800, 1900]
POS_TO_LABEL = {pos: idx for idx, pos in enumerate(POSITIONS)}
# e.g. {100: 0, 200: 1, 300: 2, ..., 1900: 11}


# ─── File loader ──────────────────────────────────────────────────────────────

def load_mat_file(filepath):
    """
    Load a single .mat file and return the 2-channel signal as a numpy array.
    Returns: np.ndarray of shape (2, 20000), dtype float32
    """
    mat = scipy.io.loadmat(filepath)
    tpd = mat['tpd']
    data = tpd['Data'][0, 0]          # shape (2, N)
    return data.astype(np.float32)


# ─── Dataset builder ──────────────────────────────────────────────────────────

def build_file_list(root_dir):
    """
    Walk the folder tree and return a list of (filepath, label) tuples.
    Folder structure: root_dir/{position}/{voltage}/*.mat
    Label = index of position in POSITIONS list.
    """
    root = Path(root_dir)
    samples = []

    for pos_folder in sorted(root.iterdir()):
        if not pos_folder.is_dir():
            continue
        try:
            position = int(pos_folder.name)
        except ValueError:
            continue  # skip non-numeric folders

        if position not in POS_TO_LABEL:
            print(f"  [Warning] Unknown position folder: {pos_folder.name}, skipping.")
            continue

        label = POS_TO_LABEL[position]

        for volt_folder in sorted(pos_folder.iterdir()):
            if not volt_folder.is_dir():
                continue

            for mat_file in sorted(volt_folder.glob('*.mat')):
                samples.append((str(mat_file), label))

    print(f"Found {len(samples)} files across {len(POSITIONS)} positions.")
    return samples


# ─── PyTorch Dataset ──────────────────────────────────────────────────────────

class PDDataset(Dataset):
    """
    PyTorch Dataset for Partial Discharge localization.

    Each sample is a 2-channel time-domain waveform of shape (2, N).
    Label is the injection position index (0–11).

    Args:
        samples     : list of (filepath, label) tuples
        normalize   : if True, normalize each channel to [-1, 1]
        crop        : if not None, tuple (start, end) sample indices to crop
    """

    def __init__(self, samples, normalize=True, crop=None):
        self.samples   = samples
        self.normalize = normalize
        self.crop      = crop

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filepath, label = self.samples[idx]

        signal = load_mat_file(filepath)           # (2, 20000)

        if self.crop is not None:
            start, end = self.crop
            signal = signal[:, start:end]          # (2, crop_len)

        if self.normalize:
            # Normalize each channel independently to [-1, 1]
            for ch in range(signal.shape[0]):
                ch_max = np.abs(signal[ch]).max()
                if ch_max > 0:
                    signal[ch] = signal[ch] / ch_max

        signal = torch.tensor(signal, dtype=torch.float32)  # (2, N)
        label  = torch.tensor(label,  dtype=torch.long)

        return signal, label


# ─── Split & DataLoader factory ───────────────────────────────────────────────

def get_dataloaders(root_dir,
                    batch_size=32,
                    normalize=True,
                    crop=None,
                    val_size=0.15,
                    test_size=0.15,
                    num_workers=0,
                    random_state=42):
    """
    Build train / val / test DataLoaders from the folder tree.

    Splits are stratified by label so every position is represented equally.

    Returns: train_loader, val_loader, test_loader, class_names
    """

    all_samples = build_file_list(root_dir)

    labels = [s[1] for s in all_samples]

    # First split off test set
    train_val, test = train_test_split(
        all_samples,
        test_size=test_size,
        stratify=labels,
        random_state=random_state
    )

    # Then split train/val from the remainder
    labels_tv = [s[1] for s in train_val]
    val_ratio  = val_size / (1.0 - test_size)

    train, val = train_test_split(
        train_val,
        test_size=val_ratio,
        stratify=labels_tv,
        random_state=random_state
    )

    print(f"\nSplit sizes → Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")

    train_ds = PDDataset(train, normalize=normalize, crop=crop)
    val_ds   = PDDataset(val,   normalize=normalize, crop=crop)
    test_ds  = PDDataset(test,  normalize=normalize, crop=crop)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    class_names = [f"{p}m" for p in POSITIONS]

    return train_loader, val_loader, test_loader, class_names



def get_dataloaders_voltage_split(root_dir,
                                   train_voltages=None,
                                   test_voltages=None,
                                   batch_size=32,
                                   normalize=True,
                                   crop=None,
                                   num_workers=0):
    """
    Split data by voltage level instead of random split.
    Train on train_voltages, test on test_voltages.
    """
    if train_voltages is None:
        train_voltages = ['1V','2V','3V','4V','5V','6V','7V','8V']
    if test_voltages is None:
        test_voltages  = ['9V','10V']

    root = Path(root_dir)
    train_samples, test_samples = [], []

    for pos_folder in sorted(root.iterdir()):
        if not pos_folder.is_dir():
            continue
        try:
            position = int(pos_folder.name)
        except ValueError:
            continue
        if position not in POS_TO_LABEL:
            continue
        label = POS_TO_LABEL[position]

        for volt_folder in sorted(pos_folder.iterdir()):
            if not volt_folder.is_dir():
                continue
            volt_name = volt_folder.name

            for mat_file in sorted(volt_folder.glob('*.mat')):
                if volt_name in train_voltages:
                    train_samples.append((str(mat_file), label))
                elif volt_name in test_voltages:
                    test_samples.append((str(mat_file), label))

    # Use 15% of train as val (random, within train voltages)
    from sklearn.model_selection import train_test_split
    labels_tr = [s[1] for s in train_samples]
    train, val = train_test_split(train_samples, test_size=0.15,
                                   stratify=labels_tr, random_state=42)

    print(f"\nVoltage-invariant split:")
    print(f"  Train voltages : {train_voltages}")
    print(f"  Test  voltages : {test_voltages}")
    print(f"  Train: {len(train)} | Val: {len(val)} | Test: {len(test_samples)}")

    train_ds = PDDataset(train, normalize=normalize, crop=crop)
    val_ds   = PDDataset(val,   normalize=normalize, crop=crop)
    test_ds  = PDDataset(test_samples, normalize=normalize, crop=crop)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)

    class_names = [f"{p}m" for p in POSITIONS]
    return train_loader, val_loader, test_loader, class_names


# ─── Quick test ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else 'data/raw/measurements'

    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        root_dir    = root,
        batch_size  = 32,
        normalize   = True,
        crop        = (3000, 7000),   # 40µs window around trigger — adjust later
    )

    print(f"\nClass names: {class_names}")

    # Peek at one batch
    signals, labels = next(iter(train_loader))
    print(f"\nBatch signal shape : {signals.shape}")   # (32, 2, 4000)
    print(f"Batch label shape  : {labels.shape}")    # (32,)
    print(f"Label values       : {labels[:8].tolist()}")
    print(f"Signal range       : [{signals.min():.3f}, {signals.max():.3f}]")