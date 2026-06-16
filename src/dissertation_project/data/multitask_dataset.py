# src/dataset_multitask.py

import os
import numpy as np
import scipy.io
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from pathlib import Path

from dataset import POSITIONS, POS_TO_LABEL, load_mat_file


def build_file_list_multitask(root_dir):
    """
    Walk folder tree, return (filepath, position_label, voltage_float).
    Folder structure: root/{position}/{voltage}/*.mat
    """
    root    = Path(root_dir)
    samples = []

    for pos_folder in sorted(root.iterdir()):
        if not pos_folder.is_dir():
            continue
        try:
            position = int(pos_folder.name)
        except ValueError:
            continue
        if position not in POS_TO_LABEL:
            continue
        pos_label = POS_TO_LABEL[position]

        for volt_folder in sorted(pos_folder.iterdir()):
            if not volt_folder.is_dir():
                continue
            # Parse voltage: "5V" → 5.0
            try:
                voltage = float(volt_folder.name.replace('V', ''))
            except ValueError:
                continue

            for mat_file in sorted(volt_folder.glob('*.mat')):
                samples.append((str(mat_file), pos_label, voltage))

    n_pos  = len(set(s[1] for s in samples))
    volts  = sorted(set(s[2] for s in samples))
    print(f"Found {len(samples)} files | "
          f"{n_pos} positions | voltages: {volts}")
    return samples


class MultiTaskDataset(Dataset):
    """
    Dataset returning (signal, position_label, voltage).
    position_label : int (0–11)
    voltage        : float normalized to [0, 1] from [1, 10]
    """
    def __init__(self, samples, normalize=True, crop=None):
        self.samples   = samples
        self.normalize = normalize
        self.crop      = crop

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filepath, pos_label, voltage = self.samples[idx]

        signal = load_mat_file(filepath)          # (2, N)

        if self.crop is not None:
            signal = signal[:, self.crop[0]:self.crop[1]]

        if self.normalize:
            for ch in range(signal.shape[0]):
                mx = np.abs(signal[ch]).max()
                if mx > 0:
                    signal[ch] = signal[ch] / mx

        signal    = torch.tensor(signal,   dtype=torch.float32)
        pos_label = torch.tensor(pos_label, dtype=torch.long)
        # Normalize voltage to [0,1]: (v-1)/9
        volt_norm = torch.tensor((voltage - 1.0) / 9.0, dtype=torch.float32)

        return signal, pos_label, volt_norm


def get_dataloaders_multitask(root_dir,
                               batch_size=32,
                               crop=(3500, 7500),
                               val_size=0.15,
                               test_size=0.15,
                               num_workers=0,
                               random_state=42):
    all_samples = build_file_list_multitask(root_dir)
    labels      = [s[1] for s in all_samples]

    train_val, test = train_test_split(
        all_samples, test_size=test_size,
        stratify=labels, random_state=random_state)

    labels_tv  = [s[1] for s in train_val]
    val_ratio  = val_size / (1.0 - test_size)
    train, val = train_test_split(
        train_val, test_size=val_ratio,
        stratify=labels_tv, random_state=random_state)

    print(f"Split → Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")

    train_ds = MultiTaskDataset(train, normalize=True, crop=crop)
    val_ds   = MultiTaskDataset(val,   normalize=True, crop=crop)
    test_ds  = MultiTaskDataset(test,  normalize=True, crop=crop)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    train_l, val_l, test_l = get_dataloaders_multitask(
        'data/raw/measurements')
    signals, pos, volt = next(iter(train_l))
    print(f"Signal: {signals.shape}")
    print(f"Position labels: {pos[:8].tolist()}")
    print(f"Voltage (normalized): {volt[:8].tolist()}")