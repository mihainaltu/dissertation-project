"""
Experiment 2 — Pulse Feature Extraction
Loads all lab files from the directory structure:
    <root>/1/         -> DAMAGED  (frequency 1 MHz)
    <root>/1-conn/    -> HEALTHY
    <root>/2/         -> DAMAGED
    <root>/2-conn/    -> HEALTHY
    ...
    <root>/20/        -> DAMAGED
    <root>/20-conn/   -> HEALTHY

Usage:
    python3 exp2_extract.py --root "data/raw/sames-cable-data" --out exp2_features.csv
"""

import scipy.io as sio
import numpy as np
import pywt
import os
import csv
import argparse
from pathlib import Path

PRE     = 25
POST    = 300
THRESH  = 0.1
MIN_SEP = 500


def extract_pulses(filepath, pre=PRE, post=POST, thresh=THRESH, min_sep=MIN_SEP):
    mat = sio.loadmat(filepath)
    key = [k for k in mat.keys() if not k.startswith('_')][0]
    d   = mat[key][0, 0]
    fs  = float(d['SampleFrequency'].flat[0])

    raw = d['Data']
    sig = raw[0].astype(float) if raw.shape[0] == 1 else raw[:, 0].astype(float)

    above = np.abs(sig) > thresh
    edges = np.where(np.diff(above.astype(int)) == 1)[0]
    if len(edges) > 1:
        gaps = np.diff(edges)
        candidates = [int(edges[0])] + [
            int(edges[i + 1]) for i, g in enumerate(gaps) if g > min_sep
        ]
    else:
        candidates = [int(e) for e in edges]

    pulses = []
    for pos in candidates:
        s    = max(0, pos - 500)
        e    = min(len(sig), pos + 500)
        peak = s + int(np.argmax(np.abs(sig[s:e])))
        if peak - pre < 0 or peak + post > len(sig):
            continue
        pulses.append((peak, sig[peak - pre: peak + post].copy(), fs))
    return pulses


def extract_features(window, fs, pre=PRE):
    feats     = {}
    peak_idx  = pre
    peak_val  = window[peak_idx]
    abs_win   = np.abs(window)
    n         = len(window)

    feats['peak_amplitude'] = float(peak_val)
    feats['peak_abs']       = float(abs_win[peak_idx])

    rising  = window[:peak_idx]
    p10, p90 = 0.1 * abs(peak_val), 0.9 * abs(peak_val)
    a10 = np.where(np.abs(rising) >= p10)[0]
    a90 = np.where(np.abs(rising) >= p90)[0]
    feats['rise_time_us'] = float((a90[0] - a10[0]) / fs * 1e6) if len(a10) and len(a90) else 0.0

    tail = window[peak_idx + 20:]
    feats['tail_energy']   = float(np.sum(tail ** 2))
    feats['tail_rms']      = float(np.sqrt(np.mean(tail ** 2)))
    feats['tail_std']      = float(np.std(tail))
    feats['tail_max_abs']  = float(np.max(np.abs(tail)))
    feats['tail_mean_abs'] = float(np.mean(np.abs(tail)))
    feats['tail_skew']     = float(np.mean(((tail - tail.mean()) / tail.std()) ** 3)) if tail.std() > 0 else 0.0
    feats['tail_kurtosis'] = float(np.mean(((tail - tail.mean()) / tail.std()) ** 4)) if tail.std() > 0 else 0.0
    feats['tail_zcr']      = float(np.sum(np.diff(np.sign(tail)) != 0)) / len(tail)

    slope, _ = np.polyfit(np.arange(len(tail)), np.log(np.abs(tail) + 1e-10), 1)
    feats['tail_decay_rate'] = float(slope)

    for pct, name in [(0.5, 'width_50pct'), (0.1, 'width_10pct')]:
        above = np.where(abs_win > pct * abs(peak_val))[0]
        feats[name] = float(len(above)) / fs * 1e6 if len(above) else 0.0

    pre_e = float(np.sum(window[:peak_idx] ** 2))
    post_e = float(np.sum(window[peak_idx:] ** 2))
    feats['energy_asymmetry']       = pre_e / (post_e + 1e-10)
    feats['reflection_ratio']       = float(np.max(np.abs(tail))) / (abs(peak_val) + 1e-10)
    ref_idx = np.where(np.abs(tail) > 0.05 * abs(peak_val))[0]
    feats['time_to_reflection_us']  = float(ref_idx[0]) / fs * 1e6 if len(ref_idx) else 1.5

    feats['full_rms']      = float(np.sqrt(np.mean(window ** 2)))
    feats['full_energy']   = float(np.sum(window ** 2))
    feats['full_skew']     = float(np.mean(((window - window.mean()) / window.std()) ** 3)) if window.std() > 0 else 0.0
    feats['full_kurtosis'] = float(np.mean(((window - window.mean()) / window.std()) ** 4)) if window.std() > 0 else 0.0

    fft_c  = np.fft.rfft(window)
    freqs  = np.fft.rfftfreq(n, 1 / fs)
    power  = np.abs(fft_c) ** 2
    tp     = power.sum() + 1e-10
    feats['spectral_centroid_mhz']  = float(np.sum(freqs * power) / tp / 1e6)
    feats['spectral_bandwidth_mhz'] = float(np.sqrt(np.sum(((freqs - feats['spectral_centroid_mhz'] * 1e6) ** 2) * power) / tp) / 1e6)
    feats['peak_freq_mhz']          = float(freqs[np.argmax(power)] / 1e6)
    for i, (lo, hi) in enumerate([(0, 10e6), (10e6, 30e6), (30e6, 60e6), (60e6, 100e6)]):
        feats[f'band_energy_{i}'] = float(power[(freqs >= lo) & (freqs < hi)].sum() / tp)
    p_norm = power / tp; p_norm = p_norm[p_norm > 0]
    feats['spectral_entropy'] = float(-np.sum(p_norm * np.log(p_norm)))

    for i, c in enumerate(pywt.wavedec(tail, 'db4', level=4)):
        feats[f'wavelet_energy_{i}'] = float(np.sum(c ** 2))
        feats[f'wavelet_std_{i}']    = float(np.std(c))

    return feats


def discover_files(root):
    """
    Walk root looking for folders named 1..20 (DAMAGED) and 1-conn..20-conn (HEALTHY).
    Returns list of (filepath, label, freq_mhz).
    """
    root = Path(root)
    pairs = []
    for freq in range(1, 21):
        for folder, label in [(str(freq), 'DAMAGED'), (f'{freq}-conn', 'HEALTHY')]:
            d = root / folder
            if not d.exists():
                continue
            mats = sorted(d.glob('*.mat'))
            if not mats:
                print(f"  WARNING: no .mat files in {d}")
                continue
            for f in mats:
                pairs.append((str(f), label, freq))
    return pairs


def process_files(file_label_freq_triples, out_csv=None):
    all_rows = []
    for fpath, label, freq in file_label_freq_triples:
        try:
            pulses = extract_pulses(fpath)
        except Exception as ex:
            print(f"  ERROR reading {fpath}: {ex}")
            continue
        print(f"  {Path(fpath).parent.name}/{Path(fpath).name} [{label}, {freq}MHz]: {len(pulses)} pulses")
        for pulse_idx, (peak, window, fs) in enumerate(pulses):
            feats = extract_features(window, fs)
            feats['label']      = label
            feats['file']       = Path(fpath).name
            feats['folder']     = Path(fpath).parent.name
            feats['freq_mhz']   = freq
            feats['peak_sample']= peak
            feats['pulse_idx']  = pulse_idx
            all_rows.append(feats)

    if out_csv and all_rows:
        keys = list(all_rows[0].keys())
        with open(out_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nSaved {len(all_rows)} rows -> {out_csv}")

    return all_rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, help='Root data directory')
    parser.add_argument('--out',  default='exp2_features.csv', help='Output CSV')
    args = parser.parse_args()

    print(f"Discovering files in: {args.root}")
    triples = discover_files(args.root)
    print(f"Found {len(triples)} files across {len(set(t[2] for t in triples))} frequencies\n")
    process_files(triples, out_csv=args.out)