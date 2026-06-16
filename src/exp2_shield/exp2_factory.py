"""
Experiment 2 - Factory Change Detection
Usage:
    python3 exp2_factory.py --root "data/raw/sames-real" --out_dir results_factory

Loads all .mat files chronologically, extracts features per pulse,
runs change detection (CUSUM + PCA), and generates thesis-quality plots.
"""

import scipy.io as sio
import numpy as np
import pandas as pd
import pywt
import os
import argparse
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pathlib import Path

# ── Style ──────────────────────────────────────────────────────────────────
COLORS = {
    'blue':   '#2563EB',
    'green':  '#16A34A',
    'red':    '#DC2626',
    'orange': '#EA580C',
    'gray':   '#6B7280',
    'light':  '#F3F4F6',
}

plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'font.size':         11,
    'axes.titlesize':    13,
    'axes.labelsize':    12,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'figure.dpi':        150,
    'savefig.dpi':       300,
    'savefig.bbox':      'tight',
    'savefig.facecolor': 'white',
})

# ── Parameters ─────────────────────────────────────────────────────────────
PRE     = 25
POST    = 300
THRESH  = 0.35   # V - factory Ch2 (galvanic) has much higher amplitude
MIN_SEP = 10000  # samples - pulses repeat every ~20ms = 4,000,000 samples at 200MHz

FEATURE_COLS = [
    'peak_amplitude', 'peak_abs', 'rise_time_us',
    'tail_energy', 'tail_rms', 'tail_std', 'tail_max_abs', 'tail_mean_abs',
    'tail_skew', 'tail_kurtosis', 'tail_zcr', 'tail_decay_rate',
    'width_50pct', 'width_10pct', 'energy_asymmetry',
    'reflection_ratio', 'time_to_reflection_us',
    'full_rms', 'full_energy', 'full_skew', 'full_kurtosis',
    'spectral_centroid_mhz', 'spectral_bandwidth_mhz', 'peak_freq_mhz',
    'band_energy_0', 'band_energy_1', 'band_energy_2', 'band_energy_3',
    'spectral_entropy',
    'wavelet_energy_0', 'wavelet_std_0',
    'wavelet_energy_1', 'wavelet_std_1',
    'wavelet_energy_2', 'wavelet_std_2',
    'wavelet_energy_3', 'wavelet_std_3',
    'wavelet_energy_4', 'wavelet_std_4',
]


# ── Extraction ─────────────────────────────────────────────────────────────

def extract_pulses(filepath, pre=PRE, post=POST, thresh=THRESH, min_sep=MIN_SEP):
    mat = sio.loadmat(filepath)
    key = [k for k in mat.keys() if not k.startswith('_')][0]
    d   = mat[key][0, 0]
    fs  = float(d['SampleFrequency'].flat[0])

    raw = d['Data']
    # factory: shape (N, 4) - use Ch2 (galvanic, index 1)
    # lab:     shape (1, N) - single channel
    if raw.shape[0] == 1:
        sig = raw[0].astype(float)
    else:
        sig = raw[:, 1].astype(float)

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
    feats    = {}
    peak_idx = pre
    peak_val = window[peak_idx]
    abs_win  = np.abs(window)
    n        = len(window)

    feats['peak_amplitude'] = float(peak_val)
    feats['peak_abs']       = float(abs_win[peak_idx])

    rising  = window[:peak_idx]
    a10 = np.where(np.abs(rising) >= 0.1 * abs(peak_val))[0]
    a90 = np.where(np.abs(rising) >= 0.9 * abs(peak_val))[0]
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

    pre_e  = float(np.sum(window[:peak_idx] ** 2))
    post_e = float(np.sum(window[peak_idx:] ** 2))
    feats['energy_asymmetry']      = pre_e / (post_e + 1e-10)
    feats['reflection_ratio']      = float(np.max(np.abs(tail))) / (abs(peak_val) + 1e-10)
    ref_idx = np.where(np.abs(tail) > 0.05 * abs(peak_val))[0]
    feats['time_to_reflection_us'] = float(ref_idx[0]) / fs * 1e6 if len(ref_idx) else 1.5

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


def load_factory_files(root):
    """Return sorted list of .mat files from root (chronological by filename)."""
    root  = Path(root)
    files = sorted(root.glob('*.mat'))
    print(f"Found {len(files)} .mat files in {root}")
    return files


def process_factory(files):
    """Extract features from all factory files. Returns DataFrame."""
    rows = []
    for file_idx, fpath in enumerate(files):
        try:
            pulses = extract_pulses(str(fpath))
        except Exception as ex:
            print(f"  ERROR {fpath.name}: {ex}")
            continue
        print(f"  [{file_idx+1:02d}/{len(files)}] {fpath.name}: {len(pulses)} pulses")
        for pulse_idx, (peak, window, fs) in enumerate(pulses):
            feats = extract_features(window, fs)
            feats['file']       = fpath.name
            feats['file_idx']   = file_idx      # chronological index
            feats['pulse_idx']  = pulse_idx
            rows.append(feats)
    return pd.DataFrame(rows)


# ── Change Detection ────────────────────────────────────────────────────────

def cusum(signal, drift=0.5):
    """
    Two-sided CUSUM on a standardised signal.
    Stops accumulating after threshold is first exceeded.
    Returns (cusum_pos, cusum_neg, change_point_idx or None).
    """
    mu    = np.mean(signal)
    sigma = np.std(signal) + 1e-10
    z     = (signal - mu) / sigma

    thresh = 5
    cp, cn = np.zeros(len(z)), np.zeros(len(z))
    cp_idx = None
    for i in range(1, len(z)):
        cp[i] = max(0, cp[i-1] + z[i] - drift)
        cn[i] = max(0, cn[i-1] - z[i] - drift)
        if cp_idx is None and ((cp[i] > thresh) or (cn[i] > thresh)):
            cp_idx = i
            break  # stop at first crossing

    # zero out everything after change point
    if cp_idx is not None:
        cp[cp_idx+1:] = np.nan
        cn[cp_idx+1:] = np.nan

    return cp, cn, cp_idx


def pca_distance(df, feature_cols, n_components=2):
    """
    PCA on file-level mean features, return PC coordinates and
    Mahalanobis distance from centroid of first 3 files (assumed healthy baseline).
    """
    # aggregate per file
    file_feats = df.groupby('file_idx')[feature_cols].mean()
    file_order = sorted(file_feats.index)
    X = file_feats.loc[file_order].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=n_components)
    coords = pca.fit_transform(X_scaled)

    # Mahalanobis-like distance from baseline centroid (first 3 files)
    baseline = coords[:3]
    centroid = baseline.mean(axis=0)
    diffs    = coords - centroid
    dist     = np.sqrt((diffs ** 2).sum(axis=1))

    return coords, dist, pca.explained_variance_ratio_, file_order


# ── Plots ───────────────────────────────────────────────────────────────────

def plot_feature_timeline(df, feature_cols, change_point, out_dir, top_n=6):
    """Plot top N most variable features over time, truncated at change point."""
    file_mean = df.groupby('file_idx')[feature_cols].mean().reset_index()
    file_mean = file_mean.sort_values('file_idx')
    if change_point is not None:
        file_mean = file_mean[file_mean['file_idx'] <= change_point]

    variances = file_mean[feature_cols].var().sort_values(ascending=False)
    top_feats = variances.index[:top_n].tolist()

    fig, axes = plt.subplots(top_n, 1, figsize=(12, 2.5 * top_n), sharex=True)

    for ax, feat in zip(axes, top_feats):
        vals = file_mean[feat].values
        idxs = file_mean['file_idx'].values
        ax.plot(idxs, vals, 'o-', color=COLORS['blue'], linewidth=1.5, markersize=5)
        if change_point is not None:
            ax.axvline(change_point, color=COLORS['red'], linewidth=1.5,
                       linestyle='--', label='Change point')
        ax.set_ylabel(feat.replace('_', '\n'), fontsize=9)
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel('File Index (chronological)')
    if change_point is not None:
        axes[0].legend(fontsize=9)
    fig.suptitle('Top Feature Trends Over Time - Factory Data\nExperiment 2', fontsize=13)
    plt.tight_layout()
    path = os.path.join(out_dir, 'factory_feature_timeline.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def plot_cusum(cusum_signal, cp_pos, cn_neg, change_point, out_dir):
    """Plot the CUSUM signal with detected change point, truncated after it."""
    # truncate display at change point
    end = (change_point + 1) if change_point is not None else len(cusum_signal)
    sig_trunc = cusum_signal[:end]
    cp_trunc  = cp_pos[:end]
    cn_trunc  = cn_neg[:end]
    x = np.arange(end)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    axes[0].plot(x, sig_trunc, 'o-', color=COLORS['blue'], linewidth=1.5, markersize=5)
    axes[0].set_ylabel('PCA Distance from Baseline')
    axes[0].set_title('Signal Used for CUSUM')
    axes[0].grid(alpha=0.3)

    axes[1].plot(x, cp_trunc, color=COLORS['green'],  linewidth=1.5, label='CUSUM+')
    axes[1].plot(x, cn_trunc, color=COLORS['orange'], linewidth=1.5, label='CUSUM−')
    axes[1].axhline(5, color='gray', linewidth=1, linestyle=':', label='Threshold (5)')
    if change_point is not None:
        for ax in axes:
            ax.axvline(change_point, color=COLORS['red'], linewidth=2,
                       linestyle='--', label=f'Change point (file {change_point})')
    axes[1].set_ylabel('CUSUM Value')
    axes[1].set_xlabel('File Index (chronological)')
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    fig.suptitle('CUSUM Change Detection - Factory Data\nExperiment 2', fontsize=13)
    plt.tight_layout()
    path = os.path.join(out_dir, 'factory_cusum.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def plot_pca_trajectory(coords, file_order, change_point, explained, out_dir):
    """PCA trajectory coloured by file index, with change point marked."""
    fig, ax = plt.subplots(figsize=(8, 6))
    n = len(coords)
    cmap = plt.cm.RdYlGn_r
    colors = [cmap(i / (n - 1)) for i in range(n)]

    for i in range(n - 1):
        ax.plot(coords[i:i+2, 0], coords[i:i+2, 1],
                '-', color=colors[i], linewidth=1.5, alpha=0.7)

    sc = ax.scatter(coords[:, 0], coords[:, 1], c=range(n),
                    cmap='RdYlGn_r', s=80, zorder=5, edgecolors='white', linewidths=0.5)

    if change_point is not None:
        cp_local = file_order.index(change_point) if change_point in file_order else None
        if cp_local is not None:
            ax.scatter(coords[cp_local, 0], coords[cp_local, 1],
                       s=200, color=COLORS['red'], zorder=6,
                       marker='*', label=f'Change point (file {change_point})')
            ax.legend(fontsize=10)

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('File Index (chronological)')
    ax.set_xlabel(f'PC1 ({explained[0]*100:.1f}% variance)')
    ax.set_ylabel(f'PC2 ({explained[1]*100:.1f}% variance)')
    ax.set_title('PCA Trajectory - Factory Files Over Time\nExperiment 2')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, 'factory_pca_trajectory.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def plot_distance_timeline(dist, file_order, change_point, out_dir):
    """Plot PCA distance from baseline over time, truncated at change point."""
    if change_point is not None:
        cp_local = change_point + 1
        dist = dist[:cp_local]
        file_order = file_order[:cp_local]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(file_order, dist, 'o-', color=COLORS['blue'], linewidth=2, markersize=7)
    ax.fill_between(file_order, dist, alpha=0.1, color=COLORS['blue'])

    if change_point is not None:
        ax.axvline(file_order[-1], color=COLORS['red'], linewidth=2,
                   linestyle='--', label=f'Detected change point (file {file_order[-1]})')
        ax.legend(fontsize=10)

    ax.axvspan(file_order[0], file_order[min(2, len(file_order)-1)],
               alpha=0.08, color=COLORS['green'])
    ax.set_xlabel('File Index (chronological)')
    ax.set_ylabel('Distance from Baseline (PCA space)')
    ax.set_title('Anomaly Score Over Time - Factory Data\nExperiment 2')
    ax.grid(alpha=0.3)
    ax.set_xticks(file_order)
    plt.tight_layout()
    path = os.path.join(out_dir, 'factory_anomaly_score.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',    required=True, help='Factory data folder')
    parser.add_argument('--out_dir', default='results_factory')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 1. Load and extract
    print(f"\nLoading factory files from: {args.root}")
    files = load_factory_files(args.root)
    df    = process_factory(files)
    print(f"\nExtracted {len(df)} pulse samples from {df['file_idx'].nunique()} files")

    # save features
    csv_path = os.path.join(args.out_dir, 'factory_features.csv')
    df.to_csv(csv_path, index=False)
    print(f"Features saved to: {csv_path}")

    # 2. PCA + distance
    print("\nRunning PCA...")
    coords, dist, explained, file_order = pca_distance(df, FEATURE_COLS)
    print(f"PC1: {explained[0]*100:.1f}%  PC2: {explained[1]*100:.1f}%  Total: {sum(explained)*100:.1f}%")

    # 3. CUSUM on PCA distance
    print("\nRunning CUSUM...")
    cp_pos, cn_neg, change_point = cusum(dist)
    if change_point is not None:
        actual_file_idx = file_order[change_point]
        print(f"Change point detected at position {change_point} → file index {actual_file_idx}: {files[actual_file_idx].name}")
    else:
        actual_file_idx = None
        print("No change point detected (CUSUM threshold not exceeded)")

    # 4. Plots
    print("\nGenerating plots...")
    plot_feature_timeline(df, FEATURE_COLS, actual_file_idx, args.out_dir)
    plot_cusum(dist, cp_pos, cn_neg, change_point, args.out_dir)
    plot_pca_trajectory(coords, file_order, actual_file_idx, explained, args.out_dir)
    plot_distance_timeline(dist, file_order, actual_file_idx, args.out_dir)

    print(f"\nAll outputs saved to: {args.out_dir}/")


if __name__ == '__main__':
    main()
