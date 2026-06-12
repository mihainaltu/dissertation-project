"""
dataset_noise.py — Extract natural noise segments from Exp1 recordings.

Uses the pre-trigger region (samples 0-3000, well before the pulse at 4000)
as genuine NonPD/background samples from the same acquisition hardware.
This is better than external datasets because it matches your exact
sensor, cable, and acquisition chain.

Produces:
  - results/nonpd_natural/noise_features_cache.npz  (122 features)
  - results/nonpd_natural/noise_comparison.png       (NonPD vs PD plot)

Usage:
    python src/dataset_noise.py
    python src/dataset_noise.py --data_dir data/raw/measurements --max_per_class 50
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.io import loadmat
from tqdm import tqdm

# ── constants ─────────────────────────────────────────────────────────────────
FS             = 100e6
NOISE_START    = 0       # well before trigger
NOISE_END      = 3000    # end before crop starts at 3500
NOISE_LEN      = NOISE_END - NOISE_START   # 3000 samples
CROP_LEN       = 4000    # PD crop length

POS_TO_LABEL = {
    100:0, 200:1, 300:2,  500:3,  700:4,  900:5,
    1000:6,1300:7,1500:8, 1600:9,1800:10,1900:11
}


# ─────────────────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────────────────

def build_file_list(root_dir):
    samples = []
    root    = Path(root_dir)
    for pos, label in POS_TO_LABEL.items():
        for volt in range(1, 11):
            volt_dir = root / str(pos) / f'{volt}V'
            if not volt_dir.exists():
                continue
            for f in volt_dir.glob('*.mat'):
                samples.append((str(f), label, volt))
    return samples


def load_noise_segment(filepath):
    """
    Load the pre-trigger region as a 2-channel noise sample.
    Returns (2, NOISE_LEN) float32.
    """
    mat  = loadmat(filepath, simplify_cells=False)
    data = mat['tpd']['Data'][0, 0].astype(np.float32)
    return data[:, NOISE_START:NOISE_END]


def load_pd_segment(filepath):
    """Load the standard PD crop. Returns (2, 4000) float32."""
    mat  = loadmat(filepath, simplify_cells=False)
    data = mat['tpd']['Data'][0, 0].astype(np.float32)
    return data[:, 3500:7500]


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction on noise segments
# ─────────────────────────────────────────────────────────────────────────────

def extract_noise_features(noise_seg):
    """
    Extract v2 features from the 3000-sample noise segment.
    The noise segment is single-ended (no pulse) so we use it as-is.
    We zero-pad to 4000 samples so features_v2 crop logic is bypassed.
    """
    from features_v2 import (
        _time_features, _freq_features, _wavelet_features,
        _analytic_features, _inter_channel_features
    )

    ch1 = noise_seg[0].astype(np.float64)
    ch2 = noise_seg[1].astype(np.float64)

    t1 = _time_features(ch1)
    t2 = _time_features(ch2)
    f1n, f1w = _freq_features(ch1)
    f2n, f2w = _freq_features(ch2)
    w1 = _wavelet_features(ch1)
    w2 = _wavelet_features(ch2)
    a1 = _analytic_features(ch1)
    a2 = _analytic_features(ch2)
    inter = _inter_channel_features(ch1, ch2)

    feats = np.concatenate([t1, t2, f1n, f1w, f2n, f2w, w1, w2, a1, a2, inter])
    feats = np.where(np.isfinite(feats), feats, 0.0)
    return feats.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Comparison visualization
# ─────────────────────────────────────────────────────────────────────────────

def plot_comparison(pd_segs, noise_segs, out_dir, n_examples=4):
    """
    Side-by-side plots: PD crop (left) vs noise segment (right).
    """
    fig, axes = plt.subplots(n_examples, 2, figsize=(10, 2.5 * n_examples),
                              constrained_layout=True)

    t_pd    = np.arange(CROP_LEN)  / FS * 1e6 - 5.0   # µs
    t_noise = np.arange(NOISE_LEN) / FS * 1e6          # µs from start

    for i in range(n_examples):
        # PD signal
        axes[i, 0].plot(t_pd, pd_segs[i][0], color='black',
                        linewidth=0.7, label='Ch1')
        axes[i, 0].plot(t_pd, pd_segs[i][1], color='#555',
                        linewidth=0.7, linestyle='--', label='Ch2')
        axes[i, 0].set_xlim(t_pd[0], t_pd[-1])
        axes[i, 0].set_ylabel('Amplitude (a.u.)')
        axes[i, 0].grid(True, alpha=0.3)
        if i == 0:
            axes[i, 0].set_title('PD crop window (-5 to +35 µs)')
            axes[i, 0].legend(fontsize=7, frameon=False)

        # Noise segment
        axes[i, 1].plot(t_noise, noise_segs[i][0], color='black',
                        linewidth=0.7, label='Ch1')
        axes[i, 1].plot(t_noise, noise_segs[i][1], color='#555',
                        linewidth=0.7, linestyle='--', label='Ch2')
        axes[i, 1].set_xlim(t_noise[0], t_noise[-1])
        axes[i, 1].grid(True, alpha=0.3)
        if i == 0:
            axes[i, 1].set_title('Pre-trigger noise segment (0 to 30 µs)')

    for ax in axes[-1]:
        ax.set_xlabel(r'Time ($\mu$s)')

    fig.suptitle('PD signals vs natural background noise (same acquisition hardware)',
                 fontsize=10)
    fig.savefig(out_dir / 'noise_vs_pd_comparison.png', dpi=150,
                bbox_inches='tight')
    plt.close(fig)
    print('Saved noise_vs_pd_comparison.png')


def plot_amplitude_distribution(pd_rms, noise_rms, out_dir):
    """Histogram of RMS amplitudes: PD vs noise."""
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    bins = np.linspace(0, max(np.percentile(pd_rms, 99),
                               np.percentile(noise_rms, 99)), 60)
    ax.hist(noise_rms, bins=bins, alpha=0.6, color='#555', label='Noise (pre-trigger)')
    ax.hist(pd_rms,    bins=bins, alpha=0.6, color='black', label='PD crop')
    ax.set_xlabel('RMS amplitude (a.u.)')
    ax.set_ylabel('Count')
    ax.set_title('Amplitude distribution: PD crop vs background noise')
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    fig.savefig(out_dir / 'noise_amplitude_distribution.png', dpi=150,
                bbox_inches='tight')
    plt.close(fig)
    print('Saved noise_amplitude_distribution.png')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'[1] Scanning {args.data_dir}...')
    all_samples = build_file_list(args.data_dir)
    print(f'    Found {len(all_samples)} files')

    # Subsample evenly across positions
    rng = np.random.default_rng(42)
    chosen = []
    for pos in POS_TO_LABEL:
        pos_files = [(f, l, v) for f, l, v in all_samples
                     if l == POS_TO_LABEL[pos]]
        n = min(args.max_per_class, len(pos_files))
        idx = rng.choice(len(pos_files), n, replace=False)
        chosen.extend([pos_files[i] for i in idx])

    print(f'    Using {len(chosen)} files for noise extraction')

    # Extract noise features
    print('\n[2] Extracting noise features...')
    from features_v2 import feature_count
    F        = feature_count()
    X_noise  = np.zeros((len(chosen), F), dtype=np.float32)
    pd_segs_sample   = []
    noise_segs_sample = []
    pd_rms   = []
    noise_rms = []

    for i, (fp, label, volt) in enumerate(tqdm(chosen, desc='Noise features')):
        try:
            noise = load_noise_segment(fp)
            X_noise[i] = extract_noise_features(noise)
            noise_rms.append(float(np.sqrt(np.mean(noise[0]**2))))

            if len(pd_segs_sample) < 8:
                pd   = load_pd_segment(fp)
                pd_segs_sample.append(pd)
                noise_segs_sample.append(noise)
                pd_rms_val = float(np.sqrt(np.mean(pd[0]**2)))
                pd_rms.append(pd_rms_val)

        except Exception as e:
            print(f'  [WARN] {fp}: {e}')

    X_noise = np.where(np.isfinite(X_noise), X_noise, 0.0)
    y_noise = np.full(len(chosen), 12, dtype=np.int32)   # class 12 = noise

    # Save cache
    cache_path = out_dir / 'noise_features_cache.npz'
    np.savez(cache_path, X=X_noise, y=y_noise)
    print(f'\nSaved noise feature cache: {cache_path}  ({len(chosen)} samples)')

    # Comparison plots
    print('\n[3] Generating comparison plots...')
    plot_comparison(pd_segs_sample[:4], noise_segs_sample[:4], out_dir)
    # collect all pd rms
    pd_rms_all = [float(np.sqrt(np.mean(load_pd_segment(fp)[0]**2)))
                  for fp, _, _ in tqdm(chosen[:500], desc='PD RMS')]
    plot_amplitude_distribution(np.array(pd_rms_all), np.array(noise_rms[:500]),
                                out_dir)

    # Print stats
    print(f'\nNoise RMS — mean: {np.mean(noise_rms):.4f}  '
          f'std: {np.std(noise_rms):.4f}  '
          f'max: {np.max(noise_rms):.4f}')
    print(f'PD RMS    — mean: {np.mean(pd_rms_all):.4f}  '
          f'std: {np.std(pd_rms_all):.4f}  '
          f'max: {np.max(pd_rms_all):.4f}')
    print(f'\nAll outputs saved to: {out_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',      default='data/raw/measurements')
    parser.add_argument('--out_dir',       default='results/nonpd_natural')
    parser.add_argument('--max_per_class', type=int, default=100,
                        help='Max noise samples per position class')
    args = parser.parse_args()
    main(args)