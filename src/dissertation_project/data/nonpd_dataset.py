"""
dataset_nonpd.py — Loader for the Rauscher et al. (2024) figshare NonPD dataset.

Files expected:
    data/raw/nonpd/Tr0.mat   (12,260 signals)
    data/raw/nonpd/Va0.mat   ( 3,066 signals)
    data/raw/nonpd/Te0.mat   (45,970 signals)

Each signal: shape (1, 400), float32, single channel, pre-cropped.

Two output modes
-----------------
  features : extract v2 features directly from the 400-sample signal.
             The crop window in features_v2 is overridden — we use the
             full 400 samples as-is.

  padded   : zero-pad to 4000 samples (centred) so the signal can be
             fed into the Exp1 CNN which expects (1 or 2, 4000) input.
             Channel 1 = padded signal, Channel 2 = zeros (no 2nd sensor).

Citation:
    Rauscher, A. et al. "Deep learning and data augmentation for partial
    discharge detection in electrical machines."
    Eng. Appl. Artif. Intell. 133 (2024) 108074.
    Data: https://doi.org/10.6084/m9.figshare.24033225  (CC BY 4.0)
"""

import numpy as np
from pathlib import Path
from scipy.io import loadmat
from tqdm import tqdm

# ── constants ─────────────────────────────────────────────────────────────────
NONPD_LABEL   = 12          # class index for NonPD (after 12 PD positions 0-11)
NONPD_SIG_LEN = 400         # original signal length in the figshare dataset
TARGET_LEN    = 4000        # Exp1 CNN input length
PAD_START     = (TARGET_LEN - NONPD_SIG_LEN) // 2   # 1800
PAD_END       = TARGET_LEN - NONPD_SIG_LEN - PAD_START  # 1800

FILE_MAP = {
    'train': 'Tr0.mat',
    'val':   'Va0.mat',
    'test':  'Te0.mat',
}


# ─────────────────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_nonpd_file(filepath: str) -> np.ndarray:
    """
    Load a single Tr0/Va0/Te0.mat file.

    Returns
    -------
    signals : np.ndarray, shape (N, 400), float32
    """
    mat  = loadmat(filepath, simplify_cells=False)
    key  = Path(filepath).stem          # 'Tr0', 'Va0', 'Te0'
    d    = mat[key]
    sigs = d['signals'][0, 0]           # shape (N, 1) of object arrays
    N    = sigs.shape[0]

    out  = np.zeros((N, NONPD_SIG_LEN), dtype=np.float32)
    for i in range(N):
        out[i] = sigs[i, 0].flatten()[:NONPD_SIG_LEN]
    return out


def load_split(root_dir: str, split: str = 'train') -> np.ndarray:
    """
    Load one split ('train', 'val', 'test').

    Returns
    -------
    signals : np.ndarray, shape (N, 400), float32
    """
    fname = FILE_MAP[split]
    fpath = Path(root_dir) / fname
    if not fpath.exists():
        raise FileNotFoundError(f"NonPD file not found: {fpath}")
    return load_nonpd_file(str(fpath))


# ─────────────────────────────────────────────────────────────────────────────
# Padding  (for CNN)
# ─────────────────────────────────────────────────────────────────────────────

def pad_to_exp1(sig400: np.ndarray) -> np.ndarray:
    """
    Zero-pad a (400,) signal to (2, 4000) matching Exp1 CNN input.
    Channel 1 = padded signal (centred), Channel 2 = zeros.

    Parameters
    ----------
    sig400 : (400,) float32

    Returns
    -------
    padded : (2, 4000) float32
    """
    ch1 = np.zeros(TARGET_LEN, dtype=np.float32)
    ch1[PAD_START: PAD_START + NONPD_SIG_LEN] = sig400
    ch2 = np.zeros(TARGET_LEN, dtype=np.float32)
    return np.stack([ch1, ch2], axis=0)


def pad_batch(signals: np.ndarray) -> np.ndarray:
    """
    Pad a batch of NonPD signals.

    Parameters
    ----------
    signals : (N, 400) float32

    Returns
    -------
    padded : (N, 2, 4000) float32
    """
    N   = signals.shape[0]
    out = np.zeros((N, 2, TARGET_LEN), dtype=np.float32)
    out[:, 0, PAD_START: PAD_START + NONPD_SIG_LEN] = signals
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction  (for RF / SVM)
# ─────────────────────────────────────────────────────────────────────────────

def extract_features_nonpd(sig400: np.ndarray) -> np.ndarray:
    """
    Extract v2 features from a 400-sample NonPD signal.

    Since the signal is single-channel and 400 samples (not 4000),
    we use a modified approach:
      - Ch1 = sig400  (the real signal)
      - Ch2 = zeros   (no second channel)
      - No crop applied (signal is already pre-cropped)

    Parameters
    ----------
    sig400 : (400,) float32

    Returns
    -------
    feats : (F,) float32  — same length as features_v2.feature_count()
    """
    from features_v2 import (
        _time_features, _freq_features, _wavelet_features,
        _analytic_features, _inter_channel_features
    )

    ch1 = sig400.astype(np.float64)
    ch2 = np.zeros_like(ch1)

    t1 = _time_features(ch1)
    t2 = _time_features(ch2)

    f1_narrow, f1_wide = _freq_features(ch1)
    f2_narrow, f2_wide = _freq_features(ch2)

    w1 = _wavelet_features(ch1)
    w2 = _wavelet_features(ch2)

    a1 = _analytic_features(ch1)
    a2 = _analytic_features(ch2)

    inter = _inter_channel_features(ch1, ch2)

    feats = np.concatenate([
        t1, t2,
        f1_narrow, f1_wide,
        f2_narrow, f2_wide,
        w1, w2,
        a1, a2,
        inter,
    ])
    feats = np.where(np.isfinite(feats), feats, 0.0)
    return feats.astype(np.float32)


def extract_features_batch(signals: np.ndarray,
                           desc: str = 'NonPD features') -> np.ndarray:
    """
    Extract features for all NonPD signals.

    Parameters
    ----------
    signals : (N, 400) float32

    Returns
    -------
    X : (N, F) float32
    """
    from features_v2 import feature_count
    F   = feature_count()
    X   = np.zeros((len(signals), F), dtype=np.float32)
    for i, sig in enumerate(tqdm(signals, desc=desc)):
        X[i] = extract_features_nonpd(sig)
    X = np.where(np.isfinite(X), X, 0.0)
    return X


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else 'data/raw/nonpd'

    print(f'Loading from: {root}')
    for split in ['train', 'val', 'test']:
        try:
            s = load_split(root, split)
            print(f'  {split:5s}: {s.shape}  min={s.min():.3f}  max={s.max():.3f}')
        except FileNotFoundError as e:
            print(f'  {split:5s}: {e}')

    # Test padding
    s = load_split(root, 'val')
    padded = pad_batch(s[:4])
    print(f'\nPadded shape: {padded.shape}')
    print(f'Signal placed at [{PAD_START}:{PAD_START+NONPD_SIG_LEN}]')

    # Test features
    feats = extract_features_batch(s[:5], desc='test')
    print(f'\nFeature shape: {feats.shape}')
    print(f'NaN count: {np.isnan(feats).sum()}')