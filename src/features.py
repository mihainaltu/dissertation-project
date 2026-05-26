# src/features.py

import numpy as np
from scipy import signal as scipy_signal


# ── Sampling config ────────────────────────────────────────────────────────────
FS      = 100e6        # 100 MHz
PRE     = 4000         # pre-trigger samples in original signal
CROP    = (3500, 7500) # our chosen window


# ── Individual feature functions ───────────────────────────────────────────────

def peak_amplitude(x):
    """Max absolute amplitude."""
    return np.max(np.abs(x))

def rms(x):
    """Root mean square energy."""
    return np.sqrt(np.mean(x ** 2))

def zero_crossing_rate(x):
    """Number of zero crossings normalized by length."""
    return np.sum(np.diff(np.sign(x)) != 0) / len(x)

def pulse_width(x, fs=FS, threshold=0.1):
    """
    Estimate pulse width (µs) as time signal stays above threshold * peak.
    """
    thr   = threshold * np.max(np.abs(x))
    above = np.where(np.abs(x) > thr)[0]
    if len(above) < 2:
        return 0.0
    return (above[-1] - above[0]) / fs * 1e6

def spectral_energy_bands(x, fs=FS):
    """
    Energy in frequency bands (normalized):
      Band 0: 0–2 MHz
      Band 1: 2–5 MHz
      Band 2: 5–10 MHz
      Band 3: 10–15 MHz
      Band 4: 15–50 MHz
    """
    N      = len(x)
    freqs  = np.fft.rfftfreq(N, d=1/fs) / 1e6   # MHz
    fft_sq = np.abs(np.fft.rfft(x)) ** 2

    bands  = [(0, 2), (2, 5), (5, 10), (10, 15), (15, 50)]
    energy = []
    total  = fft_sq.sum() + 1e-12
    for lo, hi in bands:
        mask = (freqs >= lo) & (freqs < hi)
        energy.append(fft_sq[mask].sum() / total)

    return np.array(energy)

def peak_frequency(x, fs=FS):
    """Dominant frequency in MHz."""
    N     = len(x)
    freqs = np.fft.rfftfreq(N, d=1/fs) / 1e6
    fft_m = np.abs(np.fft.rfft(x))
    return freqs[np.argmax(fft_m)]

def tdoa(ch1, ch2, fs=FS):
    """
    Time Difference of Arrival between Ch1 and Ch2 via cross-correlation (µs).
    Positive = Ch1 arrives first.
    """
    corr   = np.correlate(ch1, ch2, mode='full')
    lag    = np.argmax(np.abs(corr)) - (len(ch2) - 1)
    return lag / fs * 1e6

def waveform_asymmetry(x):
    """Skewness-like: ratio of positive to total energy."""
    pos = np.sum(x[x > 0] ** 2)
    tot = np.sum(x ** 2) + 1e-12
    return pos / tot


# ── Full feature vector ────────────────────────────────────────────────────────

def extract_features(signal_2ch):
    """
    Extract a fixed-length feature vector from a 2-channel signal.

    Args:
        signal_2ch: np.ndarray of shape (2, N), float32
    Returns:
        np.ndarray of shape (F,) — flat feature vector
    """
    ch1, ch2 = signal_2ch[0], signal_2ch[1]
    features = []

    for ch in [ch1, ch2]:
        features.append(peak_amplitude(ch))
        features.append(rms(ch))
        features.append(zero_crossing_rate(ch))
        features.append(pulse_width(ch))
        features.append(peak_frequency(ch))
        features.append(waveform_asymmetry(ch))
        features.extend(spectral_energy_bands(ch))   # 5 values

    # Cross-channel features
    features.append(tdoa(ch1, ch2))
    features.append(peak_amplitude(ch1) / (peak_amplitude(ch2) + 1e-12))  # amplitude ratio

    return np.array(features, dtype=np.float32)


def feature_names():
    names = []
    for ch in ['Ch1', 'Ch2']:
        names += [
            f'{ch}_peak', f'{ch}_rms', f'{ch}_zcr', f'{ch}_pulse_width',
            f'{ch}_peak_freq', f'{ch}_asymmetry',
            f'{ch}_band0_0-2MHz', f'{ch}_band1_2-5MHz',
            f'{ch}_band2_5-10MHz', f'{ch}_band3_10-15MHz', f'{ch}_band4_15-50MHz',
        ]
    names += ['tdoa_us', 'amplitude_ratio']
    return names