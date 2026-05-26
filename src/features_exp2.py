# src/features_exp2.py

import numpy as np
from scipy import signal as scipy_signal

FS         = 200e6
PRE        = 1000000
CROP_START = PRE - 20
CROP_END   = PRE + 1000
CROP_LEN   = 1020


def load_exp2_crop(filepath):
    import scipy.io
    mat  = scipy.io.loadmat(filepath)
    tpd  = mat['tpd']
    data = tpd['Data'][0, 0].flatten().astype(np.float32)
    pre  = int(tpd['PreSampleCount'][0, 0][0, 0])
    return data[pre - 20 : pre + 1000]   # (1020,)


# ── Feature functions ──────────────────────────────────────────────────────────

def peak_amplitude(x):
    return np.max(np.abs(x))

def rms(x):
    return np.sqrt(np.mean(x ** 2))

def decay_rate(x, fs=FS):
    """
    Estimate signal decay rate by fitting envelope to exponential.
    Returns the decay constant (larger = faster decay = cleaner shield).
    """
    analytic  = scipy_signal.hilbert(x)
    envelope  = np.abs(analytic)
    t         = np.arange(len(x)) / fs * 1e6   # µs
    # Fit log(envelope) = -alpha*t + c in the decay region (after peak)
    peak_idx  = np.argmax(envelope)
    if peak_idx >= len(x) - 5:
        return 0.0
    t_decay   = t[peak_idx:]
    env_decay = envelope[peak_idx:]
    env_decay = np.clip(env_decay, 1e-6, None)
    try:
        coeffs = np.polyfit(t_decay, np.log(env_decay), 1)
        return -coeffs[0]   # positive = decaying
    except Exception:
        return 0.0

def oscillation_energy_tail(x, fs=FS):
    """
    Energy in the tail region (0.3–1.5µs) relative to total energy.
    Higher = more oscillation lingering (damaged shield signature).
    """
    t        = np.arange(len(x)) / fs * 1e6
    tail     = x[(t >= 0.3) & (t <= 1.5)]
    total    = x[t >= 0.0]
    if len(tail) == 0 or len(total) == 0:
        return 0.0
    return np.sum(tail**2) / (np.sum(total**2) + 1e-12)

def zero_crossing_rate(x):
    return np.sum(np.diff(np.sign(x)) != 0) / len(x)

def spectral_energy_bands(x, fs=FS):
    """Energy ratio in bands: 0–5, 5–15, 15–30, 30–50, 50–100 MHz"""
    N      = len(x)
    freqs  = np.fft.rfftfreq(N, d=1/fs) / 1e6
    fft_sq = np.abs(np.fft.rfft(x)) ** 2
    total  = fft_sq.sum() + 1e-12
    bands  = [(0, 5), (5, 15), (15, 30), (30, 50), (50, 100)]
    return np.array([fft_sq[(freqs>=lo)&(freqs<hi)].sum()/total
                     for lo, hi in bands], dtype=np.float32)

def peak_frequency(x, fs=FS):
    N     = len(x)
    freqs = np.fft.rfftfreq(N, d=1/fs) / 1e6
    fft_m = np.abs(np.fft.rfft(x))
    return freqs[np.argmax(fft_m)]

def envelope_std(x):
    """Std of the Hilbert envelope — captures oscillation irregularity."""
    envelope = np.abs(scipy_signal.hilbert(x))
    return np.std(envelope)

def waveform_asymmetry(x):
    pos = np.sum(x[x > 0] ** 2)
    tot = np.sum(x ** 2) + 1e-12
    return pos / tot


# ── Full feature vector ────────────────────────────────────────────────────────

def extract_features_exp2(crop):
    """
    Extract feature vector from a single cropped signal (400,).
    Returns np.ndarray of shape (F,)
    """
    features = [
        peak_amplitude(crop),
        rms(crop),
        zero_crossing_rate(crop),
        decay_rate(crop),
        oscillation_energy_tail(crop),
        envelope_std(crop),
        peak_frequency(crop),
        waveform_asymmetry(crop),
    ]
    features.extend(spectral_energy_bands(crop))   # 5 values
    return np.array(features, dtype=np.float32)


def feature_names_exp2():
    return [
        'peak_amplitude', 'rms', 'zcr', 'decay_rate',
        'tail_energy_ratio', 'envelope_std', 'peak_frequency',
        'waveform_asymmetry',
        'band_0-5MHz', 'band_5-15MHz', 'band_15-30MHz',
        'band_30-50MHz', 'band_50-100MHz',
    ]