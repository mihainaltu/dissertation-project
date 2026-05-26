"""
features_v2.py — Extended feature bank for PD localization (Experiment 1)
~80 features per sample, CPU-friendly (no heavy STFT, no nonlinear entropy)

Feature groups:
  A. Time domain          (~20 features per channel = 40 total)
  B. Frequency domain     (~14 features per channel = 28 total)
  C. Wavelet (DWT)        (5 levels x 2 stats x 2 channels = 20 total)
  D. Analytic / Hilbert   (~6 features per channel = 12 total)
  E. Inter-channel        (8 features)

Total: ~108 raw → NaN/Inf safe-guarded → consistent vector
"""

import numpy as np
from scipy.signal import hilbert
from scipy.stats import kurtosis, skew
import pywt  # pip install PyWavelets  (very fast, CPU friendly)

# ── crop window (same as features.py) ────────────────────────────────────────
CROP = (3500, 7500)          # 4000 samples @ 100 MHz → 40 µs
FS   = 100e6                 # Hz

# Frequency bands (Hz) — 5 narrow + 5 wide for richer coverage
BANDS_NARROW = [
    (0,    2e6),
    (2e6,  5e6),
    (5e6, 10e6),
    (10e6,15e6),
    (15e6,50e6),
]
BANDS_WIDE = [
    (0,    5e6),
    (5e6, 15e6),
    (15e6,25e6),
    (25e6,40e6),
    (40e6,50e6),
]

WAVELET    = 'db4'   # Daubechies-4: good for transient signals
DWT_LEVELS = 5       # 5 levels → frequency resolution down to ~1.5 MHz


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe(x, fallback=0.0):
    """Replace NaN / Inf with fallback."""
    if not np.isfinite(x):
        return float(fallback)
    return float(x)


def _band_energy(fft_mag, freqs, flo, fhi):
    """Energy in a frequency band (trapezoidal sum of |FFT|^2)."""
    mask = (freqs >= flo) & (freqs < fhi)
    if mask.sum() == 0:
        return 0.0
    return float(np.sum(fft_mag[mask] ** 2))


def _peak_freq_safe(fft_mag, freqs):
    """Peak frequency — returns 0 if spectrum is flat / all-zero."""
    if fft_mag.max() == 0:
        return 0.0
    return float(freqs[np.argmax(fft_mag)])


# ─────────────────────────────────────────────────────────────────────────────
# A. Time-domain features  (per channel, 1-D array)
# ─────────────────────────────────────────────────────────────────────────────

def _time_features(sig):
    """
    Returns 20 scalar features for a single-channel signal.
    sig: 1-D np.ndarray, cropped waveform
    """
    n   = len(sig)
    env = np.abs(hilbert(sig))          # Hilbert envelope (used here too)

    # Basic statistics
    peak      = _safe(np.max(np.abs(sig)))
    rms       = _safe(np.sqrt(np.mean(sig ** 2)))
    crest     = _safe(peak / (rms + 1e-12))
    p2p       = _safe(np.max(sig) - np.min(sig))
    energy    = _safe(np.sum(sig ** 2))
    kurt      = _safe(kurtosis(sig))
    skewness  = _safe(skew(sig))
    asymmetry = _safe((np.max(sig) + np.min(sig)) / (peak + 1e-12))

    # Zero-crossing rate
    zcr = _safe(np.sum(np.diff(np.sign(sig)) != 0) / n)

    # Pulse width at 50% of peak
    half = peak * 0.5
    above = np.where(np.abs(sig) >= half)[0]
    pulse_width = _safe(len(above) / FS * 1e6) if len(above) > 0 else 0.0  # µs

    # Rise time: 10%→90% of peak on the envelope
    idx_peak = int(np.argmax(env))
    rise_time = 0.0
    if idx_peak > 0:
        lo = peak * 0.1; hi = peak * 0.9
        lo_idx = np.where(env[:idx_peak] >= lo)[0]
        hi_idx = np.where(env[:idx_peak] >= hi)[0]
        if len(lo_idx) > 0 and len(hi_idx) > 0:
            rise_time = _safe((hi_idx[0] - lo_idx[0]) / FS * 1e9)  # ns

    # Decay rate: fit exp to envelope tail (from peak to end)
    decay_rate = 0.0
    tail = env[idx_peak:]
    if len(tail) > 10 and tail[0] > 1e-12:
        t_tail = np.arange(len(tail)) / FS
        log_tail = np.log(tail / (tail[0] + 1e-12) + 1e-12)
        try:
            coeffs = np.polyfit(t_tail, log_tail, 1)
            decay_rate = _safe(-coeffs[0])          # positive = decaying
        except Exception:
            pass

    # Waveform centroid (center of mass in time)
    t_vec    = np.arange(n) / FS
    centroid = _safe(np.sum(t_vec * sig ** 2) / (energy + 1e-12) * 1e6)  # µs

    # Tail energy ratio (last 25% of crop vs total)
    tail_start = int(0.75 * n)
    tail_energy_ratio = _safe(
        np.sum(sig[tail_start:] ** 2) / (energy + 1e-12)
    )

    # Envelope statistics
    env_mean = _safe(np.mean(env))
    env_std  = _safe(np.std(env))
    env_peak = _safe(np.max(env))

    return np.array([
        peak, rms, crest, p2p, energy,
        kurt, skewness, asymmetry, zcr, pulse_width,
        rise_time, decay_rate, centroid, tail_energy_ratio,
        env_mean, env_std, env_peak,
        # 3 spare slots kept for future use (set to 0 → won't hurt RF/SVM)
        0.0, 0.0, 0.0,
    ], dtype=np.float32)   # 20 features


_TIME_NAMES = [
    'peak', 'rms', 'crest_factor', 'peak_to_peak', 'energy',
    'kurtosis', 'skewness', 'asymmetry', 'zcr', 'pulse_width_us',
    'rise_time_ns', 'decay_rate', 'centroid_us', 'tail_energy_ratio',
    'env_mean', 'env_std', 'env_peak',
    '_t18', '_t19', '_t20',
]


# ─────────────────────────────────────────────────────────────────────────────
# B. Frequency-domain features  (per channel)
# ─────────────────────────────────────────────────────────────────────────────

def _freq_features(sig):
    """
    Returns 14 scalar features for a single-channel signal.
    """
    n      = len(sig)
    win    = np.hanning(n)
    fft_c  = np.fft.rfft(sig * win)
    fft_m  = np.abs(fft_c)
    freqs  = np.fft.rfftfreq(n, d=1.0 / FS)
    total  = np.sum(fft_m ** 2) + 1e-12

    # 5 narrow bands
    narrow = [_band_energy(fft_m, freqs, lo, hi) / total
              for lo, hi in BANDS_NARROW]

    # 5 wide bands
    wide = [_band_energy(fft_m, freqs, lo, hi) / total
            for lo, hi in BANDS_WIDE]

    # Spectral moments
    centroid = _safe(np.sum(freqs * fft_m ** 2) / total / 1e6)   # MHz
    spread   = _safe(np.sqrt(
        np.sum(((freqs / 1e6 - centroid) ** 2) * fft_m ** 2) / total
    ))
    rolloff_idx = np.searchsorted(
        np.cumsum(fft_m ** 2), 0.85 * (total - 1e-12)
    )
    rolloff  = _safe(freqs[min(rolloff_idx, len(freqs)-1)] / 1e6)  # MHz

    peak_f   = _safe(_peak_freq_safe(fft_m, freqs) / 1e6)          # MHz (NaN-safe)

    spectral_flatness = _safe(
        np.exp(np.mean(np.log(fft_m + 1e-12))) / (np.mean(fft_m) + 1e-12)
    )
    spectral_entropy  = _safe(
        -np.sum((fft_m**2 / total) * np.log2(fft_m**2 / total + 1e-12))
    )

    # Only return narrow bands + spectral moments (14 features)
    # Wide bands kept separate so feature_names stays aligned
    return (
        np.array(narrow + [centroid, spread, rolloff, peak_f,
                           spectral_flatness, spectral_entropy,
                           0.0, 0.0, 0.0],   # 3 spares
                 dtype=np.float32),           # 14 features
        np.array(wide, dtype=np.float32)      # 5 wide-band features
    )


_FREQ_NAMES_NARROW = [
    'band0_0-2MHz', 'band1_2-5MHz', 'band2_5-10MHz',
    'band3_10-15MHz', 'band4_15-50MHz',
    'spectral_centroid_MHz', 'spectral_spread_MHz',
    'spectral_rolloff_MHz', 'peak_freq_MHz',
    'spectral_flatness', 'spectral_entropy',
    '_f12', '_f13', '_f14',
]
_FREQ_NAMES_WIDE = [
    'wide0_0-5MHz', 'wide1_5-15MHz', 'wide2_15-25MHz',
    'wide3_25-40MHz', 'wide4_40-50MHz',
]


# ─────────────────────────────────────────────────────────────────────────────
# C. Wavelet (DWT) features  (per channel)
# ─────────────────────────────────────────────────────────────────────────────

def _wavelet_features(sig):
    """
    Discrete Wavelet Transform — energy and std per subband.
    Returns 2 * DWT_LEVELS = 10 features per channel.
    pywt.wavedec returns [cA_N, cD_N, cD_{N-1}, ..., cD_1]
    We use the detail coefficients (cD) — they capture transient edges.
    """
    coeffs = pywt.wavedec(sig, WAVELET, level=DWT_LEVELS)
    # coeffs[0] = approximation at level N
    # coeffs[1..N] = details at levels N..1
    feats = []
    for c in coeffs:                            # N+1 subbands
        e = _safe(np.sum(c ** 2) / (len(c) + 1e-12))   # energy density
        s = _safe(np.std(c))
        feats.extend([e, s])
    # We asked for DWT_LEVELS, so len(coeffs) = DWT_LEVELS + 1
    # → 2*(DWT_LEVELS+1) = 12 features per channel
    return np.array(feats, dtype=np.float32)


def _wavelet_names():
    names = []
    labels = ['approx'] + [f'detail_L{DWT_LEVELS - i}' for i in range(DWT_LEVELS)]
    for lbl in labels:
        names += [f'dwt_{lbl}_energy', f'dwt_{lbl}_std']
    return names   # 12 names


# ─────────────────────────────────────────────────────────────────────────────
# D. Analytic / Hilbert features  (per channel)
# ─────────────────────────────────────────────────────────────────────────────

def _analytic_features(sig):
    """
    Instantaneous amplitude, frequency, and phase features.
    Returns 6 features per channel.
    """
    analytic  = hilbert(sig)
    env       = np.abs(analytic)
    phase     = np.unwrap(np.angle(analytic))
    inst_freq = np.diff(phase) / (2.0 * np.pi) * FS / 1e6    # MHz

    # Clip extreme inst_freq values (numerical noise at zero crossings)
    inst_freq = np.clip(inst_freq, 0, FS / 2e6)

    env_skew  = _safe(skew(env))
    env_kurt  = _safe(kurtosis(env))
    if_mean   = _safe(np.mean(inst_freq))
    if_std    = _safe(np.std(inst_freq))
    phase_std = _safe(np.std(phase))
    phase_range = _safe(np.ptp(phase))

    return np.array([
        env_skew, env_kurt,
        if_mean, if_std,
        phase_std, phase_range,
    ], dtype=np.float32)   # 6 features


_ANALYTIC_NAMES = [
    'env_skew', 'env_kurtosis',
    'inst_freq_mean_MHz', 'inst_freq_std_MHz',
    'phase_std', 'phase_range',
]


# ─────────────────────────────────────────────────────────────────────────────
# E. Inter-channel features  (uses both channels)
# ─────────────────────────────────────────────────────────────────────────────

def _inter_channel_features(sig1, sig2):
    """
    8 features capturing the relationship between Ch1 and Ch2.
    """
    n = min(len(sig1), len(sig2))
    s1, s2 = sig1[:n], sig2[:n]

    # Amplitude ratio (robust)
    rms1 = np.sqrt(np.mean(s1 ** 2)) + 1e-12
    rms2 = np.sqrt(np.mean(s2 ** 2)) + 1e-12
    amp_ratio = _safe(rms1 / rms2)

    # Peak amplitude ratio
    peak_ratio = _safe(np.max(np.abs(s1)) / (np.max(np.abs(s2)) + 1e-12))

    # Cross-correlation: peak value and lag (TDOA)
    xcorr = np.correlate(s1 / (rms1), s2 / (rms2), mode='full')
    lags  = np.arange(-(n - 1), n)
    idx   = int(np.argmax(xcorr))
    tdoa_samples = lags[idx]
    tdoa_us      = _safe(tdoa_samples / FS * 1e6)
    xcorr_peak   = _safe(xcorr[idx] / n)

    # Width of cross-correlation peak (samples above 50% of max)
    half = xcorr[idx] * 0.5
    above = np.where(xcorr >= half)[0]
    xcorr_width  = _safe(len(above) / FS * 1e6)   # µs

    # Frequency-domain coherence per wide band
    fft1 = np.fft.rfft(s1 * np.hanning(n))
    fft2 = np.fft.rfft(s2 * np.hanning(n))
    freqs = np.fft.rfftfreq(n, d=1.0 / FS)
    # Mean coherence in two broad bands
    coh = np.abs(fft1 * np.conj(fft2)) / (np.abs(fft1) * np.abs(fft2) + 1e-12)
    mask_lo = (freqs >= 0) & (freqs < 10e6)
    mask_hi = (freqs >= 10e6) & (freqs < 50e6)
    coh_lo  = _safe(np.mean(coh[mask_lo]))
    coh_hi  = _safe(np.mean(coh[mask_hi]))

    # Phase difference at peak frequency of Ch1
    pk_idx      = int(np.argmax(np.abs(fft1)))
    phase_diff  = _safe(np.angle(fft1[pk_idx]) - np.angle(fft2[pk_idx]))

    return np.array([
        amp_ratio, peak_ratio,
        tdoa_us, xcorr_peak, xcorr_width,
        coh_lo, coh_hi,
        phase_diff,
    ], dtype=np.float32)   # 8 features


_INTER_NAMES = [
    'amp_ratio_rms', 'amp_ratio_peak',
    'tdoa_us', 'xcorr_peak', 'xcorr_width_us',
    'coherence_lo_0-10MHz', 'coherence_hi_10-50MHz',
    'phase_diff_at_peak',
]


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(signal_2ch: np.ndarray) -> np.ndarray:
    """
    Main entry point.

    Parameters
    ----------
    signal_2ch : np.ndarray, shape (2, N) or (2, ≥7500)
        Two-channel raw signal from load_mat_file().
        Cropping is applied internally (CROP = (3500, 7500)).

    Returns
    -------
    feats : np.ndarray, shape (F,)  where F = feature_count()
        All features as a flat float32 vector.
        Guaranteed NaN/Inf free.
    """
    # Crop
    lo, hi = CROP
    ch1 = signal_2ch[0, lo:hi].astype(np.float64)
    ch2 = signal_2ch[1, lo:hi].astype(np.float64)

    # A. Time domain
    t1 = _time_features(ch1)
    t2 = _time_features(ch2)

    # B. Frequency domain (narrow bands + spectral moments, and wide bands)
    f1_narrow, f1_wide = _freq_features(ch1)
    f2_narrow, f2_wide = _freq_features(ch2)

    # C. Wavelet
    w1 = _wavelet_features(ch1)
    w2 = _wavelet_features(ch2)

    # D. Analytic
    a1 = _analytic_features(ch1)
    a2 = _analytic_features(ch2)

    # E. Inter-channel
    inter = _inter_channel_features(ch1, ch2)

    feats = np.concatenate([
        t1, t2,
        f1_narrow, f1_wide,
        f2_narrow, f2_wide,
        w1, w2,
        a1, a2,
        inter,
    ])

    # Final NaN/Inf guard
    feats = np.where(np.isfinite(feats), feats, 0.0)
    return feats.astype(np.float32)


def feature_names() -> list:
    """
    Returns the list of feature name strings in the same order as
    extract_features() output. Use for ANOVA plots, importance bars, etc.
    """
    ch1_time  = [f'Ch1_{n}' for n in _TIME_NAMES]
    ch2_time  = [f'Ch2_{n}' for n in _TIME_NAMES]
    ch1_freq  = [f'Ch1_{n}' for n in _FREQ_NAMES_NARROW]
    ch1_wide  = [f'Ch1_{n}' for n in _FREQ_NAMES_WIDE]
    ch2_freq  = [f'Ch2_{n}' for n in _FREQ_NAMES_NARROW]
    ch2_wide  = [f'Ch2_{n}' for n in _FREQ_NAMES_WIDE]
    ch1_wav   = [f'Ch1_{n}' for n in _wavelet_names()]
    ch2_wav   = [f'Ch2_{n}' for n in _wavelet_names()]
    ch1_ana   = [f'Ch1_{n}' for n in _ANALYTIC_NAMES]
    ch2_ana   = [f'Ch2_{n}' for n in _ANALYTIC_NAMES]
    return (
        ch1_time + ch2_time +
        ch1_freq + ch1_wide +
        ch2_freq + ch2_wide +
        ch1_wav  + ch2_wav  +
        ch1_ana  + ch2_ana  +
        _INTER_NAMES
    )


def feature_count() -> int:
    return len(feature_names())


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke-test  (run:  python features_v2.py)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print(f'PyWavelets version: {pywt.__version__}')
    # Synthetic test signal — 2 channels, 20000 samples
    rng  = np.random.default_rng(0)
    fake = rng.standard_normal((2, 20000)).astype(np.float32)
    # Add a fake pulse at sample 4000
    t    = np.arange(4000) / FS
    pulse = np.exp(-t / 5e-7) * np.sin(2 * np.pi * 5e6 * t)
    fake[0, 3500:7500] += pulse[:4000].astype(np.float32)
    fake[1, 3500:7500] += (pulse[:4000] * 0.6).astype(np.float32)

    feats = extract_features(fake)
    names = feature_names()

    print(f'\nTotal features : {len(feats)}')
    print(f'Name list len  : {len(names)}')
    print(f'Any NaN?       : {np.any(np.isnan(feats))}')
    print(f'Any Inf?       : {np.any(np.isinf(feats))}')
    print('\nFirst 10 features:')
    for n, v in zip(names[:10], feats[:10]):
        print(f'  {n:<35s} {v:.6f}')
    print('\nLast 10 features:')
    for n, v in zip(names[-10:], feats[-10:]):
        print(f'  {n:<35s} {v:.6f}')
