import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / "exp1_localization"))

"""
plot_pulse_representations.py — Figure 1.4 for thesis.

Extracts a tight window around the pulse peak, then plots:
  (a) time-domain waveform of the windowed pulse
  (b) magnitude spectrum (FFT)
  (c) spectrogram (STFT)

Usage:
    python src/plot_pulse_representations.py
    python src/plot_pulse_representations.py --position 100 --voltage 5
    python src/plot_pulse_representations.py --save_pdf
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy.io import loadmat
from scipy.signal import hilbert
from scipy.signal.windows import tukey, hann

plt.rcParams.update({
    'font.family':       'serif',
    'font.size':         10,
    'axes.titlesize':    10,
    'axes.labelsize':    10,
    'xtick.labelsize':   9,
    'ytick.labelsize':   9,
    'axes.spines.top':   False,
    'axes.spines.right': False,
})

FS         = 100e6
CROP       = (3500, 7500)
PULSE_PRE  = 100    # samples before peak
PULSE_POST = 500    # samples after peak


# ─────────────────────────────────────────────────────────────────────────────

def load_pulse(data_dir, position, voltage, file_idx=0):
    root     = Path(data_dir)
    volt_dir = root / str(position) / f'{voltage}V'
    files    = sorted(volt_dir.glob('*.mat'))
    if not files:
        raise FileNotFoundError(f'No .mat files in {volt_dir}')
    mat = loadmat(str(files[file_idx % len(files)]), simplify_cells=False)
    sig = mat['tpd']['Data'][0, 0].astype(np.float32)
    lo, hi = CROP
    return sig[:, lo:hi]


def extract_window(ch):
    """Return tight pulse window and time axis in µs (trigger at 0)."""
    env      = np.abs(hilbert(ch.astype(np.float64)))
    peak_idx = int(np.argmax(env))
    start    = max(0, peak_idx - PULSE_PRE)
    end      = min(len(ch), peak_idx + PULSE_POST)
    pulse    = ch[start:end].astype(np.float64)
    # time axis: sample 4000 in full signal = trigger = 0 µs
    # full crop starts at sample 3500, so trigger is at index 500 in crop
    t_us = (np.arange(start, end) - 500) / FS * 1e6
    peak_local = peak_idx - start
    return pulse, t_us, peak_local


def compute_fft(pulse):
    N   = len(pulse)
    win = tukey(N, alpha=0.25)
    nfft = max(N * 4, 2048)   # zero-pad for smoother spectrum
    X   = np.fft.rfft(pulse * win, n=nfft)
    mag = np.abs(X)
    mag /= mag.max() + 1e-12
    f   = np.fft.rfftfreq(nfft, d=1.0/FS) / 1e6   # MHz
    return f, mag


def compute_spectrogram(pulse, t_us):
    N       = len(pulse)
    nperseg = min(64, N // 4)
    hop     = max(1, nperseg // 8)
    win     = hann(nperseg)

    n_frames = max(1, (N - nperseg) // hop + 1)
    n_freqs  = nperseg // 2 + 1
    Sxx      = np.zeros((n_freqs, n_frames))

    for i in range(n_frames):
        s     = i * hop
        frame = pulse[s:s + nperseg]
        if len(frame) < nperseg:
            frame = np.pad(frame, (0, nperseg - len(frame)))
        Sxx[:, i] = np.abs(np.fft.rfft(frame * win, n=nperseg)) ** 2

    # time axis aligned to pulse t_us
    dt    = 1.0 / FS * 1e6   # µs per sample
    t_sg  = t_us[0] + (np.arange(n_frames) * hop + nperseg // 2) * dt
    f_sg  = np.fft.rfftfreq(nperseg, d=1.0/FS) / 1e6   # MHz

    Sxx_dB = 10 * np.log10(Sxx / (Sxx.max() + 1e-12) + 1e-6)
    Sxx_dB = np.clip(Sxx_dB, -50, 0)
    return t_sg, f_sg, Sxx_dB


def get_rise_time(pulse, t_us, peak_local):
    env  = np.abs(hilbert(pulse))
    pv   = env[peak_local]
    pre  = env[:peak_local + 1]
    lo_i = np.where(pre >= 0.10 * pv)[0]
    hi_i = np.where(pre >= 0.90 * pv)[0]
    t_10 = t_us[lo_i[0]] if len(lo_i) else None
    t_90 = t_us[hi_i[0]] if len(hi_i) else None
    return t_10, t_90


# ─────────────────────────────────────────────────────────────────────────────

def make_figure(sig, channel, position, voltage):
    ch = sig[channel]
    pulse, t_us, peak_local = extract_window(ch)

    f_fft, mag          = compute_fft(pulse)
    t_sg, f_sg, Sxx_dB  = compute_spectrogram(pulse, t_us)
    t_10, t_90          = get_rise_time(pulse, t_us, peak_local)
    pk_val              = pulse[peak_local]

    # adaptive freq axis
    cum   = np.cumsum(mag); cum /= cum[-1]
    f_max = float(np.clip(f_fft[np.searchsorted(cum, 0.99)] * 1.5, 8.0, 50.0))

    fig = plt.figure(figsize=(13, 4.2))
    gs  = gridspec.GridSpec(1, 3, figure=fig,
                            left=0.07, right=0.97,
                            bottom=0.14, top=0.87,
                            wspace=0.38)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    # ── (a) windowed time domain ──────────────────────────────────────────
    ax1.plot(t_us, pulse, color='#1d3557', linewidth=1.0)
    ax1.axvline(0, color='#888', linestyle='--', linewidth=0.7, alpha=0.6)
    ax1.axhline(0, color='#ccc', linewidth=0.5)

    ax1.set_xlabel(r'Time ($\mu$s)')
    ax1.set_ylabel('Amplitude (a.u.)')
    ax1.set_title('(a) Windowed time-domain pulse')
    ax1.set_xlim(t_us[0], t_us[-1])
    ax1.grid(True, alpha=0.2)

    # ── (b) FFT ───────────────────────────────────────────────────────────
    mask = f_fft <= f_max
    ax2.plot(f_fft[mask], mag[mask], color='#1d3557', linewidth=1.0)
    ax2.fill_between(f_fft[mask], mag[mask], alpha=0.18, color='#1d3557')
    ax2.set_xlabel('Frequency (MHz)')
    ax2.set_ylabel('Normalised magnitude')
    ax2.set_title('(b) Magnitude spectrum (FFT)')
    ax2.set_xlim(0, f_max)
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.25)

    # ── (c) Spectrogram ───────────────────────────────────────────────────
    f_mask = f_sg <= f_max
    im = ax3.pcolormesh(t_sg, f_sg[f_mask], Sxx_dB[f_mask, :],
                        cmap='inferno', shading='auto',
                        vmin=-50, vmax=0)
    ax3.axvline(0, color='white', linestyle='--', linewidth=0.7, alpha=0.7)
    ax3.set_xlabel(r'Time ($\mu$s)')
    ax3.set_ylabel('Frequency (MHz)')
    ax3.set_title('(c) Spectrogram (STFT)')
    ax3.set_xlim(t_sg[0], t_sg[-1])
    ax3.set_ylim(0, f_max)
    cbar = fig.colorbar(im, ax=ax3, pad=0.02, aspect=25)
    cbar.set_label('Power (dB)', fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    fig.suptitle(
        f'PD pulse representations  —  {position} m,  {voltage} V,  '
        f'Ch{channel+1}',
        fontsize=10, y=0.98
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────

def make_distance_comparison(data_dir, voltage, channel, file_idx,
                             positions=None):
    if positions is None:
        positions = [100, 500, 1000, 1900]

    fig, axes = plt.subplots(len(positions), 3,
                              figsize=(13, 3.0 * len(positions)))

    for row, pos in enumerate(positions):
        try:
            s = load_pulse(data_dir, pos, voltage, file_idx)
        except FileNotFoundError:
            print(f'  Skipping {pos}m'); continue

        ch = s[channel]
        pulse, t_us, peak_local = extract_window(ch)
        f_fft, mag              = compute_fft(pulse)
        t_sg, f_sg, Sxx_dB      = compute_spectrogram(pulse, t_us)

        cum   = np.cumsum(mag); cum /= cum[-1]
        f_max = float(np.clip(f_fft[np.searchsorted(cum, 0.99)] * 1.5, 8.0, 50.0))
        f_mask = f_sg <= f_max

        axes[row, 0].plot(t_us, pulse, color='#1d3557', linewidth=0.8)
        axes[row, 0].axvline(0, color='#888', linestyle='--', linewidth=0.6, alpha=0.6)
        axes[row, 0].set_ylabel(f'{pos} m\nAmplitude', fontsize=9)
        axes[row, 0].set_xlim(t_us[0], t_us[-1])
        axes[row, 0].grid(True, alpha=0.2)

        mask = f_fft <= f_max
        axes[row, 1].plot(f_fft[mask], mag[mask], color='#1d3557', linewidth=0.8)
        axes[row, 1].fill_between(f_fft[mask], mag[mask], alpha=0.15, color='#1d3557')
        axes[row, 1].set_xlim(0, f_max); axes[row, 1].set_ylim(0, 1.05)
        axes[row, 1].grid(True, alpha=0.2)

        axes[row, 2].pcolormesh(t_sg, f_sg[f_mask], Sxx_dB[f_mask, :],
                                 cmap='inferno', shading='auto', vmin=-50, vmax=0)
        axes[row, 2].axvline(0, color='white', linestyle='--', linewidth=0.6, alpha=0.7)
        axes[row, 2].set_xlim(t_sg[0], t_sg[-1]); axes[row, 2].set_ylim(0, f_max)

    for ax, title in zip(axes[0], ['Time domain', 'FFT', 'Spectrogram']):
        ax.set_title(title, fontsize=10)
    for col, xl in enumerate([r'Time ($\mu$s)', 'Frequency (MHz)', r'Time ($\mu$s)']):
        axes[-1, col].set_xlabel(xl, fontsize=9)

    fig.suptitle(f'Signal representations — {voltage} V, Ch{channel+1}', fontsize=10)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sig = load_pulse(args.data_dir, args.position, args.voltage, args.file_idx)

    fig = make_figure(sig, args.channel, args.position, args.voltage)
    png = out_dir / f'fig_pulse_repr_{args.position}m_{args.voltage}V.png'
    fig.savefig(str(png), dpi=150, bbox_inches='tight')
    print(f'Saved: {png}')
    if args.save_pdf:
        fig.savefig(str(png).replace('.png', '.pdf'), bbox_inches='tight')
    plt.close(fig)

    fig2 = make_distance_comparison(args.data_dir, args.voltage,
                                    args.channel, args.file_idx)
    png2 = out_dir / f'fig_distance_comparison_{args.voltage}V.png'
    fig2.savefig(str(png2), dpi=150, bbox_inches='tight')
    print(f'Saved: {png2}')
    if args.save_pdf:
        fig2.savefig(str(png2).replace('.png', '.pdf'), bbox_inches='tight')
    plt.close(fig2)
    print('Done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',  default='data/raw/measurements')
    parser.add_argument('--out_dir',   default='results/figures')
    parser.add_argument('--position',  type=int, default=100)
    parser.add_argument('--voltage',   type=int, default=5)
    parser.add_argument('--channel',   type=int, default=0)
    parser.add_argument('--file_idx',  type=int, default=0)
    parser.add_argument('--save_pdf',  action='store_true')
    args = parser.parse_args()
    main(args)
