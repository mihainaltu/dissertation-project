"""
plot_two_channel_waveform.py — Figure for thesis.
Clean academic style: two stacked panels, minimal decoration.
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.io import loadmat

plt.rcParams.update({
    'font.family':      'serif',
    'font.size':        10,
    'axes.titlesize':   10,
    'axes.labelsize':   10,
    'xtick.labelsize':  9,
    'ytick.labelsize':  9,
    'lines.linewidth':  0.8,
    'axes.linewidth':   0.6,
    'grid.linewidth':   0.4,
    'grid.alpha':       0.3,
    'legend.fontsize':  9,
    'legend.frameon':   False,
})

FS          = 100e6
PRE_TRIGGER = 4000
CROP_LO     = 3500
CROP_HI     = 7500


def load_full(data_dir, position, voltage, file_idx=0):
    root     = Path(data_dir)
    volt_dir = root / str(position) / f'{voltage}V'
    files    = sorted(volt_dir.glob('*.mat'))
    if not files:
        raise FileNotFoundError(f'No .mat files in {volt_dir}')
    mat = loadmat(str(files[file_idx % len(files)]), simplify_cells=False)
    return mat['tpd']['Data'][0, 0].astype(np.float32)


def make_figure(sig, position, voltage):
    N      = sig.shape[1]
    t_full = (np.arange(N) - PRE_TRIGGER) / FS * 1e6
    t_crop = t_full[CROP_LO:CROP_HI]
    t0     = t_full[CROP_LO]
    t1     = t_full[CROP_HI - 1]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5.5),
                                    constrained_layout=True)

    # ── (a) full window ───────────────────────────────────────────────────
    ax1.plot(t_full, sig[0], color='black',    linewidth=0.7, label='Ch1')
    ax1.plot(t_full, sig[1], color='#555555',  linewidth=0.7,
             linestyle='--', label='Ch2')
    ax1.axvline(0,  color='black', linewidth=0.8, linestyle=':')
    ax1.axvline(t0, color='black', linewidth=0.6, linestyle='-.')
    ax1.axvline(t1, color='black', linewidth=0.6, linestyle='-.')
    ax1.set_xlim(t_full[0], t_full[-1])
    ax1.set_xlabel(r'Time ($\mu$s)')
    ax1.set_ylabel('Amplitude (V)')
    ax1.set_title('(a) Full acquisition window (200 µs)')
    ax1.legend(loc='upper right')
    ax1.grid(True)


    # ── (b) crop window ───────────────────────────────────────────────────
    ax2.plot(t_crop, sig[0, CROP_LO:CROP_HI], color='black',
             linewidth=0.8, label='Ch1')
    ax2.plot(t_crop, sig[1, CROP_LO:CROP_HI], color='#555555',
             linewidth=0.8, linestyle='--', label='Ch2')
    ax2.axvline(0, color='black', linewidth=0.8, linestyle=':')
    ax2.set_xlim(t_crop[0], t_crop[-1])
    ax2.set_xlabel(r'Time ($\mu$s)')
    ax2.set_ylabel('Amplitude (V)')
    ax2.set_title(r'(b) Crop window ($-$5 µs to +35 µs)')
    ax2.legend(loc='upper right')
    ax2.grid(True)

    return fig


def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sig = load_full(args.data_dir, args.position, args.voltage, args.file_idx)
    fig = make_figure(sig, args.position, args.voltage)
    png = out_dir / f'fig_two_channel_{args.position}m_{args.voltage}V.png'
    fig.savefig(str(png), dpi=150, bbox_inches='tight')
    print(f'Saved: {png}')
    if args.save_pdf:
        fig.savefig(str(png).replace('.png', '.pdf'), bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',  default='data/raw/measurements')
    parser.add_argument('--out_dir',   default='results/figures')
    parser.add_argument('--position',  type=int, default=900)
    parser.add_argument('--voltage',   type=int, default=5)
    parser.add_argument('--file_idx',  type=int, default=0)
    parser.add_argument('--save_pdf',  action='store_true')
    args = parser.parse_args()
    main(args)