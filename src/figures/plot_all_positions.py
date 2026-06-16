import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / "exp1_localization"))

"""
plot_all_positions.py — Figure for thesis.
12-panel grid showing one representative pulse per injection position,
all at the same voltage level. Envelope (absolute value) plotted for
clarity, x-axis in sample index within the crop window.

Usage:
    python src/plot_all_positions.py
    python src/plot_all_positions.py --voltage 5 --save_pdf
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.io import loadmat
from scipy.signal import hilbert

plt.rcParams.update({
    'font.family':      'serif',
    'font.size':        8,
    'axes.titlesize':   8,
    'axes.labelsize':   7,
    'xtick.labelsize':  6,
    'ytick.labelsize':  6,
    'lines.linewidth':  0.7,
    'axes.linewidth':   0.5,
    'grid.linewidth':   0.3,
    'grid.alpha':       0.4,
})

FS       = 100e6
CROP     = (3500, 7500)
POSITIONS = [100, 200, 300, 500, 700, 900, 1000, 1300, 1500, 1600, 1800, 1900]


def load_crop(data_dir, position, voltage, file_idx=0):
    root     = Path(data_dir)
    volt_dir = root / str(position) / f'{voltage}V'
    files    = sorted(volt_dir.glob('*.mat'))
    if not files:
        return None
    mat = loadmat(str(files[file_idx % len(files)]), simplify_cells=False)
    sig = mat['tpd']['Data'][0, 0].astype(np.float32)
    lo, hi = CROP
    return sig[0, lo:hi]   # Ch1 only, shape (4000,)


def make_figure(data_dir, voltage, file_idx, channel, use_envelope):
    # First pass: load all signals and find global peak amplitude
    signals = {}
    for pos in POSITIONS:
        sig = load_crop(data_dir, pos, voltage, file_idx)
        if sig is None:
            continue
        y = sig.astype(np.float64)
        y = y - y[:200].mean()   # remove DC
        if use_envelope:
            y = np.abs(hilbert(y))
        signals[pos] = y

    # Global y scale: max absolute value across all positions
    global_max = max(np.abs(y).max() for y in signals.values())
    ymax = global_max * 1.15

    fig, axes = plt.subplots(3, 4, figsize=(10, 6.5),
                              constrained_layout=True)
    axes = axes.flatten()

    for idx, pos in enumerate(POSITIONS):
        ax = axes[idx]
        if pos not in signals:
            ax.set_visible(False)
            continue

        y = signals[pos]

        # Find peak as the sample with maximum absolute value
        # in the full crop window — no search restriction
        peak = int(np.argmax(np.abs(y)))

        # Centre 1000-sample window on peak
        start = max(0, peak - 500)
        end   = start + 1000
        if end > len(y):
            end   = len(y)
            start = max(0, end - 1000)

        x = np.arange(start, end)
        ax.plot(x, y[start:end], color='black', linewidth=0.7)
        ax.set_title(f'{pos} m', fontweight='bold')
        ax.set_xlim(start, start + 1000)
        seg  = y[start:end]
        ymax = max(np.abs(seg).max() * 1.2, 0.01)
        ax.set_ylim(-ymax, ymax)
        ax.set_xlabel('Sample index')
        ax.set_ylabel('Amplitude')
        ax.grid(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.suptitle(
        f'Crop window signals at all injection positions - {voltage} V, Ch{channel+1}',
        fontsize=9
    )
    return fig


def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = make_figure(args.data_dir, args.voltage,
                      args.file_idx, args.channel,
                      args.envelope)

    suffix = 'env' if args.envelope else 'raw'
    png = out_dir / f'fig_all_positions_{args.voltage}V_{suffix}.png'
    fig.savefig(str(png), dpi=150, bbox_inches='tight')
    print(f'Saved: {png}')

    if args.save_pdf:
        fig.savefig(str(png).replace('.png', '.pdf'), bbox_inches='tight')
        print(f'Saved PDF')

    plt.close(fig)
    print('Done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',  default='data/raw/measurements')
    parser.add_argument('--out_dir',   default='results/figures')
    parser.add_argument('--voltage',   type=int, default=5)
    parser.add_argument('--channel',   type=int, default=0)
    parser.add_argument('--file_idx',  type=int, default=0)
    parser.add_argument('--envelope',  action='store_true',
                        help='Plot Hilbert envelope instead of raw waveform')
    parser.add_argument('--save_pdf',  action='store_true')
    args = parser.parse_args()
    main(args)
