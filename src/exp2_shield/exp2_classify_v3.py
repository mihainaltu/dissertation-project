"""
Experiment 2 - Classification v3
Usage:
    python3 exp2_classify_v3.py --features exp2_features.csv --out_dir results/

Runs:
  1. All models with freq_mhz as feature (baseline)
  2. XGB with frequency filtering analysis (drop low-performing freqs)
  3. Per-frequency accuracy breakdown
  4. Thesis-quality plots saved to out_dir
"""

import pandas as pd
import numpy as np
import argparse
import os
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# ── Style ──────────────────────────────────────────────────────────────────
COLORS = {
    'blue':   '#2563EB',
    'green':  '#16A34A',
    'red':    '#DC2626',
    'orange': '#EA580C',
    'purple': '#7C3AED',
    'gray':   '#6B7280',
    'light':  '#F3F4F6',
}
MODEL_COLORS = {
    'RF':  COLORS['blue'],
    'SVM': COLORS['orange'],
    'kNN': COLORS['purple'],
    'XGB': COLORS['green'],
    'GBT': COLORS['green'],
}

plt.rcParams.update({
    'font.family':      'DejaVu Sans',
    'font.size':        11,
    'axes.titlesize':   13,
    'axes.labelsize':   12,
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'figure.dpi':       150,
    'savefig.dpi':      300,
    'savefig.bbox':     'tight',
    'savefig.facecolor':'white',
})

# ── Features ───────────────────────────────────────────────────────────────
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
    'freq_mhz',
]


def get_models():
    models = {
        'RF':  RandomForestClassifier(n_estimators=200, random_state=42),
        'SVM': SVC(kernel='rbf', C=10, gamma='scale', random_state=42),
        'kNN': KNeighborsClassifier(n_neighbors=5),
    }
    if HAS_XGB:
        models['XGB'] = XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            eval_metric='logloss', random_state=42
        )
    else:
        models['GBT'] = GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42
        )
    return models


def clone_model(name):
    return get_models()[name]


def lofo_cv(df, model_name, feature_cols):
    files = df['file'].unique()
    y_true_all, y_pred_all, freq_all = [], [], []
    label_map = {l: i for i, l in enumerate(sorted(df['label'].unique()))}
    inv_map   = {i: l for l, i in label_map.items()}

    for test_file in files:
        train_df = df[df['file'] != test_file]
        test_df  = df[df['file'] == test_file]

        X_train = train_df[feature_cols].values
        y_train = np.array([label_map[l] for l in train_df['label'].values])
        X_test  = test_df[feature_cols].values
        y_test  = test_df['label'].values
        freq    = test_df['freq_mhz'].values

        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        clf = clone_model(model_name)
        clf.fit(X_train, y_train)
        preds_enc = clf.predict(X_test)
        try:
            preds = np.array([inv_map[p] for p in preds_enc])
        except KeyError:
            preds = preds_enc

        y_true_all.extend(y_test)
        y_pred_all.extend(preds)
        freq_all.extend(freq)

    return np.array(y_true_all), np.array(y_pred_all), np.array(freq_all)


def per_freq_accuracy(y_true, y_pred, freqs, all_freqs):
    results = {}
    for f in all_freqs:
        mask = freqs == f
        if mask.sum() == 0:
            continue
        results[f] = accuracy_score(y_true[mask], y_pred[mask])
    return results


# ── Plots ───────────────────────────────────────────────────────────────────

def plot_model_comparison(model_results, out_dir):
    """Bar chart comparing all models."""
    fig, ax = plt.subplots(figsize=(8, 5))
    names = list(model_results.keys())
    accs  = [model_results[n] * 100 for n in names]
    colors = [MODEL_COLORS.get(n, COLORS['gray']) for n in names]
    bars = ax.bar(names, accs, color=colors, width=0.5, zorder=3)

    ax.set_ylim(50, 100)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Model Comparison - LOFO-CV Accuracy\nExperiment 2 (Lab Data, All Frequencies)')
    ax.axhline(90, color='gray', linewidth=0.8, linestyle='--', alpha=0.6, zorder=2)
    ax.grid(axis='y', alpha=0.3, zorder=0)

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)

    plt.tight_layout()
    path = os.path.join(out_dir, 'model_comparison.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def plot_per_frequency(freq_acc_all, freq_acc_filtered, all_freqs, out_dir):
    """Per-frequency accuracy: all freqs vs filtered, side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, freq_acc, title in zip(
        axes,
        [freq_acc_all, freq_acc_filtered],
        ['All Frequencies (1–20 MHz)', 'Filtered (≥4 MHz)']
    ):
        freqs  = sorted(freq_acc.keys())
        accs   = [freq_acc[f] * 100 for f in freqs]
        colors = [COLORS['red'] if a < 80 else COLORS['orange'] if a < 90 else COLORS['green'] for a in accs]
        bars = ax.bar([str(int(f)) for f in freqs], accs, color=colors, width=0.6, zorder=3)
        ax.set_ylim(0, 110)
        ax.set_xlabel('Injection Frequency (MHz)')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(f'Per-Frequency Accuracy - XGB\n{title}')
        ax.axhline(90, color='gray', linewidth=0.8, linestyle='--', alpha=0.5, zorder=2)
        ax.grid(axis='y', alpha=0.3, zorder=0)
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{acc:.0f}', ha='center', va='bottom', fontsize=8)

        # Legend
        patches = [
            mpatches.Patch(color=COLORS['green'],  label='≥ 90%'),
            mpatches.Patch(color=COLORS['orange'], label='80–90%'),
            mpatches.Patch(color=COLORS['red'],    label='< 80%'),
        ]
        ax.legend(handles=patches, loc='lower right', fontsize=9)

    plt.tight_layout()
    path = os.path.join(out_dir, 'per_frequency_accuracy.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def plot_confusion_matrices(cms, out_dir):
    """Confusion matrices for all models, side by side."""
    names = list(cms.keys())
    fig, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 4))
    if len(names) == 1:
        axes = [axes]

    for ax, name in zip(axes, names):
        cm = cms[name]
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=['DAMAGED', 'HEALTHY']
        )
        disp.plot(ax=ax, colorbar=False, cmap='Blues')
        acc = (cm[0,0] + cm[1,1]) / cm.sum() * 100
        ax.set_title(f'{name}  ({acc:.1f}%)', fontweight='bold')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')

    fig.suptitle('Confusion Matrices - LOFO-CV\nExperiment 2 (Lab Data)', fontsize=13, y=1.02)
    plt.tight_layout()
    path = os.path.join(out_dir, 'confusion_matrices.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def plot_feature_importance(df, feature_cols, out_dir, top_n=20):
    """Horizontal bar chart of RF feature importances."""
    X = StandardScaler().fit_transform(df[feature_cols].values)
    y = df['label'].values
    rf = RandomForestClassifier(n_estimators=500, random_state=42)
    rf.fit(X, y)
    importances = sorted(zip(feature_cols, rf.feature_importances_), key=lambda x: x[1])
    importances = importances[-top_n:]

    fig, ax = plt.subplots(figsize=(8, 7))
    names = [f for f, _ in importances]
    vals  = [v for _, v in importances]
    colors = [COLORS['blue'] if v > np.percentile(vals, 75) else COLORS['gray'] for v in vals]
    bars = ax.barh(names, vals, color=colors, height=0.6)
    ax.set_xlabel('Feature Importance')
    ax.set_title(f'Top {top_n} Feature Importances (RF)\nExperiment 2 - Lab Data')
    ax.grid(axis='x', alpha=0.3)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=8)
    plt.tight_layout()
    path = os.path.join(out_dir, 'feature_importance.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def plot_frequency_filter_sweep(df, model_name, out_dir):
    """Accuracy vs minimum frequency threshold - shows effect of dropping low freqs."""
    all_freqs = sorted(df['freq_mhz'].unique())
    thresholds = range(1, 15)
    accs = []

    for min_freq in thresholds:
        df_f = df[df['freq_mhz'] >= min_freq]
        if df_f['label'].nunique() < 2 or df_f['file'].nunique() < 5:
            accs.append(None)
            continue
        y_true, y_pred, _ = lofo_cv(df_f, model_name, FEATURE_COLS)
        accs.append(accuracy_score(y_true, y_pred) * 100)

    fig, ax = plt.subplots(figsize=(8, 5))
    valid = [(t, a) for t, a in zip(thresholds, accs) if a is not None]
    ts, as_ = zip(*valid)
    ax.plot(ts, as_, 'o-', color=COLORS['blue'], linewidth=2, markersize=7)
    ax.fill_between(ts, as_, alpha=0.1, color=COLORS['blue'])
    ax.axhline(max(as_), color=COLORS['green'], linewidth=1, linestyle='--', alpha=0.7,
               label=f'Best: {max(as_):.1f}% (min freq = {ts[as_.index(max(as_))]} MHz)')
    ax.set_xlabel('Minimum Injection Frequency Used (MHz)')
    ax.set_ylabel('LOFO-CV Accuracy (%)')
    ax.set_title(f'Effect of Dropping Low Frequencies - {model_name}\nExperiment 2 (Lab Data)')
    ax.set_xticks(list(ts))
    ax.set_ylim(80, 102)
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    path = os.path.join(out_dir, 'frequency_filter_sweep.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")
    return valid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', required=True)
    parser.add_argument('--out_dir',  default='results')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.features)
    print(f"Loaded {len(df)} pulse samples from {df['file'].nunique()} files")
    print(f"Label distribution:\n{df['label'].value_counts().to_string()}\n")

    all_freqs = sorted(df['freq_mhz'].unique())
    best_model = 'XGB' if HAS_XGB else 'GBT'

    # ── Part 1: All models ─────────────────────────────────────────────────
    print("="*52)
    print("PART 1: All models (LOFO-CV, all frequencies)")
    print("="*52)
    model_results = {}
    cms = {}
    for name in get_models():
        y_true, y_pred, freqs = lofo_cv(df, name, FEATURE_COLS)
        acc = accuracy_score(y_true, y_pred)
        model_results[name] = acc
        cms[name] = confusion_matrix(y_true, y_pred, labels=['DAMAGED', 'HEALTHY'])
        print(f"\n{name}: {acc*100:.2f}%")
        print(classification_report(y_true, y_pred, digits=3))

    # ── Part 2: Per-frequency breakdown ───────────────────────────────────
    print("\n" + "="*52)
    print(f"PART 2: Per-frequency breakdown ({best_model})")
    print("="*52)
    y_true, y_pred, freqs = lofo_cv(df, best_model, FEATURE_COLS)
    freq_acc_all = per_freq_accuracy(y_true, y_pred, freqs, all_freqs)
    print(f"\n  {'Freq':>6}  {'N':>5}  {'Accuracy':>10}")
    print(f"  {'-'*28}")
    for f in all_freqs:
        n = int((freqs == f).sum())
        acc = freq_acc_all[f] * 100
        print(f"  {f:>6.0f}  {n:>5}  {acc:>9.1f}%")
    print(f"\n  OVERALL: {accuracy_score(y_true, y_pred)*100:.2f}%")

    # ── Part 3: Frequency filtering ───────────────────────────────────────
    print("\n" + "="*52)
    print(f"PART 3: Frequency filter sweep ({best_model})")
    print("="*52)
    sweep = plot_frequency_filter_sweep(df, best_model, args.out_dir)
    best_thresh = max(sweep, key=lambda x: x[1])
    print(f"\nBest threshold: min_freq = {best_thresh[0]} MHz → {best_thresh[1]:.2f}%")

    # Run on filtered dataset
    df_filtered = df[df['freq_mhz'] >= best_thresh[0]]
    y_true_f, y_pred_f, freqs_f = lofo_cv(df_filtered, best_model, FEATURE_COLS)
    acc_filtered = accuracy_score(y_true_f, y_pred_f)
    freq_acc_filtered = per_freq_accuracy(y_true_f, y_pred_f, freqs_f, sorted(df_filtered['freq_mhz'].unique()))
    print(f"Filtered accuracy ({best_thresh[0]}+ MHz): {acc_filtered*100:.2f}%")
    print(classification_report(y_true_f, y_pred_f, digits=3))

    # ── Plots ─────────────────────────────────────────────────────────────
    print("\nGenerating plots...")
    plot_model_comparison(model_results, args.out_dir)
    plot_confusion_matrices(cms, args.out_dir)
    plot_per_frequency(freq_acc_all, freq_acc_filtered, all_freqs, args.out_dir)
    plot_feature_importance(df, FEATURE_COLS, args.out_dir)

    print(f"\nAll plots saved to: {args.out_dir}/")
    print("\nSummary:")
    print(f"  Best model overall:          {best_model} - {model_results[best_model]*100:.2f}%")
    print(f"  Best with frequency filter:  {best_model} (≥{best_thresh[0]} MHz) - {acc_filtered*100:.2f}%")
    print(f"  Worst frequency:             1 MHz ({freq_acc_all.get(1.0, 0)*100:.1f}%)")


if __name__ == '__main__':
    main()