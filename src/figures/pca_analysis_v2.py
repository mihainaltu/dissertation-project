"""
pca_analysis_v2.py — PCA / LDA / t-SNE on the v2 feature matrix (122 features)
Loads the cached features_cache.npz from baseline_v2, no re-extraction needed.

Usage:
    python pca_analysis_v2.py
    python pca_analysis_v2.py --cache results/baseline_v2/features_cache.npz
    python pca_analysis_v2.py --max_per_class 200  # subsample for speed
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE

# ── label map ────────────────────────────────────────────────────────────────
LABEL_TO_POS = {
    0: 100, 1: 200, 2: 300,  3: 500,  4: 700,  5: 900,
    6:1000, 7:1300, 8:1500,  9:1600, 10:1800, 11:1900
}
N_CLASSES  = 12
POS_LABELS = [f'{LABEL_TO_POS[i]}m' for i in range(N_CLASSES)]

# Consistent colour palette across all plots
# Consistent high-contrast colour palette across all plots
# Chosen to be more distinguishable than tab20 for 12 classes
COLORS = [
    '#1f77b4',  # 100 m  - blue
    '#ff7f0e',  # 200 m  - orange
    '#2ca02c',  # 300 m  - green
    '#d62728',  # 500 m  - red
    '#9467bd',  # 700 m  - purple
    '#8c564b',  # 900 m  - brown
    '#e377c2',  # 1000 m - pink
    '#7f7f7f',  # 1300 m - grey
    '#bcbd22',  # 1500 m - olive
    '#17becf',  # 1600 m - cyan
    '#000000',  # 1800 m - black
    '#FFD700',  # 1900 m - yellow/gold
]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_cache(cache_path):
    c = np.load(cache_path)
    X, y, volts = c['X'], c['y'], c['volts']
    # Replace any NaN/Inf just in case
    X = np.where(np.isfinite(X), X, 0.0)
    print(f'Loaded: X={X.shape}  classes={np.unique(y)}  voltages={np.unique(volts)}')
    return X, y, volts


def subsample(X, y, volts, max_per_class):
    """Keep at most max_per_class samples per class (stratified)."""
    idx = []
    rng = np.random.default_rng(42)
    for c in range(N_CLASSES):
        ci = np.where(y == c)[0]
        if len(ci) > max_per_class:
            ci = rng.choice(ci, max_per_class, replace=False)
        idx.append(ci)
    idx = np.concatenate(idx)
    return X[idx], y[idx], volts[idx]


def scatter2d(ax, Z, labels, color_by, cmap_vals, label_names,
              title, alpha=0.5, s=18):
    """Generic 2-D scatter coloured by an integer array."""
    for c in np.unique(color_by):
        mask = color_by == c
        ax.scatter(Z[mask, 0], Z[mask, 1],
                   color=cmap_vals[int(c)],
                   label=label_names[int(c)],
                   alpha=alpha, s=s, linewidths=0)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('Component 1'); ax.set_ylabel('Component 2')
    ax.grid(True, alpha=0.2)
    
    
def scatter3d(ax, Z, color_by, cmap_vals, label_names,
              title, alpha=0.75, s=28):
    """Generic 3-D scatter coloured by an integer array."""
    for c in np.unique(color_by):
        mask = color_by == c
        ax.scatter(
            Z[mask, 0], Z[mask, 1], Z[mask, 2],
            color=cmap_vals[int(c)],
            label=label_names[int(c)],
            alpha=alpha,
            s=s,
            depthshade=True,
            edgecolors='none'
        )

    ax.set_title(title, fontsize=11)
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_zlabel('Component 3')
    ax.grid(True, alpha=0.2)


def add_legend(ax, label_names, ncol=2):
    ax.legend(label_names, fontsize=7, markerscale=1.2,
              loc='best', ncol=ncol,
              framealpha=0.7, handlelength=1)


# ─────────────────────────────────────────────────────────────────────────────
# 1. PCA
# ─────────────────────────────────────────────────────────────────────────────

def run_pca(X_sc, y, volts, out_dir):
    print('  Running PCA ...')
    pca   = PCA(n_components=min(30, X_sc.shape[1]))
    Z_all = pca.fit_transform(X_sc)

    # ── scree plot ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    cumvar = np.cumsum(pca.explained_variance_ratio_) * 100
    ax.bar(range(1, len(cumvar)+1),
           pca.explained_variance_ratio_ * 100, color='steelblue', alpha=0.7)
    ax.plot(range(1, len(cumvar)+1), cumvar, 'o-', color='red', ms=4)
    ax.axhline(95, color='gray', linestyle='--', linewidth=0.8, label='95%')
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Explained Variance (%)')
    ax.set_title('PCA Scree Plot — v2 features (122)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / 'pca_scree_v2.png', dpi=150)
    plt.close(fig)
    n95 = int(np.searchsorted(cumvar, 95.0)) + 1
    print(f'    Components for 95% variance: {n95}')

    Z = Z_all[:, :2]

    # ── by position ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter2d(ax, Z, y, y, COLORS, POS_LABELS,
              'PCA — Colored by Injection Position (v2 features)')
    add_legend(ax, POS_LABELS)
    plt.tight_layout()
    fig.savefig(out_dir / 'pca_by_position_v2.png', dpi=150)
    plt.close(fig)

    # ── by voltage ───────────────────────────────────────────────────────
    v_cmap   = cm.get_cmap('plasma', 10)
    v_colors = [v_cmap(i) for i in range(10)]
    v_labels = [f'{v}V' for v in range(1, 11)]
    fig, ax  = plt.subplots(figsize=(10, 8))
    scatter2d(ax, Z, y, volts - 1, v_colors, v_labels,
              'PCA — Colored by Voltage (v2 features)', alpha=0.4)
    add_legend(ax, v_labels, ncol=2)
    plt.tight_layout()
    fig.savefig(out_dir / 'pca_by_voltage_v2.png', dpi=150)
    plt.close(fig)

    return Z_all, pca, n95


# ─────────────────────────────────────────────────────────────────────────────
# 2. LDA
# ─────────────────────────────────────────────────────────────────────────────

def run_lda(X_sc, y, out_dir):
    print('  Running LDA ...')
    lda = LDA(n_components=min(N_CLASSES - 1, X_sc.shape[1]))
    Z   = lda.fit_transform(X_sc, y)

    # ── explained variance ratio ─────────────────────────────────────────
    evr = lda.explained_variance_ratio_
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(range(1, len(evr)+1), evr * 100, color='darkorange')
    ax.set_xlabel('LDA Component')
    ax.set_ylabel('Explained Variance (%)')
    ax.set_title('LDA Explained Variance — v2 features')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / 'lda_variance_v2.png', dpi=150)
    plt.close(fig)
    print(f'    LD1 explains {evr[0]*100:.1f}%,  LD2 {evr[1]*100:.1f}%')

    # ── 2D scatter ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter2d(ax, Z, y, y, COLORS, POS_LABELS,
              'LDA — Maximally Separating Injection Positions (v2 features)')
    add_legend(ax, POS_LABELS)
    plt.tight_layout()
    fig.savefig(out_dir / 'lda_by_position_v2.png', dpi=150)
    plt.close(fig)

    # ── LD1 vs LD3 (often reveals extra structure) ───────────────────────
    if Z.shape[1] >= 3:
        fig, ax = plt.subplots(figsize=(10, 8))
        Z13 = Z[:, [0, 2]]
        scatter2d(ax, Z13, y, y, COLORS, POS_LABELS,
                  'LDA — LD1 vs LD3 (v2 features)')
        add_legend(ax, POS_LABELS)
        plt.tight_layout()
        fig.savefig(out_dir / 'lda_ld1_ld3_v2.png', dpi=150)
        plt.close(fig)

    # ── class overlap heatmap: mean pairwise distance in LDA space ───────
    means = np.array([Z[y == c].mean(axis=0) for c in range(N_CLASSES)])
    dist  = np.sqrt(((means[:, None] - means[None, :]) ** 2).sum(axis=2))
    fig, ax = plt.subplots(figsize=(9, 7))
    import seaborn as sns
    sns.heatmap(dist, annot=True, fmt='.1f', cmap='YlOrRd_r',
                xticklabels=POS_LABELS, yticklabels=POS_LABELS, ax=ax)
    ax.set_title('Pairwise Distance Between Class Centroids in LDA Space (v2)')
    plt.tight_layout()
    fig.savefig(out_dir / 'lda_centroid_distances_v2.png', dpi=150)
    plt.close(fig)
    print('    Saved centroid distance heatmap')

    return Z


# ─────────────────────────────────────────────────────────────────────────────
# 3. t-SNE
# ─────────────────────────────────────────────────────────────────────────────

def run_tsne(X_sc, y, volts, out_dir, max_per_class=200):
    print(f'  Running 3D t-SNE (max {max_per_class} per class) ...')

    # Subsample for speed — t-SNE is O(n^2) on CPU
    Xs, ys, vs = subsample(X_sc, y, volts, max_per_class)
    print(f'    Subsampled to {len(ys)} points')

    # 3D t-SNE
    tsne = TSNE(
        n_components=3,
        perplexity=40,
        learning_rate='auto',
        init='pca',
        n_iter=1000,
        random_state=42,
        n_jobs=-1,
    )

    Z = tsne.fit_transform(Xs)

    # ── 3D by position ───────────────────────────────────────────────────
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    scatter3d(
        ax, Z, ys, COLORS, POS_LABELS,
        '3D t-SNE — Colored by Injection Position (v2 features)',
        alpha=0.8,
        s=30
    )

    ax.view_init(elev=22, azim=45)
    ax.legend(
        fontsize=8,
        markerscale=1.3,
        loc='upper left',
        bbox_to_anchor=(1.02, 1.0),
        framealpha=0.8
    )

    plt.tight_layout()
    fig.savefig(out_dir / 'tsne_3d_by_position_v2.png', dpi=200, bbox_inches='tight')
    plt.close(fig)

    # ── 3D by voltage ────────────────────────────────────────────────────
    v_cmap   = cm.get_cmap('plasma', 10)
    v_colors = [v_cmap(i) for i in range(10)]
    v_labels = [f'{v}V' for v in range(1, 11)]

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    scatter3d(
        ax, Z, vs - 1, v_colors, v_labels,
        '3D t-SNE — Colored by Voltage (v2 features)',
        alpha=0.65,
        s=26
    )

    ax.view_init(elev=22, azim=45)
    ax.legend(
        fontsize=8,
        markerscale=1.3,
        loc='upper left',
        bbox_to_anchor=(1.02, 1.0),
        framealpha=0.8
    )

    plt.tight_layout()
    fig.savefig(out_dir / 'tsne_3d_by_voltage_v2.png', dpi=200, bbox_inches='tight')
    plt.close(fig)

    # ── Optional: keep a 2D projection of the 3D t-SNE for thesis comparison ──
    fig, ax = plt.subplots(figsize=(11, 9))
    scatter2d(
        ax, Z[:, :2], ys, ys, COLORS, POS_LABELS,
        't-SNE — First Two Components of 3D Embedding',
        alpha=0.7,
        s=24
    )
    add_legend(ax, POS_LABELS)
    plt.tight_layout()
    fig.savefig(out_dir / 'tsne_3d_projection_by_position_v2.png', dpi=200)
    plt.close(fig)

    print('    Saved 3D t-SNE plots')

    return Z


# ─────────────────────────────────────────────────────────────────────────────
# 4. Voltage effect: within-class variance decomposition
# ─────────────────────────────────────────────────────────────────────────────

def voltage_variance_analysis(X_sc, y, volts, out_dir):
    """
    For each class: compute ratio of within-voltage variance vs total variance.
    Low ratio → voltage has little effect → good for generalization.
    Compare v2 features (all) vs v1 proxy (first 24 features).
    """
    print('  Voltage variance analysis ...')

    def voltage_var_ratio(Xf, yf, vf):
        ratios = []
        for c in range(N_CLASSES):
            ci    = yf == c
            Xc    = Xf[ci]; vc = vf[ci]
            total = np.var(Xc, axis=0).mean() + 1e-12
            # within-voltage variance: average variance within each voltage group
            within = np.mean([
                np.var(Xc[vc == v], axis=0).mean()
                for v in np.unique(vc)
                if (vc == v).sum() > 1
            ])
            ratios.append(within / total)
        return np.array(ratios)

    ratios_all = voltage_var_ratio(X_sc, y, volts)

    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(N_CLASSES)
    ax.bar(x, ratios_all * 100, color='steelblue', label='v2 (122 feat)')
    ax.set_xticks(x); ax.set_xticklabels(POS_LABELS, rotation=45)
    ax.set_ylabel('Within-voltage variance / Total variance (%)')
    ax.set_title('Voltage Effect per Class — lower = more voltage-robust (v2 features)')
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    for i, v in enumerate(ratios_all):
        ax.text(i, v * 100 + 0.5, f'{v*100:.0f}%', ha='center', fontsize=8)
    plt.tight_layout()
    fig.savefig(out_dir / 'voltage_variance_v2.png', dpi=150)
    plt.close(fig)
    print('    Saved voltage variance plot')

    # Print the hard positions
    print('\n    Within-voltage variance ratio per class:')
    for c in range(N_CLASSES):
        flag = ' ← hard' if c in [8, 9, 10] else ''
        print(f'      {POS_LABELS[c]:8s}: {ratios_all[c]*100:.1f}%{flag}')


# ─────────────────────────────────────────────────────────────────────────────
# 5. Feature correlation heatmap (top-30 by RF importance)
# ─────────────────────────────────────────────────────────────────────────────

def feature_correlation(X_sc, out_dir, ranking_csv=None, top_n=30):
    print(f'  Feature correlation heatmap (top {top_n}) ...')
    import pandas as pd
    import seaborn as sns
    from features_v2 import feature_names

    names = feature_names()

    if ranking_csv and Path(ranking_csv).exists():
        df_rank = pd.read_csv(ranking_csv)
        top_names = df_rank['feature'].iloc[:top_n].tolist()
        top_idx   = [names.index(n) for n in top_names if n in names]
    else:
        top_idx   = list(range(min(top_n, X_sc.shape[1])))
        top_names = [names[i] for i in top_idx]

    Xt  = X_sc[:, top_idx]
    corr = np.corrcoef(Xt.T)

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(corr, annot=False, cmap='RdBu_r', center=0,
                vmin=-1, vmax=1,
                xticklabels=top_names, yticklabels=top_names, ax=ax)
    ax.set_title(f'Feature Correlation — Top {top_n} (v2 features)')
    ax.tick_params(axis='x', rotation=90, labelsize=7)
    ax.tick_params(axis='y', rotation=0,  labelsize=7)
    plt.tight_layout()
    fig.savefig(out_dir / 'feature_correlation_v2.png', dpi=150)
    plt.close(fig)
    print('    Saved feature correlation heatmap')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load
    X, y, volts = load_cache(args.cache)

    # Scale once — all analyses use the same scaled matrix
    print('Scaling features ...')
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    # Subsample for t-SNE speed; PCA/LDA use full data
    print(f'\nTotal samples: {len(y)}')

    print('\n── PCA ──────────────────────────────────────────')
    run_pca(X_sc, y, volts, out_dir)

    print('\n── LDA ──────────────────────────────────────────')
    run_lda(X_sc, y, out_dir)

    print('\n── t-SNE ────────────────────────────────────────')
    run_tsne(X_sc, y, volts, out_dir, max_per_class=args.max_per_class)

    print('\n── Voltage variance ─────────────────────────────')
    voltage_variance_analysis(X_sc, y, volts, out_dir)

    print('\n── Feature correlation ──────────────────────────')
    ranking_csv = str(Path(args.cache).parent / 'feature_ranking.csv')
    feature_correlation(X_sc, out_dir,
                        ranking_csv=ranking_csv, top_n=30)

    print(f'\nDone. All plots saved to: {out_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache',
                        default='results/baseline_v2/features_cache.npz',
                        help='Path to features_cache.npz')
    parser.add_argument('--out_dir',
                        default='results/pca_v2',
                        help='Output directory for plots')
    parser.add_argument('--max_per_class', type=int, default=200,
                        help='Max samples per class for t-SNE (speed)')
    args = parser.parse_args()
    main(args)