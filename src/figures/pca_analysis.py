# src/pca_analysis.py

import sys, os
sys.path.append(os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from dataset import build_file_list, load_mat_file, POSITIONS
from features import extract_features, feature_names, CROP

# ── Config ────────────────────────────────────────────────────────────────────

ROOT        = 'data/raw/measurements'
RESULTS     = Path('results/pca')
RESULTS.mkdir(parents=True, exist_ok=True)

# Sample at most N files per (position, voltage) to keep it fast
MAX_PER_FOLDER = 20
N_COMPONENTS   = 10    # PCA components to compute

COLORS  = cm.tab20(np.linspace(0, 1, 12))
POS_STR = [f"{p}m" for p in POSITIONS]


# ── Build feature matrix ───────────────────────────────────────────────────────

def build_matrix(root_dir, max_per_folder=MAX_PER_FOLDER):
    """
    Build feature matrix X, position labels y_pos, voltage labels y_volt.
    Samples max_per_folder files per (position, voltage) folder.
    """
    root    = Path(root_dir)
    X, y_pos, y_volt, y_pos_m = [], [], [], []

    from dataset import POS_TO_LABEL
    pos_idx = 0
    for pos_folder in tqdm(sorted(root.iterdir()), desc='Positions'):
        if not pos_folder.is_dir():
            continue
        try:
            position = int(pos_folder.name)
        except ValueError:
            continue
        if position not in POS_TO_LABEL:
            continue
        label = POS_TO_LABEL[position]

        for volt_folder in sorted(pos_folder.iterdir()):
            if not volt_folder.is_dir():
                continue
            try:
                voltage = float(volt_folder.name.replace('V', ''))
            except ValueError:
                continue

            files = sorted(volt_folder.glob('*.mat'))[:max_per_folder]
            for f in files:
                try:
                    sig = load_mat_file(str(f))
                    sig = sig[:, CROP[0]:CROP[1]]
                    feat = extract_features(sig)
                    X.append(feat)
                    y_pos.append(label)
                    y_volt.append(voltage)
                    y_pos_m.append(position)
                except Exception:
                    continue

    X      = np.array(X,     dtype=np.float32)
    y_pos  = np.array(y_pos, dtype=np.int32)
    y_volt = np.array(y_volt,dtype=np.float32)
    y_pos_m = np.array(y_pos_m, dtype=np.int32)

    print(f"\nFeature matrix: {X.shape}")
    print(f"Positions: {len(np.unique(y_pos))} | "
          f"Voltages: {sorted(np.unique(y_volt).tolist())}")
    return X, y_pos, y_volt, y_pos_m


# ── Plot helpers ──────────────────────────────────────────────────────────────

def scatter2d(coords, labels, label_names, colors, title, path,
              alpha=0.4, s=12):
    fig, ax = plt.subplots(figsize=(10, 8))
    for i, name in enumerate(label_names):
        mask = labels == i
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   color=colors[i], label=name,
                   alpha=alpha, s=s, linewidths=0)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.legend(loc='upper right', fontsize=7,
              ncol=2, markerscale=2)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():

    # 1. Build feature matrix
    X, y_pos, y_volt, y_pos_m = build_matrix(ROOT)

    # 2. Scale
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)
    fn     = feature_names()

    # ── A. PCA ────────────────────────────────────────────────────────────────

    print("\nRunning PCA...")
    pca    = PCA(n_components=N_COMPONENTS)
    X_pca  = pca.fit_transform(X_sc)
    evr    = pca.explained_variance_ratio_

    # Scree plot
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(1, N_COMPONENTS+1), evr*100, color='#3498db')
    ax.plot(range(1, N_COMPONENTS+1), np.cumsum(evr)*100,
            'r-o', markersize=4, label='Cumulative')
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Explained Variance (%)')
    ax.set_title('PCA Scree Plot — Exp1 Features')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS / 'pca_scree.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"PCA: PC1={evr[0]*100:.1f}% | PC2={evr[1]*100:.1f}% | "
          f"Top3={sum(evr[:3])*100:.1f}%")

    # PCA scatter — colored by position
    scatter2d(X_pca, y_pos, POS_STR, COLORS,
              'PCA — Colored by Injection Position',
              RESULTS / 'pca_by_position.png')

    # PCA scatter — colored by voltage
    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(X_pca[:, 0], X_pca[:, 1],
                    c=y_volt, cmap='plasma', alpha=0.4, s=12)
    plt.colorbar(sc, ax=ax, label='Voltage (V)')
    ax.set_title('PCA — Colored by Voltage Level', fontsize=12,
                 fontweight='bold')
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(RESULTS / 'pca_by_voltage.png', dpi=150, bbox_inches='tight')
    plt.close()

    # PCA loadings — which features drive PC1 and PC2
    loadings = pca.components_[:2]   # (2, n_features)
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    for i, ax in enumerate(axes):
        idx  = np.argsort(np.abs(loadings[i]))[::-1][:10]
        vals = loadings[i][idx]
        cols = ['#e74c3c' if v > 0 else '#3498db' for v in vals]
        ax.bar([fn[j] for j in idx], vals, color=cols)
        ax.set_title(f'PC{i+1} Loadings — Top 10 Features '
                     f'({evr[i]*100:.1f}% variance)')
        ax.set_ylabel('Loading')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, axis='y', alpha=0.3)
        ax.axhline(0, color='k', linewidth=0.8)
    plt.tight_layout()
    plt.savefig(RESULTS / 'pca_loadings.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ── B. LDA ────────────────────────────────────────────────────────────────

    print("Running LDA...")
    lda   = LDA(n_components=2)
    X_lda = lda.fit_transform(X_sc, y_pos)

    scatter2d(X_lda, y_pos, POS_STR, COLORS,
              'LDA — Maximally Separating Injection Positions',
              RESULTS / 'lda_by_position.png', alpha=0.5, s=15)

    # ── C. t-SNE ──────────────────────────────────────────────────────────────

    print("Running t-SNE (this takes a minute)...")
    # Use PCA-reduced data as input to t-SNE for speed
    tsne   = TSNE(n_components=2, perplexity=40, random_state=42,
                  n_iter=1000, init='pca', learning_rate='auto')
    X_tsne = tsne.fit_transform(X_pca)

    scatter2d(X_tsne, y_pos, POS_STR, COLORS,
              't-SNE — Colored by Injection Position',
              RESULTS / 'tsne_by_position.png', alpha=0.5, s=15)

    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(X_tsne[:, 0], X_tsne[:, 1],
                    c=y_volt, cmap='plasma', alpha=0.5, s=15)
    plt.colorbar(sc, ax=ax, label='Voltage (V)')
    ax.set_title('t-SNE — Colored by Voltage Level',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2')
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(RESULTS / 'tsne_by_voltage.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ── D. Feature correlation heatmap ────────────────────────────────────────

    print("Computing feature correlations...")
    corr = np.corrcoef(X_sc.T)
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(corr, xticklabels=fn, yticklabels=fn,
                cmap='coolwarm', center=0, vmin=-1, vmax=1,
                ax=ax, linewidths=0.3)
    ax.set_title('Feature Correlation Matrix — Exp1')
    plt.tight_layout()
    plt.savefig(RESULTS / 'feature_correlation.png',
                dpi=150, bbox_inches='tight')
    plt.close()

    # ── E. Feature distributions per position ─────────────────────────────────

    print("Plotting feature distributions...")
    # Pick the top 6 most important features from RF analysis
    top_features = ['amplitude_ratio', 'Ch1_band1_2-5MHz',
                    'Ch2_band1_2-5MHz', 'Ch2_pulse_width',
                    'Ch1_band2_5-10MHz', 'tdoa_us']
    top_idx = [fn.index(f) for f in top_features if f in fn]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Feature Distributions per Injection Position',
                 fontsize=13, fontweight='bold')

    for ax, idx in zip(axes.flat, top_idx):
        feat_name = fn[idx]
        for i, pos in enumerate(POSITIONS):
            mask = y_pos == i
            vals = X_sc[mask, idx]
            ax.plot(sorted(vals),
                    np.linspace(0, 1, mask.sum()),
                    color=COLORS[i], alpha=0.7,
                    linewidth=1.2, label=f'{pos}m')
        ax.set_title(feat_name, fontsize=9)
        ax.set_xlabel('Standardized value')
        ax.set_ylabel('CDF')
        ax.grid(True, alpha=0.2)

    axes.flat[0].legend(fontsize=6, ncol=2)
    plt.tight_layout()
    plt.savefig(RESULTS / 'feature_distributions.png',
                dpi=150, bbox_inches='tight')
    plt.close()

    # ── F. Position separability score ────────────────────────────────────────

    print("Computing per-feature separability...")
    from sklearn.feature_selection import f_classif
    f_scores, p_values = f_classif(X_sc, y_pos)
    idx_sorted = np.argsort(f_scores)[::-1]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(range(len(fn)), f_scores[idx_sorted], color='#2ecc71')
    ax.set_xticks(range(len(fn)))
    ax.set_xticklabels([fn[i] for i in idx_sorted],
                       rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('F-score (ANOVA)')
    ax.set_title('Feature Separability for Position Classification '
                 '(higher = more discriminative)')
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS / 'feature_separability_position.png',
                dpi=150, bbox_inches='tight')
    plt.close()

    # Voltage separability
    from sklearn.feature_selection import f_regression
    f_scores_v, _ = f_regression(X_sc, y_volt)
    idx_sorted_v  = np.argsort(f_scores_v)[::-1]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(range(len(fn)), f_scores_v[idx_sorted_v], color='#e74c3c')
    ax.set_xticks(range(len(fn)))
    ax.set_xticklabels([fn[i] for i in idx_sorted_v],
                       rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('F-score (regression)')
    ax.set_title('Feature Separability for Voltage Regression '
                 '(higher = more predictive)')
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS / 'feature_separability_voltage.png',
                dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nTop 5 features for POSITION:")
    for i in idx_sorted[:5]:
        print(f"  {fn[i]:35s} F={f_scores[i]:.1f}")

    print(f"\nTop 5 features for VOLTAGE:")
    for i in idx_sorted_v[:5]:
        print(f"  {fn[i]:35s} F={f_scores_v[i]:.1f}")

    print(f"\nAll plots saved to {RESULTS}/")


if __name__ == '__main__':
    main()