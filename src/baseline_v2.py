"""
baseline_v2.py — RF + SVM on the extended feature bank (features_v2.py)
Includes:
  - Feature extraction with progress bar
  - ANOVA + Mutual Information + RF importance ranking
  - Feature selection (top-K) and comparison
  - Voltage-invariant split evaluation
  - Saves results to results/baseline_v2/

Usage:
    python baseline_v2.py --data_dir data/raw/measurements --top_k 40
"""

import os
import argparse
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from tqdm import tqdm
from pathlib import Path

from scipy.io import loadmat
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)
from sklearn.feature_selection import (
    f_classif, mutual_info_classif, SelectKBest
)
from sklearn.model_selection import StratifiedShuffleSplit

# ── import our feature extractor ─────────────────────────────────────────────
from features_v2 import extract_features, feature_names, feature_count, CROP

# ── constants ─────────────────────────────────────────────────────────────────
POS_TO_LABEL = {
    100: 0, 200: 1, 300: 2, 500: 3, 700: 4, 900: 5,
    1000: 6, 1300: 7, 1500: 8, 1600: 9, 1800: 10, 1900: 11
}
LABEL_TO_POS = {v: k for k, v in POS_TO_LABEL.items()}
N_CLASSES    = 12
RANDOM_STATE = 42


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def build_file_list(root_dir):
    """Returns list of (filepath, position_label, voltage_int)."""
    samples = []
    root = Path(root_dir)
    for pos, label in POS_TO_LABEL.items():
        pos_dir = root / str(pos)
        if not pos_dir.exists():
            continue
        for volt in range(1, 11):
            volt_dir = pos_dir / f'{volt}V'   # ← was str(volt), now '1V', '2V' etc.
            if not volt_dir.exists():
                continue
            for f in volt_dir.glob('*.mat'):
                samples.append((str(f), label, volt))
    return samples


def load_mat_file(filepath):
    """Load .mat → shape (2, N)."""
    mat = loadmat(filepath, simplify_cells=False)
    data = mat['tpd']['Data'][0, 0]    # (2, 20000)
    return data.astype(np.float32)


def extract_all(samples, desc='Extracting features'):
    """Extract features for all samples. Returns X (N, F), y (N,), volts (N,)."""
    F  = feature_count()
    X  = np.zeros((len(samples), F), dtype=np.float32)
    y  = np.zeros(len(samples), dtype=np.int32)
    vs = np.zeros(len(samples), dtype=np.int32)

    for i, (fp, label, volt) in enumerate(tqdm(samples, desc=desc)):
        try:
            sig = load_mat_file(fp)
            X[i] = extract_features(sig)
        except Exception as e:
            print(f'  [WARN] {fp}: {e}')
        y[i]  = label
        vs[i] = volt

    # Replace any surviving NaN/Inf
    X = np.where(np.isfinite(X), X, 0.0)
    return X, y, vs


# ─────────────────────────────────────────────────────────────────────────────
# Feature importance analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyze_features(X_train, y_train, out_dir, top_k=40):
    """
    Compute ANOVA F-scores, Mutual Information, and RF importance.
    Saves a combined ranking plot and returns sorted feature indices.
    """
    names = feature_names()
    F     = X_train.shape[1]

    print('  Computing ANOVA F-scores ...')
    f_scores, _ = f_classif(X_train, y_train)
    f_scores     = np.nan_to_num(f_scores)

    print('  Computing Mutual Information ...')
    mi_scores = mutual_info_classif(
        X_train, y_train, random_state=RANDOM_STATE, n_jobs=-1
    )

    print('  Training RF for feature importance ...')
    rf_imp = RandomForestClassifier(
        n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE
    )
    rf_imp.fit(X_train, y_train)
    imp = rf_imp.feature_importances_

    # Normalize each to [0, 1] then average for combined rank
    def norm01(a):
        r = a - a.min()
        return r / (r.max() + 1e-12)

    combined = (norm01(f_scores) + norm01(mi_scores) + norm01(imp)) / 3.0
    rank_idx  = np.argsort(combined)[::-1]   # descending

    # ── save full ranking as CSV ─────────────────────────────────────────
    df = pd.DataFrame({
        'feature'    : [names[i] for i in range(F)],
        'anova_f'    : f_scores,
        'mutual_info': mi_scores,
        'rf_importance': imp,
        'combined'   : combined,
    }).sort_values('combined', ascending=False)
    df.to_csv(out_dir / 'feature_ranking.csv', index=False)
    print(f'  Saved feature_ranking.csv  ({F} features)')

    # ── plot top-40 combined score ───────────────────────────────────────
    topN = min(top_k, F)
    top_idx   = rank_idx[:topN]
    top_names = [names[i] for i in top_idx]
    top_comb  = combined[top_idx]

    fig, axes = plt.subplots(3, 1, figsize=(16, 18))

    for ax, scores, label in zip(
        axes,
        [f_scores[top_idx], mi_scores[top_idx], imp[top_idx]],
        ['ANOVA F-score', 'Mutual Information', 'RF Importance']
    ):
        ax.bar(range(topN), scores, color='steelblue')
        ax.set_xticks(range(topN))
        ax.set_xticklabels(top_names, rotation=90, fontsize=7)
        ax.set_title(f'Top {topN} features — {label}')
        ax.set_ylabel(label)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_dir / 'feature_importance_v2.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  Saved feature_importance_v2.png')

    return rank_idx


# ─────────────────────────────────────────────────────────────────────────────
# Model training helpers
# ─────────────────────────────────────────────────────────────────────────────

def train_rf(X_tr, y_tr):
    clf = RandomForestClassifier(
        n_estimators=500, n_jobs=-1, random_state=RANDOM_STATE
    )
    clf.fit(X_tr, y_tr)
    return clf


def train_svm(X_tr, y_tr):
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('svm',    SVC(kernel='rbf', C=10, gamma='scale',
                       decision_function_shape='ovr', random_state=RANDOM_STATE))
    ])
    pipe.fit(X_tr, y_tr)
    return pipe


def evaluate(clf, X_te, y_te, label_names):
    preds = clf.predict(X_te)
    acc   = accuracy_score(y_te, preds)
    report = classification_report(
        y_te, preds,
        target_names=label_names,
        digits=4
    )
    cm = confusion_matrix(y_te, preds)
    return acc, report, cm, preds


def plot_cm(cm, label_names, title, save_path):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=label_names, yticklabels=label_names, ax=ax
    )
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_per_class_acc(cm, label_names, title, save_path):
    per_class = cm.diagonal() / cm.sum(axis=1)
    fig, ax   = plt.subplots(figsize=(10, 4))
    colors    = ['green' if a >= 0.95 else 'orange' if a >= 0.85 else 'red'
                 for a in per_class]
    ax.bar(label_names, per_class * 100, color=colors)
    ax.axhline(95, color='gray', linestyle='--', linewidth=0.8)
    ax.set_ylim(0, 105)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(title)
    ax.set_xticklabels(label_names, rotation=45)
    for i, v in enumerate(per_class):
        ax.text(i, v * 100 + 1, f'{v*100:.1f}%', ha='center', fontsize=8)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    label_names = [f'{LABEL_TO_POS[i]}m' for i in range(N_CLASSES)]
    results     = {}

    # ── 1. Build file list ───────────────────────────────────────────────
    print(f'\n[1] Scanning {args.data_dir} ...')
    samples = build_file_list(args.data_dir)
    print(f'    Found {len(samples)} files')

    # ── 2. Extract features ──────────────────────────────────────────────
    cache_path = out_dir / 'features_cache.npz'
    if args.use_cache and cache_path.exists():
        print('[2] Loading cached features ...')
        c = np.load(cache_path)
        X, y, volts = c['X'], c['y'], c['volts']
    else:
        print(f'[2] Extracting {feature_count()} features per file ...')
        t0 = time.time()
        X, y, volts = extract_all(samples)
        elapsed = time.time() - t0
        print(f'    Done in {elapsed:.1f}s  ({elapsed/len(samples)*1000:.1f}ms/file)')
        np.savez(cache_path, X=X, y=y, volts=volts)
        print(f'    Saved cache → {cache_path}')

    print(f'    Feature matrix: {X.shape}  (NaN: {np.isnan(X).sum()})')

    # ── 3. Random stratified split (70/15/15) ────────────────────────────
    print('\n[3] Splitting (random 70/15/15) ...')
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=RANDOM_STATE)
    tr_idx, tmp_idx = next(sss.split(X, y))
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=RANDOM_STATE)
    val_idx, te_idx = next(sss2.split(X[tmp_idx], y[tmp_idx]))
    val_idx = tmp_idx[val_idx]; te_idx = tmp_idx[te_idx]

    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_te, y_te = X[te_idx], y[te_idx]
    print(f'    Train: {len(tr_idx)}  Test: {len(te_idx)}')

    # ── 4. Feature importance analysis ──────────────────────────────────
    print('\n[4] Analysing features ...')
    rank_idx = analyze_features(X_tr, y_tr, out_dir, top_k=args.top_k)

    # ── 5. Train & evaluate — ALL features ──────────────────────────────
    print('\n[5] Training RF (all features) ...')
    rf_all = train_rf(X_tr, y_tr)
    acc_rf_all, rep_rf_all, cm_rf_all, _ = evaluate(rf_all, X_te, y_te, label_names)
    print(f'    RF  (all {feature_count()} feats): {acc_rf_all*100:.2f}%')
    results['RF_all_features'] = round(acc_rf_all * 100, 4)
    plot_cm(cm_rf_all, label_names,
            f'RF — All features ({acc_rf_all*100:.2f}%)',
            out_dir / 'cm_rf_all.png')
    plot_per_class_acc(cm_rf_all, label_names,
                       f'RF per-class — All features',
                       out_dir / 'per_class_rf_all.png')

    print('\n[5b] Training SVM (all features) ...')
    svm_all = train_svm(X_tr, y_tr)
    acc_svm_all, _, cm_svm_all, _ = evaluate(svm_all, X_te, y_te, label_names)
    print(f'    SVM (all {feature_count()} feats): {acc_svm_all*100:.2f}%')
    results['SVM_all_features'] = round(acc_svm_all * 100, 4)
    plot_cm(cm_svm_all, label_names,
            f'SVM — All features ({acc_svm_all*100:.2f}%)',
            out_dir / 'cm_svm_all.png')

    # ── 6. Train & evaluate — top-K features ────────────────────────────
    K = args.top_k
    top_idx = rank_idx[:K]
    X_tr_k  = X_tr[:, top_idx]
    X_te_k  = X_te[:, top_idx]

    print(f'\n[6] Training RF (top {K} features) ...')
    rf_k = train_rf(X_tr_k, y_tr)
    acc_rf_k, rep_rf_k, cm_rf_k, _ = evaluate(rf_k, X_te_k, y_te, label_names)
    print(f'    RF  (top {K} feats): {acc_rf_k*100:.2f}%')
    results[f'RF_top{K}'] = round(acc_rf_k * 100, 4)
    plot_cm(cm_rf_k, label_names,
            f'RF — Top {K} features ({acc_rf_k*100:.2f}%)',
            out_dir / f'cm_rf_top{K}.png')
    plot_per_class_acc(cm_rf_k, label_names,
                       f'RF per-class — Top {K} features',
                       out_dir / f'per_class_rf_top{K}.png')

    print(f'\n[6b] Training SVM (top {K} features) ...')
    svm_k = train_svm(X_tr_k, y_tr)
    acc_svm_k, _, cm_svm_k, _ = evaluate(svm_k, X_te_k, y_te, label_names)
    print(f'    SVM (top {K} feats): {acc_svm_k*100:.2f}%')
    results[f'SVM_top{K}'] = round(acc_svm_k * 100, 4)

    # ── 7. Voltage-invariant evaluation (RF top-K) ───────────────────────
    print('\n[7] Voltage-invariant evaluation (train 1-8V, test 9-10V) ...')
    tr_vi = np.where(volts <= 8)[0]
    te_vi = np.where(volts >= 9)[0]
    X_tr_vi = X[tr_vi][:, top_idx];  y_tr_vi = y[tr_vi]
    X_te_vi = X[te_vi][:, top_idx];  y_te_vi = y[te_vi]

    rf_vi = train_rf(X_tr_vi, y_tr_vi)
    acc_vi, _, cm_vi, _ = evaluate(rf_vi, X_te_vi, y_te_vi, label_names)
    print(f'    RF voltage-invariant (top {K}): {acc_vi*100:.2f}%')
    results[f'RF_top{K}_voltage_invariant'] = round(acc_vi * 100, 4)
    plot_cm(cm_vi, label_names,
            f'RF Voltage-Invariant — Top {K} ({acc_vi*100:.2f}%)',
            out_dir / 'cm_rf_voltage_invariant.png')
    plot_per_class_acc(cm_vi, label_names,
                       f'RF per-class Voltage-Invariant — Top {K}',
                       out_dir / 'per_class_rf_voltage_invariant.png')

    # ── 8. Summary bar chart ─────────────────────────────────────────────
    print('\n[8] Saving summary ...')
    # Previous baseline results (from your recap)
    prev = {
        'RF v1 (24 feat, random)': 99.80,
        'SVM v1 (24 feat, random)': 98.30,
        'RF v1 (volt-invariant)': 99.53,
    }
    all_results = {**prev, **{k: v for k, v in results.items()}}

    fig, ax = plt.subplots(figsize=(12, 5))
    names_r = list(all_results.keys())
    vals_r  = list(all_results.values())
    colors  = ['#aaaaaa'] * len(prev) + ['steelblue'] * len(results)
    bars    = ax.bar(names_r, vals_r, color=colors)
    ax.set_ylim(min(vals_r) - 2, 101)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Model Comparison — v1 baseline vs v2 extended features')
    ax.set_xticklabels(names_r, rotation=35, ha='right', fontsize=9)
    for bar, v in zip(bars, vals_r):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1, f'{v:.2f}%',
                ha='center', fontsize=8)
    plt.tight_layout()
    fig.savefig(out_dir / 'comparison_v2.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Save JSON
    with open(out_dir / 'results_v2.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary table
    print('\n' + '=' * 55)
    print('  RESULTS SUMMARY')
    print('=' * 55)
    for k, v in {**prev, **results}.items():
        marker = ' ◄ NEW' if k in results else ''
        print(f'  {k:<42s} {v:.2f}%{marker}')
    print('=' * 55)
    print(f'\nAll outputs saved to: {out_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',  default='data/raw/measurements',
                        help='Path to measurements root dir')
    parser.add_argument('--out_dir',   default='results/baseline_v2',
                        help='Output directory')
    parser.add_argument('--top_k',     type=int, default=40,
                        help='Number of top features to select')
    parser.add_argument('--use_cache', action='store_true',
                        help='Load cached features_cache.npz if it exists')
    args = parser.parse_args()
    main(args)
