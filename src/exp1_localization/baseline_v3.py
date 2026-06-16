"""
baseline_v3.py — Extended classical ML comparison for Experiment 1.

Adds to the existing RF + SVM baseline:
  - k-NN (k=5 and k=11)
  - Gradient Boosting (XGBoost if available, else sklearn GBT)
  - LDA classifier
  - MLP (multi-layer perceptron on features)

All models use the same top-40 feature vector from features_v2.py.
Loads the cached feature matrix from baseline_v2 — no re-extraction needed.

Usage:
    python src/baseline_v3.py
    python src/baseline_v3.py --exp1_cache results/baseline_v2/features_cache.npz
"""

import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_selection import f_classif, mutual_info_classif

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print('[INFO] XGBoost not installed — using sklearn GradientBoostingClassifier instead.')
    print('       To install: pip install xgboost')

RANDOM_STATE = 42
N_CLASSES    = 12
LABEL_TO_POS = {
    0:100, 1:200, 2:300,  3:500,  4:700,  5:900,
    6:1000,7:1300,8:1500, 9:1600,10:1800,11:1900
}
LABEL_NAMES  = [f'{LABEL_TO_POS[i]}m' for i in range(N_CLASSES)]


# ─────────────────────────────────────────────────────────────────────────────
# Data loading and feature selection
# ─────────────────────────────────────────────────────────────────────────────

def load_cache(cache_path):
    c = np.load(cache_path)
    X, y, volts = c['X'], c['y'], c['volts']
    X = np.where(np.isfinite(X), X, 0.0)
    print(f'Loaded: X={X.shape}  classes={np.unique(y)}')
    return X, y, volts


def select_top_k(X_tr, y_tr, X_te, k=40):
    """Combined ANOVA + MI + RF ranking, return top-k indices and reduced sets."""
    from features_v2 import feature_names
    names = feature_names()
    F     = X_tr.shape[1]

    f_scores, _ = f_classif(X_tr, y_tr)
    f_scores     = np.nan_to_num(f_scores)
    mi           = mutual_info_classif(X_tr, y_tr,
                                       random_state=RANDOM_STATE, n_jobs=-1)
    rf_imp       = RandomForestClassifier(
        n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE
    ).fit(X_tr, y_tr).feature_importances_

    def norm(a):
        r = a - a.min()
        return r / (r.max() + 1e-12)

    combined  = (norm(f_scores) + norm(mi) + norm(rf_imp)) / 3.0
    rank_idx  = np.argsort(combined)[::-1][:k]
    return X_tr[:, rank_idx], X_te[:, rank_idx], rank_idx


def stratified_split(X, y):
    sss = StratifiedShuffleSplit(1, test_size=0.30, random_state=RANDOM_STATE)
    tr, tmp = next(sss.split(X, y))
    sss2 = StratifiedShuffleSplit(1, test_size=0.50, random_state=RANDOM_STATE)
    va, te = next(sss2.split(X[tmp], y[tmp]))
    return tr, tmp[va], tmp[te]


# ─────────────────────────────────────────────────────────────────────────────
# Model definitions
# ─────────────────────────────────────────────────────────────────────────────

def get_models():
    models = {}

    # k-NN variants
    models['kNN_k5']  = Pipeline([
        ('scaler', StandardScaler()),
        ('clf',    KNeighborsClassifier(n_neighbors=5, n_jobs=-1,
                                        metric='euclidean'))
    ])
    models['kNN_k11'] = Pipeline([
        ('scaler', StandardScaler()),
        ('clf',    KNeighborsClassifier(n_neighbors=11, n_jobs=-1,
                                        metric='euclidean'))
    ])

    # LDA classifier (also acts as dimensionality reduction)
    models['LDA'] = Pipeline([
        ('scaler', StandardScaler()),
        ('clf',    LinearDiscriminantAnalysis(solver='svd'))
    ])

    # Gradient boosting
    if HAS_XGB:
        models['XGBoost'] = Pipeline([
            ('scaler', StandardScaler()),
            ('clf',    XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                use_label_encoder=False,
                eval_metric='mlogloss',
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbosity=0,
            ))
        ])
    else:
        models['GradientBoosting'] = Pipeline([
            ('scaler', StandardScaler()),
            ('clf',    GradientBoostingClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                random_state=RANDOM_STATE,
            ))
        ])

    # MLP
    models['MLP'] = Pipeline([
        ('scaler', StandardScaler()),
        ('clf',    MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation='relu',
            solver='adam',
            learning_rate_init=1e-3,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=RANDOM_STATE,
            verbose=False,
        ))
    ])

    # RF and SVM for direct comparison (already in baseline_v2 but repeat here)
    models['RandomForest'] = RandomForestClassifier(
        n_estimators=500, n_jobs=-1, random_state=RANDOM_STATE
    )
    models['SVM_RBF'] = Pipeline([
        ('scaler', StandardScaler()),
        ('clf',    SVC(kernel='rbf', C=10, gamma='scale',
                       decision_function_shape='ovr',
                       random_state=RANDOM_STATE))
    ])

    return models


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(clf, X_te, y_te, name, out_dir):
    preds = clf.predict(X_te)
    acc   = accuracy_score(y_te, preds)
    cm    = confusion_matrix(y_te, preds)
    print(f'  {name:<22s}: {acc*100:.2f}%')

    # confusion matrix
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, ax=ax)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(f'{name} ({acc*100:.2f}%)')
    plt.tight_layout()
    safe = name.replace(' ', '_').replace('/', '_')
    fig.savefig(out_dir / f'cm_{safe}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    return acc, preds


def plot_comparison(results, prev_results, out_dir):
    all_res = {**prev_results, **results}
    names   = list(all_res.keys())
    vals    = list(all_res.values())
    colors  = ['#aaaaaa'] * len(prev_results) + ['#2b6cb0'] * len(results)

    fig, ax = plt.subplots(figsize=(14, 5))
    bars    = ax.bar(names, vals, color=colors)
    ax.set_ylim(min(vals) - 2, 101)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Model comparison — all classifiers, top-40 features')
    ax.set_xticklabels(names, rotation=40, ha='right', fontsize=9)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.1, f'{v:.2f}%',
                ha='center', fontsize=8)
    plt.tight_layout()
    fig.savefig(out_dir / 'comparison_all_models.png', dpi=150,
                bbox_inches='tight')
    plt.close(fig)
    print('Saved comparison_all_models.png')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print('[1] Loading features...')
    X, y, volts = load_cache(args.exp1_cache)

    print('[2] Splitting...')
    tr_idx, va_idx, te_idx = stratified_split(X, y)
    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_te, y_te = X[te_idx], y[te_idx]

    print('[3] Feature selection (top-40)...')
    X_tr_k, X_te_k, _ = select_top_k(X_tr, y_tr, X_te, k=args.top_k)
    print(f'    Input shape: {X_tr_k.shape}')

    print('\n[4] Training and evaluating all models...')
    models  = get_models()
    results = {}
    import time
    for name, clf in models.items():
        t0 = time.time()
        clf.fit(X_tr_k, y_tr)
        elapsed = time.time() - t0
        acc, _ = evaluate(clf, X_te_k, y_te, name, out_dir)
        results[name] = round(acc * 100, 4)
        print(f'    Training time: {elapsed:.1f}s')

    # Voltage-invariant evaluation for all models
    print('\n[5] Voltage-invariant evaluation...')
    tr_vi = np.where(volts[tr_idx] <= 8)[0]
    te_vi = np.where(volts <= 8)[0]   # build from full dataset
    tr_vi_full = np.where(volts <= 8)[0]
    te_vi_full = np.where(volts >= 9)[0]
    X_tr_vi, _ , _ = select_top_k(X[tr_vi_full], y[tr_vi_full],
                                   X[te_vi_full], k=args.top_k)
    _, X_te_vi, _  = select_top_k(X[tr_vi_full], y[tr_vi_full],
                                   X[te_vi_full], k=args.top_k)

    vi_results = {}
    for name in ['RandomForest', 'kNN_k5', 'LDA',
                 'XGBoost' if HAS_XGB else 'GradientBoosting', 'MLP', 'SVM_RBF']:
        if name not in models:
            continue
        clf = models[name]
        clf.fit(X_tr_vi, y[tr_vi_full])
        preds = clf.predict(X_te_vi)
        acc   = accuracy_score(y[te_vi_full], preds)
        vi_results[f'{name}_vi'] = round(acc * 100, 4)
        print(f'  {name:<22s} volt-invariant: {acc*100:.2f}%')

    # Previous baselines for comparison chart
    prev = {
        'RF v1 random':      99.80,
        'SVM v1 random':     98.30,
        'RF v1 volt-inv':    99.53,
        'CNN v1 random':     99.15,
        'CNN v1 volt-inv':   94.48,
    }

    print('\n' + '='*55)
    print('  RESULTS SUMMARY')
    print('='*55)
    for k, v in {**prev, **results}.items():
        marker = ' ◄' if k in results else ''
        print(f'  {k:<28s} {v:.2f}%{marker}')
    print('\n  Voltage-invariant:')
    for k, v in vi_results.items():
        print(f'  {k:<28s} {v:.2f}%')
    print('='*55)

    plot_comparison(results, prev, out_dir)

    all_results = {**results, **vi_results}
    json.dump(all_results, open(out_dir / 'results_v3.json', 'w'), indent=2)
    print(f'\nAll outputs saved to: {out_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp1_cache', default='results/baseline_v2/features_cache.npz')
    parser.add_argument('--out_dir',    default='results/baseline_v3')
    parser.add_argument('--top_k',      type=int, default=40)
    args = parser.parse_args()
    main(args)