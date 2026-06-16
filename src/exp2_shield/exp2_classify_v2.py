"""
Experiment 2 — Extended Classification
Usage:
    python3 exp2_classify_v2.py --features exp2_features.csv

Runs:
  1. RF / SVM / kNN / XGBoost with freq_mhz as extra feature
  2. RF with top-N feature selection
  3. Per-frequency accuracy breakdown for best model
"""

import pandas as pd
import numpy as np
import argparse
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_selection import SelectFromModel

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("XGBoost not installed — install with: pip install xgboost")
    print("Falling back to GradientBoostingClassifier\n")

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
]

FEATURE_COLS_WITH_FREQ = FEATURE_COLS + ['freq_mhz']


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
    """Leave-one-file-out cross-validation. Returns (y_true, y_pred, freq_per_sample)."""
    files = df['file'].unique()
    y_true_all, y_pred_all, freq_all = [], [], []

    # encode labels to 0/1 for XGBoost compatibility
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
        # decode back to string labels if needed
        try:
            preds = np.array([inv_map[p] for p in preds_enc])
        except KeyError:
            preds = preds_enc  # already string labels (RF/SVM/kNN)

        y_true_all.extend(y_test)
        y_pred_all.extend(preds)
        freq_all.extend(freq)

    return np.array(y_true_all), np.array(y_pred_all), np.array(freq_all)


def print_results(name, y_true, y_pred, note=''):
    acc = accuracy_score(y_true, y_pred)
    print(f"{'='*52}")
    print(f"Model: {name}  {note}")
    print(f"{'='*52}")
    print(f"Accuracy: {acc*100:.2f}%\n")
    print(classification_report(y_true, y_pred, digits=3))
    cm = confusion_matrix(y_true, y_pred, labels=['DAMAGED', 'HEALTHY'])
    print("Confusion matrix (rows=true, cols=pred):")
    print(f"               DAMAGED  HEALTHY")
    print(f"  DAMAGED       {cm[0,0]:5d}    {cm[0,1]:5d}")
    print(f"  HEALTHY       {cm[1,0]:5d}    {cm[1,1]:5d}")
    print()
    return acc


def select_top_features(df, feature_cols, top_n=20):
    """Use RF importance to select top N features."""
    X = df[feature_cols].values
    y = df['label'].values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    rf = RandomForestClassifier(n_estimators=500, random_state=42)
    rf.fit(X, y)
    importances = list(zip(feature_cols, rf.feature_importances_))
    importances.sort(key=lambda x: -x[1])
    top = [f for f, _ in importances[:top_n]]
    print(f"Top {top_n} features selected:")
    for f, imp in importances[:top_n]:
        bar = '█' * int(imp * 300)
        print(f"  {f:<35} {imp:.4f}  {bar}")
    print()
    return top


def per_frequency_breakdown(df, model_name, feature_cols):
    """Show accuracy per injection frequency."""
    y_true, y_pred, freqs = lofo_cv(df, model_name, feature_cols)
    print(f"\nPer-frequency accuracy ({model_name}, LOFO-CV):")
    print(f"  {'Freq (MHz)':<12} {'N':>5} {'Accuracy':>10}")
    print(f"  {'-'*30}")
    results = []
    for freq in sorted(df['freq_mhz'].unique()):
        mask = freqs == freq
        if mask.sum() == 0:
            continue
        acc = accuracy_score(y_true[mask], y_pred[mask])
        n   = mask.sum()
        bar = '█' * int(acc * 20)
        print(f"  {freq:<12.0f} {n:>5} {acc*100:>9.1f}%  {bar}")
        results.append((freq, acc, n))
    overall = accuracy_score(y_true, y_pred)
    print(f"  {'OVERALL':<12} {len(y_true):>5} {overall*100:>9.1f}%")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', required=True)
    parser.add_argument('--top_n', type=int, default=20,
                        help='Number of top features to keep in feature-selection run')
    args = parser.parse_args()

    df = pd.read_csv(args.features)
    print(f"Loaded {len(df)} pulse samples from {df['file'].nunique()} files")
    print(f"Label distribution:\n{df['label'].value_counts().to_string()}\n")

    # ── 1. All models WITH freq_mhz as feature ────────────────────────────
    print("\n" + "="*52)
    print("PART 1 — All models (with freq_mhz as feature)")
    print("="*52 + "\n")
    best_acc, best_model = 0, None
    for name in get_models():
        y_true, y_pred, _ = lofo_cv(df, name, FEATURE_COLS_WITH_FREQ)
        acc = print_results(name, y_true, y_pred, note='(with freq_mhz)')
        if acc > best_acc:
            best_acc, best_model = acc, name

    # ── 2. RF with top-N feature selection ────────────────────────────────
    print("\n" + "="*52)
    print(f"PART 2 — RF with top-{args.top_n} feature selection")
    print("="*52 + "\n")
    top_features = select_top_features(df, FEATURE_COLS_WITH_FREQ, top_n=args.top_n)
    y_true, y_pred, _ = lofo_cv(df, 'RF', top_features)
    print_results('RF', y_true, y_pred, note=f'(top {args.top_n} features)')

    # ── 3. Per-frequency breakdown for best model ─────────────────────────
    print("\n" + "="*52)
    print(f"PART 3 — Per-frequency breakdown (best model: {best_model})")
    print("="*52)
    per_frequency_breakdown(df, best_model, FEATURE_COLS_WITH_FREQ)


if __name__ == '__main__':
    main()
