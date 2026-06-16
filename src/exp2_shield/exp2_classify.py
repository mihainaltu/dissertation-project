"""
Experiment 2 — Classification
Usage:
    python3 exp2_classify.py --features exp2_features.csv

Runs RF, SVM, kNN with leave-one-file-out cross-validation.
Prints accuracy, confusion matrix, and per-class metrics.
"""

import pandas as pd
import numpy as np
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

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

MODELS = {
    'RF':  RandomForestClassifier(n_estimators=200, random_state=42),
    'SVM': SVC(kernel='rbf', C=10, gamma='scale', random_state=42),
    'kNN': KNeighborsClassifier(n_neighbors=5),
}


def lofo_cv(df, model_name, model):
    """Leave-one-file-out cross-validation."""
    files = df['file'].unique()
    y_true_all, y_pred_all = [], []

    for test_file in files:
        train_df = df[df['file'] != test_file]
        test_df  = df[df['file'] == test_file]

        X_train = train_df[FEATURE_COLS].values
        y_train = train_df['label'].values
        X_test  = test_df[FEATURE_COLS].values
        y_test  = test_df['label'].values

        # scale (fit on train only)
        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        clf = clone_model(model_name)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)

        y_true_all.extend(y_test)
        y_pred_all.extend(preds)

    return np.array(y_true_all), np.array(y_pred_all)


def clone_model(name):
    """Return a fresh unfitted model instance."""
    return {
        'RF':  RandomForestClassifier(n_estimators=200, random_state=42),
        'SVM': SVC(kernel='rbf', C=10, gamma='scale', random_state=42),
        'kNN': KNeighborsClassifier(n_neighbors=5),
    }[name]


def feature_importance(df):
    """Print RF feature importances trained on full dataset."""
    X = df[FEATURE_COLS].values
    y = df['label'].values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    rf = RandomForestClassifier(n_estimators=500, random_state=42)
    rf.fit(X, y)
    importances = sorted(zip(FEATURE_COLS, rf.feature_importances_), key=lambda x: -x[1])
    print("\nTop 15 RF feature importances (full dataset):")
    for feat, imp in importances[:15]:
        bar = '█' * int(imp * 200)
        print(f"  {feat:<35} {imp:.4f}  {bar}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', required=True, help='CSV from exp2_extract.py')
    args = parser.parse_args()

    df = pd.read_csv(args.features)
    print(f"Loaded {len(df)} pulse samples from {df['file'].nunique()} files")
    print(f"Label distribution:\n{df['label'].value_counts().to_string()}\n")

    # Check all feature columns present
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"WARNING: missing feature columns: {missing}")
        FEATURE_COLS[:] = [c for c in FEATURE_COLS if c in df.columns]

    for name in MODELS:
        print(f"{'='*50}")
        print(f"Model: {name}  (leave-one-file-out CV)")
        print(f"{'='*50}")
        y_true, y_pred = lofo_cv(df, name, MODELS[name])
        acc = accuracy_score(y_true, y_pred)
        print(f"Accuracy: {acc*100:.2f}%\n")
        print(classification_report(y_true, y_pred, digits=3))
        cm = confusion_matrix(y_true, y_pred, labels=['DAMAGED', 'HEALTHY'])
        print("Confusion matrix (rows=true, cols=pred):")
        print(f"               DAMAGED  HEALTHY")
        print(f"  DAMAGED       {cm[0,0]:5d}    {cm[0,1]:5d}")
        print(f"  HEALTHY       {cm[1,0]:5d}    {cm[1,1]:5d}")
        print()

    feature_importance(df)


if __name__ == '__main__':
    main()
