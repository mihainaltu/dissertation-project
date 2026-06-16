# src/baseline_exp2.py

import sys, os
sys.path.append(os.path.dirname(__file__))

import numpy as np
from pathlib import Path
from tqdm import tqdm
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns

from dataset_exp2 import build_file_list_exp2
from features_exp2 import load_exp2_crop, extract_features_exp2, feature_names_exp2


# ── Build feature matrix ───────────────────────────────────────────────────────

def build_feature_matrix(samples, desc="Extracting"):
    X, y = [], []
    for filepath, label in tqdm(samples, desc=desc):
        crop = load_exp2_crop(filepath)
        X.append(extract_features_exp2(crop))
        y.append(label)
    return np.array(X), np.array(y)


# ── Confusion matrix ───────────────────────────────────────────────────────────

def plot_cm(y_true, y_pred, title, save_path):
    cm     = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=['Damaged', 'Healthy'],
                yticklabels=['Damaged', 'Healthy'], ax=ax)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    ROOT    = 'data/raw/sames-cable-data'
    RESULTS = Path('results/exp2')
    RESULTS.mkdir(parents=True, exist_ok=True)

    all_samples = build_file_list_exp2(ROOT)
    labels      = [s[1] for s in all_samples]

    # Split — no augmentation for classical ML
    train_val, test = train_test_split(all_samples, test_size=0.15,
                                       stratify=labels, random_state=42)
    labels_tv       = [s[1] for s in train_val]
    train, val      = train_test_split(train_val, test_size=0.15/0.85,
                                       stratify=labels_tv, random_state=42)

    X_train, y_train = build_feature_matrix(train, "Train")
    X_val,   y_val   = build_feature_matrix(val,   "Val  ")
    X_test,  y_test  = build_feature_matrix(test,  "Test ")

    # Combine train+val for classical ML (no need to hold val separately)
    X_tv = np.concatenate([X_train, X_val])
    y_tv = np.concatenate([y_train, y_val])

    scaler = StandardScaler()
    X_tv_s = scaler.fit_transform(X_tv)
    X_test_s = scaler.transform(X_test)

    print(f"\nFeature vector size: {X_train.shape[1]}")
    print(f"Train+Val: {len(X_tv)} | Test: {len(X_test)}")

    # ── Random Forest ──────────────────────────────────────────────────────────
    print("\n[1/2] Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42)
    rf.fit(X_tv_s, y_tv)
    rf_acc = accuracy_score(y_test, rf.predict(X_test_s))
    print(f"  Test accuracy: {rf_acc*100:.2f}%")
    print(classification_report(y_test, rf.predict(X_test_s),
                                 target_names=['Damaged', 'Healthy']))
    plot_cm(y_test, rf.predict(X_test_s),
            f'Random Forest — {rf_acc*100:.1f}%',
            RESULTS / 'cm_rf_exp2.png')

    # Feature importances
    fi  = rf.feature_importances_
    fn  = feature_names_exp2()
    idx = np.argsort(fi)[::-1]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(fn)), fi[idx])
    ax.set_xticks(range(len(fn)))
    ax.set_xticklabels([fn[i] for i in idx], rotation=45, ha='right')
    ax.set_title('Feature Importances — Random Forest (Exp2)')
    plt.tight_layout()
    plt.savefig(RESULTS / 'rf_feature_importance_exp2.png', dpi=150)
    plt.close()

    # ── SVM ────────────────────────────────────────────────────────────────────
    print("\n[2/2] Training SVM...")
    svm = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
    svm.fit(X_tv_s, y_tv)
    svm_acc = accuracy_score(y_test, svm.predict(X_test_s))
    print(f"  Test accuracy: {svm_acc*100:.2f}%")
    print(classification_report(y_test, svm.predict(X_test_s),
                                 target_names=['Damaged', 'Healthy']))
    plot_cm(y_test, svm.predict(X_test_s),
            f'SVM (RBF) — {svm_acc*100:.1f}%',
            RESULTS / 'cm_svm_exp2.png')

    print(f"\n── Summary ──────────────────────────")
    print(f"  Random Forest : {rf_acc*100:.2f}%")
    print(f"  SVM (RBF)     : {svm_acc*100:.2f}%")