# src/baseline.py

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
import matplotlib.pyplot as plt
import seaborn as sns

from exp1_dataset import build_file_list, load_mat_file, POSITIONS
from features import extract_features, feature_names, CROP


# ── Build feature matrix ───────────────────────────────────────────────────────

def build_feature_matrix(samples, desc="Extracting features"):
    X, y = [], []
    for filepath, label in tqdm(samples, desc=desc):
        sig = load_mat_file(filepath)
        sig = sig[:, CROP[0]:CROP[1]]          # apply crop
        X.append(extract_features(sig))
        y.append(label)
    return np.array(X), np.array(y)


# ── Confusion matrix plot ──────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, class_names, title, save_path):
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    ROOT       = 'data/raw/measurements'
    RESULTS    = Path('results')
    RESULTS.mkdir(exist_ok=True)

    from exp1_dataset import build_file_list, POSITIONS
    from sklearn.model_selection import train_test_split

    class_names = [f"{p}m" for p in POSITIONS]

    # Build file list & split
    all_samples = build_file_list(ROOT)
    labels      = [s[1] for s in all_samples]

    train_val, test = train_test_split(all_samples, test_size=0.15,
                                       stratify=labels, random_state=42)
    labels_tv       = [s[1] for s in train_val]
    train, val      = train_test_split(train_val, test_size=0.15/0.85,
                                       stratify=labels_tv, random_state=42)

    print(f"\nExtracting features for {len(train)} train samples...")
    X_train, y_train = build_feature_matrix(train, "Train")

    print(f"Extracting features for {len(val)} val samples...")
    X_val, y_val     = build_feature_matrix(val,   "Val")

    print(f"Extracting features for {len(test)} test samples...")
    X_test, y_test   = build_feature_matrix(test,  "Test")

    # Scale
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    # ── Random Forest ──────────────────────────────────────────────────────────
    print("\n[1/2] Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)

    rf_val_acc  = accuracy_score(y_val,  rf.predict(X_val))
    rf_test_acc = accuracy_score(y_test, rf.predict(X_test))
    print(f"  Val  accuracy: {rf_val_acc*100:.2f}%")
    print(f"  Test accuracy: {rf_test_acc*100:.2f}%")
    print(classification_report(y_test, rf.predict(X_test),
                                 target_names=class_names))
    plot_confusion_matrix(y_test, rf.predict(X_test), class_names,
                          f'Random Forest — Test Accuracy {rf_test_acc*100:.1f}%',
                          RESULTS / 'cm_random_forest.png')

    # Feature importances
    fi   = rf.feature_importances_
    fn   = feature_names()
    idx  = np.argsort(fi)[::-1][:15]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(15), fi[idx])
    ax.set_xticks(range(15))
    ax.set_xticklabels([fn[i] for i in idx], rotation=45, ha='right')
    ax.set_title('Top 15 Feature Importances — Random Forest')
    plt.tight_layout()
    plt.savefig(RESULTS / 'rf_feature_importance.png', dpi=150)
    plt.close()

    # ── SVM ────────────────────────────────────────────────────────────────────
    print("\n[2/2] Training SVM (RBF kernel)...")
    svm = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
    svm.fit(X_train, y_train)

    svm_val_acc  = accuracy_score(y_val,  svm.predict(X_val))
    svm_test_acc = accuracy_score(y_test, svm.predict(X_test))
    print(f"  Val  accuracy: {svm_val_acc*100:.2f}%")
    print(f"  Test accuracy: {svm_test_acc*100:.2f}%")
    print(classification_report(y_test, svm.predict(X_test),
                                 target_names=class_names))
    plot_confusion_matrix(y_test, svm.predict(X_test), class_names,
                          f'SVM (RBF) — Test Accuracy {svm_test_acc*100:.1f}%',
                          RESULTS / 'cm_svm.png')

    # Save models
    joblib.dump(rf,     RESULTS / 'rf_model.pkl')
    joblib.dump(svm,    RESULTS / 'svm_model.pkl')
    joblib.dump(scaler, RESULTS / 'scaler.pkl')
    print("\nModels saved to results/")

    print("\n── Summary ──────────────────────────────")
    print(f"  Random Forest : {rf_test_acc*100:.2f}% test accuracy")
    print(f"  SVM (RBF)     : {svm_test_acc*100:.2f}% test accuracy")
