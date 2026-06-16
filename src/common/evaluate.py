import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / "exp1_localization"))

# src/evaluate.py

import sys, os
sys.path.append(os.path.dirname(__file__))

import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix,
                              ConfusionMatrixDisplay)
from pathlib import Path

from exp1_dataset import POSITIONS

RESULTS = Path('results')
class_names = [f"{p}m" for p in POSITIONS]


# ── 1. Load data ──────────────────────────────────────────────────────────────

preds  = np.load(RESULTS / 'cnn_preds.npy')
labels = np.load(RESULTS / 'cnn_labels.npy')

with open(RESULTS / 'cnn_history.json') as f:
    history = json.load(f)

epochs = range(1, len(history['train_loss']) + 1)


# ── 2. Training curves ────────────────────────────────────────────────────────

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('1D-CNN Training History', fontsize=13, fontweight='bold')

ax1.plot(epochs, history['train_loss'], label='Train', linewidth=2)
ax1.plot(epochs, history['val_loss'],   label='Val',   linewidth=2)
ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
ax1.set_title('Cross-Entropy Loss')
ax1.legend(); ax1.grid(True, alpha=0.3)

ax2.plot(epochs, [x*100 for x in history['train_acc']], label='Train', linewidth=2)
ax2.plot(epochs, [x*100 for x in history['val_acc']],   label='Val',   linewidth=2)
ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Accuracy')
ax2.legend(); ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(RESULTS / 'cnn_training_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: cnn_training_curves.png")


# ── 3. Confusion matrix ───────────────────────────────────────────────────────

cm = confusion_matrix(labels, preds)
cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names, ax=ax)
ax.set_xlabel('Predicted'); ax.set_ylabel('True')
test_acc = (preds == labels).mean()
ax.set_title(f'1D-CNN — Test Accuracy {test_acc*100:.2f}%')
plt.tight_layout()
plt.savefig(RESULTS / 'cm_cnn.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: cm_cnn.png")


# ── 4. Per-class accuracy bar chart ──────────────────────────────────────────

per_class_acc = cm.diagonal() / cm.sum(axis=1) * 100
colors = ['#2ecc71' if a >= 99 else '#f39c12' if a >= 95 else '#e74c3c'
          for a in per_class_acc]

fig, ax = plt.subplots(figsize=(12, 5))
bars = ax.bar(class_names, per_class_acc, color=colors, edgecolor='white', linewidth=0.5)
ax.axhline(99, color='green',  linestyle='--', linewidth=1, label='99% threshold')
ax.axhline(95, color='orange', linestyle='--', linewidth=1, label='95% threshold')
ax.set_ylim([80, 101])
ax.set_xlabel('Injection Position')
ax.set_ylabel('Accuracy (%)')
ax.set_title('1D-CNN — Per-Class Test Accuracy')
ax.legend()
ax.grid(True, axis='y', alpha=0.3)
for bar, acc in zip(bars, per_class_acc):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            f'{acc:.1f}%', ha='center', va='bottom', fontsize=8)
plt.tight_layout()
plt.savefig(RESULTS / 'cnn_per_class_accuracy.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: cnn_per_class_accuracy.png")


# ── 5. Model comparison bar chart ─────────────────────────────────────────────

models      = ['SVM (RBF)', 'Random Forest', '1D-CNN\n(raw waveform)']
accuracies  = [98.3, 99.8, 99.15]
bar_colors  = ['#3498db', '#2ecc71', '#e74c3c']

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(models, accuracies, color=bar_colors, width=0.5,
              edgecolor='white', linewidth=0.5)
ax.set_ylim([96, 100.5])
ax.set_ylabel('Test Accuracy (%)')
ax.set_title('Model Comparison — PD Localization')
ax.grid(True, axis='y', alpha=0.3)
for bar, acc in zip(bars, accuracies):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.savefig(RESULTS / 'model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: model_comparison.png")


# ── 6. Classification report ──────────────────────────────────────────────────

print("\n── Classification Report ─────────────────────────────")
print(classification_report(labels, preds, target_names=class_names))
