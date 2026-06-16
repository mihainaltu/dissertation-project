# src/analyze_voltage_invariant.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from pathlib import Path

POSITIONS   = [100,200,300,500,700,900,1000,1300,1500,1600,1800,1900]
class_names = [f"{p}m" for p in POSITIONS]

preds  = np.load('results/voltage_invariant/preds.npy')
labels = np.load('results/voltage_invariant/labels.npy')

cm        = confusion_matrix(labels, preds)
per_class = cm.diagonal() / cm.sum(axis=1) * 100

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle('Voltage-Invariant Test (train 1–8V, test 9–10V)',
             fontsize=13, fontweight='bold')

# Per-class bar
colors = ['#2ecc71' if a >= 99 else '#f39c12' if a >= 95 else '#e74c3c'
          for a in per_class]
axes[0].bar(class_names, per_class, color=colors, edgecolor='white')
axes[0].axhline(99, color='green',  linestyle='--', linewidth=1, label='99%')
axes[0].axhline(95, color='orange', linestyle='--', linewidth=1, label='95%')
axes[0].set_ylim([50, 101])
axes[0].set_xlabel('Position')
axes[0].set_ylabel('Accuracy (%)')
axes[0].set_title('1D-CNN Per-Class Accuracy (Voltage-Invariant)')
axes[0].legend()
axes[0].grid(True, axis='y', alpha=0.3)
for bar, acc in zip(axes[0].patches, per_class):
    axes[0].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.3,
                 f'{acc:.1f}%', ha='center', va='bottom', fontsize=7)

# Confusion matrix
cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names, ax=axes[1])
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('True')
axes[1].set_title('Confusion Matrix — Voltage-Invariant CNN')

plt.tight_layout()
Path('results/voltage_invariant').mkdir(parents=True, exist_ok=True)
plt.savefig('results/voltage_invariant/analysis.png', dpi=150, bbox_inches='tight')
plt.show()

print("Per-class accuracy (Voltage-Invariant CNN):")
for pos, acc in zip(POSITIONS, per_class):
    marker = ' ← struggling' if acc < 95 else ''
    print(f"  {pos:4d}m : {acc:.1f}%{marker}")
print(f"\nOverall: {(preds==labels).mean()*100:.2f}%")
