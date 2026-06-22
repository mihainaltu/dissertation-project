# Dissertation Project

**Deep Learning and Machine Learning Methods for High-Level Interpretation of Complex Sensor Measurements**

ML/DL pipeline for classification, localisation, and change detection of transient electrical signals in power cable systems, using data acquired with TiePie HS5 hardware.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Data Layout](#data-layout)
- [Installation](#installation)
- [Experiment 1 — PD Source Localisation](#experiment-1--pd-source-localisation)
- [Experiment 2 — Cable Shield Integrity](#experiment-2--cable-shield-integrity)
- [Figure Scripts](#figure-scripts)
- [Requirements Notes](#requirements-notes)

---

## Project Structure

```
dissertation-project/
├── src/
│   ├── common/
│   │   └── evaluate.py                  # Shared evaluation utilities
│   ├── exp1_localization/               # Experiment 1: PD source localisation
│   │   ├── exp1_dataset.py              # Dataset loader (12-class)
│   │   ├── features.py / features_v2.py # Handcrafted feature extraction
│   │   ├── model.py                     # 1D-CNN architecture
│   │   ├── model_multitask.py           # Multi-task CNN (position + voltage)
│   │   ├── baseline.py                  # RF + SVM (v1)
│   │   ├── baseline_v2.py               # RF + SVM with feature selection (v2)
│   │   ├── baseline_v3.py               # XGBoost + further ablations
│   │   ├── train.py                     # CNN training (12 classes)
│   │   ├── train_voltage_invariant.py   # Voltage-invariant train/test split
│   │   ├── train_multitask.py           # Multi-task CNN
│   │   ├── train_regression.py          # Open-set: detection + regression
│   │   ├── train_13class.py             # 12 PD classes + NonPD rejection
│   │   ├── noise_robustness_13class.py  # Noise robustness evaluation
│   │   ├── analyze_voltage_invariant.py # Voltage-invariant result plots
│   │   ├── noise_dataset.py             # Noise augmentation dataset
│   │   ├── nonpd_dataset.py             # NonPD (Rauscher) dataset
│   │   └── multitask_dataset.py         # Multi-task dataset
│   ├── exp2_shield/                     # Experiment 2: cable shield integrity
│   │   ├── exp2_dataset.py              # Dataset loader (damaged / healthy)
│   │   ├── features_exp2.py             # Feature extraction for Exp 2
│   │   ├── model_exp2.py                # CNN for Exp 2
│   │   ├── exp2_extract.py              # Extract pulse features → CSV
│   │   ├── exp2_classify.py             # RF / SVM / kNN (leave-one-file-out)
│   │   ├── exp2_classify_v2.py          # Improved classifiers (v2)
│   │   ├── exp2_classify_v3.py          # Further improvements (v3)
│   │   ├── baseline_exp2.py             # Baseline feature-based classifiers
│   │   ├── train_exp2.py                # CNN training for Exp 2
│   │   ├── transfer_learning.py         # Transfer Exp1 CNN → Exp2
│   │   └── exp2_factory.py              # Factory / real-cable change detection
│   └── figures/                         # Thesis figure generation
│       ├── pca_analysis.py
│       ├── pca_analysis_v2.py
│       ├── plot_all_positions.py
│       ├── plot_pulse_representations.py
│       └── plot_two_channel_waveform.py
├── data/                                # ← NOT in Git (see Data Layout below)
├── artifacts/                           # ← NOT in Git (models, predictions, logs)
├── requirements.txt                     # Pinned dependencies (Windows, CPU PyTorch)
├── requirements-core.txt                # Minimal cross-platform dependencies
└── .gitignore
```

---

## Data Layout

Scripts expect data at specific paths **relative to the repo root**.
Place your datasets like this:

```
data/
└── raw/
    ├── measurements/          # Experiment 1 — 32 k+ .mat files
    │   ├── 100/               # Position folder (distance in metres)
    │   │   ├── 10kV/          # Voltage-level subfolder
    │   │   │   ├── file1.mat
    │   │   │   └── ...
    │   │   └── 20kV/
    │   ├── 200/
    │   └── ...                # Positions: 100 200 300 500 700 900
    │                          #            1000 1300 1500 1600 1800 1900
    ├── nonpd/                 # Rauscher NonPD signals (.mat)
    ├── sames-cable-data/      # Experiment 2 — lab cable data
    │   ├── 1/                 # DAMAGED  (measurement 1)
    │   ├── 1-conn/            # HEALTHY  (measurement 1, connected)
    │   ├── 2/
    │   ├── 2-conn/
    │   └── ...                # Pairs up to 20 / 20-conn
    └── sames-real/            # Experiment 2 — factory real-cable data (optional)
```

> All `.mat` files are excluded from Git via `.gitignore`.

---

## Installation

**All commands below must be run from the repo root.**

```bash
# 1. Clone the repo
git clone https://github.com/mihainaltu/dissertation-project.git
cd dissertation-project

# 2. Create and activate a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

# 3. Install dependencies
#    Cross-platform (recommended for Linux/macOS):
pip install -r requirements-core.txt
#    Or exact original environment (Windows, CPU PyTorch):
pip install -r requirements.txt
```

### GPU / CUDA

`requirements.txt` installs the CPU-only PyTorch build.
For GPU support install the matching build from https://pytorch.org/get-started/locally/, e.g.:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Then set `'device': 'cuda'` in the `CFG` dict of any training script (it auto-detects by default).

---

## Experiment 1 — PD Source Localisation

> All scripts must be run **from the repo root**, e.g. `python src/exp1_localization/train.py`.

### 1a. Feature-based baselines (RF + SVM)

```bash
# RF + SVM, all features
python src/exp1_localization/baseline.py

# RF + SVM with feature selection (recommended)
python src/exp1_localization/baseline_v2.py \
    --data_dir data/raw/measurements \
    --out_dir  artifacts/baseline_v2 \
    --top_k    40

# XGBoost + extended ablations
python src/exp1_localization/baseline_v3.py \
    --data_dir data/raw/measurements \
    --out_dir  artifacts/baseline_v3
```

**To adapt `baseline.py`:** change `ROOT` and `RESULTS` near the top of the file.

### 1b. CNN training (12-class localisation)

```bash
python src/exp1_localization/train.py
```

Edit the `CFG` dict at the top of the file to adapt:

| Key | Default | What to change |
|-----|---------|----------------|
| `root_dir` | `data/raw/measurements` | Path to Experiment 1 data |
| `results_dir` | `results` | Where to save predictions / history |
| `models_dir` | `models` | Where to save checkpoints |
| `epochs` | `40` | Training epochs |
| `batch_size` | `64` | Batch size |
| `lr` | `1e-3` | Learning rate |

### 1c. Voltage-invariant split

```bash
python src/exp1_localization/train_voltage_invariant.py
```

Train/test split along voltage levels — tests generalisation to unseen voltages.

### 1d. Multi-task CNN (position + voltage joint prediction)

```bash
python src/exp1_localization/train_multitask.py
```

### 1e. Open-set: NonPD rejection + continuous regression

```bash
python src/exp1_localization/train_13class.py \
    --data_dir  data/raw/measurements \
    --nonpd_dir data/raw/nonpd \
    --mode      all
```

`--mode` options: `features` (RF + SVM only, fast CPU), `cnn`, `all`

### 1f. Noise robustness evaluation

```bash
python src/exp1_localization/noise_robustness_13class.py
```

Evaluates pretrained RF and CNN13 at controlled SNR levels.
Run after step 1e so model checkpoints exist.

---

## Experiment 2 — Cable Shield Integrity

### 2a. Extract pulse features → CSV

```bash
python src/exp2_shield/exp2_extract.py \
    --root data/raw/sames-cable-data \
    --out  artifacts/exp2_features.csv
```

### 2b. Feature-based classification (leave-one-file-out CV)

```bash
python src/exp2_shield/exp2_classify.py \
    --features artifacts/exp2_features.csv

# Improved versions
python src/exp2_shield/exp2_classify_v2.py --features artifacts/exp2_features.csv
python src/exp2_shield/exp2_classify_v3.py --features artifacts/exp2_features.csv
```

### 2c. CNN training

```bash
python src/exp2_shield/train_exp2.py
```

Edit the `CFG` dict to adapt:

| Key | Default | What to change |
|-----|---------|----------------|
| `root_dir` | `data/raw/sames-cable-data` | Path to Experiment 2 lab data |
| `results_dir` | `results/exp2` | Output directory |
| `epochs` | `80` | Training epochs (more needed; small dataset) |
| `aug_factor` | `10` | Online augmentation multiplier |

### 2d. Transfer learning (Exp1 CNN backbone → Exp2)

```bash
python src/exp2_shield/transfer_learning.py
```

Freezes the Exp1 backbone and fine-tunes a new classification head on Exp2 data.

### 2e. Factory / real-cable change detection

```bash
python src/exp2_shield/exp2_factory.py \
    --root    data/raw/sames-real \
    --out_dir artifacts/factory
```

---

## Figure Scripts

Run from the repo root after the relevant experiment has produced output:

```bash
python src/figures/pca_analysis.py
python src/figures/pca_analysis_v2.py
python src/figures/plot_all_positions.py
python src/figures/plot_pulse_representations.py
python src/figures/plot_two_channel_waveform.py
```

---

## Requirements Notes

| File | Use when |
|------|---------|
| `requirements.txt` | Reproducing the exact original Windows/CPU environment |
| `requirements-core.txt` | Any platform (Linux, macOS, Windows); no OS-specific pins |

`requirements.txt` includes `pywin32` (Windows-only) and CPU-only PyTorch.
Use `requirements-core.txt` on Linux / macOS or when installing with CUDA.