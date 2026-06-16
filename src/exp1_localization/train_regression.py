"""
train_regression.py — PD localization as open-set regression.

Architecture: two-stage detection + regression.

  Stage 1 (detection): binary classifier — PD vs NonPD
    - Input: top-40 features OR raw waveform
    - Output: PD (1) or NonPD (0)

  Stage 2 (regression): continuous distance estimator — only runs on
    samples classified as PD by Stage 1
    - Input: same features / waveform
    - Output: distance in metres (100–1900 m)

Training data:
  - PD signals: 32,926 files from data/raw/measurements
  - Natural noise: pre-trigger region (samples 0–3000) from same files
  - Rauscher NonPD: resampled signals from data/raw/nonpd (optional)

Classical models (feature-based):
  - RF, SVR, XGBoost, MLP — two separate models per algorithm
  - Evaluation: rejection accuracy + localization MAE on accepted samples

CNN model (raw waveform):
  - Shared backbone (5 ConvBlocks)
  - Detection head: FC → sigmoid (binary)
  - Regression head: FC → sigmoid scaled to [100, 1900]
  - Combined loss: L = L_detect + lambda * L_reg (only on PD samples)

Metrics:
  - Detection: precision, recall, F1 on NonPD rejection
  - Regression: MAE (m), RMSE (m), % within ±50m, ±100m
  - Per-position MAE breakdown
  - True vs predicted scatter

Usage:
    python src/train_regression.py
    python src/train_regression.py --mode all --epochs 50 --batch_size 128
    python src/train_regression.py --mode features  # classical only, fast
"""

import argparse
import json
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, classification_report,
                             mean_absolute_error, mean_squared_error,
                             r2_score, confusion_matrix)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_selection import f_classif, mutual_info_classif
from scipy.io import loadmat

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

RANDOM_STATE = 42
POSITIONS    = [100,200,300,500,700,900,1000,1300,1500,1600,1800,1900]
POS_TO_LABEL = {p:i for i,p in enumerate(POSITIONS)}
LABEL_TO_POS = {i:p for i,p in enumerate(POSITIONS)}
FS           = 100e6
CROP         = (3500, 7500)
NOISE_LEN    = 3000       # pre-trigger samples used as natural noise
D_MIN, D_MAX = 100.0, 1900.0

def norm_dist(d):   return (np.array(d) - D_MIN) / (D_MAX - D_MIN)
def denorm_dist(d): return np.array(d) * (D_MAX - D_MIN) + D_MIN


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def build_file_list(data_dir):
    samples = []
    root    = Path(data_dir)
    for pos, label in POS_TO_LABEL.items():
        for volt in range(1, 11):
            volt_dir = root / str(pos) / f'{volt}V'
            if not volt_dir.exists(): continue
            for f in volt_dir.glob('*.mat'):
                samples.append((str(f), float(pos), label))
    return samples


def load_exp1_cache(cache_path):
    c = np.load(cache_path)
    X, y, volts = c['X'], c['y'], c['volts']
    X = np.where(np.isfinite(X), X, 0.0)
    d = np.array([LABEL_TO_POS[int(yi)] for yi in y], dtype=np.float32)
    return X, y, d, volts


def load_nonpd_rauscher(nonpd_dir):
    """Load Rauscher NonPD features from cache if available."""
    cache = Path('results/baseline_13class/nonpd_features_cache.npz')
    if cache.exists():
        c = np.load(cache)
        print(f'  Loaded Rauscher NonPD features: {c["X"].shape}')
        return c['X']
    # try loading raw
    try:
        from dataset_nonpd import load_split, extract_features_batch
        sigs = []
        for split in ['train', 'val', 'test']:
            try: sigs.append(load_split(nonpd_dir, split))
            except FileNotFoundError: pass
        if sigs:
            all_sigs = np.vstack(sigs)
            print(f'  Extracting Rauscher NonPD features ({len(all_sigs)} signals)...')
            return extract_features_batch(all_sigs, desc='Rauscher NonPD')
    except Exception as e:
        print(f'  [WARN] Could not load Rauscher data: {e}')
    return None


def extract_noise_features_batch(samples, max_n=5000, seed=42):
    """Extract features from pre-trigger noise segments."""
    from features_v2 import (
        _time_features, _freq_features, _wavelet_features,
        _analytic_features, _inter_channel_features, feature_count
    )
    rng     = np.random.default_rng(seed)
    chosen  = rng.choice(len(samples), min(max_n, len(samples)), replace=False)
    F       = feature_count()
    X_noise = np.zeros((len(chosen), F), dtype=np.float32)

    for i, idx in enumerate(tqdm(chosen, desc='Natural noise features')):
        fp = samples[idx][0]
        try:
            mat  = loadmat(fp, simplify_cells=False)
            data = mat['tpd']['Data'][0, 0].astype(np.float32)
            ch1  = data[0, 0:NOISE_LEN].astype(np.float64)
            ch2  = data[1, 0:NOISE_LEN].astype(np.float64)
            t1 = _time_features(ch1); t2 = _time_features(ch2)
            f1n, f1w = _freq_features(ch1)
            f2n, f2w = _freq_features(ch2)
            w1 = _wavelet_features(ch1); w2 = _wavelet_features(ch2)
            a1 = _analytic_features(ch1); a2 = _analytic_features(ch2)
            inter = _inter_channel_features(ch1, ch2)
            feats = np.concatenate([t1,t2,f1n,f1w,f2n,f2w,w1,w2,a1,a2,inter])
            X_noise[i] = np.where(np.isfinite(feats), feats, 0.0)
        except Exception as e:
            pass

    return X_noise


def select_top_k(X_tr, y_tr, X_te, k=40):
    f_sc, _ = f_classif(X_tr, y_tr)
    f_sc    = np.nan_to_num(f_sc)
    mi      = mutual_info_classif(X_tr, y_tr,
                                   random_state=RANDOM_STATE, n_jobs=-1)
    imp     = RandomForestClassifier(
        n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE
    ).fit(X_tr, y_tr).feature_importances_

    def n01(a):
        r = a - a.min()
        return r / (r.max() + 1e-12)

    idx = np.argsort((n01(f_sc) + n01(mi) + n01(imp)) / 3)[::-1][:k]
    return X_tr[:, idx], X_te[:, idx], idx


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def reg_metrics(d_true, d_pred, name=''):
    mae  = mean_absolute_error(d_true, d_pred)
    rmse = np.sqrt(mean_squared_error(d_true, d_pred))
    r2   = r2_score(d_true, d_pred)
    w50  = np.mean(np.abs(d_true - d_pred) <= 50)  * 100
    w100 = np.mean(np.abs(d_true - d_pred) <= 100) * 100
    print(f'  {name:<30s} MAE={mae:.1f}m  RMSE={rmse:.1f}m  '
          f'R²={r2:.4f}  ±50m={w50:.1f}%  ±100m={w100:.1f}%')
    return dict(mae=mae, rmse=rmse, r2=r2, within_50m=w50, within_100m=w100)


def det_metrics(y_true, y_pred, name=''):
    acc = accuracy_score(y_true, y_pred) * 100
    tn  = np.sum((y_true == 0) & (y_pred == 0))
    fp  = np.sum((y_true == 0) & (y_pred == 1))
    fn  = np.sum((y_true == 1) & (y_pred == 0))
    tp  = np.sum((y_true == 1) & (y_pred == 1))
    precision = tp / (tp + fp + 1e-12) * 100
    recall    = tp / (tp + fn + 1e-12) * 100
    rej_acc   = tn / (tn + fp + 1e-12) * 100
    print(f'  {name:<30s} Det.acc={acc:.2f}%  '
          f'Precision={precision:.2f}%  Recall={recall:.2f}%  '
          f'NonPD rejection={rej_acc:.2f}%')
    return dict(det_acc=acc, precision=precision,
                recall=recall, nonpd_rejection=rej_acc)


def per_pos_mae_plot(d_true, d_pred, out_dir, name):
    maes = [mean_absolute_error(d_true[d_true==p], d_pred[d_true==p])
            if (d_true==p).sum() > 0 else 0 for p in POSITIONS]
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    ax.bar([f'{p}m' for p in POSITIONS], maes, color='#2b6cb0')
    ax.set_xlabel('Injection position')
    ax.set_ylabel('MAE (m)')
    ax.set_title(f'Per-position MAE — {name}')
    ax.set_xticklabels([f'{p}m' for p in POSITIONS], rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(maes):
        ax.text(i, v+1, f'{v:.0f}', ha='center', fontsize=7)
    fig.savefig(out_dir / f'per_pos_{name.replace(" ","_")}.png',
                dpi=150, bbox_inches='tight')
    plt.close(fig)


def scatter_plot(d_true, d_pred, out_dir, name):
    fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
    ax.scatter(d_true, d_pred, alpha=0.15, s=4, color='black')
    ax.plot([D_MIN,D_MAX],[D_MIN,D_MAX],'r--',linewidth=1.0)
    ax.set_xlabel('True distance (m)')
    ax.set_ylabel('Predicted distance (m)')
    ax.set_title(f'True vs predicted — {name}')
    ax.set_xlim(D_MIN-50, D_MAX+50)
    ax.set_ylim(D_MIN-50, D_MAX+50)
    ax.grid(True, alpha=0.3)
    fig.savefig(out_dir / f'scatter_{name.replace(" ","_")}.png',
                dpi=150, bbox_inches='tight')
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Two-stage classical pipeline
# ─────────────────────────────────────────────────────────────────────────────

class TwoStagePipeline:
    """
    Stage 1: binary detector  (PD=1, NonPD=0)
    Stage 2: distance regressor (runs only on PD-accepted samples)
    """
    def __init__(self, detector, regressor, name='TwoStage'):
        self.detector  = detector
        self.regressor = regressor
        self.name      = name

    def fit(self, X_pd, d_pd, X_nonpd):
        # Stage 1: binary
        X_bin = np.vstack([X_pd, X_nonpd])
        y_bin = np.concatenate([np.ones(len(X_pd)),
                                 np.zeros(len(X_nonpd))]).astype(int)
        self.detector.fit(X_bin, y_bin)

        # Stage 2: regression on PD only
        self.regressor.fit(X_pd, d_pd)

    def predict(self, X):
        """
        Returns (is_pd, distance) where distance=NaN for NonPD samples.
        """
        is_pd    = self.detector.predict(X).astype(bool)
        dist_out = np.full(len(X), np.nan)
        if is_pd.sum() > 0:
            dist_out[is_pd] = self.regressor.predict(X[is_pd])
        return is_pd, dist_out


def get_classical_pipelines():
    pipes = {}

    # RF
    pipes['RF'] = TwoStagePipeline(
        detector=RandomForestClassifier(
            n_estimators=500, n_jobs=-1, random_state=RANDOM_STATE),
        regressor=RandomForestRegressor(
            n_estimators=500, n_jobs=-1, random_state=RANDOM_STATE),
        name='RF'
    )

    # SVM
    pipes['SVM'] = TwoStagePipeline(
        detector=Pipeline([('sc', StandardScaler()),
                           ('clf', SVC(kernel='rbf', C=10,
                                       gamma='scale', random_state=RANDOM_STATE))]),
        regressor=Pipeline([('sc', StandardScaler()),
                            ('reg', SVR(kernel='rbf', C=100,
                                        gamma='scale', epsilon=10))]),
        name='SVM'
    )

    # XGBoost / GBT
    if HAS_XGB:
        pipes['XGBoost'] = TwoStagePipeline(
            detector=Pipeline([('sc', StandardScaler()),
                               ('clf', XGBClassifier(n_estimators=300,
                                   max_depth=6, learning_rate=0.1,
                                   random_state=RANDOM_STATE,
                                   eval_metric='logloss', verbosity=0))]),
            regressor=Pipeline([('sc', StandardScaler()),
                                ('reg', XGBRegressor(n_estimators=300,
                                    max_depth=6, learning_rate=0.1,
                                    random_state=RANDOM_STATE, verbosity=0))]),
            name='XGBoost'
        )
    else:
        pipes['GBT'] = TwoStagePipeline(
            detector=GradientBoostingClassifier(
                n_estimators=200, random_state=RANDOM_STATE),
            regressor=GradientBoostingRegressor(
                n_estimators=200, random_state=RANDOM_STATE),
            name='GBT'
        )

    # MLP
    pipes['MLP'] = TwoStagePipeline(
        detector=Pipeline([('sc', StandardScaler()),
                           ('clf', MLPClassifier(
                               hidden_layer_sizes=(256,128,64),
                               max_iter=500, early_stopping=True,
                               random_state=RANDOM_STATE))]),
        regressor=Pipeline([('sc', StandardScaler()),
                            ('reg', MLPRegressor(
                                hidden_layer_sizes=(256,128,64),
                                max_iter=500, early_stopping=True,
                                random_state=RANDOM_STATE))]),
        name='MLP'
    )

    return pipes


def run_classical(X_pd_tr, d_pd_tr, X_nonpd_tr,
                  X_pd_te, d_pd_te, X_nonpd_te,
                  out_dir, volts_pd=None, y_cls_pd=None,
                  X_pd_all=None, d_pd_all=None, volts_all=None,
                  top_k=40):

    # Feature selection using PD training set + binary labels
    X_bin_tr = np.vstack([X_pd_tr, X_nonpd_tr])
    y_bin_tr = np.concatenate([np.ones(len(X_pd_tr)),
                                np.zeros(len(X_nonpd_tr))]).astype(int)
    X_bin_te = np.vstack([X_pd_te, X_nonpd_te])
    y_bin_te = np.concatenate([np.ones(len(X_pd_te)),
                                np.zeros(len(X_nonpd_te))]).astype(int)

    X_bin_tr_k, X_bin_te_k, feat_idx = select_top_k(
        X_bin_tr, y_bin_tr, X_bin_te, k=top_k)
    X_pd_tr_k  = X_pd_tr[:, feat_idx]
    X_pd_te_k  = X_pd_te[:, feat_idx]
    X_nonpd_tr_k = X_nonpd_tr[:, feat_idx]
    X_nonpd_te_k = X_nonpd_te[:, feat_idx]

    results = {}
    pipes   = get_classical_pipelines()

    print('\nClassical two-stage pipelines:')
    for name, pipe in pipes.items():
        t0 = time.time()
        pipe.fit(X_pd_tr_k, d_pd_tr, X_nonpd_tr_k)
        elapsed = time.time() - t0

        # Evaluate on mixed test set
        is_pd_pred, dist_pred = pipe.predict(X_bin_te_k)

        # Detection metrics (full test set)
        dm = det_metrics(y_bin_te, is_pd_pred.astype(int), name)

        # Regression metrics (PD samples only, correctly detected)
        pd_mask  = y_bin_te == 1
        acc_mask = pd_mask & is_pd_pred
        if acc_mask.sum() > 0:
            rm = reg_metrics(d_pd_te[acc_mask[:len(d_pd_te)]],
                             dist_pred[acc_mask][:acc_mask.sum()], name)
        else:
            rm = dict(mae=np.nan, rmse=np.nan, r2=np.nan,
                      within_50m=0, within_100m=0)

        # Per-position and scatter (PD accepted only)
        pd_te_accepted_mask = acc_mask[:len(d_pd_te)]
        if pd_te_accepted_mask.sum() > 5:
            per_pos_mae_plot(d_pd_te[pd_te_accepted_mask],
                             dist_pred[pd_mask][:pd_te_accepted_mask.sum()],
                             out_dir, name)
            scatter_plot(d_pd_te[pd_te_accepted_mask],
                         dist_pred[pd_mask][:pd_te_accepted_mask.sum()],
                         out_dir, name)

        results[name] = {**dm, **rm, 'train_time_s': elapsed}
        print(f'    Training time: {elapsed:.1f}s')

    # Voltage-invariant RF
    if volts_all is not None and X_pd_all is not None:
        print('\n  Voltage-invariant RF:')
        tr_vi = volts_all <= 8
        te_vi = volts_all >= 9
        X_pd_vi_tr = X_pd_all[tr_vi][:, feat_idx]
        X_pd_vi_te = X_pd_all[te_vi][:, feat_idx]
        d_pd_vi_tr = d_pd_all[tr_vi]
        d_pd_vi_te = d_pd_all[te_vi]

        rf_vi = TwoStagePipeline(
            detector=RandomForestClassifier(
                n_estimators=500, n_jobs=-1, random_state=RANDOM_STATE),
            regressor=RandomForestRegressor(
                n_estimators=500, n_jobs=-1, random_state=RANDOM_STATE),
            name='RF_vi'
        )
        # use same nonpd for vi training
        rf_vi.fit(X_pd_vi_tr, d_pd_vi_tr, X_nonpd_tr_k)
        X_vi_te = np.vstack([X_pd_vi_te, X_nonpd_te_k])
        y_vi_te = np.concatenate([np.ones(len(X_pd_vi_te)),
                                   np.zeros(len(X_nonpd_te_k))]).astype(int)
        is_pd_pred_vi, dist_pred_vi = rf_vi.predict(X_vi_te)
        det_metrics(y_vi_te, is_pd_pred_vi.astype(int), 'RF_volt_invariant')
        acc_vi = (y_vi_te == 1) & is_pd_pred_vi
        if acc_vi.sum() > 0:
            rm_vi = reg_metrics(d_pd_vi_te[acc_vi[:len(d_pd_vi_te)]],
                                dist_pred_vi[acc_vi][:acc_vi.sum()],
                                'RF_volt_invariant')
            results['RF_volt_invariant'] = rm_vi

    return results


# ─────────────────────────────────────────────────────────────────────────────
# CNN two-head model
# ─────────────────────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=7, dropout=0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, padding=kernel//2),
            nn.BatchNorm1d(out_ch), nn.ReLU(),
            nn.MaxPool1d(2), nn.Dropout(dropout),
        )
    def forward(self, x): return self.block(x)


class TwoHeadCNN(nn.Module):
    """
    Shared backbone + two heads:
      - detection_head: sigmoid → P(PD)
      - regression_head: sigmoid scaled to [0,1] → denorm to metres
    """
    def __init__(self, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(2,  32, dropout=dropout),
            ConvBlock(32, 64, dropout=dropout),
            ConvBlock(64,128, dropout=dropout),
            ConvBlock(128,256,dropout=dropout),
            ConvBlock(256,256,dropout=dropout),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.detection_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1), nn.Sigmoid()
        )
        self.regression_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 1), nn.Sigmoid()
        )

    def forward(self, x):
        feat  = self.pool(self.features(x))
        p_pd  = self.detection_head(feat).squeeze(1)    # (B,)
        d_norm = self.regression_head(feat).squeeze(1)  # (B,) in [0,1]
        return p_pd, d_norm


class TwoHeadDataset(Dataset):
    """
    Returns (signal, is_pd, d_norm) where:
      - is_pd = 1 for PD signals, 0 for noise
      - d_norm = normalised distance for PD, 0.0 for noise (masked in loss)
    """
    def __init__(self, pd_samples, noise_samples, rauscher_signals=None):
        self.pd_samples       = pd_samples       # list of (filepath, dist_m)
        self.noise_samples    = noise_samples    # list of filepath
        self.rauscher_signals = rauscher_signals # np array (N,2,4000) or None
        self.n_pd      = len(pd_samples)
        self.n_noise   = len(noise_samples)
        self.n_rauscher = len(rauscher_signals) if rauscher_signals is not None else 0

    def __len__(self):
        return self.n_pd + self.n_noise + self.n_rauscher

    def __getitem__(self, idx):
        if idx < self.n_pd:
            fp, dist = self.pd_samples[idx]
            mat  = loadmat(fp, simplify_cells=False)
            sig  = mat['tpd']['Data'][0,0].astype(np.float32)[:,3500:7500]
            mu   = sig.mean(); std = sig.std() + 1e-8
            sig  = (sig - mu) / std
            return (torch.tensor(sig),
                    torch.tensor(1.0),
                    torch.tensor(float(norm_dist(dist))))

        elif idx < self.n_pd + self.n_noise:
            fp = self.noise_samples[idx - self.n_pd]
            mat  = loadmat(fp, simplify_cells=False)
            sig  = mat['tpd']['Data'][0,0].astype(np.float32)[:,0:NOISE_LEN]
            # pad to 4000 (repeat to fill)
            n = sig.shape[1]
            reps = int(np.ceil(4000 / n))
            sig  = np.tile(sig, (1, reps))[:, :4000]
            mu   = sig.mean(); std = sig.std() + 1e-8
            sig  = (sig - mu) / std
            return (torch.tensor(sig),
                    torch.tensor(0.0),
                    torch.tensor(0.0))

        else:
            i   = idx - self.n_pd - self.n_noise
            sig = self.rauscher_signals[i].copy()
            mu  = sig.mean(); std = sig.std() + 1e-8
            sig = (sig - mu) / std
            return (torch.tensor(sig),
                    torch.tensor(0.0),
                    torch.tensor(0.0))


def train_cnn(pd_samples, noise_files, rauscher_sigs,
              out_dir, device, epochs=50, batch_size=64,
              lambda_reg=1.0):
    print(f'\nTraining Two-Head CNN on {device}')
    rng = np.random.default_rng(RANDOM_STATE)

    # Split pd samples
    rng.shuffle(pd_samples)
    n = len(pd_samples)
    n_tr = int(0.70*n); n_va = int(0.15*n)
    pd_tr = pd_samples[:n_tr]
    pd_va = pd_samples[n_tr:n_tr+n_va]
    pd_te = pd_samples[n_tr+n_va:]

    # Split noise files
    rng.shuffle(noise_files)
    nn_ = len(noise_files)
    noise_tr = noise_files[:int(0.70*nn_)]
    noise_va = noise_files[int(0.70*nn_):int(0.85*nn_)]
    noise_te = noise_files[int(0.85*nn_):]

    # Rauscher split
    r_tr = r_va = r_te = None
    if rauscher_sigs is not None:
        rng.shuffle(rauscher_sigs)
        nr = len(rauscher_sigs)
        r_tr = rauscher_sigs[:int(0.70*nr)]
        r_va = rauscher_sigs[int(0.70*nr):int(0.85*nr)]
        r_te = rauscher_sigs[int(0.85*nr):]

    tr_ds = TwoHeadDataset(pd_tr, noise_tr, r_tr)
    va_ds = TwoHeadDataset(pd_va, noise_va, r_va)
    te_ds = TwoHeadDataset(pd_te, noise_te, r_te)

    print(f'  Train: {len(tr_ds)}  Val: {len(va_ds)}  Test: {len(te_ds)}')

    tr_dl = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,
                       num_workers=2, pin_memory=True)
    va_dl = DataLoader(va_ds, batch_size=batch_size, shuffle=False,
                       num_workers=2, pin_memory=True)
    te_dl = DataLoader(te_ds, batch_size=batch_size, shuffle=False,
                       num_workers=2, pin_memory=True)

    model = TwoHeadCNN().to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    bce   = nn.BCELoss()
    mse   = nn.MSELoss(reduction='none')

    best_va = float('inf')
    history = {'tr_det':[], 'va_det':[], 'tr_mae':[], 'va_mae':[]}

    for epoch in range(1, epochs+1):
        model.train()
        tr_det_loss = tr_reg_loss = tr_n = 0
        for X, is_pd, d_norm in tqdm(tr_dl, desc=f'E{epoch}', leave=False):
            X, is_pd, d_norm = X.to(device), is_pd.to(device), d_norm.to(device)
            opt.zero_grad()
            p_pd, d_pred = model(X)

            # detection loss (all samples)
            l_det = bce(p_pd, is_pd)

            # regression loss (PD samples only)
            pd_mask = is_pd > 0.5
            if pd_mask.sum() > 0:
                l_reg = mse(d_pred[pd_mask], d_norm[pd_mask]).mean()
            else:
                l_reg = torch.tensor(0.0, device=device)

            loss = l_det + lambda_reg * l_reg
            loss.backward(); opt.step()
            tr_det_loss += l_det.item() * len(X)
            tr_reg_loss += l_reg.item() * len(X)
            tr_n        += len(X)
        sched.step()

        model.eval()
        va_preds_pd, va_true_pd = [], []
        va_det_correct = va_det_total = 0
        with torch.no_grad():
            for X, is_pd, d_norm in va_dl:
                X, is_pd, d_norm = X.to(device), is_pd.to(device), d_norm.to(device)
                p_pd, d_pred = model(X)
                pred_bin = (p_pd > 0.5).float()
                va_det_correct += (pred_bin == is_pd).sum().item()
                va_det_total   += len(X)
                pd_mask = is_pd > 0.5
                if pd_mask.sum() > 0:
                    va_preds_pd.extend(
                        denorm_dist(d_pred[pd_mask].cpu().numpy()))
                    va_true_pd.extend(
                        denorm_dist(d_norm[pd_mask].cpu().numpy()))

        va_det_acc = va_det_correct / va_det_total * 100
        va_mae     = mean_absolute_error(va_true_pd, va_preds_pd) \
                     if va_preds_pd else float('inf')

        history['tr_det'].append(tr_det_loss / tr_n)
        history['va_det'].append(va_det_acc)
        history['va_mae'].append(va_mae)

        combined = (100 - va_det_acc) + va_mae / 100
        if combined < best_va:
            best_va = combined
            torch.save(model.state_dict(),
                       out_dir / 'cnn_twohead_best.pt')

        if epoch % 5 == 0 or epoch == 1:
            print(f'  E{epoch:3d}: det_acc={va_det_acc:.2f}%  '
                  f'reg_MAE={va_mae:.1f}m')

    # ── Test ─────────────────────────────────────────────────────────────────
    model.load_state_dict(torch.load(out_dir / 'cnn_twohead_best.pt',
                                     map_location=device))
    model.eval()
    te_is_pd_true, te_is_pd_pred = [], []
    te_d_true,     te_d_pred     = [], []

    with torch.no_grad():
        for X, is_pd, d_norm in te_dl:
            X = X.to(device)
            p_pd, d_pred = model(X)
            pred_bin = (p_pd > 0.5).cpu().numpy()
            te_is_pd_true.extend(is_pd.numpy())
            te_is_pd_pred.extend(pred_bin)
            pd_mask = is_pd > 0.5
            if pd_mask.sum() > 0:
                te_d_true.extend(
                    denorm_dist(d_norm[pd_mask].numpy()))
                te_d_pred.extend(
                    denorm_dist(d_pred[pd_mask.to(device)].cpu().numpy()))

    te_is_pd_true = np.array(te_is_pd_true)
    te_is_pd_pred = np.array(te_is_pd_pred)
    te_d_true     = np.array(te_d_true)
    te_d_pred     = np.array(te_d_pred)

    print('\nTest results:')
    dm = det_metrics(te_is_pd_true, te_is_pd_pred, 'CNN_TwoHead')
    rm = {}
    if len(te_d_true) > 0:
        rm = reg_metrics(te_d_true, te_d_pred, 'CNN_TwoHead')
        per_pos_mae_plot(te_d_true, te_d_pred, out_dir, 'CNN_TwoHead')
        scatter_plot(te_d_true, te_d_pred, out_dir, 'CNN_TwoHead')

    # Training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4),
                                    constrained_layout=True)
    ax1.plot(history['va_det']); ax1.set_title('Val detection accuracy (%)')
    ax1.set_xlabel('Epoch'); ax1.grid(True, alpha=0.3)
    ax2.plot(history['va_mae']); ax2.set_title('Val regression MAE (m)')
    ax2.set_xlabel('Epoch'); ax2.grid(True, alpha=0.3)
    fig.suptitle('Two-Head CNN training history')
    fig.savefig(out_dir / 'cnn_twohead_history.png', dpi=150)
    plt.close(fig)

    json.dump(history, open(out_dir / 'cnn_twohead_history.json', 'w'))
    return {**dm, **rm}


# ─────────────────────────────────────────────────────────────────────────────
# Summary plot
# ─────────────────────────────────────────────────────────────────────────────

def summary_plot(all_results, out_dir):
    names = [n for n, r in all_results.items() if 'mae' in r]
    maes  = [all_results[n]['mae']         for n in names]
    w50   = [all_results[n]['within_50m']  for n in names]
    rej   = [all_results[n].get('nonpd_rejection', 0) for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)

    axes[0].bar(names, maes, color='#2b6cb0')
    axes[0].set_ylabel('MAE (m)'); axes[0].set_title('Localization MAE')
    axes[0].set_xticklabels(names, rotation=40, ha='right', fontsize=8)
    axes[0].grid(True, alpha=0.3, axis='y')

    axes[1].bar(names, w50, color='#2b6cb0')
    axes[1].set_ylabel('% within ±50m')
    axes[1].set_title('Predictions within ±50m')
    axes[1].set_xticklabels(names, rotation=40, ha='right', fontsize=8)
    axes[1].set_ylim(0, 105); axes[1].grid(True, alpha=0.3, axis='y')

    rej_names = [n for n in all_results if 'nonpd_rejection' in all_results[n]]
    rej_vals  = [all_results[n]['nonpd_rejection'] for n in rej_names]
    axes[2].bar(rej_names, rej_vals, color='#2b6cb0')
    axes[2].set_ylabel('NonPD rejection rate (%)')
    axes[2].set_title('NonPD rejection accuracy')
    axes[2].set_xticklabels(rej_names, rotation=40, ha='right', fontsize=8)
    axes[2].set_ylim(0, 105); axes[2].grid(True, alpha=0.3, axis='y')

    fig.suptitle('Open-set regression — detection + localization', fontsize=10)
    fig.savefig(out_dir / 'regression_summary.png', dpi=150,
                bbox_inches='tight')
    plt.close(fig)
    print('Saved regression_summary.png')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print('[1] Loading PD features...')
    X_pd, y_cls, d_pd, volts = load_exp1_cache(args.exp1_cache)

    print('[2] Splitting PD data...')
    sss = StratifiedShuffleSplit(1, test_size=0.30, random_state=RANDOM_STATE)
    tr, tmp = next(sss.split(X_pd, y_cls))
    sss2 = StratifiedShuffleSplit(1, test_size=0.50, random_state=RANDOM_STATE)
    va, te = next(sss2.split(X_pd[tmp], y_cls[tmp]))
    va, te = tmp[va], tmp[te]

    X_pd_tr, d_tr = X_pd[tr],  d_pd[tr]
    X_pd_te, d_te = X_pd[te],  d_pd[te]

    print('[3] Loading NonPD data...')
    all_files = []
    root = Path(args.data_dir)
    for pos in POSITIONS:
        for volt in range(1, 11):
            vd = root / str(pos) / f'{volt}V'
            if vd.exists():
                all_files.extend([(str(f), float(pos), 0)
                                   for f in vd.glob('*.mat')])

    rng = np.random.default_rng(RANDOM_STATE)
    rng.shuffle(all_files)
    noise_files_all = [f[0] for f in all_files[:args.max_nonpd * 2]]

    print(f'  Extracting natural noise features ({min(args.max_nonpd, len(noise_files_all))} samples)...')
    X_noise = extract_noise_features_batch(
        [(f, 0, 0) for f in noise_files_all], max_n=args.max_nonpd)
    n_noise = len(X_noise)
    n_tr_noise = int(0.70 * n_noise)
    X_noise_tr = X_noise[:n_tr_noise]
    X_noise_te = X_noise[n_tr_noise:]

    # Optional Rauscher data
    X_rauscher = None
    if args.nonpd_dir and Path(args.nonpd_dir).exists():
        X_rauscher = load_nonpd_rauscher(args.nonpd_dir)
        if X_rauscher is not None:
            # Add to noise pool
            nr = len(X_rauscher)
            X_noise_tr = np.vstack([X_noise_tr,
                                     X_rauscher[:int(0.70*nr)]])
            X_noise_te = np.vstack([X_noise_te,
                                     X_rauscher[int(0.70*nr):]])
            print(f'  Rauscher NonPD added: {nr} samples')

    print(f'  NonPD train: {len(X_noise_tr)}  test: {len(X_noise_te)}')

    all_results = {}

    if args.mode in ('features', 'all'):
        print('\n[4] Classical two-stage pipelines...')
        res = run_classical(
            X_pd_tr, d_tr, X_noise_tr,
            X_pd_te, d_te, X_noise_te,
            out_dir,
            volts_all=volts, X_pd_all=X_pd, d_pd_all=d_pd,
            top_k=args.top_k
        )
        all_results.update(res)

    if args.mode in ('cnn', 'all'):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'\n[5] CNN Two-Head model on {device}...')

        pd_samples = [(str(f), float(pos)) for f, pos, _ in all_files
                      if pos in POSITIONS]
        noise_fps  = noise_files_all[:args.max_nonpd]

        # Rauscher resampled signals for CNN
        rauscher_cnn = None
        if args.nonpd_dir and Path(args.nonpd_dir).exists():
            try:
                from dataset_nonpd import load_split, resample_batch
                segs = []
                for sp in ['train', 'val', 'test']:
                    try: segs.append(load_split(args.nonpd_dir, sp))
                    except: pass
                if segs:
                    all_r = np.vstack(segs)
                    rauscher_cnn = resample_batch(all_r)
                    print(f'  Rauscher CNN signals: {rauscher_cnn.shape}')
            except Exception as e:
                print(f'  [WARN] Rauscher CNN load failed: {e}')

        res_cnn = train_cnn(pd_samples, noise_fps, rauscher_cnn,
                             out_dir, device,
                             epochs=args.epochs,
                             batch_size=args.batch_size)
        all_results['CNN_TwoHead'] = res_cnn

    if all_results:
        summary_plot(all_results, out_dir)
        json.dump(
            {k: {kk: float(vv) for kk, vv in v.items()}
             for k, v in all_results.items()},
            open(out_dir / 'regression_results.json', 'w'), indent=2
        )

    print(f'\nAll outputs saved to: {out_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp1_cache', default='results/baseline_v2/features_cache.npz')
    parser.add_argument('--data_dir',   default='data/raw/measurements')
    parser.add_argument('--nonpd_dir',  default='data/raw/nonpd',
                        help='Path to Rauscher NonPD .mat files (optional)')
    parser.add_argument('--out_dir',    default='results/regression')
    parser.add_argument('--mode',       default='all',
                        choices=['features', 'cnn', 'all'])
    parser.add_argument('--top_k',      type=int, default=40)
    parser.add_argument('--max_nonpd',  type=int, default=5000)
    parser.add_argument('--epochs',     type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()
    main(args)