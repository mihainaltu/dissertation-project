"""
noise_robustness_13class.py — Noise robustness for 13-class RF and CNN13.

Evaluates RF and CNN13 on the 13-class position + NonPD rejection task
at controlled SNR levels. Noise is added to raw waveforms before feature
extraction (RF) or inference (CNN13). Models are NOT retrained.

Output columns (fills Table 5.X in the report):
  Noise level | RF Acc. (%) | CNN13 Acc. (%)

The Detection Acc. (%) and MAE (m) columns come from the two-head CNN
experiment and are NOT recomputed here.

Usage:
    python src/noise_robustness_13class.py
    python src/noise_robustness_13class.py --n_trials 1
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from scipy.io import loadmat
from scipy.signal import resample as scipy_resample

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_selection import f_classif, mutual_info_classif

import torch
import torch.nn as nn
import sys
sys.path.insert(0, str(Path(__file__).parent))

RANDOM_STATE = 42
POSITIONS    = [100,200,300,500,700,900,1000,1300,1500,1600,1800,1900]
POS_TO_LABEL = {p:i for i,p in enumerate(POSITIONS)}
LABEL_TO_POS = {i:p for i,p in enumerate(POSITIONS)}
NONPD_LABEL  = 12
FS           = 100e6
CROP         = (3500, 7500)
NONPD_SIG_LEN = 400
TARGET_LEN    = 4000

SNR_LEVELS   = ['Clean', 30, 20, 10, 5, 0]


# ─────────────────────────────────────────────────────────────────────────────
# CNN13 architecture — must match cnn_13class_best.pt exactly
# ─────────────────────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=7, dropout=0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, padding=kernel//2),  # bias=True
            nn.BatchNorm1d(out_ch), nn.ReLU(),
            nn.MaxPool1d(2), nn.Dropout(dropout),
        )
    def forward(self, x): return self.block(x)


class CNN13Class(nn.Module):
    def __init__(self, num_classes=13, dropout=0.3):
        super().__init__()
        self.convs = nn.Sequential(
            ConvBlock(2,   32, dropout=dropout),
            ConvBlock(32,  64, dropout=dropout),
            ConvBlock(64, 128, dropout=dropout),
            ConvBlock(128,256, dropout=dropout),
            ConvBlock(256,256, dropout=dropout),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256,128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )
    def forward(self, x):
        return self.head(self.pool(self.convs(x)))


# ─────────────────────────────────────────────────────────────────────────────
# Noise
# ─────────────────────────────────────────────────────────────────────────────

def add_awgn(sig, snr_db, rng):
    sig     = sig.astype(np.float64)
    p_sig   = np.mean(sig ** 2)
    if p_sig < 1e-12:
        return sig.astype(np.float32)
    p_noise = p_sig / (10 ** (snr_db / 10.0))
    return (sig + rng.normal(0, np.sqrt(p_noise), sig.shape)).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def build_file_list(data_dir):
    samples = []
    root = Path(data_dir)
    for pos, label in POS_TO_LABEL.items():
        for volt in range(1, 11):
            vd = root / str(pos) / f'{volt}V'
            if not vd.exists(): continue
            for f in sorted(vd.glob('*.mat')):
                samples.append((str(f), label, volt))
    return samples


def load_cache(cache_path):
    c = np.load(cache_path)
    X, y, volts = c['X'], c['y'], c['volts']
    return np.where(np.isfinite(X), X, 0.0), y, volts


def load_nonpd_signals(nonpd_dir):
    """
    Load Rauscher NonPD signals and resample to TARGET_LEN.
    Returns (N, 2, TARGET_LEN) float32 — ch2 = zeros.
    """
    root = Path(nonpd_dir)
    sigs = []
    for fname in ['Te0.mat']:   # test split only
        fp = root / fname
        if not fp.exists():
            print(f'  [WARN] NonPD file not found: {fp}')
            continue
        mat = loadmat(str(fp), simplify_cells=False)
        # shape: (N, 400) or (N, 1, 400)
        data = mat.get('data', mat.get('X', None))
        if data is None:
            keys = [k for k in mat if not k.startswith('__')]
            data = mat[keys[0]]
        # handle struct: shape (1,1) with 'signals' and 'labels' fields
        if hasattr(data, 'dtype') and data.dtype.names:
            if 'signals' in data.dtype.names:
                data = data['signals'][0, 0]  # (N, 1) object array
            else:
                data = data[data.dtype.names[0]][0, 0]
        # data is (N, 1) object array where each element is (1,) float array
        # stack into (N, 400) float32
        data = np.vstack([np.array(data[i, 0]).flatten()
                          for i in range(len(data))]).astype(np.float32)
        if data.ndim == 3:
            data = data[:, 0, :]
        sigs.append(data)

    if not sigs:
        print('  [WARN] No NonPD signals loaded — NonPD class will be empty')
        return np.zeros((0, 2, TARGET_LEN), dtype=np.float32)

    all_sigs = np.vstack(sigs)   # (N, 400)
    N = len(all_sigs)
    out = np.zeros((N, 2, TARGET_LEN), dtype=np.float32)
    for i in range(N):
        out[i, 0] = scipy_resample(
            all_sigs[i].astype(np.float64), TARGET_LEN
        ).astype(np.float32)
    print(f'  Loaded {N} NonPD signals (resampled 400→{TARGET_LEN})')
    return out


def load_pd_waveforms(all_files, te_idx):
    """Load raw cropped waveforms for the PD test split."""
    signals, labels = [], []
    for i in tqdm(te_idx, desc='Loading PD waveforms', leave=False):
        fp, label, _ = all_files[i]
        try:
            mat = loadmat(fp, simplify_cells=False)
            sig = mat['tpd']['Data'][0,0].astype(np.float32)
            lo, hi = CROP
            signals.append(sig[:, lo:hi])
            labels.append(label)
        except Exception as e:
            print(f'  [WARN] {fp}: {e}')
    return np.stack(signals), np.array(labels)


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_features_batch(waveforms, desc='Features'):
    """
    Extract 122 v2 features from a batch of (N, 2, L) waveforms.
    Returns (N, 122) float32.
    """
    from features_v2 import (
        _time_features, _freq_features, _wavelet_features,
        _analytic_features, _inter_channel_features, feature_count
    )
    N = len(waveforms)
    F = feature_count()
    X = np.zeros((N, F), dtype=np.float32)
    for i in tqdm(range(N), desc=desc, leave=False):
        ch1 = waveforms[i, 0].astype(np.float64)
        ch2 = waveforms[i, 1].astype(np.float64)
        t1=_time_features(ch1); t2=_time_features(ch2)
        f1n,f1w=_freq_features(ch1); f2n,f2w=_freq_features(ch2)
        w1=_wavelet_features(ch1); w2=_wavelet_features(ch2)
        a1=_analytic_features(ch1); a2=_analytic_features(ch2)
        inter=_inter_channel_features(ch1,ch2)
        feats=np.concatenate([t1,t2,f1n,f1w,f2n,f2w,w1,w2,a1,a2,inter])
        # replace inf/nan and clip extreme values to prevent feature explosion
        # on zero-channel signals (NonPD ch2=0) under noise
        feats = np.where(np.isfinite(feats), feats, 0.0)
        feats = np.clip(feats, -1e6, 1e6)
        X[i] = feats.astype(np.float32)
    return X


def select_top_k(X_tr, y_tr, k=40):
    f_sc, _ = f_classif(X_tr, y_tr)
    f_sc    = np.nan_to_num(f_sc)
    mi      = mutual_info_classif(X_tr, y_tr,
                                   random_state=RANDOM_STATE, n_jobs=-1)
    imp     = RandomForestClassifier(
        n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE
    ).fit(X_tr, y_tr).feature_importances_
    def n01(a):
        r = a - a.min(); return r / (r.max() + 1e-12)
    return np.argsort((n01(f_sc)+n01(mi)+n01(imp))/3)[::-1][:k]


# ─────────────────────────────────────────────────────────────────────────────
# RF 13-class pipeline
# ─────────────────────────────────────────────────────────────────────────────

def build_rf_13class(X_pd_tr, y_pd_tr, X_nonpd_tr, feat_idx, top_k=40):
    """
    Train a 13-class RF on PD training features + NonPD features.
    Returns (rf, scaler) fitted on training data only.
    """
    # NonPD features (ch2=zeros, full 4000 samples)
    print('  Extracting NonPD training features...')

    X_tr = np.vstack([X_pd_tr[:, feat_idx], X_nonpd_tr[:, feat_idx]])
    y_tr = np.concatenate([y_pd_tr,
                            np.full(len(X_nonpd_tr), NONPD_LABEL)])

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)

    rf = RandomForestClassifier(n_estimators=500, n_jobs=-1,
                                 random_state=RANDOM_STATE)
    rf.fit(X_tr_s, y_tr)
    return rf, scaler


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation at one SNR level
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_at_snr(snr, pd_waveforms, pd_labels,
                    nonpd_waveforms, nonpd_labels,
                    rf, scaler, feat_idx,
                    cnn13, device, n_trials, rng_base):
    """
    Returns (rf_acc_mean, rf_acc_std, cnn_acc_mean, cnn_acc_std)
    """
    rf_accs, cnn_accs = [], []

    for trial in range(n_trials):
        rng = np.random.default_rng(trial * 100 + 42)

        # ── Add noise ────────────────────────────────────────────────────
        if snr == 'Clean':
            pd_noisy   = pd_waveforms.copy()
            npd_noisy  = nonpd_waveforms.copy()
        else:
            pd_noisy  = np.stack([add_awgn(s, snr, rng) for s in pd_waveforms])
            npd_noisy = np.stack([add_awgn(s, snr, rng) for s in nonpd_waveforms]) \
                        if len(nonpd_waveforms) > 0 else nonpd_waveforms.copy()

        # ── RF: extract features → scale → predict ────────────────────────
        X_pd_feat  = extract_features_batch(pd_noisy,  desc=f'RF PD SNR={snr}')
        if len(npd_noisy) > 0:
            X_npd_feat = extract_features_batch(npd_noisy,
                                                desc=f'RF NonPD SNR={snr}')
            X_te = np.vstack([X_pd_feat[:, feat_idx],
                               X_npd_feat[:, feat_idx]])
        else:
            X_te = X_pd_feat[:, feat_idx]

        y_te = np.concatenate([pd_labels,
                                np.full(len(npd_noisy), NONPD_LABEL)])

        X_te_s    = scaler.transform(X_te)
        rf_preds  = rf.predict(X_te_s)
        rf_accs.append(accuracy_score(y_te, rf_preds) * 100)

        # ── CNN13: normalise → predict ─────────────────────────────────────
        if len(npd_noisy) > 0:
            all_waves = np.vstack([pd_noisy, npd_noisy])
        else:
            all_waves = pd_noisy.copy()

        mu  = all_waves.mean(axis=(1,2), keepdims=True)
        std = all_waves.std(axis=(1,2),  keepdims=True) + 1e-8
        all_norm = (all_waves - mu) / std

        cnn_preds = []
        cnn13.eval()
        with torch.no_grad():
            for i in range(0, len(all_norm), 128):
                X = torch.tensor(all_norm[i:i+128]).to(device)
                cnn_preds.extend(cnn13(X).argmax(1).cpu().numpy())
        cnn_accs.append(accuracy_score(y_te, cnn_preds) * 100)

    return (np.mean(rf_accs),  np.std(rf_accs),
            np.mean(cnn_accs), np.std(cnn_accs))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ── Load feature cache + PD split ────────────────────────────────────
    print('[1] Loading feature cache...')
    X, y, volts = load_cache(args.exp1_cache)
    sss  = StratifiedShuffleSplit(1, test_size=0.30, random_state=RANDOM_STATE)
    tr, tmp = next(sss.split(X, y))
    sss2 = StratifiedShuffleSplit(1, test_size=0.50, random_state=RANDOM_STATE)
    _, te = next(sss2.split(X[tmp], y[tmp]))
    te    = tmp[te]
    X_tr, y_tr = X[tr], y[tr]

    print('[2] Feature selection (top-40)...')
    feat_idx = select_top_k(X_tr, y_tr, k=40)

    # ── Load PD test waveforms ────────────────────────────────────────────
    print('[3] Loading PD test waveforms...')
    all_files   = build_file_list(args.data_dir)
    pd_waves, pd_labels = load_pd_waveforms(all_files, te)
    print(f'  PD test samples: {len(pd_waves)}')

    # ── Load NonPD test waveforms ─────────────────────────────────────────
    print('[4] Loading NonPD test waveforms...')
    nonpd_waves = load_nonpd_signals(args.nonpd_dir)
    nonpd_labels = np.full(len(nonpd_waves), NONPD_LABEL)

    # Cap NonPD test size to avoid very long evaluation
    if len(nonpd_waves) > args.max_nonpd_test:
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(len(nonpd_waves), args.max_nonpd_test, replace=False)
        nonpd_waves  = nonpd_waves[idx]
        nonpd_labels = nonpd_labels[idx]
    print(f'  NonPD test samples: {len(nonpd_waves)}')

    # ── Build and train 13-class RF (or load if already saved) ──────────
    print('[5] Building 13-class RF...')
    rf13_path  = out_dir / 'rf13_model.pkl'
    sca13_path = out_dir / 'rf13_scaler.pkl'
    feat_path  = out_dir / 'rf13_feat_idx.npy'

    if rf13_path.exists() and sca13_path.exists() and feat_path.exists():
        import pickle
        print('  Loading saved RF13 from disk...')
        with open(rf13_path,  'rb') as f: rf13     = pickle.load(f)
        with open(sca13_path, 'rb') as f: scaler13 = pickle.load(f)
        feat_idx = np.load(feat_path)
        feat_idx = feat_idx.astype(int)
    else:
        # NonPD training features
        nonpd_tr_dir = Path(args.nonpd_dir)
        nonpd_tr_waves_list = []
        for fname in ['Tr0.mat', 'Va0.mat']:
            fp = nonpd_tr_dir / fname
            if not fp.exists(): continue
            mat  = loadmat(str(fp), simplify_cells=False)
            keys = [k for k in mat if not k.startswith('__')]
            raw  = mat[keys[0]]
            if hasattr(raw, 'dtype') and raw.dtype.names:
                if 'signals' in raw.dtype.names:
                    raw = raw['signals'][0, 0]
                else:
                    raw = raw[raw.dtype.names[0]][0, 0]
            data = np.vstack([np.array(raw[i, 0]).flatten()
                              for i in range(len(raw))]).astype(np.float32)
            if data.ndim == 3: data = data[:,0,:]
            out = np.zeros((len(data), 2, TARGET_LEN), dtype=np.float32)
            for i in range(len(data)):
                out[i,0] = scipy_resample(
                    data[i].astype(np.float64), TARGET_LEN
                ).astype(np.float32)
            nonpd_tr_waves_list.append(out)

        if nonpd_tr_waves_list:
            nonpd_tr_waves = np.vstack(nonpd_tr_waves_list)
            if len(nonpd_tr_waves) > 5000:
                rng = np.random.default_rng(RANDOM_STATE)
                idx = rng.choice(len(nonpd_tr_waves), 5000, replace=False)
                nonpd_tr_waves = nonpd_tr_waves[idx]
            print(f'  Extracting NonPD training features ({len(nonpd_tr_waves)} signals)...')
            X_nonpd_tr = extract_features_batch(nonpd_tr_waves,
                                                 desc='NonPD train features')
        else:
            print('  [WARN] No NonPD training data found')
            X_nonpd_tr = np.zeros((0, X_tr.shape[1]), dtype=np.float32)

        rf13, scaler13 = build_rf_13class(X_tr, y_tr, X_nonpd_tr,
                                           feat_idx, top_k=40)

        # Save for future runs
        import pickle
        with open(rf13_path,  'wb') as f: pickle.dump(rf13,     f)
        with open(sca13_path, 'wb') as f: pickle.dump(scaler13, f)
        np.save(feat_path, feat_idx)
        print('  Saved RF13 model to disk')

    # ── Load CNN13 ────────────────────────────────────────────────────────
    print('[6] Loading CNN13...')
    cnn13 = CNN13Class(num_classes=13).to(device)
    cnn13.load_state_dict(torch.load(args.cnn13_path, map_location=device))
    cnn13.eval()

    # Sanity check on clean data
    print('[7] Sanity check (clean accuracy)...')
    res_clean = evaluate_at_snr('Clean', pd_waves, pd_labels,
                                 nonpd_waves, nonpd_labels,
                                 rf13, scaler13, feat_idx,
                                 cnn13, device, n_trials=1,
                                 rng_base=42)
    print(f'  RF13 clean:  {res_clean[0]:.2f}%')
    print(f'  CNN13 clean: {res_clean[2]:.2f}%')

    # ── Noise sweep ───────────────────────────────────────────────────────
    print('\n[8] Noise robustness sweep...')
    rows = []
    for snr in SNR_LEVELS:
        label = 'Clean' if snr == 'Clean' else f'+{snr} dB' if snr > 0 else f'{snr} dB'
        print(f'  SNR = {label}')
        rf_m, rf_s, cnn_m, cnn_s = evaluate_at_snr(
            snr, pd_waves, pd_labels,
            nonpd_waves, nonpd_labels,
            rf13, scaler13, feat_idx,
            cnn13, device, n_trials=args.n_trials,
            rng_base=42
        )
        rows.append({
            'SNR': label,
            'RF_acc_mean':  rf_m,  'RF_acc_std':  rf_s,
            'CNN_acc_mean': cnn_m, 'CNN_acc_std': cnn_s,
        })
        print(f'    RF:  {rf_m:.2f}% ± {rf_s:.2f}%   '
              f'CNN: {cnn_m:.2f}% ± {cnn_s:.2f}%')

    # ── Output ────────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    csv_path = out_dir / 'noise_robustness_13class.csv'
    df.to_csv(csv_path, index=False)
    print(f'\nSaved: {csv_path}')

    # ── Print final table ─────────────────────────────────────────────────
    print('\n' + '='*58)
    print(f'  {"SNR":<10}  {"RF Acc. (%)":>14}  {"CNN13 Acc. (%)":>16}')
    print('='*58)
    for r in rows:
        rf_str  = f'{r["RF_acc_mean"]:.2f} ± {r["RF_acc_std"]:.2f}'
        cnn_str = f'{r["CNN_acc_mean"]:.2f} ± {r["CNN_acc_std"]:.2f}'
        print(f'  {r["SNR"]:<10}  {rf_str:>14}  {cnn_str:>16}')
    print('='*58)

    # ── Full table with two-head CNN columns ─────────────────────────────
    twohead = {
        'Clean':  ('99.81', '11.8'),
        '+30 dB': ('99.62', '20.5'),
        '+20 dB': ('97.17', '71.2'),
        '+10 dB': ('9.67',  '728.9'),
        '+5 dB':  ('0.00',  '900.9'),
        '0 dB':   ('0.00',  '901.5'),
    }

    print('\nFull noise robustness table:')
    print('='*72)
    print(f'  {"Noise level":<12}  {"RF Acc. (%)":>12}  '
          f'{"CNN13 Acc. (%)":>14}  {"Det. Acc. (%)":>14}  {"MAE (m)":>8}')
    print('='*72)
    for r in rows:
        snr = r['SNR']
        rf  = f'{r["RF_acc_mean"]:.2f}'
        cnn = f'{r["CNN_acc_mean"]:.2f}'
        det, mae = twohead.get(snr, ('--', '--'))
        print(f'  {snr:<12}  {rf:>12}  {cnn:>14}  {det:>14}  {mae:>8}')
    print('='*72)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp1_cache',    default='results/baseline_v2/features_cache.npz')
    parser.add_argument('--data_dir',      default='data/raw/measurements')
    parser.add_argument('--nonpd_dir',     default='data/raw/nonpd')
    parser.add_argument('--cnn13_path',    default='results/baseline_13class/cnn_13class_best.pt')
    parser.add_argument('--out_dir',       default='results/noise_robustness_13class')
    parser.add_argument('--n_trials',      type=int, default=3)
    parser.add_argument('--max_nonpd_test',type=int, default=5000,
                        help='Cap NonPD test size to speed up evaluation')
    args = parser.parse_args()
    main(args)