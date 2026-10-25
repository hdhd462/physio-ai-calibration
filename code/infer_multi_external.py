# -*- coding: utf-8 -*-
"""Multi-task FusionNet external inference: per-class metrics + cross-fitted Platt/Beta repair.

Cohorts: Chapman-Shaoxing (China) and CODE-15% part17 (Brazil).
Requires: fusion_multi.pt, chapman_multilabel.npz, code15_multilabel.npz, ecg_features_code15.csv.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA_DIR, PROJ_DIR, FIG_DIR, SUB_DIR, PATCH_DIR, NHANES_DIR, PROJ1_DIR, SHAOXING_DIR, PTBXL_DIR
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, DATA_DIR)
import numpy as np, pandas as pd, torch, torch.nn as nn, os
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from eval_multi import eval_all

OUT = DATA_DIR
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
COLS = ['age', 'sex', 'hr_std', 'rr_mean', 'rmssd', 'sdnn', 'pnn50', 'cv_rr', 'rmssd_norm',
        'dom_freq', 'spec_ent', 'qrs_dur', 'pr_int', 'qt_int', 'p_amp', 'sig_std_v2', 'sig_std_i', 'n_peaks']
K = 5
NAMES = ['AF', 'RBBB', 'LBBB', 'IAVB', 'NORM']
print('device:', DEVICE)

class FusionNetMulti(nn.Module):
    def __init__(self, d_model=64, nhead=4, nlayers=2, ntab=18, k=K):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(12, 32, 7, stride=2, padding=3), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, d_model, 5, stride=2, padding=2), nn.BatchNorm1d(d_model), nn.ReLU())
        self.pos = nn.Parameter(torch.randn(1, d_model, 250) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=128,
                                            dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=nlayers)
        self.tab = nn.Sequential(nn.Linear(ntab, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 16), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(d_model + 16, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, k))
    def forward(self, x, t):
        h = self.conv(x).transpose(1, 2)
        h = self.encoder(h).mean(dim=1)
        f = self.tab(t)
        return self.head(torch.cat([h, f], dim=1))

ptb = np.load(f'{OUT}\\ptbxl_100hz.npz', allow_pickle=True)
X_p, fold = ptb['X'], ptb['fold']
feat_p = pd.read_csv(f'{OUT}\\ecg_features_ptbxl.csv')
feat_p['fold'] = fold[feat_p['idx'].values]
df_tr = feat_p[feat_p.fold <= 8]
mu = X_p[fold <= 8].mean(axis=(0, 2), keepdims=True)
sd = X_p[fold <= 8].std(axis=(0, 2), keepdims=True) + 1e-8
def norm(x): return ((x - mu) / sd).astype(np.float32)
imp, sc = SimpleImputer(strategy='median'), StandardScaler()
sc.fit(imp.fit_transform(df_tr[COLS]))
def tab(df): return sc.transform(imp.transform(df[COLS])).astype(np.float32)

model = FusionNetMulti().to(DEVICE)
model.load_state_dict(torch.load(f'{OUT}\\fusion_multi.pt', map_location=DEVICE))
model.eval()

def predict(X, T, bs=256):
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(X[i:i+bs]).to(DEVICE)
            tb = torch.from_numpy(T[i:i+bs].astype(np.float32)).to(DEVICE)
            out.append(torch.sigmoid(model(xb, tb)).cpu().numpy())
    return np.concatenate(out)

def run_cohort(name, X, T, Y, ids):
    print(f'\n===== {name} (n={len(Y)}) =====')
    P = predict(X, T)
    res = {}
    for k, cls in enumerate(NAMES):
        ev = int(Y[:, k].sum())
        if ev < 2 or ev > len(Y) - 2:
            print(f'  SKIP {cls}: only {ev} events (labels unavailable/unstable)')
            continue
        res[cls] = eval_all(Y[:, k], P[:, k], f'{name} {cls}')
    return P, res

summary = {}
# ── Chapman ──
if os.path.exists(f'{OUT}\\chapman_multilabel.npz'):
    ch = np.load(f'{OUT}\\chapman_100hz.npz', allow_pickle=True)
    mlc = np.load(f'{OUT}\\chapman_multilabel.npz', allow_pickle=True)
    assert np.array_equal(mlc['rec_id'].astype(str), ch['rec_id'].astype(str)), 'chapman id order mismatch'
    feat_c = pd.read_csv(f'{OUT}\\ecg_features_chapman.csv')
    ok = ~np.isnan(ch['X']).any(axis=(1, 2))
    valid = feat_c[feat_c['idx'].isin(np.where(ok)[0])].sort_values('idx').reset_index(drop=True)
    idx = valid['idx'].values
    P, res = run_cohort('Chapman', norm(ch['X'][idx]), tab(valid),
                        mlc['Y'][idx], ch['rec_id'][idx])
    summary['chapman'] = (P, idx, res)
else:
    print('SKIP Chapman: chapman_multilabel.npz missing')

# ── CODE-15% ──
if os.path.exists(f'{OUT}\\ecg_features_code15.csv') and os.path.exists(f'{OUT}\\code15_multilabel.npz'):
    cd = np.load(f'{OUT}\\code15_part17_100hz.npz', allow_pickle=True)
    mlcd = np.load(f'{OUT}\\code15_multilabel.npz', allow_pickle=True)
    assert np.array_equal(mlcd['rec_id'].astype(str), cd['rec_id'].astype(str)), 'code15 id order mismatch'
    feat_d = pd.read_csv(f'{OUT}\\ecg_features_code15.csv')
    ok = ~np.isnan(cd['X']).any(axis=(1, 2))
    valid = feat_d[feat_d['idx'].isin(np.where(ok)[0])].sort_values('idx').reset_index(drop=True)
    idx = valid['idx'].values
    P, res = run_cohort('CODE-15%', norm(cd['X'][idx]), tab(valid),
                        mlcd['Y'][idx], cd['rec_id'][idx])
    summary['code15'] = (P, idx, res)
else:
    print('SKIP CODE-15%: features or labels missing')

out = {}
for coh, (P, idx, res) in summary.items():
    out[f'{coh}_P'] = P
    out[f'{coh}_idx'] = idx
    for cls, (m, pp, pb) in res.items():
        for variant in ['raw', 'platt', 'beta']:
            for key, val in m[variant].items():
                out[f'{coh}_{cls}_{variant}_{key}'] = val
np.savez_compressed(f'{OUT}\\fusion_multi_external.npz', **out)
print('\nsaved: fusion_multi_external.npz')
