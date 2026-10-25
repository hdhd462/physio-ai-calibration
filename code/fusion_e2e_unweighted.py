import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA_DIR, PROJ_DIR, FIG_DIR, SUB_DIR, PATCH_DIR, NHANES_DIR, PROJ1_DIR, SHAOXING_DIR, PTBXL_DIR
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, random, json, time, warnings
import pandas as pd
import torch
import torch.nn as nn

SEED = 42
np.random.seed(SEED); random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
from torch.utils.data import DataLoader, TensorDataset
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
from scipy.special import logit
import warnings
warnings.filterwarnings('ignore')

OUT = DATA_DIR
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device: {DEVICE}', flush=True)

ptb = np.load(f'{OUT}\\ptbxl_100hz.npz', allow_pickle=True)
chp = np.load(f'{OUT}\\chapman_100hz.npz', allow_pickle=True)
fold = ptb['fold']
X_p = ptb['X']
ok = ~np.isnan(chp['X']).any(axis=(1, 2))

feat_p = pd.read_csv(f'{OUT}\\ecg_features_ptbxl.csv')
feat_p['fold'] = fold[feat_p['idx'].values]
feat_c = pd.read_csv(f'{OUT}\\ecg_features_chapman.csv')
valid_c = feat_c[feat_c['idx'].isin(np.where(ok)[0])].sort_values('idx').reset_index(drop=True)

mu = X_p[fold <= 8].mean(axis=(0, 2), keepdims=True)
sd = X_p[fold <= 8].std(axis=(0, 2), keepdims=True) + 1e-8
def norm(x): return ((x - mu) / sd).astype(np.float32)

COLS = ['age', 'sex', 'hr_std', 'rr_mean', 'rmssd', 'sdnn', 'pnn50', 'cv_rr', 'rmssd_norm',
        'dom_freq', 'spec_ent', 'qrs_dur', 'pr_int', 'qt_int', 'p_amp', 'sig_std_v2', 'sig_std_i', 'n_peaks']
df_tr = feat_p[feat_p.fold <= 8]
imp, sc = SimpleImputer(strategy='median'), StandardScaler()
T_all_p = sc.fit_transform(imp.fit_transform(df_tr[COLS]))
T_all_c = sc.transform(imp.transform(valid_c[COLS]))

tr_idx = df_tr['idx'].values
va_df = feat_p[feat_p.fold == 9]
te_df = feat_p[feat_p.fold == 10].sort_values('idx').reset_index(drop=True)

Xtr, ytr = norm(X_p[tr_idx]), df_tr['AF'].values.astype(np.float32)
Tab_tr = T_all_p
Xva, yva = norm(X_p[va_df['idx'].values]), va_df['AF'].values.astype(np.float32)
Tab_va = sc.transform(imp.transform(va_df[COLS]))
Xte, yte = norm(X_p[te_df['idx'].values]), te_df['AF'].values.astype(np.float32)
Tab_te = sc.transform(imp.transform(te_df[COLS]))
Xcx = norm(chp['X'][valid_c['idx'].values])
ycx = valid_c['AF'].values.astype(np.float32)
Tab_cx = T_all_c
print(f'Train={len(ytr)} (AF={int(ytr.sum())}) Val={len(yva)} Test={len(yte)} Ext={len(ycx)} (AF={int(ycx.sum())})', flush=True)


class FusionNet(nn.Module):
    def __init__(self, d_model=64, nhead=4, nlayers=2, ntab=18):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(12, 32, 7, stride=2, padding=3), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, d_model, 5, stride=2, padding=2), nn.BatchNorm1d(d_model), nn.ReLU())
        self.pos = nn.Parameter(torch.randn(1, d_model, 250) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=128,
                                            dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=nlayers)
        self.tab = nn.Sequential(nn.Linear(ntab, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 16), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(d_model + 16, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1))

    def forward(self, x, t):
        h = self.conv(x).transpose(1, 2)
        h = self.encoder(h).mean(dim=1)
        f = self.tab(t)
        return self.head(torch.cat([h, f], dim=1)).squeeze(-1)


def loader(X, T, y, batch=64, shuffle=False):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(T.astype(np.float32)), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=0, pin_memory=True)


model = FusionNet().to(DEVICE)
pos, neg = ytr.sum(), len(ytr) - ytr.sum()
print(f'pos_weight = 1.0 (unweighted); per-class_weighted would be {neg / pos:.3f}', flush=True)
crit = nn.BCEWithLogitsLoss()
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30)
tl, vl = loader(Xtr, Tab_tr, ytr, shuffle=True), loader(Xva, Tab_va, yva)
best_auc, best_state = 0, None
t0 = time.time()
for ep in range(30):
    model.train()
    for xb, tb, yb in tl:
        xb, tb, yb = xb.to(DEVICE), tb.to(DEVICE), yb.to(DEVICE)
        opt.zero_grad()
        loss = crit(model(xb, tb), yb)
        loss.backward()
        opt.step()
    sched.step()
    model.eval()
    ps = []
    with torch.no_grad():
        for xb, tb, _ in vl:
            ps.append(torch.sigmoid(model(xb.to(DEVICE), tb.to(DEVICE))).cpu().numpy())
    auc = roc_auc_score(yva, np.concatenate(ps))
    if auc > best_auc:
        best_auc = auc
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    if (ep + 1) % 5 == 0:
        print(f'  ep{ep+1}: valAUC={auc:.4f} best={best_auc:.4f} ({time.time()-t0:.0f}s)', flush=True)
print(f'trained in {time.time()-t0:.0f}s, fold9 best AUC={best_auc:.4f}', flush=True)

model.load_state_dict(best_state)
model.eval()


def predict(X, T):
    ps = []
    with torch.no_grad():
        for xb, tb, _ in loader(X, T, np.zeros(len(X)), batch=256):
            ps.append(torch.sigmoid(model(xb.to(DEVICE), tb.to(DEVICE))).cpu().numpy())
    return np.concatenate(ps)


p_te = predict(Xte, Tab_te)
p_cx = predict(Xcx, Tab_cx)


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    tot = 0.0
    n = len(p)
    for b in range(bins):
        m = idx == b
        if m.sum() == 0:
            continue
        tot += m.sum() / n * abs(y[m].mean() - p[m].mean())
    return float(tot)


def slope_int(y, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    lr = LogisticRegression(C=1e6, solver='lbfgs', max_iter=2000)
    lr.fit(logit(p).reshape(-1, 1), y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def met(y, p):
    s, i = slope_int(y, p)
    return dict(n=int(len(y)), events=int(y.sum()), prev=float(y.mean()),
                mean_pred=float(np.mean(p)), median_pred=float(np.median(p)),
                pct_ge_09=float(np.mean(p >= 0.9)),
                auc=float(roc_auc_score(y, p)), brier=float(brier_score_loss(y, p)),
                ece=ece(y, p), slope=s, intercept=i)


np.savez_compressed(f'{OUT}\\fusion_unweighted_probs.npz',
                    y_te=yte, y_ext=ycx, int=p_te, ext=p_cx)

ref = np.load(f'{OUT}\\common_probs.npz', allow_pickle=True)
summary = {
    'unweighted': {'internal': met(yte, p_te), 'chapman': met(ycx, p_cx)},
    'weighted': {'internal': met(yte, ref['int_FusionNet'].astype(float)),
                 'chapman': met(ycx, ref['ext_FusionNet'].astype(float))},
    'val_auc': float(best_auc),
}
with open(f'{OUT}\\unweighted_vs_weighted.json', 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2)

print()
print('=== weighted vs unweighted ===')
for k in ['weighted', 'unweighted']:
    for c in ['internal', 'chapman']:
        m = summary[k][c]
        print(f"{k:11s} {c:9s} AUC={m['auc']:.4f} Brier={m['brier']:.4f} ECE={m['ece']:.4f} "
              f"slope={m['slope']:.3f} mean_pred={m['mean_pred']:.4f} pct>=0.9={100*m['pct_ge_09']:.1f}%")
print('saved: fusion_unweighted_probs.npz, unweighted_vs_weighted.json')
