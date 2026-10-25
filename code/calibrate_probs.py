import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA_DIR, PROJ_DIR, FIG_DIR, SUB_DIR, PATCH_DIR, NHANES_DIR, PROJ1_DIR, SHAOXING_DIR, PTBXL_DIR
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import random
import pandas as pd
import torch
import torch.nn as nn
SEED = 42
np.random.seed(SEED); random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.calibration import calibration_curve
import time, warnings
warnings.filterwarnings('ignore')

OUT = DATA_DIR
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'设备: {DEVICE}')

# ── 数据 ──
ptb = np.load(f'{OUT}\\ptbxl_100hz.npz', allow_pickle=True)
chp = np.load(f'{OUT}\\chapman_100hz.npz', allow_pickle=True)
fold = ptb['fold']
X_p, y_p = ptb['X'], ptb['y']
X_c, y_c = chp['X'], chp['y']
ok = ~np.isnan(X_c).any(axis=(1, 2))
X_c, y_c = X_c[ok], y_c[ok]

tr, va, te = fold <= 8, fold == 9, fold == 10
mu = X_p[tr].mean(axis=(0, 2), keepdims=True)
sd = X_p[tr].std(axis=(0, 2), keepdims=True) + 1e-8
def norm(x): return ((x - mu) / sd).astype(np.float32)

# ── 模型 ──
class CNNTransformer(nn.Module):
    def __init__(self, d_model=64, nhead=4, nlayers=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(12, 32, 7, stride=2, padding=3), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, d_model, 5, stride=2, padding=2), nn.BatchNorm1d(d_model), nn.ReLU())
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=128,
                                           dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=nlayers)
        self.head = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1))
    def forward(self, x):
        h = self.conv(x).transpose(1, 2)
        h = self.encoder(h).mean(dim=1)
        return self.head(h).squeeze(-1)

def to_loader(X, y, batch=64, shuffle=False):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y.astype(np.float32)))
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=0, pin_memory=True)

Xtr, ytr = norm(X_p[tr]), y_p[tr]
Xva, yva = norm(X_p[va]), y_p[va]
Xte, yte = norm(X_p[te]), y_p[te]
Xcx = norm(X_c)

# ── DL训练（folds1-8训练, fold9早停+校准） ──
print('训练 CNNTransformer...')
model = CNNTransformer().to(DEVICE)
pos, neg = ytr.sum(), len(ytr) - ytr.sum()
crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / pos], device=DEVICE))
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30)
tl, vl = to_loader(Xtr, ytr, shuffle=True), to_loader(Xva, yva)
best_auc, best_state = 0, None
t0 = time.time()
for ep in range(30):
    model.train()
    for xb, yb in tl:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        opt.zero_grad()
        loss = crit(model(xb), yb)
        loss.backward()
        opt.step()
    sched.step()
    model.eval()
    ps = []
    with torch.no_grad():
        for xb, _ in vl:
            ps.append(torch.sigmoid(model(xb.to(DEVICE))).cpu().numpy())
    auc = roc_auc_score(yva, np.concatenate(ps))
    if auc > best_auc:
        best_auc = auc
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
print(f'  完成 {time.time()-t0:.0f}s best valAUC={best_auc:.4f}')

model.load_state_dict(best_state)
model.eval()
def predict_dl(X, batch=256):
    ps = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(X[i:i+batch]).to(DEVICE)
            ps.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(ps)

p9_dl = predict_dl(Xva)
pte_dl = predict_dl(Xte)
pcx_dl = predict_dl(Xcx)

# ── LR M3（folds1-8训练, fold9校准） ──
print('训练 LR M3...')
feat = pd.read_csv(f'{OUT}\\ecg_features_ptbxl.csv')
chf = pd.read_csv(f'{OUT}\\ecg_features_chapman.csv')
feat['fold'] = fold[feat['idx'].values]
COLS = ['age', 'sex', 'hr_std', 'rr_mean', 'rmssd', 'sdnn', 'pnn50', 'cv_rr',
        'rmssd_norm', 'dom_freq', 'spec_ent',
        'qrs_dur', 'pr_int', 'qt_int', 'p_amp', 'sig_std_v2', 'sig_std_i', 'n_peaks']
def lr_probs(df_tr, dfs):
    imp = SimpleImputer(strategy='median')
    sc = StandardScaler()
    Xtr_ = sc.fit_transform(imp.fit_transform(df_tr[COLS]))
    ytr_ = df_tr['AF'].values
    w = np.where(ytr_ == 1, (1 - ytr_.mean()) / ytr_.mean(), ytr_.mean() / (1 - ytr_.mean()))
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr_, ytr_, sample_weight=w)
    return [clf.predict_proba(sc.transform(imp.transform(d[COLS])))[:, 1] for d in dfs]

df_tr = feat[feat.fold <= 8]
p9_lr, pte_lr, pcx_lr = lr_probs(df_tr, [feat[feat.fold == 9], feat[feat.fold == 10], chf])
y9_lr = feat[feat.fold == 9]['AF'].values
yte_lr = feat[feat.fold == 10]['AF'].values
y_ext_lr = chf['AF'].values

# ── Platt校准（fold9, 自然患病率） ──
def platt(p_cal, y_cal, *args):
    lg = np.log(p_cal / (1 - p_cal) + 1e-12).reshape(-1, 1)
    c = LogisticRegression(C=1e6).fit(lg, y_cal)
    out = []
    for p in args:
        out.append(c.predict_proba(np.log(p / (1 - p) + 1e-12).reshape(-1, 1))[:, 1])
    return out

cte_dl, ccx_dl = platt(p9_dl, y_p[va], pte_dl, pcx_dl)
cte_lr, ccx_lr = platt(p9_lr, y9_lr, pte_lr, pcx_lr)
print(f'DL AUC校准前后(外部): {roc_auc_score(y_c, pcx_dl):.4f} -> {roc_auc_score(y_c, ccx_dl):.4f}')
print(f'LR AUC校准前后(外部): {roc_auc_score(y_ext_lr, pcx_lr):.4f} -> {roc_auc_score(y_ext_lr, ccx_lr):.4f}')
print(f'DL外部概率范围: {np.percentile(ccx_dl, [5,50,95]).round(3)}')

np.savez_compressed(f'{OUT}\\calibrated_probs.npz',
                    y_val=y_p[va], y_int=yte, y_int_lr=yte_lr, y_ext=y_c, y_ext_lr=y_ext_lr,
                    val_dl=p9_dl, int_dl=pte_dl, ext_dl=pcx_dl,
                    cal_int_dl=cte_dl, cal_ext_dl=ccx_dl,
                    val_lr=p9_lr, int_lr=pte_lr, ext_lr=pcx_lr,
                    cal_int_lr=cte_lr, cal_ext_lr=ccx_lr)
print('保存: calibrated_probs.npz')