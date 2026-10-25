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
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from scipy import stats
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

def rows_of(df): return df['idx'].values
tr_idx = rows_of(df_tr)
va_df = feat_p[feat_p.fold == 9]
te_df = feat_p[feat_p.fold == 10].sort_values('idx').reset_index(drop=True)
assert np.array_equal(te_df['AF'].values, np.load(f'{OUT}\\common_probs.npz', allow_pickle=True)['y_te'])

Xtr, ytr = norm(X_p[tr_idx]), df_tr['AF'].values.astype(np.float32)
Tab_tr = T_all_p
Xva, yva = norm(X_p[rows_of(va_df)]), va_df['AF'].values.astype(np.float32)
Tab_va = sc.transform(imp.transform(va_df[COLS]))
Xte, yte = norm(X_p[rows_of(te_df)]), te_df['AF'].values.astype(np.float32)
Tab_te = sc.transform(imp.transform(te_df[COLS]))
Xcx = norm(chp['X'][valid_c['idx'].values])
ycx = valid_c['AF'].values.astype(np.float32)
Tab_cx = T_all_c
print(f'Train={len(ytr)} Val={len(yva)} Test={len(yte)} Ext={len(ycx)}')

# ── 融合网络: CNN-T卷积支路 + 表格MLP ──
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
crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / pos], device=DEVICE))
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
    if (ep + 1) % 10 == 0:
        print(f'  ep{ep+1}: valAUC={auc:.4f} best={best_auc:.4f}')
print(f'训练 {time.time()-t0:.0f}s, fold9 best AUC={best_auc:.4f}')

model.load_state_dict(best_state)
model.eval()
def predict(X, T):
    ps = []
    with torch.no_grad():
        for xb, tb, _ in loader(X, T, np.zeros(len(X)), batch=256):
            ps.append(torch.sigmoid(model(xb.to(DEVICE), tb.to(DEVICE))).cpu().numpy())
    return np.concatenate(ps)

p_te, p_cx = predict(Xte, Tab_te), predict(Xcx, Tab_cx)
ai, ae = roc_auc_score(yte, p_te), roc_auc_score(ycx, p_cx)

def boot_ci(y, p, n=1000):
    r = np.random.RandomState(42); aucs = []
    for _ in range(n):
        i = r.choice(len(y), len(y), replace=True)
        if len(np.unique(y[i])) < 2: continue
        aucs.append(roc_auc_score(y[i], p[i]))
    return np.percentile(aucs, [2.5, 97.5])

def paired_p(y, p1, p2, n=1000):
    r = np.random.RandomState(42); diffs = []
    for _ in range(n):
        i = r.choice(len(y), len(y), replace=True)
        if len(np.unique(y[i])) < 2: continue
        diffs.append(roc_auc_score(y[i], p1[i]) - roc_auc_score(y[i], p2[i]))
    diffs = np.array(diffs)
    return 2 * (1 - stats.norm.cdf(abs(diffs.mean()) / (diffs.std() + 1e-12)))

def youden(y, p):
    fpr, tpr, ths = roc_curve(y, p)
    th = ths[np.argmax(tpr - fpr)]
    tn, fp, fn, tp = confusion_matrix(y, (p >= th).astype(int)).ravel()
    return tp / (tp + fn), tn / (tn + fp)

b = np.load(f'{OUT}\\common_probs.npz', allow_pickle=True)
f2 = np.load(f'{OUT}\\fusion2_probs.npz', allow_pickle=True)
li, hi = boot_ci(yte, p_te); le, he = boot_ci(ycx, p_cx)
si, sp = youden(yte, p_te); se, ep_ = youden(ycx, p_cx)
p_int_cnn = paired_p(yte, p_te, b['int_CNNTransformer'])
p_ext_cnn = paired_p(ycx, p_cx, b['ext_CNNTransformer'])
p_ext_g3 = paired_p(ycx, p_cx, f2['sel_ext'])

# 与G3堆叠 (fold9需融合网络预测 -> 本脚本已算val probs? 补算)
p_va = predict(Xva, Tab_va)
np.savez_compressed(f'{OUT}\\fusion3_probs.npz', y_te=yte, y_ext=ycx, val=p_va, int=p_te, ext=p_cx)
print(f'\n端到端融合: 内部{ai:.4f} ({li:.3f}-{hi:.3f}) 外部{ae:.4f} ({le:.3f}-{he:.3f})')
print(f'  vs CNN-T: int P={p_int_cnn:.4f} ext P={p_ext_cnn:.4f} | vs G3: ext P={p_ext_g3:.4f}')
print(f'  Youden外部 Sens={se:.3f} Spec={ep_:.3f}')
print('保存: fusion3_probs.npz')
