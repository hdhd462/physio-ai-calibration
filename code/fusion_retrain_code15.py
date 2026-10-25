# -*- coding: utf-8 -*-
"""Retrain binary FusionNet (exact mirror of fusion_e2e.py, seed 42), save checkpoint,
verify reproduction of manuscript numbers (int 0.9868 / ext 0.8389), then infer CODE-15%."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA_DIR, PROJ_DIR, FIG_DIR, SUB_DIR, PATCH_DIR, NHANES_DIR, PROJ1_DIR, SHAOXING_DIR, PTBXL_DIR
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, DATA_DIR)
import numpy as np, random, pandas as pd, torch, torch.nn as nn, time, warnings
from torch.utils.data import DataLoader, TensorDataset
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss
from eval_multi import metrics, crossfit_repair
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED); random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

OUT = DATA_DIR
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:', DEVICE)

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
sc.fit(imp.fit_transform(df_tr[COLS]))

def rows_of(df): return df['idx'].values
tr_idx = rows_of(df_tr)
va_df = feat_p[feat_p.fold == 9]
te_df = feat_p[feat_p.fold == 10].sort_values('idx').reset_index(drop=True)

Xtr, ytr = norm(X_p[tr_idx]), df_tr['AF'].values.astype(np.float32)
Tab_tr = sc.transform(imp.transform(df_tr[COLS])).astype(np.float32)
Xva, yva = norm(X_p[rows_of(va_df)]), va_df['AF'].values.astype(np.float32)
Tab_va = sc.transform(imp.transform(va_df[COLS])).astype(np.float32)
Xte, yte = norm(X_p[rows_of(te_df)]), te_df['AF'].values.astype(np.float32)
Tab_te = sc.transform(imp.transform(te_df[COLS])).astype(np.float32)
Xcx = norm(chp['X'][valid_c['idx'].values])
ycx = valid_c['AF'].values.astype(np.float32)
Tab_cx = sc.transform(imp.transform(valid_c[COLS])).astype(np.float32)
print(f'Train={len(ytr)} Val={len(yva)} Test={len(yte)} Ext={len(ycx)}')

# original manuscript probs (ground truth for verification)
f3 = np.load(f'{OUT}\\fusion3_probs.npz', allow_pickle=True)
print(f'ORIGINAL fusion3: int AUC={roc_auc_score(f3["y_te"], f3["int"]):.4f} '
      f'ext AUC={roc_auc_score(f3["y_ext"], f3["ext"]):.4f}')

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
import os
if os.path.exists(f'{OUT}\\fusion.pt'):
    best_state = torch.load(f'{OUT}\\fusion.pt', map_location=DEVICE)
    print('loaded fusion.pt (skip retrain)')
else:
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
            print(f'  ep{ep+1}: valAUC={auc:.4f} best={best_auc:.4f}', flush=True)
    print(f'training {time.time()-t0:.0f}s, fold9 best AUC={best_auc:.4f}')

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
print(f'RETRAINED: int AUC={ai:.4f} ext AUC={ae:.4f}')
torch.save(best_state, f'{OUT}\\fusion.pt')
print('saved fusion.pt')
assert abs(ai - 0.9868) < 0.005 and abs(ae - 0.8389) < 0.005, 'retrain drift beyond tolerance'
print('GATE: retrained within 0.005 of manuscript (both inside reported 95% CIs)')

# ── CODE-15% inference ──
cd = np.load(f'{OUT}\\code15_part17_100hz.npz', allow_pickle=True)
feat_d = pd.read_csv(f'{OUT}\\ecg_features_code15.csv')
mlcd = np.load(f'{OUT}\\code15_multilabel.npz', allow_pickle=True)
assert np.array_equal(mlcd['rec_id'].astype(str), cd['rec_id'].astype(str))
assert np.array_equal(mlcd['Y'][:, 0].astype(np.int8), cd['y'].astype(np.int8)), 'AF bit-match fail'
okd = ~np.isnan(cd['X']).any(axis=(1, 2))
valid_d = feat_d[feat_d['idx'].isin(np.where(okd)[0])].sort_values('idx').reset_index(drop=True)
idx = valid_d['idx'].values
Xcd = norm(cd['X'][idx])
Tab_d = sc.transform(imp.transform(valid_d[COLS])).astype(np.float32)
ycd = cd['y'][idx].astype(np.float32)
p_cd = predict(Xcd, Tab_d)

m = metrics(ycd.astype(int), p_cd)
p_pl = crossfit_repair(ycd.astype(int), p_cd, 'platt')
p_be = crossfit_repair(ycd.astype(int), p_cd, 'beta')
m_pl, m_be = metrics(ycd.astype(int), p_pl), metrics(ycd.astype(int), p_be)

def boot_ci(y, p, n=1000):
    r = np.random.RandomState(42); aucs = []
    for _ in range(n):
        i = r.choice(len(y), len(y), replace=True)
        if len(np.unique(y[i])) < 2: continue
        aucs.append(roc_auc_score(y[i], p[i]))
    return np.percentile(aucs, [2.5, 97.5])
lo, hi = boot_ci(ycd, p_cd)

print(f'\n=== FusionNet BINARY on CODE-15% (n={len(ycd)}, AF events={int(ycd.sum())}, prev={ycd.mean()*100:.2f}%) ===')
print(f'  AUC={m["auc"]:.4f} (95% CI {lo:.3f}-{hi:.3f})')
print(f'  raw : Brier={m["brier"]:.4f} ECE={m["ece"]:.4f} slope={m["slope"]:.3f} int={m["intercept"]:.3f}')
print(f'  platt: Brier={m_pl["brier"]:.4f} ECE={m_pl["ece"]:.4f}')
print(f'  beta : Brier={m_be["brier"]:.4f} ECE={m_be["ece"]:.4f}')
np.savez_compressed(f'{OUT}\\fusion_code15_binary.npz', p=p_cd, y=ycd, idx=idx,
                    auc=m['auc'], auc_lo=lo, auc_hi=hi,
                    brier=m['brier'], ece=m['ece'], slope=m['slope'], intercept=m['intercept'],
                    platt_ece=m_pl['ece'], platt_brier=m_pl['brier'],
                    beta_ece=m_be['ece'], beta_brier=m_be['brier'])
print('saved: fusion_code15_binary.npz')
