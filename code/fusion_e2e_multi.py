# -*- coding: utf-8 -*-
"""Path B: multi-task FusionNet (K=5) on PTB-XL, breadth layer.

Architecture identical to fusion_e2e.py except head outputs K logits.
Checkpoint selected by macro-average validation AUC across the K classes.
Outputs: fusion_multi_probs.npz (internal test + val probs, Y cols), fusion_multi.pt
Gates: splits identical to binary pipeline (fold <=8 / ==9 / ==10).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA_DIR, PROJ_DIR, FIG_DIR, SUB_DIR, PATCH_DIR, NHANES_DIR, PROJ1_DIR, SHAOXING_DIR, PTBXL_DIR
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, random, pandas as pd, torch, torch.nn as nn, time, warnings
from torch.utils.data import DataLoader, TensorDataset
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED); random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

OUT = DATA_DIR
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device: {DEVICE}')

COLS = ['age', 'sex', 'hr_std', 'rr_mean', 'rmssd', 'sdnn', 'pnn50', 'cv_rr', 'rmssd_norm',
        'dom_freq', 'spec_ent', 'qrs_dur', 'pr_int', 'qt_int', 'p_amp', 'sig_std_v2', 'sig_std_i', 'n_peaks']
K = 5
NAMES = ['AF', 'RBBB', 'LBBB', 'IAVB', 'NORM']

ptb = np.load(f'{OUT}\\ptbxl_100hz.npz', allow_pickle=True)
ml = np.load(f'{OUT}\\ptbxl_multilabel.npz', allow_pickle=True)
assert list(ml['cols']) == NAMES, f'col mismatch {ml["cols"]}'
assert np.array_equal(ptb['ecg_id'], ml['ecg_id']), 'ecg_id misalignment'
assert np.array_equal(ptb['fold'], ml['fold']), 'fold misalignment'
X_p, Y_p, fold = ptb['X'], ml['Y'].astype(np.float32), ptb['fold']
assert np.array_equal(Y_p[:, 0].astype(np.int8), ptb['y'].astype(np.int8)), 'AF column != binary y'

feat_p = pd.read_csv(f'{OUT}\\ecg_features_ptbxl.csv')
feat_p['fold'] = fold[feat_p['idx'].values]

mu = X_p[fold <= 8].mean(axis=(0, 2), keepdims=True)
sd = X_p[fold <= 8].std(axis=(0, 2), keepdims=True) + 1e-8
def norm(x): return ((x - mu) / sd).astype(np.float32)

df_tr = feat_p[feat_p.fold <= 8]
imp, sc = SimpleImputer(strategy='median'), StandardScaler()
sc.fit(imp.fit_transform(df_tr[COLS]))

def tab(df):
    return sc.transform(imp.transform(df[COLS])).astype(np.float32)

def rows_of(df): return df['idx'].values
tr_idx = rows_of(df_tr)
va_df = feat_p[feat_p.fold == 9]
te_df = feat_p[feat_p.fold == 10].sort_values('idx').reset_index(drop=True)

Xtr, Ytr = norm(X_p[tr_idx]), Y_p[tr_idx]
Xva, Yva = norm(X_p[rows_of(va_df)]), Y_p[rows_of(va_df)]
Xte, Yte = norm(X_p[rows_of(te_df)]), Y_p[rows_of(te_df)]
Tab_tr, Tab_va, Tab_te = tab(df_tr), tab(va_df), tab(te_df)
print(f'Train={len(Ytr)} Val={len(Yva)} Test={len(Yte)}')
print('train prev:', {n: round(float(Ytr[:, k].mean()), 4) for k, n in enumerate(NAMES)})

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

def loader(X, T, Y, batch=64, shuffle=False):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(T), torch.from_numpy(Y))
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=0, pin_memory=True)

model = FusionNetMulti().to(DEVICE)
pw = torch.tensor([(Ytr[:, k] == 0).sum() / max(Ytr[:, k].sum(), 1) for k in range(K)], device=DEVICE)
print('pos_weight:', [round(float(w), 2) for w in pw])
crit = nn.BCEWithLogitsLoss(pos_weight=pw)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30)
tl, vl = loader(Xtr, Tab_tr, Ytr, shuffle=True), loader(Xva, Tab_va, Yva)
best_mauc, best_state = -1, None
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
    Pva = np.concatenate(ps)
    aucs = [roc_auc_score(Yva[:, k], Pva[:, k]) for k in range(K)]
    mauc = float(np.mean(aucs))
    if mauc > best_mauc:
        best_mauc = mauc
        best_state = {k2: v.cpu().clone() for k2, v in model.state_dict().items()}
    if (ep + 1) % 5 == 0:
        print(f'  ep{ep+1}: valAUC ' + ' '.join(f'{n}={a:.4f}' for n, a in zip(NAMES, aucs)) + f' macro={mauc:.4f}', flush=True)
print(f'training {time.time()-t0:.0f}s, best macro val AUC={best_mauc:.4f}')

model.load_state_dict(best_state)
model.eval()
def predict(X, T):
    ps = []
    with torch.no_grad():
        for xb, tb, _ in loader(X, T, np.zeros((len(X), K), dtype=np.float32), batch=256):
            ps.append(torch.sigmoid(model(xb.to(DEVICE), tb.to(DEVICE))).cpu().numpy())
    return np.concatenate(ps)

P_te, P_va = predict(Xte, Tab_te), predict(Xva, Tab_va)
print('\n=== internal test per-class AUC ===')
for k, n in enumerate(NAMES):
    auc = roc_auc_score(Yte[:, k], P_te[:, k])
    print(f'  {n}: AUC={auc:.4f} (prev={Yte[:, k].mean()*100:.2f}%)')
torch.save(best_state, f'{OUT}\\fusion_multi.pt')
np.savez_compressed(f'{OUT}\\fusion_multi_probs.npz',
                    Y_te=Yte, Y_va=Yva, P_te=P_te, P_va=P_va,
                    cols=np.array(NAMES), ecg_id_te=te_df['idx'].values, ecg_id_va=va_df['idx'].values)
print(f'saved: fusion_multi.pt, fusion_multi_probs.npz')
