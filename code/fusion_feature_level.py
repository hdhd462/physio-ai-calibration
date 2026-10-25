import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA_DIR, PROJ_DIR, FIG_DIR, SUB_DIR, PATCH_DIR, NHANES_DIR, PROJ1_DIR, SHAOXING_DIR, PTBXL_DIR
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

OUT = DATA_DIR
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ── 模型定义 (与routeA一致) ──
class CNN1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(12, 32, 7, padding=3), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1))
    def forward(self, x): return self.head(self.features(x).squeeze(-1)).squeeze(-1)
    def embed(self, x): return self.features(x).squeeze(-1)

class ResBlock(nn.Module):
    def __init__(self, cin, cout, stride=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(cin, cout, 3, stride=stride, padding=1), nn.BatchNorm1d(cout), nn.ReLU(),
            nn.Conv1d(cout, cout, 3, padding=1), nn.BatchNorm1d(cout))
        self.relu = nn.ReLU()
        self.down = nn.Identity() if stride == 1 and cin == cout else nn.Sequential(
            nn.Conv1d(cin, cout, 1, stride=stride), nn.BatchNorm1d(cout))
    def forward(self, x): return self.relu(self.conv(x) + self.down(x))

class ResNet1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(12, 32, 7, stride=2, padding=3), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2))
        self.layer1 = ResBlock(32, 64); self.layer2 = ResBlock(64, 128); self.layer3 = ResBlock(128, 128)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1))
    def forward(self, x):
        x = self.stem(x); x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)
        return self.head(self.pool(x).squeeze(-1)).squeeze(-1)
    def embed(self, x):
        x = self.stem(x); x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)
        return self.pool(x).squeeze(-1)

class CNNTransformer(nn.Module):
    def __init__(self, d_model=64, nhead=4, nlayers=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(12, 32, 7, stride=2, padding=3), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, d_model, 5, stride=2, padding=2), nn.BatchNorm1d(d_model), nn.ReLU())
        self.pos = nn.Parameter(torch.randn(1, d_model, 250) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=128,
                                            dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=nlayers)
        self.head = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1))
    def forward(self, x):
        h = self.conv(x).transpose(1, 2)
        h = self.encoder(h).mean(dim=1)
        return self.head(h).squeeze(-1)
    def embed(self, x):
        h = self.conv(x).transpose(1, 2)
        return self.encoder(h).mean(dim=1)

# ── 数据 ──
ptb = np.load(f'{OUT}\\ptbxl_100hz.npz', allow_pickle=True)
chp = np.load(f'{OUT}\\chapman_100hz.npz', allow_pickle=True)
fold = ptb['fold']
ok = ~np.isnan(chp['X']).any(axis=(1, 2))
X_p, y_p = ptb['X'], ptb['y']
X_c, y_c = chp['X'][ok], chp['y'][ok]

mu = X_p[fold <= 8].mean(axis=(0, 2), keepdims=True)
sd = X_p[fold <= 8].std(axis=(0, 2), keepdims=True) + 1e-8
def norm(x): return ((x - mu) / sd).astype(np.float32)

feat_p = pd.read_csv(f'{OUT}\\ecg_features_ptbxl.csv')
feat_p['fold'] = fold[feat_p['idx'].values]
feat_c = pd.read_csv(f'{OUT}\\ecg_features_chapman.csv')

# Chapman特征行中X有效的 (对齐共同队列)
ok_idx = np.where(ok)[0]
valid_c = feat_c[feat_c['idx'].isin(ok_idx)].reset_index(drop=True)

f9 = feat_p[feat_p.fold == 9].sort_values('idx').reset_index(drop=True)
f10 = feat_p[feat_p.fold == 10].sort_values('idx').reset_index(drop=True)
valid_c = valid_c.sort_values('idx').reset_index(drop=True)
print(f'fold9={len(f9)} fold10={len(f10)} chapman={len(valid_c)}')

def embed_all(model, X, name, batch=256):
    model = model.to(DEVICE).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(X[i:i+batch]).to(DEVICE)
            out.append(model.embed(xb).cpu().numpy())
    e = np.concatenate(out)
    print(f'  {name} embed: {e.shape}')
    return e

print('提取倒数层特征...')
EMB = {}
for nm, cls in [('CNN', CNN1D), ('ResNet', ResNet1D), ('CNNT', CNNTransformer)]:
    m = cls().to(DEVICE)
    sd_state = torch.load(f'{OUT}\\{nm if nm != "CNNT" else "CNNTransformer"}.pt', map_location=DEVICE)
    m.load_state_dict(sd_state)
    EMB[nm] = {
        'f9': embed_all(m, norm(X_p[f9['idx'].values]), nm),
        'f10': embed_all(m, norm(X_p[f10['idx'].values]), nm),
        'fc': embed_all(m, norm(chp['X'][valid_c['idx'].values]), nm),
    }

# ── 表格特征 (folds1-8统计量) ──
COLS = ['age', 'sex', 'hr_std', 'rr_mean', 'rmssd', 'sdnn', 'pnn50', 'cv_rr', 'rmssd_norm',
        'dom_freq', 'spec_ent', 'qrs_dur', 'pr_int', 'qt_int', 'p_amp', 'sig_std_v2', 'sig_std_i', 'n_peaks']
df_tr = feat_p[feat_p.fold <= 8]
imp, sc = SimpleImputer(strategy='median'), StandardScaler()
imp.fit(df_tr[COLS]); sc.fit(imp.transform(df_tr[COLS]))
TAB = {'f9': sc.transform(imp.transform(f9[COLS])),
       'f10': sc.transform(imp.transform(f10[COLS])),
       'fc': sc.transform(imp.transform(valid_c[COLS]))}

y9, yte, ycx = f9['AF'].values, f10['AF'].values, valid_c['AF'].values

def assemble(keys, part):  # keys: ['CNN','ResNet','CNNT'] 或 ['TAB']
    if keys == ['TAB']: return TAB[part]
    return np.column_stack([EMB[k][part] for k in keys])

VAR = {'G1': ['CNN', 'ResNet', 'CNNT'],                # 仅DL嵌入
       'G2': ['TAB'],                                   # 仅表格
       'G3': ['CNN', 'ResNet', 'CNNT', 'TAB']}          # DL嵌入+表格 (全融合)
VAR['G3'] = ['DL+TAB']

def assemble2(spec, part):
    if spec == ['DL+TAB']:
        return np.column_stack([EMB['CNN'][part], EMB['ResNet'][part], EMB['CNNT'][part], TAB[part]])
    return assemble(spec, part)

# ── fold9 CV选型 (C网格+变体) ──
cv_best = {}
for name, spec in VAR.items():
    X = assemble2(spec, 'f9')
    best_c, best_auc = None, 0
    for C in [0.01, 0.1, 1.0, 10.0]:
        aucs = []
        skf = StratifiedKFold(5, shuffle=True, random_state=42)
        for tr_, te_ in skf.split(X, y9):
            scz = StandardScaler().fit(X[tr_])
            clf = LogisticRegression(C=C, max_iter=2000).fit(scz.transform(X[tr_]), y9[tr_])
            p = clf.predict_proba(scz.transform(X[te_]))[:, 1]
            aucs.append(roc_auc_score(y9[te_], p))
        m = np.mean(aucs)
        if m > best_auc: best_auc, best_c = m, C
    cv_best[name] = (best_auc, best_c)
    print(f'{name}: fold9-CV AUC={best_auc:.4f} (C={best_c}) dim={X.shape[1]}')

sel = max(cv_best, key=lambda k: cv_best[k][0])
print(f'CV选型: {sel}')

# ── 最终: fold9全量训练 → fold10/Chapman ──
def run_final(spec, C):
    X9, Xt, Xc = assemble2(spec, 'f9'), assemble2(spec, 'f10'), assemble2(spec, 'fc')
    scz = StandardScaler().fit(X9)
    clf = LogisticRegression(C=C, max_iter=2000).fit(scz.transform(X9), y9)
    return clf.predict_proba(scz.transform(Xt))[:, 1], clf.predict_proba(scz.transform(Xc))[:, 1]

# 基线 (共同对齐的已发表预测, 本脚本的fold10/Chapman行=f共同队列)
b = np.load(f'{OUT}\\common_probs.npz', allow_pickle=True)
assert np.array_equal(b['y_te'], yte), 'fold10标签不一致'
assert np.array_equal(b['y_ext'], ycx), 'Chapman标签不一致'

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

rows = []
for nm in ['CNNTransformer', 'M3']:
    ai = roc_auc_score(yte, b['int_' + nm]); ae = roc_auc_score(ycx, b['ext_' + nm])
    rows.append(dict(Model=nm, Dim='', C='', Int=round(ai, 4), Ext=round(ae, 4)))
for name in ['G1', 'G2', 'G3']:
    auc9, C = cv_best[name]
    pt, pe = run_final(VAR[name], C)
    ai, ae = roc_auc_score(yte, pt), roc_auc_score(ycx, pe)
    li, hi = boot_ci(yte, pt); le, he = boot_ci(ycx, pe)
    si, sp = youden(yte, pt); se, ep = youden(ycx, pe)
    p_int = paired_p(yte, pt, b['int_CNNTransformer'])
    p_ext = paired_p(ycx, pe, b['ext_CNNTransformer'])
    mark = '*' if name == sel else ''
    rows.append(dict(Model=name + mark, Dim=assemble2(VAR[name], 'f9').shape[1], C=C,
                     Int=round(ai, 4), Ext=round(ae, 4),
                     Int_CI=f'{li:.3f}-{hi:.3f}', Ext_CI=f'{le:.3f}-{he:.3f}',
                     Ext_Sens=round(se, 3), Ext_Spec=round(ep, 3),
                     P_int=f'{p_int:.4f}', P_ext=f'{p_ext:.4f}'))
    print(f'{name}{mark}: 内部{ai:.4f} 外部{ae:.4f} | vs CNN-T int P={p_int:.4f} ext P={p_ext:.4f}')
    if name == sel:
        np.savez_compressed(f'{OUT}\\fusion2_probs.npz', y_te=yte, y_ext=ycx, sel_int=pt, sel_ext=pe)

df = pd.DataFrame(rows)
print('\n=== 特征级融合结果 (★=fold9 CV选型) ===')
print(df.to_string(index=False))
df.to_csv(f'{OUT}\\fusion2_results.csv', index=False, encoding='utf-8-sig')
print('保存: fusion2_results.csv, fusion2_probs.npz')
