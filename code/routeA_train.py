import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA_DIR, PROJ_DIR, FIG_DIR, SUB_DIR, PATCH_DIR, NHANES_DIR, PROJ1_DIR, SHAOXING_DIR, PTBXL_DIR
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import random
import torch
import torch.nn as nn
SEED = 42
np.random.seed(SEED); random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score, roc_curve
import time, warnings
warnings.filterwarnings('ignore')

OUT = DATA_DIR
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'设备: {DEVICE}')
if DEVICE == 'cuda':
    print(f'GPU: {torch.cuda.get_device_name(0)}, 显存: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB')

# ── 数据加载 ──
print('加载数据...')
ptb = np.load(f'{OUT}\\ptbxl_100hz.npz', allow_pickle=True)
chp = np.load(f'{OUT}\\chapman_100hz.npz', allow_pickle=True)

X_p, y_p, fold = ptb['X'], ptb['y'], ptb['fold']
X_c, y_c = chp['X'], chp['y']
ok = ~np.isnan(X_c).any(axis=(1, 2))
if not ok.all():
    print(f'Chapman过滤NaN: 去除{( ~ok).sum()}条, 剩{ok.sum()}')
    X_c, y_c = X_c[ok], y_c[ok]

# 划分
tr_mask = fold <= 8
va_mask = fold == 9
te_mask = fold == 10
Xtr, ytr = X_p[tr_mask], y_p[tr_mask]
Xva, yva = X_p[va_mask], y_p[va_mask]
Xte, yte = X_p[te_mask], y_p[te_mask]
print(f'Train={len(Xtr)} AF={ytr.mean()*100:.1f}% | Val={len(Xva)} | Test={len(Xte)} | Ext={len(X_c)}')

# 标准化（按训练集逐导联）
mu = Xtr.mean(axis=(0, 2), keepdims=True)
sd = Xtr.std(axis=(0, 2), keepdims=True) + 1e-8
def norm(x): return ((x - mu) / sd).astype(np.float32)
Xtr, Xva, Xte, Xc = norm(Xtr), norm(Xva), norm(Xte), norm(X_c)

def to_loader(X, y, batch=64, shuffle=False):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y.astype(np.float32)))
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=0, pin_memory=True)

# ── 模型 ──
class CNN1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(12, 32, 7, padding=3), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1))
    def forward(self, x):
        return self.head(self.features(x).squeeze(-1)).squeeze(-1)

class ResBlock(nn.Module):
    def __init__(self, cin, cout, stride=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(cin, cout, 3, stride=stride, padding=1), nn.BatchNorm1d(cout), nn.ReLU(),
            nn.Conv1d(cout, cout, 3, padding=1), nn.BatchNorm1d(cout))
        self.relu = nn.ReLU()
        self.down = nn.Identity() if stride == 1 and cin == cout else nn.Sequential(
            nn.Conv1d(cin, cout, 1, stride=stride), nn.BatchNorm1d(cout))
    def forward(self, x):
        return self.relu(self.conv(x) + self.down(x))

class ResNet1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(12, 32, 7, stride=2, padding=3), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2))
        self.layer1 = ResBlock(32, 64)
        self.layer2 = ResBlock(64, 128)
        self.layer3 = ResBlock(128, 128)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1))
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)
        return self.head(self.pool(x).squeeze(-1)).squeeze(-1)

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
        h = self.conv(x)
        h = h.transpose(1, 2)  # B, T, C
        h = self.encoder(h).mean(dim=1)
        return self.head(h).squeeze(-1)

# ── 训练 ──
def train_model(model, name, epochs=30, lr=1e-3, batch=64):
    print(f'\n=== {name} ===')
    model = model.to(DEVICE)
    pos = ytr.sum(); neg = len(ytr) - pos
    weight = torch.tensor([neg / pos], device=DEVICE)
    crit = nn.BCEWithLogitsLoss(pos_weight=weight)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_auc, best_state, best_ep = 0, None, 0
    tl = to_loader(Xtr, ytr, batch, shuffle=True)
    vl = to_loader(Xva, yva, batch)
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        tot, n = 0, 0
        for xb, yb in tl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            logit = model(xb)
            loss = crit(logit, yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb); n += len(xb)
        sched.step()
        # 验证
        model.eval()
        ps = []
        with torch.no_grad():
            for xb, yb in vl:
                ps.append(torch.sigmoid(model(xb.to(DEVICE))).cpu().numpy())
        p = np.concatenate(ps)
        auc = roc_auc_score(yva, p)
        if auc > best_auc:
            best_auc = auc; best_ep = ep
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if (ep + 1) % 5 == 0:
            print(f'  ep{ep+1}: loss={tot/n:.4f} valAUC={auc:.4f} best={best_auc:.4f}@{best_ep}')
    print(f'  训练耗时 {time.time()-t0:.0f}s, best valAUC={best_auc:.4f}')
    torch.save(best_state, f'{OUT}\\{name}.pt')

    # 测试
    model.load_state_dict(best_state)
    model.eval()
    def predict(X):
        ps = []
        dl = to_loader(X, np.zeros(len(X)), batch)
        with torch.no_grad():
            for xb, _ in dl:
                ps.append(torch.sigmoid(model(xb.to(DEVICE))).cpu().numpy())
        return np.concatenate(ps)
    p_te = predict(Xte)
    p_ext = predict(Xc)
    auc_te = roc_auc_score(yte, p_te)
    auc_ext = roc_auc_score(y_c, p_ext)
    # Youden阈值
    fpr, tpr, ths = roc_curve(yte, p_te)
    th = ths[np.argmax(tpr - fpr)]
    yh = (p_te >= th).astype(int)
    tn, fp, fn, tp = confusion_matrix(yte, yh).ravel()
    sens = tp/(tp+fn); spec = tn/(tn+fp)
    print(f'  内部AUC={auc_te:.4f} 外部AUC={auc_ext:.4f} dAUC={auc_te-auc_ext:.4f}')
    print(f'  Youden阈值={th:.3f} Sens={sens:.3f} Spec={spec:.3f}')
    return dict(name=name, internal_auc=round(auc_te,4), external_auc=round(auc_ext,4),
                d_auc=round(auc_te-auc_ext,4), sens=round(sens,3), spec=round(spec,3),
                p_test=p_te, p_ext=p_ext)

if __name__ == '__main__':
    results = []
    probs = {'y_test': yte, 'y_ext': y_c}
    for mname, mcls in [('CNN', CNN1D), ('ResNet', ResNet1D), ('CNNTransformer', CNNTransformer)]:
        r = train_model(mcls(), mname)
        probs[f'test_{r["name"]}'] = r.pop('p_test')
        probs[f'ext_{r["name"]}'] = r.pop('p_ext')
        results.append(r)
    import pandas as pd
    df = pd.DataFrame(results)
    print('\n=== Route A 结果 ===')
    print(df.to_string(index=False))
    df.to_csv(f'{OUT}\\table_routeA_results.csv', index=False, encoding='utf-8-sig')
    np.savez_compressed(f'{OUT}\\routeA_probs.npz', **probs)
    print(f'保存: table_routeA_results.csv, routeA_probs.npz')
