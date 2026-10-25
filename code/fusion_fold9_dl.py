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
from sklearn.metrics import roc_auc_score
import time, warnings
warnings.filterwarnings('ignore')

OUT = DATA_DIR
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'设备: {DEVICE}')

ptb = np.load(f'{OUT}\\ptbxl_100hz.npz', allow_pickle=True)
X_p, y_p, fold = ptb['X'], ptb['y'], ptb['fold']
tr, va = fold <= 8, fold == 9
Xtr, ytr, Xva, yva = X_p[tr], y_p[tr], X_p[va], y_p[va]
mu = X_p[tr].mean(axis=(0, 2), keepdims=True)
sd = X_p[tr].std(axis=(0, 2), keepdims=True) + 1e-8
def norm(x): return ((x - mu) / sd).astype(np.float32)
Xtr, Xva = norm(Xtr), norm(Xva)

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
        h = self.conv(x).transpose(1, 2)
        h = self.encoder(h).mean(dim=1)
        return self.head(h).squeeze(-1)

def to_loader(X, y, batch=64, shuffle=False):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y.astype(np.float32)))
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=0, pin_memory=True)

def train_predict(model, name, epochs=30, lr=1e-3, batch=64):
    print(f'\n=== {name} (fold9预测) ===')
    model = model.to(DEVICE)
    pos, neg = ytr.sum(), len(ytr) - ytr.sum()
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / pos], device=DEVICE))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    tl, vl = to_loader(Xtr, ytr, batch, shuffle=True), to_loader(Xva, yva, batch)
    best_auc, best_state = 0, None
    t0 = time.time()
    for ep in range(epochs):
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
        if (ep + 1) % 10 == 0:
            print(f'  ep{ep+1}: valAUC={auc:.4f} best={best_auc:.4f}')
    print(f'  {time.time()-t0:.0f}s, best valAUC={best_auc:.4f}')
    model.load_state_dict(best_state)
    model.eval()
    ps = []
    with torch.no_grad():
        for xb, _ in to_loader(Xva, yva, batch):
            ps.append(torch.sigmoid(model(xb.to(DEVICE))).cpu().numpy())
    return np.concatenate(ps)

out = {'y_val': y_p[va]}
for nm, cls in [('CNN', CNN1D), ('ResNet', ResNet1D), ('CNNTransformer', CNNTransformer)]:
    out[f'val_{nm}'] = train_predict(cls(), nm)

np.savez_compressed(f'{OUT}\\fusion_dl9.npz', **out)
print('\n保存: fusion_dl9.npz', {k: v.shape for k, v in out.items()})
