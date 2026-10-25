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
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import TwoSlopeNorm

rcParams['font.family'] = 'Arial'
rcParams['axes.linewidth'] = 0.8
rcParams['pdf.fonttype'] = 42
rcParams['font.size'] = 11

OUT = DATA_DIR
FIG = FIG_DIR
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'设备: {DEVICE}')

# ── 模型定义（与routeA_train.py一致） ──
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

# ── 数据 ──
ptb = np.load(f'{OUT}\\ptbxl_100hz.npz', allow_pickle=True)
chp = np.load(f'{OUT}\\chapman_100hz.npz', allow_pickle=True)
fold = ptb['fold']
Xtr = ptb['X'][fold <= 8]
mu = Xtr.mean(axis=(0, 2), keepdims=True)
sd = Xtr.std(axis=(0, 2), keepdims=True) + 1e-8
def norm(x): return ((x - mu) / sd).astype(np.float32)

X_c, y_c = chp['X'], chp['y']
ok = ~np.isnan(X_c).any(axis=(1, 2))
X_c, y_c = X_c[ok], y_c[ok]
X_c = norm(X_c)

# 外部测试子集：100 AF + 100 非AF（用于全局重要性）
rng = np.random.RandomState(0)
i_af = np.where(y_c == 1)[0]
i_na = np.where(y_c == 0)[0]
sel_af = rng.choice(i_af, 50, replace=False)
sel_na = rng.choice(i_na, 50, replace=False)
sel = np.concatenate([sel_af, sel_na])
X_sub = torch.from_numpy(X_c[sel])

# 代表样本：预测概率最接近中位数者
# ── 模型 ──
model = CNN1D().to(DEVICE)
model.load_state_dict(torch.load(f'{OUT}\\CNN.pt', map_location=DEVICE))
model.eval()

class Wrapper(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
    def forward(self, x):
        return self.m(x).unsqueeze(1)

model_w = Wrapper(model)

def f(x):
    return model(x).unsqueeze(1)

# GradientSHAP（baseline = 32条随机训练样本）
bg_idx = rng.choice(len(Xtr), 16, replace=False)
baseline = torch.from_numpy(norm(Xtr[bg_idx]))
print('计算GradientSHAP (GradientExplainer)...')
gs = shap.GradientExplainer(model_w, baseline.to(DEVICE))
sv = gs.shap_values(X_sub.to(DEVICE))          # (200, 12, 1000)
if hasattr(sv, 'detach'):
    sv = sv.detach().cpu().numpy()
sv = np.asarray(sv, dtype=np.float32)
if sv.ndim == 4:
    sv = sv[..., 0]
print(f'SHAP形状: {sv.shape}')

# 全局重要性：每导联 mean|SHAP|
leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
imp = np.abs(sv).mean(axis=(0, 2))   # (12,)

# 代表样本：基于RR不规则度（lead II R峰检测）
from scipy.signal import find_peaks
def rr_irreg(sig):
    pk, _ = find_peaks(sig, distance=40)
    if len(pk) < 6:
        return np.inf, np.nan
    rr = np.diff(pk) / 100.0
    return rr.std() / (rr.mean() + 1e-9), rr.mean()

with torch.no_grad():
    probs = torch.sigmoid(model(X_sub.to(DEVICE))).cpu().numpy()
p_af = probs[:50]         # AF样本（sel前半）
p_na = probs[50:]         # 非AF样本

# AF候选：RR最不规则的25例中取中位概率者
irr_af = np.array([rr_irreg(X_c[sel[i], 1, :])[0] for i in range(50)])
cand_af = np.argsort(irr_af)[:25]
iaf = int(cand_af[np.argmin(np.abs(p_af[cand_af] - np.median(p_af[cand_af])))])

# 非AF候选：窦性心律（RR 0.6-1.0s）且最规则的25例中取中位概率者
stats_na = [rr_irreg(X_c[sel[50 + i], 1, :]) for i in range(50)]
irr_na = np.array([s[0] for s in stats_na])
meanrr_na = np.array([s[1] for s in stats_na])
cand_na = np.where((meanrr_na > 0.6) & (meanrr_na < 1.0))[0]
if len(cand_na) < 5:
    cand_na = np.argsort(irr_na)[:25]
else:
    cand_na = cand_na[np.argsort(irr_na[cand_na])[:25]]
ina = 50 + int(cand_na[np.argmin(np.abs(p_na[cand_na] - np.median(p_na[cand_na])))])
print(f'代表AF样本: 子集#{iaf} p={probs[iaf]:.3f} irr={irr_af[iaf]:.3f} | '
      f'非AF: 子集#{ina} p={probs[ina]:.3f} irr={irr_na[ina-50]:.3f} meanRR={meanrr_na[ina-50]:.2f}s')

# ── 绘图 ──
fig = plt.figure(figsize=(14, 7.5), dpi=300)
gs_grid = fig.add_gridspec(2, 3, height_ratios=[1, 3.2], width_ratios=[1, 1.35, 1.35],
                           hspace=0.32, wspace=0.25)

# Panel A: 每导联全局重要性
axA = fig.add_subplot(gs_grid[:, 0])
order = np.argsort(imp)
axA.barh(np.arange(12), imp[order], color=['#5B9BD5' if l not in ('II', 'V1') else '#ED7D31'
                                           for l in np.array(leads)[order]], height=0.7)
axA.set_yticks(np.arange(12))
axA.set_yticklabels(np.array(leads)[order], fontsize=9)
axA.set_xlabel('Mean |SHAP value|', fontsize=11)
axA.set_title('(A) Global lead importance', fontsize=12, fontweight='bold', loc='left')
axA.spines['top'].set_visible(False)
axA.spines['right'].set_visible(False)

# Panel B/C: 代表样本 SHAP热图（上方lead II原始波形）
clip = np.percentile(np.abs(sv), 99)
norm_c = TwoSlopeNorm(vmin=-clip, vcenter=0, vmax=clip)
for col, (idx, tag, ttl) in enumerate([(ina, 'B', 'Non-AF representative'), (iaf, 'C', 'AF representative')], start=1):
    axt = fig.add_subplot(gs_grid[0, col])
    axh = fig.add_subplot(gs_grid[1, col])
    sig = X_c[sel[idx], 1, :]                 # lead II 原始（标准化后）
    t = np.arange(1000) / 100
    axt.plot(t, sig, color='#333', lw=0.7)
    axt.set_xlim(0, 10)
    axt.set_ylim(np.percentile(sig, 0.5), np.percentile(sig, 99.5))
    axt.set_xticks([])
    axt.set_ylabel('II', fontsize=9, rotation=0, labelpad=8, ha='right', va='center')
    axt.tick_params(labelsize=8)
    axt.spines['top'].set_visible(False)
    axt.spines['right'].set_visible(False)
    axt.set_title(f'({tag}) {ttl} (p={probs[idx]:.2f})', fontsize=12, fontweight='bold', loc='left')

    im = axh.imshow(sv[idx], aspect='auto', cmap='RdBu_r', norm=norm_c,
                    extent=[0, 10, 11.5, -0.5], interpolation='nearest')
    axh.set_yticks(np.arange(12))
    axh.set_yticklabels(leads, fontsize=7)
    axh.set_xlabel('Time (s)', fontsize=11)
    axh.set_ylabel('')
    axh.tick_params(labelsize=8)

cb = fig.colorbar(im, ax=[fig.axes[2], fig.axes[4]], fraction=0.03, pad=0.02)
cb.set_label('SHAP value (logit)', fontsize=9)
cb.ax.tick_params(labelsize=8)

plt.savefig(f'{FIG}\\figure8_shap.png', dpi=600, bbox_inches='tight')
plt.savefig(f'{FIG}\\figure8_shap.pdf', bbox_inches='tight')
plt.close()
print('Saved: figure8_shap.png/pdf')

# 保存导联重要性
import pandas as pd
pd.DataFrame({'Lead': leads, 'MeanAbsSHAP': imp}).to_csv(
    f'{OUT}\\shap_lead_importance.csv', index=False, encoding='utf-8-sig')
print(pd.DataFrame({'Lead': leads, 'MeanAbsSHAP': np.round(imp, 5)}).sort_values(
    'MeanAbsSHAP', ascending=False).to_string(index=False))