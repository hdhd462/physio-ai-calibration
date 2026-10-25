import sys
sys.stdout.reconfigure(encoding='utf-8')
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA_DIR, PROJ_DIR, FIG_DIR, SUB_DIR, PATCH_DIR, NHANES_DIR, PROJ1_DIR, SHAOXING_DIR, PTBXL_DIR
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.family'] = 'Arial'
rcParams['axes.linewidth'] = 0.8
rcParams['pdf.fonttype'] = 42
rcParams['font.size'] = 11

OUT = DATA_DIR
FIG = FIG_DIR

d = np.load(f'{OUT}\\routeB_probs.npz')

# 用外部验证集（Chapman, n=22218, AF=7.9%）
y = d['ext_y']
p1 = d['ext_M1']
p2 = d['ext_M2']
p3 = d['ext_M3']
prev = y.mean()

# ── 先验校正（prior correction）──
# routeB_logistic.py 用类别平衡权重 w1=(1-pi)/pi, w0=pi/(1-pi) 拟合；
# predict_proba 输出带 +log(w1/w0) 的先验偏移（不是人群概率）。
# 单调变换：AUC/sens/spec/ROC/表格完全不受影响；校准与 DCA 必须用真实概率。
ptb = pd.read_csv(f'{OUT}\\ecg_features_ptbxl.csv', usecols=['AF', 'idx'])
fold = np.load(f'{OUT}\\ptbxl_100hz.npz')['fold']
fold_sel = fold[ptb['idx'].to_numpy()]
train = fold_sel <= 9
pi = ptb['AF'].to_numpy()[train].mean()
w1 = (1 - pi) / pi
w0 = pi / (1 - pi)
L = np.log(w1 / w0)
print(f'training AF prevalence pi={pi:.4f}, logit shift L={L:.3f}')

def to_pop(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    z = np.log(p / (1 - p)) - L
    return 1 / (1 + np.exp(-z))

p1c, p2c, p3c = to_pop(p1), to_pop(p2), to_pop(p3)
for name, raw, corr in [('M1', p1, p1c), ('M2', p2, p2c), ('M3', p3, p3c)]:
    a_raw = roc_auc_score(y, raw)
    a_corr = roc_auc_score(y, corr)
    assert abs(a_raw - a_corr) < 1e-9, f'{name}: AUC changed {a_raw} vs {a_corr}'
    print(f'{name}: AUC={a_corr:.4f} (invariant) corrected range [{corr.min():.4f}, {corr.max():.4f}]')

# ── Figure ──
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.3, 4.7), dpi=300)

# Panel A: calibration curve
for p, lab, col in [(p1c, 'M1 (Demographics)', '#E74C3C'),
                    (p2c, 'M2 (Demo+HRV)', '#3498DB'),
                    (p3c, 'M3 (Full ECG)', '#2ECC71')]:
    frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy='quantile')
    ax1.plot(mean_pred, frac_pos, '-o', color=col, ms=5, lw=2, label=lab)

ax1.plot([0, 1], [0, 1], ls='--', color='#888', lw=1.3, zorder=1)
ax1.set_xlabel('Predicted probability')
ax1.set_ylabel('Observed AF rate')
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.legend(loc='upper left', frameon=True, fancybox=False, edgecolor='#CCC', fontsize=9)
ax1.set_title('(A) Calibration (External)', fontsize=12, fontweight='bold', loc='left')
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.text(-0.078, 1.045, 'A', transform=ax1.transAxes, fontsize=15, fontweight='bold', va='center')

# ── Panel B: DCA ──
thresholds = np.linspace(0.001, 0.5, 500)
# Treat All
nb_all = prev - (1 - prev) * thresholds / (1 - thresholds)
ax2.plot(thresholds * 100, nb_all, color='#CCCCCC', lw=2.5, label='Treat All', zorder=2)
ax2.axhline(0, color='#444444', lw=1.5, label='Treat None', zorder=2)

n = len(y)
for p, lab, col in [(p1c, 'M1 (Demographics)', '#E74C3C'),
                    (p2c, 'M2 (Demo+HRV)', '#3498DB'),
                    (p3c, 'M3 (Full ECG)', '#2ECC71')]:
    xs, ys = [], []
    for pt in thresholds:
        treat = p >= pt
        if not treat.any():
            break  # beyond max predicted probability: nobody flagged (curve would sit on treat-none)
        tp = (treat & (y == 1)).sum()
        fp = (treat & (y == 0)).sum()
        nb = tp / n - fp / n * pt / (1 - pt)
        xs.append(pt * 100)
        ys.append(nb)
        if nb < -0.005:
            break  # below y-axis floor: stop instead of re-entering far right
    ax2.plot(xs, ys, color=col, lw=2, label=lab)

ax2.set_xlabel('Threshold probability (%)')
ax2.set_ylabel('Net benefit')
ax2.set_xlim(0, 50)
ax2.set_ylim(-0.005, 0.08)
ax2.set_xticks([0, 10, 20, 30, 40, 50])
ax2.set_yticks(np.arange(0, 0.081, 0.01))
ax2.set_yticklabels([f'{v:.2f}' for v in np.arange(0, 0.081, 0.01)])
ax2.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='#CCC', fontsize=9)
ax2.set_title('(B) Decision curve (External)', fontsize=12, fontweight='bold', loc='left')
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.text(-0.078, 1.045, 'B', transform=ax2.transAxes, fontsize=15, fontweight='bold', va='center')

plt.tight_layout()
plt.savefig(f'{FIG}\\Figure S3_cal_dca_external.png', dpi=600, bbox_inches='tight')
plt.savefig(f'{FIG}\\Figure S3_cal_dca_external.pdf', bbox_inches='tight')
plt.close()
print('Saved: Figure S3_cal_dca_external.png/pdf')
