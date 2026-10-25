import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA_DIR, PROJ_DIR, FIG_DIR, SUB_DIR, PATCH_DIR, NHANES_DIR, PROJ1_DIR, SHAOXING_DIR, PTBXL_DIR
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.ndimage import uniform_filter1d

rcParams['font.family'] = 'Arial'
rcParams['axes.linewidth'] = 0.8
rcParams['pdf.fonttype'] = 42
rcParams['font.size'] = 11

OUT = DATA_DIR
FIG = FIG_DIR
d = np.load(f'{OUT}\\calibrated_probs.npz')

# ── Figure 10: 校准曲线（Platt校准后） ──
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.6), dpi=300)
panels = [
    (ax1, [(d['y_int_lr'], d['cal_int_lr'], 'Logistic (M3)'),
           (d['y_int'], d['cal_int_dl'], 'CNN-Transformer')], 'A', '(A) Internal test (PTB-XL)'),
    (ax2, [(d['y_ext_lr'], d['cal_ext_lr'], 'Logistic (M3)'),
           (d['y_ext'], d['cal_ext_dl'], 'CNN-Transformer')], 'B', '(B) External validation (Chapman-Shaoxing)'),
]
for ax, curves, letter, title in panels:
    for y, p, label in curves:
        auc = roc_auc_score(y, p)
        color = '#2ECC71' if 'Logistic' in label else '#2C3E50'
        frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy='quantile')
        ax.plot(mean_pred, frac_pos, color=color, marker='o', markersize=5,
                lw=1.5, label=f'{label} (AUC={auc:.3f})')
    ax.plot([0, 1], [0, 1], ls='--', color='#888', lw=1, label='Ideal')
    ax.set_xlabel('Predicted probability')
    ax.set_ylabel('Observed AF rate')
    ax.set_xlim(0, 0.85)
    ax.set_ylim(0, 0.85)
    ax.legend(loc='upper left', frameon=True, fancybox=False, edgecolor='#CCC', fontsize=9)
    ax.set_title(title, fontsize=12, fontweight='bold', loc='left')
    ax.text(-0.08, 1.06, letter, transform=ax.transAxes, fontsize=15, fontweight='bold', va='top')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(f'{FIG}\\figure10_calibration.png', dpi=600, bbox_inches='tight')
plt.savefig(f'{FIG}\\figure10_calibration.pdf', bbox_inches='tight')
plt.close()
print('Saved: figure10_calibration.png/pdf')

# ── Figure 9: DCA（外部验证, Platt校准后; DL与LR各自队列） ──
y_all = d['y_ext']
n_all = len(y_all)
prev = y_all.mean()
fig, ax = plt.subplots(figsize=(6.4, 5.6), dpi=300)
thresholds = np.linspace(0.001, 0.5, 500)
nb_all = prev - (1 - prev) * thresholds / (1 - thresholds)
ax.plot(thresholds * 100, nb_all, color='#AAAAAA', lw=2.5, label='Treat All', zorder=2)
ax.axhline(0, color='#444', lw=1.5, label='Treat None', zorder=2)
for y, p, label, color in [(d['y_ext_lr'], d['cal_ext_lr'], 'Logistic (M3)', '#2ECC71'),
                           (d['y_ext'], d['cal_ext_dl'], 'CNN-Transformer', '#2C3E50')]:
    n = len(y)
    status = y == 1
    nbs = []
    for pt in thresholds:
        treat = p >= pt
        tp = (treat & status).sum()
        fp = (treat & ~status).sum()
        nbs.append((tp / n) - (fp / n) * (pt / (1 - pt)))
    nbs = uniform_filter1d(nbs, size=11, mode='nearest')
    ax.plot(thresholds * 100, nbs, color=color, lw=1.6, label=label, zorder=3)
ax.set_xlabel('Threshold probability (%)')
ax.set_ylabel('Net benefit')
ax.set_xlim(0, 40)
ax.set_ylim(-0.005, 0.09)
ax.set_xticks([0, 10, 20, 30, 40])
ax.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='#CCC', fontsize=9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(f'{FIG}\\figure9_dca.png', dpi=600, bbox_inches='tight')
plt.savefig(f'{FIG}\\figure9_dca.pdf', bbox_inches='tight')
plt.close()
print('Saved: figure9_dca.png/pdf')