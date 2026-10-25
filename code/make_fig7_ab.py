import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA_DIR, PROJ_DIR, FIG_DIR, SUB_DIR, PATCH_DIR, NHANES_DIR, PROJ1_DIR, SHAOXING_DIR, PTBXL_DIR
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from sklearn.metrics import roc_curve, auc
from scipy.interpolate import PchipInterpolator
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

def smooth_roc(fpr, tpr, n_points=300):
    """Smooth ROC curve using PCHIP interpolation."""
    # Add endpoints
    fpr = np.concatenate([[0], fpr, [1]])
    tpr = np.concatenate([[0], tpr, [1]])
    # Remove duplicates for interpolation
    if len(fpr) > 1:
        keep = np.ones(len(fpr), dtype=bool)
        keep[1:] = np.diff(fpr) > 1e-6
        fpr_unique = fpr[keep]
        tpr_unique = tpr[keep]
    else:
        fpr_unique = fpr
        tpr_unique = tpr
    # Ensure at least 2 points for interpolation
    if len(fpr_unique) < 2:
        fpr_unique = np.array([0, 1])
        tpr_unique = np.array([0, 1])
    # Interpolate
    fpr_smooth = np.linspace(0, 1, n_points)
    interp = PchipInterpolator(fpr_unique, tpr_unique)
    tpr_smooth = interp(fpr_smooth)
    # Clip
    tpr_smooth = np.clip(tpr_smooth, 0, 1)
    return fpr_smooth, tpr_smooth

d = np.load(f'{OUT}\\common_probs.npz')
fe = np.load(f'{OUT}\\fusion3_probs.npz')

CURVES = [
    ('int_M1', 'M1 (Demographics)', '#E74C3C', '--', 1.4),
    ('int_M2', 'M2 (Demo+HRV)', '#3498DB', '--', 1.4),
    ('int_M3', 'M3 (Demo+HRV+Morph)', '#2ECC71', '--', 1.4),
    ('int_CNN', '1D-CNN', '#8E44AD', '-', 1.8),
    ('int_ResNet', 'ResNet1D', '#E67E22', '-', 1.8),
    ('int_CNNTransformer', 'CNN-Transformer', '#2C3E50', '-', 2.0),
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

y_int, y_ext = d['y_te'], d['y_ext']
for key, label, color, ls, lw in CURVES:
    ek = key.replace('int_', 'ext_')
    # Calculate AUC from original data
    fpr_raw, tpr_raw, _ = roc_curve(y_int, d[key])
    auc_int = auc(fpr_raw, tpr_raw)
    # Plot smoothed curve
    fpr_s, tpr_s = smooth_roc(fpr_raw, tpr_raw)
    ax1.plot(fpr_s, tpr_s, color=color, ls=ls, lw=lw, label=f'{label} (AUC={auc_int:.3f})')
    
    fpr_raw, tpr_raw, _ = roc_curve(y_ext, d[ek])
    auc_ext = auc(fpr_raw, tpr_raw)
    fpr_s, tpr_s = smooth_roc(fpr_raw, tpr_raw)
    ax2.plot(fpr_s, tpr_s, color=color, ls=ls, lw=lw, label=f'{label} (AUC={auc_ext:.3f})')

# 融合网络（端到端, 红色加粗）
fpr_raw, tpr_raw, _ = roc_curve(y_int, fe['int'])
auc_int = auc(fpr_raw, tpr_raw)
fpr_s, tpr_s = smooth_roc(fpr_raw, tpr_raw)
ax1.plot(fpr_s, tpr_s, color='#C0392B', ls='-', lw=2.2, label=f'Fusion network (AUC={auc_int:.3f})')

fpr_raw, tpr_raw, _ = roc_curve(y_ext, fe['ext'])
auc_ext = auc(fpr_raw, tpr_raw)
fpr_s, tpr_s = smooth_roc(fpr_raw, tpr_raw)
ax2.plot(fpr_s, tpr_s, color='#C0392B', ls='-', lw=2.2, label=f'Fusion network (AUC={auc_ext:.3f})')

for ax, letter, title in [(ax1, 'A', '(A) Internal test (PTB-XL)'),
                          (ax2, 'B', '(B) External validation (Chapman-Shaoxing)')]:
    ax.plot([0, 1], [0, 1], ls='--', color='#888', lw=1)
    ax.set_xlabel('1 - Specificity')
    ax.set_ylabel('Sensitivity')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc='lower right', frameon=True, fancybox=False, edgecolor='#CCC', fontsize=8)
    ax.set_title(title, fontsize=12, fontweight='bold', loc='left')
    ax.text(-0.08, 1.06, letter, transform=ax.transAxes, fontsize=15, fontweight='bold', va='top')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(f'{FIG}\\figure7_roc.png', dpi=600, bbox_inches='tight')
plt.savefig(f'{FIG}\\figure7_roc.pdf', bbox_inches='tight')
plt.close()
print('Saved: figure7_roc.png/pdf')