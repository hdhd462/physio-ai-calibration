import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA_DIR, PROJ_DIR, FIG_DIR, SUB_DIR, PATCH_DIR, NHANES_DIR, PROJ1_DIR, SHAOXING_DIR, PTBXL_DIR
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from sklearn.metrics import roc_curve, auc
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
colors = {'M1': '#E74C3C', 'M2': '#3498DB', 'M3': '#2ECC71'}
labels = {'M1': 'M1 (Demographics)', 'M2': 'M2 (Demo+HRV)', 'M3': 'M3 (Full ECG)'}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

# Panel A: 内部测试（PTB-XL fold10）
y_int = d['test_y']
for m in ['M1', 'M2', 'M3']:
    p = d[f'test_{m}']
    fpr, tpr, _ = roc_curve(y_int, p)
    a = auc(fpr, tpr)
    ax1.plot(fpr, tpr, color=colors[m], lw=1.8, label=f'{labels[m]} (AUC={a:.3f})')
ax1.plot([0, 1], [0, 1], ls='--', color='#888', lw=1)
ax1.set_xlabel('1 - Specificity')
ax1.set_ylabel('Sensitivity')
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.legend(loc='lower right', frameon=True, fancybox=False, edgecolor='#CCC', fontsize=9)
ax1.set_title('(A) Internal test (PTB-XL)', fontsize=12, fontweight='bold', loc='left')
ax1.text(-0.08, 1.06, 'A', transform=ax1.transAxes, fontsize=15, fontweight='bold', va='top')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Panel B: 外部验证（Chapman）
y_ext = d['ext_y']
for m in ['M1', 'M2', 'M3']:
    p = d[f'ext_{m}']
    fpr, tpr, _ = roc_curve(y_ext, p)
    a = auc(fpr, tpr)
    ax2.plot(fpr, tpr, color=colors[m], lw=1.8, label=f'{labels[m]} (AUC={a:.3f})')
ax2.plot([0, 1], [0, 1], ls='--', color='#888', lw=1)
ax2.set_xlabel('1 - Specificity')
ax2.set_ylabel('Sensitivity')
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.legend(loc='lower right', frameon=True, fancybox=False, edgecolor='#CCC', fontsize=9)
ax2.set_title('(B) External validation (Chapman-Shaoxing)', fontsize=12, fontweight='bold', loc='left')
ax2.text(-0.08, 1.06, 'B', transform=ax2.transAxes, fontsize=15, fontweight='bold', va='top')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(f'{FIG}\\figure7_routeB_roc.png', dpi=600, bbox_inches='tight')
plt.savefig(f'{FIG}\\figure7_routeB_roc.pdf', bbox_inches='tight')
plt.close()
print('Saved: figure7_routeB_roc.png/pdf')
