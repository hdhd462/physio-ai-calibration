import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA_DIR, PROJ_DIR, FIG_DIR, SUB_DIR, PATCH_DIR, NHANES_DIR, PROJ1_DIR, SHAOXING_DIR, PTBXL_DIR
import numpy as np, os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from sklearn.linear_model import LogisticRegression
from scipy.special import logit, expit

sys.stdout.reconfigure(encoding='utf-8')
base = DATA_DIR
out = PATCH_DIR
os.makedirs(out, exist_ok=True)

plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 8,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

def fit_platt(p, y):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    lr = LogisticRegression(C=1e6, solver='lbfgs', max_iter=3000)
    lr.fit(logit(p).reshape(-1, 1), y)
    a, b = float(lr.coef_[0, 0]), float(lr.intercept_[0])
    return lambda q: expit(a * logit(np.clip(q, 1e-9, 1 - 1e-9)) + b)

def fit_beta(p, y):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    X = np.column_stack([np.log(p), -np.log(1 - p)])
    lr = LogisticRegression(C=1e6, solver='lbfgs', max_iter=5000)
    lr.fit(X, y)
    a, b = lr.coef_[0]
    c = float(lr.intercept_[0])
    return lambda q: expit(a * np.log(np.clip(q, 1e-9, 1 - 1e-9)) - b * np.log(1 - np.clip(q, 1e-9, 1 - 1e-9)) + c)

def reliability(y, p, bins=10):
    qs = np.quantile(p, np.linspace(0, 1, bins + 1))
    qs = np.unique(qs)
    idx = np.clip(np.digitize(p, qs[1:-1]), 0, len(qs) - 2)
    xs, ys, ns = [], [], []
    for b in range(len(qs) - 1):
        m = idx == b
        if m.sum() < 10:
            continue
        xs.append(p[m].mean()); ys.append(y[m].mean()); ns.append(m.sum())
    return np.array(xs), np.array(ys), np.array(ns)

common = np.load(os.path.join(base, 'common_probs.npz'))
y_int = common['y_te'].astype(int); y_ext = common['y_ext'].astype(int)
p_int = common['int_FusionNet'].astype(float); p_ext = common['ext_FusionNet'].astype(float)
code = np.load(os.path.join(base, 'fusion_code15_binary.npz'))
y_code = code['y'].astype(int); p_code = code['p'].astype(float)

platt_ch = fit_platt(p_ext, y_ext)
beta_ch = fit_beta(p_ext, y_ext)
platt_co = fit_platt(p_code, y_code)

fig = plt.figure(figsize=(7.0, 6.4))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.1], hspace=0.38, wspace=0.28)

# Panel A: prediction distributions
axA = fig.add_subplot(gs[0, :])
bins = np.linspace(0, 1, 51)
for name, p, color, ls in [
    ('PTB-XL internal (Germany)', p_int, '#1f77b4', '-'),
    ('Chapman-Shaoxing (China)', p_ext, '#d62728', '-'),
    ('CODE-15% (Brazil)', p_code, '#2ca02c', '-'),
]:
    h, edges = np.histogram(p, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    axA.plot(centers, h / len(p) * 100, color=color, lw=1.2, ls=ls, label=name)
axA.set_xlim(0, 1)
axA.set_xlabel('Predicted probability')
axA.set_ylabel('Share of records (%)')
axA.legend(frameon=False, fontsize=6.8, loc='upper center')
axA.text(0.985, 0.92, 'scores $\\geq$0.9: 7.0% / 39.8% / 2.9%',
         transform=axA.transAxes, ha='right', va='top', fontsize=6.5)
for s in ['top', 'right']:
    axA.spines[s].set_visible(False)

# Panel B: Chapman reliability
axB = fig.add_subplot(gs[1, 0])
axB.plot([0, 1], [0, 1], color='0.4', lw=0.8, ls='--')
xs, ys, _ = reliability(y_ext, p_ext); axB.plot(xs, ys, 'o-', color='#d62728', ms=3, lw=1.0, label='Raw')
xs, ys, _ = reliability(y_ext, platt_ch(p_ext)); axB.plot(xs, ys, 's-', color='#1f77b4', ms=3, lw=1.0, label='Platt (local)')
xs, ys, _ = reliability(y_ext, beta_ch(p_ext)); axB.plot(xs, ys, '^--', color='#2ca02c', ms=3, lw=1.0, label='Beta (local)')
axB.set_xlim(0, 1); axB.set_ylim(0, 1)
axB.set_xlabel('Mean predicted probability'); axB.set_ylabel('Observed AF prevalence')
axB.set_title('Chapman-Shaoxing', fontsize=8, pad=4)
axB.legend(frameon=False, fontsize=6.3, loc='upper left')
for s in ['top', 'right']:
    axB.spines[s].set_visible(False)

# Panel C: CODE reliability, cross-site vs local
axC = fig.add_subplot(gs[1, 1])
axC.plot([0, 1], [0, 1], color='0.4', lw=0.8, ls='--')
xs, ys, _ = reliability(y_code, p_code); axC.plot(xs, ys, 'o-', color='#2ca02c', ms=3, lw=1.0, label='Raw')
xs, ys, _ = reliability(y_code, platt_ch(p_code)); axC.plot(xs, ys, 's--', color='#9467bd', ms=3, lw=1.0, label='Platt (from Chapman)')
xs, ys, _ = reliability(y_code, platt_co(p_code)); axC.plot(xs, ys, '^:', color='#1f77b4', ms=3, lw=1.0, label='Platt (local)')
axC.set_xlim(0, 1); axC.set_ylim(0, 1)
axC.set_xlabel('Mean predicted probability'); axC.set_ylabel('Observed AF prevalence')
axC.set_title('CODE-15%', fontsize=8, pad=4)
axC.legend(frameon=False, fontsize=6.3, loc='upper left')
for s in ['top', 'right']:
    axC.spines[s].set_visible(False)

for ax, letter in [(axA, 'A'), (axB, 'B'), (axC, 'C')]:
    ax.text(-0.02, 1.06, letter, transform=ax.transAxes, fontsize=9, fontweight='bold', va='bottom', ha='right')

pdf = os.path.join(out, 'Figure_S7_prediction_distribution_reliability.pdf')
png = os.path.join(out, 'Figure_S7_prediction_distribution_reliability.png')
fig.savefig(pdf, bbox_inches='tight')
fig.savefig(png, dpi=300, bbox_inches='tight')
plt.close(fig)
print('Saved:', pdf)
print('Saved:', png)
