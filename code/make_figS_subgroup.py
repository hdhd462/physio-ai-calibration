import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA_DIR, PROJ_DIR, FIG_DIR, SUB_DIR, PATCH_DIR, NHANES_DIR, PROJ1_DIR, SHAOXING_DIR, PTBXL_DIR
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import statsmodels.api as sm
from scipy.stats import chi2
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
chp = pd.read_csv(f'{OUT}\\ecg_features_chapman.csv')
y = d['ext_y']
p3 = d['ext_M3']  # 完整模型

# 对齐（特征提取时有少量失败，prob来自同一过滤后的集？检查）
# routeB_probs的ext来自chp整表（fit_predict用chp df），需对齐
assert len(y) == len(chp), f'{len(y)} vs {len(chp)}'

chp = chp.reset_index(drop=True)
chp['p'] = p3
# score = logit(p); routeB M3 is a class-weighted logistic model, so logit(p)
# recovers its linear predictor (up to an additive constant, irrelevant for slope tests)
chp['score'] = np.log(np.clip(chp['p'], 1e-12, 1 - 1e-12) / (1 - np.clip(chp['p'], 1e-12, 1 - 1e-12)))

def lrt_pint(df, factor, levels):
    """LRT of score x subgroup interaction in AF ~ score * subgroup (logistic)."""
    sub = df[df[factor].isin(levels)].dropna(subset=['score', 'AF'])
    g = pd.get_dummies(sub[factor], drop_first=True, dtype=float).to_numpy()
    X_r = np.column_stack([np.ones(len(sub)), sub['score'].to_numpy(), g])
    X_f = np.column_stack([X_r, sub['score'].to_numpy()[:, None] * g])
    yy = sub['AF'].to_numpy().astype(float)
    ll_r = sm.Logit(yy, X_r).fit(disp=0).llf
    ll_f = sm.Logit(yy, X_f).fit(disp=0).llf
    return float(chi2.sf(2 * (ll_f - ll_r), g.shape[1]))

def fmt_p(pv):
    return '<0.001' if pv < 0.001 else f'{pv:.3f}'

# 亚组
subgroups = []
def add_sub(label, mask, group):
    sub = chp[mask]
    if sub.AF.sum() < 10 or (sub.AF == 0).sum() < 10:
        return
    auc = roc_auc_score(sub.AF, sub.p)
    # bootstrap CI
    rng = np.random.RandomState(42)
    aucs = []
    for _ in range(500):
        i = rng.choice(len(sub), len(sub), replace=True)
        s = sub.iloc[i]
        if s.AF.nunique() < 2:
            continue
        aucs.append(roc_auc_score(s.AF, s.p))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    subgroups.append(dict(Group=group, Subgroup=label, N=len(sub),
                          AF=int(sub.AF.sum()), AUC=round(auc, 3),
                          Lo=round(lo, 3), Hi=round(hi, 3)))

add_sub('Overall', slice(None), 'All')
# 年龄
chp['age_g'] = pd.cut(chp.age, bins=[-1, 40, 60, 200], labels=['<40', '40-60', '>60'])
for g in ['<40', '40-60', '>60']:
    add_sub(f'Age {g}', chp.age_g == g, 'Age')
# 性别
chp['sex_g'] = chp.sex.map({0: 'Male', 1: 'Female'})
for g in ['Male', 'Female']:
    add_sub(g, chp.sex_g == g, 'Sex')

p_age = lrt_pint(chp, 'age_g', ['<40', '40-60', '>60'])
p_sex = lrt_pint(chp, 'sex_g', ['Male', 'Female'])
print(f'P_int age (LRT, 2 df) = {p_age:.4g} -> {fmt_p(p_age)}')
print(f'P_int sex (LRT, 1 df) = {p_sex:.4g} -> {fmt_p(p_sex)}')

df_sub = pd.DataFrame(subgroups)
df_sub['P_int'] = df_sub['Group'].map({'All': '', 'Age': fmt_p(p_age), 'Sex': fmt_p(p_sex)}).fillna('')
print(df_sub.to_string(index=False))
df_sub.to_csv(f'{OUT}\\tableS_subgroup_auc.csv', index=False, encoding='utf-8-sig')

# ── 森林图（Table S1 / Figure S1 三线表样式）──
fig, ax = plt.subplots(figsize=(9.6, 5.2))

rows = []
for _, r in df_sub.iterrows():
    rows.append(r)

x_text_subgroup = -0.12
x_text_event = 0.11
x_text_auc = 0.26
auc_x = lambda a: 0.50 + (a - 0.7) * (0.40 / 0.3)  # AUC -> axes coords; visible range 0.7-1.0
x_text_p = 1.08

plot_rows = []
plot_rows.append(('header', 'Subgroup', 'Event (%)', 'AUC (95% CI)', 'P for interaction'))
# Overall
r = rows[0]
plot_rows.append(('subheader', 'Overall', '', '', '', None, None, None, None, None))
evt = f"{r.AF} ({r.AF / r.N * 100:.1f}%)"
auc_txt = f"{r.AUC:.3f} ({r.Lo:.3f}\u2013{r.Hi:.3f})"
plot_rows.append(('data', 'All participants', evt, auc_txt, '', r.AUC, r.Lo, r.Hi, '#C71585', 22))
plot_rows.append(('spacer',))
# Age
plot_rows.append(('subheader', 'Age (years)', '', '', fmt_p(p_age), None, None, None, None, None))
for r in rows[1:4]:
    evt = f"{r.AF} ({r.AF / r.N * 100:.1f}%)"
    auc_txt = f"{r.AUC:.3f} ({r.Lo:.3f}\u2013{r.Hi:.3f})"
    plot_rows.append(('data', r.Subgroup.replace('Age ', '').replace('40-60', '40\u201360'), evt, auc_txt, '',
                      r.AUC, r.Lo, r.Hi, '#20B2AA', 13))
plot_rows.append(('spacer',))
# Sex
plot_rows.append(('subheader', 'Sex', '', '', fmt_p(p_sex), None, None, None, None, None))
for r in rows[4:6]:
    evt = f"{r.AF} ({r.AF / r.N * 100:.1f}%)"
    auc_txt = f"{r.AUC:.3f} ({r.Lo:.3f}\u2013{r.Hi:.3f})"
    plot_rows.append(('data', r.Subgroup, evt, auc_txt, '',
                      r.AUC, r.Lo, r.Hi, '#20B2AA', 13))

y = 0
spacing = 0.8
y_coords = []
for row in plot_rows:
    y_coords.append(y)
    y += spacing
y_coords = [max(y_coords) - yc for yc in y_coords]

for yy, row in zip(y_coords, plot_rows):
    if row[0] == 'header':
        ax.text(x_text_subgroup, yy, row[1], fontsize=10, fontweight='bold', va='center')
        ax.text(x_text_event, yy, row[2], fontsize=10, fontweight='bold', va='center')
        ax.text(x_text_auc, yy, row[3], fontsize=10, fontweight='bold', va='center')
        ax.text(x_text_p, yy, row[4], fontsize=10, fontweight='bold', va='center', ha='right')
        ax.axhline(y=yy - 0.3, xmin=0.02, xmax=0.98, color='black', lw=0.8)
    elif row[0] == 'subheader':
        ax.text(x_text_subgroup, yy, row[1], fontsize=10, fontweight='bold', va='center')
        if row[4]:
            ax.text(x_text_p, yy, row[4], fontsize=9, va='center', ha='right')
    elif row[0] == 'data':
        ax.text(x_text_subgroup, yy, '  ' + row[1], fontsize=9, va='center')
        ax.text(x_text_event, yy, row[2], fontsize=9, va='center')
        ax.text(x_text_auc, yy, row[3], fontsize=9, va='center')
        ax.plot([auc_x(row[6]), auc_x(row[7])], [yy, yy], color=row[8], lw=1.5, zorder=4)
        ax.scatter(auc_x(row[5]), yy, marker='o', color=row[8], s=row[9], zorder=5)

ax.axvline(x=auc_x(0.7), color='gray', lw=0.8, ls='--', zorder=1, ymin=0.05, ymax=0.95)
ax.set_xlim([-0.17, 1.10])
ax.set_ylim([-0.5, y + 0.5])
ax.set_xticks([auc_x(v) for v in (0.7, 0.8, 0.9, 1.0)])
ax.set_xticklabels(['0.7', '0.8', '0.9', '1.0'], fontsize=10)
ax.set_xlabel('AUC (95% CI)', fontsize=12, labelpad=10)
ax.set_yticks([])
ax.spines['left'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_position(('data', -0.3))

plt.tight_layout()
plt.savefig(f'{FIG}\\Figure S2_ECG_subgroup_AUC.png', dpi=600, bbox_inches='tight')
plt.savefig(f'{FIG}\\Figure S2_ECG_subgroup_AUC.pdf', bbox_inches='tight')
plt.close()
print('\nSaved: Figure S2_ECG_subgroup_AUC.png/pdf')
