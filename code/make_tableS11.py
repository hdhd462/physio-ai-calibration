import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA_DIR, PROJ_DIR, FIG_DIR, SUB_DIR, PATCH_DIR, NHANES_DIR, PROJ1_DIR, SHAOXING_DIR, PTBXL_DIR
import os, sys, json, copy
import numpy as np
import docx
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from scipy.special import logit, expit

sys.stdout.reconfigure(encoding='utf-8')
base = DATA_DIR
tabdir = os.path.join(SUB_DIR, "tables")
template = os.path.join(tabdir, 'Table S7.docx')


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    tot, n = 0.0, len(p)
    for b in range(bins):
        m = idx == b
        if m.sum() == 0:
            continue
        tot += m.sum() / n * abs(y[m].mean() - p[m].mean())
    return float(tot)


def cal_slope(y, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    lr = LogisticRegression(C=1e6, solver='lbfgs', max_iter=2000)
    lr.fit(logit(p).reshape(-1, 1), y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def fit_platt(p, y):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    lr = LogisticRegression(C=1e6, solver='lbfgs', max_iter=2000)
    lr.fit(logit(p).reshape(-1, 1), y)
    a, b = float(lr.coef_[0, 0]), float(lr.intercept_[0])
    return lambda q: expit(a * logit(np.clip(q, 1e-9, 1 - 1e-9)) + b)


def met(y, p):
    s, i = cal_slope(y, p)
    return dict(n=int(len(y)), events=int(y.sum()), prev=float(y.mean()),
                mean_pred=float(np.mean(p)), pct_ge_09=float(np.mean(p >= 0.9)),
                auc=float(roc_auc_score(y, p)), brier=float(brier_score_loss(y, p)),
                ece=ece(y, p), slope=s, intercept=i)


uw = np.load(os.path.join(base, 'fusion_unweighted_probs.npz'))
w = np.load(os.path.join(base, 'common_probs.npz'))
y_te = uw['y_te'].astype(int); y_ext = uw['y_ext'].astype(int)
p_uw_te = uw['int'].astype(float); p_uw_ext = uw['ext'].astype(float)
p_w_te = w['int_FusionNet'].astype(float); p_w_ext = w['ext_FusionNet'].astype(float)

platt_uw = fit_platt(p_uw_ext, y_ext)
platt_w = fit_platt(p_w_ext, y_ext)

rows = [
    ('Class-weighted BCE (as reported)', 'PTB-XL internal test', met(y_te, p_w_te)),
    ('', 'Chapman-Shaoxing', met(y_ext, p_w_ext)),
    ('', '+ Platt fitted on Chapman', met(y_ext, platt_w(p_w_ext))),
    ('Unweighted BCE (sensitivity analysis)', 'PTB-XL internal test', met(y_te, p_uw_te)),
    ('', 'Chapman-Shaoxing', met(y_ext, p_uw_ext)),
    ('', '+ Platt fitted on Chapman', met(y_ext, platt_uw(p_uw_ext))),
]

with open(os.path.join(PATCH_DIR, 'tableS11_source.json'), 'w', encoding='utf-8') as f:
    json.dump([dict(training=r[0], cohort=r[1], **r[2]) for r in rows], f, indent=2)

HEAD = ['Training loss', 'Cohort', 'n', 'AF, n (%)', 'Mean predicted', 'Scores \u22650.9, %',
        'Brier', 'ECE', 'Calibration slope', 'AUC']

data = []
for tr, co, m in rows:
    data.append([tr, co, f"{m['n']:,}", f"{m['events']:,} ({100 * m['prev']:.1f}%)",
                 f"{m['mean_pred']:.3f}", f"{100 * m['pct_ge_09']:.1f}",
                 f"{m['brier']:.3f}", f"{m['ece']:.3f}", f"{m['slope']:.2f}", f"{m['auc']:.3f}"])

doc = docx.Document(template)
t = doc.tables[0]
while len(t.rows) - 1 > len(data):
    t._tbl.remove(t.rows[-1]._tr)
while len(t.rows) - 1 < len(data):
    t._tbl.append(copy.deepcopy(t.rows[-1]._tr))
for j, txt in enumerate(HEAD):
    c = t.rows[0].cells[j]
    c.paragraphs[0].runs[0].text = txt
    for e in c.paragraphs[0].runs[1:]:
        e.text = ''
for i, row in enumerate(data, start=1):
    for j, txt in enumerate(row):
        c = t.rows[i].cells[j]
        c.paragraphs[0].runs[0].text = txt
        for e in c.paragraphs[0].runs[1:]:
            e.text = ''

doc.paragraphs[0].runs[0].text = 'Table S11. Class-weighted versus unweighted training: effect on internal and external calibration'
doc.paragraphs[1].runs[0].text = (
    'Note: Both models are the same FusionNet architecture trained on the same folds with the same seed, optimiser, schedule and 30-epoch budget; '
    'the only difference is the loss. The reported model used BCEWithLogitsLoss with pos_weight = 13.4 (the negative-to-positive ratio, 16,164:1,204); '
    'the sensitivity model used an unweighted BCE (pos_weight = 1.0). Checkpoint selection was by fold-9 validation AUC in both cases '
    '(0.9929 unweighted). Unweighted training improved internal calibration (ECE 0.019 \u2192 0.010; slope 0.47 \u2192 0.69) and attenuated, but did not '
    'remove, external drift: Chapman ECE remained 0.29, calibration slope 0.41 and mean predicted risk 0.37 against an observed prevalence of 8.0%, '
    'with 30.6% of records at or above 0.9. Class weighting therefore accounts for part of the severity of the drift but not for its presence; '
    'the failure reproduces under an unweighted objective.'
)
for e in doc.paragraphs[1].runs[1:]:
    e.text = ''

out = os.path.join(tabdir, 'Table S11.docx')
doc.save(out)
print('Saved:', out)
for r in data:
    print(' | '.join(r))
