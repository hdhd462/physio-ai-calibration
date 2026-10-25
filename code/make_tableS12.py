import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA_DIR, PROJ_DIR, FIG_DIR, SUB_DIR, PATCH_DIR, NHANES_DIR, PROJ1_DIR, SHAOXING_DIR, PTBXL_DIR
import os, sys, json, copy
import numpy as np
import docx
from docx.shared import Pt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from scipy.special import logit, expit

sys.stdout.reconfigure(encoding='utf-8')
base = DATA_DIR
sub = SUB_DIR
tabdir = os.path.join(sub, 'tables')
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


def fit_beta(p, y):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    X = np.column_stack([np.log(p), -np.log(1 - p)])
    lr = LogisticRegression(C=1e6, solver='lbfgs', max_iter=5000)
    lr.fit(X, y)
    a, b = lr.coef_[0]
    c = float(lr.intercept_[0])
    return lambda q: expit(a * np.log(np.clip(q, 1e-9, 1 - 1e-9))
                           - b * np.log(1 - np.clip(q, 1e-9, 1 - 1e-9)) + c)


def metrics(y, p):
    s, i = cal_slope(y, p)
    return dict(n=int(len(y)), events=int(y.sum()), prev=float(y.mean()),
                mean_pred=float(np.mean(p)), pct_ge_09=float(np.mean(p >= 0.9)),
                auc=float(roc_auc_score(y, p)), brier=float(brier_score_loss(y, p)),
                ece=ece(y, p), slope=s, intercept=i)


common = np.load(os.path.join(base, 'common_probs.npz'))
y_int = common['y_te'].astype(int)
y_ext = common['y_ext'].astype(int)
p_int = common['int_FusionNet'].astype(float)
p_ext = common['ext_FusionNet'].astype(float)
code = np.load(os.path.join(base, 'fusion_code15_binary.npz'))
p_code = code['p'].astype(float)
y_code = code['y'].astype(int)

platt_chap = fit_platt(p_ext, y_ext)
beta_chap = fit_beta(p_ext, y_ext)
platt_code = fit_platt(p_code, y_code)
beta_code = fit_beta(p_code, y_code)

rows = [
    ('PTB-XL internal (transport reference), raw', y_int, p_int),
    ('Chapman-Shaoxing, raw', y_ext, p_ext),
    ('Chapman-Shaoxing, Platt (fitted on Chapman)', y_ext, platt_chap(p_ext)),
    ('Chapman-Shaoxing, Beta (fitted on Chapman)', y_ext, beta_chap(p_ext)),
    ('CODE-15%, raw', y_code, p_code),
    ('CODE-15%, Platt transferred from Chapman', y_code, platt_chap(p_code)),
    ('CODE-15%, Beta transferred from Chapman', y_code, beta_chap(p_code)),
    ('CODE-15%, Platt (fitted on CODE-15%)', y_code, platt_code(p_code)),
    ('Chapman-Shaoxing, Platt transferred from CODE-15%', y_ext, platt_code(p_ext)),
]

computed = []
for label, y, p in rows:
    m = metrics(y, p)
    m['setting'] = label
    computed.append(m)

with open(os.path.join(PATCH_DIR, 'tableS12_source.json'), 'w', encoding='utf-8') as f:
    json.dump(computed, f, indent=2)

HEAD = ['Setting', 'n', 'AF, n (%)', 'Mean predicted', 'Scores \u22650.9, %',
        'Brier', 'ECE', 'Calibration slope', 'Calibration-in-the-large', 'AUC']


def fmt(m):
    return [
        m['setting'],
        f"{m['n']:,}",
        f"{m['events']:,} ({100 * m['prev']:.1f}%)",
        f"{m['mean_pred']:.3f}",
        f"{100 * m['pct_ge_09']:.1f}",
        f"{m['brier']:.3f}",
        f"{m['ece']:.3f}",
        f"{m['slope']:.2f}",
        f"{m['intercept']:.2f}",
        f"{m['auc']:.3f}",
    ]


def build_table(out_path, head, data, caption, note, widths=None):
    doc = docx.Document(template)
    t = doc.tables[0]
    need = len(data) + 1
    while len(t.rows) - 1 > len(data):
        t._tbl.remove(t.rows[-1]._tr)
    while len(t.rows) - 1 < len(data):
        t._tbl.append(copy.deepcopy(t.rows[-1]._tr))
    for j, txt in enumerate(head):
        cell = t.rows[0].cells[j]
        cell.paragraphs[0].runs[0].text = txt
        for extra in cell.paragraphs[0].runs[1:]:
            extra.text = ''
    for i, row in enumerate(data, start=1):
        for j, txt in enumerate(row):
            cell = t.rows[i].cells[j]
            cell.paragraphs[0].runs[0].text = txt
            for extra in cell.paragraphs[0].runs[1:]:
                extra.text = ''
    doc.paragraphs[0].runs[0].text = caption
    if len(doc.paragraphs) > 1:
        doc.paragraphs[1].runs[0].text = note
        for extra in doc.paragraphs[1].runs[1:]:
            extra.text = ''
    doc.save(out_path)
    print('Saved:', out_path)


build_table(
    os.path.join(tabdir, 'Table S12.docx'),
    HEAD,
    [fmt(m) for m in computed],
    'Table S12. Transport of calibration maps between sites',
    'Note: All rows use the same frozen FusionNet scores; only the post-hoc calibration map changes. Platt and Beta maps were fitted by logistic regression on the logit or on both log-transformed probabilities of the predicted risk, respectively; rows labelled "fitted on" are apparent (in-sample) performance and are shown only to indicate the ceiling that a site-specific map can reach. Transferred maps were fitted in one cohort and applied unchanged to the other. Calibration slope and calibration-in-the-large were estimated as in Table S7. ECE uses 10 equal-width bins; the reliability curves in Figure S7 use 10 quantile bins because equal-width binning leaves most bins unpopulated when predictions are bimodal. AUC is unaffected by any monotone recalibration. The transferred maps lower ECE numerically but invert the calibration slope (1.29\u20131.80) and drive the mean predicted risk below the observed prevalence (0.007\u20130.008 vs 0.028), i.e. they achieve average calibration by ceasing to identify high-risk records: no record receives a predicted risk above 0.9 and the share of records flagged at the source threshold falls from 15.9% to 5.7%.'
)
