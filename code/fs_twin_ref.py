import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA_DIR, PROJ_DIR, FIG_DIR, SUB_DIR, PATCH_DIR, NHANES_DIR, PROJ1_DIR, SHAOXING_DIR, PTBXL_DIR
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
import json

OUT = DATA_DIR
ptb = pd.read_csv(f'{OUT}\\fs_raw_ptbxl.csv')
train = ptb[ptb.fold <= 9].reset_index(drop=True)

DEMO = ['age', 'sex']
HRV = ['hr_std', 'rr_mean', 'rmssd', 'sdnn', 'pnn50', 'cv_rr', 'rmssd_norm', 'dom_freq', 'spec_ent']
MORPH = ['qrs_dur', 'pr_int', 'qt_int', 'p_amp', 'sig_std_v2', 'sig_std_i', 'n_peaks']
MODELS = {'m1': DEMO, 'm2': DEMO + HRV, 'm3': DEMO + HRV + MORPH}

y = train['AF'].astype(int).values
p = y.mean()
w = np.where(y == 1, (1 - p) / p, p / (1 - p))
print(f'n={len(y)}, AF={y.sum()}, w1={w[y==1][0]:.6f}, w0={w[y==0][0]:.6f}')

out = {}
for name, feats in MODELS.items():
    d = train[feats].copy()
    if d['sex'].dtype == object or str(d['sex'].dtype) == 'category':
        d['sex'] = (d['sex'].astype(float)).values
    else:
        d['sex'] = d['sex'].astype(float)
    imp = SimpleImputer(strategy='median')
    X = imp.fit_transform(d)
    clf = LogisticRegression(C=1e12, max_iter=10000, solver='lbfgs')
    clf.fit(X, y, sample_weight=w)
    ors = np.exp(clf.coef_[0])
    out[name] = {f: float(o) for f, o in zip(feats, ors)}
    out[name]['_intercept'] = float(np.exp(clf.intercept_[0]))
    print(name, {f: f'{o:.4g}' for f, o in zip(feats, ors)})

json.dump(out, open(f'{OUT}\\fs_twin_ref.json', 'w'), indent=1)
print('saved fs_twin_ref.json')