import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA_DIR, PROJ_DIR, FIG_DIR, SUB_DIR, PATCH_DIR, NHANES_DIR, PROJ1_DIR, SHAOXING_DIR, PTBXL_DIR
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

OUT = DATA_DIR
rng = np.random.RandomState(42)

ptb = pd.read_csv(f'{OUT}\\ecg_features_ptbxl.csv')
npz = np.load(f'{OUT}\\ptbxl_100hz.npz', allow_pickle=True)
ptb['fold'] = npz['fold'][ptb['idx'].values]
train_df = ptb[ptb.fold <= 9].reset_index(drop=True)

# 去共线：删hr_mean(=60000/rr_mean)、rr_std(=sdnn)
DEMO = ['age', 'sex']
HRV = ['hr_std', 'rr_mean', 'rmssd', 'sdnn', 'pnn50',
       'cv_rr', 'rmssd_norm', 'dom_freq', 'spec_ent']
MORPH = ['qrs_dur', 'pr_int', 'qt_int', 'p_amp', 'sig_std_v2', 'sig_std_i', 'n_peaks']

MODELS = {
    'M1': DEMO,
    'M2': DEMO + HRV,
    'M3': DEMO + HRV + MORPH,
}

def fit_boot(df, feats, n_boot=500):
    d = df[feats + ['AF']].copy()
    imp = SimpleImputer(strategy='median')
    X = imp.fit_transform(d[feats])
    sc = StandardScaler()
    X = sc.fit_transform(X)
    y = d['AF'].values
    w = np.where(y == 1, (1 - y.mean()) / y.mean(), y.mean() / (1 - y.mean()))

    def fit_one(X_, y_, w_):
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(X_, y_, sample_weight=w_)
        return clf.coef_[0]

    coef = fit_one(X, y, w)
    # bootstrap CI
    boots = []
    n = len(y)
    for _ in range(n_boot):
        i = rng.choice(n, n, replace=True)
        yi = y[i]
        if len(np.unique(yi)) < 2:
            continue
        wi = np.where(yi == 1, (1 - yi.mean()) / yi.mean(), yi.mean() / (1 - yi.mean()))
        try:
            boots.append(fit_one(X[i], yi, wi))
        except Exception:
            pass
    boots = np.array(boots)
    lo = np.percentile(boots, 2.5, axis=0)
    hi = np.percentile(boots, 97.5, axis=0)
    # P值（正态近似）
    se = boots.std(axis=0)
    z = coef / (se + 1e-12)
    pvals = 2 * (1 - norm.cdf(np.abs(z)))

    rows = []
    for f, c, l, h, p in zip(feats, coef, lo, hi, pvals):
        rows.append({'Feature': f, 'OR': f'{np.exp(c):.3f}',
                     '95%CI': f'{np.exp(l):.2f}-{np.exp(h):.2f}',
                     'P': f'{p:.4f}' if p >= 0.0001 else '<0.0001'})
    auc = roc_auc_score(y, LogisticRegression(max_iter=2000, C=1.0).fit(
        X, y, sample_weight=w).predict_proba(X)[:, 1])
    return pd.DataFrame(rows), round(auc, 4)

all_res = []
for mname, feats in MODELS.items():
    tbl, auc = fit_boot(train_df, feats)
    tbl.insert(0, 'Model', f'{mname} (n={len(feats)})')
    tbl.insert(1, 'Train_AUC', auc)
    all_res.append(tbl)
    print(f'{mname}: AUC={auc}, {len(feats)}特征')

out = pd.concat(all_res, ignore_index=True)
out.to_csv(f'{OUT}\\table7_logistic_or.csv', index=False, encoding='utf-8-sig')
print(f'\n=== Table 7 ===')
print(out.to_string(index=False))
print(f'\n保存: table7_logistic_or.csv')
