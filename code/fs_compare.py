import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA_DIR, PROJ_DIR, FIG_DIR, SUB_DIR, PATCH_DIR, NHANES_DIR, PROJ1_DIR, SHAOXING_DIR, PTBXL_DIR
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
import json
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

OUT = DATA_DIR
DEMO = ['age', 'sex']
HRV = ['hr_std', 'rr_mean', 'rmssd', 'sdnn', 'pnn50', 'cv_rr', 'rmssd_norm', 'dom_freq', 'spec_ent']
MORPH = ['qrs_dur', 'pr_int', 'qt_int', 'p_amp', 'sig_std_v2', 'sig_std_i', 'n_peaks']
MODELS = {'M1': DEMO, 'M2': DEMO + HRV, 'M3': DEMO + HRV + MORPH}

ptb = pd.read_csv(f'{OUT}\\fs_raw_ptbxl.csv')
train = ptb[ptb.fold <= 9].reset_index(drop=True)
y = train['AF'].astype(int).values
p = y.mean()
w = np.where(y == 1, (1 - p) / p, p / (1 - p))

# 标准化尺度（与Table7的StandardScaler一致）
def scaler_for(feats):
    d = train[feats].copy()
    d['sex'] = d['sex'].astype(float)
    imp = SimpleImputer(strategy='median')
    X = imp.fit_transform(d)
    sc = StandardScaler()
    sc.fit(X)
    return imp, sc

def freestat_tbl(m):
    df = pd.read_csv(f'{OUT}\\fs_logw_{m.lower()}\\artifacts\\table-3.csv', encoding='utf-8-sig')
    res = {}
    for _, row in df.iterrows():
        v = str(row['Variable']).strip()
        if v in ('(Intercept)',):
            continue
        if v.startswith('sex'):
            key = 'sex'
        else:
            key = v
        adj = str(row['adj.OR (95%CI)']).split('(')[0].strip()
        crude = str(row['crude.OR (95%CI)']).split('(')[0].strip()
        try:
            res[key] = float(adj)
        except ValueError:
            res[key] = np.nan
        res[key + '__crude'] = None
        try:
            res[key + '__crude'] = float(crude)
        except ValueError:
            pass
    return res

t7 = pd.read_csv(f'{OUT}\\table7_logistic_or.csv', encoding='utf-8-sig')
twin = json.load(open(f'{OUT}\\fs_twin_ref.json'))

rows = []
for mname, feats in MODELS.items():
    imp, sc = scaler_for(feats)
    fs_adj = freestat_tbl(mname)
    # sklearn多变量加权-无正则-原始单位（z空间算β再转raw已在twin里）
    twin_raw = {k: v for k, v in twin[mname.lower()].items() if not k.startswith('_')}
    # 单变量加权-原始单位
    uni = {}
    for f in feats:
        d = train[[f]].copy()
        if f == 'sex':
            d[f] = d[f].astype(float)
        Xi = SimpleImputer(strategy='median').fit_transform(d)
        clf = LogisticRegression(C=1e12, max_iter=5000)
        clf.fit(Xi, y, sample_weight=w)
        uni[f] = float(np.exp(clf.coef_[0][0]))
    for f in feats:
        row = {'Model': mname, 'Feature': f}
        row['Freestat_adj_raw'] = fs_adj.get(f, np.nan)
        row['Sklearn_adj_raw_twin'] = twin_raw.get(f, np.nan)
        if fs_adj.get(f) and twin_raw.get(f) and np.isfinite(fs_adj.get(f)) and np.isfinite(twin_raw.get(f)):
            r = fs_adj[f] / twin_raw[f]
            row['ratio_fs_sk'] = round(r, 4) if 0.01 < r < 100 else f'{r:.2e}'
        else:
            row['ratio_fs_sk'] = np.nan
        row['Freestat_crude_raw'] = fs_adj.get(f + '__crude')
        row['Sklearn_crude_raw'] = round(uni[f], 6) if np.isfinite(uni[f]) else np.nan
        # Table7 (z单位, 加权L2) 桥接: freestat raw -> z
        col = t7[(t7.Model.str.startswith(mname)) & (t7.Feature == f)]
        if len(col):
            t7_or = float(col.iloc[0]['OR'])
            sd = sc.scale_[feats.index(f)]
            row['Table7_z_L2'] = t7_or
            row['Freestat_adj_as_z'] = round(fs_adj.get(f, np.nan) ** sd, 4) if np.isfinite(fs_adj.get(f, np.nan)) else np.nan
        rows.append(row)

out = pd.DataFrame(rows)
out.to_csv(f'{OUT}\\fs_verification_table.csv', index=False, encoding='utf-8-sig')
pd.set_option('display.width', 250)
print(out.to_string(index=False))