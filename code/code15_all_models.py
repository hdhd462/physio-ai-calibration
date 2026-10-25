# -*- coding: utf-8 -*-
"""Table 4 completion: CODE-15% AUC (95% CI) for M1-M3 (routeB recipe), FusionStack F4 (frozen fold9 meta).
Gates: M1-M3 int/ext AUC match verify_all REF +/-0.001; F4 int/ext match fusion_probs <=1e-6."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA_DIR, PROJ_DIR, FIG_DIR, SUB_DIR, PATCH_DIR, NHANES_DIR, PROJ1_DIR, SHAOXING_DIR, PTBXL_DIR
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, warnings
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore')

OUT = DATA_DIR

ptb = pd.read_csv(f'{OUT}\\ecg_features_ptbxl.csv')
npz = np.load(f'{OUT}\\ptbxl_100hz.npz', allow_pickle=True)
ptb['fold'] = npz['fold'][ptb['idx'].values]
chp = pd.read_csv(f'{OUT}\\ecg_features_chapman.csv')

cd = np.load(f'{OUT}\\code15_part17_100hz.npz', allow_pickle=True)
feat_d = pd.read_csv(f'{OUT}\\ecg_features_code15.csv')
mlcd = np.load(f'{OUT}\\code15_multilabel.npz', allow_pickle=True)
assert np.array_equal(mlcd['Y'][:, 0].astype(np.int8), cd['y'].astype(np.int8)), 'AF bit-match fail'
okd = ~np.isnan(cd['X']).any(axis=(1, 2))
valid_d = feat_d[feat_d['idx'].isin(np.where(okd)[0])].sort_values('idx').reset_index(drop=True)
idx_d = valid_d['idx'].values
y_cd = cd['y'][idx_d].astype(int)
print(f'CODE-15% tabular rows: {len(valid_d)} (AF={int(y_cd.sum())})')

DEMO = ['age', 'sex']
HRV = ['hr_mean', 'hr_std', 'rr_mean', 'rr_std', 'rmssd', 'sdnn', 'pnn50',
       'cv_rr', 'rmssd_norm', 'dom_freq', 'spec_ent']
MORPH = ['qrs_dur', 'pr_int', 'qt_int', 'p_amp', 'sig_std_v2', 'sig_std_i', 'n_peaks']
MODELS = {'M1': DEMO, 'M2': DEMO + HRV, 'M3': DEMO + HRV + MORPH}

def fit_predict(train_df, test_df, feats):
    tr = train_df[feats + ['AF']]
    te = test_df[feats + ['AF']]
    imp = SimpleImputer(strategy='median')
    Xtr = imp.fit_transform(tr[feats])
    Xte = imp.transform(te[feats])
    sc = StandardScaler()
    Xtr = sc.fit_transform(Xtr)
    Xte = sc.transform(Xte)
    ytr = tr['AF'].values
    w = np.where(ytr == 1, (1 - ytr.mean()) / ytr.mean(), ytr.mean() / (1 - ytr.mean()))
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(Xtr, ytr, sample_weight=w)
    return clf.predict_proba(Xte)[:, 1]

def boot_ci(y, p, n=1000):
    r = np.random.RandomState(42)
    aucs = []
    for _ in range(n):
        i = r.choice(len(y), len(y), replace=True)
        if len(np.unique(y[i])) < 2:
            continue
        aucs.append(roc_auc_score(y[i], p[i]))
    return np.percentile(aucs, [2.5, 97.5])

full_train = pd.concat([ptb[ptb.fold <= 8], ptb[ptb.fold == 9]])
test_df = ptb[ptb.fold == 10]
REF = {'M1': (0.766589, 0.643254), 'M2': (0.967173, 0.811443), 'M3': (0.970088, 0.810863)}

results = {}
p_cd = {}
for name, feats in MODELS.items():
    p_te = fit_predict(full_train, test_df, feats)
    p_ex = fit_predict(full_train, chp, feats)
    p_d = fit_predict(full_train, valid_d, feats)
    ai, ae = roc_auc_score(test_df.AF.values, p_te), roc_auc_score(chp.AF.values, p_ex)
    ri, re_ = REF[name]
    assert abs(ai - ri) < 0.001 and abs(ae - re_) < 0.001, f'{name} REF mismatch {ai:.6f}/{ae:.6f}'
    lo, hi = boot_ci(y_cd, p_d)
    ad = roc_auc_score(y_cd, p_d)
    results[name] = (ad, lo, hi)
    p_cd[name] = p_d
    print(f'{name}: REF gate PASS (int={ai:.6f} ext={ae:.6f}) | CODE-15% AUC={ad:.4f} ({lo:.3f}-{hi:.3f})')

# ── FusionStack F4 (mirror fusion_stacking.py) ──
cal = np.load(f'{OUT}\\calibrated_probs.npz', allow_pickle=True)
dl9z = np.load(f'{OUT}\\fusion_dl9.npz', allow_pickle=True)
fold = npz['fold']
pos9 = np.where(fold == 9)[0]
feat9 = ptb[ptb.fold == 9].reset_index(drop=True)
p9 = pd.Index(pos9).get_indexer(feat9['idx'].values)
assert (p9 >= 0).all()
y9 = feat9['AF'].values
assert np.array_equal(y9, dl9z['y_val'][p9]) and np.array_equal(y9, cal['y_val'][p9])

FCOLS = {'M1': ['age', 'sex'],
         'M2': ['age', 'sex', 'hr_std', 'rr_mean', 'rmssd', 'sdnn', 'pnn50', 'cv_rr', 'rmssd_norm', 'dom_freq', 'spec_ent'],
         'M3': ['age', 'sex', 'hr_std', 'rr_mean', 'rmssd', 'sdnn', 'pnn50', 'cv_rr', 'rmssd_norm', 'dom_freq', 'spec_ent',
                'qrs_dur', 'pr_int', 'qt_int', 'p_amp', 'sig_std_v2', 'sig_std_i', 'n_peaks']}
df_tr = ptb[ptb.fold <= 8]

def lr9(cols):
    imp, sc = SimpleImputer(strategy='median'), StandardScaler()
    Xt = sc.fit_transform(imp.fit_transform(df_tr[cols]))
    yt = df_tr['AF'].values
    w = np.where(yt == 1, (1 - yt.mean()) / yt.mean(), yt.mean() / (1 - yt.mean()))
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(Xt, yt, sample_weight=w)
    return clf.predict_proba(sc.transform(imp.transform(feat9[cols])))[:, 1]

D9 = {'CNN': dl9z['val_CNN'][p9], 'ResNet': dl9z['val_ResNet'][p9],
      'CNNT': dl9z['val_CNNTransformer'][p9],
      'M1': lr9(FCOLS['M1']), 'M2': lr9(FCOLS['M2']), 'M3': cal['val_lr']}

c = np.load(f'{OUT}\\common_probs.npz', allow_pickle=True)
T = {'CNN': (c['int_CNN'], c['ext_CNN']), 'ResNet': (c['int_ResNet'], c['ext_ResNet']),
     'CNNT': (c['int_CNNTransformer'], c['ext_CNNTransformer']),
     'M1': (c['int_M1'], c['ext_M1']), 'M2': (c['int_M2'], c['ext_M2']), 'M3': (c['int_M3'], c['ext_M3'])}

def lg(p): return np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
def F(keys, src): return np.column_stack([lg(src[k]) for k in keys])
keys = ['CNN', 'ResNet', 'CNNT', 'M1', 'M2', 'M3']

sc = StandardScaler().fit(F(keys, D9))
clf = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(F(keys, D9)), y9)
fs_int = clf.predict_proba(sc.transform(F(keys, {k: T[k][0] for k in T})))[:, 1]
fs_ext = clf.predict_proba(sc.transform(F(keys, {k: T[k][1] for k in T})))[:, 1]

fs = np.load(f'{OUT}\\fusion_probs.npz', allow_pickle=True)
d_i = np.max(np.abs(fs_int - fs['int_F4']))
d_e = np.max(np.abs(fs_ext - fs['ext_F4']))
assert d_i < 1e-6 and d_e < 1e-6, f'F4 repro mismatch {d_i:.2e}/{d_e:.2e}'
ai4, ae4 = roc_auc_score(c['y_te'], fs_int), roc_auc_score(c['y_ext'], fs_ext)
print(f'F4 gate PASS (maxabsdiff int={d_i:.2e} ext={d_e:.2e}; AUC int={ai4:.4f} ext={ae4:.4f})')

cnpz = np.load(f'{OUT}\\code15_part17_probs.npz', allow_pickle=True)
assert np.array_equal(cnpz['y'].astype(np.int8), cd['y'].astype(np.int8))
src_cd = {'CNN': cnpz['CNN'][idx_d], 'ResNet': cnpz['ResNet'][idx_d], 'CNNT': cnpz['CNNT'][idx_d],
          'M1': p_cd['M1'], 'M2': p_cd['M2'], 'M3': p_cd['M3']}
fs_d = clf.predict_proba(sc.transform(F(keys, src_cd)))[:, 1]
lo, hi = boot_ci(y_cd, fs_d)
ad = roc_auc_score(y_cd, fs_d)
results['FusionStack'] = (ad, lo, hi)
print(f'FusionStack: CODE-15% AUC={ad:.4f} ({lo:.3f}-{hi:.3f})')

fn = np.load(f'{OUT}\\fusion_code15_binary.npz', allow_pickle=True)
results['FusionNet'] = (float(fn['auc']), float(fn['auc_lo']), float(fn['auc_hi']))
print(f'FusionNet: CODE-15% AUC={fn["auc"]:.4f} ({fn["auc_lo"]:.3f}-{fn["auc_hi"]:.3f})')

np.savez_compressed(f'{OUT}\\code15_table4_auc.npz',
                    **{f'{k}_auc': v[0] for k, v in results.items()},
                    **{f'{k}_lo': v[1] for k, v in results.items()},
                    **{f'{k}_hi': v[2] for k, v in results.items()},
                    n=len(y_cd), events=int(y_cd.sum()))
print('\n=== CODE-15% column (Table 4) ===')
for k, (a, l, h) in results.items():
    print(f'  {k:13s} {a:.3f} ({l:.3f}-{h:.3f})')
print(f'n={len(y_cd)} events={int(y_cd.sum())}')
print('saved: code15_table4_auc.npz')
