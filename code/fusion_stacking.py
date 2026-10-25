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
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

OUT = DATA_DIR

# ── 元训练集: fold9对齐 ──
cal = np.load(f'{OUT}\\calibrated_probs.npz', allow_pickle=True)
dl9z = np.load(f'{OUT}\\fusion_dl9.npz', allow_pickle=True)
ptb = np.load(f'{OUT}\\ptbxl_100hz.npz', allow_pickle=True)
fold = ptb['fold']
feat_p = pd.read_csv(f'{OUT}\\ecg_features_ptbxl.csv')
feat_p['fold'] = fold[feat_p['idx'].values]
chf = pd.read_csv(f'{OUT}\\ecg_features_chapman.csv')

pos9 = np.where(fold == 9)[0]
feat9 = feat_p[feat_p.fold == 9].reset_index(drop=True)
p9 = pd.Index(pos9).get_indexer(feat9['idx'].values)
assert (p9 >= 0).all()
y9 = feat9['AF'].values
assert np.array_equal(y9, dl9z['y_val'][p9]) and np.array_equal(y9, cal['y_val'][p9])

# M1/M2/M3 fold9预测 (folds1-8训练, 与calibrate同法)
COLS = {'M1': ['age', 'sex'],
        'M2': ['age', 'sex', 'hr_std', 'rr_mean', 'rmssd', 'sdnn', 'pnn50', 'cv_rr', 'rmssd_norm', 'dom_freq', 'spec_ent'],
        'M3': ['age', 'sex', 'hr_std', 'rr_mean', 'rmssd', 'sdnn', 'pnn50', 'cv_rr', 'rmssd_norm', 'dom_freq', 'spec_ent',
               'qrs_dur', 'pr_int', 'qt_int', 'p_amp', 'sig_std_v2', 'sig_std_i', 'n_peaks']}
df_tr = feat_p[feat_p.fold <= 8]
def lr9(cols):
    imp, sc = SimpleImputer(strategy='median'), StandardScaler()
    Xt = sc.fit_transform(imp.fit_transform(df_tr[cols]))
    yt = df_tr['AF'].values
    w = np.where(yt == 1, (1 - yt.mean()) / yt.mean(), yt.mean() / (1 - yt.mean()))
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(Xt, yt, sample_weight=w)
    return clf.predict_proba(sc.transform(imp.transform(feat9[cols])))[:, 1]

D9 = {'CNN': dl9z['val_CNN'][p9], 'ResNet': dl9z['val_ResNet'][p9],
      'CNNT': dl9z['val_CNNTransformer'][p9],
      'M1': lr9(COLS['M1']), 'M2': lr9(COLS['M2']), 'M3': cal['val_lr']}
print(f'元训练 n={len(y9)} AF={y9.mean()*100:.2f}% fold9 AUC: ' + ' '.join(f'{k}={roc_auc_score(y9, v):.4f}' for k, v in D9.items()))

# ── 测试集 ──
c = np.load(f'{OUT}\\common_probs.npz', allow_pickle=True)
y_te, y_ext = c['y_te'], c['y_ext']
T = {'CNN': (c['int_CNN'], c['ext_CNN']), 'ResNet': (c['int_ResNet'], c['ext_ResNet']),
     'CNNT': (c['int_CNNTransformer'], c['ext_CNNTransformer']),
     'M1': (c['int_M1'], c['ext_M1']), 'M2': (c['int_M2'], c['ext_M2']), 'M3': (c['int_M3'], c['ext_M3'])}

def lg(p): return np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
def F(keys, src): return np.column_stack([lg(src[k]) for k in keys])
def fit_meta(X, y, *Xt):
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(X), y)
    return [clf.predict_proba(sc.transform(x))[:, 1] for x in Xt]

VAR = {'F1': ['CNNT', 'M3'],
       'F2': ['CNN', 'ResNet', 'CNNT', 'M3'],
       'F4': ['CNN', 'ResNet', 'CNNT', 'M1', 'M2', 'M3']}

# fold9 5折CV选型 (元模型仅用fold9, CV防元过拟合)
cv_scores = {}
for name, keys in VAR.items():
    X = F(keys, D9)
    aucs = []
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    for tr_, te_ in skf.split(X, y9):
        p = fit_meta(X[tr_], y9[tr_], X[te_])[0]
        aucs.append(roc_auc_score(y9[te_], p))
    cv_scores[name] = np.mean(aucs)
    print(f'{name} fold9-CV AUC={np.mean(aucs):.4f}±{np.std(aucs):.4f}')

# alpha融合 (fold9网格)
best_a, best_cv = 0, 0
for a in np.arange(0, 1.01, 0.05):
    pa = a * D9['CNNT'] + (1 - a) * D9['M3']
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    aucs = [roc_auc_score(y9[te_], pa[te_]) for _, te_ in skf.split(pa, y9)]
    m = np.mean(aucs)
    if m > best_cv: best_cv, best_a = m, a
cv_scores['F3'] = best_cv
print(f'F3 (alpha*CNNT+(1-a)*M3) fold9-CV AUC={best_cv:.4f} (alpha={best_a:.2f})')

sel = max(cv_scores, key=cv_scores.get)
print(f'CV选型: {sel}')

# ── 最终元模型: fold9全量训练 → 内部测试/外部 ──
results = {}
for name, keys in VAR.items():
    p_te, p_ex = fit_meta(F(keys, D9), y9, F(keys, {k: T[k][0] for k in T}), F(keys, {k: T[k][1] for k in T}))
    results[name] = (p_te, p_ex)
results['F3'] = (best_a * c['int_CNNTransformer'] + (1 - best_a) * c['int_M3'],
                 best_a * c['ext_CNNTransformer'] + (1 - best_a) * c['ext_M3'])

def boot_ci(y, p, n=1000):
    r = np.random.RandomState(42); aucs = []
    for _ in range(n):
        i = r.choice(len(y), len(y), replace=True)
        if len(np.unique(y[i])) < 2: continue
        aucs.append(roc_auc_score(y[i], p[i]))
    return np.percentile(aucs, [2.5, 97.5])

def paired_p(y, p1, p2, n=1000):
    r = np.random.RandomState(42); diffs = []
    for _ in range(n):
        i = r.choice(len(y), len(y), replace=True)
        if len(np.unique(y[i])) < 2: continue
        diffs.append(roc_auc_score(y[i], p1[i]) - roc_auc_score(y[i], p2[i]))
    diffs = np.array(diffs)
    return 2 * (1 - stats.norm.cdf(abs(diffs.mean()) / (diffs.std() + 1e-12)))

def youden(y, p):
    fpr, tpr, ths = roc_curve(y, p)
    th = ths[np.argmax(tpr - fpr)]
    tn, fp, fn, tp = confusion_matrix(y, (p >= th).astype(int)).ravel()
    return tp / (tp + fn), tn / (tn + fp)

rows = []
for name in ['M1', 'M2', 'M3', 'CNN', 'ResNet', 'CNNT']:
    ai, ae = roc_auc_score(y_te, T[name][0]), roc_auc_score(y_ext, T[name][1])
    li, hi = boot_ci(y_te, T[name][0]); le, he = boot_ci(y_ext, T[name][1])
    si, sp = youden(y_te, T[name][0]); se, ep = youden(y_ext, T[name][1])
    rows.append(dict(Model=name, CV9='', Selected='', Int_AUC=round(ai, 4), Int_CI=f'{li:.3f}-{hi:.3f}',
                     Ext_AUC=round(ae, 4), Ext_CI=f'{le:.3f}-{he:.3f}',
                     Int_Sens=round(si, 3), Int_Spec=round(sp, 3), Ext_Sens=round(se, 3), Ext_Spec=round(ep, 3)))

for name in ['F1', 'F2', 'F3', 'F4']:
    pt, pe = results[name]
    ai, ae = roc_auc_score(y_te, pt), roc_auc_score(y_ext, pe)
    li, hi = boot_ci(y_te, pt); le, he = boot_ci(y_ext, pe)
    si, sp = youden(y_te, pt); se, ep = youden(y_ext, pe)
    p_int = paired_p(y_te, pt, c['int_CNNTransformer'])
    p_ext = paired_p(y_ext, pe, c['ext_CNNTransformer'])
    rows.append(dict(Model=name, CV9=round(cv_scores.get(name, np.nan), 4),
                     Selected='*' if name == sel else '',
                     Int_AUC=round(ai, 4), Int_CI=f'{li:.3f}-{hi:.3f}',
                     Ext_AUC=round(ae, 4), Ext_CI=f'{le:.3f}-{he:.3f}',
                     Int_Sens=round(si, 3), Int_Spec=round(sp, 3), Ext_Sens=round(se, 3), Ext_Spec=round(ep, 3),
                     P_int=f'{p_int:.4f}', P_ext=f'{p_ext:.4f}'))
    print(f'{name}: 内部{ai:.4f} 外部{ae:.4f} | vs CNN-T int P={p_int:.4f} ext P={p_ext:.4f}')

df = pd.DataFrame(rows)
print('\n=== 融合结果 (★=fold9 CV选型) ===')
print(df.to_string(index=False))
df.to_csv(f'{OUT}\\fusion_results.csv', index=False, encoding='utf-8-sig')
np.savez_compressed(f'{OUT}\\fusion_probs.npz', y_te=y_te, y_ext=y_ext,
                    **{f'int_{k}': v[0] for k, v in results.items()},
                    **{f'ext_{k}': v[1] for k, v in results.items()})
print('\n保存: fusion_results.csv, fusion_probs.npz')
