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
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score, roc_curve
from scipy import stats
import warnings, json
warnings.filterwarnings('ignore')

OUT = DATA_DIR

ptb = pd.read_csv(f'{OUT}\\ecg_features_ptbxl.csv')
chp = pd.read_csv(f'{OUT}\\ecg_features_chapman.csv')
npz = np.load(f'{OUT}\\ptbxl_100hz.npz', allow_pickle=True)
ptb['fold'] = npz['fold'][ptb['idx'].values]
print(f'PTB-XL: {len(ptb)} (AF={ptb.AF.sum()}) | Chapman: {len(chp)} (AF={chp.AF.sum()})')

DEMO = ['age', 'sex']
HRV = ['hr_mean', 'hr_std', 'rr_mean', 'rr_std', 'rmssd', 'sdnn', 'pnn50',
       'cv_rr', 'rmssd_norm', 'dom_freq', 'spec_ent']
MORPH = ['qrs_dur', 'pr_int', 'qt_int', 'p_amp', 'sig_std_v2', 'sig_std_i', 'n_peaks']
ALL = HRV + MORPH
MODELS = {
    'M1': DEMO,
    'M2': DEMO + HRV,
    'M3': DEMO + ALL,
}

def fit_predict(train_df, test_df, feats):
    tr = train_df[feats + ['AF']]
    te = test_df[feats + ['AF']]
    imp = SimpleImputer(strategy='median')
    Xtr = imp.fit_transform(tr[feats])
    Xte = imp.transform(te[feats])
    sc = StandardScaler()
    Xtr = sc.fit_transform(Xtr)
    Xte = sc.transform(Xte)
    # 平衡权重（AF~7%）
    ytr = tr['AF'].values
    w = np.where(ytr == 1, (1 - ytr.mean()) / ytr.mean(), ytr.mean() / (1 - ytr.mean()))
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(Xtr, ytr, sample_weight=w)
    p = clf.predict_proba(Xte)[:, 1]
    return p, clf, imp, sc

def metrics(y, p, th=None):
    auc = roc_auc_score(y, p)
    if th is None:
        # 约登指数最优阈值
        fpr, tpr, ths = roc_curve(y, p)
        j = tpr - fpr
        th = ths[np.argmax(j)]
    yh = (p >= th).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yh).ravel()
    return dict(auc=round(auc, 4), threshold=round(float(th), 4),
                sens=round(tp/(tp+fn), 3) if tp+fn else np.nan,
                spec=round(tn/(tn+fp), 3) if tn+fp else np.nan,
                ppv=round(tp/(tp+fp), 3) if tp+fp else np.nan,
                npv=round(tn/(tn+fn), 3) if tn+fn else np.nan,
                f1=round(f1_score(y, yh, zero_division=0), 3))

def boot_ci(y, p, n=1000, seed=42):
    rng = np.random.RandomState(seed)
    aucs = []
    for _ in range(n):
        i = rng.choice(len(y), len(y), replace=True)
        if len(np.unique(y[i])) < 2:
            continue
        aucs.append(roc_auc_score(y[i], p[i]))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return f'{p if False else ""}{lo:.3f}-{hi:.3f}'

def auc_diff_p(y, p1, p2, n=1000, seed=42):
    rng = np.random.RandomState(seed)
    diffs = []
    for _ in range(n):
        i = rng.choice(len(y), len(y), replace=True)
        if len(np.unique(y[i])) < 2:
            continue
        diffs.append(roc_auc_score(y[i], p1[i]) - roc_auc_score(y[i], p2[i]))
    diffs = np.array(diffs)
    m, s = diffs.mean(), diffs.std()
    pval = 2 * (1 - stats.norm.cdf(abs(m) / (s + 1e-12)))
    return pval

# 划分 PTB-XL
train_df = ptb[ptb.fold <= 8]
val_df = ptb[ptb.fold == 9]
test_df = ptb[ptb.fold == 10]
print(f'Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}')

results = []
probs_store = {}

for name, feats in MODELS.items():
    # 用train+val训练，test内部测试
    full_train = pd.concat([train_df, val_df])
    p_test, *_ = fit_predict(full_train, test_df, feats)
    p_ext, *_ = fit_predict(full_train, chp, feats)
    m_int = metrics(test_df.AF.values, p_test)
    m_ext = metrics(chp.AF.values, p_ext)
    ci_int = boot_ci(test_df.AF.values, p_test)
    ci_ext = boot_ci(chp.AF.values, p_ext)
    results.append({
        'Model': name, 'Features': len(feats),
        'Internal_AUC': m_int['auc'], 'Internal_CI': ci_int,
        'Internal_Sens': m_int['sens'], 'Internal_Spec': m_int['spec'],
        'External_AUC': m_ext['auc'], 'External_CI': ci_ext,
        'External_Sens': m_ext['sens'], 'External_Spec': m_ext['spec'],
        'dAUC': round(m_int['auc'] - m_ext['auc'], 4),
    })
    probs_store[name] = (p_test, p_ext)
    print(f'{name}: 内部AUC={m_int["auc"]} ({ci_int}) 外部AUC={m_ext["auc"]} ({ci_ext}) dAUC={m_int["auc"]-m_ext["auc"]:.4f}')

df_res = pd.DataFrame(results)
print('\n=== Route B 结果 ===')
print(df_res.to_string(index=False))

# 模型间比较（外部）
print('\n外部AUC比较 (bootstrap P):')
names = list(MODELS.keys())
for i in range(len(names)-1):
    for j in range(i+1, len(names)):
        p1 = probs_store[names[i]][1]
        p2 = probs_store[names[j]][1]
        pv = auc_diff_p(chp.AF.values, p1, p2)
        print(f'  {names[i]} vs {names[j]}: P={pv:.4f}')

df_res.to_csv(f'{OUT}\\table_routeB_results.csv', index=False, encoding='utf-8-sig')
# 保存预测概率（画ROC用）
np.savez_compressed(f'{OUT}\\routeB_probs.npz',
    test_y=test_df.AF.values, ext_y=chp.AF.values,
    **{f'test_{k}': v[0] for k, v in probs_store.items()},
    **{f'ext_{k}': v[1] for k, v in probs_store.items()})
print(f'\n保存: table_routeB_results.csv, routeB_probs.npz')
