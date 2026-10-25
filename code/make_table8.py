import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA_DIR, PROJ_DIR, FIG_DIR, SUB_DIR, PATCH_DIR, NHANES_DIR, PROJ1_DIR, SHAOXING_DIR, PTBXL_DIR
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

OUT = DATA_DIR
rng = np.random.RandomState(42)

# ── 加载 ──
a = np.load(f'{OUT}\\routeA_probs.npz', allow_pickle=True)
b = np.load(f'{OUT}\\routeB_probs.npz', allow_pickle=True)
ptb_npz = np.load(f'{OUT}\\ptbxl_100hz.npz', allow_pickle=True)
chp_npz = np.load(f'{OUT}\\chapman_100hz.npz', allow_pickle=True)

fold = ptb_npz['fold']
y_ptb, y_chp = ptb_npz['y'], chp_npz['y']

feat_p = pd.read_csv(f'{OUT}\\ecg_features_ptbxl.csv')
feat_c = pd.read_csv(f'{OUT}\\ecg_features_chapman.csv')

# ── 内部测试集对齐（共同idx） �─
idxA_te = np.where(fold == 10)[0]                     # A顺序 = npz顺序
feat_p['fold'] = fold[feat_p['idx'].values]
fb_te = feat_p.loc[feat_p.fold == 10, 'idx'].values    # B顺序 = csv顺序
common_te = np.intersect1d(idxA_te, fb_te)
posA_te = pd.Index(idxA_te).get_indexer(common_te)
posB_te = pd.Index(fb_te).get_indexer(common_te)
y_te = y_ptb[common_te]
# 校验标签一致
assert np.array_equal(y_te, a['y_test'][posA_te]), 'test标签A不一致'
assert np.array_equal(y_te, b['test_y'][posB_te]), 'test标签B不一致'

# ── 外部测试集对齐 ──
ok = ~np.isnan(chp_npz['X']).any(axis=(1, 2))
idxA_ext = np.where(ok)[0]
fb_ext = feat_c['idx'].values
common_ext = np.intersect1d(idxA_ext, fb_ext)
posA_ext = pd.Index(idxA_ext).get_indexer(common_ext)
posB_ext = pd.Index(fb_ext).get_indexer(common_ext)
y_ext = y_chp[common_ext]
assert np.array_equal(y_ext, a['y_ext'][posA_ext]), 'ext标签A不一致'
assert np.array_equal(y_ext, b['ext_y'][posB_ext]), 'ext标签B不一致'

print(f'内部共同样本: {len(common_te)} (AF={y_te.mean()*100:.1f}%)  外部共同样本: {len(common_ext)} (AF={y_ext.mean()*100:.1f}%)')

P_INT = {'M1': b['test_M1'][posB_te], 'M2': b['test_M2'][posB_te], 'M3': b['test_M3'][posB_te],
         'CNN': a['test_CNN'][posA_te], 'ResNet': a['test_ResNet'][posA_te],
         'CNNTransformer': a['test_CNNTransformer'][posA_te]}
P_EXT = {'M1': b['ext_M1'][posB_ext], 'M2': b['ext_M2'][posB_ext], 'M3': b['ext_M3'][posB_ext],
         'CNN': a['ext_CNN'][posA_ext], 'ResNet': a['ext_ResNet'][posA_ext],
         'CNNTransformer': a['ext_CNNTransformer'][posA_ext]}

# 融合模型（F4 stacking + 端到端融合网络）
fs = np.load(f'{OUT}\\fusion_probs.npz', allow_pickle=True)
fe = np.load(f'{OUT}\\fusion3_probs.npz', allow_pickle=True)
assert np.array_equal(fs['y_te'], y_te) and np.array_equal(fs['y_ext'], y_ext), 'fusion stacking标签不一致'
assert np.array_equal(fe['y_te'], y_te) and np.array_equal(fe['y_ext'], y_ext), 'fusion e2e标签不一致'
P_INT['FusionStack'] = fs['int_F4']; P_EXT['FusionStack'] = fs['ext_F4']
P_INT['FusionNet'] = fe['int']; P_EXT['FusionNet'] = fe['ext']

def boot_ci(y, p, n=1000):
    r = np.random.RandomState(42)
    aucs = []
    for _ in range(n):
        i = r.choice(len(y), len(y), replace=True)
        if len(np.unique(y[i])) < 2:
            continue
        aucs.append(roc_auc_score(y[i], p[i]))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return lo, hi

def paired_p(y, p1, p2, n=1000):
    r = np.random.RandomState(42)
    diffs = []
    for _ in range(n):
        i = r.choice(len(y), len(y), replace=True)
        if len(np.unique(y[i])) < 2:
            continue
        diffs.append(roc_auc_score(y[i], p1[i]) - roc_auc_score(y[i], p2[i]))
    diffs = np.array(diffs)
    return 2 * (1 - stats.norm.cdf(abs(diffs.mean()) / (diffs.std() + 1e-12)))

def youden(y, p):
    fpr, tpr, ths = roc_curve(y, p)
    return ths[np.argmax(tpr - fpr)]

def sens_spec(y, p, th):
    yh = (p >= th).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yh).ravel()
    return tp / (tp + fn), tn / (tn + fp)

META = {
    'M1': ('Logistic regression', 'Demographics'),
    'M2': ('Logistic regression', 'Demographics + HRV'),
    'M3': ('Logistic regression', 'Demographics + HRV + morphology'),
    'CNN': ('1D-CNN', 'Raw ECG'),
    'ResNet': ('ResNet1D', 'Raw ECG'),
    'CNNTransformer': ('CNN-Transformer', 'Raw ECG'),
    'FusionStack': ('Multimodal stacking', 'DL scores + LR scores'),
    'FusionNet': ('Multimodal fusion network', 'Raw ECG + clinical/HRV'),
}

rows = []
for name in ['M1', 'M2', 'M3', 'CNN', 'ResNet', 'CNNTransformer', 'FusionStack', 'FusionNet']:
    pi, pe = P_INT[name], P_EXT[name]
    auc_i, auc_e = roc_auc_score(y_te, pi), roc_auc_score(y_ext, pe)
    lo_i, hi_i = boot_ci(y_te, pi)
    lo_e, hi_e = boot_ci(y_ext, pe)
    th = youden(y_te, pi)                       # 约登阈值来自内部测试
    s_i, p_i = sens_spec(y_te, pi, th)          # 内部
    s_e, p_e = sens_spec(y_ext, pe, th)         # 同一阈值应用于外部
    rows.append(dict(Model=name, Approach=META[name][0], Input=META[name][1],
                     Int_AUC=round(auc_i, 4), Int_CI=f'{lo_i:.3f}-{hi_i:.3f}',
                     Ext_AUC=round(auc_e, 4), Ext_CI=f'{lo_e:.3f}-{hi_e:.3f}',
                     dAUC=round(auc_i - auc_e, 4),
                     Int_Sens=round(s_i, 3), Int_Spec=round(p_i, 3),
                     Ext_Sens=round(s_e, 3), Ext_Spec=round(p_e, 3)))

df = pd.DataFrame(rows)

# 配对检验：DL主模型 vs LR主模型
p_int_M3 = paired_p(y_te, P_INT['CNNTransformer'], P_INT['M3'])
p_ext_M3 = paired_p(y_ext, P_EXT['CNNTransformer'], P_EXT['M3'])
p_ext_RN = paired_p(y_ext, P_EXT['ResNet'], P_EXT['M3'])
p_ext_DL = paired_p(y_ext, P_EXT['CNNTransformer'], P_EXT['ResNet'])
print(f'CNNTransformer vs M3: 内部P={p_int_M3:.4f} 外部P={p_ext_M3:.4f}')
print(f'ResNet vs M3: 外部P={p_ext_RN:.4f} | CNNTransformer vs ResNet: 外部P={p_ext_DL:.4f}')

df['P_vs_M3_int'] = ''
df['P_vs_M3_ext'] = ''
df.loc[df.Model == 'CNNTransformer', 'P_vs_M3_int'] = f'{p_int_M3:.4f}' if p_int_M3 >= 0.0001 else '<0.0001'
df.loc[df.Model == 'CNNTransformer', 'P_vs_M3_ext'] = f'{p_ext_M3:.4f}' if p_ext_M3 >= 0.0001 else '<0.0001'
# 融合模型 vs CNN-T
df['P_vs_CNNT_int'] = ''
df['P_vs_CNNT_ext'] = ''
for fname in ['FusionStack', 'FusionNet']:
    pi = paired_p(y_te, P_INT[fname], P_INT['CNNTransformer'])
    pe = paired_p(y_ext, P_EXT[fname], P_EXT['CNNTransformer'])
    df.loc[df.Model == fname, 'P_vs_CNNT_int'] = f'{pi:.4f}' if pi >= 0.0001 else '<0.0001'
    df.loc[df.Model == fname, 'P_vs_CNNT_ext'] = f'{pe:.4f}' if pe >= 0.0001 else '<0.0001'
    print(f'{fname} vs CNNTransformer: 内部P={pi:.4f} 外部P={pe:.4f}')
p_ext_net_stack = paired_p(y_ext, P_EXT['FusionNet'], P_EXT['FusionStack'])
print(f'FusionNet vs FusionStack: 外部P={p_ext_net_stack:.4f}')
df.to_csv(f'{OUT}\\table8_ab_comparison.csv', index=False, encoding='utf-8-sig')
print('\n=== Table 8 ===')
print(df[['Model', 'Int_AUC', 'Int_CI', 'Ext_AUC', 'Ext_CI', 'dAUC', 'Int_Sens', 'Int_Spec', 'Ext_Sens', 'Ext_Spec']].to_string(index=False))

# 保存共同队列预测（Figure 7用）
np.savez_compressed(f'{OUT}\\common_probs.npz', y_te=y_te, y_ext=y_ext,
                    **{f'int_{k}': v for k, v in P_INT.items()},
                    **{f'ext_{k}': v for k, v in P_EXT.items()})
print('\n保存: table8_ab_comparison.csv, common_probs.npz')