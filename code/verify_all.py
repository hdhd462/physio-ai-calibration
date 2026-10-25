import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA_DIR, PROJ_DIR, FIG_DIR, SUB_DIR, PATCH_DIR, NHANES_DIR, PROJ1_DIR, SHAOXING_DIR, PTBXL_DIR
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

OUT = DATA_DIR
npz = lambda f: np.load(f'{OUT}\\{f}', allow_pickle=True)
PASS, FAIL = [], []

def check(name, cond, detail=''):
    (PASS if cond else FAIL).append(name)
    print(f'  [{"PASS" if cond else "FAIL"}] {name} {detail}')

print('=== 1. Route A: npz反算AUC vs table_routeA_results.csv ===')
a, tA = npz('routeA_probs.npz'), pd.read_csv(f'{OUT}\\table_routeA_results.csv')
for m in ['CNN', 'ResNet', 'CNNTransformer']:
    ai, ae = roc_auc_score(a['y_test'], a[f'test_{m}']), roc_auc_score(a['y_ext'], a[f'ext_{m}'])
    row = tA[tA.name == m].iloc[0]
    check(f'RouteA {m}', abs(ai - row['internal_auc']) < 1e-4 and abs(ae - row['external_auc']) < 1e-4,
          f'npz=({ai:.4f},{ae:.4f}) csv=({row["internal_auc"]:.4f},{row["external_auc"]:.4f})')

print('\n=== 2. Route B: npz反算AUC vs table_routeB_results.csv ===')
b, tB = npz('routeB_probs.npz'), pd.read_csv(f'{OUT}\\table_routeB_results.csv')
for m in ['M1', 'M2', 'M3']:
    ai, ae = roc_auc_score(b['test_y'], b[f'test_{m}']), roc_auc_score(b['ext_y'], b[f'ext_{m}'])
    row = tB[tB.Model == m].iloc[0]
    check(f'RouteB {m}', abs(ai - row['Internal_AUC']) < 1e-4 and abs(ae - row['External_AUC']) < 1e-4,
          f'npz=({ai:.4f},{ae:.4f}) csv=({row["Internal_AUC"]:.4f},{row["External_AUC"]:.4f})')

print('\n=== 3. Table 8: npz反算AUC vs csv（含融合） ===')
c, fs, fe, t8 = npz('common_probs.npz'), npz('fusion_probs.npz'), npz('fusion3_probs.npz'), pd.read_csv(f'{OUT}\\table8_ab_comparison.csv')
check('标签对齐', np.array_equal(c['y_te'], fs['y_te']) and np.array_equal(c['y_ext'], fe['y_ext']))
SRC = {**{k: (c[f'int_{k}'], c[f'ext_{k}']) for k in ['M1', 'M2', 'M3', 'CNN', 'ResNet', 'CNNTransformer']},
       'FusionStack': (fs['int_F4'], fs['ext_F4']), 'FusionNet': (fe['int'], fe['ext'])}
for m, (pi, pe) in SRC.items():
    ai, ae = roc_auc_score(c['y_te'], pi), roc_auc_score(c['y_ext'], pe)
    row = t8[t8.Model == m].iloc[0]
    check(f'Table8 {m}', abs(ai - row['Int_AUC']) < 5e-4 and abs(ae - row['Ext_AUC']) < 5e-4,
          f'npz=({ai:.4f},{ae:.4f}) csv=({row["Int_AUC"]:.3f},{row["Ext_AUC"]:.3f})')

print('\n=== 4. 融合CSV vs npz ===')
f1, f2 = npz('fusion_probs.npz'), npz('fusion2_probs.npz')
tF = pd.read_csv(f'{OUT}\\fusion_results.csv')
for m in ['F1', 'F2', 'F3', 'F4']:
    ai, ae = roc_auc_score(c['y_te'], f1[f'int_{m}']), roc_auc_score(c['y_ext'], f1[f'ext_{m}'])
    row = tF[tF.Model == m].iloc[0]
    check(f'fusion {m}', abs(ai - row['Int_AUC']) < 5e-4 and abs(ae - row['Ext_AUC']) < 5e-4,
          f'npz=({ai:.4f},{ae:.4f})')
tG = pd.read_csv(f'{OUT}\\fusion2_results.csv')
ai, ae = roc_auc_score(c['y_te'], f2['sel_int']), roc_auc_score(c['y_ext'], f2['sel_ext'])
row = tG[tG.Model.str.startswith('G3')].iloc[0]
check('fusion2 G3', abs(ai - row['Int']) < 5e-4 and abs(ae - row['Ext']) < 5e-4, f'npz=({ai:.4f},{ae:.4f})')
ai, ae = roc_auc_score(fe['y_te'], fe['int']), roc_auc_score(fe['y_ext'], fe['ext'])
check('fusion3 e2e', abs(ai - 0.9868) < 1e-4 and abs(ae - 0.8389) < 1e-4, f'({ai:.4f},{ae:.4f})')

print('\n=== 5. freestat基准值对照 (2026-10-13验证) ===')
REF = {('M1', 'int'): 0.766589, ('M2', 'int'): 0.967173, ('M3', 'int'): 0.970088,
       ('M1', 'ext'): 0.643254, ('M2', 'ext'): 0.811443, ('M3', 'ext'): 0.810863}
for (m, side), ref in REF.items():
    p = b[f'{"test" if side == "int" else "ext"}_{m}']; y = b['test_y' if side == 'int' else 'ext_y']
    auc = roc_auc_score(y, p)
    check(f'freestat {m}-{side}', abs(auc - ref) < 5e-4, f'{auc:.6f} vs {ref:.6f}')

print('\n=== 6. 备份对比（确定性脚本应完全一致） ===')
D = f'{OUT}\\verify_backup'
pairs = [('table_routeB_results.csv', 'table_routeB_results.csv', True),
         ('table7_logistic_or.csv', 'table7_logistic_or.csv', True),
         ('tableS_subgroup_auc.csv', 'tableS_subgroup_auc.csv', True),
         ('shap_lead_importance.csv', 'shap_lead_importance.csv', True),
         ('fusion2_results.csv', 'fusion2_results.csv', True)]
for old_f, new_f, strict in pairs:
    o = pd.read_csv(f'{D}\\{old_f}'); n = pd.read_csv(f'{OUT}\\{new_f}')
    same = o.equals(n)
    check(f'{new_f} 与备份一致', same if strict else same, '' if same else '有差异(种子重训预期)')
t8o = pd.read_csv(f'{D}\\table8_ab_comparison.csv')
merged = t8o.merge(t8o.iloc[:, 0], on=t8o.columns[0])  # 占位
base_ok = t8o.iloc[:, :12].reset_index(drop=True).equals(
    t8[t8.Model.isin(t8o.Model)].iloc[:, :12].reset_index(drop=True))
check('table8基础6行与备份一致', base_ok)

print(f'\n========== 结果: {len(PASS)} PASS / {len(FAIL)} FAIL ==========')
if FAIL:
    print('FAIL项:', FAIL)
