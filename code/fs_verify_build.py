import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA_DIR, PROJ_DIR, FIG_DIR, SUB_DIR, PATCH_DIR, NHANES_DIR, PROJ1_DIR, SHAOXING_DIR, PTBXL_DIR
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd

OUT = DATA_DIR

ptb = pd.read_csv(f'{OUT}\\ecg_features_ptbxl.csv')
npz = np.load(f'{OUT}\\ptbxl_100hz.npz', allow_pickle=True)
ptb['fold'] = npz['fold'][ptb['idx'].values]
chp = pd.read_csv(f'{OUT}\\ecg_features_chapman.csv')

DEMO = ['age', 'sex']
HRV = ['hr_std', 'rr_mean', 'rmssd', 'sdnn', 'pnn50', 'cv_rr', 'rmssd_norm', 'dom_freq', 'spec_ent']
MORPH = ['qrs_dur', 'pr_int', 'qt_int', 'p_amp', 'sig_std_v2', 'sig_std_i', 'n_peaks']
FEATS = DEMO + HRV + MORPH

train = ptb[ptb.fold <= 9].reset_index(drop=True)
med = train[FEATS].median()
Xtr = train[FEATS].fillna(med)
mu, sd = Xtr.mean(), Xtr.std(ddof=0)
sd = sd.replace(0, 1)
Ztr = (Xtr - mu) / sd

y = train['AF'].values.astype(int)
p = y.mean()
w = np.where(y == 1, (1 - p) / p, p / (1 - p))

fs_train = pd.DataFrame({'AF': y, 'w': np.round(w, 6)})
for f in FEATS:
    fs_train['z_' + f] = np.round(Ztr[f].values, 6)
fs_train.to_csv(f'{OUT}\\fs_train_z.csv', index=False, encoding='utf-8-sig')
print(f'fs_train_z.csv: n={len(fs_train)}, AF={y.sum()}, 列数={len(fs_train.columns)}')

b = np.load(f'{OUT}\\routeB_probs.npz', allow_pickle=True)
test_df = ptb[ptb.fold == 10].reset_index(drop=True)
assert len(test_df) == len(b['test_y']), f"fold10 n={len(test_df)} vs probs n={len(b['test_y'])}"

fs_test = pd.DataFrame({'AF': b['test_y'].astype(int)})
for m in ['M1', 'M2', 'M3']:
    fs_test[f'pred_{m}'] = np.round(b[f'test_{m}'], 6)
fs_test.to_csv(f'{OUT}\\fs_test_preds.csv', index=False, encoding='utf-8-sig')
print(f'fs_test_preds.csv: n={len(fs_test)}, AF={int(fs_test.AF.sum())}')

assert len(chp) == len(b['ext_y']), f"chapman n={len(chp)} vs probs n={len(b['ext_y'])}"
fs_chp = pd.DataFrame({'AF': b['ext_y'].astype(int)})
for m in ['M1', 'M2', 'M3']:
    fs_chp[f'pred_{m}'] = np.round(b[f'ext_{m}'], 6)
fs_chp.to_csv(f'{OUT}\\fs_chapman_preds.csv', index=False, encoding='utf-8-sig')
print(f'fs_chapman_preds.csv: n={len(fs_chp)}, AF={int(fs_chp.AF.sum())}')
print('完成')