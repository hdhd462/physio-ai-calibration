import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA_DIR, PROJ_DIR, FIG_DIR, SUB_DIR, PATCH_DIR, NHANES_DIR, PROJ1_DIR, SHAOXING_DIR, PTBXL_DIR
import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
import ast
import wfdb
from scipy.signal import butter, filtfilt
import os, time

BASE = PTBXL_DIR
OUT = DATA_DIR
os.makedirs(OUT, exist_ok=True)

FS = 100
SIG_LEN = 1000

def bandpass(x, fs=100, low=0.5, high=40, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, x, axis=0)

def quality_ok(sig):
    if np.any(np.isnan(sig)):
        if np.mean(np.isnan(sig)) > 0.01:
            return False
        sig = np.nan_to_num(sig, nan=0.0)
    flat = np.std(sig, axis=0) < 1e-6
    if flat.sum() >= 6:
        return False
    if np.mean(np.abs(sig) > 10) > 0.05:
        return False
    return True

t0 = time.time()
df = pd.read_csv(f'{BASE}\\ptbxl_database.csv')
print(f'元数据: {len(df)} 条')

df['scp'] = df['scp_codes'].apply(ast.literal_eval)
df['AF'] = df['scp'].apply(lambda x: 1 if 'AFIB' in x else 0)
df['flutter'] = df['scp'].apply(lambda x: 1 if 'FLUT' in x else 0)
print(f'AF阳性: {df.AF.sum()} ({df.AF.mean()*100:.1f}%) | 房扑: {df.flutter.sum()}')
print(f'sex取值: {df.sex.value_counts().to_dict()} | age范围: {df.age.min()}-{df.age.max()}')

X_list, y_list, age_list, sex_list, fold_list, id_list = [], [], [], [], [], []
fail = 0

for i, row in enumerate(df.itertuples()):
    path = f'{BASE}\\{row.filename_lr}'
    try:
        rec = wfdb.rdrecord(path)
        sig = rec.p_signal.astype(np.float64)
        if sig.shape != (SIG_LEN, 12):
            fail += 1
            continue
        if not quality_ok(sig):
            fail += 1
            continue
        sig = bandpass(sig, FS)
        X_list.append(sig.T.astype(np.float32))
        y_list.append(row.AF)
        age_list.append(row.age if pd.notna(row.age) else np.nan)
        sex_list.append(row.sex)
        fold_list.append(row.strat_fold)
        id_list.append(row.ecg_id)
    except Exception:
        fail += 1
    if (i + 1) % 3000 == 0:
        print(f'  进度 {i+1}/{len(df)}, 有效={len(X_list)}, 失败={fail}, {time.time()-t0:.0f}s')

X = np.stack(X_list)
y = np.array(y_list, dtype=np.int8)
age = np.array(age_list, dtype=np.float32)
sex = np.array(sex_list, dtype=np.int8)
fold = np.array(fold_list, dtype=np.int8)
ids = np.array(id_list, dtype=np.int32)

np.savez_compressed(f'{OUT}\\ptbxl_100hz.npz', X=X, y=y, age=age, sex=sex, fold=fold, ecg_id=ids)
print(f'\n完成: X={X.shape}, AF={y.sum()} ({y.mean()*100:.1f}%), 失败={fail}')
print(f'保存: {OUT}\\ptbxl_100hz.npz, 耗时 {time.time()-t0:.0f}s')
