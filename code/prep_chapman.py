import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA_DIR, PROJ_DIR, FIG_DIR, SUB_DIR, PATCH_DIR, NHANES_DIR, PROJ1_DIR, SHAOXING_DIR, PTBXL_DIR
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import wfdb
from scipy.signal import butter, filtfilt, resample_poly
import os, time, glob, re

ROOT = SHAOXING_DIR
OUT = DATA_DIR
os.makedirs(OUT, exist_ok=True)

FS_SRC = 500
FS_DST = 100
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

def parse_comments(comments):
    age = np.nan
    sex = 0
    dx = []
    for c in comments:
        if c.startswith('Age:'):
            try:
                age = float(c.split(':', 1)[1].strip())
            except ValueError:
                pass
        elif c.startswith('Sex:'):
            v = c.split(':', 1)[1].strip().lower()
            sex = 1 if v == 'female' else 0
        elif c.startswith('Dx:'):
            dx = [s.strip() for s in c.split(':', 1)[1].split(',')]
    return age, sex, dx

t0 = time.time()
hea_files = glob.glob(f'{ROOT}\\WFDBRecords\\**\\*.hea', recursive=True)
print(f'找到 {len(hea_files)} 条记录（仅顶层WFDBRecords）')

X_list, y_list, age_list, sex_list, id_list, flut_list = [], [], [], [], [], []
fail = 0
AF_CODE = '164889003'
FLUT_CODE = '164890007'

for i, hea in enumerate(hea_files):
    try:
        rec = wfdb.rdrecord(hea[:-4])
        sig = rec.p_signal.astype(np.float64)
        if sig.shape != (FS_SRC * 10, 12):
            fail += 1
            continue
        if not quality_ok(sig):
            fail += 1
            continue
        sig = resample_poly(sig, FS_DST, FS_SRC, axis=0)
        if sig.shape[0] != SIG_LEN:
            fail += 1
            continue
        sig = bandpass(sig, FS_DST)
        age, sex, dx = parse_comments(rec.comments)
        X_list.append(sig.T.astype(np.float32))
        y_list.append(1 if AF_CODE in dx else 0)
        flut_list.append(1 if FLUT_CODE in dx else 0)
        age_list.append(age)
        sex_list.append(sex)
        id_list.append(os.path.basename(hea)[:-4])
    except Exception:
        fail += 1
    if (i + 1) % 3000 == 0:
        print(f'  进度 {i+1}/{len(hea_files)}, 有效={len(X_list)}, 失败={fail}, {time.time()-t0:.0f}s')

X = np.stack(X_list)
y = np.array(y_list, dtype=np.int8)
flut = np.array(flut_list, dtype=np.int8)
age = np.array(age_list, dtype=np.float32)
sex = np.array(sex_list, dtype=np.int8)

np.savez_compressed(f'{OUT}\\chapman_100hz.npz', X=X, y=y, flutter=flut, age=age, sex=sex, rec_id=np.array(id_list))
print(f'\n完成: X={X.shape}, AF={y.sum()} ({y.mean()*100:.1f}%), 房扑={flut.sum()}, 失败={fail}')
print(f'age缺失: {np.isnan(age).sum()}, sex分布: 男={(sex==0).sum()} 女={(sex==1).sum()}')
print(f'保存: {OUT}\\chapman_100hz.npz, 耗时 {time.time()-t0:.0f}s')
