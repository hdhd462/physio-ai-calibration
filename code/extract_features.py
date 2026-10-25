import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA_DIR, PROJ_DIR, FIG_DIR, SUB_DIR, PATCH_DIR, NHANES_DIR, PROJ1_DIR, SHAOXING_DIR, PTBXL_DIR
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
import neurokit2 as nk
from scipy.signal import welch
from multiprocessing import Pool
import os, time, warnings
warnings.filterwarnings('ignore')

OUT = DATA_DIR
FS = 100

def extract_one(args):
    idx, sig12, y, age, sex = args
    try:
        # Lead II (index 1) 为主要导联
        x = sig12[1, :].astype(float)
        # R峰检测（注意：索引在info中，peaks返回的是二值标记数组）
        peaks, info = nk.ecg_peaks(x, sampling_rate=FS, correct_artifacts=True)
        r_idx = np.asarray(info['ECG_R_Peaks'])
        if len(r_idx) < 3:
            return None
        # RR间期
        rr = np.diff(r_idx) / FS * 1000  # ms
        if len(rr) < 2:
            return None
        # HRV时域
        rmssd = np.sqrt(np.mean(np.diff(rr) ** 2))
        sdnn = np.std(rr)
        pnn50 = np.mean(np.abs(np.diff(rr)) > 50) * 100
        rr_mean = np.mean(rr)
        rr_std = np.std(rr)
        hr_mean = 60000 / rr_mean if rr_mean > 0 else np.nan
        hr_std = np.std(60000 / rr) if len(rr) > 0 else np.nan
        # RR不规则系数
        cv_rr = rr_std / rr_mean if rr_mean > 0 else np.nan
        rmssd_norm = rmssd / rr_mean if rr_mean > 0 else np.nan
        # 频域（RR序列）
        if len(rr) >= 8:
            f, pxx = welch(rr, fs=1.0/ (np.median(rr)/1000) if np.median(rr)>0 else 1, nperseg=min(len(rr), 32))
            dom_freq = f[np.argmax(pxx)] if len(f) > 0 else np.nan
            spec_ent = -np.sum((pxx/np.sum(pxx)+1e-12) * np.log(pxx/np.sum(pxx)+1e-12)) if np.sum(pxx)>0 else np.nan
        else:
            dom_freq = np.nan
            spec_ent = np.nan
        # ECG波形特征（neurokit2，QRS起止为ECG_R_Onsets/Offsets）
        try:
            _, info2 = nk.ecg_process(x, sampling_rate=FS)
            qrs_dur = np.nan
            p_amp = np.nan
            pr_int = np.nan
            qt_int = np.nan
            # QRS时长（R_Onsets~R_Offsets）
            if 'ECG_R_Onsets' in info2 and 'ECG_R_Offsets' in info2:
                on = np.asarray(info2['ECG_R_Onsets'], dtype=float)
                off = np.asarray(info2['ECG_R_Offsets'], dtype=float)
                valid = (~np.isnan(on)) & (~np.isnan(off))
                if valid.sum() > 0:
                    qrs_dur = np.mean((off - on)[valid]) / FS * 1000
            # PR间期（P_Onsets~R_Onsets）
            if 'ECG_P_Onsets' in info2 and 'ECG_R_Onsets' in info2:
                pon = np.asarray(info2['ECG_P_Onsets'], dtype=float)
                ron = np.asarray(info2['ECG_R_Onsets'], dtype=float)
                valid = (~np.isnan(pon)) & (~np.isnan(ron))
                if valid.sum() > 0:
                    pr_int = np.mean((ron - pon)[valid]) / FS * 1000
            # QT间期（Q_Peaks~T_Offsets）
            if 'ECG_Q_Peaks' in info2 and 'ECG_T_Offsets' in info2:
                q = np.asarray(info2['ECG_Q_Peaks'], dtype=float)
                to = np.asarray(info2['ECG_T_Offsets'], dtype=float)
                valid = (~np.isnan(q)) & (~np.isnan(to))
                if valid.sum() > 0:
                    qt_int = np.mean((to - q)[valid]) / FS * 1000
            # P波振幅
            if 'ECG_P_Peaks' in info2:
                pp = np.asarray(info2['ECG_P_Peaks'], dtype=float)
                valid = ~np.isnan(pp)
                if valid.sum() > 0:
                    p_amp = np.nanmean(x[pp[valid].astype(int)])
        except Exception:
            pass
        # 多导联R波振幅（V2导联 index 7）
        x_v2 = sig12[7, :].astype(float)
        r_amp_v2 = np.std(x_v2)
        # I导联（index 0）std
        r_amp_i = np.std(sig12[0, :].astype(float))
        return {
            'idx': idx, 'AF': y, 'age': age, 'sex': sex,
            'hr_mean': hr_mean, 'hr_std': hr_std,
            'rr_mean': rr_mean, 'rr_std': rr_std,
            'rmssd': rmssd, 'sdnn': sdnn, 'pnn50': pnn50,
            'cv_rr': cv_rr, 'rmssd_norm': rmssd_norm,
            'dom_freq': dom_freq, 'spec_ent': spec_ent,
            'qrs_dur': qrs_dur, 'pr_int': pr_int, 'qt_int': qt_int,
            'p_amp': p_amp,
            'sig_std_v2': r_amp_v2, 'sig_std_i': r_amp_i,
            'n_peaks': len(r_idx),
        }
    except Exception:
        return None

def run_extract(npz_name, csv_name, max_n=None):
    t0 = time.time()
    d = np.load(f'{OUT}\\{npz_name}', allow_pickle=True)
    X, y, age, sex = d['X'], d['y'], d['age'], d['sex']
    n = len(y)
    if max_n:
        n = min(n, max_n)
    print(f'{npz_name}: n={n}, 开始特征提取...')
    args = [(i, X[i], int(y[i]), float(age[i]) if not np.isnan(age[i]) else np.nan, int(sex[i]))
            for i in range(n)]
    with Pool(processes=12) as pool:
        results = []
        for i, r in enumerate(pool.imap_unordered(extract_one, args, chunksize=32)):
            if r is not None:
                results.append(r)
            if (i + 1) % 3000 == 0:
                print(f'  进度 {i+1}/{n}, 有效={len(results)}, {time.time()-t0:.0f}s')
    df = pd.DataFrame(results)
    df.to_csv(f'{OUT}\\{csv_name}', index=False, encoding='utf-8-sig')
    print(f'完成: {len(df)}/{n} 有效, AF={df.AF.sum()}, 保存 {csv_name}, 耗时 {time.time()-t0:.0f}s')
    return df

if __name__ == '__main__':
    run_extract('ptbxl_100hz.npz', 'ecg_features_ptbxl.csv')
    run_extract('chapman_100hz.npz', 'ecg_features_chapman.csv')
