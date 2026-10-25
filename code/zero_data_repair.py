# -*- coding: utf-8 -*-
"""Zero-data prevalence-only repair (label-shift prior correction).

odds' = odds * [pit/(1-pit)] / [pi0/(1-pi0)]  <=>  logit(p') = logit(p) + logit(pit) - logit(pi0)

pi0 (development prior, estimated WITHOUT labels at deployment):
  primary   = mean model prediction on internal validation set (fold 9) -- model-implied prior
  secondary = true fold-9 prevalence (reference variant)
pit (target prevalence): observed external prevalence (in deployment sourced from local
  epidemiology/dataset documentation -- zero labelled target records used).
AUC is unchanged by construction (strictly monotone map).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA_DIR, PROJ_DIR, FIG_DIR, SUB_DIR, PATCH_DIR, NHANES_DIR, PROJ1_DIR, SHAOXING_DIR, PTBXL_DIR
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, DATA_DIR)
import numpy as np
from eval_multi import metrics, logit, EPS

OUT = DATA_DIR
COLS = ['AF', 'RBBB', 'LBBB', 'IAVB', 'NORM']

probs = np.load(f'{OUT}\\fusion_multi_probs.npz', allow_pickle=True)
P_va, Y_va = probs['P_va'], probs['Y_va']
pi0_mean = P_va.mean(axis=0)          # model-implied development prior (unlabeled)
pi_dev = Y_va.mean(axis=0)            # true development prevalence (reference variant)
print('dev priors per class:')
for k, c in enumerate(COLS):
    print(f'  {c}: mean(p_val)={pi0_mean[k]:.4f}  prev(val)={pi_dev[k]:.4f}')

ext = np.load(f'{OUT}\\fusion_multi_external.npz', allow_pickle=True)

def prior_fix(p, pi0, pit):
    return 1.0 / (1.0 + np.exp(-(logit(p) + logit(pit) - logit(pi0))))

cohorts = {}
ch = np.load(f'{OUT}\\chapman_multilabel.npz', allow_pickle=True)
idx = ext['chapman_idx']
cohorts['chapman'] = (ext['chapman_P'], ch['Y'][idx])
cd = np.load(f'{OUT}\\code15_multilabel.npz', allow_pickle=True)
idx = ext['code15_idx']
cohorts['code15'] = (ext['code15_P'], cd['Y'][idx])

saved = {}
print('\n=== zero-data prevalence-only repair vs labelled repairs ===')
print(f'{"cohort":9s} {"cls":5s} {"prev_ext":>8s} {"pi0m":>6s} | '
      f'{"ECE raw":>7s} {"prior":>7s} {"priorD":>7s} {"Platt":>7s} {"beta":>7s} | '
      f'{"slope raw":>9s} {"prior":>6s} | {"int raw":>8s} {"prior":>7s}')
for coh, (P, Y) in cohorts.items():
    for k, cls in enumerate(COLS):
        y, p = Y[:, k].astype(int), P[:, k]
        pit = y.mean()
        m_raw = metrics(y, p)
        m_raw_clip = metrics(y, np.clip(p, EPS, 1 - EPS))  # tie-consistent baseline
        p_pm = prior_fix(p, pi0_mean[k], pit)
        p_pd = prior_fix(p, pi_dev[k], pit)
        m_pm, m_pd = metrics(y, p_pm), metrics(y, p_pd)
        m_pl = {'ece': ext[f'{coh}_{cls}_platt_ece'], 'slope': ext[f'{coh}_{cls}_platt_slope']}
        m_be = {'ece': ext[f'{coh}_{cls}_beta_ece']}
        print(f'{coh:9s} {cls:5s} {pit*100:7.2f}% {pi0_mean[k]:6.3f} | '
              f'{m_raw["ece"]:7.4f} {m_pm["ece"]:7.4f} {m_pd["ece"]:7.4f} {m_pl["ece"]:7.4f} {m_be["ece"]:7.4f} | '
              f'{m_raw["slope"]:9.3f} {m_pm["slope"]:6.3f} | '
              f'{m_raw["intercept"]:8.3f} {m_pm["intercept"]:7.3f}')
        d = abs(m_pm['auc'] - m_raw_clip['auc'])
        assert d < 1e-6, f'AUC changed under prior correction! delta={d:.2e}'
        for tag, m, pp in [('raw', m_raw, p), ('prior_meanp', m_pm, p_pm),
                           ('prior_devprev', m_pd, p_pd)]:
            for key, val in m.items():
                saved[f'{coh}_{cls}_{tag}_{key}'] = val
            saved[f'{coh}_{cls}_{tag}_p'] = pp

np.savez_compressed(f'{OUT}\\zero_data_repair.npz', **saved)
print('\nsaved: zero_data_repair.npz')
