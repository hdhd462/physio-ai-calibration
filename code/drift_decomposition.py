#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A3: Quantify the cross-geographic calibration drift (review point 2a/2b).

Part 1 - FusionNet drift decomposition ladder on Chapman-Shaoxing:
    raw ECE -> prevalence-difference-only intercept correction (Bayesian prior shift,
    pi_PTBXL -> pi_Chapman) -> fitted intercept-only (calibration-in-the-large) ->
    Platt (intercept+slope). Attributes the drift to (i) the 1.1-pp prevalence
    difference, (ii) overall score-level shift, (iii) slope/spread misalignment.
Part 2 - SAME model (CNN-Transformer) across all three cohorts:
    PTB-XL vs Chapman-Shaoxing vs CODE-15% - prevalence shift magnitude, score
    saturation, calibration intercept/slope/ECE. Quantifies the
    "China drifts, Brazil does not" contradiction: CODE-15 has a 6x LARGER
    prevalence shift than Chapman yet stays calibrated.
Part 3 - FusionNet score saturation internal vs external.
Outputs: drift_decomposition_stats.csv
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA_DIR, PROJ_DIR, FIG_DIR, SUB_DIR, PATCH_DIR, NHANES_DIR, PROJ1_DIR, SHAOXING_DIR, PTBXL_DIR
import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from scipy.optimize import minimize_scalar

OUT = DATA_DIR
EPS = 1e-6


def logit(x):
    x = np.clip(x, EPS, 1 - EPS)
    return np.log(x / (1 - x))


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))


def ece_weighted(y, p, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(p, bins) - 1
    e = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum():
            e += m.sum() / len(y) * abs(y[m].mean() - p[m].mean())
    return e


def calib_slope(y, p):
    return LogisticRegression(C=1e6).fit(logit(p).reshape(-1, 1), y).coef_[0][0]


def citl_intercept(y, p):
    """calibration-in-the-large: intercept with logit(p) as fixed offset"""
    z = logit(p)
    f = lambda a: -np.mean(y * np.log(sigmoid(z + a) + 1e-12) + (1 - y) * np.log(1 - sigmoid(z + a) + 1e-12))
    return float(minimize_scalar(f, bracket=(-10, 0)).x)


def row(dataset, y, p, extra=None):
    r = dict(dataset=dataset, n=len(y), events=int(y.sum()), prevalence=float(y.mean()),
             mean_pred=float(p.mean()), median_pred=float(np.median(p)),
             p75_pred=float(np.percentile(p, 75)), p95_pred=float(np.percentile(p, 95)),
             pct_ge_09=float((p >= 0.9).mean() * 100), pct_ge_095=float((p >= 0.95).mean() * 100),
             Brier=brier_score_loss(y, p), ECE=ece_weighted(y, p),
             calib_slope=calib_slope(y, p), citl=citl_intercept(y, p),
             AUC=roc_auc_score(y, p))
    if extra:
        r.update(extra)
    return r


rows = []

# ── data ──
f3 = np.load(f"{OUT}\\fusion3_probs.npz")
y_int, p_int_f = f3["y_te"], f3["int"]
y_ext, p_ext_f = f3["y_ext"], f3["ext"]
cm = np.load(f"{OUT}\\common_probs.npz")
assert np.array_equal(cm["y_te"], y_int) and np.array_equal(cm["y_ext"], y_ext)
p_int_cnnt, p_ext_cnnt = cm["int_CNNTransformer"], cm["ext_CNNTransformer"]
cd = np.load(f"{OUT}\\code15_part17_probs.npz", allow_pickle=True)
y_cd, p_cd_cnnt = cd["y"], cd["CNNT"]

# gates vs published numbers
csv_cal = pd.read_csv(f"{OUT}\\cal_dca_fusion_stats.csv").set_index("dataset")
prev_int = float(csv_cal.loc["Internal Test", "AF_prev"])
prev_ext = float(csv_cal.loc["External (Chapman-Shaoxing)", "AF_prev"])
assert np.isclose(ece_weighted(y_ext, p_ext_f), 0.35836415449044035, rtol=1e-9)
assert np.isclose(roc_auc_score(y_ext, p_ext_cnnt), 0.8252, atol=1e-3)
assert np.isclose(roc_auc_score(y_int, p_int_cnnt), 0.9727, atol=1e-3)
assert np.isclose(roc_auc_score(y_cd, p_cd_cnnt), 0.966, atol=1e-3)
assert len(y_cd) == 5694 and int(y_cd.sum()) == 159
print("GATES PASS: raw ECE, CNN-T AUCs (3 cohorts), CODE-15 n/events")

# ── Part 1: decomposition ladder ──
ece_raw = ece_weighted(y_ext, p_ext_f)
d_prev = logit(prev_ext) - logit(prev_int)                       # prevalence-difference-only logit shift
p_prev_only = sigmoid(logit(p_ext_f) + d_prev)                   # Bayesian prior-shift correction
ece_prev_only = ece_weighted(y_ext, p_prev_only)
a_citl = citl_intercept(y_ext, p_ext_f)                          # fitted level shift (slope fixed 1)
p_intercept = sigmoid(logit(p_ext_f) + a_citl)
ece_intercept = ece_weighted(y_ext, p_intercept)
lr = LogisticRegression(C=1e6).fit(logit(p_ext_f).reshape(-1, 1), y_ext)
p_platt = lr.predict_proba(logit(p_ext_f).reshape(-1, 1))[:, 1]
ece_platt = ece_weighted(y_ext, p_platt)
print("\n=== Part 1: FusionNet drift decomposition (Chapman) ===")
print(f"prevalence: PTB-XL {prev_int:.4f} -> Chapman {prev_ext:.4f} "
      f"(delta logit = {d_prev:+.3f})")
print(f"1) raw ECE                          = {ece_raw:.4f}")
print(f"2) prevalence-difference correction = {ece_prev_only:.4f}  (explains "
      f"{100*(ece_raw-ece_prev_only)/ece_raw:.1f}% of drift)")
print(f"3) intercept-only (fitted CITL a={a_citl:+.3f}; prevalence alone implies {d_prev:+.3f}) "
      f"= {ece_intercept:.4f}  (level-shift share {100*(ece_raw-ece_intercept)/ece_raw:.1f}%)")
print(f"4) Platt intercept+slope (slope={lr.coef_[0][0]:.3f}) = {ece_platt:.4f} "
      f"(slope/spread adds {100*(ece_intercept-ece_platt)/ece_raw:.1f}%)")
rows += [
    dict(section="P1_ladder", item="raw", value=ece_raw),
    dict(section="P1_ladder", item="prevalence_diff_only_correction", value=ece_prev_only),
    dict(section="P1_ladder", item="intercept_only_fitted", value=ece_intercept),
    dict(section="P1_ladder", item="platt_full", value=ece_platt),
    dict(section="P1_ladder", item="delta_logit_prevalence", value=d_prev),
    dict(section="P1_ladder", item="fitted_CITL_intercept", value=a_citl),
]

# ── Part 2: same model (CNN-Transformer) across 3 cohorts ──
rows2 = [
    row("CNN-T PTB-XL (source, internal test)", y_int, p_int_cnnt, {"cohort": "Germany (PTB-XL)"}),
    row("CNN-T Chapman-Shaoxing (external)", y_ext, p_ext_cnnt, {"cohort": "China (Chapman)"}),
    row("CNN-T CODE-15% (third cohort)", y_cd, p_cd_cnnt, {"cohort": "Brazil (CODE-15% part17)"}),
]
df2 = pd.DataFrame(rows2)
print("\n=== Part 2: CNN-Transformer across 3 cohorts (same model) ===")
print(df2[["dataset", "n", "events", "prevalence", "mean_pred", "p75_pred", "pct_ge_09",
           "Brier", "ECE", "calib_slope", "citl", "AUC"]].to_string(index=False,
          float_format=lambda v: f"{v:.4f}"))
sh_cn = logit(0.07999099) - logit(0.0694381)
sh_br = logit(y_cd.mean()) - logit(prev_int)
print(f"\nprevalence shift (logit) vs source: China {sh_cn:+.3f} | Brazil {sh_br:+.3f} "
      f"(|Brazil| is {abs(sh_br)/abs(sh_cn):.1f}x larger)")
rows2_summary = [
    dict(section="P2_shift", item="logit_shift_china", value=sh_cn),
    dict(section="P2_shift", item="logit_shift_brazil", value=sh_br),
    dict(section="P2_shift", item="ratio_abs", value=abs(sh_br) / abs(sh_cn)),
]

# ── Part 3: FusionNet saturation internal vs external ──
sat = [row("FusionNet PTB-XL internal", y_int, p_int_f),
       row("FusionNet Chapman external", y_ext, p_ext_f)]
df3 = pd.DataFrame(sat)
print("\n=== Part 3: FusionNet score saturation ===")
print(df3[["dataset", "n", "prevalence", "mean_pred", "median_pred", "p75_pred", "p95_pred",
           "pct_ge_09", "pct_ge_095", "Brier", "ECE", "calib_slope", "AUC"]].to_string(
          index=False, float_format=lambda v: f"{v:.4f}"))
assert np.isclose(np.percentile(p_ext_f, 75), 0.998, atol=2e-3), "plan line 315 p75=0.998"

out = pd.DataFrame(rows + rows2_summary)
out.to_csv(f"{OUT}\\drift_decomposition_stats.csv", index=False, encoding="utf-8-sig")
df2.assign(section="P2_cohorts").to_csv(f"{OUT}\\drift_cohort_table.csv", index=False,
                                        encoding="utf-8-sig")
print("\nSaved: drift_decomposition_stats.csv, drift_cohort_table.csv")
