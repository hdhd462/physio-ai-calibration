#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A1: Cross-fitted (held-out) recalibration evaluation for FusionNet on Chapman-Shaoxing.

Answers reviewer criticism "Platt/beta fitted and evaluated on the same external set".
Design: 10-fold stratified cross-fitting (seed 42). Platt/Beta fitted on 9 folds,
evaluated on the held-out fold; pooled out-of-fold predictions scored on the full
cohort (n = 22,190). Metric definitions copied verbatim from the existing pipeline:
  - ECE: occupancy-weighted, 10 equal-width bins (cal_dca_fusion definition; matches
    recalibration_stats.csv / cal_dca_fusion_stats.csv bit-for-bit - gated below)
  - slope: LogisticRegression(C=1e6) of outcome on logit(p)
  - NB@t: standard net benefit at single threshold (single-element dca_nb call of
    make_fig_recalibration.py - the 11-point moving average is a no-op there)
Outputs: crossfit_recal_stats.csv, crossfit_recal_params.csv
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA_DIR, PROJ_DIR, FIG_DIR, SUB_DIR, PATCH_DIR, NHANES_DIR, PROJ1_DIR, SHAOXING_DIR, PTBXL_DIR
import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import brier_score_loss, roc_auc_score
from scipy.optimize import minimize

OUT = DATA_DIR
EPS = 1e-6


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


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


def fit_platt(p, y):
    lr = LogisticRegression(C=1e6).fit(logit(p).reshape(-1, 1), y)
    return lr


def apply_platt(lr, p):
    return lr.predict_proba(logit(p).reshape(-1, 1))[:, 1]


def beta_loss(params, p, y):
    a, b, c = params
    z = a * np.log(np.clip(p, EPS, 1 - EPS)) - b * np.log(np.clip(1 - p, EPS, 1 - EPS)) + c
    q = np.clip(sigmoid(z), 1e-8, 1 - 1e-8)
    return -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))


def fit_beta(p, y):
    return minimize(beta_loss, [1, 1, 0], args=(p, y), method="Nelder-Mead").x


def apply_beta(params, p):
    a, b, c = params
    z = a * np.log(np.clip(p, EPS, 1 - EPS)) - b * np.log(np.clip(1 - p, EPS, 1 - EPS)) + c
    return sigmoid(z)


def nb_at(y, p, pt):
    n = len(y)
    status = y == 1
    treat = p >= pt
    tp = (treat & status).sum()
    fp = (treat & ~status).sum()
    return tp / n - fp / n * (pt / (1 - pt))


def metrics(y, p):
    return dict(
        Brier=brier_score_loss(y, p),
        ECE=ece_weighted(y, p),
        calib_slope=calib_slope(y, p),
        NB_5=nb_at(y, p, 0.05),
        NB_10=nb_at(y, p, 0.10),
        NB_20=nb_at(y, p, 0.20),
        AUC=roc_auc_score(y, p),
    )


d = np.load(f"{OUT}\\fusion3_probs.npz")
y, p_raw = d["y_ext"], d["ext"]
assert len(y) == 22190 and int(y.sum()) == 1775, (len(y), int(y.sum()))

# ── GATE 1: reproduce recalibration_stats.csv bit-for-bit (raw/platt/beta in-sample)
csv = pd.read_csv(f"{OUT}\\recalibration_stats.csv").set_index("method")
p_ins_platt = apply_platt(fit_platt(p_raw, y), p_raw)
p_ins_beta = apply_beta(fit_beta(p_raw, y), p_raw)
for name, p in [("Raw", p_raw), ("Platt", p_ins_platt), ("Beta", p_ins_beta)]:
    m = metrics(y, p)
    for col, key in [("Brier", "Brier"), ("ECE", "ECE"), ("calib_slope", "calib_slope"),
                     ("NB@5%", "NB_5"), ("NB@10%", "NB_10"), ("NB@20%", "NB_20"), ("AUC", "AUC")]:
        got, want = m[key], csv.loc[name, col]
        assert np.isclose(got, want, rtol=1e-9, atol=1e-12), f"{name}.{col}: {got} != {want}"
print("GATE 1 PASS: in-sample Raw/Platt/Beta reproduce recalibration_stats.csv (rtol 1e-9)")

# ── A1: 10-fold stratified cross-fitting
K, SEED = 10, 42
skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED)
oof_platt = np.zeros(len(y))
oof_beta = np.zeros(len(y))
rows_params, fold_ece_platt, fold_ece_beta = [], [], []
for k, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y)):
    lr = fit_platt(p_raw[tr], y[tr])
    bp = fit_beta(p_raw[tr], y[tr])
    oof_platt[te] = apply_platt(lr, p_raw[te])
    oof_beta[te] = apply_beta(bp, p_raw[te])
    rows_params.append(dict(fold=k, n_fit=len(tr), events_fit=int(y[tr].sum()),
                            platt_slope=lr.coef_[0][0], platt_intercept=lr.intercept_[0],
                            beta_a=bp[0], beta_b=bp[1], beta_c=bp[2]))
    fold_ece_platt.append(ece_weighted(y[te], oof_platt[te]))
    fold_ece_beta.append(ece_weighted(y[te], oof_beta[te]))
assert (oof_platt > 0).all() and (oof_beta > 0).all()

rows = [dict(method="Raw", fit_scheme="none", eval_set="full external", **metrics(y, p_raw)),
        dict(method="Platt", fit_scheme="in-sample (all 22,190)", eval_set="full external (optimistic)",
             **metrics(y, p_ins_platt)),
        dict(method="Platt", fit_scheme="10-fold cross-fit", eval_set="pooled out-of-fold",
             **metrics(y, oof_platt)),
        dict(method="Beta", fit_scheme="in-sample (all 22,190)", eval_set="full external (optimistic)",
             **metrics(y, p_ins_beta)),
        dict(method="Beta", fit_scheme="10-fold cross-fit", eval_set="pooled out-of-fold",
             **metrics(y, oof_beta))]
stats = pd.DataFrame(rows)
stats.to_csv(f"{OUT}\\crossfit_recal_stats.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(rows_params).to_csv(f"{OUT}\\crossfit_recal_params.csv", index=False, encoding="utf-8-sig")

print("\n=== A1 cross-fitted recalibration (external, n=22,190, AF=1,775) ===")
print(stats.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
print(f"\nper-fold held-out ECE: Platt {np.mean(fold_ece_platt):.6f} +- {np.std(fold_ece_platt):.6f}"
      f" | Beta {np.mean(fold_ece_beta):.6f} +- {np.std(fold_ece_beta):.6f}")
ins, cf = stats[stats.fit_scheme.str.startswith("in")].set_index("method"), \
    stats[stats.fit_scheme == "10-fold cross-fit"].set_index("method")
for m in ["Platt", "Beta"]:
    print(f"optimism {m}: ECE {ins.loc[m,'ECE']:.6f} -> {cf.loc[m,'ECE']:.6f}"
          f" (delta {cf.loc[m,'ECE']-ins.loc[m,'ECE']:+.6f}); "
          f"Brier {ins.loc[m,'Brier']:.6f} -> {cf.loc[m,'Brier']:.6f}; "
          f"NB@20% {ins.loc[m,'NB_20']:.6f} -> {cf.loc[m,'NB_20']:.6f}")
print("\nSaved: crossfit_recal_stats.csv, crossfit_recal_params.csv")
