#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A2: Recalibration sample-size learning curve for FusionNet on Chapman-Shaoxing.

Answers reviewer criticism "'a small local labelled sample' - how small is small?".
Design: stratified 50/50 split of the external cohort (seed 42) into a calibration
FIT pool and a locked held-out EVAL set. For each target size n, draw a
prevalence-stratified subsample from the fit pool (200 repeats for n<=800,
100 repeats for n=1600/3200; identical subsamples used for Platt and Beta),
fit recalibration parameters, and score on the untouched eval set.
Metrics: occupancy-weighted 10-bin ECE, Brier, calibration slope (C=1e6),
net benefit at 10%/20% - definitions identical to the existing pipeline
(gated against recalibration_stats.csv on the full cohort).
Outputs: recal_sample_size_curve.csv (long format, median + quantiles per n)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA_DIR, PROJ_DIR, FIG_DIR, SUB_DIR, PATCH_DIR, NHANES_DIR, PROJ1_DIR, SHAOXING_DIR, PTBXL_DIR
import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import brier_score_loss, roc_auc_score
from scipy.optimize import minimize

OUT = DATA_DIR
EPS = 1e-6
SIZES = [(25, 200), (50, 200), (100, 200), (200, 200), (400, 200),
         (800, 200), (1600, 100), (3200, 100)]  # (n, repeats); full pool added as anchor


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
    return LogisticRegression(C=1e6).fit(logit(p).reshape(-1, 1), y)


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
    treat = p >= pt
    tp = (treat & (y == 1)).sum()
    fp = (treat & (y == 0)).sum()
    return tp / n - fp / n * (pt / (1 - pt))


def eval_all(y, p):
    return dict(ECE=ece_weighted(y, p), Brier=brier_score_loss(y, p),
                calib_slope=calib_slope(y, p),
                NB_10=nb_at(y, p, 0.10), NB_20=nb_at(y, p, 0.20),
                AUC=roc_auc_score(y, p))


d = np.load(f"{OUT}\\fusion3_probs.npz")
y, p_raw = d["y_ext"], d["ext"]
assert len(y) == 22190 and int(y.sum()) == 1775

# GATE: metric definitions reproduce recalibration_stats.csv (raw row)
csv = pd.read_csv(f"{OUT}\\recalibration_stats.csv").set_index("method")
m_raw = eval_all(y, p_raw)
assert np.isclose(m_raw["ECE"], csv.loc["Raw", "ECE"], rtol=1e-9)
assert np.isclose(m_raw["Brier"], csv.loc["Raw", "Brier"], rtol=1e-9)
assert np.isclose(m_raw["calib_slope"], csv.loc["Raw", "calib_slope"], rtol=1e-9)
assert np.isclose(m_raw["NB_20"], csv.loc["Raw", "NB@20%"], rtol=1e-9)
print("GATE PASS: metric definitions reproduce recalibration_stats.csv (raw row)")

# locked 50/50 fit-pool / eval split (stratified, seed 42)
sss = StratifiedShuffleSplit(n_splits=1, train_size=0.5, random_state=42)
pool_idx, eval_idx = next(sss.split(np.zeros(len(y)), y))
y_pool, p_pool = y[pool_idx], p_raw[pool_idx]
y_ev, p_ev = y[eval_idx], p_raw[eval_idx]
prev_pool = y_pool.mean()
print(f"fit pool: n={len(y_pool)} events={int(y_pool.sum())} ({prev_pool:.4f}) | "
      f"locked eval: n={len(y_ev)} events={int(y_ev.sum())} ({y_ev.mean():.4f})")
print(f"raw model on locked eval: ECE={m_raw['ECE']:.6f} (reference line)")


def strat_subsample(n, rng):
    n_ev = max(1, int(round(n * prev_pool)))
    n_ev = min(n_ev, int(y_pool.sum()), n - 1)
    ev = rng.choice(np.flatnonzero(y_pool == 1), n_ev, replace=False)
    ne = rng.choice(np.flatnonzero(y_pool == 0), n - n_ev, replace=False)
    idx = np.concatenate([ev, ne])
    rng.shuffle(idx)
    return idx


records = []
# anchor: full fit pool (single deterministic fit per method)
anchors = {"Platt": apply_platt(fit_platt(p_pool, y_pool), p_ev),
           "Beta": apply_beta(fit_beta(p_pool, y_pool), p_ev)}
for method, p_eval in anchors.items():
    records.append(dict(method=method, n=len(y_pool), reps=1, **eval_all(y_ev, p_eval)))

n_beta_fail = 0
for n, reps in SIZES:
    for rep in range(reps):
        rng = np.random.default_rng(42000 + n * 1000 + rep)   # same subsample for both methods
        idx = strat_subsample(n, rng)
        ys, ps = y_pool[idx], p_pool[idx]
        p_platt_ev = apply_platt(fit_platt(ps, ys), p_ev)
        bp = fit_beta(ps, ys)
        p_beta_ev = apply_beta(bp, p_ev)
        if not np.isfinite(bp).all():
            n_beta_fail += 1
        for method, p_ev_fit in [("Platt", p_platt_ev), ("Beta", p_beta_ev)]:
            records.append(dict(method=method, n=n, reps=reps, **eval_all(y_ev, p_ev_fit)))
    print(f"  n={n}: {reps} repeats done")

rec = pd.DataFrame(records)
long_rows = []
for (method, n), g in rec.groupby(["method", "n"]):
    for metric in ["ECE", "Brier", "calib_slope", "NB_10", "NB_20"]:
        v = g[metric].values
        long_rows.append(dict(method=method, n=n, reps=len(v), eval_n=len(y_ev),
                              eval_events=int(y_ev.sum()), metric=metric,
                              median=float(np.median(v)), p05=float(np.percentile(v, 5)),
                              p25=float(np.percentile(v, 25)), p75=float(np.percentile(v, 75)),
                              p95=float(np.percentile(v, 95)),
                              mean=float(v.mean()), sd=float(v.std(ddof=1)) if len(v) > 1 else 0.0))
curve = pd.DataFrame(long_rows)
curve.to_csv(f"{OUT}\\recal_sample_size_curve.csv", index=False, encoding="utf-8-sig")
rec.to_csv(f"{OUT}\\recal_sample_size_raw.csv", index=False, encoding="utf-8-sig")
if n_beta_fail:
    print(f"WARNING: {n_beta_fail} beta fits with non-finite params")

print("\n=== A2 learning curve: ECE (weighted) on locked eval set, median [p25-p75] ===")
ece = curve[curve.metric == "ECE"].copy()
ece["txt"] = ece.apply(lambda r: f"{r['median']:.4f} [{r['p25']:.4f}-{r['p75']:.4f}]", axis=1)
piv = ece.pivot(index="n", columns="method", values="txt").reset_index()
print(piv.to_string(index=False))
for method in ["Platt", "Beta"]:
    sub = ece[(ece.method == method) & (ece.n <= len(y_pool))].sort_values("n")
    ok = sub[(sub["median"] <= 0.01) & (sub["p75"] <= 0.015)]
    if len(ok):
        print(f"{method}: smallest n with median ECE<=0.01 and p75<=0.015 -> n={int(ok.iloc[0]['n'])}")
    else:
        print(f"{method}: no n reaches median ECE<=0.01 within tested range")
br = curve[curve.metric == "Brier"]
print("\nBrier median by n:")
print(br.pivot(index="n", columns="method", values="median").to_string())
sl = curve[curve.metric == "calib_slope"]
print("\nslope median by n:")
print(sl.pivot(index="n", columns="method", values="median").to_string())
print("\nSaved: recal_sample_size_curve.csv, recal_sample_size_raw.csv")
