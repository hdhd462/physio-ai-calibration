#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""NHANES survival model calibration analysis."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA_DIR, PROJ_DIR, FIG_DIR, SUB_DIR, PATCH_DIR, NHANES_DIR, PROJ1_DIR, SHAOXING_DIR, PTBXL_DIR
import sys
sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd
import numpy as np
from sklearn.metrics import brier_score_loss
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.family"] = "Arial"
rcParams["axes.linewidth"] = 0.8
rcParams["pdf.fonttype"] = 42
rcParams["font.size"] = 11

OUT = PROJ_DIR
FIG = FIG_DIR

# Load test predictions
import os
base = PROJ1_DIR
test_fp = None
for root, dirs, fnames in os.walk(base):
    for f in fnames:
        if "test_predprob" in f.lower():
            test_fp = os.path.join(root, f)
            break

df = pd.read_csv(test_fp)
print(f"Test set: n={len(df)}, deaths={df['status'].sum()}")

# 5-year risk: predicted risk = 1 - 5yr survival prob
df["pred_risk_5yr"] = 1 - df["5"]

# Observed 5-year mortality (simple binary, exclude censored before 5yr)
T5 = 5.0
# Method: those with time >= 5 are confirmed survivors at 5yr
# those with time < 5 and status=1 died before 5yr (event)
# those with time < 5 and status=0 are censored before 5yr -> exclude for simple analysis
complete = df["time"] >= T5
died_before = (df["time"] < T5) & (df["status"] == 1)
censored_before = (df["time"] < T5) & (df["status"] == 0)

df["obs_death_5yr"] = np.nan
df.loc[complete, "obs_death_5yr"] = 0  # survived >= 5yr
df.loc[died_before, "obs_death_5yr"] = 1  # died before 5yr

n_analyzed = df["obs_death_5yr"].notna().sum()
n_excluded = censored_before.sum()
print(f"5yr analysis: analyzed={n_analyzed}, excluded(censored<5yr)={n_excluded}")
print(f"Observed 5yr mortality: {df['obs_death_5yr'].sum():.0f}/{n_analyzed} = {df['obs_death_5yr'].mean()*100:.1f}%")

# Drop excluded
d = df.dropna(subset=["obs_death_5yr"]).copy()
y = d["obs_death_5yr"].astype(int).values
p = d["pred_risk_5yr"].values

# Brier
brier = brier_score_loss(y, p)
print(f"Brier (5yr): {brier:.4f}")

# ECE
def compute_ece(y, p, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(p, bins) - 1
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if mask.sum() == 0:
            continue
        ece += mask.sum() / len(y) * abs(y[mask].mean() - p[mask].mean())
    return ece

ece = compute_ece(y, p)
print(f"ECE: {ece:.4f}")

# Calibration slope
from sklearn.linear_model import LogisticRegression
eps = 1e-6
logit_p = np.log(np.clip(p, eps, 1-eps) / (1 - np.clip(p, eps, 1-eps)))
lr = LogisticRegression(C=1e6).fit(logit_p.reshape(-1, 1), y)
slope = lr.coef_[0][0]
intercept = lr.intercept_[0]
print(f"Calibration slope: {slope:.3f}, intercept: {intercept:.3f}")

# AUC (5yr binary)
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(y, p)
print(f"AUC (5yr binary): {auc:.4f}")

# Calibration curve
from sklearn.calibration import calibration_curve
frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), dpi=300)

# Panel A: Calibration
ax1.plot(mean_pred, frac_pos, color="#2C3E50", marker="o", markersize=6, lw=1.8, label=f"NHANES Cox (n={len(y)})")
ax1.plot([0, 0.5], [0, 0.5], ls="--", color="#999", lw=1)
ax1.set_xlabel("Predicted 5-year CV mortality risk")
ax1.set_ylabel("Observed 5-year mortality rate")
ax1.set_xlim(0, 0.5)
ax1.set_ylim(0, 0.5)
ax1.legend(loc="upper left", frameon=True, fancybox=False, edgecolor="#CCC", fontsize=9)
ax1.set_title("(A) NHANES Calibration (5-year)", fontsize=12, fontweight="bold", loc="left")
ax1.text(-0.08, 1.06, "A", transform=ax1.transAxes, fontsize=15, fontweight="bold", va="top")
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# Panel B: Calibration zoom (0-20%)
mask = mean_pred <= 0.2
ax2.plot(mean_pred[mask], frac_pos[mask], color="#2C3E50", marker="o", markersize=6, lw=1.8)
ax2.plot([0, 0.2], [0, 0.2], ls="--", color="#999", lw=1)
ax2.set_xlabel("Predicted risk (zoomed)")
ax2.set_ylabel("Observed rate")
ax2.set_xlim(0, 0.2)
ax2.set_ylim(0, 0.2)
ax2.set_title("(B) Calibration Zoom (0-20%)", fontsize=12, fontweight="bold", loc="left")
ax2.text(-0.08, 1.06, "B", transform=ax2.transAxes, fontsize=15, fontweight="bold", va="top")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(f"{FIG}\\figure_nhanes_calibration.png", dpi=600, bbox_inches="tight")
plt.savefig(f"{FIG}\\figure_nhanes_calibration.pdf", bbox_inches="tight")
plt.close()
print(f"\nSaved: figure_nhanes_calibration.png/pdf")

# Stats
stats = {
    "n_analyzed": len(y),
    "n_events": int(y.sum()),
    "event_rate": y.mean(),
    "Brier": brier,
    "ECE": ece,
    "calib_slope": slope,
    "calib_intercept": intercept,
    "AUC_5yr": auc,
}
print("\n=== Summary ===")
for k, v in stats.items():
    print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

import json
with open(os.path.join(PROJ_DIR, "nhanes_cal_stats.json"), "w") as f:
    json.dump(stats, f, indent=2)
