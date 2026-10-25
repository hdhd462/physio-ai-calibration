#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Combined 3-panel figure: calibration drift + recalibration fix."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA_DIR, PROJ_DIR, FIG_DIR, SUB_DIR, PATCH_DIR, NHANES_DIR, PROJ1_DIR, SHAOXING_DIR, PTBXL_DIR
import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from scipy.optimize import minimize
from scipy.ndimage import uniform_filter1d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.family"] = "Arial"
rcParams["axes.linewidth"] = 0.8
rcParams["pdf.fonttype"] = 42
rcParams["font.size"] = 11

OUT = DATA_DIR
FIG = FIG_DIR

d = np.load(f"{OUT}\\fusion3_probs.npz")
y_int, p_int = d["y_te"], d["int"]
y_ext, p_ext = d["y_ext"], d["ext"]

eps = 1e-6
def logit(p):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Platt scaling
lr_ext = LogisticRegression(C=1e6).fit(logit(p_ext).reshape(-1, 1), y_ext)
p_ext_platt = lr_ext.predict_proba(logit(p_ext).reshape(-1, 1))[:, 1]

# Beta calibration
def beta_cal_loss(params, p, y):
    a, b, c = params
    z = a * np.log(np.clip(p, eps, 1-eps)) - b * np.log(np.clip(1-p, eps, 1-eps)) + c
    q = sigmoid(z)
    q = np.clip(q, 1e-8, 1 - 1e-8)
    return -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))
res = minimize(beta_cal_loss, [1, 1, 0], args=(p_ext, y_ext), method="Nelder-Mead")
a, b, c = res.x
z = a * np.log(np.clip(p_ext, eps, 1-eps)) - b * np.log(np.clip(1-p_ext, eps, 1-eps)) + c
p_ext_beta = sigmoid(z)

# DCA function
def dca_nb(y, p, thresholds):
    n = len(y)
    status = y == 1
    nbs = []
    for pt in thresholds:
        treat = p >= pt
        tp = (treat & status).sum()
        fp = (treat & ~status).sum()
        nb = (tp / n) - (fp / n) * (pt / (1 - pt))
        nbs.append(nb)
    return uniform_filter1d(nbs, size=11, mode="nearest")

thresholds = np.linspace(0.001, 0.5, 500)
prev_ext = y_ext.mean()

# Create figure - 2x2 layout
fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=300)
(ax1, ax2), (ax3, ax4) = axes

# Panel A: Calibration drift (internal vs external)
for (y, p, color, label) in [
    (y_int, p_int, "#2C3E50", f"Internal (n={len(y_int)}, AF={y_int.mean()*100:.1f}%)"),
    (y_ext, p_ext, "#E67E22", f"External (n={len(y_ext)}, AF={y_ext.mean()*100:.1f}%)"),
]:
    frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
    ax1.plot(mean_pred, frac_pos, color=color, marker="o", markersize=5, lw=1.8, label=label)
ax1.plot([0, 0.7], [0, 0.7], ls="--", color="#999", lw=1)
ax1.set_xlabel("Predicted probability")
ax1.set_ylabel("Observed AF rate")
ax1.set_xlim(0, 0.7)
ax1.set_ylim(0, 0.7)
ax1.legend(loc="upper left", frameon=True, fancybox=False, edgecolor="#CCC", fontsize=9)
ax1.set_title("(A) Calibration Drift", fontsize=12, fontweight="bold", loc="left")
ax1.text(-0.08, 1.06, "A", transform=ax1.transAxes, fontsize=15, fontweight="bold", va="top")
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# Panel B: Calibration zoom (0-30%)
for (y, p, color, label) in [
    (y_int, p_int, "#2C3E50", "Internal"),
    (y_ext, p_ext, "#E67E22", "External"),
]:
    frac_pos, mean_pred = calibration_curve(y, p, n_bins=15, strategy="quantile")
    mask = mean_pred <= 0.3
    ax2.plot(mean_pred[mask], frac_pos[mask], color=color, marker="o", markersize=4, lw=1.5, label=label)
ax2.plot([0, 0.3], [0, 0.3], ls="--", color="#999", lw=1)
ax2.set_xlabel("Predicted probability (zoomed)")
ax2.set_ylabel("Observed AF rate")
ax2.set_xlim(0, 0.3)
ax2.set_ylim(0, 0.3)
ax2.legend(loc="upper left", frameon=True, fancybox=False, edgecolor="#CCC", fontsize=9)
ax2.set_title("(B) Calibration Zoom (0-30%)", fontsize=12, fontweight="bold", loc="left")
ax2.text(-0.08, 1.06, "B", transform=ax2.transAxes, fontsize=15, fontweight="bold", va="top")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

# Panel C: Recalibration (before/after)
for (p, color, label) in [
    (p_ext, "#E67E22", "Raw FusionNet"),
    (p_ext_platt, "#27AE60", "Platt recalibrated"),
    (p_ext_beta, "#8E44AD", "Beta calibrated"),
]:
    frac_pos, mean_pred = calibration_curve(y_ext, p, n_bins=10, strategy="quantile")
    ax3.plot(mean_pred, frac_pos, color=color, marker="o", markersize=5, lw=1.8, label=label)
ax3.plot([0, 0.7], [0, 0.7], ls="--", color="#999", lw=1)
ax3.set_xlabel("Predicted probability")
ax3.set_ylabel("Observed AF rate")
ax3.set_xlim(0, 0.7)
ax3.set_ylim(0, 0.7)
ax3.legend(loc="upper left", frameon=True, fancybox=False, edgecolor="#CCC", fontsize=9)
ax3.set_title("(C) Recalibration Fix", fontsize=12, fontweight="bold", loc="left")
ax3.text(-0.08, 1.06, "C", transform=ax3.transAxes, fontsize=15, fontweight="bold", va="top")
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)

# Panel D: DCA before/after
nb_all = prev_ext - (1 - prev_ext) * thresholds / (1 - thresholds)
ax4.plot(thresholds * 100, nb_all, color="#CCCCCC", lw=2.5, label="Treat All", zorder=2)
ax4.axhline(0, color="#444", lw=1.5, label="Treat None", zorder=2)
for (p, color, label) in [
    (p_ext, "#E67E22", "Raw"),
    (p_ext_platt, "#27AE60", "Platt"),
    (p_ext_beta, "#8E44AD", "Beta"),
]:
    nbs = dca_nb(y_ext, p, thresholds)
    ax4.plot(thresholds * 100, nbs, color=color, lw=1.8, label=label, zorder=3)
ax4.set_xlabel("Threshold probability (%)")
ax4.set_ylabel("Net benefit")
ax4.set_xlim(0, 30)
ax4.set_ylim(-0.005, 0.10)
ax4.set_xticks([0, 10, 20, 30])
ax4.legend(loc="upper right", frameon=True, fancybox=False, edgecolor="#CCC", fontsize=9)
ax4.set_title("(D) DCA Recovery", fontsize=12, fontweight="bold", loc="left")
ax4.text(-0.08, 1.06, "D", transform=ax4.transAxes, fontsize=15, fontweight="bold", va="top")
ax4.spines["top"].set_visible(False)
ax4.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(f"{FIG}/Figure4_calibration_recalibration.png", dpi=600, bbox_inches="tight", facecolor="white")
plt.savefig(f"{FIG}/Figure4_calibration_recalibration.pdf", bbox_inches="tight", facecolor="white")
plt.close()
print("Saved: Figure4_calibration_recalibration.png/pdf")
