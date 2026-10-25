#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Fusion model calibration + DCA (internal vs external)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA_DIR, PROJ_DIR, FIG_DIR, SUB_DIR, PATCH_DIR, NHANES_DIR, PROJ1_DIR, SHAOXING_DIR, PTBXL_DIR
import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.ndimage import uniform_filter1d

rcParams["font.family"] = "Arial"
rcParams["axes.linewidth"] = 0.8
rcParams["pdf.fonttype"] = 42
rcParams["font.size"] = 11

OUT = DATA_DIR
FIG = FIG_DIR

d = np.load(f"{OUT}\\fusion3_probs.npz")
y_int = d["y_te"]
p_int = d["int"]
y_ext = d["y_ext"]
p_ext = d["ext"]

prev_int = y_int.mean()
prev_ext = y_ext.mean()
brier_int = brier_score_loss(y_int, p_int)
brier_ext = brier_score_loss(y_ext, p_ext)

print(f"internal: n={len(y_int)}, AF={y_int.sum()} ({prev_int*100:.1f}%), Brier={brier_int:.4f}")
print(f"external: n={len(y_ext)}, AF={y_ext.sum()} ({prev_ext*100:.1f}%), Brier={brier_ext:.4f}")

# ECE (Expected Calibration Error)
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

ece_int = compute_ece(y_int, p_int)
ece_ext = compute_ece(y_ext, p_ext)
print(f"ECE internal={ece_int:.4f}, external={ece_ext:.4f}")

# Calibration intercept & slope (logistic recalibration)
from sklearn.linear_model import LogisticRegression
eps = 1e-6
logit_int = np.log(np.clip(p_int, eps, 1-eps) / (1 - np.clip(p_int, eps, 1-eps)))
logit_ext = np.log(np.clip(p_ext, eps, 1-eps) / (1 - np.clip(p_ext, eps, 1-eps)))
# slope: fit y ~ logit(p); intercept: fit y ~ offset(logit(p))
lr_slope_int = LogisticRegression(C=1e6).fit(logit_int.reshape(-1,1), y_int)
lr_slope_ext = LogisticRegression(C=1e6).fit(logit_ext.reshape(-1,1), y_ext)
print(f"calib slope internal={lr_slope_int.coef_[0][0]:.3f}, external={lr_slope_ext.coef_[0][0]:.3f}")

fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=300)
(ax1, ax2), (ax3, ax4) = axes

# ── Panel A: Calibration curves (internal vs external) ──
for ax, (y, p, prev, brier, ece, title) in zip(
    [ax1],
    [(y_int, p_int, prev_int, brier_int, ece_int, "Internal Test"),
     ]
):
    pass  # placeholder

# Combined calibration plot
for (y, p, color, label, ls) in [
    (y_int, p_int, "#2C3E50", f"Internal (n={len(y_int)}, AF={prev_int*100:.1f}%)", "-"),
    (y_ext, p_ext, "#E67E22", f"External (n={len(y_ext)}, AF={prev_ext*100:.1f}%)", "-"),
]:
    frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
    ax1.plot(mean_pred, frac_pos, color=color, marker="o", markersize=5, lw=1.8, ls=ls, label=label)
ax1.plot([0, 1], [0, 1], ls="--", color="#999", lw=1)
ax1.set_xlabel("Predicted probability")
ax1.set_ylabel("Observed AF rate")
ax1.set_xlim(0, 0.7)
ax1.set_ylim(0, 0.7)
ax1.legend(loc="upper left", frameon=True, fancybox=False, edgecolor="#CCC", fontsize=9)
ax1.set_title("(A) Calibration: FusionNet", fontsize=12, fontweight="bold", loc="left")
ax1.text(-0.08, 1.06, "A", transform=ax1.transAxes, fontsize=15, fontweight="bold", va="top")
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# ── Panel B: Calibration zoom (0-30%) ──
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

# ── Panel C: DCA internal ──
thresholds = np.linspace(0.001, 0.5, 500)
nb_all_int = prev_int - (1 - prev_int) * thresholds / (1 - thresholds)
ax3.plot(thresholds * 100, nb_all_int, color="#CCCCCC", lw=2.5, label="Treat All", zorder=2)
ax3.axhline(0, color="#444", lw=1.5, label="Treat None", zorder=2)
status = y_int == 1
n = len(y_int)
nbs = []
for pt in thresholds:
    treat = p_int >= pt
    tp = (treat & status).sum()
    fp = (treat & ~status).sum()
    nb = (tp / n) - (fp / n) * (pt / (1 - pt))
    nbs.append(nb)
nbs = uniform_filter1d(nbs, size=11, mode="nearest")
ax3.plot(thresholds * 100, nbs, color="#2C3E50", lw=1.8, label="FusionNet", zorder=3)
ax3.set_xlabel("Threshold probability (%)")
ax3.set_ylabel("Net benefit")
ax3.set_xlim(0, 50)
ax3.set_ylim(-0.005, 0.12)
ax3.set_xticks([0, 10, 20, 30, 40, 50])
ax3.legend(loc="upper right", frameon=True, fancybox=False, edgecolor="#CCC", fontsize=9)
ax3.set_title("(C) DCA: Internal Test", fontsize=12, fontweight="bold", loc="left")
ax3.text(-0.08, 1.06, "C", transform=ax3.transAxes, fontsize=15, fontweight="bold", va="top")
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)

# ── Panel D: DCA external ──
nb_all_ext = prev_ext - (1 - prev_ext) * thresholds / (1 - thresholds)
ax4.plot(thresholds * 100, nb_all_ext, color="#CCCCCC", lw=2.5, label="Treat All", zorder=2)
ax4.axhline(0, color="#444", lw=1.5, label="Treat None", zorder=2)
status = y_ext == 1
n = len(y_ext)
nbs = []
for pt in thresholds:
    treat = p_ext >= pt
    tp = (treat & status).sum()
    fp = (treat & ~status).sum()
    nb = (tp / n) - (fp / n) * (pt / (1 - pt))
    nbs.append(nb)
nbs = uniform_filter1d(nbs, size=11, mode="nearest")
ax4.plot(thresholds * 100, nbs, color="#E67E22", lw=1.8, label="FusionNet", zorder=3)
ax4.set_xlabel("Threshold probability (%)")
ax4.set_ylabel("Net benefit")
ax4.set_xlim(0, 50)
ax4.set_ylim(-0.005, 0.12)
ax4.set_xticks([0, 10, 20, 30, 40, 50])
ax4.legend(loc="upper right", frameon=True, fancybox=False, edgecolor="#CCC", fontsize=9)
ax4.set_title("(D) DCA: External (Chapman)", fontsize=12, fontweight="bold", loc="left")
ax4.text(-0.08, 1.06, "D", transform=ax4.transAxes, fontsize=15, fontweight="bold", va="top")
ax4.spines["top"].set_visible(False)
ax4.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(f"{FIG}\\figure_cal_dca_fusion.png", dpi=600, bbox_inches="tight")
plt.savefig(f"{FIG}\\figure_cal_dca_fusion.pdf", bbox_inches="tight")
plt.close()
print(f"Saved: figure_cal_dca_fusion.png/pdf")

# Also save stats to CSV for manuscript
import pandas as pd
stats = pd.DataFrame({
    "dataset": ["Internal Test", "External (Chapman-Shaoxing)"],
    "n": [len(y_int), len(y_ext)],
    "AF_n": [int(y_int.sum()), int(y_ext.sum())],
    "AF_prev": [prev_int, prev_ext],
    "Brier": [brier_int, brier_ext],
    "ECE": [ece_int, ece_ext],
    "calib_slope": [lr_slope_int.coef_[0][0], lr_slope_ext.coef_[0][0]],
})
stats.to_csv(f"{OUT}\\cal_dca_fusion_stats.csv", index=False, encoding="utf-8-sig")
print("Saved: cal_dca_fusion_stats.csv")
print(stats.to_string(index=False))
