#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Combined calibration figure for merged manuscript."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA_DIR, PROJ_DIR, FIG_DIR, SUB_DIR, PATCH_DIR, NHANES_DIR, PROJ1_DIR, SHAOXING_DIR, PTBXL_DIR
import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

rcParams["font.family"] = "Arial"
rcParams["axes.linewidth"] = 0.8
rcParams["pdf.fonttype"] = 42
rcParams["font.size"] = 11

FIG = FIG_DIR

# Load NHANES test data
import os
base = PROJ1_DIR
test_fp = None
for root, dirs, fnames in os.walk(base):
    for f in fnames:
        if "test_predprob" in f.lower():
            test_fp = os.path.join(root, f)
            break

df = pd.read_csv(test_fp)
df["pred_risk"] = 1 - df["5"]
# 5yr binary
complete = df["time"] >= 5.0
died_before = (df["time"] < 5.0) & (df["status"] == 1)
df["obs_death"] = np.nan
df.loc[complete, "obs_death"] = 0
df.loc[died_before, "obs_death"] = 1
d = df.dropna(subset=["obs_death"]).copy()
y_nhanes = d["obs_death"].astype(int).values
p_nhanes = d["pred_risk"].values

# Load ECG data (if available)
ecg_cal = pd.read_csv(os.path.join(DATA_DIR, "cal_dca_fusion_stats.csv"))
ecg_rec = pd.read_csv(os.path.join(DATA_DIR, "recalibration_stats.csv"))
print("ECG calibration stats loaded")
print(ecg_cal)
print(ecg_rec)

# For ECG, we need the actual predictions to plot calibration curve
# Check if available
ecg_pred_fp = os.path.join(DATA_DIR, "cal_predictions.csv")
has_ecg_pred = os.path.exists(ecg_pred_fp)
print(f"ECG predictions file exists: {has_ecg_pred}")

# Create figure
fig = plt.figure(figsize=(14, 10), dpi=300)
gs = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.35)

# Panel A: NHANES Calibration
ax1 = fig.add_subplot(gs[0, 0])
frac_pos, mean_pred = calibration_curve(y_nhanes, p_nhanes, n_bins=10, strategy="quantile")
ax1.plot(mean_pred, frac_pos, color="#2C3E50", marker="o", markersize=7, lw=2, label=f"NHANES (n={len(y_nhanes)})")
ax1.plot([0, 0.5], [0, 0.5], ls="--", color="#999", lw=1)
ax1.set_xlabel("Predicted 5-year CV mortality risk")
ax1.set_ylabel("Observed rate")
ax1.set_xlim(0, 0.5)
ax1.set_ylim(0, 0.5)
ax1.legend(loc="upper left", frameon=True, fancybox=False, edgecolor="#CCC", fontsize=9)
ax1.set_title("(A) NHANES: Development + Validation", fontsize=12, fontweight="bold", loc="left")
ax1.text(-0.08, 1.06, "A", transform=ax1.transAxes, fontsize=15, fontweight="bold", va="top")
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# Panel B: ECG Internal vs External (using reported stats for annotation)
ax2 = fig.add_subplot(gs[0, 1])
# Since we don't have raw predictions, plot the summary stats
datasets = ["Internal", "External\n(Chapman)"]
briers = [0.019, 0.343]
eces = [0.019, 0.358]
x = np.arange(len(datasets))
width = 0.35
bars1 = ax2.bar(x - width/2, briers, width, label="Brier", color="#2C3E50", alpha=0.8)
bars2 = ax2.bar(x + width/2, eces, width, label="ECE", color="#E74C3C", alpha=0.8)
ax2.set_xlabel("Dataset")
ax2.set_ylabel("Error")
ax2.set_xticks(x)
ax2.set_xticklabels(datasets)
ax2.legend(loc="upper left")
ax2.set_ylim(0, 0.45)
ax2.set_title("(B) ECG: Calibration Drift", fontsize=12, fontweight="bold", loc="left")
ax2.text(-0.08, 1.06, "B", transform=ax2.transAxes, fontsize=15, fontweight="bold", va="top")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

# Annotate bars
for bar, val in zip(bars1, briers):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f"{val:.3f}", ha="center", fontsize=9)
for bar, val in zip(bars2, eces):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f"{val:.3f}", ha="center", fontsize=9)

# Panel C: Recalibration Results
ax3 = fig.add_subplot(gs[1, 0])
methods = ecg_rec["method"].values
brier_rec = ecg_rec["Brier"].values
ece_rec = ecg_rec["ECE"].values
x = np.arange(len(methods))
bars1 = ax3.bar(x - width/2, brier_rec, width, label="Brier", color="#2C3E50", alpha=0.8)
bars2 = ax3.bar(x + width/2, ece_rec, width, label="ECE", color="#E74C3C", alpha=0.8)
ax3.set_xlabel("Recalibration Method")
ax3.set_ylabel("Error")
ax3.set_xticks(x)
ax3.set_xticklabels(methods)
ax3.legend(loc="upper right")
ax3.set_title("(C) ECG: Recalibration Fixes Drift", fontsize=12, fontweight="bold", loc="left")
ax3.text(-0.08, 1.06, "C", transform=ax3.transAxes, fontsize=15, fontweight="bold", va="top")
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)

# Annotate
for bar, val in zip(bars1, brier_rec):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, f"{val:.3f}", ha="center", fontsize=8)
for bar, val in zip(bars2, ece_rec):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, f"{val:.3f}", ha="center", fontsize=8)

# Panel D: Summary Comparison
ax4 = fig.add_subplot(gs[1, 1])
# Compare NHANES vs ECG External vs ECG Recalibrated
labels = ["NHANES\n(5yr mortality)", "ECG External\n(AF detection)", "ECG Recalibrated\n(Beta)"]
eces_summary = [0.010, 0.358, 0.008]
colors = ["#27AE60", "#E74C3C", "#2980B9"]
bars = ax4.bar(labels, eces_summary, color=colors, alpha=0.8)
ax4.set_ylabel("ECE (lower = better)")
ax4.set_ylim(0, 0.45)
ax4.set_title("(D) Calibration Quality Comparison", fontsize=12, fontweight="bold", loc="left")
ax4.text(-0.08, 1.06, "D", transform=ax4.transAxes, fontsize=15, fontweight="bold", va="top")
ax4.spines["top"].set_visible(False)
ax4.spines["right"].set_visible(False)
for bar, val in zip(bars, eces_summary):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f"{val:.3f}", ha="center", fontsize=10)

# Panel E: Narrative Summary
ax5 = fig.add_subplot(gs[2, :])
ax5.axis("off")
summary_text = (
    "UNIFIED FRAMEWORK: Physiological Signal AI Model Deployment\n\n"
    "NHANES (Development + Validation)          ECG (External Deployment)\n"
    "  • PPG wearable features → CV mortality      • ECG → AF detection\n"
    "  • Internal calibration: EXCELLENT          • Cross-continent drift: SEVERE\n"
    "    Brier=0.041, ECE=0.010                     Brier=0.343, ECE=0.358\n"
    "  • Demonstrates proper development           • Demonstrates deployment challenge\n\n"
    "SOLUTION: Recalibration with local data\n"
    "  • Beta calibration: ECE 0.358 → 0.008 (98% reduction)\n"
    "  • Discrimination preserved: AUC=0.839\n\n"
    "CONCLUSION: Physiological signal AI requires VALIDATION + CALIBRATION at deployment"
)
ax5.text(0.02, 0.95, summary_text, transform=ax5.transAxes, fontsize=10, va="top", ha="left",
         fontfamily="monospace", bbox=dict(boxstyle="round", facecolor="#F8F9FA", edgecolor="#DEE2E6", alpha=0.8))
ax5.set_title("(E) Unified Framework", fontsize=12, fontweight="bold", loc="left")
ax5.text(-0.02, 1.08, "E", transform=ax5.transAxes, fontsize=15, fontweight="bold", va="top")

plt.savefig(f"{FIG}\\figure_combined_calibration.png", dpi=600, bbox_inches="tight")
plt.savefig(f"{FIG}\\figure_combined_calibration.pdf", bbox_inches="tight")
print(f"\nSaved: figure_combined_calibration.png/pdf")
