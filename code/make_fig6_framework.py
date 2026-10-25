#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Combined calibration framework figure for merged manuscript."""
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

rcParams["font.family"] = "Arial"
rcParams["axes.linewidth"] = 0.8
rcParams["pdf.fonttype"] = 42
rcParams["font.size"] = 11

OUT = DATA_DIR
FIG = FIG_DIR

# Load stats
ecg_cal = pd.read_csv(f"{OUT}\\cal_dca_fusion_stats.csv")
ecg_rec = pd.read_csv(f"{OUT}\\recalibration_stats.csv")
nhanes_stats = {
    "Brier": 0.041, "ECE": 0.010, "calib_slope": 1.074
}

# Create 3-panel figure
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.5), dpi=300)

# Panel A: NHANES calibration quality (bar chart of metrics)
metrics = ["Brier", "ECE", "Calibration\nSlope"]
nhanes_vals = [nhanes_stats["Brier"], nhanes_stats["ECE"], nhanes_stats["calib_slope"]]
colors_bar = ["#2C3E50", "#E74C3C", "#27AE60"]
bars = ax1.bar(metrics, nhanes_vals, color=colors_bar, alpha=0.8, width=0.6)
ax1.set_ylabel("Value")
ax1.set_ylim(0, 1.3)
ax1.axhline(y=1.0, color="#999", ls="--", lw=1, label="Ideal slope = 1.0")
ax1.legend(loc="upper right", fontsize=9)
ax1.set_title("(A) NHANES: Development (5-yr all-cause mortality)", fontsize=11, fontweight="bold", loc="left")
ax1.text(-0.08, 1.06, "A", transform=ax1.transAxes, fontsize=15, fontweight="bold", va="top")
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
# Add value labels
for bar, val in zip(bars, nhanes_vals):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f"{val:.3f}", ha="center", fontsize=10)

# Panel B: ECG calibration drift (internal vs external)
datasets = ["Internal\n(PTB-XL)", "External\n(Chapman)"]
brier_vals = ecg_cal["Brier"].values
ece_vals = ecg_cal["ECE"].values
x = np.arange(len(datasets))
width = 0.35
bars1 = ax2.bar(x - width/2, brier_vals, width, label="Brier", color="#2C3E50", alpha=0.8)
bars2 = ax2.bar(x + width/2, ece_vals, width, label="ECE", color="#E74C3C", alpha=0.8)
ax2.set_xlabel("Dataset")
ax2.set_ylabel("Calibration Error")
ax2.set_xticks(x)
ax2.set_xticklabels(datasets)
ax2.set_ylim(0, 0.45)
ax2.legend(loc="upper left")
ax2.set_title("(B) ECG: Cross-continent Calibration Drift", fontsize=11, fontweight="bold", loc="left")
ax2.text(-0.08, 1.06, "B", transform=ax2.transAxes, fontsize=15, fontweight="bold", va="top")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
# Add value labels
for bar, val in zip(bars1, brier_vals):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015, f"{val:.3f}", ha="center", fontsize=9)
for bar, val in zip(bars2, ece_vals):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015, f"{val:.3f}", ha="center", fontsize=9)

# Panel C: Recalibration fix (ECE comparison)
methods = ["Raw", "Platt", "Beta"]
ece_rec = ecg_rec["ECE"].values
brier_rec = ecg_rec["Brier"].values
x = np.arange(len(methods))
bars1 = ax3.bar(x - width/2, brier_rec, width, label="Brier", color="#2C3E50", alpha=0.8)
bars2 = ax3.bar(x + width/2, ece_rec, width, label="ECE", color="#E74C3C", alpha=0.8)
# Add NHANES reference line
ax3.axhline(y=nhanes_stats["ECE"], color="#27AE60", ls="--", lw=1.5, label=f"NHANES ECE={nhanes_stats['ECE']:.3f}")
ax3.set_xlabel("Recalibration Method")
ax3.set_ylabel("Calibration Error")
ax3.set_xticks(x)
ax3.set_xticklabels(methods)
ax3.set_ylim(0, 0.5)
ax3.legend(loc="upper right", fontsize=9)
ax3.set_title("(C) ECG: Recalibration Restores Quality", fontsize=11, fontweight="bold", loc="left")
ax3.text(-0.08, 1.06, "C", transform=ax3.transAxes, fontsize=15, fontweight="bold", va="top")
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)
# Add value labels
for bar, val in zip(bars1, brier_rec):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015, f"{val:.3f}", ha="center", fontsize=9)
for bar, val in zip(bars2, ece_rec):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015, f"{val:.3f}", ha="center", fontsize=9)

plt.tight_layout()
plt.savefig(f"{FIG}/Figure6_unified_framework.png", dpi=600, bbox_inches="tight", facecolor="white")
plt.savefig(f"{FIG}/Figure6_unified_framework.pdf", bbox_inches="tight", facecolor="white")
plt.close()
print("Saved: Figure6_unified_framework.png/pdf")
