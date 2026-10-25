#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Regenerate all main figures to submission folder."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA_DIR, PROJ_DIR, FIG_DIR, SUB_DIR, PATCH_DIR, NHANES_DIR, PROJ1_DIR, SHAOXING_DIR, PTBXL_DIR
import sys
sys.stdout.reconfigure(encoding="utf-8")
import os
import shutil
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, brier_score_loss, roc_auc_score
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize
from scipy.ndimage import uniform_filter1d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import rcParams

rcParams["font.family"] = "Arial"
rcParams["axes.linewidth"] = 0.8
rcParams["pdf.fonttype"] = 42
rcParams["font.size"] = 11

OUT = DATA_DIR
FIG = FIG_DIR
SUB = SUB_DIR

# Ensure submission dir exists
os.makedirs(SUB, exist_ok=True)

# ─── Smooth ROC function ───
def smooth_roc(fpr, tpr, n_points=300):
    fpr = np.concatenate([[0], fpr, [1]])
    tpr = np.concatenate([[0], tpr, [1]])
    if len(fpr) > 1:
        keep = np.ones(len(fpr), dtype=bool)
        keep[1:] = np.diff(fpr) > 1e-6
        fpr_unique, tpr_unique = fpr[keep], tpr[keep]
    else:
        fpr_unique, tpr_unique = fpr, tpr
    if len(fpr_unique) < 2:
        fpr_unique, tpr_unique = np.array([0, 1]), np.array([0, 1])
    fpr_smooth = np.linspace(0, 1, n_points)
    tpr_smooth = np.clip(PchipInterpolator(fpr_unique, tpr_unique)(fpr_smooth), 0, 1)
    return fpr_smooth, tpr_smooth

# ─── Figures 1 & 2: rebuilt natively (vector) in make_fig12_new.py ───
from make_fig12_new import build_figure1, build_figure2
build_figure1()
build_figure2()

# ─── Figure 3: ECG ROC (smooth) ───
d = np.load(f"{OUT}\\common_probs.npz")
fe = np.load(f"{OUT}\\fusion3_probs.npz")
y_int, y_ext = d['y_te'], d['y_ext']

CURVES = [
    ('int_M1', 'M1 (Demographics)', '#E74C3C', '--', 1.4),
    ('int_M2', 'M2 (Demo+HRV)', '#3498DB', '--', 1.4),
    ('int_M3', 'M3 (Demo+HRV+Morph)', '#2ECC71', '--', 1.4),
    ('int_CNN', '1D-CNN', '#8E44AD', '-', 1.8),
    ('int_ResNet', 'ResNet1D', '#E67E22', '-', 1.8),
    ('int_CNNTransformer', 'CNN-Transformer', '#2C3E50', '-', 2.0),
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 4.9), dpi=300)
for key, label, color, ls, lw in CURVES:
    ek = key.replace('int_', 'ext_')
    fpr_raw, tpr_raw, _ = roc_curve(y_int, d[key])
    auc_int = auc(fpr_raw, tpr_raw)
    fpr_s, tpr_s = smooth_roc(fpr_raw, tpr_raw)
    ax1.plot(fpr_s, tpr_s, color=color, ls=ls, lw=lw, label=f'{label} AUC={auc_int:.3f}')
    fpr_raw, tpr_raw, _ = roc_curve(y_ext, d[ek])
    auc_ext = auc(fpr_raw, tpr_raw)
    fpr_s, tpr_s = smooth_roc(fpr_raw, tpr_raw)
    ax2.plot(fpr_s, tpr_s, color=color, ls=ls, lw=lw, label=f'{label} AUC={auc_ext:.3f}')

fpr_raw, tpr_raw, _ = roc_curve(y_int, fe['int'])
auc_int = auc(fpr_raw, tpr_raw)
fpr_s, tpr_s = smooth_roc(fpr_raw, tpr_raw)
ax1.plot(fpr_s, tpr_s, color='#C0392B', ls='-', lw=2.2, label=f'Fusion network AUC={auc_int:.3f}')
fpr_raw, tpr_raw, _ = roc_curve(y_ext, fe['ext'])
auc_ext = auc(fpr_raw, tpr_raw)
fpr_s, tpr_s = smooth_roc(fpr_raw, tpr_raw)
ax2.plot(fpr_s, tpr_s, color='#C0392B', ls='-', lw=2.2, label=f'Fusion network AUC={auc_ext:.3f}')

for ax, letter, title in [(ax1, 'A', 'Internal test (PTB-XL)'),
                          (ax2, 'B', 'External validation (Chapman-Shaoxing)')]:
    ax.plot([0, 1], [0, 1], ls='--', color='#888', lw=1)
    ax.set_xlabel('1 - Specificity')
    ax.set_ylabel('Sensitivity')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc='lower right', frameon=True, fancybox=False, edgecolor='#CCC', fontsize=9)
    ax.set_title(title, fontsize=12, fontweight='bold', loc='left')
    ax.text(-0.08, 1.05, letter, transform=ax.transAxes, fontsize=15, fontweight='bold', va='bottom')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(f"{SUB}/Figure3_ROC.png", dpi=600, bbox_inches="tight", facecolor="white")
plt.savefig(f"{SUB}/Figure3_ROC.pdf", bbox_inches="tight", facecolor="white")
plt.close()
print("Figure3: ECG ROC (smooth)")

# ─── Figure 4: Calibration + Recalibration ───
y_int4, p_int = fe["y_te"], fe["int"]
y_ext4, p_ext = fe["y_ext"], fe["ext"]
eps = 1e-6
def logit(p):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

lr_ext = LogisticRegression(C=1e6).fit(logit(p_ext).reshape(-1, 1), y_ext4)
p_ext_platt = lr_ext.predict_proba(logit(p_ext).reshape(-1, 1))[:, 1]

def beta_cal_loss(params, p, y):
    a, b, c = params
    z = a * np.log(np.clip(p, eps, 1-eps)) - b * np.log(np.clip(1-p, eps, 1-eps)) + c
    q = np.clip(sigmoid(z), 1e-8, 1 - 1e-8)
    return -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))
res = minimize(beta_cal_loss, [1, 1, 0], args=(p_ext, y_ext4), method="Nelder-Mead")
a, b, c = res.x
z = a * np.log(np.clip(p_ext, eps, 1-eps)) - b * np.log(np.clip(1-p_ext, eps, 1-eps)) + c
p_ext_beta = sigmoid(z)

def dca_nb(y, p, thresholds):
    n = len(y)
    status = y == 1
    nbs = []
    for pt in thresholds:
        treat = p >= pt
        tp = (treat & status).sum()
        fp = (treat & ~status).sum()
        nbs.append((tp / n) - (fp / n) * (pt / (1 - pt)))
    return uniform_filter1d(nbs, size=11, mode="nearest")

thresholds = np.linspace(0.001, 0.5, 500)
prev_ext = y_ext4.mean()

fig, axes = plt.subplots(2, 2, figsize=(9.7, 8.1), dpi=300)
(ax1, ax2), (ax3, ax4) = axes

for (y, p, color, label) in [
    (y_int4, p_int, "#2C3E50", f"Internal (n={len(y_int4):,}, AF={y_int4.mean()*100:.1f}%)"),
    (y_ext4, p_ext, "#E67E22", f"External (n={len(y_ext4):,}, AF={y_ext4.mean()*100:.1f}%)"),
]:
    frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
    ax1.plot(mean_pred, frac_pos, color=color, marker="o", markersize=5, lw=1.8, label=label)
ax1.plot([0, 0.7], [0, 0.7], ls="--", color="#999", lw=1)
ax1.set_xlabel("Predicted probability")
ax1.set_ylabel("Observed AF rate")
ax1.set_xlim(0, 0.7)
ax1.set_ylim(0, 0.7)
ax1.legend(loc="upper left", frameon=True, fancybox=False, edgecolor="#CCC", fontsize=9.5)
ax1.set_title("Calibration Drift", fontsize=12, fontweight="bold", loc="left")
ax1.text(-0.08, 1.05, "A", transform=ax1.transAxes, fontsize=15, fontweight="bold", va="bottom")
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

for (y, p, color, label) in [
    (y_int4, p_int, "#2C3E50", "Internal"),
    (y_ext4, p_ext, "#E67E22", "External"),
]:
    frac_pos, mean_pred = calibration_curve(y, p, n_bins=15, strategy="quantile")
    mask = mean_pred <= 0.3
    ax2.plot(mean_pred[mask], frac_pos[mask], color=color, marker="o", markersize=4, lw=1.5, label=label)
ax2.plot([0, 0.3], [0, 0.3], ls="--", color="#999", lw=1)
ax2.set_xlabel("Predicted probability (zoomed)")
ax2.set_ylabel("Observed AF rate")
ax2.set_xlim(0, 0.3)
ax2.set_ylim(0, 0.3)
ax2.legend(loc="upper left", frameon=True, fancybox=False, edgecolor="#CCC", fontsize=9.5)
ax2.set_title("Calibration Zoom (0-30%)", fontsize=12, fontweight="bold", loc="left")
ax2.text(-0.08, 1.05, "B", transform=ax2.transAxes, fontsize=15, fontweight="bold", va="bottom")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

for (p, color, label) in [
    (p_ext, "#E67E22", "Raw FusionNet"),
    (p_ext_platt, "#27AE60", "Platt recalibrated"),
    (p_ext_beta, "#8E44AD", "Beta calibrated"),
]:
    frac_pos, mean_pred = calibration_curve(y_ext4, p, n_bins=10, strategy="quantile")
    ax3.plot(mean_pred, frac_pos, color=color, marker="o", markersize=5, lw=1.8, label=label)
ax3.plot([0, 0.7], [0, 0.7], ls="--", color="#999", lw=1)
ax3.set_xlabel("Predicted probability")
ax3.set_ylabel("Observed AF rate")
ax3.set_xlim(0, 0.7)
ax3.set_ylim(0, 0.7)
ax3.legend(loc="upper left", frameon=True, fancybox=False, edgecolor="#CCC", fontsize=9.5)
ax3.set_title("Recalibration Fix", fontsize=12, fontweight="bold", loc="left")
ax3.text(-0.08, 1.05, "C", transform=ax3.transAxes, fontsize=15, fontweight="bold", va="bottom")
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)

nb_all = prev_ext - (1 - prev_ext) * thresholds / (1 - thresholds)
ax4.plot(thresholds * 100, nb_all, color="#CCCCCC", lw=2.5, label="Treat All", zorder=2)
ax4.axhline(0, color="#444", lw=1.5, label="Treat None", zorder=2)
for (p, color, label) in [
    (p_ext, "#E67E22", "Raw"),
    (p_ext_platt, "#27AE60", "Platt"),
    (p_ext_beta, "#8E44AD", "Beta"),
]:
    nbs = dca_nb(y_ext4, p, thresholds)
    ax4.plot(thresholds * 100, nbs, color=color, lw=1.8, label=label, zorder=3)
ax4.set_xlabel("Threshold probability (%)")
ax4.set_ylabel("Net benefit")
ax4.set_xlim(0, 30)
ax4.set_ylim(-0.005, 0.10)
ax4.set_xticks([0, 10, 20, 30])
ax4.legend(loc="upper right", frameon=True, fancybox=False, edgecolor="#CCC", fontsize=9.5)
ax4.set_title("DCA Recovery", fontsize=12, fontweight="bold", loc="left")
ax4.text(-0.08, 1.05, "D", transform=ax4.transAxes, fontsize=15, fontweight="bold", va="bottom")
ax4.spines["top"].set_visible(False)
ax4.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(f"{SUB}/Figure4_calibration_recalibration.png", dpi=600, bbox_inches="tight", facecolor="white")
plt.savefig(f"{SUB}/Figure4_calibration_recalibration.pdf", bbox_inches="tight", facecolor="white")
plt.close()
print("Figure4: Calibration+Recalibration")

# ─── Figure 5: CODE-15% ───
cd = np.load(f"{OUT}\\code15_part17_probs.npz", allow_pickle=True)
y_cd = cd["y"]
p_cnn_cd, p_res_cd, p_cnnt_cd = cd["CNN"], cd["ResNet"], cd["CNNT"]
y_ext5, p_ext5 = fe["y_ext"], fe["ext"]

fig, axes = plt.subplots(2, 2, figsize=(9.7, 8.1), dpi=300)
(ax1, ax2), (ax3, ax4) = axes

colors = {"CNN": "#3498DB", "ResNet": "#2ECC71", "CNN-T": "#E74C3C"}

for name, p, color in [("CNN", p_cnn_cd, colors["CNN"]),
                        ("ResNet", p_res_cd, colors["ResNet"]),
                        ("CNN-T", p_cnnt_cd, colors["CNN-T"])]:
    auc_val = roc_auc_score(y_cd, p)
    fpr_raw, tpr_raw, _ = roc_curve(y_cd, p)
    fpr_s, tpr_s = smooth_roc(fpr_raw, tpr_raw)
    ax1.plot(fpr_s, tpr_s, color=color, lw=2, label=f"{name} (AUC={auc_val:.3f})")
ax1.plot([0, 1], [0, 1], ls="--", color="#999", lw=1)
ax1.set_xlabel("1 - Specificity")
ax1.set_ylabel("Sensitivity")
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.legend(loc="lower right", frameon=True, fancybox=False, edgecolor="#CCC", fontsize=9.5)
ax1.set_title(f"ROC: CODE-15% Brazil (n={len(y_cd):,}, AF={int(y_cd.sum()):,})", fontsize=12, fontweight="bold", loc="left")
ax1.text(-0.08, 1.05, "A", transform=ax1.transAxes, fontsize=15, fontweight="bold", va="bottom")
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

auc_cd = roc_auc_score(y_cd, p_cnnt_cd)
auc_ch = roc_auc_score(y_ext5, p_ext5)
fpr_cd_raw, tpr_cd_raw, _ = roc_curve(y_cd, p_cnnt_cd)
fpr_ch_raw, tpr_ch_raw, _ = roc_curve(y_ext5, p_ext5)
fpr_cd_s, tpr_cd_s = smooth_roc(fpr_cd_raw, tpr_cd_raw)
fpr_ch_s, tpr_ch_s = smooth_roc(fpr_ch_raw, tpr_ch_raw)
ax2.plot(fpr_ch_s, tpr_ch_s, color="#E67E22", lw=2, label=f"Chapman China (AUC={auc_ch:.3f})")
ax2.plot(fpr_cd_s, tpr_cd_s, color="#2C3E50", lw=2, label=f"CODE-15% Brazil (AUC={auc_cd:.3f})")
ax2.plot([0, 1], [0, 1], ls="--", color="#999", lw=1)
ax2.set_xlabel("1 - Specificity")
ax2.set_ylabel("Sensitivity")
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.legend(loc="lower right", frameon=True, fancybox=False, edgecolor="#CCC", fontsize=9.5)
ax2.set_title("External Validation: China vs Brazil", fontsize=12, fontweight="bold", loc="left")
ax2.text(-0.08, 1.05, "B", transform=ax2.transAxes, fontsize=15, fontweight="bold", va="bottom")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

for name, p, y, color, marker in [
    ("CNN-T CODE-15%", p_cnnt_cd, y_cd, "#2C3E50", "o"),
    ("FusionNet Chapman", p_ext5, y_ext5, "#E67E22", "s"),
]:
    frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
    ax3.plot(mean_pred, frac_pos, color=color, marker=marker, markersize=6, lw=1.8, label=name)
ax3.plot([0, 0.6], [0, 0.6], ls="--", color="#999", lw=1)
ax3.set_xlabel("Predicted probability")
ax3.set_ylabel("Observed AF rate")
ax3.set_xlim(0, 0.6)
ax3.set_ylim(0, 0.6)
ax3.legend(loc="upper left", frameon=True, fancybox=False, edgecolor="#CCC", fontsize=9.5)
ax3.set_title("Calibration: CODE-15% vs Chapman", fontsize=12, fontweight="bold", loc="left")
ax3.text(-0.08, 1.05, "C", transform=ax3.transAxes, fontsize=15, fontweight="bold", va="bottom")
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)

for y, p, color, label in [
    (y_cd, p_cnnt_cd, "#2C3E50", "CNN-T CODE-15%"),
    (y_ext5, p_ext5, "#E67E22", "FusionNet Chapman"),
]:
    prev = y.mean()
    n = len(y)
    status = y == 1
    nb_all = prev - (1 - prev) * thresholds / (1 - thresholds)
    if label.startswith("CNN-T"):
        ax4.plot(thresholds * 100, nb_all, color="#CCCCCC", lw=2.5, label="Treat All", zorder=2)
        ax4.axhline(0, color="#444", lw=1.5, label="Treat None", zorder=2)
    nbs = dca_nb(y, p, thresholds)
    ax4.plot(thresholds * 100, nbs, color=color, lw=1.8, label=label, zorder=3)
ax4.set_xlabel("Threshold probability (%)")
ax4.set_ylabel("Net benefit")
ax4.set_xlim(0, 50)
ax4.set_ylim(-0.005, 0.12)
ax4.set_xticks([0, 10, 20, 30, 40, 50])
ax4.legend(loc="upper right", frameon=True, fancybox=False, edgecolor="#CCC", fontsize=9.5)
ax4.set_title("DCA: CODE-15% vs Chapman", fontsize=12, fontweight="bold", loc="left")
ax4.text(-0.08, 1.05, "D", transform=ax4.transAxes, fontsize=15, fontweight="bold", va="bottom")
ax4.spines["top"].set_visible(False)
ax4.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(f"{SUB}/Figure5_CODE15.png", dpi=600, bbox_inches="tight", facecolor="white")
plt.savefig(f"{SUB}/Figure5_CODE15.pdf", bbox_inches="tight", facecolor="white")
plt.close()
print("Figure5: CODE-15%")

# ─── Figure 6: Unified Framework ───
ecg_cal = pd.read_csv(f"{OUT}\\cal_dca_fusion_stats.csv")
ecg_rec = pd.read_csv(f"{OUT}\\recalibration_stats.csv")
nhanes_stats = {"Brier": 0.041, "ECE": 0.010, "calib_slope": 1.074}

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(7.6, 7.8), dpi=300)

# —— Panel A: NHANES scores (left axis, 0-0.12) + calibration slope (right axis) ——
metrics = ["Brier", "ECE", "Calibration\nSlope"]
nhanes_vals = [nhanes_stats["Brier"], nhanes_stats["ECE"], nhanes_stats["calib_slope"]]
colors_bar = ["#2C3E50", "#E74C3C", "#27AE60"]
bars = ax1.bar([0, 1], nhanes_vals[:2], color=colors_bar[:2], alpha=0.8, width=0.55)
ax1.set_ylabel("Score (lower is better)", fontsize=10)
ax1.set_ylim(0, 0.12)
ax1.set_yticks([0, 0.03, 0.06, 0.09, 0.12])
ax1.set_xlim(-0.6, 2.6)
ax1.set_xticks([0, 1, 2])
ax1.set_xticklabels(metrics, fontsize=10)
for bar, val in zip(bars, nhanes_vals[:2]):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.004, f"{val:.3f}",
             ha="center", fontsize=9)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

ax1r = ax1.twinx()
bar_r = ax1r.bar([2], [nhanes_vals[2]], color=colors_bar[2], alpha=0.8, width=0.55)
ax1r.set_ylim(0, 1.3)
ax1r.set_yticks([0, 0.5, 1.0])
ax1r.set_ylabel("Calibration slope", fontsize=10)
ax1r.axhline(y=1.0, color="#999", ls="--", lw=1, label="Ideal slope = 1.0")
ax1r.text(2, nhanes_vals[2] + 0.05, f"{nhanes_vals[2]:.3f}", ha="center", fontsize=9)
ax1r.legend(loc="upper left", fontsize=9, frameon=False)
ax1r.spines["top"].set_visible(False)
ax1r.spines["left"].set_visible(False)
ax1r.spines["bottom"].set_visible(False)
ax1.set_title("NHANES: Development (5-yr all-cause mortality)", fontsize=11, fontweight="bold", loc="left")
ax1.text(-0.08, 1.05, "A", transform=ax1.transAxes, fontsize=15, fontweight="bold", va="bottom")

# —— Panel B: ECG calibration drift ——
datasets = ["Internal\n(PTB-XL)", "External\n(Chapman)"]
brier_vals = ecg_cal["Brier"].values
ece_vals = ecg_cal["ECE"].values
x = np.arange(len(datasets))
width = 0.35
bars1 = ax2.bar(x - width/2, brier_vals, width, label="Brier", color="#2C3E50", alpha=0.8)
bars2 = ax2.bar(x + width/2, ece_vals, width, label="ECE", color="#E74C3C", alpha=0.8)
ax2.set_xlabel("Dataset", fontsize=10)
ax2.set_ylabel("Score (lower is better)", fontsize=10)
ax2.set_xticks(x)
ax2.set_xticklabels(datasets, fontsize=10)
ax2.set_ylim(0, 0.45)
ax2.legend(loc="upper left", fontsize=9, frameon=False)
ax2.set_title("ECG: Cross-continent Calibration Drift", fontsize=11, fontweight="bold", loc="left")
ax2.text(-0.08, 1.05, "B", transform=ax2.transAxes, fontsize=15, fontweight="bold", va="bottom")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
for bar, val in zip(bars1, brier_vals):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.012, f"{val:.3f}", ha="center", fontsize=9)
for bar, val in zip(bars2, ece_vals):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.012, f"{val:.3f}", ha="center", fontsize=9)

# —— Panel C: ECG recalibration ——
methods = ["Raw", "Platt", "Beta"]
ece_rec = ecg_rec["ECE"].values
brier_rec = ecg_rec["Brier"].values
x = np.arange(len(methods))
bars1 = ax3.bar(x - width/2, brier_rec, width, label="Brier", color="#2C3E50", alpha=0.8)
bars2 = ax3.bar(x + width/2, ece_rec, width, label="ECE", color="#E74C3C", alpha=0.8)
ax3.axhline(y=nhanes_stats["ECE"], color="#27AE60", ls="--", lw=1.5, label=f"NHANES ECE={nhanes_stats['ECE']:.3f}")
ax3.set_xlabel("Recalibration Method", fontsize=10)
ax3.set_ylabel("Score (lower is better)", fontsize=10)
ax3.set_xticks(x)
ax3.set_xticklabels(methods, fontsize=10)
ax3.set_ylim(0, 0.5)
ax3.legend(loc="upper right", fontsize=9, frameon=False)
ax3.set_title("ECG: Recalibration Restores Quality", fontsize=11, fontweight="bold", loc="left")
ax3.text(-0.08, 1.05, "C", transform=ax3.transAxes, fontsize=15, fontweight="bold", va="bottom")
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)
for bar, val in zip(bars1, brier_rec):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.014, f"{val:.3f}", ha="center", fontsize=9)
for bar, val in zip(bars2, ece_rec):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.014, f"{val:.3f}", ha="center", fontsize=9)

plt.tight_layout()
plt.savefig(f"{SUB}/Figure6_unified_framework.png", dpi=600, bbox_inches="tight", facecolor="white")
plt.savefig(f"{SUB}/Figure6_unified_framework.pdf", bbox_inches="tight", facecolor="white")
plt.close()
print("Figure6: Unified Framework")

# ─── List final files ───
print("\n=== Submission folder contents ===")
for f in sorted(os.listdir(SUB)):
    fp = os.path.join(SUB, f)
    if os.path.isfile(fp):
        print(f"  {f} ({os.path.getsize(fp)/1024:.0f} KB)")
