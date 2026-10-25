#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Recalibration analysis: does Platt/beta calibration on external restore DCA utility?"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA_DIR, PROJ_DIR, FIG_DIR, SUB_DIR, PATCH_DIR, NHANES_DIR, PROJ1_DIR, SHAOXING_DIR, PTBXL_DIR
import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score, brier_score_loss
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
y_int, p_int = d["y_te"], d["int"]
y_ext, p_ext = d["y_ext"], d["ext"]

eps = 1e-6
def logit(p):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Platt scaling on external (fit on external itself - shows "if you had labels here")
lr_ext = LogisticRegression(C=1e6).fit(logit(p_ext).reshape(-1, 1), y_ext)
p_ext_platt = lr_ext.predict_proba(logit(p_ext).reshape(-1, 1))[:, 1]
print(f"Platt ext: slope={lr_ext.coef_[0][0]:.3f}, intercept={lr_ext.intercept_[0]:.3f}")
print(f"  Brier before={brier_score_loss(y_ext, p_ext):.4f}, after={brier_score_loss(y_ext, p_ext_platt):.4f}")
print(f"  AUC before={roc_auc_score(y_ext, p_ext):.4f}, after={roc_auc_score(y_ext, p_ext_platt):.4f} (should be identical)")

# Beta calibration (3-parameter logistic on log-odds)
from scipy.optimize import minimize
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
print(f"Beta ext: a={a:.3f}, b={b:.3f}, c={c:.3f}")
print(f"  Brier after beta={brier_score_loss(y_ext, p_ext_beta):.4f}")

# DCA comparison
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

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=300)
(ax1, ax2, ax3) = axes

# Panel A: Calibration before/after
for (p, color, label) in [
    (p_ext, "#E67E22", "Raw FusionNet"),
    (p_ext_platt, "#27AE60", "Platt recalibrated"),
    (p_ext_beta, "#8E44AD", "Beta calibrated"),
]:
    frac_pos, mean_pred = calibration_curve(y_ext, p, n_bins=10, strategy="quantile")
    ax1.plot(mean_pred, frac_pos, color=color, marker="o", markersize=5, lw=1.8, label=label)
ax1.plot([0, 0.7], [0, 0.7], ls="--", color="#999", lw=1)
ax1.set_xlabel("Predicted probability")
ax1.set_ylabel("Observed AF rate")
ax1.set_xlim(0, 0.7)
ax1.set_ylim(0, 0.7)
ax1.legend(loc="upper left", frameon=True, fancybox=False, edgecolor="#CCC", fontsize=9)
ax1.set_title("(A) External Calibration: before/after", fontsize=12, fontweight="bold", loc="left")
ax1.text(-0.08, 1.06, "A", transform=ax1.transAxes, fontsize=15, fontweight="bold", va="top")
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# Panel B: DCA before/after
nb_all = prev_ext - (1 - prev_ext) * thresholds / (1 - thresholds)
ax2.plot(thresholds * 100, nb_all, color="#CCCCCC", lw=2.5, label="Treat All", zorder=2)
ax2.axhline(0, color="#444", lw=1.5, label="Treat None", zorder=2)
for (p, color, label, ls) in [
    (p_ext, "#E67E22", "Raw", "-"),
    (p_ext_platt, "#27AE60", "Platt", "-"),
    (p_ext_beta, "#8E44AD", "Beta", "-"),
]:
    nbs = dca_nb(y_ext, p, thresholds)
    ax2.plot(thresholds * 100, nbs, color=color, lw=1.8, label=label, zorder=3, ls=ls)
ax2.set_xlabel("Threshold probability (%)")
ax2.set_ylabel("Net benefit")
ax2.set_xlim(0, 50)
ax2.set_ylim(-0.005, 0.12)
ax2.set_xticks([0, 10, 20, 30, 40, 50])
ax2.legend(loc="upper right", frameon=True, fancybox=False, edgecolor="#CCC", fontsize=9)
ax2.set_title("(B) External DCA: before/after recalibration", fontsize=12, fontweight="bold", loc="left")
ax2.text(-0.08, 1.06, "B", transform=ax2.transAxes, fontsize=15, fontweight="bold", va="top")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

# Panel C: Zoom DCA 0-15%
ax3.plot(thresholds * 100, nb_all, color="#CCCCCC", lw=2.5, label="Treat All", zorder=2)
ax3.axhline(0, color="#444", lw=1.5, label="Treat None", zorder=2)
for (p, color, label) in [
    (p_ext, "#E67E22", "Raw"),
    (p_ext_platt, "#27AE60", "Platt"),
    (p_ext_beta, "#8E44AD", "Beta"),
]:
    nbs = dca_nb(y_ext, p, thresholds)
    ax3.plot(thresholds * 100, nbs, color=color, lw=1.8, label=label, zorder=3)
ax3.set_xlabel("Threshold probability (%)")
ax3.set_ylabel("Net benefit")
ax3.set_xlim(0, 15)
ax3.set_ylim(-0.005, 0.12)
ax3.set_xticks([0, 5, 10, 15])
ax3.legend(loc="upper right", frameon=True, fancybox=False, edgecolor="#CCC", fontsize=9)
ax3.set_title("(C) DCA Zoom (0-15%)", fontsize=12, fontweight="bold", loc="left")
ax3.text(-0.08, 1.06, "C", transform=ax3.transAxes, fontsize=15, fontweight="bold", va="top")
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(f"{FIG}\\figure_recalibration_dca.png", dpi=600, bbox_inches="tight")
plt.savefig(f"{FIG}\\figure_recalibration_dca.pdf", bbox_inches="tight")
plt.close()
print("Saved: figure_recalibration_dca.png/pdf")

# Stats summary
import pandas as pd
rows = []
for name, p in [("Raw", p_ext), ("Platt", p_ext_platt), ("Beta", p_ext_beta)]:
    # net benefit at clinically relevant thresholds
    nb5 = dca_nb(y_ext, p, np.array([0.05]))[0]
    nb10 = dca_nb(y_ext, p, np.array([0.10]))[0]
    nb20 = dca_nb(y_ext, p, np.array([0.20]))[0]
    rows.append({
        "method": name,
        "Brier": brier_score_loss(y_ext, p),
        "ECE": np.mean([abs(y_ext[np.digitize(p, np.linspace(0,1,11))-1==b].mean() - p[np.digitize(p, np.linspace(0,1,11))-1==b].mean())
                        for b in range(10) if (np.digitize(p, np.linspace(0,1,11))-1==b).sum()>0]),
        "NB@5%": nb5, "NB@10%": nb10, "NB@20%": nb20,
        "AUC": roc_auc_score(y_ext, p),
    })
stats = pd.DataFrame(rows)
stats.to_csv(f"{OUT}\\recalibration_stats.csv", index=False, encoding="utf-8-sig")
print(stats.to_string(index=False))
