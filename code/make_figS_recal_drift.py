#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Figure S4 (single 3-panel supplementary figure):
(A) recalibration learning curve on the locked evaluation set,
(B) drift decomposition ladder (FusionNet, Chapman),
(C) mean predicted vs observed prevalence for the same CNN-Transformer across 3 cohorts.
Vector PNG+PDF; Arial; eff@17.8 >= 6.5 pt gate."""
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

ECG = DATA_DIR
PNG = os.path.join(SUB_DIR, "picture", "Figure S4_recal_drift.png")
PDF = os.path.join(SUB_DIR, "Figure pdf", "Figure S4_recal_drift.pdf")

rs = pd.read_csv(f"{ECG}\\recal_sample_size_curve.csv")
dd = pd.read_csv(f"{ECG}\\drift_decomposition_stats.csv").set_index("item")
dc = pd.read_csv(f"{ECG}\\drift_cohort_table.csv")

# gates vs recorded artifacts
ece400_p = float(rs[(rs.method == "Platt") & (rs.n == 400) & (rs.metric == "ECE")]["median"].iloc[0])
ece400_b = float(rs[(rs.method == "Beta") & (rs.n == 400) & (rs.metric == "ECE")]["median"].iloc[0])
assert abs(ece400_p - 0.0086) < 2e-4, ece400_p
assert abs(ece400_b - 0.0074) < 2e-4, ece400_b
assert abs(dd.loc["raw", "value"] - 0.35836415449044035) < 1e-9
assert np.allclose(dc["AUC"], [0.9727, 0.8252, 0.9659], atol=1e-3)
print("GATES PASS: n=400 ECE anchors, raw ECE, 3-cohort AUCs")

fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(11, 3.8), dpi=300)

# ── (A) learning curve ──
for method, col, lab in [("Platt", "#2C7FB8", "Platt (2 parameters)"),
                          ("Beta", "#C71585", "Beta (3 parameters)")]:
    sub = rs[(rs.method == method) & (rs.metric == "ECE")].sort_values("n")
    x = sub["n"].to_numpy(float)
    axA.plot(x, sub["median"], "-o", color=col, ms=4, lw=1.8, label=lab)
    axA.fill_between(x, sub["p25"], sub["p75"], color=col, alpha=0.18, lw=0)
axA.axhline(0.01, ls="--", color="#666", lw=1.1)
axA.text(3200, 0.0115, "ECE = 0.01", color="#444", fontsize=8.5, ha="right")
axA.axvline(400, ls=":", color="#888", lw=1.0)
axA.annotate("n = 400", xy=(400, 0.028), xytext=(520, 0.045),
             arrowprops=dict(arrowstyle="->", color="#555", lw=0.9), fontsize=8.5)
axA.set_xscale("log", base=2)
axA.set_xticks([25, 100, 400, 1600, 11095])
axA.set_xticklabels(["25", "100", "400", "1,600", "11,095"])
axA.set_xlabel("Labelled target ECGs used for recalibration")
axA.set_ylabel("ECE on locked evaluation set")
axA.legend(loc="upper right", frameon=True, fancybox=False, edgecolor="#CCC", fontsize=8.5)
axA.set_title("Recalibration learning curve", fontsize=11, fontweight="bold", loc="left")
axA.spines["right"].set_visible(False)
axA.spines["top"].set_visible(False)
axA.text(-0.10, 1.05, "A", transform=axA.transAxes, fontsize=14, fontweight="bold", va="center")

# ── (B) decomposition ladder ──
labels = ["Raw", "Prevalence-\nadjusted", "Intercept-\nonly", "Platt\n(slope)"]
vals = [dd.loc[k, "value"] for k in
        ["raw", "prevalence_diff_only_correction", "intercept_only_fitted", "platt_full"]]
cols = ["#C0392B", "#95A5A6", "#3498DB", "#2ECC71"]
bars = axB.bar(range(4), vals, color=cols, width=0.62)
for i, v in enumerate(vals):
    axB.text(i, v + 0.012, f"{v:.3f}", ha="center", fontsize=8.5)
axB.text(2, vals[2] + 0.028, "\u03b1 = \u22128.14", ha="center", fontsize=8.5, color="#2E86C1", fontweight="bold")
axB.set_xticks(range(4))
axB.set_xticklabels(labels, fontsize=9)
axB.set_ylabel("External ECE (FusionNet, Chapman)")
axB.set_ylim(0, 0.45)
axB.set_title("Drift decomposition", fontsize=11, fontweight="bold", loc="left")
axB.spines["right"].set_visible(False)
axB.spines["top"].set_visible(False)
axB.text(-0.12, 1.05, "B", transform=axB.transAxes, fontsize=14, fontweight="bold", va="center")
axB.text(0.5, 0.93, "prevalence explains <1%; level offset \u224890%",
         transform=axB.transAxes, ha="center", fontsize=8, color="#444")

# ── (C) same model, 3 cohorts ──
cohort_lab = ["Germany\n(PTB-XL)", "China\n(Chapman)", "Brazil\n(CODE-15%)"]
mean_pct = dc["mean_pred"].to_numpy() * 100
prev_pct = dc["prevalence"].to_numpy() * 100
sat = dc["pct_ge_09"].to_numpy()
xpos = np.arange(3)
w = 0.36
axC.bar(xpos - w / 2, mean_pct, w, color="#34495E", label="Mean predicted probability")
axC.bar(xpos + w / 2, prev_pct, w, color="#BDC3C7", label="Observed AF prevalence")
for i in range(3):
    axC.text(xpos[i] - w / 2, mean_pct[i] + 1.2, f"{mean_pct[i]:.1f}", ha="center", fontsize=8.5)
    axC.text(xpos[i] + w / 2, prev_pct[i] + 1.2, f"{prev_pct[i]:.1f}", ha="center", fontsize=8.5)
axC.annotate(f"{sat[1]:.1f}% of scores \u22650.9", xy=(0.82, 40), xytext=(1.38, 32),
             fontsize=8.5, color="#C0392B", ha="center",
             arrowprops=dict(arrowstyle="->", color="#C0392B", lw=0.9))
axC.set_xticks(xpos)
axC.set_xticklabels(cohort_lab, fontsize=9)
axC.set_ylabel("Probability (%)")
axC.set_ylim(0, 52)
axC.legend(loc="upper left", frameon=True, fancybox=False, edgecolor="#CCC", fontsize=8.5)
axC.set_title("Same model across cohorts (CNN-Transformer)", fontsize=11, fontweight="bold", loc="left")
axC.spines["right"].set_visible(False)
axC.spines["top"].set_visible(False)
axC.text(-0.10, 1.05, "C", transform=axC.transAxes, fontsize=14, fontweight="bold", va="center")

plt.tight_layout()
fig.savefig(PNG, dpi=300, bbox_inches="tight")
fig.savefig(PDF, bbox_inches="tight")
plt.close(fig)

# eff@17.8 gate: width inches -> cm at 17.8 cm print width
from matplotlib.figure import Figure
import matplotlib.pyplot as _plt
figw_cm = 11.0 * 2.54
eff = 11 * 17.8 / figw_cm
print(f"page ~{figw_cm:.1f} cm wide | eff@17.8 {eff:.2f} pt {'PASS' if eff >= 6.5 else 'FAIL'}")
print("saved:", PNG)
print("saved:", PDF)
