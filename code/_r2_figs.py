# -*- coding: utf-8 -*-
"""Round-2 figure rebuilds: Figure 1 (full 0-1 axes), Figure 4 (same model across
cohorts, per-cohort treat-all, full axes), Figure S6 (legend + label fixes)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA_DIR, PROJ_DIR, FIG_DIR, SUB_DIR, PATCH_DIR, NHANES_DIR, PROJ1_DIR, SHAOXING_DIR, PTBXL_DIR
import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, brier_score_loss
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

ECG = DATA_DIR
PIC = os.path.join(SUB_DIR, "picture")
PDF = os.path.join(SUB_DIR, "Figure pdf")

eps = 1e-6
logit = lambda p: np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
sigmoid = lambda z: 1 / (1 + np.exp(-z))


def dca_nb(y, p, thresholds):
    n = len(y)
    status = y == 1
    out = []
    for pt in thresholds:
        treat = p >= pt
        tp = (treat & status).sum()
        fp = (treat & ~status).sum()
        out.append(tp / n - (fp / n) * (pt / (1 - pt)))
    return uniform_filter1d(out, size=11, mode="nearest")


def panel_letter(ax, s, x=-0.09, y=1.05):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=15, fontweight="bold", va="bottom")


def tidy(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ══════════════════════ Figure 1: drift + recalibration (FusionNet) ══════════════
fe = np.load(f"{ECG}\\fusion3_probs.npz")
y_int, p_int = fe["y_te"], fe["int"]
y_ext, p_ext = fe["y_ext"], fe["ext"]


def beta_fit(p, y):
    def loss(par):
        a, b, c = par
        z = a * np.log(np.clip(p, eps, 1 - eps)) - b * np.log(np.clip(1 - p, eps, 1 - eps)) + c
        q = np.clip(sigmoid(z), 1e-8, 1 - 1e-8)
        return -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))
    a, b, c = minimize(loss, [1, 1, 0], method="Nelder-Mead").x
    z = a * np.log(np.clip(p, eps, 1 - eps)) - b * np.log(np.clip(1 - p, eps, 1 - eps)) + c
    return sigmoid(z)


p_platt = LogisticRegression(C=1e6).fit(logit(p_ext).reshape(-1, 1), y_ext).predict_proba(
    logit(p_ext).reshape(-1, 1))[:, 1]
p_beta = beta_fit(p_ext, y_ext)

assert abs(roc_auc_score(y_ext, p_ext) - 0.839) < 0.002
assert abs(brier_score_loss(y_ext, p_ext) - 0.343) < 0.002

thresholds = np.linspace(0.001, 0.5, 500)
prev_ext = y_ext.mean()

fig, axes = plt.subplots(2, 2, figsize=(9.7, 8.1), dpi=300)
(ax1, ax2), (ax3, ax4) = axes

for (y, p, color, label) in [
    (y_int, p_int, "#2C3E50", f"Internal (n={len(y_int):,}, AF={y_int.mean()*100:.1f}%)"),
    (y_ext, p_ext, "#E67E22", f"External (n={len(y_ext):,}, AF={y_ext.mean()*100:.1f}%)"),
]:
    frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
    ax1.plot(mean_pred, frac_pos, color=color, marker="o", markersize=5, lw=1.8, label=label)
ax1.plot([0, 1], [0, 1], ls="--", color="#999", lw=1)
ax1.set_xlabel("Predicted probability")
ax1.set_ylabel("Observed AF rate")
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.set_xticks(np.arange(0, 1.01, 0.2))
ax1.set_yticks(np.arange(0, 1.01, 0.2))
ax1.legend(loc="upper left", frameon=True, fancybox=False, edgecolor="#CCC", fontsize=9.5)
ax1.set_title("Calibration Drift", fontsize=12, fontweight="bold", loc="left")
panel_letter(ax1, "A")
tidy(ax1)

for (y, p, color, label) in [(y_int, p_int, "#2C3E50", "Internal"),
                             (y_ext, p_ext, "#E67E22", "External")]:
    frac_pos, mean_pred = calibration_curve(y, p, n_bins=15, strategy="quantile")
    m = mean_pred <= 0.3
    ax2.plot(mean_pred[m], frac_pos[m], color=color, marker="o", markersize=4, lw=1.5, label=label)
ax2.plot([0, 0.3], [0, 0.3], ls="--", color="#999", lw=1)
ax2.set_xlabel("Predicted probability (zoomed)")
ax2.set_ylabel("Observed AF rate")
ax2.set_xlim(0, 0.3)
ax2.set_ylim(0, 0.3)
ax2.legend(loc="upper left", frameon=True, fancybox=False, edgecolor="#CCC", fontsize=9.5)
ax2.set_title("Calibration Zoom (0-30%)", fontsize=12, fontweight="bold", loc="left")
panel_letter(ax2, "B")
tidy(ax2)

for (p, color, label) in [(p_ext, "#E67E22", "Raw FusionNet"),
                          (p_platt, "#27AE60", "Platt recalibrated"),
                          (p_beta, "#8E44AD", "Beta calibrated")]:
    frac_pos, mean_pred = calibration_curve(y_ext, p, n_bins=10, strategy="quantile")
    ax3.plot(mean_pred, frac_pos, color=color, marker="o", markersize=5, lw=1.8, label=label)
ax3.plot([0, 1], [0, 1], ls="--", color="#999", lw=1)
ax3.set_xlabel("Predicted probability")
ax3.set_ylabel("Observed AF rate")
ax3.set_xlim(0, 1)
ax3.set_ylim(0, 1)
ax3.set_xticks(np.arange(0, 1.01, 0.2))
ax3.set_yticks(np.arange(0, 1.01, 0.2))
ax3.legend(loc="upper left", frameon=True, fancybox=False, edgecolor="#CCC", fontsize=9.5)
ax3.set_title("Recalibration Fix", fontsize=12, fontweight="bold", loc="left")
panel_letter(ax3, "C")
tidy(ax3)

nb_all = prev_ext - (1 - prev_ext) * thresholds / (1 - thresholds)
ax4.plot(thresholds * 100, nb_all, color="#CCCCCC", lw=2.5, label="Treat All", zorder=2)
ax4.axhline(0, color="#444", lw=1.5, label="Treat None", zorder=2)
for (p, color, label) in [(p_ext, "#E67E22", "Raw"), (p_platt, "#27AE60", "Platt"),
                          (p_beta, "#8E44AD", "Beta")]:
    ax4.plot(thresholds * 100, dca_nb(y_ext, p, thresholds), color=color, lw=1.8,
             label=label, zorder=3)
ax4.set_xlabel("Threshold probability (%)")
ax4.set_ylabel("Net benefit")
ax4.set_xlim(0, 30)
ax4.set_ylim(-0.006, 0.10)
ax4.set_xticks([0, 10, 20, 30])
ax4.legend(loc="upper right", frameon=True, fancybox=False, edgecolor="#CCC", fontsize=9.5)
ax4.set_title("DCA Recovery", fontsize=12, fontweight="bold", loc="left")
panel_letter(ax4, "D")
tidy(ax4)

plt.tight_layout()
fig.savefig(f"{PIC}\\Figure 1.png", dpi=600, bbox_inches="tight", facecolor="white")
fig.savefig(f"{PDF}\\Figure 1.pdf", bbox_inches="tight", facecolor="white")
plt.close(fig)
print("Figure 1 rebuilt")

# ══════════════════════ Figure 4: CODE-15% vs Chapman, same model ═══════════════
cp = np.load(f"{ECG}\\common_probs.npz")
y_ch, p_ch = cp["y_ext"], cp["ext_CNNTransformer"]
cd = np.load(f"{ECG}\\code15_part17_probs.npz", allow_pickle=True)
y_cd = cd["y"]
p_cnn, p_res, p_cnt = cd["CNN"], cd["ResNet"], cd["CNNT"]

auc_ch = roc_auc_score(y_ch, p_ch)
auc_cd = roc_auc_score(y_cd, p_cnt)
assert abs(auc_ch - 0.825) < 0.002, auc_ch
assert abs(auc_cd - 0.966) < 0.002, auc_cd


def smooth(fpr, tpr, n=300):
    from scipy.interpolate import PchipInterpolator
    fpr = np.concatenate([[0], fpr, [1]])
    tpr = np.concatenate([[0], tpr, [1]])
    keep = np.ones(len(fpr), bool)
    keep[1:] = np.diff(fpr) > 1e-6
    f = np.linspace(0, 1, n)
    return f, np.clip(PchipInterpolator(fpr[keep], tpr[keep])(f), 0, 1)


fig, axes = plt.subplots(2, 2, figsize=(9.7, 8.1), dpi=300)
(ax1, ax2), (ax3, ax4) = axes

for name, p, color in [("1D-CNN", p_cnn, "#3498DB"), ("ResNet1D", p_res, "#2ECC71"),
                       ("CNN-Transformer", p_cnt, "#E74C3C")]:
    a = roc_auc_score(y_cd, p)
    f, t = smooth(*roc_curve(y_cd, p)[:2])
    ax1.plot(f, t, color=color, lw=2, label=f"{name} (AUC={a:.3f})")
ax1.plot([0, 1], [0, 1], ls="--", color="#999", lw=1)
ax1.set_xlabel("1 - Specificity")
ax1.set_ylabel("Sensitivity")
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.legend(loc="lower right", frameon=True, fancybox=False, edgecolor="#CCC", fontsize=9)
ax1.set_title(f"ROC: CODE-15% Brazil (n={len(y_cd):,}, AF={int(y_cd.sum())})",
              fontsize=12, fontweight="bold", loc="left")
panel_letter(ax1, "A")
tidy(ax1)

for y, p, color, label in [(y_ch, p_ch, "#E67E22", f"Chapman China (AUC={auc_ch:.3f})"),
                           (y_cd, p_cnt, "#2C3E50", f"CODE-15% Brazil (AUC={auc_cd:.3f})")]:
    f, t = smooth(*roc_curve(y, p)[:2])
    ax2.plot(f, t, color=color, lw=2, label=label)
ax2.plot([0, 1], [0, 1], ls="--", color="#999", lw=1)
ax2.set_xlabel("1 - Specificity")
ax2.set_ylabel("Sensitivity")
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.legend(loc="lower right", frameon=True, fancybox=False, edgecolor="#CCC", fontsize=9)
ax2.set_title("Same model, two external cohorts", fontsize=12, fontweight="bold", loc="left")
panel_letter(ax2, "B")
tidy(ax2)

for y, p, color, marker, label in [
    (y_ch, p_ch, "#E67E22", "s", "Chapman China (CNN-Transformer)"),
    (y_cd, p_cnt, "#2C3E50", "o", "CODE-15% Brazil (CNN-Transformer)"),
]:
    frac, mp = calibration_curve(y, p, n_bins=10, strategy="quantile")
    ax3.plot(mp, frac, color=color, marker=marker, markersize=6, lw=1.8, label=label)
ax3.plot([0, 1], [0, 1], ls="--", color="#999", lw=1)
ax3.set_xlabel("Predicted probability")
ax3.set_ylabel("Observed AF rate")
ax3.set_xlim(0, 1)
ax3.set_ylim(0, 1)
ax3.set_xticks(np.arange(0, 1.01, 0.2))
ax3.set_yticks(np.arange(0, 1.01, 0.2))
ax3.legend(loc="upper left", frameon=True, fancybox=False, edgecolor="#CCC", fontsize=9)
ax3.set_title("Calibration: China vs Brazil", fontsize=12, fontweight="bold", loc="left")
panel_letter(ax3, "C")
tidy(ax3)

ax4.axhline(0, color="#444", lw=1.5, label="Treat None", zorder=2)
for y, p, color, tcol, tls, label, tlabel in [
    (y_ch, p_ch, "#E67E22", "#F5CBA7", "-", "CNN-Transformer, China", "Treat all (China, 8.0%)"),
    (y_cd, p_cnt, "#2C3E50", "#AED6F1", "-", "CNN-Transformer, Brazil", "Treat all (Brazil, 2.8%)"),
]:
    prev = y.mean()
    nb_all = prev - (1 - prev) * thresholds / (1 - thresholds)
    ax4.plot(thresholds * 100, nb_all, color=tcol, lw=2.5, label=tlabel, zorder=2)
    ax4.plot(thresholds * 100, dca_nb(y, p, thresholds), color=color, lw=1.8,
             label=label, zorder=3)
ax4.set_xlabel("Threshold probability (%)")
ax4.set_ylabel("Net benefit")
ax4.set_xlim(0, 50)
ax4.set_ylim(-0.02, 0.10)
ax4.set_xticks([0, 10, 20, 30, 40, 50])
ax4.legend(loc="upper right", frameon=True, fancybox=False, edgecolor="#CCC", fontsize=8.5)
ax4.set_title("DCA: China vs Brazil", fontsize=12, fontweight="bold", loc="left")
panel_letter(ax4, "D")
tidy(ax4)

plt.tight_layout()
fig.savefig(f"{PIC}\\Figure 4.png", dpi=600, bbox_inches="tight", facecolor="white")
fig.savefig(f"{PDF}\\Figure 4.pdf", bbox_inches="tight", facecolor="white")
plt.close(fig)
print("Figure 4 rebuilt")

# ══════════════════════ Figure S6: learning curve + decomposition + cohorts ══════
rs = pd.read_csv(f"{ECG}\\recal_sample_size_curve.csv")
dd = pd.read_csv(f"{ECG}\\drift_decomposition_stats.csv").set_index("item")
dc = pd.read_csv(f"{ECG}\\drift_cohort_table.csv")

assert abs(float(dd.loc["raw", "value"]) - 0.35836415449044035) < 1e-9
assert np.allclose(dc["AUC"], [0.9727, 0.8252, 0.9659], atol=1e-3)

fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(11, 3.8), dpi=300)

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

labels = ["Raw", "Prevalence-\nadjusted", "Intercept-\nonly", "Platt\n(full)"]
vals = [dd.loc[k, "value"] for k in
        ["raw", "prevalence_diff_only_correction", "intercept_only_fitted", "platt_full"]]
cols = ["#C0392B", "#95A5A6", "#3498DB", "#2ECC71"]
axB.bar(range(4), vals, color=cols, width=0.62)
for i, v in enumerate(vals):
    axB.text(i, v + 0.012, f"{v:.3f}", ha="center", fontsize=8.5)
axB.text(2, vals[2] + 0.03, "\u03b1 = \u22128.14", ha="center", fontsize=8.5,
         color="#2E86C1", fontweight="bold")
axB.set_xticks(range(4))
axB.set_xticklabels(labels, fontsize=8.5)
axB.set_ylabel("External ECE (FusionNet, Chapman)")
axB.set_ylim(0, 0.45)
axB.set_title("Drift decomposition", fontsize=11, fontweight="bold", loc="left")
axB.spines["right"].set_visible(False)
axB.spines["top"].set_visible(False)
axB.text(-0.12, 1.05, "B", transform=axB.transAxes, fontsize=14, fontweight="bold", va="center")
axB.text(0.5, 0.93, "prevalence explains <1%; level offset \u224890%",
         transform=axB.transAxes, ha="center", fontsize=8, color="#444")

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
axC.set_xticklabels(cohort_lab, fontsize=8.5)
axC.set_ylabel("Probability (%)")
axC.set_ylim(0, 58)
axC.legend(loc="upper left", frameon=True, fancybox=False, edgecolor="#CCC", fontsize=8.5)
axC.set_title("Same model across cohorts (CNN-Transformer)", fontsize=11, fontweight="bold", loc="left")
axC.spines["right"].set_visible(False)
axC.spines["top"].set_visible(False)
axC.text(-0.10, 1.05, "C", transform=axC.transAxes, fontsize=14, fontweight="bold", va="center")

plt.tight_layout()
fig.savefig(f"{PIC}\\Figure S6.png", dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(f"{PDF}\\Figure S6.pdf", bbox_inches="tight", facecolor="white")
plt.close(fig)
print("Figure S6 rebuilt")
