#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Rebuild submission Figure 1 and Figure 2 as native matplotlib (vector, Arial, print-sized).

Data sources (all audited, no re-estimation):
  Fig1A KM      : recomputed from nhanes_analysis.csv (same logic as make_km.py);
                  risk-table counts asserted against figure 1.pdf values.
  Fig1B RCS     : curve + 95% band polygon extracted from vector
                  freestat run RCS_2_6\\RCS_plot.pdf (tick-line transforms).
  Fig2A td-AUC  : td_auc_results.csv (make_roc_dca.py output).
  Fig2B DCA     : six curves extracted from vector table and figure\figure 4.pdf
                  (make_roc_dca.py output; colors keyed, legend samples skipped).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA_DIR, PROJ_DIR, FIG_DIR, SUB_DIR, PATCH_DIR, NHANES_DIR, PROJ1_DIR, SHAOXING_DIR, PTBXL_DIR
import sys, os, glob, json
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import pandas as pd
import pymupdf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from lifelines import KaplanMeierFitter

rcParams["font.family"] = "Arial"
rcParams["axes.linewidth"] = 0.9
rcParams["pdf.fonttype"] = 42
rcParams["font.size"] = 11
rcParams["axes.unicode_minus"] = False

FIG = FIG_DIR
NH = NHANES_DIR
SUB = SUB_DIR
RCS_PDF = glob.glob(os.path.join(NH, r"Project1\result\*RCS_2_6", "RCS_plot.pdf"))[0]

LETTER_FS = 15
EXPECTED_RISK = {
    "T1 (Low)":    [3089, 2917, 2650, 1011, 0],
    "T2 (Medium)": [3058, 3040, 3000, 1214, 0],
    "T3 (High)":   [3063, 3048, 3033, 1196, 0],
}


# ───────────────────────────── helpers ─────────────────────────────
def chain_points(items):
    """Chain [('l', p1, p2), ...] into a point list."""
    pts = [items[0][1]]
    for it in items:
        pts.append(it[2])
    return pts


def color_key(c, nd=3):
    return tuple(round(v, nd) for v in c) if c else None


# ───────────────────────────── Fig 1 panel A: KM ─────────────────────────────
def draw_km(ax):
    df = pd.read_csv(os.path.join(NH, "nhanes_analysis.csv"))
    df["group"] = df["mvpa_tertile"].map({"T1": "T1 (Low)", "T2": "T2 (Medium)", "T3": "T3 (High)"})
    colors = {"T1 (Low)": "#66CDAA", "T2 (Medium)": "#7B68EE", "T3 (High)": "#F08080"}
    groups = ["T1 (Low)", "T2 (Medium)", "T3 (High)"]
    kmf = KaplanMeierFitter()
    for g in groups:
        m = df["group"] == g
        kmf.fit(df.loc[m, "followup_years"].values, df.loc[m, "is_dead"].values, label=g)
        sf = kmf.survival_function_.iloc[:, 0]
        ci = kmf.confidence_interval_
        lo = ci.filter(like="lower", axis=1).iloc[:, 0].values * 100
        hi = ci.filter(like="upper", axis=1).iloc[:, 0].values * 100
        ax.step(sf.index, sf.values * 100, where="post", color=colors[g], lw=1.8, label=g)
        ax.fill_between(ci.index, lo, hi, step="post", color=colors[g], alpha=0.12)

    ax.set_xlabel("Time (years)", fontweight="bold", fontsize=10.5)
    ax.set_ylabel("Survival probability (%)", fontweight="bold", fontsize=10.5)
    ax.set_ylim(70, 101)
    ax.set_xlim(0, 10)
    ax.set_yticks([70, 75, 80, 85, 90, 95, 100])
    ax.set_xticks([0, 2, 4, 6, 8, 10])
    ax.tick_params(labelsize=9.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    leg = ax.legend(loc="lower center", bbox_to_anchor=(0.5, 0.02), ncol=3, frameon=True,
                    fancybox=False, fontsize=9.5, framealpha=0.95, edgecolor="gray")
    leg.get_frame().set_linewidth(0.8)
    ax.text(0.5, 0.17, "P < 0.0001", transform=ax.transAxes, fontsize=10, fontweight="bold",
            va="bottom", ha="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray",
                      alpha=0.95, linewidth=0.8))
    ax.set_title("Kaplan-Meier Survival Curve by MVPA Tertile", fontweight="bold", fontsize=11.5, pad=8)
    return df, groups, colors


def draw_risk_table(fig, ax, df, groups):
    times = [0, 2.5, 5, 7.5, 10]
    pos = ax.get_position()
    fig.text(pos.x0 + 0.5 * pos.width, pos.y0 - 0.155, "Number at risk",
             fontsize=10, fontweight="bold", ha="center", va="center")
    for i, g in enumerate(groups):
        gd = df[df["group"] == g]
        y = pos.y0 - 0.205 - i * 0.052
        fig.text(pos.x0 - 0.012, y, g, fontsize=9.5, fontweight="bold",
                 ha="right", va="center")
        counts = [(gd["followup_years"] >= t).sum() for t in times]
        assert counts == EXPECTED_RISK[g], f"risk mismatch {g}: {counts}"
        for j, n in enumerate(counts):
            x = pos.x0 + (0.07 + 0.215 * j) * pos.width
            fig.text(x, y, f"{n:,}", fontsize=9.5, ha="center", va="center")
    y_t = pos.y0 - 0.34
    fig.text(pos.x0 - 0.012, y_t, "Time (years)", fontsize=9.5, fontstyle="italic",
             ha="right", va="center")
    for j, t in enumerate(times):
        x = pos.x0 + (0.07 + 0.215 * j) * pos.width
        fig.text(x, y_t, f"{t:g}", fontsize=9.5, ha="center", va="center")


# ───────────────────────────── Fig 1 panel B: RCS ─────────────────────────────
def draw_rcs(ax):
    doc = pymupdf.open(RCS_PDF)
    pg = doc[0]
    drawings = pg.get_drawings()
    def is_rcs(c):
        if not c:
            return False
        return abs(c[0] - 0.0) < 0.01 and abs(c[1] - 0.439) < 0.02 and abs(c[2] - 0.725) < 0.02

    band = curve = None
    for d in drawings:
        if d["type"] == "f" and is_rcs(d.get("fill")) and len(d["items"]) > 100:
            band = d
        if d["type"] == "s" and is_rcs(d.get("color")) and len(d["items"]) > 100:
            curve = d
    assert band and curve, "RCS band/curve not found"
    doc.close()

    # transforms from tick lines: x 49.6px=0, 389px=40 ; y 376.6px=0, 77.9px=20
    fx = lambda px: (px - 49.6) / ((389.0 - 49.6) / 40.0)
    fy = lambda py: (376.6 - py) / ((376.6 - 77.9) / 20.0)

    def to_data(pts):
        a = np.array(pts)
        return fx(a[:, 0]), fy(a[:, 1])

    bx, by = to_data(chain_points(band["items"]))
    cx, cy = to_data(chain_points(curve["items"]))
    assert cx.min() > -5 and cx.max() < 50 and cy.max() < 30, "RCS data out of range"

    verts = np.column_stack([bx, by])
    ax.add_patch(PathPatch(Path(verts, closed=True), facecolor="#0070B9",
                           edgecolor="none", alpha=0.18, zorder=1))
    ax.plot(cx, cy, color="#0070B9", lw=1.8, zorder=3)

    ax.set_xlim(-2.07, 43.6)
    ax.set_ylim(-1.17, 24.86)
    ax.axvline(1.54, color="black", ls="--", lw=1.0, zorder=2)
    ax.axhline(1.0, color="black", ls="--", lw=1.0, zorder=2)
    ax.set_xticks([0, 10, 20, 30, 40])
    ax.set_yticks([0, 5, 10, 15, 20])
    ax.tick_params(labelsize=9.5)
    ax.set_xlabel("MVPA (per 10 min/day)", fontweight="bold", fontsize=10.5)
    ax.set_ylabel("Hazard Ratio (HR)", fontweight="bold", fontsize=10.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(0.985, 0.975,
            "P for non-linearity: <0.001\nP for overall: <0.001\nRef point: 1.54",
            transform=ax.transAxes, fontsize=9, ha="right", va="top", linespacing=1.35,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none",
                      alpha=0.85))


def build_figure1():
    fig = plt.figure(figsize=(9.6, 4.9))
    axA = fig.add_axes([0.105, 0.40, 0.37, 0.50])
    axB = fig.add_axes([0.585, 0.40, 0.36, 0.50])
    df, groups, _ = draw_km(axA)
    draw_risk_table(fig, axA, df, groups)
    draw_rcs(axB)
    fig.text(0.055, 0.92, "A", fontsize=LETTER_FS, fontweight="bold", ha="left", va="center")
    fig.text(0.535, 0.92, "B", fontsize=LETTER_FS, fontweight="bold", ha="left", va="center")
    fig.text(0.765, 0.925, "RCS: MVPA and Hazard Ratio",
             fontsize=11, fontweight="bold", ha="center", va="center")
    fig.savefig(os.path.join(SUB, "Figure1_KM_RCS.png"), dpi=600, facecolor="white")
    fig.savefig(os.path.join(SUB, "Figure1_KM_RCS.pdf"), facecolor="white")
    plt.close(fig)
    print("Figure1: KM+RCS (rebuilt, vector)")


# ───────────────────────────── Fig 2 panel A: time-dependent AUC ─────────────────
def draw_tdauc(ax):
    auc = pd.read_csv(os.path.join(NH, "td_auc_results.csv"), encoding="utf-8-sig")
    eval_t = [3.0, 5.0, 7.0, 8.5]
    colors = {"Model 1": "#888888", "Model 2": "#2E8B57", "Model 3": "#6A5ACD", "Model 4": "#E84393"}
    markers = {"Model 1": "o", "Model 2": "s", "Model 3": "^", "Model 4": "D"}
    for name in ["Model 1", "Model 2", "Model 3", "Model 4"]:
        row = auc[auc["Model"] == name].iloc[0]
        aucs = [row[f"AUC_{t}y"] for t in eval_t]
        ax.plot(eval_t, aucs, color=colors[name], marker=markers[name],
                markersize=6, lw=2.0, label=name, clip_on=False)
    ax.set_xlabel("Follow-up time (years)", fontsize=10.5)
    ax.set_ylabel("AUC", fontsize=10.5)
    ax.set_xticks(eval_t)
    ax.tick_params(labelsize=9.5)
    ax.set_ylim([0.78, 0.90])
    ax.legend(loc="lower left", fontsize=9.5, frameon=False)
    ax.set_title("Time-dependent AUC by Follow-up Time", fontsize=11, fontweight="bold", pad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ───────────────────────────── Fig 2 panel B: DCA ─────────────────────────────
def draw_dca(ax):
    doc = pymupdf.open(os.path.join(FIG, "figure 4.pdf"))
    pg = doc[0]
    fx = lambda px: (px - 54.8) / ((409.8 - 54.8) / 0.5)
    fy = lambda py: (228.3 - py) / ((296.8 - 228.3) / 0.04)  # y tick 0.00@228.3, -0.04@296.8
    wanted = {
        (0.533, 0.533, 0.533): ("Model 1", "#888888", "-", 2.0),
        (0.18, 0.545, 0.341): ("Model 2", "#2E8B57", "-", 2.0),
        (0.416, 0.353, 0.804): ("Model 3", "#6A5ACD", "-", 2.0),
        (0.91, 0.263, 0.576): ("Model 4", "#E84393", "-", 2.0),
        (0.0, 0.0, 0.0): ("Treat all", "black", "--", 1.2),
        (0.6, 0.6, 0.6): ("Treat none", "#999999", ":", 1.0),
    }
    found = {}
    for d in pg.get_drawings():
        if d["type"] != "s" or not d.get("color"):
            continue
        key = color_key(d["color"], 3)
        if key in wanted:
            label = wanted[key][0]
            if label == "Treat none" and d["rect"].width < 100:
                continue  # legend sample
            if label.startswith(("Model", "Treat all")) and len(d["items"]) < 5:
                continue  # legend sample
            if label not in found or len(d["items"]) > len(found[label]["items"]):
                found[label] = d
    doc.close()
    assert len(found) == 6, f"DCA curves missing: {set(k[0] for k in wanted.values()) - set(found)}"
    for key, (label, color, ls, lw) in wanted.items():
        d = found[label]
        pts = chain_points(d["items"])
        a = np.array(pts)
        ax.plot(fx(a[:, 0]), fy(a[:, 1]), color=color, ls=ls, lw=lw,
                label=label,
                zorder=4 if label == "Model 1" else
                       (3 if label.startswith(("Model", "Treat all")) else 2),
                solid_capstyle="round" if ls != ":" else "butt")
    ax.set_xlim([0, 0.5])
    ax.set_ylim([-0.05, 0.12])
    ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax.set_xlabel("Threshold Probability", fontsize=10.5)
    ax.set_ylabel("Net Benefit", fontsize=10.5)
    ax.tick_params(labelsize=9.5)
    ax.legend(loc="upper right", fontsize=9.5, frameon=True, edgecolor="#CCCCCC", fancybox=False)
    ax.set_title("Decision Curve Analysis at 5 years", fontsize=11, fontweight="bold", pad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def build_figure2():
    fig = plt.figure(figsize=(9.6, 4.3))
    axA = fig.add_axes([0.065, 0.155, 0.385, 0.70])
    axB = fig.add_axes([0.565, 0.155, 0.40, 0.70])
    draw_tdauc(axA)
    draw_dca(axB)
    fig.text(0.018, 0.90, "A", fontsize=LETTER_FS, fontweight="bold", ha="left", va="center")
    fig.text(0.515, 0.90, "B", fontsize=LETTER_FS, fontweight="bold", ha="left", va="center")
    fig.savefig(os.path.join(SUB, "Figure2_AUC_DCA.png"), dpi=600, facecolor="white")
    fig.savefig(os.path.join(SUB, "Figure2_AUC_DCA.pdf"), facecolor="white")
    plt.close(fig)
    print("Figure2: AUC+DCA (rebuilt, vector)")


if __name__ == "__main__":
    build_figure1()
    build_figure2()
