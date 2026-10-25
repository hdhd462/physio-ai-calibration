#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build supplementary Tables S6 (recalibration learning curve) and S7
(same-model three-cohort calibration) as three-line .docx matching S1-S5 style."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA_DIR, PROJ_DIR, FIG_DIR, SUB_DIR, PATCH_DIR, NHANES_DIR, PROJ1_DIR, SHAOXING_DIR, PTBXL_DIR
import sys
sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

ECG = DATA_DIR
TAB = os.path.join(SUB_DIR, "tables")

rs = pd.read_csv(f"{ECG}\\recal_sample_size_curve.csv")
dc = pd.read_csv(f"{ECG}\\drift_cohort_table.csv")

def set_borders(table):
    tbl = table._tbl
    for el in tbl.xpath('./w:tblPr/w:tblBorders'):
        tbl.xpath('./w:tblPr')[0].remove(el)
    borders = OxmlElement('w:tblBorders')
    for edge in ('top', 'bottom'):
        el = OxmlElement(f'w:{edge}')
        el.set(qn('w:val'), 'single'); el.set(qn('w:sz'), '8')
        el.set(qn('w:space'), '0'); el.set(qn('w:color'), '000000')
        borders.append(el)
    for edge in ('left', 'right', 'insideH', 'insideV'):
        el = OxmlElement(f'w:{edge}'); el.set(qn('w:val'), 'none')
        borders.append(el)
    tbl.xpath('./w:tblPr')[0].append(borders)

def hdr_cell_borders(cell):
    tcPr = cell._tc.get_or_add_tcPr()
    b = OxmlElement('w:tcBorders')
    for edge in ('bottom',):
        el = OxmlElement(f'w:{edge}')
        el.set(qn('w:val'), 'single'); el.set(qn('w:sz'), '8')
        el.set(qn('w:space'), '0'); el.set(qn('w:color'), '000000')
        b.append(el)
    tcPr.append(b)

def make_doc(title, headers, rows, note):
    d = Document()
    s = d.sections[0]
    s.left_margin = s.right_margin = s.top_margin = s.bottom_margin = Cm(1.5)
    t = d.add_paragraph()
    r = t.add_run(title)
    r.bold = True; r.font.size = Pt(10); r.font.name = 'Arial'
    tbl = d.add_table(rows=1 + len(rows), cols=len(headers))
    set_borders(tbl)
    for j, h in enumerate(headers):
        c = tbl.rows[0].cells[j]
        c.text = ''
        run = c.paragraphs[0].add_run(h)
        run.bold = True; run.font.size = Pt(8); run.font.name = 'Arial'
        hdr_cell_borders(c)
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            c = tbl.rows[i + 1].cells[j]
            c.text = ''
            run = c.paragraphs[0].add_run(str(v))
            run.font.size = Pt(8); run.font.name = 'Arial'
    p = d.add_paragraph()
    rn = p.add_run('Note: '); rn.font.size = Pt(8); rn.font.name = 'Arial'
    rt = p.add_run(note); rt.font.size = Pt(8); rt.font.name = 'Arial'
    return d

def cell_ece(med, p25, p75):
    return f"{med:.3f} ({p25:.3f}\u2013{p75:.3f})"

# ── Table S6 ──
ns_levels = [25, 50, 100, 200, 400, 800, 1600, 3200, 11095]
rows6 = []
for n in ns_levels:
    sub = rs[rs['n'] == n]
    reps = int(sub['reps'].iloc[0])
    def get(m, method, col):
        v = sub[(sub['method'] == method) & (sub['metric'] == m)]
        return float(v[col].iloc[0])
    n_lab = "Full pool" if n == 11095 else f"{n:,}"
    rows6.append([
        n_lab, reps if n != 11095 else "\u2014",
        cell_ece(get('ECE', 'Platt', 'median'), get('ECE', 'Platt', 'p25'), get('ECE', 'Platt', 'p75')),
        f"{get('Brier', 'Platt', 'median'):.4f}",
        f"{get('calib_slope', 'Platt', 'median'):.2f}",
        cell_ece(get('ECE', 'Beta', 'median'), get('ECE', 'Beta', 'p25'), get('ECE', 'Beta', 'p75')),
        f"{get('Brier', 'Beta', 'median'):.4f}",
        f"{get('calib_slope', 'Beta', 'median'):.2f}",
    ])
d6 = make_doc(
    "Table S6. Recalibration learning curve on a locked external evaluation set",
    ["Labelled sample size, n", "Repeats",
     "Platt ECE, median (IQR)", "Platt Brier, median", "Platt slope, median",
     "Beta ECE, median (IQR)", "Beta Brier, median", "Beta slope, median"],
    rows6,
    ("The external Chapman-Shaoxing cohort was split 1:1 (stratified by outcome, seed 42) into a fitting pool "
     "(n = 11,095; 887 events) and a locked evaluation set (n = 11,095; 887 events). For each sample size, Platt "
     "(two-parameter logistic) and beta (three-parameter) recalibration models were fitted on stratified random "
     "subsamples of the pool (200 replicates for n \u2264 800; 100 replicates for n = 1,600 and 3,200) and scored "
     "on the locked evaluation set; the full-pool anchor was fitted on all 11,095 pool records and scored on the "
     "same locked evaluation set. Values are replicate medians; interquartile ranges are shown for ECE. ECE, "
     "expected calibration error (10 equal-width probability bins weighted by bin occupancy); IQR, interquartile "
     "range; Brier, Brier score; slope, calibration slope from logistic regression of outcome on the logit of "
     "recalibrated probability."))
d6.save(f"{TAB}\\Table S6.docx")
print("saved Table S6.docx | rows:", len(rows6))

# ── Table S7 ──
rows7 = []
for _, r_ in dc.iterrows():
    rows7.append([
        r_['cohort'],
        f"{int(r_['n']):,}",
        f"{int(r_['events']):,} ({100 * r_['prevalence']:.1f}%)",
        f"{r_['mean_pred']:.3f}",
        f"{r_['pct_ge_09']:.1f}",
        f"{r_['Brier']:.4f}",
        f"{r_['ECE']:.4f}",
        f"{r_['calib_slope']:.2f}",
        f"{r_['citl']:+.2f}",
        f"{r_['AUC']:.4f}",
    ])
d7 = make_doc(
    "Table S7. Cross-cohort calibration of the same CNN-Transformer across three cohorts",
    ["Cohort", "n", "AF, n (%)", "Mean predicted", "Scores \u22650.9, %",
     "Brier", "ECE", "Calibration slope", "Calibration-in-the-large", "AUC"],
    rows7,
    ("The CNN-Transformer was applied without refitting to the German source internal test set, the Chinese external "
     "cohort, and the Brazilian third cohort. Prevalence shift in logit terms versus the source: China +0.15; "
     "Brazil \u22120.96 (6.2-fold larger in absolute magnitude). Calibration slope from logistic regression of "
     "outcome on the logit of predicted probability; calibration-in-the-large fitted with the logit of predicted "
     "probability as a fixed offset. ECE, expected calibration error (10 equal-width probability bins weighted by "
     "bin occupancy)."))
d7.save(f"{TAB}\\Table S7.docx")
print("saved Table S7.docx | rows:", len(rows7))
