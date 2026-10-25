#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Table S8: multi-outcome external calibration + repair ladder (raw/zero-data/Platt/beta)
across five ECG outcomes x two external cohorts, three-line .docx matching S1-S7 style."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA_DIR, PROJ_DIR, FIG_DIR, SUB_DIR, PATCH_DIR, NHANES_DIR, PROJ1_DIR, SHAOXING_DIR, PTBXL_DIR
import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
from docx import Document
from docx.shared import Pt, Cm
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

ECG = DATA_DIR
TAB = os.path.join(SUB_DIR, "tables")

ext = np.load(f"{ECG}\\fusion_multi_external.npz", allow_pickle=True)
zd = np.load(f"{ECG}\\zero_data_repair.npz", allow_pickle=True)
mlp = np.load(f"{ECG}\\ptbxl_multilabel.npz", allow_pickle=True)
names = [str(x) for x in mlp["names"]] if "names" in mlp.files else ["AF", "RBBB", "LBBB", "IAVB", "NORM"]
prev_dev = {names[k]: float(mlp["Y"][:, k].mean()) for k in range(len(names))}
print("dev prevalence:", {k: round(v, 4) for k, v in prev_dev.items()})

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
    el = OxmlElement('w:bottom')
    el.set(qn('w:val'), 'single'); el.set(qn('w:sz'), '8')
    el.set(qn('w:space'), '0'); el.set(qn('w:color'), '000000')
    b.append(el)
    tcPr.append(b)

rows = []
for cohort, label in [("chapman", "China (Chapman-Shaoxing)"), ("code15", "Brazil (CODE-15%)")]:
    for o in ["AF", "RBBB", "LBBB", "IAVB", "NORM"]:
        r_ = lambda stage, m: float(ext[f"{cohort}_{o}_{stage}_{m}"])
        n = int(r_("raw", "n")); ev = int(r_("raw", "events")); prev = r_("raw", "prev")
        zde = float(zd[f"{cohort}_{o}_prior_meanp_ece"])
        rows.append([
            o, label, f"{n:,} ({ev:,})",
            f"{100 * prev_dev[o]:.1f}% \u2192 {100 * prev:.1f}%",
            f"{r_('raw', 'auc'):.3f}",
            f"{r_('raw', 'ece'):.3f}",
            f"{zde:.3f}",
            f"{r_('platt', 'ece'):.3f}",
            f"{r_('beta', 'ece'):.3f}",
            f"{r_('raw', 'slope'):.2f}",
            f"{r_('raw', 'intercept'):+.2f}",
        ])

def make_doc(title, headers, rows, note):
    d = Document()
    s = d.sections[0]
    s.left_margin = s.right_margin = s.top_margin = s.bottom_margin = Cm(1.5)
    t = d.add_paragraph()
    r_ = t.add_run(title)
    r_.bold = True; r_.font.size = Pt(10); r_.font.name = 'Arial'
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

note = (
    "AF, atrial fibrillation; RBBB, right bundle branch block; LBBB, left bundle branch block; "
    "IAVB, first-degree atrioventricular block; NORM, normal ECG; ECE, expected calibration error "
    "(10 equal-width probability bins weighted by bin occupancy). One multi-task model (shared backbone, "
    "five-output head) was trained on PTB-XL (folds 1\u20138; fold 9 checkpoint selection by macro validation AUC) "
    "and applied without refitting to the PTB-XL test fold (prevalence reference) and to both external cohorts. "
    "Outcome definitions are operational: PTB-XL scp codes AFIB, CRBBB|IRBBB, CLBBB|ILBBB, 1AVB, NORM; CODE-15% "
    "labels AF, RBBB, LBBB, 1dAVb, normal_ecg; Chapman-Shaoxing SNOMED 164889003, 59118001, 713427006/713426006/164909002, "
    "270492004, and normal ECG proxied by isolated sinus rhythm (426783006) with no competing diagnoses (Chapman has no "
    "dedicated normal-ECG code); atrial flutter (164890007) excluded by design. The multi-task model takes the raw "
    "waveform plus the same 18 waveform-derived tabular inputs (demographics, heart-rate variability, morphology) used "
    "by the fusion network. Repair ladder: raw, zero-data prior "
    "correction (logit shift by target prevalence minus model-implied development prior, i.e., mean validation-fold "
    "prediction; needs target prevalence but no labelled target sample), Platt and beta recalibration both fitted by "
    "10-fold cross-fitting within each external cohort. AUC is invariant to strictly monotone recalibration. All ten "
    "external cells met the pre-specified \u226550-event reporting gate (range 96\u20131,775 events)."
)

d8 = make_doc(
    "Table S8. Per-outcome calibration transport and repair ladder across external cohorts",
    ["Outcome", "External cohort", "n (events)", "Prevalence, dev \u2192 external", "AUC (raw)",
     "ECE raw", "ECE zero-data", "ECE Platt", "ECE beta", "Slope (raw)", "Intercept (raw)"],
    rows, note)
d8.save(f"{TAB}\\Table S8.docx")
print(f"saved Table S8.docx | rows: {len(rows)}")
for r_ in rows:
    print("  ", r_)
