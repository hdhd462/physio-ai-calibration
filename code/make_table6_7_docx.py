import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA_DIR, PROJ_DIR, FIG_DIR, SUB_DIR, PATCH_DIR, NHANES_DIR, PROJ1_DIR, SHAOXING_DIR, PTBXL_DIR
import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
from docx import Document
from docx.shared import Pt, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn, nsdecls
from docx.oxml import parse_xml

OUT = DATA_DIR
FIG = FIG_DIR

def new_doc():
    doc = Document()
    style = doc.styles['Normal']
    style.font.name = 'Times New Roman'
    style.font.size = Pt(9)
    style.element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
    for section in doc.sections:
        section.left_margin = Cm(1.5)
        section.right_margin = Cm(1.5)
        section.top_margin = Cm(2.0)
        section.bottom_margin = Cm(2.0)
    return doc

def set_cell_text(cell, text, bold=False, align='center', size=9):
    cell.text = ''
    p = cell.paragraphs[0]
    p.alignment = {'center': WD_ALIGN_PARAGRAPH.CENTER, 'left': WD_ALIGN_PARAGRAPH.LEFT}[align]
    p.paragraph_format.space_before = Pt(1)
    p.paragraph_format.space_after = Pt(1)
    p.paragraph_format.line_spacing = Pt(10)
    run = p.add_run(str(text))
    run.font.size = Pt(size)
    run.font.name = 'Times New Roman'
    run.bold = bold

def three_line(table):
    tblPr = table._tbl.tblPr
    for e in tblPr.findall(qn('w:tblBorders')):
        tblPr.remove(e)
    tblPr.append(parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f'<w:top w:val="single" w:sz="12" w:space="0" w:color="000000"/>'
        f'<w:bottom w:val="single" w:sz="12" w:space="0" w:color="000000"/>'
        f'<w:insideH w:val="none" w:sz="0" w:space="0" w:color="000000"/>'
        f'<w:insideV w:val="none" w:sz="0" w:space="0" w:color="000000"/>'
        f'<w:left w:val="none" w:sz="0" w:space="0" w:color="000000"/>'
        f'<w:right w:val="none" w:sz="0" w:space="0" w:color="000000"/>'
        f'</w:tblBorders>'))

def cell_border(cell, edge, sz=6):
    tcPr = cell._tc.get_or_add_tcPr()
    b = tcPr.find(qn('w:tcBorders'))
    if b is None:
        b = parse_xml(f'<w:tcBorders {nsdecls("w")}/>')
        tcPr.append(b)
    for e in b.findall(qn(f'w:{edge}')):
        b.remove(e)
    b.append(parse_xml(f'<w:{edge} {nsdecls("w")} w:val="single" w:sz="{sz}" w:space="0" w:color="000000"/>'))

def add_title(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(6)
    run = p.add_run(text)
    run.bold = True
    run.font.size = Pt(10)
    run.font.name = 'Times New Roman'

def add_note(doc, text):
    doc.add_paragraph()
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(0)
    run = p.add_run(text)
    run.font.size = Pt(8)
    run.font.name = 'Times New Roman'

# ══════ Table 6 ══════
doc = new_doc()
add_title(doc, 'Table 6. Baseline characteristics of the derivation and external validation cohorts')

headers = ['ᵃDatabase', 'Total\nrecords', 'ᵇAF, n (%)', 'Age, mean ± SD', '', '',
           'ᶜFemale, %', '', 'ᵈP']
rows_data = [
    ['PTB-XL (Germany)', '21,798', '1,514 (6.9)', '70.2 ± 12.0', '58.8 ± 16.8', '<0.001',
     '44.3', '48.2', '0.003'],
    ['Chapman-Shaoxing (China)', '22,526', '1,780 (7.9)', '73.2 ± 11.0', '64.6 ± 17.0', '<0.001',
     '41.5', '41.6', '0.96'],
]

n_cols = 9
table = doc.add_table(rows=2 + len(rows_data), cols=n_cols)
table.alignment = WD_TABLE_ALIGNMENT.CENTER
three_line(table)

r0 = table.rows[0]
set_cell_text(r0.cells[0], '', size=8)
set_cell_text(r0.cells[1], '', size=8)
set_cell_text(r0.cells[2], '', size=8)
r0.cells[3].merge(r0.cells[5])
set_cell_text(r0.cells[3], 'Age, mean ± SD', bold=True, size=8)
set_cell_text(r0.cells[6], 'ᶜFemale, %', bold=True, size=8)
set_cell_text(r0.cells[8], 'ᵈP', bold=True, size=8)

r1 = table.rows[1]
set_cell_text(r1.cells[0], 'ᵃDatabase', bold=True, size=8, align='left')
set_cell_text(r1.cells[1], 'Total records', bold=True, size=8)
set_cell_text(r1.cells[2], 'ᵇAF, n (%)', bold=True, size=8)
set_cell_text(r1.cells[3], 'AF', bold=True, size=8)
set_cell_text(r1.cells[4], 'Non-AF', bold=True, size=8)
set_cell_text(r1.cells[5], 'ᵈP', bold=True, size=8)
set_cell_text(r1.cells[6], 'AF', bold=True, size=8)
set_cell_text(r1.cells[7], 'Non-AF', bold=True, size=8)
set_cell_text(r1.cells[8], 'ᵈP', bold=True, size=8)

for j in range(n_cols):
    cell_border(r0.cells[j], 'bottom', sz=12)
    cell_border(r1.cells[j], 'bottom', sz=6)

for i, row in enumerate(rows_data):
    r = table.rows[i + 2]
    set_cell_text(r.cells[0], row[0], align='left', size=9)
    for j in range(1, n_cols):
        set_cell_text(r.cells[j], row[j], size=9)

widths = [Cm(4.2), Cm(1.9), Cm(2.2), Cm(1.8), Cm(1.8), Cm(1.2), Cm(1.5), Cm(1.7), Cm(1.2)]
for r in table.rows:
    seen = set()
    for j, w in enumerate(widths):
        if j < len(r.cells):
            c = r.cells[j]
            if id(c._tc) in seen:
                continue
            seen.add(id(c._tc))
            c.width = w

add_note(doc, 'Note: AF was defined as presence of AFIB in scp_codes (PTB-XL) or SNOMED 164889003 in ECG statements (Chapman-Shaoxing). '
              'ᵃCohort; ᵇatrial fibrillation; ᶜfemale sex; ᵈP value, independent-samples t-test (age) or chi-square test (sex), AF vs non-AF within each cohort.')
doc.save(f'{FIG}\\table6_baseline.docx')
print('已保存: table6_baseline.docx')

# ══════ Table 7 ══════
doc = new_doc()
add_title(doc, 'Table 7. Multivariable logistic regression models for atrial fibrillation detection (derivation cohort)')

t7 = pd.read_csv(f'{OUT}\\table7_logistic_or.csv')
n_cols = 5
table = doc.add_table(rows=1 + len(t7), cols=n_cols)
table.alignment = WD_TABLE_ALIGNMENT.CENTER
three_line(table)

r0 = table.rows[0]
set_cell_text(r0.cells[0], 'ᵃModel', bold=True, size=8, align='left')
set_cell_text(r0.cells[1], 'ᵇTrain AUC', bold=True, size=8)
set_cell_text(r0.cells[2], 'ᶜFeature', bold=True, size=8, align='left')
set_cell_text(r0.cells[3], 'ᵈOR (95% ᶜCI)', bold=True, size=8)
set_cell_text(r0.cells[4], 'ᵈP value', bold=True, size=8)
for j in range(n_cols):
    cell_border(r0.cells[j], 'bottom', sz=6)

prev_model = None
for i, (_, row) in enumerate(t7.iterrows()):
    r = table.rows[i + 1]
    m = row['Model'] if row['Model'] != prev_model else ''
    prev_model = row['Model']
    bold = m != ''
    set_cell_text(r.cells[0], m, bold=bold, align='left', size=9)
    set_cell_text(r.cells[1], row['Train_AUC'] if m != '' else '', size=9)
    set_cell_text(r.cells[2], row['Feature'], align='left', size=9)
    set_cell_text(r.cells[3], f"{row['OR']} ({row['95%CI']})", size=9)
    set_cell_text(r.cells[4], row['P'], size=9)

widths = [Cm(3.2), Cm(2.2), Cm(3.0), Cm(3.4), Cm(2.4)]
for r in table.rows:
    for j, w in enumerate(widths):
        if j < len(r.cells):
            r.cells[j].width = w

add_note(doc, 'Note: Models fitted on the PTB-XL derivation cohort (folds 1–9, n = 19,538) with class-balanced weights. '
              'ᵃM1: demographics (age, sex); M2: M1 + heart rate and HRV features; M3: M2 + morphological features '
              '(collinear variables, i.e., mean heart rate and a duplicate RR standard deviation identical to SDNN, excluded). '
              'ᵇAUC, area under the receiver operating characteristic curve (in-sample). '
              'ᶜOR, odds ratio (per 1 SD increase for continuous variables); ᶜCI, confidence interval from 500 bootstrap resamples; '
              'ᵈP value from bootstrap standard error.')
doc.save(f'{FIG}\\table7_logistic_or.docx')
print('已保存: table7_logistic_or.docx')