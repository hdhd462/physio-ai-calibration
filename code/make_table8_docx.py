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
df = pd.read_csv(f'{OUT}\\table8_ab_comparison.csv')

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

def set_table_borders_three_line(table):
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

def set_cell_border(cell, edge, sz=6):
    tcPr = cell._tc.get_or_add_tcPr()
    b = tcPr.find(qn('w:tcBorders'))
    if b is None:
        b = parse_xml(f'<w:tcBorders {nsdecls("w")}/>')
        tcPr.append(b)
    for e in b.findall(qn(f'w:{edge}')):
        b.remove(e)
    b.append(parse_xml(f'<w:{edge} {nsdecls("w")} w:val="single" w:sz="{sz}" w:space="0" w:color="000000"/>'))

p = doc.add_paragraph()
p.paragraph_format.space_after = Pt(6)
run = p.add_run('Table 8. Performance of logistic regression and deep learning models for atrial fibrillation detection')
run.bold = True
run.font.size = Pt(10)
run.font.name = 'Times New Roman'

n_cols = 9
n_rows = 2 + len(df)
table = doc.add_table(rows=n_rows, cols=n_cols)
table.alignment = WD_TABLE_ALIGNMENT.CENTER
set_table_borders_three_line(table)

# row0 合并组
r0 = table.rows[0]
r0.cells[2].merge(r0.cells[3])
r0.cells[5].merge(r0.cells[8])
# row1 合并（须在row0合并后独立进行）
r1 = table.rows[1]
r1.cells[5].merge(r1.cells[6])
r1.cells[7].merge(r1.cells[8])

# 注意：merge后表格列引用变化，直接按逻辑位置写
# 结构：row0: [ ][ ][AUC组跨2][ΔAUC][Sens/Spec组跨4]
#        row1: [Model][Approach][Internal][External][ ][Internal跨2][External跨2]
# python-docx merge 后 row.cells 返回的列表可能重复引用，逐行用已合并对象写

# 重新取每行的逻辑单元格（merge后 row.cells 仍返回 n_cols 个引用，其中重复）
def rowcells(row):
    return list(row.cells)

c0 = rowcells(r0)
set_cell_text(c0[0], '', size=8)
set_cell_text(c0[1], '', size=8)
set_cell_text(c0[2], 'ᶜAUC (95% ᶜCI)', bold=True, size=8)
set_cell_text(c0[4], 'ᵈΔAUC', bold=True, size=8)
set_cell_text(c0[5], 'ᵉSensitivity / Specificity', bold=True, size=8)

c1 = rowcells(r1)
set_cell_text(c1[0], 'ᵃModel', bold=True, size=8, align='left')
set_cell_text(c1[1], 'ᵇApproach', bold=True, size=8)
set_cell_text(c1[2], 'Internal', bold=True, size=8)
set_cell_text(c1[3], 'External', bold=True, size=8)
set_cell_text(c1[4], '', bold=True, size=8)
set_cell_text(c1[5], 'Internal', bold=True, size=8)
set_cell_text(c1[7], 'External', bold=True, size=8)

for j in range(n_cols):
    set_cell_border(c0[j], 'bottom', sz=12)
    set_cell_border(c1[j], 'bottom', sz=6)

for i, (_, row) in enumerate(df.iterrows()):
    r = table.rows[i + 2]
    rc = rowcells(r)
    is_dl = row['Model'] in ('CNN', 'ResNet', 'CNNTransformer', 'FusionStack', 'FusionNet')
    # 数据行合并
    rc[5].merge(rc[6])
    rc[7].merge(rc[8])
    set_cell_text(rc[0], row['Model'], bold=is_dl, align='left', size=9)
    set_cell_text(rc[1], row['Approach'], size=8)
    set_cell_text(rc[2], f"{row['Int_AUC']:.3f} ({row['Int_CI']})", size=9)
    set_cell_text(rc[3], f"{row['Ext_AUC']:.3f} ({row['Ext_CI']})", size=9)
    set_cell_text(rc[4], f"{row['dAUC']:.3f}", size=9)
    set_cell_text(rc[5], f"{row['Int_Sens']:.3f} / {row['Int_Spec']:.3f}", size=9)
    set_cell_text(rc[7], f"{row['Ext_Sens']:.3f} / {row['Ext_Spec']:.3f}", size=9)

widths = [Cm(3.0), Cm(3.2), Cm(2.6), Cm(2.6), Cm(1.4), Cm(1.8), Cm(0.1), Cm(1.8), Cm(0.1)]
for r in table.rows:
    seen = set()
    for j, w in enumerate(widths):
        if j < len(r.cells):
            c = r.cells[j]
            if id(c._tc) in seen:
                continue
            seen.add(id(c._tc))
            c.width = w

doc.add_paragraph()
p = doc.add_paragraph()
p.paragraph_format.space_before = Pt(6)
p.paragraph_format.space_after = Pt(0)
run = p.add_run(
    'Note: Metrics computed on the common evaluation cohort (internal n=2,189; external n=22,190). '
    'ᵃM1–M3, logistic regression models; CNN, ResNet, CNNTransformer, deep learning models trained on raw ECG. '
    'ᶜCI, confidence interval (1,000 bootstrap resamples). '
    'ᵈΔAUC, generalization gap (internal AUC − external AUC). '
    'ᵉYouden index threshold derived from the internal test set and applied unchanged to both cohorts. '
    'CNN-Transformer vs M3: external P<0.001; CNN-Transformer vs ResNet external P=0.648 (paired bootstrap). '
    'Fusion network vs CNN-Transformer: internal P=0.021, external P<0.001; fusion network vs fusion stacking external P=0.755.'
)
run.font.size = Pt(8)
run.font.name = 'Times New Roman'

p = doc.add_paragraph()
p.paragraph_format.space_before = Pt(2)
run = p.add_run(
    'ᵃModel; ᵇApproach; ᶜAUC, area under the receiver operating characteristic curve; ᶜCI, confidence interval; '
    'ᵈΔAUC, difference in AUC; ᵉSens, sensitivity; Spec, specificity; HRV, heart rate variability.'
)
run.font.size = Pt(8)
run.font.name = 'Times New Roman'

path = f'{FIG}\\table8_ab_comparison.docx'
doc.save(path)
print(f'已保存: {path}')