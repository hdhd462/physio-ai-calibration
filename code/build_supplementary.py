import docx, os, sys, copy
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA_DIR, PROJ_DIR, FIG_DIR, SUB_DIR, PATCH_DIR, NHANES_DIR, PROJ1_DIR, SHAOXING_DIR, PTBXL_DIR
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn

sys.stdout.reconfigure(encoding='utf-8')

base = SUB_DIR
out = os.path.join(base, 'F2_Supplementary_Materials.docx')

doc = docx.Document()

style = doc.styles['Normal']
style.font.name = 'Times New Roman'
style.font.size = Pt(10.5)
style.element.rPr.rFonts.set(qn('w:eastAsia'), 'Times New Roman')

h = doc.add_paragraph()
h.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = h.add_run('Supplementary Materials')
r.font.bold = True
r.font.size = Pt(16)

def caption(text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(text)
    run.font.bold = True
    run.font.size = Pt(10)
    p.paragraph_format.space_before = Pt(12)
    p.paragraph_format.space_after = Pt(6)
    return p

sectPr = doc.element.body.find(qn('w:sectPr'))

def add_table(src_path):
    src = docx.Document(src_path)
    for child in src.element.body:
        if child.tag == qn('w:sectPr'):
            continue
        el = copy.deepcopy(child)
        if sectPr is not None:
            sectPr.addprevious(el)
        else:
            doc.element.body.append(el)

def add_figure(img_path, width=5.5):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(img_path, width=Inches(width))
    p.paragraph_format.space_before = Pt(6)

for i in range(1, 14):
    name = f'Table S{i}'
    path = f'{base}/tables/{name}.docx'
    if not os.path.exists(path):
        continue
    doc.add_page_break()
    add_table(path)
    print('Added', name)

for i in range(1, 8):
    name = f'Figure S{i}'
    path = f'{base}/picture/{name}.png'
    if not os.path.exists(path):
        continue
    doc.add_page_break()
    add_figure(path)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(name)
    run.font.bold = True
    run.font.size = Pt(10)
    p.paragraph_format.space_before = Pt(6)
    print('Added', name)

doc.save(out)
print()
print('Saved:', out)
