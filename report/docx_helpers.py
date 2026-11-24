"""Low-level helpers for building the NTCC report with python-docx."""
from __future__ import annotations

import copy
import re

from docx.enum.section import WD_SECTION
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

FONT = "Times New Roman"
BODY_PT = 12


def set_margins(section):
    section.left_margin = section.right_margin = Inches(1.25)
    section.top_margin = section.bottom_margin = Inches(1.0)
    section.header_distance = section.footer_distance = Inches(0.5)
    section.page_width, section.page_height = Inches(8.27), Inches(11.69)


def set_page_numbering(section, fmt: str | None, start: int | None):
    sp = section._sectPr
    for el in sp.findall(qn("w:pgNumType")):
        sp.remove(el)
    if fmt is None and start is None:
        return
    el = OxmlElement("w:pgNumType")
    if fmt:
        el.set(qn("w:fmt"), fmt)
    if start is not None:
        el.set(qn("w:start"), str(start))
    cols = sp.find(qn("w:cols"))
    if cols is not None:
        cols.addprevious(el)
    else:
        sp.append(el)


def add_field(paragraph, instr: str, placeholder: str = ""):
    """Insert a complex field (PAGE, TOC, ...) into ``paragraph``."""
    def fld(t):
        r = OxmlElement("w:r")
        f = OxmlElement("w:fldChar")
        f.set(qn("w:fldCharType"), t)
        if t == "begin":
            f.set(qn("w:dirty"), "true")
        r.append(f)
        return r
    paragraph._p.append(fld("begin"))
    r = OxmlElement("w:r")
    it = OxmlElement("w:instrText")
    it.set(qn("xml:space"), "preserve")
    it.text = f" {instr} "
    r.append(it)
    paragraph._p.append(r)
    paragraph._p.append(fld("separate"))
    run = paragraph.add_run(placeholder)
    paragraph._p.append(fld("end"))
    return run


def clear_hf(part):
    el = part._element
    for child in list(el):
        el.remove(child)


def page_number_paragraph(part, align):
    clear_hf(part)
    p = part.add_paragraph()
    p.alignment = align
    r = add_field(p, "PAGE", "1")
    r.font.name = FONT
    r.font.size = Pt(11)
    return p


def style_font(style, size=None, bold=None, italic=None, color=None):
    style.font.name = FONT
    rpr = style.element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.insert(0, rfonts)
    for a in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rfonts.set(qn(a), FONT)
    for a in ("w:asciiTheme", "w:hAnsiTheme", "w:cstheme", "w:eastAsiaTheme"):
        if rfonts.get(qn(a)) is not None:
            del rfonts.attrib[qn(a)]
    if size:
        style.font.size = Pt(size)
    if bold is not None:
        style.font.bold = bold
    if italic is not None:
        style.font.italic = italic
    if color is not None:
        style.font.color.rgb = RGBColor.from_string(color)


def ensure_paragraph_style(doc, name, base="Normal"):
    try:
        return doc.styles[name]
    except KeyError:
        s = doc.styles.add_style(name, WD_STYLE_TYPE.PARAGRAPH)
        s.base_style = doc.styles[base]
        return s


def setup_styles(doc):
    st = doc.styles
    for name in ("Normal", "Body Text"):
        s = st[name]
        style_font(s, BODY_PT)
    h1 = st["Heading 1"]
    style_font(h1, 16, True, color="000000")
    h1.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    h1.paragraph_format.space_before = Pt(0)
    h1.paragraph_format.space_after = Pt(18)
    h1.paragraph_format.keep_with_next = True
    h1.paragraph_format.first_line_indent = Inches(0)
    h2 = st["Heading 2"]
    style_font(h2, 14, True, color="000000")
    h2.paragraph_format.space_before = Pt(12)
    h2.paragraph_format.space_after = Pt(6)
    h2.paragraph_format.keep_with_next = True
    h2.paragraph_format.first_line_indent = Inches(0)
    h2.paragraph_format.line_spacing = 1.5
    h3 = st["Heading 3"]
    style_font(h3, 12, True, color="000000")
    h3.paragraph_format.space_before = Pt(8)
    h3.paragraph_format.space_after = Pt(4)
    h3.paragraph_format.keep_with_next = True
    h3.paragraph_format.first_line_indent = Inches(0)
    for h in (h1, h2, h3):
        h.font.italic = False
        h.paragraph_format.keep_together = True

    body = ensure_paragraph_style(doc, "Report Body")
    style_font(body, BODY_PT)
    pf = body.paragraph_format
    pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    pf.line_spacing = 1.5
    pf.first_line_indent = Inches(0.5)
    pf.space_before = Pt(0)
    pf.space_after = Pt(6)

    for name in ("Figure Caption", "Table Caption"):
        c = ensure_paragraph_style(doc, name)
        style_font(c, 11, False, italic=False)
        c.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
        c.paragraph_format.line_spacing = 1.0
        c.paragraph_format.first_line_indent = Inches(0)
        c.paragraph_format.space_before = Pt(4)
        c.paragraph_format.space_after = Pt(10)
    st["Table Caption"].paragraph_format.keep_with_next = True
    st["Table Caption"].paragraph_format.space_before = Pt(8)
    st["Table Caption"].paragraph_format.space_after = Pt(4)

    fig = ensure_paragraph_style(doc, "Figure Holder")
    fig.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    fig.paragraph_format.first_line_indent = Inches(0)
    fig.paragraph_format.keep_with_next = True
    fig.paragraph_format.space_before = Pt(6)
    fig.paragraph_format.space_after = Pt(0)
    fig.paragraph_format.line_spacing = 1.0

    ref = ensure_paragraph_style(doc, "Reference Entry")
    style_font(ref, BODY_PT)
    ref.paragraph_format.left_indent = Inches(0.5)
    ref.paragraph_format.first_line_indent = Inches(-0.5)
    ref.paragraph_format.line_spacing = 1.5
    ref.paragraph_format.space_after = Pt(4)
    ref.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT

    code = ensure_paragraph_style(doc, "Code Block")
    style_font(code, 8.5)
    code.font.name = "Consolas"
    code.element.get_or_add_rPr().find(qn("w:rFonts")).set(qn("w:ascii"), "Consolas")
    code.element.get_or_add_rPr().find(qn("w:rFonts")).set(qn("w:hAnsi"), "Consolas")
    code.paragraph_format.line_spacing = 1.0
    code.paragraph_format.space_after = Pt(0)
    code.paragraph_format.first_line_indent = Inches(0)
    code.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT

    eq = ensure_paragraph_style(doc, "Equation")
    style_font(eq, BODY_PT)
    eq.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    eq.paragraph_format.first_line_indent = Inches(0)
    eq.paragraph_format.space_before = Pt(4)
    eq.paragraph_format.space_after = Pt(8)

    lst = ensure_paragraph_style(doc, "Report List")
    style_font(lst, BODY_PT)
    lst.paragraph_format.left_indent = Inches(0.75)
    lst.paragraph_format.first_line_indent = Inches(-0.25)
    lst.paragraph_format.line_spacing = 1.5
    lst.paragraph_format.space_after = Pt(2)
    lst.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    for n in ("TOC 1", "TOC 2", "TOC 3", "Table of Figures"):
        try:
            s = st[n]
        except KeyError:
            s = doc.styles.add_style(n, WD_STYLE_TYPE.PARAGRAPH)
            s.base_style = st["Normal"]
        style_font(s, BODY_PT)
        s.paragraph_format.space_after = Pt(3)
        s.paragraph_format.line_spacing = 1.15
    st["TOC 2"].paragraph_format.left_indent = Inches(0.3)
    st["TOC 1"].font.bold = True
    st["TOC 3"].paragraph_format.left_indent = Inches(0)
    st["TOC 3"].font.bold = False


_MARK = re.compile(r"(\*\*.+?\*\*|\*.+?\*|_\{.+?\}|\^\{.+?\})")


def add_rich_text(paragraph, text: str, size=None):
    """Very small markup: **bold**, *italic*, _{sub}, ^{sup}."""
    for part in _MARK.split(text):
        if not part:
            continue
        if part.startswith("**"):
            r = paragraph.add_run(part[2:-2]); r.bold = True
        elif part.startswith("_{"):
            r = paragraph.add_run(part[2:-1]); r.font.subscript = True
        elif part.startswith("^{"):
            r = paragraph.add_run(part[2:-1]); r.font.superscript = True
        elif part.startswith("*"):
            r = paragraph.add_run(part[1:-1]); r.italic = True
        else:
            r = paragraph.add_run(part)
        if size:
            r.font.size = Pt(size)
    return paragraph


def page_break(doc):
    p = doc.add_paragraph()
    p.paragraph_format.first_line_indent = Inches(0)
    p.add_run().add_break(WD_BREAK.PAGE)
    return p


def shade_cell(cell, hex_fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_fill)
    tcPr.append(shd)


def set_cell_margins(table, top=40, bottom=40, left=80, right=80):
    tblPr = table._tbl.tblPr
    mar = OxmlElement("w:tblCellMar")
    for k, v in (("top", top), ("left", left), ("bottom", bottom), ("right", right)):
        e = OxmlElement(f"w:{k}")
        e.set(qn("w:w"), str(v))
        e.set(qn("w:type"), "dxa")
        mar.append(e)
    tblPr.append(mar)


def set_table_borders(table, color="808080", size=4):
    tblPr = table._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        e = OxmlElement(f"w:{edge}")
        e.set(qn("w:val"), "single")
        e.set(qn("w:sz"), str(size))
        e.set(qn("w:space"), "0")
        e.set(qn("w:color"), color)
        borders.append(e)
    tblPr.append(borders)
    layout = OxmlElement("w:tblLayout")
    layout.set(qn("w:type"), "fixed")
    tblPr.append(layout)


def add_table(doc, header, rows, widths_in, font_pt=10, align=None):
    t = doc.add_table(rows=1 + len(rows), cols=len(header))
    set_table_borders(t)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_cell_margins(t)
    if align is None:
        align = ["left"] + ["left" if max(len(str(r[c])) for r in rows) > 28 else "center"
                            for c in range(1, len(header))]
    amap = {"left": WD_ALIGN_PARAGRAPH.LEFT, "center": WD_ALIGN_PARAGRAPH.CENTER, "right": WD_ALIGN_PARAGRAPH.RIGHT}
    keep_whole = len(rows) <= 25
    for r_i, row in enumerate([header] + rows):
        tr = t.rows[r_i]
        trPr = tr._tr.get_or_add_trPr()
        cs = OxmlElement("w:cantSplit")
        cs.set(qn("w:val"), "true")
        trPr.append(cs)
        if r_i == 0:
            h = OxmlElement("w:tblHeader")
            h.set(qn("w:val"), "true")
            trPr.append(h)
        for c_i, val in enumerate(row):
            cell = tr.cells[c_i]
            cell.width = Inches(widths_in[c_i])
            p = cell.paragraphs[0]
            p.style = doc.styles["Normal"]
            p.paragraph_format.first_line_indent = Inches(0)
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.0
            p.alignment = amap[align[c_i]] if r_i > 0 else WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.keep_with_next = keep_whole and r_i < len(rows)
            add_rich_text(p, str(val), size=font_pt)
            if r_i == 0:
                for r in p.runs:
                    r.bold = True
                shade_cell(cell, "D9E2F3")
    return t


def copy_sectpr(src_sectPr):
    return copy.deepcopy(src_sectPr)


def new_section_paragraph(doc, sectPr):
    """Append an empty paragraph carrying ``sectPr`` (ends the current section)."""
    p = doc.add_paragraph()
    p.paragraph_format.first_line_indent = Inches(0)
    pPr = p._p.get_or_add_pPr()
    pPr.append(sectPr)
    return p


__all__ = [n for n in dir() if not n.startswith("_")] + ["WD_ALIGN_PARAGRAPH", "WD_SECTION", "Inches", "Pt"]
