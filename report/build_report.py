"""Assemble the NTCC report (.docx) from the template and the experiment results.

python report/build_report.py  ->  report/NTCC_PD_Report.docx
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import docx
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT
from docx.shared import Inches, Pt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import docx_helpers as H  # noqa: E402
from references import REFERENCES  # noqa: E402
from results_context import load_context  # noqa: E402
from text_intro import CH1, CH2, TITLE  # noqa: E402
from text_method import ch3  # noqa: E402
from text_results import ABSTRACT, KEYWORDS, appendices, ch4, ch5  # noqa: E402

TEMPLATE = ROOT / "ntcc_regulations" / "NTCC_v2.docx"
OUT = HERE / "NTCC_PD_Report.docx"
FIG = ROOT / "artifacts" / "figures"

STUDENT = "[Student Name]"
ENROL = "[Enrollment No.]"
GUIDE = "[Faculty Guide Name]"
GUIDE_POS = "[Designation]"
PROGRAMME_SHORT = "B.Sc (IT)"
SEMESTER = "Semester [__] Section [__]"
BATCH = "Batch: [____]"


def set_runs(p, runs):
    for r in list(p.runs):
        r._r.getparent().remove(r._r)
    for text, bold in runs:
        r = p.add_run(text)
        r.bold = bold
    return p


class Writer:
    def __init__(self, doc):
        self.doc = doc
        self.text_width = 8.27 - 2.5

    def para(self, text, style="Report Body"):
        p = self.doc.add_paragraph(style=style)
        H.add_rich_text(p, text)
        return p

    def h1(self, text, new_page=True):
        p = self.doc.add_paragraph(text, style="Heading 1")
        p.paragraph_format.page_break_before = new_page
        return p

    def h2(self, text):
        return self.doc.add_paragraph(text, style="Heading 2")

    def h3(self, text):
        return self.doc.add_paragraph(text, style="Heading 3")

    def numbered(self, items):
        for i, it in enumerate(items, 1):
            p = self.doc.add_paragraph(style="Report List")
            p.paragraph_format.tab_stops.add_tab_stop(Inches(0.75))
            H.add_rich_text(p, f"{i}.\t{it}")

    def bullets(self, items):
        for it in items:
            p = self.doc.add_paragraph(style="Report List")
            p.paragraph_format.tab_stops.add_tab_stop(Inches(0.75))
            H.add_rich_text(p, f"–\t{it}")

    def figure(self, fname, caption, width):
        p = self.doc.add_paragraph(style="Figure Holder")
        p.add_run().add_picture(str(FIG / fname), width=Inches(min(width, self.text_width)))
        c = self.doc.add_paragraph(style="Figure Caption")
        H.add_rich_text(c, caption)

    def table(self, caption, header, rows, widths, font_pt=10, align=None):
        c = self.doc.add_paragraph(style="Table Caption")
        H.add_rich_text(c, caption)
        scale = min(1.0, self.text_width / sum(widths))
        H.add_table(self.doc, header, rows, [w * scale for w in widths], font_pt, align)
        sp = self.doc.add_paragraph(style="Normal")
        sp.paragraph_format.space_after = Pt(4)
        sp.paragraph_format.line_spacing = 1.0

    def equation(self, text, number):
        p = self.doc.add_paragraph(style="Equation")
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        ts = p.paragraph_format.tab_stops
        ts.add_tab_stop(Inches(self.text_width / 2), WD_TAB_ALIGNMENT.CENTER)
        ts.add_tab_stop(Inches(self.text_width), WD_TAB_ALIGNMENT.RIGHT)
        H.add_rich_text(p, "\t" + text + "\t" + number)
        for r in p.runs[:-1]:
            r.italic = r.italic if r.italic is not None else False

    def code(self, lines):
        for ln in lines:
            p = self.doc.add_paragraph(style="Code Block")
            p.add_run(ln if ln else " ")

    def render(self, blocks):
        for b in blocks:
            kind = b[0]
            if kind == "h1":
                self.h1(b[1])
            elif kind == "h2":
                self.h2(b[1])
            elif kind == "h3":
                self.h3(b[1])
            elif kind == "p":
                self.para(b[1])
            elif kind == "list":
                self.numbered(b[1])
            elif kind == "bullets":
                self.bullets(b[1])
            elif kind == "fig":
                self.figure(b[1], b[2], b[3])
            elif kind == "table":
                self.table(*b[1:])
            elif kind == "eq":
                self.equation(b[1], b[2])
            elif kind == "code":
                self.code(b[1])
            elif kind == "pagebreak":
                H.page_break(self.doc)
            else:
                raise ValueError(kind)


def fill_front_matter(doc):
    P = doc.paragraphs
    for idx, text, size in ((3, TITLE, 16), (14, STUDENT, 14), (15, ENROL, 14), (19, GUIDE, 14)):
        P[idx].style = doc.styles["Normal"]
        P[idx].alignment = WD_ALIGN_PARAGRAPH.CENTER
        P[idx].paragraph_format.space_after = Pt(4)
        set_runs(P[idx], [(text, True)])
        for r in P[idx].runs:
            r.font.size = Pt(size)
    set_runs(P[20], [(GUIDE_POS, True)])
    set_runs(P[33], [("I, ", False), (STUDENT, True), (f", student of {PROGRAMME_SHORT} {SEMESTER}, hereby declare that "
                     "the project titled ", False), (f"“{TITLE}”", True),
                     (", which is submitted by me to the Department of Information Technology and Engineering, Amity "
                      "University in Tashkent, Uzbekistan, in partial fulfilment of the requirement for the award of "
                      "the degree of Bachelor of Science in Information Technology, has not previously formed the "
                      "basis for the award of any degree, diploma or other similar title or recognition.", False)])
    set_runs(P[35], [("The author attests that permission has been obtained for the use of any copyrighted material "
                      "appearing in this project report other than brief excerpts requiring only proper "
                      "acknowledgement in scholarly writing, and that all such use is acknowledged.", False)])
    set_runs(P[42], [(STUDENT + " ", False), (ENROL, True)])
    set_runs(P[43], [(PROGRAMME_SHORT, True)])
    set_runs(P[44], [(SEMESTER, True)])
    set_runs(P[45], [(BATCH, True)])
    set_runs(P[49], [("This is to certify that ", False), (STUDENT, True),
                     (", student of Bachelor of Science in Information Technology, has carried out the work presented "
                      "in the term paper report entitled ", False), (f"“{TITLE}”", True),
                     (" as a part of the programme of Bachelor of Science in Information Technology at Amity "
                      "University in Tashkent, Uzbekistan, under my supervision.", False)])
    set_runs(P[55], [(GUIDE, True)])
    set_runs(P[57], [(GUIDE_POS + ", Department of Information Technology and Engineering", False)])
    set_runs(P[62], [("I would like to thank Amity University in Tashkent for the opportunity to work on this project "
                      "and for the resources provided during the semester. I am grateful to my faculty guide, ", False),
                     (GUIDE, True),
                     (", for the advice on the structure of the work, for the questions that made me check my "
                      "results more carefully, and for the patience with which my drafts were reviewed. I also thank "
                      "the teachers of the Department of Information Technology and Engineering, whose courses on "
                      "programming, statistics and machine learning gave me the background needed for this study. "
                      "Finally, I thank my family and my fellow students for their support and for the discussions "
                      "that helped me to see the problem from different sides.", False)])
    set_runs(P[70], [(STUDENT, False)])
    set_runs(P[71], [(ENROL, True)])
    last = P[71]._p
    # tighten the title page so that it fits on one page with the regulation margins
    for idx in (7, 8, 10, 18, 21, 22, 23):
        P[idx]._p.getparent().remove(P[idx]._p)
    return last


def truncate_after(doc, keep):
    body = doc.element.body
    started = False
    for el in list(body):
        if started and el.tag != docx.oxml.ns.qn("w:sectPr"):
            body.remove(el)
        if el is keep:
            started = True


def pre_pages(doc, W: Writer, R):
    # table of contents
    p = doc.add_paragraph(style="Normal")
    p.paragraph_format.page_break_before = True
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(18)
    r = p.add_run("TABLE OF CONTENTS")
    r.bold = True
    r.font.size = Pt(16)
    p = doc.add_paragraph(style="Normal")
    H.add_field(p, 'TOC \\o "1-2" \\h \\z \\u', "Right-click and choose Update Field to build the table of contents.")

    W.h1("LIST OF TABLES")
    p = doc.add_paragraph(style="Normal")
    H.add_field(p, 'TOC \\h \\z \\t "Table Caption,3"', "Update field to build the list of tables.")
    W.h1("LIST OF FIGURES")
    p = doc.add_paragraph(style="Normal")
    H.add_field(p, 'TOC \\h \\z \\t "Figure Caption,3"', "Update field to build the list of figures.")

    W.h1("ABSTRACT")
    for t in ABSTRACT(R):
        W.para(t)
    kp = W.para(f"**Keywords:** {KEYWORDS}")
    kp.paragraph_format.first_line_indent = Inches(0)


def configure_sections(doc):
    secs = doc.sections
    for s in secs:
        H.set_margins(s)
    # 0: title page, no number
    s0 = secs[0]
    for part in (s0.header, s0.footer, s0.first_page_header, s0.first_page_footer,
                 s0.even_page_header, s0.even_page_footer):
        part.is_linked_to_previous = False
        H.clear_hf(part)
        part.add_paragraph()
    H.set_page_numbering(s0, None, None)
    # 1..n-2: preliminary pages, lower-case Roman, bottom centre, start at ii
    for i, s in enumerate(secs[1:-1], 1):
        s.start_type = docx.enum.section.WD_SECTION.NEW_PAGE
        if i == 1:
            s.header.is_linked_to_previous = False
            s.footer.is_linked_to_previous = False
            H.clear_hf(s.header)
            s.header.add_paragraph()
            H.page_number_paragraph(s.footer, WD_ALIGN_PARAGRAPH.CENTER)
            H.set_page_numbering(s, "lowerRoman", 2)
        else:
            s.header.is_linked_to_previous = True
            s.footer.is_linked_to_previous = True
            H.set_page_numbering(s, "lowerRoman", None)
    # last: main text, Arabic, top right, start at 1
    sl = secs[-1]
    sl.start_type = docx.enum.section.WD_SECTION.NEW_PAGE
    sl.header.is_linked_to_previous = False
    sl.footer.is_linked_to_previous = False
    H.page_number_paragraph(sl.header, WD_ALIGN_PARAGRAPH.RIGHT)
    H.clear_hf(sl.footer)
    sl.footer.add_paragraph()
    H.set_page_numbering(sl, "decimal", 1)


def main():
    R = load_context()
    doc = docx.Document(str(TEMPLATE))
    H.setup_styles(doc)
    last = fill_front_matter(doc)
    truncate_after(doc, last)
    W = Writer(doc)
    pre_pages(doc, W, R)

    # close the preliminary-pages section with a copy of the body section properties
    body_sect = doc.element.body.find(docx.oxml.ns.qn("w:sectPr"))
    H.new_section_paragraph(doc, copy.deepcopy(body_sect))

    blocks = CH1 + CH2 + ch3(R) + ch4(R) + ch5(R)
    blocks[0] = ("h1_first", blocks[0][1])
    for b in blocks:
        if b[0] == "h1_first":
            W.h1(b[1], new_page=False)
        else:
            W.render([b])

    W.h1("REFERENCES")
    for ref in sorted(REFERENCES, key=lambda s: s.lower()):
        p = doc.add_paragraph(style="Reference Entry")
        H.add_rich_text(p, ref)

    W.render(appendices(R))
    configure_sections(doc)

    doc.core_properties.title = TITLE
    doc.core_properties.author = STUDENT
    doc.core_properties.comments = ""
    doc.save(str(OUT))
    print("saved", OUT)


if __name__ == "__main__":
    main()
