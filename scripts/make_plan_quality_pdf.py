"""Typeset the CHAI-vs-CheckMate plan-quality note as a shareable PDF.

Mirrors reports/CHAI_vs_CheckMate_plan_quality.md and embeds the two figures
produced by scripts/make_colleague_figures.py.
"""
import os

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether,
                                PageBreak, PageTemplate, Paragraph, Spacer,
                                Table, TableStyle)

OUT = "reports/CHAI_vs_CheckMate_plan_quality.pdf"
FIG = "reports/figures"

INK = colors.HexColor("#22201E")
MUTED = colors.HexColor("#6B6660")
C_BASE = colors.HexColor("#C25E3A")
C_CHAI = colors.HexColor("#1F6F7A")
RULE = colors.HexColor("#D8D3CC")
BAND = colors.HexColor("#F4F1ED")
WARN_BG = colors.HexColor("#FBF3EE")
TEAL_BG = colors.HexColor("#EDF3F4")

PAGE_W, PAGE_H = A4
MARGIN = 20 * mm
CONTENT_W = PAGE_W - 2 * MARGIN

getSampleStyleSheet()


def S(name, **kw):
    base = dict(name=name, fontName="Helvetica", fontSize=9.5, leading=13.6,
                textColor=INK, alignment=TA_LEFT, spaceAfter=0)
    base.update(kw)
    return ParagraphStyle(**base)


TITLE = S("t", fontName="Helvetica-Bold", fontSize=19, leading=23, spaceAfter=5)
DECK = S("d", fontSize=11, leading=15.5, textColor=C_CHAI,
         fontName="Helvetica-Bold", spaceAfter=9)
META = S("m", fontSize=8, leading=11.5, textColor=MUTED, spaceAfter=2)
H2 = S("h2", fontName="Helvetica-Bold", fontSize=13, leading=16,
       spaceBefore=15, spaceAfter=6)
H3 = S("h3", fontName="Helvetica-Bold", fontSize=10.5, leading=14,
       spaceBefore=11, spaceAfter=4)
BODY = S("b", spaceAfter=7)
SMALL = S("s", fontSize=8.4, leading=11.8, textColor=MUTED, spaceAfter=5)
CAP = S("c", fontSize=8.2, leading=11.5, textColor=MUTED, spaceBefore=3,
        spaceAfter=2)
BULLET = S("bl", spaceAfter=5, leftIndent=11, bulletIndent=1)
WARN = S("w", fontSize=9, leading=12.8)
WARNH = S("wh", fontName="Helvetica-Bold", fontSize=9, leading=12.8,
          textColor=C_BASE, spaceAfter=3)
WARNH_T = S("wht", fontName="Helvetica-Bold", fontSize=9, leading=12.8,
            textColor=C_CHAI, spaceAfter=3)


def callout(title, body, bg=WARN_BG, bar=C_BASE, head=WARNH):
    inner = Table([[Paragraph(title, head)], [Paragraph(body, WARN)]],
                  colWidths=[CONTENT_W - 15 * mm])
    inner.setStyle(TableStyle([
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    outer = Table([[inner]], colWidths=[CONTENT_W])
    outer.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg),
        ("LINEBEFORE", (0, 0), (0, -1), 2.4, bar),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("RIGHTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    return outer


def comparison_table():
    hdr = ParagraphStyle("th", parent=BODY, fontName="Helvetica-Bold",
                         fontSize=9, leading=12, textColor=colors.white)
    lab = ParagraphStyle("tl", parent=BODY, fontSize=9, leading=12, spaceAfter=0)
    rows = [
        ["", "B &nbsp;CheckMate + 10 operators", "C &nbsp;CHAI, same 30 operators"],
        ["Control plane", "off", "<b>on</b>"],
        ["Operator catalogue", "T1-T30 (32 incl. auto-gen)", "identical"],
        ["Operators offered to the planner", "32 (all)",
         "<b>16 of 32</b> <font size=7.5>(median; 15-17)</font>"],
        ["Distinct per-function menus <font size=7.5>(ar-iclib)</font>", "1", "<b>7</b>"],
        ["Accepted plans", "20", "13"],
        ["Hardware-impossible operators delivered", "1", "<b>0</b>"],
        ["Unsupported labels <i>detected</i> in generation", "35", "37"],
        ["Unsupported labels <i>shipped</i> in the plan", "<b>10</b>", "<b>0</b>"],
        ["Plans carrying at least one false label",
         "<b>10 of 20 &nbsp;(50%)</b>", "<b>0 of 13 &nbsp;(0%)</b>"],
    ]
    data = [[Paragraph(c, hdr) for c in rows[0]]]
    for r in rows[1:]:
        data.append([Paragraph(c, lab) for c in r])

    t = Table(data, colWidths=[CONTENT_W * 0.42, CONTENT_W * 0.29,
                               CONTENT_W * 0.29], repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), INK),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("LINEBELOW", (0, 1), (-1, -2), 0.4, RULE),
        ("BOX", (0, 0), (-1, -1), 0.6, RULE),
        ("LINEAFTER", (0, 0), (0, -1), 0.4, RULE),
        ("LINEAFTER", (1, 0), (1, -1), 0.4, RULE),
    ]
    for i in range(1, len(data)):
        if i % 2 == 0:
            style.append(("BACKGROUND", (0, i), (-1, i), BAND))
    style += [
        ("BACKGROUND", (0, 9), (-1, 9), TEAL_BG),
        ("TEXTCOLOR", (1, 8), (1, 9), C_BASE),
        ("TEXTCOLOR", (2, 8), (2, 9), C_CHAI),
    ]
    t.setStyle(TableStyle(style))
    return t


def figure(name, width_frac=1.0):
    path = os.path.join(FIG, name + ".png")
    iw, ih = ImageReader(path).getSize()
    w = CONTENT_W * width_frac
    return Image(path, width=w, height=w * ih / iw)


def on_page(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(MARGIN, 12 * mm,
                      "CHAI - plan-quality evidence  |  "
                      "run planquality_20260906_164509")
    canvas.drawRightString(PAGE_W - MARGIN, 12 * mm, str(doc.page))
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.5)
    canvas.line(MARGIN, 15.5 * mm, PAGE_W - MARGIN, 15.5 * mm)
    canvas.restoreState()


def build():
    doc = BaseDocTemplate(OUT, pagesize=A4, leftMargin=MARGIN,
                          rightMargin=MARGIN, topMargin=MARGIN,
                          bottomMargin=22 * mm,
                          title="CHAI vs CheckMate - plan quality",
                          author="CHAI")
    frame = Frame(MARGIN, 22 * mm, CONTENT_W, PAGE_H - MARGIN - 22 * mm,
                  id="body")
    doc.addPageTemplates([PageTemplate(id="p", frames=[frame], onPage=on_page)])

    f = []
    f.append(Paragraph("Does CHAI plan better than CheckMate "
                       "with the same operators?", TITLE))
    f.append(Paragraph("Yes, on plan quality &mdash; and the gain comes from "
                       "the control plane, not the extra operators.", DECK))
    f.append(Paragraph(
        "Target MSP430FR5994 (no FPU, no SIMD, 10&nbsp;KB RAM, 16&nbsp;MHz) "
        "&nbsp;|&nbsp; 6 benchmarks x 3 repeats &nbsp;|&nbsp; 7B planner, "
        "temperature 0 &nbsp;|&nbsp; source "
        "<font face='Courier'>reports/plan_quality_data.json</font>", META))
    f.append(Spacer(1, 9))

    f.append(Paragraph("The comparison that matters", H2))
    f.append(Paragraph(
        "The obvious objection to CHAI is <i>&ldquo;you just added ten more "
        "operators.&rdquo;</i> Arm <b>B</b> rules that out: it is CheckMate with "
        "the <b>identical 30-operator catalogue</b> CHAI uses, and no control "
        "plane. Arm <b>C</b> is CHAI with the same catalogue and the control "
        "plane on. Both arms choose from exactly the same menu, so any "
        "difference between them is the control plane.", BODY))
    f.append(Spacer(1, 3))
    f.append(comparison_table())
    f.append(Spacer(1, 13))

    f.append(Paragraph("1.&nbsp; Both planners mislabel at the same rate. "
                       "Only one gets caught.", H3))
    f.append(Paragraph(
        "35 versus 37 unsupported operator labels were detected during "
        "generation &mdash; statistically indistinguishable. The model is not "
        "smarter under CHAI. What differs is what survives to the delivered "
        "artifact: <b>10 versus 0</b>. Exactly half of arm B&rsquo;s plans ship "
        "at least one operator label the code does not implement. CHAI detects "
        "the same mistakes and removes them before acceptance. Figure 1 "
        "overleaf shows both quantities side by side.", BODY))
    f.append(callout(
        "Caveat &mdash; state this plainly when citing the figure",
        "CHAI&rsquo;s zero is an <i>enforcement outcome</i>, not evidence of a "
        "better planner. CHAI prunes disproven labels before writing the "
        "accepted artifact, so the zero is true by construction. The honest "
        "claim is <b>&ldquo;CHAI detects and removes the false labels that "
        "CheckMate ships&rdquo;</b> &mdash; never &ldquo;CHAI&rsquo;s model "
        "makes fewer mistakes.&rdquo; The 35-vs-37 detection bars are shown "
        "precisely so the figure cannot be misread."))

    f.append(PageBreak())

    f.append(figure("chai_vs_checkmate_provenance", 1.0))
    f.append(Paragraph("<b>Figure 1.</b> Grey = operator labels the planner "
                       "got wrong during generation. Coloured = labels that "
                       "survived into the delivered plan. Both arms draw from "
                       "the identical 30-operator catalogue.", CAP))

    f.append(Paragraph("Where the control plane intervenes", H2))
    f.append(Paragraph("2.&nbsp; The hardware gate removes half the menu "
                       "before the model sees it.", H3))
    f.append(Paragraph(
        "CHAI filters the catalogue against the target&rsquo;s real "
        "capabilities: 32 operators down to a median of 16. Arm B sees all 32, "
        "including operators the chip physically cannot execute.", BODY))
    f.append(callout(
        "Concrete failure",
        "On <font face='Courier'>fft-iclib</font>, arm B selected "
        "<b>T24 &ldquo;Magnitude-Only FFT&rdquo; &mdash; which requires an FPU "
        "&mdash; on a chip with no FPU.</b> The same plan also declared T24 "
        "without implementing it. One operator, two independent defects, "
        "shipped in a delivered plan. CHAI never offered T24 for this target.",
        bg=TEAL_BG, bar=C_CHAI, head=WARNH_T))

    f.append(PageBreak())

    f.append(figure("chai_vs_checkmate_hardware", 0.88))
    f.append(Paragraph("<b>Figure 2.</b> The three stages an operator "
                       "catalogue alone cannot reach.", CAP))
    f.append(Spacer(1, 6))

    f.append(Paragraph("3.&nbsp; CHAI plans per function; CheckMate plans once.",
                       H3))
    f.append(Paragraph(
        "On <font face='Courier'>ar-iclib</font>, arm B gave every function the "
        "<i>identical</i> operator menu (1 distinct menu, all three repeats). "
        "CHAI produced <b>7 distinct menus</b> across the same functions "
        "&mdash; each derived from what that function actually does.", BODY))

    doc.build(f)
    print("wrote " + OUT)


if __name__ == "__main__":
    build()
