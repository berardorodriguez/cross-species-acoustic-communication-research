from pathlib import Path
import re

import reportlab
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    PageTemplate,
    Paragraph,
    PageBreak,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[2]
MD_PATH = ROOT / "supplementary" / "methods" / "Supplementary_Methods.md"
PDF_PATH = ROOT / "supplementary" / "methods" / "Supplementary_Methods.pdf"
FONT_PATH = Path(reportlab.__file__).resolve().parent / "fonts"
ARTICLE_TITLE = (
    "Affective Information Transmission in Cross-Species Acoustic Communication: "
    "Evidence for Multiscale Acoustic–Affective Organization"
)


EQUATION_BLOCKS = {
    "z_i = \\frac{x_i - mean_i}{SD_i}": (
        "z<sub>i</sub> = (x<sub>i</sub> − mean<sub>i</sub>) / SD<sub>i</sub>"
    ),
    "d_k = \\sqrt{\\sum_{j=1}^{10}(p_j - c_{k,j})^2}": (
        "d<sub>k</sub> = sqrt[sum<sub>j=1</sub><super>10</super> (p<sub>j</sub> − c<sub>k,j</sub>)²]"
    ),
    "RAI_{A,B} = \\frac{d_B - d_A}{d_B + d_A}": (
        "RAI<sub>A,B</sub> = (d<sub>B</sub> − d<sub>A</sub>) / (d<sub>B</sub> + d<sub>A</sub>)"
    ),
    "P_{source} = \\frac{p1_{mean} + p2_{mean} + p3_{mean}}{herd_{est}}": (
        "P<sub>source</sub> = (p1<sub>mean</sub> + p2<sub>mean</sub> + p3<sub>mean</sub>) / herd<sub>est</sub>"
    ),
    "NBS = \\frac{M_{2-8\\,kHz} - M_{0-2\\,kHz}}{M_{2-8\\,kHz} + M_{0-2\\,kHz}}": (
        "NBS = (M<sub>2–8 kHz</sub> − M<sub>0–2 kHz</sub>) / (M<sub>2–8 kHz</sub> + M<sub>0–2 kHz</sub>)"
    ),
    "TL = \\sum_{k=1}^{12} \\sqrt{\\sum_{j=1}^{3}(x_{k+1,j} - x_{k,j})^2}": (
        "TL = sum<sub>k=1</sub><super>12</super> sqrt[sum<sub>j=1</sub><super>3</super> (x<sub>k+1,j</sub> − x<sub>k,j</sub>)²]"
    ),
    "TD = \\sum_{j=1}^{3} \\left[\\frac{1}{13}\\sum_{k=1}^{13}(x_{k,j} - mean_j)^2\\right]": (
        "TD = sum<sub>j=1</sub><super>3</super> [(1/13) sum<sub>k=1</sub><super>13</super> "
        "(x<sub>k,j</sub> − mean<sub>j</sub>)²]"
    ),
    "mean_j = \\frac{1}{13}\\sum_{k=1}^{13}x_{k,j}": (
        "mean<sub>j</sub> = (1/13) sum<sub>k=1</sub><super>13</super> x<sub>k,j</sub>"
    ),
    "D_{mn} = a_m a_n (e^{-3.5x_{mn}} - e^{-5.75x_{mn}})": (
        "D<sub>mn</sub> = a<sub>m</sub>a<sub>n</sub>"
        "(e<super>−3.5x<sub>mn</sub></super> − e<super>−5.75x<sub>mn</sub></super>)"
    ),
    "x_{mn} = \\frac{0.24|f_m-f_n|}{0.021\\min(f_m,f_n)+19}": (
        "x<sub>mn</sub> = 0.24|f<sub>m</sub> − f<sub>n</sub>| / "
        "[0.021 min(f<sub>m</sub>, f<sub>n</sub>) + 19]"
    ),
    "R_t = \\sum_{m<n}\\max(D_{mn}, 0)": (
        "R<sub>t</sub> = sum<sub>m&lt;n</sub> max(D<sub>mn</sub>, 0)"
    ),
    "A_{base} = \\operatorname{logistic}(0.20s_{RMS} + 0.18s_{onset} + 0.15s_{BPM} + 0.12s_{centroid} + 0.10s_{ZCR} + 0.08s_{bandwidth} + 0.07s_{HF/LF} + 0.05s_{tilt} + 0.05s_{SNR})": (
        "A<sub>base</sub> = logistic(0.20s<sub>RMS</sub> + 0.18s<sub>onset</sub> + 0.15s<sub>BPM</sub> + "
        "0.12s<sub>centroid</sub> + 0.10s<sub>ZCR</sub> + 0.08s<sub>bandwidth</sub> + "
        "0.07s<sub>HF/LF</sub> + 0.05s<sub>tilt</sub> + 0.05s<sub>SNR</sub>)"
    ),
    "A_{adj} = \\operatorname{clip}(A_{base} + 0.08P_{HF} - 0.06R_{LF}, 0, 1)": (
        "A<sub>adj</sub> = clip(A<sub>base</sub> + 0.08P<sub>HF</sub> − 0.06R<sub>LF</sub>, 0, 1)"
    ),
}


def xml_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def inline_markup(text: str) -> str:
    text = xml_escape(text)
    text = re.sub(r"`([^`]+)`", r"<font name='Vera'>\1</font>", text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\*(.+?)\*", r"<i>\1</i>", text)
    return text


def draw_page(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#D9E0E6"))
    canvas.setLineWidth(0.5)
    canvas.line(22 * mm, 16 * mm, A4[0] - 22 * mm, 16 * mm)
    canvas.setFont("Vera", 8)
    canvas.setFillColor(colors.HexColor("#59636E"))
    canvas.drawString(22 * mm, 10 * mm, "Supplementary Methods")
    canvas.drawRightString(A4[0] - 22 * mm, 10 * mm, str(doc.page))
    canvas.restoreState()


def deterministic_canvasmaker(*args, **kwargs):
    kwargs["invariant"] = 1
    return Canvas(*args, **kwargs)


def build_pdf(markdown: str) -> None:
    pdfmetrics.registerFont(TTFont("Vera", str(FONT_PATH / "Vera.ttf")))
    pdfmetrics.registerFont(TTFont("Vera-Bold", str(FONT_PATH / "VeraBd.ttf")))
    pdfmetrics.registerFont(TTFont("Vera-Italic", str(FONT_PATH / "VeraIt.ttf")))
    pdfmetrics.registerFontFamily("Vera", normal="Vera", bold="Vera-Bold", italic="Vera-Italic", boldItalic="Vera-Bold")

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="BodyClean", parent=styles["BodyText"], fontName="Vera", fontSize=9.1, leading=12.6, alignment=TA_JUSTIFY, spaceAfter=6.3, textColor=colors.HexColor("#202830")))
    styles.add(ParagraphStyle(name="Section1", parent=styles["Heading1"], fontName="Vera-Bold", fontSize=14, leading=17, spaceBefore=13, spaceAfter=7, keepWithNext=True, textColor=colors.HexColor("#163B56")))
    styles.add(ParagraphStyle(name="Section2", parent=styles["Heading2"], fontName="Vera-Bold", fontSize=11, leading=14, spaceBefore=10, spaceAfter=5, keepWithNext=True, textColor=colors.HexColor("#245B78")))
    styles.add(ParagraphStyle(name="Section3", parent=styles["Heading3"], fontName="Vera-Bold", fontSize=9.8, leading=12, spaceBefore=8, spaceAfter=4, keepWithNext=True, textColor=colors.HexColor("#34495A")))
    styles.add(ParagraphStyle(name="Eq", fontName="Vera", fontSize=10.6, leading=16, alignment=TA_CENTER, textColor=colors.HexColor("#142E40")))
    styles.add(ParagraphStyle(name="DocTitle", fontName="Vera-Bold", fontSize=22, leading=27, alignment=TA_LEFT, spaceAfter=12, textColor=colors.HexColor("#163B56")))
    styles.add(ParagraphStyle(name="ArticleTitle", fontName="Vera-Italic", fontSize=11.4, leading=16, alignment=TA_LEFT, spaceAfter=17, textColor=colors.HexColor("#465A68")))

    story = []
    blocks = re.split(r"\n\s*\n", markdown.strip())
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        if block.startswith("$$") and block.endswith("$$"):
            latex = block[2:-2].strip()
            if latex not in EQUATION_BLOCKS:
                raise ValueError(f"No PDF equation rendering defined for: {latex}")
            p = Paragraph(EQUATION_BLOCKS[latex], styles["Eq"])
            box = Table([[p]], colWidths=[165 * mm], hAlign="CENTER")
            box.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F5F8FA")),
                ("BOX", (0, 0), (-1, -1), 0.4, colors.HexColor("#DCE5EA")),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]))
            story.extend([Spacer(1, 2), box, Spacer(1, 5)])
            continue
        if block.startswith("# "):
            heading = block[2:].strip()
            if heading == "Supplementary Methods":
                story.append(Paragraph(inline_markup(heading), styles["DocTitle"]))
            else:
                story.append(Paragraph(inline_markup(heading), styles["Section1"]))
            continue
        if block.startswith("## "):
            if block == "## Transcultural, cross-species, and instrumental comparisons":
                story.append(PageBreak())
            story.append(Paragraph(inline_markup(block[3:].strip()), styles["Section2"]))
            continue
        if block.startswith("### "):
            story.append(Paragraph(inline_markup(block[4:].strip()), styles["Section3"]))
            continue
        if block.startswith("#### "):
            story.append(Paragraph(inline_markup(block[5:].strip()), styles["Section3"]))
            continue
        if block.startswith("**Article title:**"):
            title = block.split(":**", 1)[-1].strip()
            story.append(Paragraph(inline_markup(title), styles["ArticleTitle"]))
            continue
        story.append(Paragraph(inline_markup(block.replace("\n", " ")), styles["BodyClean"]))

    frame = Frame(22 * mm, 16 * mm, A4[0] - 44 * mm, A4[1] - 32 * mm, id="normal", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    doc = BaseDocTemplate(str(PDF_PATH), pagesize=A4, leftMargin=22 * mm, rightMargin=22 * mm, topMargin=16 * mm, bottomMargin=16 * mm, title="Supplementary Methods - " + ARTICLE_TITLE, author="Research team")
    doc.addPageTemplates([PageTemplate(id="all", frames=[frame], onPage=draw_page)])
    doc.build(story, canvasmaker=deterministic_canvasmaker)


def main() -> None:
    markdown = MD_PATH.read_text(encoding="utf-8")
    build_pdf(markdown)
    print(f"Wrote PDF: {PDF_PATH}")


if __name__ == "__main__":
    main()
