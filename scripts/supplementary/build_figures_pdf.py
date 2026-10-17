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
    Image,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
)


ROOT = Path(__file__).resolve().parents[2]
MD_INPUT = ROOT / "supplementary" / "figures" / "Supplementary_Figures.md"
PDF_OUTPUT = ROOT / "supplementary" / "figures" / "Supplementary_Figures.pdf"
ARTICLE_TITLE = (
    "Affective Information Transmission in Cross-Species Acoustic Communication: "
    "Evidence for Multiscale Acoustic–Affective Organization"
)


def figure_sources():
    if not MD_INPUT.exists():
        raise FileNotFoundError(f"Missing repository Markdown source: {MD_INPUT}")
    raw = MD_INPUT.read_text(encoding="utf-8")
    pattern = re.compile(
        r"(?ms)^## Supplementary Figure S(\d+)\s*\n+"
        r"!\[Supplementary Figure S\1\]\(([^)]+)\)\s*\n+"
        r"(.+?)(?=\n\s*## Supplementary Figure S\d+|\Z)"
    )
    items = []
    for match in pattern.finditer(raw):
        number = int(match.group(1))
        image = (MD_INPUT.parent / match.group(2)).resolve()
        if not image.is_relative_to(ROOT.resolve()):
            raise ValueError(f"Figure S{number} image path escapes the repository: {image}")
        if not image.is_file():
            raise FileNotFoundError(f"Missing packaged image for Figure S{number}: {image}")
        caption = next((part.strip() for part in re.split(r"\n\s*\n", match.group(3).strip()) if part.strip()), None)
        if caption is None or not caption.startswith(f"Supplementary Figure S{number}."):
            raise ValueError(f"No reader-facing caption found for Figure S{number} in {MD_INPUT}")
        items.append((number, image, caption))
    if [number for number, _, _ in items] != list(range(1, 6)):
        raise ValueError("Expected exactly Supplementary Figures S1 through S5, in order, in the provided Markdown")
    return raw, items


def inline_markup(text):
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#D9E0E6"))
    canvas.setLineWidth(0.5)
    canvas.line(18 * mm, 16 * mm, A4[0] - 18 * mm, 16 * mm)
    canvas.setFont("Vera", 8)
    canvas.setFillColor(colors.HexColor("#59636E"))
    canvas.drawString(18 * mm, 10 * mm, "Supplementary Figures")
    canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, str(doc.page))
    canvas.restoreState()


def deterministic_canvasmaker(*args, **kwargs):
    kwargs["invariant"] = 1
    return Canvas(*args, **kwargs)


def build_pdf(items):
    fonts = Path(reportlab.__file__).resolve().parent / "fonts"
    pdfmetrics.registerFont(TTFont("Vera", str(fonts / "Vera.ttf")))
    pdfmetrics.registerFont(TTFont("Vera-Bold", str(fonts / "VeraBd.ttf")))
    pdfmetrics.registerFont(TTFont("Vera-Italic", str(fonts / "VeraIt.ttf")))
    pdfmetrics.registerFontFamily("Vera", normal="Vera", bold="Vera-Bold", italic="Vera-Italic", boldItalic="Vera-Bold")

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitleCustom", fontName="Vera-Bold", fontSize=18, leading=22, spaceAfter=5, textColor=colors.HexColor("#163B56")))
    styles.add(ParagraphStyle(name="ArticleCustom", fontName="Vera-Italic", fontSize=9.3, leading=12, spaceAfter=10, textColor=colors.HexColor("#465A68")))
    styles.add(ParagraphStyle(name="FigureHeading", fontName="Vera-Bold", fontSize=11.5, leading=14, spaceBefore=3, spaceAfter=5, keepWithNext=True, textColor=colors.HexColor("#245B78")))
    styles.add(ParagraphStyle(name="Caption", fontName="Vera", fontSize=8.5, leading=11, alignment=TA_JUSTIFY, spaceBefore=5, textColor=colors.HexColor("#202830")))
    styles.add(ParagraphStyle(name="Intro", fontName="Vera", fontSize=9, leading=12, spaceAfter=8, textColor=colors.HexColor("#202830")))

    story = [
        Paragraph("Supplementary Figures", styles["TitleCustom"]),
        Paragraph(inline_markup(ARTICLE_TITLE), styles["ArticleCustom"]),
        Paragraph("The figures and captions below are provided as a separate supplementary deliverable.", styles["Intro"]),
    ]
    for idx, (n, image_path, caption) in enumerate(items):
        if idx:
            story.append(PageBreak())
        story.append(Paragraph(f"Supplementary Figure S{n}", styles["FigureHeading"]))
        figure = Image(str(image_path))
        figure._restrictSize(168 * mm, 144 * mm)
        figure.hAlign = "CENTER"
        story.extend([figure, Spacer(1, 2 * mm), Paragraph(inline_markup(caption), styles["Caption"])])

    frame = Frame(18 * mm, 21 * mm, A4[0] - 36 * mm, A4[1] - 39 * mm, id="figures", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    doc = BaseDocTemplate(str(PDF_OUTPUT), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=17 * mm, bottomMargin=21 * mm, title="Supplementary Figures - " + ARTICLE_TITLE)
    doc.addPageTemplates([PageTemplate(id="all", frames=[frame], onPage=footer)])
    doc.build(story, canvasmaker=deterministic_canvasmaker)


def main():
    _, items = figure_sources()
    build_pdf(items)
    print(f"Created {PDF_OUTPUT}")


if __name__ == "__main__":
    main()
