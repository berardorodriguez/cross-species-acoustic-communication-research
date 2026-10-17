from pathlib import Path
import re

import reportlab
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    PageBreak,
    PageTemplate,
    Paragraph,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[2]
MD_INPUT = ROOT / "supplementary" / "tables" / "Supplementary_Tables.md"
PDF_OUTPUT = ROOT / "supplementary" / "tables" / "Supplementary_Tables.pdf"
ARTICLE_TITLE = (
    "Affective Information Transmission in Cross-Species Acoustic Communication: "
    "Evidence for Multiscale Acoustic–Affective Organization"
)


def split_cells(line):
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def parse_markdown():
    if not MD_INPUT.is_file():
        raise FileNotFoundError(f"Missing repository Markdown source: {MD_INPUT}")
    text = MD_INPUT.read_text(encoding="utf-8")
    sections = list(re.finditer(r"(?m)^## Supplementary Table S([12])\s*$", text))
    if [int(match.group(1)) for match in sections] != [1, 2]:
        raise ValueError("Expected Supplementary Tables S1 and S2, in order, in the provided Markdown")

    tables = {}
    for index, match in enumerate(sections):
        number = int(match.group(1))
        end = sections[index + 1].start() if index + 1 < len(sections) else len(text)
        body = text[match.end():end]
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", body.strip()) if part.strip()]
        title = next((part.strip("*").strip() for part in paragraphs if part.startswith("**")), None)
        if title is None:
            raise ValueError(f"Missing title for Supplementary Table S{number}")
        note_block = next((part for part in paragraphs if part.startswith("**Note.**")), None)
        if note_block is None:
            raise ValueError(f"Missing note for Supplementary Table S{number}")
        note = note_block.replace("**Note.**", "", 1).strip()
        table_lines = [line for line in body.splitlines() if line.strip().startswith("|")]
        rows = [split_cells(line) for line in table_lines if not re.fullmatch(r"\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?", line.strip())]
        if len(rows) < 2 or any(len(row) != len(rows[0]) for row in rows):
            raise ValueError(f"Malformed Markdown table for Supplementary Table S{number}")
        tables[number] = {"title": title, "note": note, "header": rows[0], "rows": rows[1:]}

    if len(tables[1]["rows"]) != 36 or len(tables[2]["rows"]) != 9:
        raise ValueError("Expected 36 rows in Table S1 and 9 rows in Table S2")
    if len(tables[1]["header"]) != 3 or len(tables[2]["header"]) != 4:
        raise ValueError("Unexpected column count in the reader-facing tables")
    return tables


def escape(text):
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def footer(canvas, doc):
    canvas.saveState()
    width, _ = landscape(A4)
    canvas.setStrokeColor(colors.HexColor("#D9E0E6"))
    canvas.setLineWidth(0.5)
    canvas.line(18 * mm, 14 * mm, width - 18 * mm, 14 * mm)
    canvas.setFont("Vera", 8)
    canvas.setFillColor(colors.HexColor("#59636E"))
    canvas.drawString(18 * mm, 9 * mm, "Supplementary Tables")
    canvas.drawRightString(width - 18 * mm, 9 * mm, str(doc.page))
    canvas.restoreState()


def deterministic_canvasmaker(*args, **kwargs):
    kwargs["invariant"] = 1
    return Canvas(*args, **kwargs)


def table_style():
    return TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "Vera"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.8),
        ("LEADING", (0, 0), (-1, -1), 10),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#202830")),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CFD8DE")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F8FA")]),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E6EEF2")),
        ("FONTNAME", (0, 0), (-1, 0), "Vera-Bold"),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#163B56")),
    ])


def render_table(header, rows, widths, styles):
    data = [[Paragraph(escape(cell), styles["TableHeaderCell"]) for cell in header]]
    data.extend([[Paragraph(escape(cell), styles["TableCell"]) for cell in row] for row in rows])
    result = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    result.setStyle(table_style())
    return result


def build_pdf(tables):
    fonts = Path(reportlab.__file__).resolve().parent / "fonts"
    pdfmetrics.registerFont(TTFont("Vera", str(fonts / "Vera.ttf")))
    pdfmetrics.registerFont(TTFont("Vera-Bold", str(fonts / "VeraBd.ttf")))
    pdfmetrics.registerFont(TTFont("Vera-Italic", str(fonts / "VeraIt.ttf")))
    pdfmetrics.registerFontFamily("Vera", normal="Vera", bold="Vera-Bold", italic="Vera-Italic", boldItalic="Vera-Bold")

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TableDocTitle", fontName="Vera-Bold", fontSize=17, leading=20, spaceAfter=3, textColor=colors.HexColor("#163B56")))
    styles.add(ParagraphStyle(name="TableArticleTitle", fontName="Vera-Italic", fontSize=8.7, leading=11, spaceAfter=7, textColor=colors.HexColor("#465A68")))
    styles.add(ParagraphStyle(name="TableHeading", fontName="Vera-Bold", fontSize=10.5, leading=13, spaceBefore=3, spaceAfter=4, textColor=colors.HexColor("#245B78")))
    styles.add(ParagraphStyle(name="TableTitle", fontName="Vera-Bold", fontSize=8.7, leading=11, spaceAfter=5, textColor=colors.HexColor("#202830")))
    styles.add(ParagraphStyle(name="TableNote", fontName="Vera", fontSize=7.7, leading=10, alignment=TA_JUSTIFY, spaceBefore=5, textColor=colors.HexColor("#303840")))
    styles.add(ParagraphStyle(name="TableCell", fontName="Vera", fontSize=7.8, leading=9.5, textColor=colors.HexColor("#202830")))
    styles.add(ParagraphStyle(name="TableHeaderCell", fontName="Vera-Bold", fontSize=7.7, leading=9.5, textColor=colors.HexColor("#163B56")))

    page_width, page_height = landscape(A4)
    full_width = page_width - 36 * mm
    story = [
        Paragraph("Supplementary Tables", styles["TableDocTitle"]),
        Paragraph(escape(ARTICLE_TITLE), styles["TableArticleTitle"]),
        Paragraph("Supplementary Table S1", styles["TableHeading"]),
        Paragraph(escape(tables[1]["title"]), styles["TableTitle"]),
    ]

    s1 = tables[1]
    s1_widths = [full_width * 0.43, full_width * 0.43, full_width * 0.14]
    story.append(render_table(s1["header"], s1["rows"][:18], s1_widths, styles))
    story.extend([
        PageBreak(),
        Paragraph("Supplementary Table S1 (continued)", styles["TableHeading"]),
        render_table(s1["header"], s1["rows"][18:], s1_widths, styles),
        Paragraph("<b>Note.</b> " + escape(s1["note"]), styles["TableNote"]),
        PageBreak(),
    ])

    s2 = tables[2]
    story.extend([
        Paragraph("Supplementary Table S2", styles["TableHeading"]),
        Paragraph(escape(s2["title"]), styles["TableTitle"]),
    ])
    s2_widths = [full_width * 0.35, full_width * 0.21, full_width * 0.36, full_width * 0.08]
    story.extend([
        render_table(s2["header"], s2["rows"], s2_widths, styles),
        Paragraph("<b>Note.</b> " + escape(s2["note"]), styles["TableNote"]),
    ])

    frame = Frame(18 * mm, 19 * mm, full_width, page_height - 34 * mm, id="tables", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    doc = BaseDocTemplate(str(PDF_OUTPUT), pagesize=landscape(A4), leftMargin=18 * mm, rightMargin=18 * mm, topMargin=15 * mm, bottomMargin=19 * mm, title="Supplementary Tables - " + ARTICLE_TITLE)
    doc.addPageTemplates([PageTemplate(id="all", frames=[frame], onPage=footer)])
    doc.build(story, canvasmaker=deterministic_canvasmaker)


def main():
    tables = parse_markdown()
    build_pdf(tables)
    print(f"Created {PDF_OUTPUT}")
    print(f"Reader-facing table rows: S1={len(tables[1]['rows'])}, S2={len(tables[2]['rows'])}")


if __name__ == "__main__":
    main()
