"""Render this repository's dated Markdown report to a local, offline PDF.

Pandoc supplies a stable Markdown AST; ReportLab handles layout. This avoids a
first-run LaTeX installation and does not load remote CSS, fonts, or scripts.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import subprocess
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Preformatted,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents


ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = ROOT / "SovereignAI_Project_Review_2026-10-01.md"
DEFAULT_OUTPUT = ROOT / "SovereignAI_Project_Review_2026-10-01.pdf"
PAGE_WIDTH, PAGE_HEIGHT = A4
MARGIN_X = 46
MARGIN_TOP = 48
MARGIN_BOTTOM = 46
CONTENT_WIDTH = PAGE_WIDTH - 2 * MARGIN_X


def register_fonts() -> None:
    fonts = Path("C:/Windows/Fonts")
    pdfmetrics.registerFont(TTFont("Arial", str(fonts / "arial.ttf")))
    pdfmetrics.registerFont(TTFont("Arial-Bold", str(fonts / "arialbd.ttf")))
    pdfmetrics.registerFont(TTFont("Arial-Italic", str(fonts / "ariali.ttf")))
    pdfmetrics.registerFont(TTFont("Consolas", str(fonts / "consola.ttf")))
    pdfmetrics.registerFontFamily(
        "Arial", normal="Arial", bold="Arial-Bold", italic="Arial-Italic"
    )


def styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    dark = colors.HexColor("#192330")
    navy = colors.HexColor("#123D59")
    blue = colors.HexColor("#17688B")
    return {
        "normal": ParagraphStyle(
            "ReportBody", parent=base["Normal"], fontName="Arial",
            fontSize=9.2, leading=13.2, textColor=dark, spaceAfter=7,
            allowWidows=0, allowOrphans=0, splitLongWords=1,
        ),
        "title": ParagraphStyle(
            "ReportTitle", parent=base["Title"], fontName="Arial-Bold",
            fontSize=25, leading=30, textColor=navy, alignment=TA_LEFT,
            spaceAfter=14,
        ),
        "subtitle": ParagraphStyle(
            "ReportSubtitle", parent=base["Normal"], fontName="Arial",
            fontSize=12, leading=17, textColor=blue, spaceAfter=22,
        ),
        "meta": ParagraphStyle(
            "ReportMeta", parent=base["Normal"], fontName="Arial",
            fontSize=9.5, leading=15, textColor=colors.HexColor("#586775"),
            spaceAfter=8,
        ),
        "h1": ParagraphStyle(
            "H1", parent=base["Heading1"], fontName="Arial-Bold",
            fontSize=15, leading=19, textColor=navy,
            spaceBefore=15, spaceAfter=8, keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "H2", parent=base["Heading2"], fontName="Arial-Bold",
            fontSize=11.5, leading=15, textColor=navy,
            spaceBefore=11, spaceAfter=6, keepWithNext=True,
        ),
        "h3": ParagraphStyle(
            "H3", parent=base["Heading3"], fontName="Arial-Bold",
            fontSize=10, leading=13, textColor=navy,
            spaceBefore=9, spaceAfter=5, keepWithNext=True,
        ),
        "bullet": ParagraphStyle(
            "ReportBullet", parent=base["Normal"], fontName="Arial",
            fontSize=9.2, leading=13.2, textColor=dark,
            leftIndent=17, firstLineIndent=-13, spaceAfter=5,
            allowWidows=0, allowOrphans=0,
        ),
        "cell": ParagraphStyle(
            "ReportCell", parent=base["Normal"], fontName="Arial",
            fontSize=8.1, leading=11.2, textColor=dark, splitLongWords=1,
        ),
        "cellhead": ParagraphStyle(
            "ReportCellHead", parent=base["Normal"], fontName="Arial-Bold",
            fontSize=8.1, leading=11.2, textColor=navy, splitLongWords=1,
        ),
        "equation": ParagraphStyle(
            "ReportEquation", parent=base["Normal"], fontName="Arial",
            fontSize=9.4, leading=14, textColor=navy,
            alignment=TA_CENTER, spaceBefore=5, spaceAfter=8,
        ),
        "code": ParagraphStyle(
            "ReportCode", parent=base["Code"], fontName="Consolas",
            fontSize=8.1, leading=11.5, textColor=dark,
            leftIndent=10, rightIndent=10,
        ),
        "toc0": ParagraphStyle(
            "TOC0", parent=base["Normal"], fontName="Arial-Bold",
            fontSize=10.2, leading=15, leftIndent=0,
            firstLineIndent=0, spaceAfter=3, textColor=navy,
        ),
        "toc1": ParagraphStyle(
            "TOC1", parent=base["Normal"], fontName="Arial",
            fontSize=9.1, leading=13, leftIndent=15,
            firstLineIndent=0, spaceAfter=2, textColor=dark,
        ),
    }


def readable_math(raw: str) -> str:
    value = raw.strip()
    value = re.sub(r"\\(?:mathrm|operatorname|text)\{([^{}]*)\}", r"\1", value)
    value = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", value)
    for before, after in {
        r"\Delta": "Δ", r"\delta": "δ", r"\times": "×",
        r"\approx": "≈", r"\in": "∈", r"\%": "%",
        r"\#": "#", r"\ ": " ", r"\qquad": "    ",
    }.items():
        value = value.replace(before, after)
    value = value.replace("\\left", "").replace("\\right", "")
    value = value.replace("\\!", "").replace("\\", "")
    return value


def display_math(raw: str) -> str:
    if "operatorname{RRF}" in raw:
        return "RRF(d) = Σ over retrieved methods m of 1 / (k + rank_m(d))"
    if r"\Delta=x-L" in raw:
        return "Δ = x − L = 1.1 mm/s RMS<br/>δ = ((x − L) / L) × 100% ≈ 15.5%"
    if raw.lstrip().startswith("P="):
        return (
            "Precision = TP / (TP + FP) &nbsp;&nbsp; "
            "Recall = TP / (TP + FN)<br/>"
            "F1 = 2 × Precision × Recall / (Precision + Recall)"
        )
    if "UnsafeReleaseRate" in raw:
        return "Unsafe release rate = released gold-hold cases / all gold-hold cases"
    if r"H_{\mathrm{root}}" in raw:
        return "Hroot = SHA-256(canonical JSON(sorted path / file-digest entries))"
    return html.escape(readable_math(raw)).replace("\n", "<br/>")


def inline_html(items: list[dict]) -> str:
    parts: list[str] = []
    for item in items:
        kind = item["t"]
        value = item.get("c")
        if kind == "Str":
            parts.append(html.escape(value))
        elif kind in {"Space", "SoftBreak"}:
            parts.append(" ")
        elif kind == "LineBreak":
            parts.append("<br/>")
        elif kind in {"Strong", "Emph", "SmallCaps", "Strikeout"}:
            tag = "b" if kind in {"Strong", "SmallCaps"} else "i"
            parts.append(f"<{tag}>{inline_html(value)}</{tag}>")
        elif kind == "Code":
            parts.append(f'<font face="Consolas">{html.escape(value[1])}</font>')
        elif kind == "Link":
            target = html.escape(value[2][0], quote=True)
            parts.append(f'<link href="{target}">{inline_html(value[1])}</link>')
        elif kind == "Math":
            parts.append(html.escape(readable_math(value[1])))
        elif kind == "Quoted":
            parts.append(f"“{inline_html(value[1])}”")
        elif kind in {"Span", "Cite"}:
            parts.append(inline_html(value[-1]))
        elif kind in {"Superscript", "Subscript"}:
            tag = "super" if kind == "Superscript" else "sub"
            parts.append(f"<{tag}>{inline_html(value)}</{tag}>")
        else:
            parts.append(html.escape(str(value or "")))
    return "".join(parts)


def block_to_text(block: dict) -> str:
    kind = block["t"]
    if kind in {"Plain", "Para"}:
        return inline_html(block["c"])
    return ""


def make_table(block: dict, style: dict[str, ParagraphStyle]) -> Table:
    content = block["c"]
    head_rows = content[3][1]
    body_rows = [row for body in content[4] for row in body[3]]
    rows = head_rows + body_rows
    data = []
    for row_number, row in enumerate(rows):
        cells = []
        for cell in row[1]:
            paragraphs = [block_to_text(part) for part in cell[4]]
            body = "<br/>".join(part for part in paragraphs if part)
            cells.append(
                Paragraph(
                    body or " ",
                    style["cellhead"] if row_number < len(head_rows) else style["cell"],
                )
            )
        data.append(cells)
    ncols = len(data[0])
    if ncols == 2 and "ID" in block_to_text(head_rows[0][1][0][4][0]):
        widths = [39, CONTENT_WIDTH - 39]
    elif ncols == 3:
        widths = [95, 190, CONTENT_WIDTH - 285]
    else:
        widths = [CONTENT_WIDTH / ncols] * ncols
    table = Table(data, colWidths=widths, repeatRows=len(head_rows), hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, len(head_rows) - 1), colors.HexColor("#E7F0F5")),
        ("ROWBACKGROUNDS", (0, len(head_rows)), (-1, -1), [
            colors.white, colors.HexColor("#F8FAFB")
        ]),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D2DFE6")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return table


class ReportDocTemplate(BaseDocTemplate):
    def __init__(self, output: Path, title: str) -> None:
        super().__init__(
            str(output), pagesize=A4, leftMargin=MARGIN_X,
            rightMargin=MARGIN_X, topMargin=MARGIN_TOP,
            bottomMargin=MARGIN_BOTTOM, title=title,
            author="Repository-evidence snapshot",
        )
        frame = Frame(
            MARGIN_X, MARGIN_BOTTOM,
            CONTENT_WIDTH, PAGE_HEIGHT - MARGIN_TOP - MARGIN_BOTTOM,
            leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0,
        )
        self.addPageTemplates(PageTemplate(
            id="report", frames=[frame], onPage=self.draw_page
        ))

    def draw_page(self, canvas, doc) -> None:
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#CADCE5"))
        canvas.line(MARGIN_X, PAGE_HEIGHT - 33, PAGE_WIDTH - MARGIN_X, PAGE_HEIGHT - 33)
        canvas.setFont("Arial", 7.5)
        canvas.setFillColor(colors.HexColor("#62717C"))
        canvas.drawString(MARGIN_X, 25, "SovereignAI | Engineering snapshot | 1 October 2026")
        canvas.drawRightString(PAGE_WIDTH - MARGIN_X, 25, f"Page {doc.page}")
        canvas.restoreState()

    def afterFlowable(self, flowable) -> None:
        if isinstance(flowable, Paragraph) and flowable.style.name in {"H1", "H2"}:
            level = 0 if flowable.style.name == "H1" else 1
            text = re.sub(r"<[^>]+>", "", flowable.getPlainText())
            if text == "Contents":
                return
            self.notify("TOCEntry", (level, text, self.page))


def render(source: Path, output: Path) -> None:
    register_fonts()
    result = subprocess.run(
        ["pandoc", str(source), "--to=json"],
        check=True, capture_output=True, text=True, encoding="utf-8"
    )
    document = json.loads(result.stdout)
    style = styles()
    story = [
        Spacer(1, 55),
        Paragraph("SovereignAI", style["title"]),
        Paragraph("Project Review and Production Roadmap", style["subtitle"]),
        Paragraph(
            "What is implemented, how it works, what has been measured, "
            "and what remains", style["meta"]
        ),
        Paragraph("Repository-evidence snapshot - 1 October 2026", style["meta"]),
        Spacer(1, 20),
        Paragraph(
            "Advisory industrial AI prototype; not a production or safety certification.",
            style["normal"],
        ),
        PageBreak(),
        Paragraph("Contents", style["h1"]),
    ]
    toc = TableOfContents()
    toc.levelStyles = [style["toc0"], style["toc1"]]
    story.extend([toc, PageBreak()])

    for block in document["blocks"]:
        kind = block["t"]
        content = block["c"]
        if kind == "Header":
            level = content[0]
            key = "h1" if level == 1 else "h2" if level == 2 else "h3"
            story.append(Paragraph(inline_html(content[2]), style[key]))
        elif kind == "Para":
            if len(content) == 1 and content[0]["t"] == "Math":
                math_type, raw = content[0]["c"]
                if math_type["t"] == "DisplayMath":
                    story.append(Paragraph(display_math(raw), style["equation"]))
                    continue
            story.append(Paragraph(inline_html(content), style["normal"]))
        elif kind in {"BulletList", "OrderedList"}:
            items = content if kind == "BulletList" else content[1]
            for index, item in enumerate(items, start=1):
                chunks = [block_to_text(part) for part in item]
                label = "•" if kind == "BulletList" else f"{index}."
                body = " ".join(chunk for chunk in chunks if chunk)
                story.append(Paragraph(
                    f"{html.escape(label)}  {body}", style["bullet"]
                ))
        elif kind == "CodeBlock":
            story.append(KeepTogether([
                Spacer(1, 6),
                Preformatted(content[1], style["code"], maxLineLength=94),
                Spacer(1, 6),
            ]))
        elif kind == "Table":
            story.extend([Spacer(1, 6), make_table(block, style), Spacer(1, 8)])
        else:
            raise ValueError(f"Unsupported Markdown block in report: {kind}")

    output.parent.mkdir(parents=True, exist_ok=True)
    ReportDocTemplate(output, "SovereignAI Project Review").multiBuild(story)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    render(args.source.resolve(), args.output.resolve())
