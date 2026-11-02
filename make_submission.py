"""Build the report, metric audit/guide PDFs, and a ten-slide presentation."""
from __future__ import annotations

import html
import json
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image,
    KeepTogether,
)
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "submission"
CRIMSON = colors.HexColor("#8C1515")
INK = colors.HexColor("#192331")
MUTED = colors.HexColor("#5E6A75")
LIGHT = colors.HexColor("#EDF1F4")


def register_font():
    choices = [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for path in choices:
        if path.exists():
            pdfmetrics.registerFont(TTFont("ProjectSans", str(path)))
            return "ProjectSans"
    return "Helvetica"


def inline_markup(text: str) -> str:
    """Handle the small subset of inline Markdown used in REPORT.md."""
    placeholders = []

    def save(value):
        placeholders.append(value)
        return f"ZZMARK{len(placeholders)-1}ZZ"

    text = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)",
                  lambda m: save(f'<link href="{html.escape(m.group(2), quote=True)}" color="#8C1515">{html.escape(m.group(1))}</link>'),
                  text)
    text = re.sub(r"`([^`]+)`", lambda m: save(f'<font color="#8C1515">{html.escape(m.group(1))}</font>'), text)
    text = html.escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    for index, value in enumerate(placeholders):
        text = text.replace(f"ZZMARK{index}ZZ", value)
    return text


def make_pdf(source_name="REPORT.md", output_name="Protein_Interface_GNN_Report.pdf"):
    font = register_font()
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ProjectTitle", fontName=font, fontSize=21, leading=27,
                              textColor=INK, spaceAfter=13))
    styles.add(ParagraphStyle(name="ProjectH2", fontName=font, fontSize=13, leading=17,
                              textColor=CRIMSON, spaceBefore=16, spaceAfter=7, keepWithNext=True))
    styles.add(ParagraphStyle(name="ProjectH3", fontName=font, fontSize=10.5, leading=14,
                              textColor=INK, spaceBefore=11, spaceAfter=5, keepWithNext=True))
    styles.add(ParagraphStyle(name="ProjectBody", fontName=font, fontSize=9.2, leading=14,
                              textColor=INK, spaceAfter=7))
    styles.add(ParagraphStyle(name="ProjectBullet", parent=styles["ProjectBody"],
                              leftIndent=16, firstLineIndent=-10, spaceAfter=4))
    styles.add(ParagraphStyle(name="ProjectSmall", fontName=font, fontSize=7.6, leading=10.5,
                              textColor=INK))
    styles.add(ParagraphStyle(name="ProjectCaption", fontName=font, fontSize=8, leading=11,
                              textColor=MUTED, alignment=TA_CENTER, spaceAfter=9))
    if source_name == "METRIC_AUDIT.md":
        styles["ProjectBody"].leading = 13.4
        styles["ProjectBody"].spaceAfter = 6
        styles["ProjectBullet"].leading = 13.4
    width = A4[0] - 4*cm
    content = (ROOT / source_name).read_text(encoding="utf-8").splitlines()
    flow = []
    i = 0
    while i < len(content):
        line = content[i].strip()
        if not line:
            i += 1
            continue
        if line.startswith("!["):
            match = re.match(r"!\[([^\]]+)\]\(([^)]+)\)", line)
            if match:
                image_path = ROOT / match.group(2)
                if image_path.exists():
                    graphic = Image(str(image_path))
                    flow.append(Image(str(image_path), width=width,
                                      height=width*graphic.imageHeight/graphic.imageWidth))
                    flow.append(Paragraph(html.escape(match.group(1)), styles["ProjectCaption"]))
            i += 1
            continue
        if line.startswith("| "):
            rows = []
            while i < len(content) and content[i].strip().startswith("|"):
                raw = [s.strip() for s in content[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-+:?", cell) for cell in raw):
                    rows.append([Paragraph(inline_markup(cell), styles["ProjectSmall"]) for cell in raw])
                i += 1
            if rows:
                widths = [width/len(rows[0])] * len(rows[0])
                table = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
                table.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), LIGHT),
                    ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#CED5DB")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 5),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]))
                flow.extend((table, Spacer(1, 8)))
            continue
        if line.startswith("# "):
            flow.append(Paragraph(inline_markup(line[2:]), styles["ProjectTitle"]))
        elif line.startswith("## "):
            flow.append(Paragraph(inline_markup(line[3:]), styles["ProjectH2"]))
        elif line.startswith("### "):
            flow.append(Paragraph(inline_markup(line[4:]), styles["ProjectH3"]))
        elif line.startswith(("- ", "> ")) or re.match(r"\d+\. ", line):
            while i+1 < len(content) and content[i+1].strip() and not re.match(
                r"^(#|\||!\[|- |\d+\. |> )", content[i+1].strip()
            ):
                i += 1
                line += " " + content[i].strip()
            label = re.sub(r"^(?:- |> |\d+\. )", "", line)
            prefix = "• " if line.startswith("- ") else ""
            if re.match(r"\d+\. ", line):
                prefix = line.split(".", 1)[0] + ". "
            flow.append(Paragraph(prefix + inline_markup(label), styles["ProjectBullet"]))
        else:
            paragraph = [line]
            while i+1 < len(content) and content[i+1].strip() and not re.match(r"^(#|\||!\[|- |\d+\. |> )", content[i+1].strip()):
                i += 1
                paragraph.append(content[i].strip())
            flow.append(Paragraph(inline_markup(" ".join(paragraph)), styles["ProjectBody"]))
        i += 1

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(CRIMSON)
        canvas.setLineWidth(1.1)
        canvas.line(2*cm, A4[1]-1.5*cm, A4[0]-2*cm, A4[1]-1.5*cm)
        canvas.setFont(font, 8)
        canvas.setFillColor(MUTED)
        canvas.drawString(2*cm, 1.45*cm, "Protein Interface Prediction · Solo GNN Project")
        canvas.drawRightString(A4[0]-2*cm, 1.45*cm, str(doc.page))
        canvas.restoreState()

    doc = SimpleDocTemplate(str(OUT / output_name), pagesize=A4,
                            leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=2.1*cm, bottomMargin=2.1*cm,
                            title=content[0].lstrip("# "),
                            author="Solo GNN course project")
    doc.build(flow, onFirstPage=footer, onLaterPages=footer)


def add_text(slide, x, y, w, h, text, size=22, color=INK, bold=False, align=PP_ALIGN.LEFT):
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = shape.text_frame
    tf.word_wrap = True
    tf.clear()
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.name = "Aptos"
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = RGBColor(int(color.red*255), int(color.green*255), int(color.blue*255))
    return shape


def add_slide(prs, title, number):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    bg = slide.background.fill
    bg.solid()
    bg.fore_color.rgb = RGBColor(255, 255, 255)
    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(0.11))
    line.fill.solid()
    line.fill.fore_color.rgb = RGBColor(140, 21, 21)
    line.line.fill.background()
    add_text(slide, 0.7, 0.45, 12, 0.6, title, 30, INK, True)
    add_text(slide, 0.7, 7.05, 10, 0.3, "Protein Interface Prediction · Solo GNN project", 10, MUTED)
    add_text(slide, 12.15, 7.05, 0.5, 0.3, str(number), 10, MUTED)
    return slide


def bullets(slide, items, x=0.9, y=1.45, w=11.4, spacing=0.82, size=21):
    for index, item in enumerate(items):
        add_text(slide, x, y+index*spacing, w, spacing-0.08, "•  " + item, size)


def make_slides():
    metrics = json.loads((ROOT / "paper_benchmark" / "results" / "benchmark_metrics.json").read_text())
    analysis = json.loads((ROOT / "paper_benchmark" / "results" / "analysis.json").read_text())
    audit = json.loads((ROOT / "paper_benchmark" / "results" / "metric_audit.json").read_text())
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    slide = add_slide(prs, "Protein Interface Prediction with GNNs", 1)
    add_text(slide, 0.85, 1.7, 11.7, 1.2,
             "Does message passing improve contact prediction between two proteins?", 27, CRIMSON, True)
    add_text(slide, 0.9, 4.1, 11.5, 0.7,
             "Paper-guided PyTorch reimplementation of Fout et al. (NeurIPS 2017)", 19, INK)
    add_text(slide, 0.9, 4.85, 11.5, 0.6,
             "Solo course project · Official Docking Benchmark 5 features and labels", 16, MUTED)

    slide = add_slide(prs, "1 · The prediction question", 2)
    bullets(slide, ["A protein interface is the region where two proteins touch.",
                    "Score a pair: one amino-acid residue from each protein.",
                    "Positive label: non-hydrogen atoms within 6 Å in the bound complex.",
                    "The bound cross-protein distance defines the answer, not an input."], size=20)

    slide = add_slide(prs, "2 · Research data and leakage control", 3)
    bullets(slide, ["Use the authors' official Docking Benchmark 5 data release.",
                    "Input: 70 residue features from each unbound protein partner.",
                    "Targets: pair contacts measured in the known bound complex.",
                    "175 development complexes and 55 separate official test complexes.",
                    "Downloaded files are checked against published MD5 hashes."], size=18)

    slide = add_slide(prs, "3 · Node-average graph convolution", 4)
    add_text(slide, 0.95, 1.55, 11.6, 0.8,
             "hᵢ(next) = ReLU(Wc hᵢ + Wn mean of 20 neighbors)", 27, CRIMSON, True)
    bullets(slide, ["Each residue is a graph node with 20 spatial nearest neighbors.",
                    "Center and neighbor features have separate learned weights.",
                    "Two shared 64-unit layers encode each protein independently.",
                    "This is a small version of Fout et al.'s Equation (1)."], y=2.65, size=19)

    slide = add_slide(prs, "4 · Pair scoring and matched baseline", 5)
    bullets(slide, ["One shared encoder produces a vector for each residue on both proteins.",
                    "Pair head uses sum, absolute difference, and elementwise product.",
                    "These features make the score invariant to swapping partners.",
                    "Baseline: same node inputs and pair head, with no neighbor aggregation."], size=19)

    slide = add_slide(prs, "5 · Experimental protocol", 6)
    bullets(slide, ["Fixed 140/35 development split; choose epoch by validation median AUC.",
                    "Retrain each selected model on all 175 development complexes.",
                    "Evaluate every released pair label in the 55 official test complexes.",
                    "Test: 4,958,317 labeled pairs, but only 4,871 contacts (0.0982%).",
                    "Primary metric: median ROC AUC across complexes; also report AP."], size=18)

    slide = add_slide(prs, "6 · Overall ranking and early retrieval", 7)
    chart = ROOT / "paper_benchmark" / "results" / "audited_metric_comparison.png"
    slide.shapes.add_picture(str(chart), Inches(0.55), Inches(1.5), width=Inches(8.5))
    gcn = metrics["models"]["Node-average GCN"]["test_summary"]
    mlp = metrics["models"]["No-convolution MLP"]["test_summary"]
    add_text(slide, 9.2, 1.85, 3.3, 0.7,
             f"GCN AUC  {gcn['median_complex_roc_auc']:.3f}", 21, CRIMSON, True)
    add_text(slide, 9.2, 2.6, 3.3, 0.7,
             f"MLP AUC  {mlp['median_complex_roc_auc']:.3f}", 21, INK)
    add_text(slide, 9.2, 3.65, 3.3, 1.2,
             f"Higher GCN AUC on {analysis['all_test']['gcn_better_auc_count']}/55 complexes.", 19, INK)
    p10_mlp = audit["models"]["mlp"]["macro"]["precision_at_10"]["mean"]
    p10_gcn = audit["models"]["gcn"]["macro"]["precision_at_10"]["mean"]
    add_text(slide, 9.2, 5.0, 3.35, 1.1,
             f"Top-10 precision: GCN {p10_gcn:.2%}; MLP {p10_mlp:.2%}.", 16, MUTED)

    slide = add_slide(prs, "7 · Why these metrics are different", 8)
    bullets(slide, ["Median complex ROC AUC: overall ordering; gives each complex one vote.",
                    "AP: contact concentration through the ranking; differs from trapezoidal PR area.",
                    "Precision/recall at K: usefulness under a limited inspection budget.",
                    "GCN retrieves 27/550 top-10 contacts; MLP retrieves 28/550.",
                    "Report the measures where GCN loses as well as where it improves."], size=18)

    slide = add_slide(prs, "8 · Raw score 0.5 is a diagnostic", 9)
    diagnostic = audit["models"]["gcn"]["pooled"]["fixed_0_5_threshold_diagnostic"]
    baseline_accuracy = audit["baselines"]["always_negative_accuracy"]
    bullets(slide, [f"GCN: {diagnostic['tp']:,} true positives and {diagnostic['fp']:,} false positives.",
                    f"Precision {diagnostic['precision']:.3%}; recall {diagnostic['recall']:.2%}.",
                    f"F1 {diagnostic['f1']:.5f}; MCC {diagnostic['mcc']:.5f}; threshold not validated.",
                    "Negative sampling and class weighting make raw sigmoid scores uncalibrated.",
                    f"Always-negative accuracy is {baseline_accuracy:.4%}, despite finding no contact."], size=18)

    slide = add_slide(prs, "9 · Audit evidence and remaining limits", 10)
    bullets(slide, ["Saved-checkpoint AUC and AP reproduced exactly; independent formulas agree.",
                    "Tie handling, labels, pair indices, checksums and split IDs were checked.",
                    "One training seed; different parameter counts; published study used ten runs.",
                    "24 test cases were used earlier; the historical 31-case AUC interval includes zero.",
                    "Next: independent validation/calibration and a fresh external test set."], size=18)
    add_text(slide, 0.95, 5.55, 11.5, 0.6,
             "Conclusion: stronger overall ranking; top-10 retrieval does not improve.", 18, CRIMSON, True)

    prs.save(OUT / "Protein_Interface_GNN_Presentation.pptx")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    make_pdf()
    make_pdf("METRIC_AUDIT.md", "Metric_Audit.pdf")
    make_pdf("METRICS_GUIDE.md", "Metrics_Explanation_Guide.pdf")
    make_slides()
    print("Created", OUT / "Protein_Interface_GNN_Report.pdf")
    print("Created", OUT / "Protein_Interface_GNN_Presentation.pptx")
    print("Created", OUT / "Metric_Audit.pdf")
    print("Created", OUT / "Metrics_Explanation_Guide.pdf")
