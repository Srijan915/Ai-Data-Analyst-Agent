"""Build tiny, deterministic PDF fixtures with reportlab for local tests."""
from pathlib import Path

from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.platypus import Table, TableStyle

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures"


def build_table_report():
    document = canvas.Canvas(str(FIXTURES / "table_report.pdf"), pagesize=(400, 500))
    document.drawString(40, 450, "Cover page summary")
    document.showPage()
    document.drawString(40, 450, "Purchase table")
    table = Table([["ID", "Amount"], ["0012", "10"], ["0013", "20"]],
                  colWidths=[110, 110], rowHeights=[24, 24, 24])
    table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 1, colors.black)]))
    table.wrapOn(document, 220, 72)
    table.drawOn(document, 40, 320)
    document.save()


def build_text_report():
    document = canvas.Canvas(str(FIXTURES / "text_report.pdf"), pagesize=(400, 500))
    document.drawString(40, 450, "Quarterly narrative without a table")
    document.save()


def build_no_text_report():
    document = canvas.Canvas(str(FIXTURES / "no_text_report.pdf"), pagesize=(400, 500))
    document.rect(40, 350, 180, 90, stroke=1, fill=0)
    document.save()


if __name__ == "__main__":
    build_table_report()
    build_text_report()
    build_no_text_report()
    (FIXTURES / "corrupt_report.pdf").write_bytes(b"%PDF-1.7\nnot a valid PDF document\n")
