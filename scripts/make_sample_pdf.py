"""Render data/sample_resume.md into data/sample_resume.pdf for testing PDF parsing.

Requires reportlab (pip install reportlab), which is only needed for this script.
"""

from __future__ import annotations

import re
from pathlib import Path

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "sample_resume.md"
TARGET = ROOT / "data" / "sample_resume.pdf"


def inline(text: str) -> str:
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)


def build() -> None:
    styles = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=styles["BodyText"], fontSize=10, leading=13)
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontSize=18, spaceAfter=4)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=12, spaceBefore=10, spaceAfter=4)

    story, bullets = [], []

    def flush() -> None:
        if bullets:
            story.append(ListFlowable([ListItem(Paragraph(b, body), leftIndent=12) for b in bullets], bulletType="bullet", start="•"))
            bullets.clear()

    for line in SOURCE.read_text(encoding="utf-8").splitlines():
        if line.startswith("- "):
            bullets.append(inline(line[2:]))
            continue
        flush()
        if line.startswith("# "):
            story.append(Paragraph(inline(line[2:]), h1))
        elif line.startswith("## "):
            story.append(Paragraph(inline(line[3:]).upper(), h2))
        elif line.strip():
            story.append(Paragraph(inline(line), body))
        else:
            story.append(Spacer(1, 4))
    flush()

    doc = SimpleDocTemplate(str(TARGET), pagesize=LETTER, leftMargin=0.8 * inch, rightMargin=0.8 * inch, topMargin=0.7 * inch, bottomMargin=0.7 * inch)
    doc.build(story)
    print(f"Wrote {TARGET}")


if __name__ == "__main__":
    build()
