"""Rebuild the resume with accepted edits and new bullets, then re-score it."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from ..analysis import Analysis, analyze
from ..matcher.embeddings import Embedder
from ..parser.job_parser import JobDescription
from ..parser.resume_parser import Resume, parse_resume_text
from ..skills import extract_skills
from .gap_prompts import PROJECTS_ANCHOR

_SECTION_TITLES = {
    "summary": "Summary", "experience": "Experience", "projects": "Projects", "skills": "Skills",
    "education": "Education", "certifications": "Certifications", "awards": "Awards",
    "publications": "Publications", "volunteering": "Volunteering",
}


@dataclass
class Addition:
    anchor: str  # "context::<job context>" or "section::<section name>"
    text: str


@dataclass
class TailoringPlan:
    edits: dict[int, str] = field(default_factory=dict)  # bullet index -> replacement text
    additions: list[Addition] = field(default_factory=list)
    new_skills: list[str] = field(default_factory=list)  # appended to the Skills section

    @property
    def is_empty(self) -> bool:
        return not (self.edits or self.additions or self.new_skills)

    def change_count(self) -> int:
        return len(self.edits) + len(self.additions) + (1 if self.new_skills else 0)


def bullet_index(resume: Resume, text: str) -> Optional[int]:
    for i, b in enumerate(resume.bullets):
        if b.text == text:
            return i
    return None


@dataclass
class Block:
    kind: str  # "name", "text", "heading", "context", "bullet", "line", "paragraph", "blank"
    text: str = ""
    added: bool = False  # new or edited, so renderers can highlight it


def build_blocks(resume: Resume, plan: TailoringPlan) -> list[Block]:
    """Walk the original layout, applying edits and inserting additions in the right places."""
    layout = resume.layout
    by_anchor: dict[str, list[str]] = {}
    for add in plan.additions:
        if add.text.strip():
            by_anchor.setdefault(add.anchor, []).append(add.text.strip())

    # Where each anchor's additions go: after the last layout item that belongs to it.
    insert_after: dict[int, list[str]] = {}
    pending = dict(by_anchor)
    for anchor in list(pending):
        pos = _anchor_position(resume, anchor)
        if pos is not None:
            insert_after.setdefault(pos, []).extend(pending.pop(anchor))

    new_skills_line: Optional[int] = None
    if plan.new_skills:
        skill_lines = [i for i, item in enumerate(layout) if item.section == "skills" and item.kind in ("line", "text", "paragraph")]
        new_skills_line = skill_lines[-1] if skill_lines else None

    blocks: list[Block] = []
    seen_body = False
    prev_kind = None
    for i, item in enumerate(layout):
        if item.kind == "heading":
            blocks += [Block("blank"), Block("heading", _title(item.text))]
            seen_body = True
        elif item.section == "header" and not seen_body:
            blocks.append(Block("name" if not blocks else "text", item.text))
        elif item.kind == "context":
            if prev_kind not in ("context", "heading"):
                blocks.append(Block("blank"))
            blocks.append(Block("context", item.text))
        elif item.kind == "bullet":
            idx = item.bullet_ids[0]
            text = plan.edits.get(idx, resume.bullets[idx].text)
            blocks.append(Block("bullet", text, added=idx in plan.edits))
        elif item.kind == "line":
            idx = item.bullet_ids[0]
            text = plan.edits.get(idx, resume.bullets[idx].text)
            added = idx in plan.edits
            if i == new_skills_line:
                text, added = _append_skills(text, plan.new_skills), True
            blocks.append(Block("line", text, added=added))
        elif item.kind == "paragraph":
            text, added = item.text, False
            for idx in item.bullet_ids:
                if idx in plan.edits:
                    text = text.replace(resume.bullets[idx].text, plan.edits[idx])
                    added = True
            blocks.append(Block("paragraph", text, added=added))
        else:
            blocks.append(Block("text", item.text))
        for extra in insert_after.get(i, []):
            blocks.append(Block("bullet", extra, added=True))
        prev_kind = item.kind

    # Anchors whose section does not exist yet (usually Projects) get a new section.
    for anchor, texts in pending.items():
        section = anchor.split("::", 1)[1] if anchor.startswith("section::") else "projects"
        blocks += [Block("blank"), Block("heading", _SECTION_TITLES.get(section, section.title()))]
        blocks += [Block("bullet", t, added=True) for t in texts]
    if plan.new_skills and new_skills_line is None:
        blocks += [Block("blank"), Block("heading", "Skills"), Block("line", ", ".join(plan.new_skills), added=True)]
    return blocks


def _title(text: str) -> str:
    return text.title() if text.isupper() else text


def _append_skills(line: str, skills: list[str]) -> str:
    existing = extract_skills(line)
    extra = [s for s in skills if s.lower() not in line.lower() and s not in existing]
    if not extra:
        return line
    return f"{line.rstrip(' ,.')}, {', '.join(extra)}"


def _anchor_position(resume: Resume, anchor: str) -> Optional[int]:
    kind, _, value = anchor.partition("::")
    layout = resume.layout
    last = None
    if kind == "context":
        ids = {i for i, b in enumerate(resume.bullets) if b.context == value}
        for i, item in enumerate(layout):
            if ids & set(item.bullet_ids):
                last = i
        if last is None:  # a job with no bullets yet: go right after its title lines
            for i, item in enumerate(layout):
                if item.kind == "context" and item.text in value:
                    last = i
        return last
    if kind == "section":
        for i, item in enumerate(layout):
            if item.section == value:
                last = i
        return last
    return None


def to_markdown(resume: Resume, plan: TailoringPlan) -> str:
    out: list[str] = []
    prev = None
    for block in build_blocks(resume, plan):
        if block.kind == "name":
            out.append(f"# {block.text}")
        elif block.kind == "heading":
            out.append(f"## {block.text}")
        elif block.kind == "context":
            out.append(f"**{block.text}**  " if prev != "context" else f"{block.text}  ")
        elif block.kind == "bullet":
            out.append(f"- {block.text}")
        elif block.kind == "line":
            out.append(f"{block.text}  ")
        elif block.kind == "blank":
            out.append("")
        else:
            out.append(block.text)
        if block.kind in ("heading", "name"):
            out.append("")
        prev = block.kind
    return _tidy("\n".join(out))


def to_text(resume: Resume, plan: TailoringPlan) -> str:
    out: list[str] = []
    for block in build_blocks(resume, plan):
        if block.kind == "name":
            out.append(block.text.upper())
        elif block.kind == "heading":
            out += [block.text.upper(), "-" * len(block.text)]
        elif block.kind == "bullet":
            out.append(f"  - {block.text}")
        elif block.kind == "blank":
            out.append("")
        else:
            out.append(block.text)
    return _tidy("\n".join(out))


def _tidy(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def to_pdf(resume: Resume, plan: TailoringPlan) -> bytes:
    """Render a clean single-column PDF. Needs reportlab (pip install reportlab)."""
    from io import BytesIO
    from xml.sax.saxutils import escape

    from reportlab.lib.pagesizes import LETTER
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer

    styles = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=styles["BodyText"], fontSize=10, leading=13, spaceAfter=1)
    name = ParagraphStyle("name", parent=styles["Heading1"], fontSize=18, spaceAfter=2)
    heading = ParagraphStyle("heading", parent=styles["Heading2"], fontSize=11.5, spaceBefore=8, spaceAfter=3)
    context = ParagraphStyle("context", parent=body, fontName="Helvetica-Bold")

    story: list = []
    bullets: list = []

    def flush() -> None:
        if bullets:
            story.append(ListFlowable([ListItem(Paragraph(b, body), leftIndent=12) for b in bullets], bulletType="bullet", start="•", leftIndent=12))
            bullets.clear()

    for block in build_blocks(resume, plan):
        text = escape(block.text)
        if block.kind == "bullet":
            bullets.append(text)
            continue
        flush()
        if block.kind == "name":
            story.append(Paragraph(text, name))
        elif block.kind == "heading":
            story.append(Paragraph(text.upper(), heading))
        elif block.kind == "context":
            story.append(Paragraph(text, context))
        elif block.kind == "blank":
            story.append(Spacer(1, 3))
        else:
            story.append(Paragraph(text, body))
    flush()

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=LETTER, leftMargin=0.75 * inch, rightMargin=0.75 * inch, topMargin=0.6 * inch, bottomMargin=0.6 * inch)
    doc.build(story)
    return buffer.getvalue()


def pdf_available() -> bool:
    try:
        import reportlab  # noqa: F401
    except ImportError:
        return False
    return True


def tailored_resume(resume: Resume, plan: TailoringPlan) -> Resume:
    """Parse the tailored markdown back into a Resume, exactly as a user would re-upload it."""
    tailored = parse_resume_text(to_markdown(resume, plan), source=resume.source)
    return tailored


def rescore(resume: Resume, plan: TailoringPlan, job: JobDescription, embedder: Embedder, max_suggestions: int = 8) -> Analysis:
    return analyze(tailored_resume(resume, plan), job, embedder=embedder, max_suggestions=max_suggestions)
