"""Help the user close gaps honestly: ask, draft from their answers, and rebuild the resume."""

from .bullet_writer import Draft, GapAnswer, check_draft, draft_bullet
from .gap_prompts import PROJECTS_ANCHOR, GapPrompt, anchor_options, build_gap_prompts
from .honest_gaps import LearningPlan, cover_letter_line, learning_plan, notes_markdown
from .tailored_resume import (
    Addition,
    TailoringPlan,
    bullet_index,
    pdf_available,
    rescore,
    tailored_resume,
    to_markdown,
    to_pdf,
    to_text,
)

__all__ = [
    "PROJECTS_ANCHOR",
    "Addition",
    "Draft",
    "GapAnswer",
    "GapPrompt",
    "LearningPlan",
    "TailoringPlan",
    "anchor_options",
    "build_gap_prompts",
    "bullet_index",
    "check_draft",
    "cover_letter_line",
    "draft_bullet",
    "learning_plan",
    "notes_markdown",
    "pdf_available",
    "rescore",
    "tailored_resume",
    "to_markdown",
    "to_pdf",
    "to_text",
]
