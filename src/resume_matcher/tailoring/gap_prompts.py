"""Turn the gaps found in an analysis into questions the user can answer."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from ..analysis import Analysis
from ..matcher.scorer import GAP, PARTIAL
from ..parser.resume_parser import Resume
from ..skills import related_present
from ..suggestions.rewrite_suggester import key_phrase

PROJECTS_ANCHOR = "section::projects"


@dataclass
class GapPrompt:
    id: str
    requirement: str
    category: str
    score: float  # 0..1
    label: str
    missing_skills: list[str]  # canonical names not found in the resume
    terms: list[str]  # the posting's spelling of those skills
    phrase: str  # the requirement without boilerplate
    question: str
    related: list[str] = field(default_factory=list)  # related skills the resume already has
    tools: list[str] = field(default_factory=list)  # named tools among the terms (Kafka, not "data pipelines")
    suggested_anchor: Optional[str] = None

    @property
    def target(self) -> str:
        """What the user would be adding: the skill names, or the requirement phrase."""
        return _join(self.tools or self.terms) if self.terms else self.phrase


def _join(items: list[str], word: str = "and") -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + f" {word} {items[-1]}"


def anchor_options(resume: Resume) -> list[tuple[str, str]]:
    """Places a new bullet can go: each job in Experience, plus Projects. Returns (key, label)."""
    options: list[tuple[str, str]] = []
    seen = set()
    for b in resume.bullets:
        if b.section == "experience" and b.context and b.context not in seen:
            seen.add(b.context)
            options.append((f"context::{b.context}", b.context))
    options.append((PROJECTS_ANCHOR, "Projects section"))
    return options


def build_gap_prompts(analysis: Analysis, limit: int = 10) -> list[GapPrompt]:
    resume, job = analysis.resume, analysis.job
    targets = [m for m in analysis.result.requirement_matches if m.label in (GAP, PARTIAL)]
    targets.sort(key=lambda m: -(m.requirement.weight * (1 - m.score)))

    prompts: list[GapPrompt] = []
    for i, m in enumerate(targets):
        req = m.requirement
        if req.years and not req.skills:
            continue  # "5+ years" is about the timeline, not something a new bullet fixes
        missing = list(m.missing_skills)  # already empty when one of "X or Y" is present
        if not missing and m.label == PARTIAL:
            continue  # the rewrite suggestions already cover partial matches with known skills
        terms = [job.skill_surface_forms.get(s, s) for s in missing]
        phrase = key_phrase(req.text)
        tools = [t for t in terms if any(ch.isupper() for ch in t)]
        if tools:
            pronoun = "it" if len(tools) == 1 else "any of them"
            question = (
                f"The posting asks for {_join(tools)}, which your resume does not mention. "
                f"Have you used {pronoun} anywhere: at work, in a project, or in school?"
            )
        else:
            # Concepts like "data pipelines" read better as the full requirement.
            question = f"The posting asks for \"{phrase}\". Have you done something like this that your resume leaves out?"
        related = sorted({r for s in missing for r in related_present(s, resume.skills)})
        anchor = None
        if m.best_bullet is not None and m.best_bullet.section == "experience" and m.best_bullet.context:
            anchor = f"context::{m.best_bullet.context}"
        prompts.append(
            GapPrompt(
                id=f"gap-{i}-{re.sub(r'[^a-z0-9]+', '-', phrase.lower())[:40].strip('-')}",
                requirement=req.text,
                category=req.category,
                score=m.score,
                label=m.label,
                missing_skills=missing,
                terms=terms,
                phrase=phrase,
                question=question,
                related=related,
                tools=tools,
                suggested_anchor=anchor,
            )
        )
        if len(prompts) >= limit:
            break
    return prompts
