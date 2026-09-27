"""Find the requirements a resume does not cover and the bullets that do not help."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..matcher.scorer import GAP, PARTIAL, STRONG, BulletMatch, MatchResult, RequirementMatch, is_any_of
from ..parser.job_parser import PREFERRED, JobDescription
from ..parser.resume_parser import EVIDENCE_SECTIONS, Resume
from ..skills import extract_skills, related_present

LOW_RELEVANCE = 0.15

# Capitalized or mixed-case technical terms the skill vocabulary may not know (TensorRT, CVPR, SOC 2).
_TERM_RE = re.compile(r"\b(?:[A-Z][a-z]*[A-Z][A-Za-z0-9]*|[A-Z]{2,}[a-z]?[0-9]*|[A-Za-z]+[0-9]+[A-Za-z]*)\b")
_COMMON_CAPS = frozenset(
    {"I", "US", "USA", "UK", "EU", "IT", "OR", "AND", "THE", "TBD", "FAQ", "HR", "CEO", "CTO", "VP", "OK", "EOE", "YOE", "N/A"}
)


@dataclass
class GapReport:
    strengths: list[RequirementMatch] = field(default_factory=list)
    partial: list[RequirementMatch] = field(default_factory=list)
    gaps: list[RequirementMatch] = field(default_factory=list)
    missing_required_skills: list[str] = field(default_factory=list)
    missing_preferred_skills: list[str] = field(default_factory=list)
    listed_not_shown: list[str] = field(default_factory=list)  # in the Skills section but never shown in a bullet
    other_missing_terms: list[str] = field(default_factory=list)
    related_evidence: dict[str, list[str]] = field(default_factory=dict)  # missing skill -> related skills present
    low_relevance_bullets: list[BulletMatch] = field(default_factory=list)


def analyze_gaps(resume: Resume, job: JobDescription, result: MatchResult, low_relevance: float = LOW_RELEVANCE) -> GapReport:
    matches = sorted(result.requirement_matches, key=lambda m: m.score)
    report = GapReport(
        strengths=sorted([m for m in matches if m.label == STRONG], key=lambda m: -m.score),
        partial=[m for m in matches if m.label == PARTIAL],
        gaps=[m for m in matches if m.label == GAP],
    )

    satisfied_by_alternative: set[str] = set()
    for m in result.requirement_matches:
        if is_any_of(m.requirement) and m.matched_skills:
            satisfied_by_alternative.update(s for s in m.requirement.skills if s not in m.matched_skills)

    for skill in job.required_skills:
        if skill not in resume.skills and skill not in satisfied_by_alternative:
            report.missing_required_skills.append(skill)
    for skill in job.preferred_skills:
        if skill not in resume.skills and skill not in satisfied_by_alternative:
            report.missing_preferred_skills.append(skill)

    for skill in report.missing_required_skills + report.missing_preferred_skills:
        related = related_present(skill, resume.skills)
        if related:
            report.related_evidence[skill] = related

    job_skills = set(job.all_skills)
    report.listed_not_shown = sorted(
        s for s in resume.listed_skills if s in job_skills and s not in resume.evidenced_skills
    )
    report.other_missing_terms = find_other_missing_terms(resume, job)

    report.low_relevance_bullets = sorted(
        (
            b
            for b in result.bullet_matches
            if b.bullet.section in EVIDENCE_SECTIONS and b.bullet.section != "summary" and b.relevance < low_relevance
        ),
        key=lambda b: b.relevance,
    )
    return report


def find_other_missing_terms(resume: Resume, job: JobDescription, limit: int = 12) -> list[str]:
    """Technical-looking terms from the requirements that never appear in the resume."""
    resume_lower = resume.raw_text.lower()
    seen: list[str] = []
    for req in job.requirements:
        if req.category == PREFERRED:
            continue
        for term in _TERM_RE.findall(req.text):
            if term in _COMMON_CAPS or term in seen or len(term) < 2:
                continue
            if extract_skills(term):
                continue  # already covered by the skill keyword lists
            if term.lower() in resume_lower:
                continue
            seen.append(term)
    for req in job.requirements:
        if req.category != PREFERRED:
            continue
        for term in _TERM_RE.findall(req.text):
            if term in _COMMON_CAPS or term in seen or extract_skills(term) or term.lower() in resume_lower:
                continue
            seen.append(term)
    return seen[:limit]
