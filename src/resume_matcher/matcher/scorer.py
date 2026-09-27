"""Compare resume bullets against job requirements and aggregate a match score."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..parser.job_parser import JobDescription, Requirement
from ..parser.resume_parser import Bullet, Resume
from ..skills import related_present
from .embeddings import Embedder

STRONG = "strong"
PARTIAL = "partial"
GAP = "gap"

STRONG_THRESHOLD = 0.70
PARTIAL_THRESHOLD = 0.40

# How much of a requirement's score comes from named skills versus overall meaning.
SKILL_BLEND = 0.4
# How much of the overall score comes from keyword coverage versus requirement coverage.
KEYWORD_BLEND = 0.3

_ANY_OF_RE = re.compile(r"\b(or|either|such as|e\.g\.|for example|like|one of)\b|/", re.IGNORECASE)


@dataclass
class RequirementMatch:
    requirement: Requirement
    best_bullet: Optional[Bullet]
    similarity: float  # raw cosine similarity with the best bullet
    semantic: float  # similarity rescaled to 0..1
    skill_coverage: Optional[float]  # None when the requirement names no skills
    matched_skills: list[str]
    missing_skills: list[str]
    score: float  # 0..1
    label: str
    runner_up: Optional[Bullet] = None


@dataclass
class BulletMatch:
    bullet: Bullet
    best_requirement: Optional[Requirement]
    similarity: float
    relevance: float  # 0..1


@dataclass
class MatchResult:
    overall_score: float  # 0..100
    requirement_score: float  # 0..100
    keyword_score: Optional[float]  # 0..100, None if the job names no known skills
    requirement_matches: list[RequirementMatch]
    bullet_matches: list[BulletMatch]
    backend: str
    similarity_matrix: np.ndarray = field(repr=False, default_factory=lambda: np.zeros((0, 0)))

    @property
    def fit_label(self) -> str:
        return fit_label(self.overall_score)


def fit_label(score: float) -> str:
    if score >= 75:
        return "Strong fit"
    if score >= 55:
        return "Good fit"
    if score >= 35:
        return "Partial fit"
    return "Weak fit"


def rescale(similarity: float, low: float, high: float) -> float:
    return float(np.clip((similarity - low) / (high - low), 0.0, 1.0))


def label_for(score: float) -> str:
    if score >= STRONG_THRESHOLD:
        return STRONG
    if score >= PARTIAL_THRESHOLD:
        return PARTIAL
    return GAP


def is_any_of(requirement: Requirement) -> bool:
    """True for requirements like "PyTorch or TensorFlow" where one named skill is enough."""
    return len(requirement.skills) > 1 and bool(_ANY_OF_RE.search(requirement.text))


def skill_coverage(requirement: Requirement, resume_skills: set[str]) -> tuple[Optional[float], list[str], list[str]]:
    if not requirement.skills:
        return None, [], []
    matched = [s for s in requirement.skills if s in resume_skills]
    missing = [s for s in requirement.skills if s not in resume_skills]
    if is_any_of(requirement):
        coverage = 1.0 if matched else 0.0
        if matched:
            missing = []
    else:
        coverage = len(matched) / len(requirement.skills)
    return coverage, matched, missing


def score_match(resume: Resume, job: JobDescription, embedder: Embedder) -> MatchResult:
    bullets = resume.matchable_bullets
    requirements = job.requirements

    bullet_texts = [_bullet_text(b) for b in bullets]
    requirement_texts = [r.text for r in requirements]
    embedder.fit(bullet_texts + requirement_texts)

    if bullets and requirements:
        bullet_vecs = embedder.encode(bullet_texts)
        requirement_vecs = embedder.encode(requirement_texts)
        sim = requirement_vecs @ bullet_vecs.T  # shape: (requirements, bullets)
    else:
        sim = np.zeros((len(requirements), len(bullets)), dtype=np.float32)

    requirement_matches = []
    for i, req in enumerate(requirements):
        if bullets:
            order = np.argsort(-sim[i])
            best, similarity = bullets[order[0]], float(sim[i, order[0]])
            runner_up = bullets[order[1]] if len(order) > 1 else None
        else:
            best, similarity, runner_up = None, 0.0, None
        semantic = rescale(similarity, embedder.low, embedder.high)
        coverage, matched, missing = skill_coverage(req, resume.skills)
        if coverage is None:
            score = semantic
        else:
            score = (1 - SKILL_BLEND) * semantic + SKILL_BLEND * coverage
        score = _apply_years(score, req, resume)
        requirement_matches.append(
            RequirementMatch(
                requirement=req,
                best_bullet=best,
                similarity=similarity,
                semantic=semantic,
                skill_coverage=coverage,
                matched_skills=matched,
                missing_skills=missing,
                score=score,
                label=label_for(score),
                runner_up=runner_up,
            )
        )

    bullet_matches = []
    for j, bullet in enumerate(bullets):
        if requirements:
            i = int(np.argmax(sim[:, j]))
            similarity = float(sim[i, j])
            bullet_matches.append(BulletMatch(bullet, requirements[i], similarity, rescale(similarity, embedder.low, embedder.high)))
        else:
            bullet_matches.append(BulletMatch(bullet, None, 0.0, 0.0))

    total_weight = sum(m.requirement.weight for m in requirement_matches)
    requirement_score = (
        100 * sum(m.score * m.requirement.weight for m in requirement_matches) / total_weight if total_weight else 0.0
    )

    skill_reqs = [m for m in requirement_matches if m.skill_coverage is not None]
    skill_weight = sum(m.requirement.weight for m in skill_reqs)
    keyword_score = (
        100 * sum(m.skill_coverage * m.requirement.weight for m in skill_reqs) / skill_weight if skill_weight else None
    )

    if keyword_score is None:
        overall = requirement_score
    else:
        overall = (1 - KEYWORD_BLEND) * requirement_score + KEYWORD_BLEND * keyword_score

    return MatchResult(
        overall_score=round(overall, 1),
        requirement_score=round(requirement_score, 1),
        keyword_score=None if keyword_score is None else round(keyword_score, 1),
        requirement_matches=requirement_matches,
        bullet_matches=bullet_matches,
        backend=embedder.name,
        similarity_matrix=sim,
    )


def _apply_years(score: float, req: Requirement, resume: Resume) -> float:
    """"N+ years" requirements are judged partly on the resume's timeline, not just its wording."""
    if not req.years or resume.years_experience is None:
        return score
    ratio = resume.years_experience / req.years
    if req.skills:
        # Total years only count toward "4+ years of X" when X (or a close relative) is present.
        present = [s for s in req.skills if s in resume.skills or related_present(s, resume.skills)]
        skills_ok = len(present) / len(req.skills)
        if ratio < 1 or not skills_ok:
            return score
        return max(score, 0.55 + 0.35 * skills_ok)
    if ratio >= 1:
        return max(score, 0.9)
    return min(max(score, ratio * 0.8), ratio)


def _bullet_text(bullet: Bullet) -> str:
    # Skills-section lines like "Languages: Python, SQL" embed better without the label.
    if bullet.section == "skills" and ":" in bullet.text:
        return bullet.text.split(":", 1)[1].strip()
    return bullet.text
