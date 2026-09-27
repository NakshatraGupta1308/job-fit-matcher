"""Turn a job description into weighted requirements and skill lists."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Union

from ..skills import extract_skills, surface_forms
from ..text_utils import (
    clean_markdown,
    is_bullet,
    join_wrapped,
    merge_wrapped_lines,
    normalize_header,
    normalize_text,
    split_sentences,
    strip_bullet,
    word_count,
)
from .text_extract import extract_text

REQUIRED = "required"
PREFERRED = "preferred"
RESPONSIBILITY = "responsibility"
GENERAL = "general"
IGNORED = "ignored"

CATEGORY_WEIGHTS = {REQUIRED: 1.0, RESPONSIBILITY: 0.8, GENERAL: 0.7, PREFERRED: 0.5}

JOB_SECTION_ALIASES: dict[str, tuple[str, ...]] = {
    PREFERRED: (
        "preferred qualifications", "preferred skills", "preferred experience", "nice to have", "nice to haves",
        "nice-to-have", "nice-to-haves", "bonus points", "bonus", "preferred", "pluses", "extra credit",
        "desired skills", "desired qualifications", "it's a plus if", "it's a plus if you have",
        "good to have", "additional qualifications", "even better if", "bonus if you have",
    ),
    REQUIRED: (
        "requirements", "qualifications", "required qualifications", "minimum qualifications",
        "basic qualifications", "what you'll need", "what you need", "what we're looking for",
        "what we are looking for", "who you are", "must have", "must haves", "must-haves", "you have",
        "you bring", "what you bring", "required skills", "skills", "requirements and qualifications",
        "about you", "skills and experience", "experience", "your experience", "your profile",
        "key requirements", "the ideal candidate", "ideal candidate", "you might be a fit if",
        "you may be a good fit if", "we'd love to hear from you if", "key skills", "competencies",
    ),
    RESPONSIBILITY: (
        "responsibilities", "key responsibilities", "what you'll do", "what you will do",
        "what you'll be doing", "what you will be doing", "the role", "your role", "role",
        "duties", "day to day", "day-to-day", "in this role you will", "in this role, you will",
        "your responsibilities", "job responsibilities", "your impact", "what you'll work on",
        "the opportunity", "your mission", "impact",
    ),
    IGNORED: (
        "about us", "about the company", "who we are", "benefits", "perks", "perks and benefits",
        "what we offer", "compensation", "salary", "pay", "equal opportunity", "equal employment opportunity",
        "why join us", "why you'll love working here", "about the team", "location", "our values",
        "company overview", "how to apply", "application process", "diversity", "eeo statement",
        "working here", "life at", "our culture",
    ),
}

_HEADER_LOOKUP = {normalize_header(a): cat for cat, aliases in JOB_SECTION_ALIASES.items() for a in aliases}

_PREFERRED_HINT_RE = re.compile(
    r"\b(preferred|nice[\s-]to[\s-]have|a plus|is a plus|are a plus|bonus|ideally|desirable|would be great|advantageous)\b",
    re.IGNORECASE,
)
_BOILERPLATE_RE = re.compile(
    r"\b(equal opportunity|without regard to|reasonable accommodation|e-verify|we offer|salary range|benefits include|"
    r"health insurance|401\(k\)|paid time off|apply now|click apply|visa sponsorship)\b",
    re.IGNORECASE,
)
_YEARS_RE = re.compile(r"(\d{1,2})\s*\+?\s*(?:-\s*\d{1,2}\s*)?(?:years|yrs)", re.IGNORECASE)


@dataclass
class Requirement:
    text: str
    category: str
    weight: float
    skills: list[str] = field(default_factory=list)
    years: Optional[int] = None


@dataclass
class JobDescription:
    raw_text: str
    title: Optional[str]
    requirements: list[Requirement] = field(default_factory=list)
    required_skills: list[str] = field(default_factory=list)
    preferred_skills: list[str] = field(default_factory=list)
    skill_surface_forms: dict[str, str] = field(default_factory=dict)
    source: Optional[str] = None

    @property
    def all_skills(self) -> list[str]:
        return self.required_skills + [s for s in self.preferred_skills if s not in self.required_skills]

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "title": self.title,
            "requirements": [asdict(r) for r in self.requirements],
            "required_skills": self.required_skills,
            "preferred_skills": self.preferred_skills,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


def detect_job_section(line: str) -> Optional[str]:
    stripped = line.strip()
    if not stripped or (is_bullet(stripped) and not stripped.startswith("#")):
        return None
    if word_count(stripped) > 9 or stripped.endswith("."):
        return None
    key = normalize_header(stripped)
    if key in _HEADER_LOOKUP:
        return _HEADER_LOOKUP[key]
    # Headers with a trailing colon get a looser match ("Must-have skills:", "About Acme:").
    looks_like_header = stripped.endswith(":") or stripped.startswith("#") or stripped.isupper() or stripped.startswith("**")
    if looks_like_header:
        for alias, cat in sorted(_HEADER_LOOKUP.items(), key=lambda kv: -len(kv[0])):
            if len(alias) >= 4 and (key.startswith(alias) or key.endswith(alias)):
                return cat
        if key.startswith("about ") or key.startswith("life at "):
            return IGNORED
    return None


def _guess_title(lines: list[str]) -> Optional[str]:
    for line in lines[:5]:
        cleaned = clean_markdown(line).strip()
        if not cleaned:
            continue
        if detect_job_section(cleaned):
            return None
        if word_count(cleaned) <= 12 and not cleaned.endswith("."):
            cleaned = re.sub(r"^(job title|title|position|role)\s*:\s*", "", cleaned, flags=re.IGNORECASE)
            return cleaned
        return None
    return None


def _requirement_for(text: str, category: str) -> Optional[Requirement]:
    text = text.strip().rstrip(";").strip()
    skills = sorted(extract_skills(text))
    if _BOILERPLATE_RE.search(text):
        return None
    if word_count(text) < 3 and not skills:
        return None
    if category in (REQUIRED, GENERAL) and _PREFERRED_HINT_RE.search(text):
        category = PREFERRED
    years = None
    m = _YEARS_RE.search(text)
    if m:
        years = int(m.group(1))
    return Requirement(text=text, category=category, weight=CATEGORY_WEIGHTS[category], skills=skills, years=years)


def parse_job_text(text: str, source: Optional[str] = None) -> JobDescription:
    text = normalize_text(text)
    lines = text.split("\n")
    title = _guess_title(lines)
    lines = merge_wrapped_lines(lines, lambda l: detect_job_section(l) is not None or l.strip() == (title or ""))

    items: list[tuple[str, str]] = []  # (text, category)
    current = GENERAL
    saw_header = False
    open_item: Optional[int] = None

    for index, raw_line in enumerate(lines):
        line = raw_line.strip()
        if not line:
            open_item = None
            continue
        if title and index < 5 and clean_markdown(line) == title:
            continue
        section = detect_job_section(line)
        if section:
            current = section
            saw_header = True
            open_item = None
            continue
        if current == IGNORED:
            continue

        if is_bullet(line):
            items.append((clean_markdown(strip_bullet(line)), current))
            open_item = len(items) - 1
            continue

        cleaned = clean_markdown(line)
        if open_item is not None and cleaned[:1].islower():
            prev_text, prev_cat = items[open_item]
            items[open_item] = (join_wrapped(prev_text, cleaned), prev_cat)
            continue
        open_item = None
        for sentence in split_sentences(cleaned):
            items.append((sentence, current))

    requirements: list[Requirement] = []
    seen = set()
    for item_text, category in items:
        if not saw_header and category == GENERAL and word_count(item_text) > 60:
            continue
        req = _requirement_for(item_text, category)
        if req and req.text.lower() not in seen:
            seen.add(req.text.lower())
            requirements.append(req)

    required_skills: list[str] = []
    preferred_skills: list[str] = []
    for req in requirements:
        target = preferred_skills if req.category == PREFERRED else required_skills
        for skill in req.skills:
            if skill not in target:
                target.append(skill)
    preferred_skills = [s for s in preferred_skills if s not in required_skills]

    return JobDescription(
        raw_text=text,
        title=title,
        requirements=requirements,
        required_skills=required_skills,
        preferred_skills=preferred_skills,
        skill_surface_forms=surface_forms(text),
        source=source,
    )


def parse_job(path_or_text: Union[str, Path]) -> JobDescription:
    """Parse a job description from a file path or from pasted text."""
    if isinstance(path_or_text, Path) or (
        isinstance(path_or_text, str) and "\n" not in path_or_text and Path(path_or_text).suffix and Path(path_or_text).exists()
    ):
        path = Path(path_or_text)
        return parse_job_text(extract_text(path), source=str(path))
    return parse_job_text(str(path_or_text))
