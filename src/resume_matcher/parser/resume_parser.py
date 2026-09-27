"""Split a resume into sections and bullet points using header heuristics."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional, Union

from ..skills import extract_skills
from ..text_utils import (
    DATE_RANGE_RE,
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

SECTION_ALIASES: dict[str, tuple[str, ...]] = {
    "summary": (
        "summary", "profile", "professional summary", "career summary", "about", "about me",
        "objective", "career objective", "professional profile", "overview",
    ),
    "experience": (
        "experience", "work experience", "professional experience", "employment", "employment history",
        "work history", "relevant experience", "career history", "industry experience", "internships",
        "internship experience",
    ),
    "projects": (
        "projects", "personal projects", "selected projects", "academic projects", "side projects",
        "key projects", "project experience", "open source", "open source contributions",
    ),
    "skills": (
        "skills", "technical skills", "core skills", "key skills", "core competencies", "competencies",
        "technologies", "tools", "skills and tools", "tools and technologies", "technical proficiencies",
        "tech stack", "languages and tools", "skills and technologies",
    ),
    "education": ("education", "academic background", "education and training", "academics", "qualifications"),
    "certifications": (
        "certifications", "certificates", "licenses and certifications", "licenses", "certifications and licenses",
        "courses", "coursework", "relevant coursework",
    ),
    "awards": ("awards", "honors", "achievements", "honors and awards", "awards and honors", "accomplishments"),
    "publications": ("publications", "papers", "research"),
    "volunteering": ("volunteering", "volunteer experience", "volunteer", "community involvement", "leadership and activities", "activities"),
}

_HEADER_LOOKUP = {alias: section for section, aliases in SECTION_ALIASES.items() for alias in aliases}

# Sections whose lines describe things the candidate did, so they are worth matching.
EVIDENCE_SECTIONS = ("summary", "experience", "projects", "volunteering", "awards", "publications", "certifications")


@dataclass
class Bullet:
    text: str
    section: str
    context: Optional[str] = None  # e.g. the job title or project name the bullet belongs to


@dataclass
class Resume:
    raw_text: str
    sections: dict[str, list[str]] = field(default_factory=dict)
    bullets: list[Bullet] = field(default_factory=list)
    skills: set[str] = field(default_factory=set)
    listed_skills: set[str] = field(default_factory=set)  # skills named in the Skills section
    evidenced_skills: set[str] = field(default_factory=set)  # skills named in experience or project bullets
    years_experience: Optional[float] = None
    source: Optional[str] = None

    @property
    def evidence_bullets(self) -> list[Bullet]:
        """Bullets that describe work or achievements (not the raw skills list)."""
        return [b for b in self.bullets if b.section in EVIDENCE_SECTIONS]

    @property
    def role_lines(self) -> list[Bullet]:
        """Distinct job titles and employers (dates removed), which are evidence on their own."""
        seen, out = set(), []
        for b in self.bullets:
            if b.section != "experience" or not b.context:
                continue
            title = DATE_RANGE_RE.sub("", b.context)
            title = re.sub(r"(\s*\|\s*)+$", "", re.sub(r"\|\s*\|", "|", title)).strip(" |,-")
            if title and title not in seen:
                seen.add(title)
                out.append(Bullet(title, "roles", None))
        return out

    @property
    def matchable_bullets(self) -> list[Bullet]:
        """Everything worth comparing against job requirements."""
        lines = [b for b in self.bullets if b.section in EVIDENCE_SECTIONS + ("skills", "education", "other")]
        return lines + self.role_lines

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "sections": self.sections,
            "bullets": [asdict(b) for b in self.bullets],
            "skills": sorted(self.skills),
            "listed_skills": sorted(self.listed_skills),
            "evidenced_skills": sorted(self.evidenced_skills),
            "years_experience": self.years_experience,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


def detect_section(line: str) -> Optional[str]:
    """Return the canonical section name if the line looks like a section header."""
    stripped = line.strip()
    if not stripped or is_bullet(stripped) and not stripped.startswith("#"):
        return None
    if word_count(stripped) > 6 or stripped.endswith("."):
        return None
    key = normalize_header(stripped)
    if key in _HEADER_LOOKUP:
        return _HEADER_LOOKUP[key]
    # Headers such as "WORK EXPERIENCE (2019 - 2024)" or "Skills Summary".
    for alias, section in sorted(_HEADER_LOOKUP.items(), key=lambda kv: -len(kv[0])):
        if key.startswith(alias + " ") and (stripped.isupper() or stripped.startswith("#")):
            return section
    return None


def _looks_like_entry_header(line: str) -> bool:
    """Job titles, company names, and date lines: context, not accomplishments."""
    if DATE_RANGE_RE.search(line):
        return True
    words = word_count(line)
    if words <= 8 and not line.rstrip().endswith("."):
        return True
    if re.search(r"\s\|\s", line) and words <= 14:
        return True
    return False


def parse_resume_text(text: str, source: Optional[str] = None) -> Resume:
    text = normalize_text(text)
    sections: dict[str, list[str]] = {}
    bullets: list[Bullet] = []

    current = "header"
    context: Optional[str] = None
    open_bullet: Optional[Bullet] = None  # the bullet that a wrapped line may continue
    last_was_header = False

    for raw_line in merge_wrapped_lines(text.split("\n"), lambda l: detect_section(l) is not None):
        line = raw_line.strip()
        if not line or set(line) <= set("-_=*#~ "):
            open_bullet = None if not line else open_bullet
            continue

        section = detect_section(line)
        if section:
            current = section
            sections.setdefault(current, [])
            context = None
            open_bullet = None
            continue

        was_header, last_was_header = last_was_header, False
        cleaned = clean_markdown(line)
        sections.setdefault(current, []).append(cleaned)

        if current == "header":
            continue

        if is_bullet(line):
            content = clean_markdown(strip_bullet(line))
            if not content:
                continue
            open_bullet = Bullet(content, current, context)
            bullets.append(open_bullet)
            continue

        # Continuation of a bullet that wrapped onto the next line.
        if open_bullet is not None and _continues(open_bullet.text, cleaned):
            open_bullet.text = join_wrapped(open_bullet.text, cleaned)
            continue

        open_bullet = None
        if current == "skills":
            bullets.append(Bullet(cleaned, current, None))
            continue

        if current in ("experience", "projects", "volunteering") and _looks_like_entry_header(cleaned):
            # Title, company, and dates often span consecutive lines; keep them together.
            context = f"{context} | {cleaned}" if was_header and context else cleaned
            last_was_header = True
            continue

        if current == "education" and word_count(cleaned) <= 14:
            bullets.append(Bullet(cleaned, current, None))
            continue

        for sentence in split_sentences(cleaned):
            if word_count(sentence) >= 3:
                bullets.append(Bullet(sentence, current, context))

    listed = set()
    for line in sections.get("skills", []):
        listed |= extract_skills(line)
    evidenced = set()
    for b in bullets:
        if b.section in EVIDENCE_SECTIONS:
            evidenced |= extract_skills(b.text)
            if b.context:
                evidenced |= extract_skills(b.context)

    return Resume(
        raw_text=text,
        sections=sections,
        bullets=bullets,
        skills=extract_skills(text),
        listed_skills=listed,
        evidenced_skills=evidenced,
        years_experience=estimate_years(text, sections.get("experience", [])),
        source=source,
    )


_CONNECTIVE_END_RE = re.compile(r"(?:,|;|:|-|&|\+|/|\b(?:and|or|of|to|the|a|an|in|on|for|with|by|from|at|into|across|via|using|over))$", re.IGNORECASE)


def _continues(previous: str, line: str) -> bool:
    """Decide whether a non-bullet line is the wrapped tail of the previous bullet."""
    if DATE_RANGE_RE.search(line) or detect_section(line) is not None:
        return False
    if previous.rstrip().endswith((".", "!", "?")):
        return False
    return line[:1].islower() or line[:1] in "(&" or bool(_CONNECTIVE_END_RE.search(previous.rstrip()))


_STATED_YEARS_RE = re.compile(r"(\d{1,2})\+?\s*(?:years|yrs)(?:\s+of)?(?:\s+\w+){0,3}\s+experience", re.IGNORECASE)
_MONTH_NUMBERS = {m: i for i, m in enumerate(("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"), 1)}
_RANGE_PARTS_RE = re.compile(
    r"(?:([a-z]{3})[a-z]*\.?\s+)?((?:19|20)\d{2})\s*(?:-|to)\s*(?:(?:([a-z]{3})[a-z]*\.?\s+)?((?:19|20)\d{2})|(present|current|now|today))",
    re.IGNORECASE,
)


def estimate_years(text: str, experience_lines: list[str], today: Optional[date] = None) -> Optional[float]:
    """Estimate total years of experience from a stated figure or from merged date ranges."""
    stated = [int(m.group(1)) for m in _STATED_YEARS_RE.finditer(text)]
    today = today or date.today()
    spans = []
    for line in experience_lines:
        for m in _RANGE_PARTS_RE.finditer(line):
            start = int(m.group(2)) + (_MONTH_NUMBERS.get((m.group(1) or "jan").lower(), 1) - 1) / 12
            if m.group(5):
                end = today.year + (today.month - 1) / 12
            else:
                # End months are inclusive: "Jan 2018 - Dec 2019" is two full years.
                end = int(m.group(4)) + _MONTH_NUMBERS.get((m.group(3) or "dec").lower(), 12) / 12
            if end >= start:
                spans.append((start, end))
    spans.sort()
    total, cur_start, cur_end = 0.0, None, None
    for start, end in spans:
        if cur_end is None or start > cur_end:
            if cur_end is not None:
                total += cur_end - cur_start
            cur_start, cur_end = start, end
        else:
            cur_end = max(cur_end, end)
    if cur_end is not None:
        total += cur_end - cur_start
    candidates = [float(y) for y in stated] + ([round(total, 1)] if total else [])
    return max(candidates) if candidates else None


def parse_resume(path_or_text: Union[str, Path]) -> Resume:
    """Parse a resume from a file path (PDF, TXT, MD) or from raw text."""
    if isinstance(path_or_text, Path) or (
        isinstance(path_or_text, str) and "\n" not in path_or_text and Path(path_or_text).suffix and Path(path_or_text).exists()
    ):
        path = Path(path_or_text)
        return parse_resume_text(extract_text(path), source=str(path))
    return parse_resume_text(str(path_or_text))
