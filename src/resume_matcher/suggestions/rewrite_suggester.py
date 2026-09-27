"""Suggest honest bullet edits that mirror the job description's language.

Every suggestion is built only from material already in the resume: the
original bullet, skills the resume already names, and the job description's
spelling of those same skills. Nothing here invents tools, employers, numbers,
or responsibilities. A final guard drops any suggested text that would name a
skill the resume does not already mention.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from ..matcher.embeddings import Embedder
from ..matcher.scorer import GAP, PARTIAL, MatchResult, RequirementMatch, rescale
from ..parser.job_parser import JobDescription
from ..parser.resume_parser import EVIDENCE_SECTIONS, Bullet, Resume
from ..matcher.embeddings import tokenize
from ..skills import extract_skills, find_skill_mentions, related_present
from .gap_analysis import GapReport

MIN_PLAUSIBLE = 0.25  # rescaled similarity needed before a bullet counts as related evidence
WEAK_PLAUSIBLE = 0.10  # below this, a bullet is not worth pointing at at all

TERMINOLOGY = "terminology"
SURFACE_SKILL = "surface-skill"
STRONGER_VERB = "stronger-verb"
REFRAME = "reframe"
MISSING = "missing"


@dataclass
class Suggestion:
    kind: str
    requirement: Optional[str]
    original: Optional[str]
    suggested: Optional[str]
    rationale: str
    caution: Optional[str] = None
    section: Optional[str] = None
    context: Optional[str] = None

    @property
    def has_rewrite(self) -> bool:
        return bool(self.suggested) and self.suggested != self.original


_IRREGULAR_PAST = {
    "build": "built", "lead": "led", "write": "wrote", "run": "ran", "make": "made", "drive": "drove",
    "teach": "taught", "set": "set", "get": "got", "give": "gave", "take": "took", "bring": "brought",
    "grow": "grew", "begin": "began", "hold": "held", "keep": "kept", "find": "found", "put": "put",
    "sell": "sold", "win": "won", "spend": "spent", "cut": "cut", "think": "thought", "oversee": "oversaw",
    "undertake": "undertook", "rebuild": "rebuilt", "rewrite": "rewrote", "understand": "understood",
}
_IRREGULAR_GERUND = {
    "building": "build", "leading": "lead", "writing": "write", "running": "run", "making": "make",
    "driving": "drive", "teaching": "teach", "setting": "set", "getting": "get", "giving": "give",
    "taking": "take", "bringing": "bring", "growing": "grow", "beginning": "begin", "holding": "hold",
    "keeping": "keep", "finding": "find", "putting": "put", "selling": "sell", "winning": "win",
    "spending": "spend", "cutting": "cut", "overseeing": "oversee", "rebuilding": "rebuild",
    "rewriting": "rewrite", "thinking": "think",
}

_WEAK_OPENERS = [
    (re.compile(r"^(?:was\s+)?responsible\s+for\s+(\w+ing)\b\s*", re.IGNORECASE), "gerund"),
    (re.compile(r"^(?:was\s+)?(?:tasked|charged)\s+with\s+(\w+ing)\b\s*", re.IGNORECASE), "gerund"),
    (re.compile(r"^(?:was\s+)?in\s+charge\s+of\s+(\w+ing)\b\s*", re.IGNORECASE), "gerund"),
    (re.compile(r"^(?:was\s+)?responsible\s+for\s+(?=the\b|a\b|an\b|our\b|all\b)", re.IGNORECASE), "owned"),
    (re.compile(r"^(?:was\s+)?in\s+charge\s+of\s+(?=the\b|a\b|an\b|our\b|all\b)", re.IGNORECASE), "owned"),
]
_ADVICE_ONLY_OPENERS = re.compile(
    r"^(worked on|helped( to)?|assisted( with| in)?|involved in|participated in|contributed to|exposure to|duties included)\b",
    re.IGNORECASE,
)

_REQUIREMENT_LEAD_RE = re.compile(
    r"^(?:(?:you have|you've got|we'd love|we want|must have|you will|you'll|will|to)\s+)?"
    r"(?:(?:at least\s+)?\d{1,2}\s*\+?\s*(?:-\s*\d{1,2}\s*)?(?:years|yrs)(?:\s+of)?\s+)?"
    r"(?:(?:strong|solid|proven|demonstrated|deep|excellent|good|hands-on|working|practical|professional|some|basic|"
    r"advanced|extensive|significant|familiarity|prior)\s+)*"
    r"(?:(?:experience|proficiency|familiarity|knowledge|understanding|background|expertise|track record|skills?|comfort)"
    r"\s+(?:with|in|of|using|on|around)?\s*)?"
    r"(?:(?:the\s+)?ability\s+to\s+)?",
    re.IGNORECASE,
)


def gerund_to_past(gerund: str) -> str:
    """Convert "migrating" to "migrated". Dropping "ing" and adding "ed" handles silent-e
    verbs (creating, created) and doubled consonants (planning, planned); irregulars use a table."""
    lower = gerund.lower()
    if lower in _IRREGULAR_GERUND:
        base = _IRREGULAR_GERUND[lower]
        past = _IRREGULAR_PAST.get(base, base + "ed")
    else:
        past = lower[:-3] + "ed"
    return past[:1].upper() + past[1:] if gerund[:1].isupper() else past


def strengthen_opener(text: str) -> Optional[str]:
    """Rewrite "Responsible for migrating X" as "Migrated X". Returns None if nothing to change."""
    for pattern, mode in _WEAK_OPENERS:
        m = pattern.match(text)
        if not m:
            continue
        rest = text[m.end() :]
        if mode == "gerund":
            verb = gerund_to_past(m.group(1))
            verb = verb[:1].upper() + verb[1:]
            return f"{verb} {rest}".strip()
        return f"Owned {rest}".strip()
    return None


def key_phrase(requirement_text: str) -> str:
    """Strip boilerplate like "5+ years of strong experience with" from a requirement."""
    phrase = _REQUIREMENT_LEAD_RE.sub("", requirement_text.strip(), count=1).strip().rstrip(".;")
    if len(phrase.split()) < 2:
        phrase = requirement_text.strip().rstrip(".;")
    first = phrase.split()[0]
    if first[1:].islower() and not extract_skills(first):
        phrase = phrase[:1].lower() + phrase[1:]
    return phrase


def align_terminology(text: str, job: JobDescription) -> tuple[str, list[tuple[str, str]]]:
    """Use the job description's spelling for skills the bullet already names."""
    swaps = []
    out = text
    for m in reversed(find_skill_mentions(text)):
        target = job.skill_surface_forms.get(m.skill)
        if not target or target.lower() == m.surface.lower():
            continue
        if target.lower() in m.surface.lower() or len(target) > 40:
            continue
        out = out[: m.start] + target + out[m.end :]
        swaps.append((m.surface, target))
    swaps.reverse()
    return out, swaps


_PLAIN_WORD_RE = re.compile(r"[A-Za-z][A-Za-z+#-]*")
_FILLER = frozenset(
    "strong solid proven experience proficiency familiarity knowledge understanding background expertise ability "
    "hands-on professional practices practice skills tools such other related environment".split()
)


def missing_terms(requirement_text: str, bullet_text: str, limit: int = 5) -> list[str]:
    """Content words from the requirement that the bullet does not use (compared by stem)."""
    bullet_stems = set(tokenize(bullet_text))
    out: list[str] = []
    for word in _PLAIN_WORD_RE.findall(requirement_text):
        lower = word.lower()
        if lower in _FILLER or len(lower) < 3:
            continue
        stems = tokenize(word)
        if not stems or any(stem in bullet_stems for stem in stems) or lower in {o.lower() for o in out}:
            continue
        out.append(lower if word[1:].islower() or word.islower() else word)
        if len(out) >= limit:
            break
    return out


def _join_skills(skills: list[str]) -> str:
    if len(skills) == 1:
        return skills[0]
    return ", ".join(skills[:-1]) + f" and {skills[-1]}"


def introduces_new_skills(suggested: str, resume: Resume) -> set[str]:
    return extract_skills(suggested) - resume.skills


class RewriteSuggester:
    def __init__(self, resume: Resume, job: JobDescription, result: MatchResult, gaps: GapReport, embedder: Embedder):
        self.resume = resume
        self.job = job
        self.result = result
        self.gaps = gaps
        self.embedder = embedder
        self._bullets = resume.matchable_bullets

    def _evidence_candidates(self, req_index: int) -> list[tuple[Bullet, float]]:
        sim = self.result.similarity_matrix
        if sim.size == 0:
            return []
        row = sim[req_index]
        ranked = sorted(range(len(self._bullets)), key=lambda j: -row[j])
        return [
            (self._bullets[j], rescale(float(row[j]), self.embedder.low, self.embedder.high))
            for j in ranked
            if self._bullets[j].section in EVIDENCE_SECTIONS and self._bullets[j].section != "summary"
        ]

    def _plausible_evidence(self, req_index: int, req_skills: set[str]) -> list[tuple[Bullet, float]]:
        out = []
        for bullet, plausibility in self._evidence_candidates(req_index):
            if plausibility >= MIN_PLAUSIBLE or req_skills & extract_skills(bullet.text):
                out.append((bullet, plausibility))
        return out

    def _has_used_evidence(self, match: RequirementMatch, req_index: int, used: set[str]) -> bool:
        """True when this requirement's evidence already carries another suggestion."""
        return any(b.text in used for b, _ in self._plausible_evidence(req_index, set(match.requirement.skills)))

    def suggest(self, limit: int = 8) -> list[Suggestion]:
        index_of = {id(m): i for i, m in enumerate(self.result.requirement_matches)}
        targets = [m for m in self.result.requirement_matches if m.label in (GAP, PARTIAL)]
        # Biggest shortfall on the most important requirements first.
        targets.sort(key=lambda m: -(m.requirement.weight * (1 - m.score)))

        # Pass 1: requirements with direct evidence claim their bullets first, so a looser
        # "related skill" hint never steals the bullet that directly proves another requirement.
        used_bullets: set[str] = set()
        chosen: dict[int, Suggestion] = {}
        for rank, match in enumerate(targets):
            s = self._suggest_for(match, index_of[id(match)], used_bullets)
            if s:
                chosen[rank] = s
                used_bullets.add(s.original)
        # Pass 2: everything else gets a related-skill hint or an honest "not found" note.
        for rank, match in enumerate(targets):
            if rank in chosen or self._has_used_evidence(match, index_of[id(match)], used_bullets):
                continue
            absent = [sk for sk in match.requirement.skills if sk not in self.resume.skills]
            s = (
                self._related_evidence(match, absent, used_bullets)
                or self._weak_evidence(match, index_of[id(match)], used_bullets)
                or self._no_evidence(match, absent)
            )
            if s.original:
                used_bullets.add(s.original)
            chosen[rank] = s
        suggestions = [chosen[rank] for rank in sorted(chosen)][:limit]

        if len(suggestions) < limit:
            suggestions.extend(self._verb_cleanups(used_bullets, limit - len(suggestions)))
        return suggestions

    def _suggest_for(self, match: RequirementMatch, req_index: int, used: set[str]) -> Optional[Suggestion]:
        req = match.requirement
        req_skills = set(req.skills)

        candidate: Optional[tuple[Bullet, float]] = None
        for bullet, plausibility in self._plausible_evidence(req_index, req_skills):
            if bullet.text not in used:
                candidate = (bullet, plausibility)
                break

        if candidate is None:
            return None

        bullet, _ = candidate
        new_text, notes = bullet.text, []

        stronger = strengthen_opener(new_text)
        if stronger:
            notes.append("leads with the action instead of \"responsible for\"")
            new_text = stronger

        new_text, swaps = align_terminology(new_text, self.job)
        for old, new in swaps:
            notes.append(f"uses the posting's term \"{new}\" instead of \"{old}\"")

        bullet_skills = extract_skills(new_text)
        surfaceable = [
            s for s in req.skills if s in self.resume.skills and s not in bullet_skills and not _any_of_satisfied(match, bullet_skills)
        ]
        caution = None
        if surfaceable:
            names = [self.job.skill_surface_forms.get(s, s) for s in surfaceable]
            new_text = f"{new_text.rstrip(' .')}, using {_join_skills(names)}"
            notes.append(f"names {_join_skills(names)}, which your resume already lists")
            caution = f"Only keep \"using {_join_skills(names)}\" if you actually used it for this work."

        if new_text != bullet.text and introduces_new_skills(new_text, self.resume):
            new_text, notes, caution = bullet.text, [], None

        phrase = key_phrase(req.text)
        if new_text == bullet.text:
            advice = _ADVICE_ONLY_OPENERS.match(bullet.text)
            rationale = (
                f"This is your closest evidence for \"{phrase}\". If it is accurate, reword it so the connection is explicit "
                "(what you did, with which tools, and the measurable result)."
            )
            unused = missing_terms(req.text, bullet.text)
            if unused:
                rationale += f" Wording from the posting this bullet does not use yet: {', '.join(unused)}."
            if advice:
                rationale += (
                    f" It opens with \"{advice.group(1)}\", which undersells the work; start with the specific action you took."
                )
            return Suggestion(
                kind=REFRAME,
                requirement=req.text,
                original=bullet.text,
                suggested=None,
                rationale=rationale,
                caution="Do not claim scope or results beyond what you actually did.",
                section=bullet.section,
                context=bullet.context,
            )

        kind = SURFACE_SKILL if surfaceable else TERMINOLOGY if swaps else STRONGER_VERB
        rationale = f"Closer to the requirement \"{phrase}\": " + "; ".join(notes) + "."
        return Suggestion(
            kind=kind,
            requirement=req.text,
            original=bullet.text,
            suggested=new_text,
            rationale=rationale,
            caution=caution,
            section=bullet.section,
            context=bullet.context,
        )

    def _related_evidence(self, match: RequirementMatch, absent: list[str], used: set[str]) -> Optional[Suggestion]:
        """Point at a bullet whose tools are commonly part of the missing skill (Airflow for ETL)."""
        req = match.requirement
        for skill in absent:
            for related_skill in related_present(skill, self.resume.skills):
                bullet = next((b for b in self.resume.evidence_bullets if related_skill in extract_skills(b.text)), None)
                if bullet is None:
                    continue
                term = self.job.skill_surface_forms.get(skill, skill)
                return Suggestion(
                    kind=REFRAME,
                    requirement=req.text,
                    original=bullet.text,
                    suggested=None,
                    rationale=(
                        f"The posting asks for {term}, which your resume never names, but this bullet mentions "
                        f"{related_skill}, which is often part of that kind of work. If this work really involved "
                        f"{term}, describe it in the posting's words (for example, \"{key_phrase(req.text)}\")."
                    ),
                    caution=f"Only use the term {term} if it accurately describes what you did.",
                    section=bullet.section,
                    context=bullet.context,
                )
        return None

    def _weak_evidence(self, match: RequirementMatch, req_index: int, used: set[str]) -> Optional[Suggestion]:
        """Some overlap, but not enough to count as evidence: point at it without overstating it."""
        req = match.requirement
        for bullet, plausibility in self._evidence_candidates(req_index):
            if plausibility < WEAK_PLAUSIBLE:
                return None
            # No rewrite is proposed here, so pointing at a bullet another suggestion uses is fine.
            unused = missing_terms(req.text, bullet.text)
            rationale = (
                f"The closest line in your resume only loosely connects to \"{key_phrase(req.text)}\". "
                "If this work did involve it, make that explicit; if not, this is a real gap."
            )
            if unused:
                rationale += f" Wording from the posting this bullet does not use yet: {', '.join(unused)}."
            return Suggestion(
                kind=REFRAME,
                requirement=req.text,
                original=bullet.text,
                suggested=None,
                rationale=rationale,
                caution="Do not stretch a loosely related bullet to claim this requirement.",
                section=bullet.section,
                context=bullet.context,
            )
        return None

    def _no_evidence(self, match: RequirementMatch, absent: list[str]) -> Suggestion:
        req = match.requirement
        listed_only = [s for s in req.skills if s in self.resume.listed_skills and s not in self.resume.evidenced_skills]
        if listed_only and not absent:
            names = _join_skills([self.job.skill_surface_forms.get(s, s) for s in listed_only])
            return Suggestion(
                kind=SURFACE_SKILL,
                requirement=req.text,
                original=None,
                suggested=None,
                rationale=(
                    f"Your Skills section lists {names}, but no experience or project bullet shows it in use. "
                    "If you have applied it at work or in a project, add or edit a bullet that says where and what it achieved."
                ),
                caution="Skip this if the skill is only something you have read about.",
            )
        if absent:
            verb = "does" if len(absent) == 1 else "do"
            rationale = f"{_join_skills(absent)} {verb} not appear anywhere in your resume."
        else:
            rationale = "Nothing in your resume clearly addresses this requirement."
        return Suggestion(
            kind=MISSING,
            requirement=req.text,
            original=None,
            suggested=None,
            rationale=rationale
            + " If you have genuine experience here, add a bullet describing it; if not, leave it out and consider "
            "addressing it in a cover letter (for example, related work or how you would ramp up).",
            caution="Never add experience you do not have.",
        )

    def _verb_cleanups(self, used: set[str], limit: int) -> list[Suggestion]:
        out = []
        for bullet in self.resume.evidence_bullets:
            if len(out) >= limit:
                break
            if bullet.text in used:
                continue
            stronger = strengthen_opener(bullet.text)
            if stronger:
                new_text, swaps = align_terminology(stronger, self.job)
                out.append(
                    Suggestion(
                        kind=STRONGER_VERB,
                        requirement=None,
                        original=bullet.text,
                        suggested=new_text,
                        rationale="Starts with what you did rather than \"responsible for\", which reads stronger to recruiters and ATS.",
                        section=bullet.section,
                        context=bullet.context,
                    )
                )
                used.add(bullet.text)
        return out


def _any_of_satisfied(match: RequirementMatch, bullet_skills: set[str]) -> bool:
    from ..matcher.scorer import is_any_of

    return is_any_of(match.requirement) and bool(set(match.requirement.skills) & bullet_skills)


def suggest_rewrites(
    resume: Resume, job: JobDescription, result: MatchResult, gaps: GapReport, embedder: Embedder, limit: int = 8
) -> list[Suggestion]:
    return RewriteSuggester(resume, job, result, gaps, embedder).suggest(limit=limit)
