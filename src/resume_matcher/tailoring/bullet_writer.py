"""Draft a resume bullet from the user's own answers about a gap.

The draft only rearranges and polishes what the user typed. It never adds tools,
numbers, or claims, and `check_draft` rejects any draft that names a skill that
is neither in the user's answers nor already in the resume.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from ..parser.job_parser import JobDescription
from ..parser.resume_parser import Resume
from ..skills import extract_skills
from ..suggestions.rewrite_suggester import _IRREGULAR_GERUND, _IRREGULAR_PAST, align_terminology, gerund_to_past


@dataclass
class GapAnswer:
    has_experience: Optional[bool] = None  # None means "not answered yet"
    what: str = ""  # what you did, in your own words
    tools: str = ""  # tools or technologies, comma separated
    result: str = ""  # outcome, ideally with a number
    anchor: Optional[str] = None  # where the bullet goes (see gap_prompts.anchor_options)


@dataclass
class Draft:
    text: str
    tips: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.text) and not self.problems


_BASE_VERBS = {
    "add", "analyze", "architect", "automate", "build", "coach", "collaborate", "configure", "create", "cut", "debug",
    "define", "deliver", "deploy", "design", "develop", "document", "drive", "enable", "establish", "evaluate",
    "expand", "grow", "help", "implement", "improve", "increase", "integrate", "introduce", "launch", "lead", "maintain",
    "manage", "mentor", "migrate", "model", "monitor", "optimize", "organize", "own", "partner", "plan", "present",
    "prototype", "publish", "rebuild", "redesign", "reduce", "refactor", "research", "run", "scale", "set", "ship",
    "simplify", "streamline", "support", "teach", "test", "train", "use", "win", "work", "write", "make", "measure",
    "tune", "upgrade", "containerize", "orchestrate", "benchmark", "instrument", "visualize", "clean", "wrangle",
    "coordinate", "negotiate", "pitch", "recruit", "hire", "review", "audit", "secure", "harden", "port", "contribute",
}
_DOUBLE_FINAL = {"plan", "ship", "run", "set", "win", "cut", "stop", "drop", "scan", "tag", "log", "chat", "map", "prep"}
_PAST_WORDS = set(_IRREGULAR_PAST.values()) | {"led", "built", "wrote", "ran", "made", "cut", "set", "grew", "won"}
_SCALE_RE = re.compile(
    r"\b(?:per|a|each|every)\s+(?:second|minute|hour|day|week|month|year)\b|"
    r"\b(?:users|customers|requests|events|records|rows|messages|transactions|queries|jobs|files)\b|\b\d+(?:\.\d+)?\s*[kmb]?\s*(?:gb|tb|pb)\b",
    re.IGNORECASE,
)
_LEADING_FILLER_RE = re.compile(r"^(?:i\s+(?:have\s+|had\s+)?|we\s+|me\s+and\s+\w+\s+|my\s+team\s+and\s+i\s+)", re.IGNORECASE)


def to_past(verb: str) -> str:
    lower = verb.lower()
    if lower in _IRREGULAR_PAST:
        return _IRREGULAR_PAST[lower]
    if lower in _DOUBLE_FINAL:
        return lower + lower[-1] + "ed"
    if lower.endswith("e"):
        return lower + "d"
    if re.search(r"[^aeiou]y$", lower):
        return lower[:-1] + "ied"
    return lower + "ed"


_PAST_TO_GERUND = {_IRREGULAR_PAST[base]: gerund for gerund, base in _IRREGULAR_GERUND.items() if base in _IRREGULAR_PAST}


def past_to_gerund(word: str) -> str:
    lower = word.lower()
    if lower.endswith("ing"):
        return lower
    if lower in _PAST_TO_GERUND:
        return _PAST_TO_GERUND[lower]
    if lower.endswith("ied"):
        return lower[:-3] + "ying"
    if lower.endswith("ed"):
        return lower[:-2] + "ing"
    return lower


def _is_past_or_gerund(word: str) -> bool:
    lower = word.lower()
    return lower in _PAST_WORDS or lower.endswith("ed") or (lower.endswith("ing") and len(lower) > 4)


def start_with_action(text: str) -> str:
    """Normalize "I build X" / "building X" / "built X" to "Built X"."""
    text = _LEADING_FILLER_RE.sub("", text.strip()).strip()
    if not text:
        return text
    first, _, rest = text.partition(" ")
    lower = first.lower()
    if lower.endswith("ing") and len(lower) > 4 and lower not in {"something", "everything", "string", "thing"}:
        first = gerund_to_past(lower)
    elif lower in _BASE_VERBS:
        first = to_past(lower)
    elif lower.endswith("s") and lower[:-1] in _BASE_VERBS:
        first = to_past(lower[:-1])  # "builds" -> "built"
    elif lower.endswith("es") and lower[:-2] in _BASE_VERBS:
        first = to_past(lower[:-2])
    return (first[:1].upper() + first[1:] + (" " + rest if rest else "")).strip()


def _split_tools(tools: str) -> list[str]:
    parts = re.split(r",|;|/|\band\b|\+", tools)
    return [p.strip() for p in parts if p.strip()]


def _join(items: list[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + f" and {items[-1]}"


def draft_bullet(answer: GapAnswer, job: Optional[JobDescription] = None, resume: Optional[Resume] = None) -> Draft:
    what = answer.what.strip().rstrip(".")
    if not what:
        return Draft("", problems=["Describe what you did first."])

    text = start_with_action(what)

    tools = [t for t in _split_tools(answer.tools) if t.lower() not in text.lower()]
    if tools:
        text += f" using {_join(tools)}"

    result = answer.result.strip().rstrip(".")
    if result:
        first = result.split()[0]
        if _is_past_or_gerund(first):
            # "reduced costs by 20%" reads as a dangling clause after a comma; "reducing" does not.
            rest = result[len(first):]
            text += ", " + past_to_gerund(first) + rest
        elif _SCALE_RE.search(result):
            text += ", handling " + result[:1].lower() + result[1:]
        else:
            text += ", resulting in " + result[:1].lower() + result[1:]

    if job is not None:
        text, _ = align_terminology(text, job)

    draft = Draft(text)
    draft.problems = check_draft(draft.text, answer, resume)
    if not re.search(r"\d", draft.text):
        draft.tips.append("Add a number if you can: how many users, how much faster, how much data, how many people.")
    if len(draft.text.split()) > 32:
        draft.tips.append("This is long for a bullet. Try to keep it to one or two lines.")
    if draft.text.split()[0].lower() in {"helped", "worked", "assisted", "participated"}:
        draft.tips.append("Start with the specific action you took rather than \"helped\" or \"worked on\".")
    return draft


def check_draft(text: str, answer: GapAnswer, resume: Optional[Resume] = None) -> list[str]:
    """Every skill in the draft must come from the user's answers or the existing resume."""
    allowed = extract_skills(" ".join([answer.what, answer.tools, answer.result]))
    if resume is not None:
        allowed |= resume.skills
    extra = extract_skills(text) - allowed
    return [f"The draft mentions {s}, which is not in your answers." for s in sorted(extra)]
