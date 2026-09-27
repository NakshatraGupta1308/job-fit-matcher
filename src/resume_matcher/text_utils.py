"""Small text helpers shared by the parsers."""

from __future__ import annotations

import re
import unicodedata
from typing import Callable

# Bullet glyphs commonly produced by word processors and PDF exports.
BULLET_CHARS = "-*+•●▪◦‣⁃·■□∙➢✓✔►▸>"
_BULLET_RE = re.compile(r"^\s*(?:[" + re.escape(BULLET_CHARS) + r"]|\d{1,2}[.)])\s+")

_ABBREVIATIONS = ("e.g.", "i.e.", "etc.", "vs.", "approx.", "incl.", "Inc.", "Ltd.", "Co.", "Jr.", "Sr.", "Dr.", "Mr.", "Ms.")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")

_MONTHS = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?"
DATE_RANGE_RE = re.compile(
    r"(?:" + _MONTHS + r"\s+)?(?:19|20)\d{2}\s*(?:-|to)\s*(?:(?:" + _MONTHS + r"\s+)?(?:19|20)\d{2}|present|current|now|today)",
    re.IGNORECASE,
)


def normalize_text(text: str) -> str:
    """Normalize unicode punctuation and whitespace so the heuristics see clean input."""
    text = unicodedata.normalize("NFKC", text)
    chars = []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat == "Pd":
            # Any dash-like punctuation becomes a plain hyphen.
            chars.append("-")
        elif ch in "‘’‛′":
            chars.append("'")
        elif ch in "“”‟″":
            chars.append('"')
        elif ch in "   \t":
            chars.append(" ")
        elif cat == "Cf":
            continue
        else:
            chars.append(ch)
    text = "".join(chars).replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r" {2,}", " ", line).rstrip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def is_bullet(line: str) -> bool:
    return bool(_BULLET_RE.match(line))


def strip_bullet(line: str) -> str:
    return _BULLET_RE.sub("", line, count=1).strip()


def clean_markdown(line: str) -> str:
    """Remove lightweight markdown decoration (headers, bold, italics, links)."""
    line = re.sub(r"^\s*#{1,6}\s*", "", line)
    line = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", line)
    line = re.sub(r"(\*\*|__)(.+?)\1", r"\2", line)
    line = re.sub(r"(?<!\w)[*_](\S(?:.*?\S)?)[*_](?!\w)", r"\1", line)
    line = line.replace("`", "")
    return line.strip()


def normalize_header(line: str) -> str:
    """Reduce a potential section header to a comparable key."""
    line = clean_markdown(line)
    line = line.strip().strip(":").strip()
    line = line.lower().replace("&", "and")
    line = re.sub(r"[^a-z0-9' ]+", " ", line)
    return re.sub(r"\s+", " ", line).strip()


def split_sentences(text: str) -> list[str]:
    """Split a paragraph into sentences, protecting common abbreviations."""
    protected = text
    for i, abbr in enumerate(_ABBREVIATIONS):
        protected = protected.replace(abbr, f"<ABBR{i}>")
    parts = _SENTENCE_SPLIT_RE.split(protected)
    out = []
    for part in parts:
        for i, abbr in enumerate(_ABBREVIATIONS):
            part = part.replace(f"<ABBR{i}>", abbr)
        part = part.strip()
        if part:
            out.append(part)
    return out


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9+#.]+", text))


_CONNECTIVE_END_RE = re.compile(
    r"(?:,|;|:|&|\+|/|\b(?:and|or|of|to|the|a|an|in|on|for|with|by|from|at|into|across|via|using|over|as|that|while))$",
    re.IGNORECASE,
)


def merge_wrapped_lines(lines: list[str], is_header: Callable[[str], bool]) -> list[str]:
    """Rejoin lines that a PDF export or text editor wrapped mid-sentence.

    A line is treated as the tail of the previous line when it is not a bullet,
    header, or date line and either starts in lowercase, follows a line that ends
    with a connective word, or follows a line that runs the full page width.
    """
    widths = sorted(len(l.strip()) for l in lines if l.strip())
    full_width = widths[int(len(widths) * 0.9)] if len(widths) >= 5 else 10_000
    out: list[str] = []
    for i, line in enumerate(lines):
        cur = line.strip()
        prev = out[-1].strip() if out else ""
        if not cur or not prev or is_bullet(cur) or is_header(cur) or is_header(prev) or DATE_RANGE_RE.search(cur):
            out.append(line)
            continue
        if prev.endswith((".", "!", "?")) or (cur.isupper() and len(cur) > 3):
            out.append(line)
            continue
        next_line = lines[i + 1].strip() if i + 1 < len(lines) else ""
        starts_lower = cur[:1].islower() or cur[:1] in "(&,;%"
        connective = bool(_CONNECTIVE_END_RE.search(prev))
        wide = len(prev) >= 0.85 * full_width and " | " not in cur and not DATE_RANGE_RE.search(next_line)
        if starts_lower or connective or wide:
            out[-1] = join_wrapped(out[-1].rstrip(), cur)
        else:
            out.append(line)
    return out


def join_wrapped(previous: str, continuation: str) -> str:
    """Join a line that was wrapped mid-sentence, handling hyphenated breaks."""
    if previous.endswith("-") and continuation[:1].islower():
        return previous[:-1] + continuation
    return f"{previous} {continuation}"
