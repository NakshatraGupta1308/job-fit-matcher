"""Render an analysis as markdown, plain text, or JSON."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Iterable, Optional

from ..analysis import Analysis
from ..matcher.scorer import GAP, PARTIAL, STRONG, RequirementMatch
from ..suggestions.rewrite_suggester import MISSING, Suggestion

FORMATS = ("markdown", "text", "json")

_LABEL_TEXT = {STRONG: "Strong", PARTIAL: "Partial", GAP: "Gap"}


def score_bar(score: float, width: int = 20) -> str:
    filled = int(round(score / 100 * width))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def _md_escape(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def _truncate(text: str, limit: int = 90) -> str:
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def _pct(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.0f}"


def _source_name(path: Optional[str]) -> Optional[str]:
    return Path(path).name if path else None


def _headline(analysis: Analysis) -> list[str]:
    r = analysis.result
    lines = [
        f"Overall match: {r.overall_score:.0f}/100 ({r.fit_label})",
        f"{score_bar(r.overall_score)}",
        f"Requirement coverage: {_pct(r.requirement_score)}/100 | Keyword coverage: {_pct(r.keyword_score)}/100",
    ]
    return lines


def _sentence(text: str) -> str:
    return text if text.endswith((".", "!", "?")) else text + "."


def _mostly_irrelevant(analysis: Analysis) -> bool:
    evidence = [b for b in analysis.result.bullet_matches if b.bullet.section not in ("summary", "skills", "education", "other")]
    return len(evidence) >= 4 and len(analysis.gaps.low_relevance_bullets) / len(evidence) > 0.6


def _with_related(skills: list[str], related: dict[str, list[str]]) -> str:
    parts = []
    for skill in skills:
        if related.get(skill):
            parts.append(f"{skill} (related in your resume: {', '.join(related[skill])})")
        else:
            parts.append(skill)
    return ", ".join(parts)


def _evidence(match: RequirementMatch) -> str:
    return match.best_bullet.text if match.best_bullet else "(no matching bullet)"


def to_markdown(analysis: Analysis) -> str:
    r, g, job = analysis.result, analysis.gaps, analysis.job
    out: list[str] = [f"# Resume match report: {analysis.job_name}", ""]
    meta = []
    if _source_name(analysis.resume.source):
        meta.append(f"**Resume:** {_source_name(analysis.resume.source)}")
    if _source_name(job.source):
        meta.append(f"**Job description:** {_source_name(job.source)}")
    meta.append(f"**Generated:** {analysis.created_at:%Y-%m-%d %H:%M}")
    meta.append(f"**Embedding backend:** {r.backend}")
    out += [" | ".join(meta), ""]

    out += [f"## Overall match: {r.overall_score:.0f}/100 ({r.fit_label})", "", f"`{score_bar(r.overall_score)}`", ""]
    out += [
        f"- **Requirement coverage:** {_pct(r.requirement_score)}/100 (how well your bullets address each requirement)",
        f"- **Keyword coverage:** {_pct(r.keyword_score)}/100 (named skills from the posting that appear in your resume)",
        f"- **Requirements:** {len(g.strengths)} strong, {len(g.partial)} partial, {len(g.gaps)} gaps",
        "",
    ]

    out += ["## Matched strengths", ""]
    if g.strengths:
        out += ["| Requirement | Best evidence | Score |", "| --- | --- | --- |"]
        for m in g.strengths:
            out.append(f"| {_md_escape(m.requirement.text)} | {_md_escape(_truncate(_evidence(m)))} | {m.score * 100:.0f} |")
    else:
        out.append("_No requirement is strongly covered yet._")
    out.append("")

    out += ["## Gaps and weak spots", ""]
    weak = g.gaps + g.partial
    if weak:
        for m in weak:
            tag = "Gap" if m.label == GAP else "Partial"
            extra = f" Missing: {', '.join(m.missing_skills)}." if m.missing_skills else ""
            out.append(f"- **{tag} ({m.score * 100:.0f})** [{m.requirement.category}] {_sentence(m.requirement.text)}{extra}")
    else:
        out.append("_No gaps found. Every requirement has solid evidence._")
    out.append("")

    out += ["## Missing keywords", ""]
    any_keywords = False
    if g.missing_required_skills:
        out.append(f"- **Required:** {_with_related(g.missing_required_skills, g.related_evidence)}")
        any_keywords = True
    if g.missing_preferred_skills:
        out.append(f"- **Nice to have:** {_with_related(g.missing_preferred_skills, g.related_evidence)}")
        any_keywords = True
    if g.other_missing_terms:
        out.append(f"- **Other terms from the posting not found in your resume:** {', '.join(g.other_missing_terms)}")
        any_keywords = True
    if g.listed_not_shown:
        out.append(
            f"- **Listed in Skills but not shown in any bullet:** {', '.join(g.listed_not_shown)} "
            "(recruiters trust skills more when a bullet shows them in use)"
        )
        any_keywords = True
    if not any_keywords:
        out.append("_No missing keywords detected._")
    out.append("")

    out += ["## Suggested edits", ""]
    out.append(
        "_Every suggestion only rephrases or emphasizes what is already in your resume. "
        "Review each one and keep it only if it is true._"
    )
    out.append("")
    if analysis.suggestions:
        for i, s in enumerate(analysis.suggestions, 1):
            out += _md_suggestion(i, s)
    else:
        out += ["_No suggestions: your bullets already mirror the posting well._", ""]

    if g.low_relevance_bullets and _mostly_irrelevant(analysis):
        out += ["## Bullets that may be less relevant for this role", ""]
        out.append(
            f"_{len(g.low_relevance_bullets)} of your bullets have little overlap with this posting. "
            "This role may be a stretch from your current experience, so weigh whether to apply or how much to retool._"
        )
        out.append("")
    elif g.low_relevance_bullets:
        out += ["## Bullets that may be less relevant for this role", ""]
        out.append("_Consider shortening or moving these lower for this application._")
        out.append("")
        for b in g.low_relevance_bullets:
            where = f" ({b.bullet.context})" if b.bullet.context else ""
            out.append(f"- {b.bullet.text}{_md_escape(where)}")
        out.append("")

    out += ["## Per-requirement breakdown", ""]
    out += ["| # | Requirement | Type | Score | Status | Closest resume line |", "| --- | --- | --- | --- | --- | --- |"]
    for i, m in enumerate(r.requirement_matches, 1):
        out.append(
            f"| {i} | {_md_escape(m.requirement.text)} | {m.requirement.category} | {m.score * 100:.0f} | "
            f"{_LABEL_TEXT[m.label]} | {_md_escape(_truncate(_evidence(m), 70))} |"
        )
    out.append("")
    return "\n".join(out)


def _md_suggestion(i: int, s: Suggestion) -> list[str]:
    title = f"### {i}. {s.requirement}" if s.requirement else f"### {i}. General polish"
    lines = [title, ""]
    if s.original:
        where = f" _({s.context})_" if s.context else ""
        lines += [f"**Current:**{where}", "", f"> {s.original}", ""]
    if s.has_rewrite:
        lines += ["**Suggested:**", "", f"> {s.suggested}", ""]
    label = "Note" if s.kind == MISSING else "Why"
    lines.append(f"**{label}:** {s.rationale}")
    if s.caution:
        lines += ["", f"**Check:** {s.caution}"]
    lines.append("")
    return lines


def to_text(analysis: Analysis, width: int = 88) -> str:
    r, g = analysis.result, analysis.gaps
    wrap = lambda text, indent="": textwrap.fill(text, width=width, initial_indent=indent, subsequent_indent=" " * len(indent))  # noqa: E731
    out: list[str] = []
    title = f"RESUME MATCH REPORT: {analysis.job_name}"
    out += [title, "=" * min(len(title), width)]
    if _source_name(analysis.resume.source):
        out.append(f"Resume: {_source_name(analysis.resume.source)}")
    if _source_name(analysis.job.source):
        out.append(f"Job description: {_source_name(analysis.job.source)}")
    out.append(f"Embedding backend: {r.backend}")
    out.append("")
    out += _headline(analysis)
    out.append(f"Requirements: {len(g.strengths)} strong, {len(g.partial)} partial, {len(g.gaps)} gaps")
    out.append("")

    out += ["MATCHED STRENGTHS", "-" * 17]
    for m in g.strengths or []:
        out.append(wrap(f"{m.requirement.text} ({m.score * 100:.0f})", "  + "))
        out.append(wrap(f"evidence: {_evidence(m)}", "      "))
    if not g.strengths:
        out.append("  (none yet)")
    out.append("")

    out += ["GAPS AND WEAK SPOTS", "-" * 19]
    for m in g.gaps + g.partial:
        tag = "GAP" if m.label == GAP else "PARTIAL"
        extra = f" Missing: {', '.join(m.missing_skills)}." if m.missing_skills else ""
        out.append(wrap(f"[{tag} {m.score * 100:.0f}] {_sentence(m.requirement.text)}{extra}", "  - "))
    if not (g.gaps or g.partial):
        out.append("  (none)")
    out.append("")

    out += ["MISSING KEYWORDS", "-" * 16]
    rows = [
        ("Required", [_with_related([s], g.related_evidence) for s in g.missing_required_skills]),
        ("Nice to have", [_with_related([s], g.related_evidence) for s in g.missing_preferred_skills]),
        ("Other terms", g.other_missing_terms),
        ("Listed but not shown in a bullet", g.listed_not_shown),
    ]
    printed = False
    for label, items in rows:
        if items:
            out.append(wrap(f"{label}: {', '.join(items)}", "  "))
            printed = True
    if not printed:
        out.append("  (none)")
    out.append("")

    out += ["SUGGESTED EDITS", "-" * 15]
    out.append(wrap("Suggestions only rephrase or emphasize what is already in your resume. Keep an edit only if it is true.", "  "))
    out.append("")
    for i, s in enumerate(analysis.suggestions, 1):
        out.append(wrap(s.requirement or "General polish", f"  {i}. "))
        if s.original:
            out.append(wrap(s.original, "     Current:   "))
        if s.has_rewrite:
            out.append(wrap(s.suggested, "     Suggested: "))
        out.append(wrap(s.rationale, "     Why:       "))
        if s.caution:
            out.append(wrap(s.caution, "     Check:     "))
        out.append("")
    if not analysis.suggestions:
        out += ["  (no suggestions)", ""]

    if g.low_relevance_bullets:
        out += ["POSSIBLY LESS RELEVANT BULLETS", "-" * 30]
        if _mostly_irrelevant(analysis):
            out.append(
                wrap(
                    f"{len(g.low_relevance_bullets)} of your bullets have little overlap with this posting. "
                    "This role may be a stretch from your current experience.",
                    "  ",
                )
            )
        else:
            for b in g.low_relevance_bullets:
                out.append(wrap(b.bullet.text, "  - "))
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def to_dict(analysis: Analysis) -> dict:
    r, g = analysis.result, analysis.gaps

    def match_dict(m: RequirementMatch) -> dict:
        return {
            "requirement": m.requirement.text,
            "category": m.requirement.category,
            "weight": m.requirement.weight,
            "score": round(m.score * 100, 1),
            "status": m.label,
            "similarity": round(m.similarity, 4),
            "best_evidence": m.best_bullet.text if m.best_bullet else None,
            "matched_skills": m.matched_skills,
            "missing_skills": m.missing_skills,
        }

    return {
        "job": analysis.job_name,
        "job_source": analysis.job.source,
        "resume_source": analysis.resume.source,
        "generated_at": analysis.created_at.isoformat(timespec="seconds"),
        "backend": r.backend,
        "overall_score": r.overall_score,
        "fit": r.fit_label,
        "requirement_score": r.requirement_score,
        "keyword_score": r.keyword_score,
        "requirements": [match_dict(m) for m in r.requirement_matches],
        "missing_required_skills": g.missing_required_skills,
        "missing_preferred_skills": g.missing_preferred_skills,
        "other_missing_terms": g.other_missing_terms,
        "related_evidence": g.related_evidence,
        "listed_not_shown": g.listed_not_shown,
        "low_relevance_bullets": [b.bullet.text for b in g.low_relevance_bullets],
        "suggestions": [
            {
                "kind": s.kind,
                "requirement": s.requirement,
                "original": s.original,
                "suggested": s.suggested if s.has_rewrite else None,
                "rationale": s.rationale,
                "caution": s.caution,
            }
            for s in analysis.suggestions
        ],
    }


def to_json(analysis: Analysis, indent: int = 2) -> str:
    return json.dumps(to_dict(analysis), indent=indent, ensure_ascii=False)


def render(analysis: Analysis, fmt: str = "markdown") -> str:
    if fmt == "markdown":
        return to_markdown(analysis)
    if fmt == "text":
        return to_text(analysis)
    if fmt == "json":
        return to_json(analysis)
    raise ValueError(f"Unknown format '{fmt}'. Choose one of: {', '.join(FORMATS)}")


def comparison_table(analyses: Iterable[Analysis], fmt: str = "markdown") -> str:
    """Rank several job descriptions against the same resume."""
    ranked = sorted(analyses, key=lambda a: -a.score)
    if fmt == "json":
        return json.dumps(
            [{"job": a.job_name, "source": a.job.source, "score": a.score, "fit": a.result.fit_label} for a in ranked],
            indent=2,
        )
    if fmt == "text":
        lines = ["JOB RANKING", "-" * 11]
        for i, a in enumerate(ranked, 1):
            lines.append(f"{i:>2}. {a.score:5.0f}  {score_bar(a.score, 10)}  {a.result.fit_label:<11}  {a.job_name}")
        return "\n".join(lines) + "\n"
    lines = ["# Job ranking", "", "| Rank | Job | Score | Fit | Gaps | Missing required skills |", "| --- | --- | --- | --- | --- | --- |"]
    for i, a in enumerate(ranked, 1):
        missing = ", ".join(a.gaps.missing_required_skills) or "none"
        lines.append(
            f"| {i} | {_md_escape(a.job_name)} | {a.score:.0f} | {a.result.fit_label} | {len(a.gaps.gaps)} | {_md_escape(missing)} |"
        )
    return "\n".join(lines) + "\n"
