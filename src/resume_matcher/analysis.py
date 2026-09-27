"""End-to-end pipeline: parse, score, find gaps, and suggest edits."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

from .matcher.embeddings import Embedder, get_embedder
from .matcher.scorer import MatchResult, score_match
from .parser.job_parser import JobDescription, parse_job
from .parser.resume_parser import Resume, parse_resume
from .suggestions.gap_analysis import GapReport, analyze_gaps
from .suggestions.rewrite_suggester import Suggestion, suggest_rewrites


@dataclass
class Analysis:
    resume: Resume
    job: JobDescription
    result: MatchResult
    gaps: GapReport
    suggestions: list[Suggestion]
    created_at: datetime = field(default_factory=datetime.now)

    @property
    def score(self) -> float:
        return self.result.overall_score

    @property
    def job_name(self) -> str:
        if self.job.title:
            return self.job.title
        if self.job.source:
            return Path(self.job.source).stem
        return "Job description"


def analyze(
    resume: Union[Resume, str, Path],
    job: Union[JobDescription, str, Path],
    embedder: Optional[Embedder] = None,
    backend: str = "auto",
    max_suggestions: int = 8,
) -> Analysis:
    """Run the full analysis. `resume` and `job` may be parsed objects, file paths, or raw text."""
    if not isinstance(resume, Resume):
        resume = parse_resume(resume)
    if not isinstance(job, JobDescription):
        job = parse_job(job)
    if embedder is None:
        embedder = get_embedder(backend)
    result = score_match(resume, job, embedder)
    gaps = analyze_gaps(resume, job, result)
    suggestions = suggest_rewrites(resume, job, result, gaps, embedder, limit=max_suggestions)
    return Analysis(resume=resume, job=job, result=result, gaps=gaps, suggestions=suggestions)
