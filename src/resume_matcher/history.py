"""Local score history, stored as JSON lines so it is easy to inspect or delete."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from .analysis import Analysis


def default_history_path() -> Path:
    base = os.environ.get("RESUME_MATCHER_HOME")
    root = Path(base) if base else Path.home() / ".resume_matcher"
    return root / "history.jsonl"


def record(analysis: Analysis, path: Optional[Path] = None) -> Path:
    path = Path(path) if path else default_history_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": analysis.created_at.isoformat(timespec="seconds"),
        "resume": analysis.resume.source,
        "job": analysis.job_name,
        "job_source": analysis.job.source,
        "score": analysis.score,
        "requirement_score": analysis.result.requirement_score,
        "keyword_score": analysis.result.keyword_score,
        "fit": analysis.result.fit_label,
        "backend": analysis.result.backend,
        "missing_required_skills": analysis.gaps.missing_required_skills,
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return path


def load(path: Optional[Path] = None) -> list[dict]:
    path = Path(path) if path else default_history_path()
    if not path.exists():
        return []
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries
