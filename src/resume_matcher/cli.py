"""Command line interface: `resume-matcher match RESUME JOB [JOB ...]`."""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import Optional

import click

from . import __version__
from . import history as history_store
from .analysis import Analysis, analyze
from .matcher.embeddings import BACKENDS, DEFAULT_MODEL, get_embedder
from .parser.job_parser import parse_job_text
from .parser.resume_parser import parse_resume
from .parser.text_extract import extract_text
from .report.report_builder import FORMATS, comparison_table, render

_EXTENSIONS = {"markdown": ".md", "text": ".txt", "json": ".json"}


def _load_job(spec: str):
    if spec == "-":
        return parse_job_text(sys.stdin.read(), source=None)
    path = Path(spec)
    return parse_job_text(extract_text(path), source=str(path))


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60] or "job"


@click.group()
@click.version_option(__version__, prog_name="resume-matcher")
@click.option("-v", "--verbose", is_flag=True, help="Show progress and backend details.")
def main(verbose: bool) -> None:
    """Score how well a resume fits job descriptions and suggest honest edits."""
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING, format="%(message)s")


@main.command()
@click.argument("resume", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("jobs", nargs=-1, required=True)
@click.option("-f", "--format", "fmt", type=click.Choice(FORMATS), default="markdown", show_default=True, help="Report format.")
@click.option("-o", "--output", type=click.Path(dir_okay=False, path_type=Path), help="Write the report to a file (single job).")
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    help="Write one report per job into this directory (useful with several jobs).",
)
@click.option("--backend", type=click.Choice(BACKENDS), default="auto", show_default=True, help="Embedding backend.")
@click.option("--model", default=DEFAULT_MODEL, show_default=True, help="sentence-transformers model name or path.")
@click.option("--max-suggestions", type=click.IntRange(0, 50), default=8, show_default=True)
@click.option("--save", is_flag=True, help="Append the score to your local history.")
@click.option("--history-file", type=click.Path(dir_okay=False, path_type=Path), help="History file (default ~/.resume_matcher/history.jsonl).")
def match(
    resume: Path,
    jobs: tuple[str, ...],
    fmt: str,
    output: Optional[Path],
    output_dir: Optional[Path],
    backend: str,
    model: str,
    max_suggestions: int,
    save: bool,
    history_file: Optional[Path],
) -> None:
    """Compare RESUME against one or more JOBS (files, or '-' to read one job from stdin)."""
    if output and len(jobs) > 1:
        raise click.UsageError("--output works with a single job. Use --output-dir for several jobs.")
    if jobs.count("-") > 1:
        raise click.UsageError("Only one job can be read from stdin.")
    for spec in jobs:
        if spec != "-" and not Path(spec).is_file():
            raise click.BadParameter(f"File not found: {spec}", param_hint="JOBS")

    try:
        parsed_resume = parse_resume(resume)
        embedder = get_embedder(backend, model)
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc

    analyses: list[Analysis] = []
    for spec in jobs:
        try:
            job = _load_job(spec)
        except Exception as exc:
            raise click.ClickException(f"Could not read job description {spec}: {exc}") from exc
        if not job.requirements:
            click.echo(f"Warning: no requirements found in {spec}; the score will not be meaningful.", err=True)
        analyses.append(analyze(parsed_resume, job, embedder=embedder, max_suggestions=max_suggestions))

    if save:
        for a in analyses:
            path = history_store.record(a, history_file)
        click.echo(f"Saved {len(analyses)} result(s) to {path}", err=True)

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        used = set()
        for a in analyses:
            name = _slug(a.job_name)
            while name in used:
                name += "-2"
            used.add(name)
            target = output_dir / f"{name}{_EXTENSIONS[fmt]}"
            target.write_text(render(a, fmt), encoding="utf-8")
            click.echo(f"{a.score:5.0f}  {a.job_name}  ->  {target}", err=True)
        if len(analyses) > 1:
            summary = output_dir / f"ranking{_EXTENSIONS[fmt]}"
            summary.write_text(comparison_table(analyses, fmt), encoding="utf-8")
            click.echo(f"Ranking written to {summary}", err=True)
        return

    if len(analyses) == 1:
        text = render(analyses[0], fmt)
    elif fmt == "json":
        import json

        from .report.report_builder import to_dict

        text = json.dumps([to_dict(a) for a in sorted(analyses, key=lambda a: -a.score)], indent=2, ensure_ascii=False)
    else:
        separator = "\n\n---\n\n" if fmt == "markdown" else "\n" + "=" * 88 + "\n\n"
        text = comparison_table(analyses, fmt) + separator + separator.join(
            render(a, fmt) for a in sorted(analyses, key=lambda a: -a.score)
        )

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
        click.echo(f"Report written to {output} (score {analyses[0].score:.0f}/100)", err=True)
    else:
        click.echo(text)


@main.command("parse-resume")
@click.argument("resume", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def parse_resume_cmd(resume: Path) -> None:
    """Print the parsed resume as JSON (sections, bullets, detected skills)."""
    try:
        click.echo(parse_resume(resume).to_json())
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc


@main.command("parse-job")
@click.argument("job")
def parse_job_cmd(job: str) -> None:
    """Print the parsed job description as JSON (requirements and skills). Use '-' for stdin."""
    try:
        click.echo(_load_job(job).to_json())
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc


@main.command()
@click.option("--history-file", type=click.Path(dir_okay=False, path_type=Path), help="History file to read.")
@click.option("-n", "--limit", type=int, default=20, show_default=True, help="Show the most recent N entries.")
def history(history_file: Optional[Path], limit: int) -> None:
    """Show saved scores across applications."""
    entries = history_store.load(history_file)
    if not entries:
        click.echo("No saved results yet. Run `resume-matcher match ... --save` to start a history.")
        return
    click.echo(f"{'Date':<17} {'Score':>5}  {'Fit':<11}  Job")
    for e in entries[-limit:]:
        date = e.get("timestamp", "")[:16].replace("T", " ")
        click.echo(f"{date:<17} {e.get('score', 0):>5.0f}  {e.get('fit', ''):<11}  {e.get('job', '')}")


if __name__ == "__main__":
    main()
