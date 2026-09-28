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


@main.command()
@click.argument("resume", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("job")
@click.option(
    "-d", "--output-dir", type=click.Path(file_okay=False, path_type=Path), default=Path("tailored"), show_default=True,
    help="Where to write the tailored resume and gap notes.",
)
@click.option("--backend", type=click.Choice(BACKENDS), default="auto", show_default=True, help="Embedding backend.")
@click.option("--model", default=DEFAULT_MODEL, show_default=True, help="sentence-transformers model name or path.")
@click.option("--max-gaps", type=click.IntRange(1, 30), default=10, show_default=True, help="How many gaps to walk through.")
def tailor(resume: Path, job: str, output_dir: Path, backend: str, model: str, max_gaps: int) -> None:
    """Walk through edits and gaps for one JOB, then write a tailored resume.

    For each gap you say whether you have real experience. If you do, a bullet is drafted
    from your own answers. If you do not, you get a cover letter line and a learning plan
    instead, and nothing is added to the resume.
    """
    from .tailoring import (
        PROJECTS_ANCHOR,
        Addition,
        GapAnswer,
        TailoringPlan,
        anchor_options,
        build_gap_prompts,
        bullet_index,
        check_draft,
        cover_letter_line,
        draft_bullet,
        learning_plan,
        notes_markdown,
        pdf_available,
        rescore,
        to_markdown,
        to_pdf,
        to_text,
    )

    if job == "-":
        raise click.UsageError("tailor asks questions on stdin, so the job description must be a file.")
    if not Path(job).is_file():
        raise click.BadParameter(f"File not found: {job}", param_hint="JOB")
    try:
        embedder = get_embedder(backend, model)
        before = analyze(parse_resume(resume), _load_job(job), embedder=embedder)
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc
    res, plan, notes = before.resume, TailoringPlan(), []

    click.echo(f"\n{before.job_name}: {before.score:.0f}/100 ({before.result.fit_label})\n")

    rewrites = [s for s in before.suggestions if s.has_rewrite]
    if rewrites:
        click.secho("Step 1 of 2: suggested edits to existing bullets", bold=True)
    for n, s in enumerate(rewrites, 1):
        click.echo(f"\n[{n}/{len(rewrites)}] {s.requirement or 'General polish'}")
        click.echo(f"  Current:   {s.original}")
        click.echo(f"  Suggested: {s.suggested}")
        if s.caution:
            click.echo(f"  Check:     {s.caution}")
        choice = click.prompt("  Apply? (y)es, (n)o, (e)dit", type=click.Choice(["y", "n", "e"]), default="n", show_choices=False)
        text = s.suggested if choice == "y" else click.prompt("  Your version", default=s.suggested) if choice == "e" else None
        idx = bullet_index(res, s.original)
        if text and idx is not None:
            plan.edits[idx] = text

    prompts = build_gap_prompts(before, limit=max_gaps)
    anchors = anchor_options(res)
    if prompts:
        click.secho("\nStep 2 of 2: gaps", bold=True)
        click.echo("Answer honestly. A bullet is only drafted from what you type.")
    for n, p in enumerate(prompts, 1):
        click.echo(f"\n[{n}/{len(prompts)}] {p.requirement}")
        click.echo(f"  {p.question}")
        choice = click.prompt("  (y)es, (n)o, (s)kip", type=click.Choice(["y", "n", "s"]), default="s", show_choices=False)
        if choice == "s":
            continue
        if choice == "n":
            notes.append((p, cover_letter_line(p, res), learning_plan(p)))
            click.echo("  Noted. You'll get a cover letter line and a learning plan in gap_notes.md.")
            continue
        answer = GapAnswer(True)
        answer.what = click.prompt("  What did you do? (for example: built a pipeline that streams order events)")
        answer.tools = click.prompt("  Tools or technologies used (Enter to skip)", default=", ".join(p.tools), show_default=bool(p.tools))
        answer.result = click.prompt("  Result or scale, ideally with a number (Enter to skip)", default="", show_default=False)
        for i, (_, label) in enumerate(anchors, 1):
            click.echo(f"    {i}. {label}")
        default_pos = next((i for i, (key, _) in enumerate(anchors, 1) if key == p.suggested_anchor), len(anchors))
        pos = click.prompt("  Where does it go?", type=click.IntRange(1, len(anchors)), default=default_pos)
        answer.anchor = anchors[pos - 1][0]
        draft = draft_bullet(answer, before.job, res)
        click.echo(f"  Draft: {draft.text}")
        for tip in draft.tips:
            click.echo(f"  Tip:   {tip}")
        choice = click.prompt("  Add it? (y)es, (n)o, (e)dit", type=click.Choice(["y", "n", "e"]), default="y", show_choices=False)
        text = draft.text if choice == "y" else click.prompt("  Your version", default=draft.text) if choice == "e" else None
        if not text:
            continue
        problems = check_draft(text, answer, res)
        if problems and not click.confirm("  " + " ".join(problems) + " Keep it anyway?", default=False):
            continue
        plan.additions.append(Addition(answer.anchor or PROJECTS_ANCHOR, text))
        new = [t for t in p.tools if t.lower() in text.lower()]
        if new and click.confirm(f"  Also add {', '.join(new)} to your Skills section?", default=True):
            plan.new_skills += [s for s in new if s not in plan.new_skills]

    after = rescore(res, plan, before.job, embedder)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "tailored_resume.md").write_text(to_markdown(res, plan), encoding="utf-8")
    (output_dir / "tailored_resume.txt").write_text(to_text(res, plan), encoding="utf-8")
    written = ["tailored_resume.md", "tailored_resume.txt"]
    if pdf_available():
        (output_dir / "tailored_resume.pdf").write_bytes(to_pdf(res, plan))
        written.append("tailored_resume.pdf")
    (output_dir / "gap_notes.md").write_text(notes_markdown(before.job_name, notes), encoding="utf-8")
    written.append("gap_notes.md")

    delta = after.score - before.score
    click.secho(f"\nScore: {before.score:.0f} -> {after.score:.0f} ({delta:+.0f})", bold=True)
    click.echo(f"{plan.change_count()} change(s). Wrote {', '.join(written)} to {output_dir}/")


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
