import json

from click.testing import CliRunner

from resume_matcher.analysis import analyze
from resume_matcher.cli import main
from resume_matcher.report import comparison_table, render
from resume_matcher.report.report_builder import score_bar


def test_score_bar():
    assert score_bar(50, 10) == "[#####-----]"
    assert score_bar(0, 4) == "[----]"


def test_markdown_report_sections(resume, job, embedder):
    md = render(analyze(resume, job, embedder=embedder), "markdown")
    for heading in ("## Overall match", "## Matched strengths", "## Gaps and weak spots", "## Missing keywords", "## Suggested edits", "## Per-requirement breakdown"):
        assert heading in md


def test_text_and_json_reports(resume, job, embedder):
    a = analyze(resume, job, embedder=embedder)
    assert "RESUME MATCH REPORT" in render(a, "text")
    data = json.loads(render(a, "json"))
    assert data["overall_score"] == a.score
    assert len(data["requirements"]) == len(job.requirements)


def test_reports_have_no_long_dashes(resume, job, weak_job, embedder):
    for posting in (job, weak_job):
        a = analyze(resume, posting, embedder=embedder)
        for fmt in ("markdown", "text", "json"):
            assert chr(0x2014) not in render(a, fmt)


def test_comparison_table(resume, job, weak_job, embedder):
    table = comparison_table([analyze(resume, weak_job, embedder=embedder), analyze(resume, job, embedder=embedder)])
    lines = [l for l in table.splitlines() if l.startswith("| 1 ")]
    assert "Senior Backend Engineer" in lines[0]


def test_cli_match_single(data_dir):
    result = CliRunner().invoke(main, ["match", str(data_dir / "sample_resume.md"), str(data_dir / "sample_job.txt"), "--backend", "tfidf"])
    assert result.exit_code == 0, result.output
    assert "# Resume match report" in result.output


def test_cli_match_many_to_dir_and_history(data_dir, tmp_path):
    out = tmp_path / "reports"
    hist = tmp_path / "history.jsonl"
    result = CliRunner().invoke(
        main,
        [
            "match", str(data_dir / "sample_resume.pdf"),
            str(data_dir / "sample_job.txt"), str(data_dir / "sample_job_weak_fit.txt"),
            "--backend", "tfidf", "--output-dir", str(out), "-f", "text", "--save", "--history-file", str(hist),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (out / "ranking.txt").exists()
    assert len(list(out.glob("*.txt"))) == 3
    assert len(hist.read_text().strip().splitlines()) == 2

    shown = CliRunner().invoke(main, ["history", "--history-file", str(hist)])
    assert "Senior Backend Engineer" in shown.output


def test_cli_stdin_and_output_file(data_dir, tmp_path):
    target = tmp_path / "r.json"
    job_text = (data_dir / "sample_job.txt").read_text()
    result = CliRunner().invoke(
        main, ["match", str(data_dir / "sample_resume.md"), "-", "-f", "json", "-o", str(target), "--backend", "tfidf"], input=job_text
    )
    assert result.exit_code == 0, result.output
    assert json.loads(target.read_text())["overall_score"] > 0


def test_cli_errors(data_dir):
    runner = CliRunner()
    missing = runner.invoke(main, ["match", str(data_dir / "sample_resume.md"), "nope.txt", "--backend", "tfidf"])
    assert missing.exit_code != 0
    both = runner.invoke(
        main,
        ["match", str(data_dir / "sample_resume.md"), str(data_dir / "sample_job.txt"), str(data_dir / "sample_job.txt"), "-o", "x.md"],
    )
    assert both.exit_code != 0


def test_cli_parse_commands(data_dir):
    runner = CliRunner()
    r = runner.invoke(main, ["parse-resume", str(data_dir / "sample_resume.md")])
    assert json.loads(r.output)["sections"]
    j = runner.invoke(main, ["parse-job", str(data_dir / "sample_job.txt")])
    assert json.loads(j.output)["title"].startswith("Senior Backend")
