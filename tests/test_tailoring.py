import pytest
from click.testing import CliRunner

from resume_matcher.analysis import analyze
from resume_matcher.cli import main
from resume_matcher.parser import parse_resume
from resume_matcher.skills import extract_skills
from resume_matcher.tailoring import (
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
    tailored_resume,
    to_markdown,
    to_pdf,
    to_text,
)
from resume_matcher.tailoring.bullet_writer import past_to_gerund, start_with_action


@pytest.fixture()
def analysis(resume, job, embedder):
    return analyze(resume, job, embedder=embedder)


# Gap questions

def test_prompts_cover_real_gaps_only(analysis):
    prompts = build_gap_prompts(analysis)
    targets = [p.target for p in prompts]
    assert "Kafka" in targets and "Terraform" in targets
    # AWS already satisfies "GCP or AWS", and "5+ years" is a timeline question, not a bullet.
    assert not any("GCP" in p.requirement for p in prompts)
    assert not any("5+ years" in p.requirement for p in prompts)


def test_tool_questions_and_concept_questions(analysis):
    by_target = {p.target: p for p in build_gap_prompts(analysis)}
    assert "Have you used it" in by_target["Kafka"].question
    concept = next(p for p in by_target.values() if not p.tools)
    assert concept.question.startswith("The posting asks for \"")


def test_anchor_options(resume):
    keys = [k for k, _ in anchor_options(resume)]
    assert keys[-1] == PROJECTS_ANCHOR
    assert any("Brightline" in k for k in keys)


# Drafting bullets from answers

@pytest.mark.parametrize(
    "what,expected",
    [
        ("built a pipeline", "Built a pipeline"),
        ("I build pipelines", "Built pipelines"),
        ("building pipelines", "Built pipelines"),
        ("designs APIs", "Designed APIs"),
        ("Led a team of 3", "Led a team of 3"),
        ("plan releases", "Planned releases"),
        ("the ingestion layer", "The ingestion layer"),
    ],
)
def test_start_with_action(what, expected):
    assert start_with_action(what) == expected


def test_past_to_gerund():
    assert [past_to_gerund(w) for w in ["reduced", "cut", "led", "copied", "improving"]] == [
        "reducing", "cutting", "leading", "copying", "improving",
    ]


def test_draft_uses_only_the_answer(job, resume):
    answer = GapAnswer(True, "built an event pipeline for order updates", "kafka", "processing 2M events per day")
    draft = draft_bullet(answer, job, resume)
    assert draft.text == "Built an event pipeline for order updates using Kafka, processing 2M events per day"
    assert draft.ok and not draft.tips


def test_draft_result_phrasing():
    base = dict(has_experience=True, what="built ingestion jobs", tools="Airflow")
    assert draft_bullet(GapAnswer(**base, result="about 1M events per day")).text.endswith(", handling about 1M events per day")
    assert draft_bullet(GapAnswer(**base, result="20% lower costs")).text.endswith(", resulting in 20% lower costs")
    assert draft_bullet(GapAnswer(**base, result="reduced costs by 20%")).text.endswith(", reducing costs by 20%")


def test_draft_tips_and_empty_answer():
    assert not draft_bullet(GapAnswer(True, "")).ok
    tips = draft_bullet(GapAnswer(True, "helped with dashboards")).tips
    assert any("number" in t for t in tips) and any("helped" in t for t in tips)


def test_check_draft_blocks_skills_not_in_answers(resume):
    answer = GapAnswer(True, "built a pipeline", "Kafka")
    assert check_draft("Built a pipeline using Kafka", answer, resume) == []
    assert check_draft("Built a pipeline using Kafka and Terraform", answer, resume) == [
        "The draft mentions Terraform, which is not in your answers."
    ]


def test_drafts_never_add_skills(job, resume):
    answers = [
        GapAnswer(True, "set up streaming jobs", "Kafka, Python", "cut lag from hours to seconds"),
        GapAnswer(True, "maintained postgres backups", "", ""),
        GapAnswer(True, "wrote infrastructure modules", "terraform", "for 3 environments"),
    ]
    for a in answers:
        text = draft_bullet(a, job, resume).text
        assert extract_skills(text) <= extract_skills(" ".join([a.what, a.tools, a.result])) | resume.skills


# Honest handling of gaps the user cannot fill

def test_cover_letter_line_uses_related_experience(analysis, resume):
    prompts = build_gap_prompts(analysis)
    pipelines = next(p for p in prompts if p.related)
    line = cover_letter_line(pipelines, resume)
    assert "Airflow" in line and "migrated nightly batch jobs" in line
    kafka = next(p for p in prompts if p.target == "Kafka")
    assert "[Add one true sentence" in cover_letter_line(kafka, resume)


def test_learning_plan_dedupes_concepts(analysis):
    terraform = next(p for p in build_gap_prompts(analysis) if p.target == "Terraform")
    plans = learning_plan(terraform)
    assert [p.skill for p in plans] == ["Terraform"]
    assert any(step.startswith("Project idea:") for step in plans[0].steps)


def test_notes_markdown(analysis, resume):
    p = build_gap_prompts(analysis)[0]
    md = notes_markdown(analysis.job_name, [(p, cover_letter_line(p, resume), learning_plan(p))])
    assert md.startswith("# Gap notes") and p.requirement in md
    assert "No open gaps" in notes_markdown("Job", [])


# Tailored resume

@pytest.mark.parametrize("name", ["sample_resume.md", "sample_resume.pdf"])
def test_empty_plan_round_trips_exactly(data_dir, name):
    resume = parse_resume(data_dir / name)
    again = tailored_resume(resume, TailoringPlan())
    assert [(b.section, b.context, b.text) for b in again.bullets] == [(b.section, b.context, b.text) for b in resume.bullets]


def _plan(analysis):
    resume = analysis.resume
    plan = TailoringPlan()
    for s in analysis.suggestions:
        if s.has_rewrite:
            plan.edits[bullet_index(resume, s.original)] = s.suggested
    anchor = next(k for k, _ in anchor_options(resume) if "Brightline" in k)
    plan.additions.append(Addition(anchor, "Built an event pipeline for order updates using Kafka, processing 2M events per day"))
    plan.additions.append(Addition(PROJECTS_ANCHOR, "Wrote Terraform modules for a staging environment"))
    plan.new_skills = ["Kafka", "Terraform"]
    return plan


def test_plan_applies_edits_additions_and_skills(analysis):
    resume, plan = analysis.resume, _plan(analysis)
    md = to_markdown(resume, plan)
    assert "Migrated nightly batch jobs to Airflow" in md
    assert "Responsible for migrating" not in md
    brightline = md.index("Brightline")
    pinecrest = md.index("Pinecrest")
    assert brightline < md.index("Kafka, processing 2M") < pinecrest  # under the right job
    assert md.index("## Projects") < md.index("Terraform modules") < md.index("## Skills")
    assert "Linux, Kafka, Terraform" in md
    assert "KAFKA" not in to_text(resume, plan) and "EXPERIENCE" in to_text(resume, plan)


def test_rescore_goes_up_and_gaps_close(analysis, embedder):
    after = rescore(analysis.resume, _plan(analysis), analysis.job, embedder)
    assert after.score > analysis.score
    assert "Kafka" not in after.gaps.missing_preferred_skills


def test_new_section_created_when_missing(embedder):
    from resume_matcher.parser import parse_resume_text

    resume = parse_resume_text("Sam Lee\n\nEXPERIENCE\nAnalyst, Foo\n2020 - 2023\n- Built reports in Excel\n")
    plan = TailoringPlan(additions=[Addition(PROJECTS_ANCHOR, "Built a Tableau dashboard for city bike data")], new_skills=["Tableau"])
    md = to_markdown(resume, plan)
    assert "## Projects" in md and "## Skills" in md and md.rstrip().endswith("Tableau")


@pytest.mark.skipif(not pdf_available(), reason="reportlab not installed")
def test_pdf_export(analysis, tmp_path):
    pdf = to_pdf(analysis.resume, _plan(analysis))
    assert pdf.startswith(b"%PDF")
    path = tmp_path / "t.pdf"
    path.write_bytes(pdf)
    reparsed = parse_resume(path)
    assert any("Kafka, processing 2M" in b.text for b in reparsed.bullets)


# CLI

def test_cli_tailor(data_dir, tmp_path):
    answers = "\n".join(
        [
            "y",  # apply first edit
            "n",  # skip second edit
            "s",  # skip the data pipelines gap
            "n",  # no Terraform experience
            "y",  # have used Kafka
            "built an event pipeline for order updates",
            "",  # tools: accept default (Kafka)
            "processing 2M events per day",
            "",  # anchor: accept default
            "y",  # add the bullet
            "y",  # add Kafka to skills
        ]
    ) + "\n"
    out = tmp_path / "t"
    result = CliRunner().invoke(
        main, ["tailor", str(data_dir / "sample_resume.md"), str(data_dir / "sample_job.txt"), "--backend", "tfidf", "-d", str(out)], input=answers
    )
    assert result.exit_code == 0, result.output
    assert "Score:" in result.output
    md = (out / "tailored_resume.md").read_text()
    assert "Migrated nightly batch jobs" in md
    assert "using Kafka, processing 2M events per day" in md
    notes = (out / "gap_notes.md").read_text()
    assert "Terraform" in notes and "Kafka" not in notes


def test_cli_tailor_rejects_stdin(data_dir):
    result = CliRunner().invoke(main, ["tailor", str(data_dir / "sample_resume.md"), "-"])
    assert result.exit_code != 0
