from resume_matcher.analysis import analyze
from resume_matcher.skills import extract_skills
from resume_matcher.suggestions.rewrite_suggester import (
    MISSING,
    align_terminology,
    gerund_to_past,
    key_phrase,
    missing_terms,
    strengthen_opener,
)


def test_gap_analysis_flags_genuinely_missing_skills(resume, job, embedder):
    a = analyze(resume, job, embedder=embedder)
    assert "Kafka" in a.gaps.missing_preferred_skills
    assert "Terraform" in a.gaps.missing_preferred_skills
    # AWS satisfies "GCP or AWS", so GCP is not reported missing.
    assert "GCP" not in a.gaps.missing_preferred_skills
    assert "Kubernetes" in a.gaps.listed_not_shown
    assert a.gaps.related_evidence.get("ETL") and "Airflow" in a.gaps.related_evidence["ETL"]
    gap_texts = {m.requirement.text for m in a.gaps.gaps}
    assert "Familiarity with Terraform and infrastructure as code" in gap_texts


def test_suggestions_never_introduce_new_skills(resume, job, weak_job, embedder):
    for posting in (job, weak_job):
        a = analyze(resume, posting, embedder=embedder, max_suggestions=20)
        for s in a.suggestions:
            if s.has_rewrite:
                assert extract_skills(s.suggested) <= resume.skills, s


def test_missing_skills_get_notes_not_rewrites(resume, weak_job, embedder):
    a = analyze(resume, weak_job, embedder=embedder)
    missing = [s for s in a.suggestions if s.kind == MISSING]
    assert missing
    assert all(s.suggested is None for s in missing)


def test_expected_rewrites_on_sample(resume, job, embedder):
    a = analyze(resume, job, embedder=embedder)
    rewrites = {s.original: s.suggested for s in a.suggestions if s.has_rewrite}
    assert rewrites["Designed a Postgres schema and query layer that cut report generation time from 40s to 6s"] == (
        "Designed a PostgreSQL schema and query layer that cut report generation time from 40s to 6s"
    )
    assert rewrites["Responsible for migrating nightly batch jobs to Airflow, reducing failed runs by 70%"] == (
        "Migrated nightly batch jobs to Airflow, reducing failed runs by 70%"
    )


def test_gerund_to_past():
    cases = {"migrating": "migrated", "planning": "planned", "building": "built", "Leading": "Led", "deploying": "deployed"}
    for gerund, past in cases.items():
        assert gerund_to_past(gerund) == past


def test_strengthen_opener():
    assert strengthen_opener("Responsible for managing a team of 4") == "Managed a team of 4"
    assert strengthen_opener("Responsible for the billing service") == "Owned the billing service"
    assert strengthen_opener("Built the billing service") is None
    # "Helped" is left alone: turning it into "Built" could overstate the work.
    assert strengthen_opener("Helped build the billing service") is None


def test_key_phrase_strips_boilerplate():
    assert key_phrase("5+ years of strong experience with Python and SQL") == "Python and SQL"
    assert key_phrase("Ability to communicate clearly") == "communicate clearly"


def test_align_terminology(job):
    text, swaps = align_terminology("Tuned Postgres queries", job)
    assert text == "Tuned PostgreSQL queries"
    assert swaps == [("Postgres", "PostgreSQL")]


def test_missing_terms():
    assert missing_terms("Mentor engineers and lead design reviews", "Mentored two engineers") == ["lead", "design", "reviews"]
