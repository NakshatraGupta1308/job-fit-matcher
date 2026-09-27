from resume_matcher.parser.job_parser import PREFERRED, REQUIRED, RESPONSIBILITY, parse_job_text


def test_title_and_categories(job):
    assert job.title == "Senior Backend Engineer, Data Platform"
    cats = {r.category for r in job.requirements}
    assert cats == {REQUIRED, RESPONSIBILITY, PREFERRED}


def test_benefits_and_about_sections_are_ignored(job):
    texts = " ".join(r.text for r in job.requirements).lower()
    assert "dental" not in texts
    assert "60-person" not in texts
    assert "equal opportunity" not in texts


def test_skill_lists(job):
    assert {"Python", "SQL", "Docker", "Kubernetes", "Airflow"} <= set(job.required_skills)
    assert "Kafka" in job.preferred_skills
    assert not set(job.required_skills) & set(job.preferred_skills)


def test_years_extracted(job):
    assert any(r.years == 5 for r in job.requirements)


def test_inline_preferred_hint():
    job = parse_job_text("Requirements:\n- Python\n- Experience with Go is a plus\n")
    cats = {r.text: r.category for r in job.requirements}
    assert cats["Experience with Go is a plus"] == PREFERRED


def test_job_without_headers_uses_sentences():
    job = parse_job_text(
        "We need a data analyst who knows SQL and Tableau. You will build dashboards for the sales team. "
        "Experience with Python is required."
    )
    assert len(job.requirements) == 3
    assert {"SQL", "Tableau", "Python"} <= set(job.required_skills)


def test_colon_headers_and_loose_matching():
    job = parse_job_text("Acme\n\nWhat you'll be doing:\n- Ship product features weekly\n- Fix bugs fast\n\nNice-to-have skills:\n- Rust\n")
    by_text = {r.text: r.category for r in job.requirements}
    assert by_text["Ship product features weekly"] == RESPONSIBILITY
    assert by_text["Rust"] == PREFERRED
