from datetime import date

from resume_matcher.parser.resume_parser import detect_section, estimate_years, parse_resume, parse_resume_text


def test_sections_detected(resume):
    assert {"summary", "experience", "projects", "skills", "education"} <= set(resume.sections)


def test_bullets_are_tagged_by_section_with_context(resume):
    exp = [b for b in resume.bullets if b.section == "experience"]
    assert len(exp) == 9
    assert exp[0].text.startswith("Built and maintained 12 Flask microservices")
    assert "Brightline Analytics" in exp[0].context
    assert "Pinecrest Health" in exp[-1].context


def test_listed_versus_evidenced_skills(resume):
    assert "Kubernetes" in resume.listed_skills
    assert "Kubernetes" not in resume.evidenced_skills
    assert "Airflow" in resume.evidenced_skills


def test_pdf_and_markdown_parse_the_same(data_dir):
    pdf = parse_resume(data_dir / "sample_resume.pdf")
    md = parse_resume(data_dir / "sample_resume.md")
    assert [(b.section, b.text) for b in pdf.bullets] == [(b.section, b.text) for b in md.bullets]


def test_header_variants():
    assert detect_section("WORK EXPERIENCE") == "experience"
    assert detect_section("## Technical Skills:") == "skills"
    assert detect_section("**Education**") == "education"
    assert detect_section("Led experience redesign for checkout.") is None


def test_plain_text_resume_without_bullet_glyphs():
    text = """Sam Lee
sam@example.com

EXPERIENCE
Data Analyst, Acme Retail
2020 - 2023
Built weekly sales dashboards in Tableau for 30 store managers. Automated inventory reports with Python and SQL.

SKILLS
Python, SQL, Tableau, Excel
"""
    resume = parse_resume_text(text)
    texts = [b.text for b in resume.bullets if b.section == "experience"]
    assert texts == [
        "Built weekly sales dashboards in Tableau for 30 store managers.",
        "Automated inventory reports with Python and SQL.",
    ]
    assert resume.bullets[0].context.startswith("Data Analyst, Acme Retail")
    assert {"Python", "SQL", "Tableau", "Excel"} <= resume.skills


def test_wrapped_bullets_are_rejoined():
    text = """EXPERIENCE
Engineer, Foo
- Reduced API latency by 40% by adding a caching layer in front of
  the pricing service
- Wrote docs
"""
    resume = parse_resume_text(text)
    assert resume.bullets[0].text.endswith("in front of the pricing service")
    assert resume.bullets[1].text == "Wrote docs"


def test_years_from_date_ranges():
    lines = ["Engineer | Jan 2018 - Dec 2019", "Senior Engineer | Jan 2020 - Present"]
    assert estimate_years("", lines, today=date(2024, 1, 1)) == 6.0


def test_years_prefers_larger_stated_value():
    assert estimate_years("Engineer with 10+ years of experience", []) == 10.0


def test_to_json_roundtrip(resume):
    import json

    data = json.loads(resume.to_json())
    assert data["bullets"] and "Python" in data["skills"]
