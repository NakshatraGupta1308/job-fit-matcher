from resume_matcher.skills import extract_skills, related_present, surface_forms


def test_aliases_map_to_canonical_names():
    skills = extract_skills("Deployed services on k8s with Postgres, node.js and scikit-learn")
    assert {"Kubernetes", "PostgreSQL", "Node.js", "scikit-learn"} <= skills


def test_symbols_in_skill_names():
    assert {"C++", "C#", ".NET", "CI/CD"} <= extract_skills("C++ and C# on ASP.NET with CI/CD")


def test_c_and_cpp_both_found():
    assert {"C", "C++"} <= extract_skills("C/C++ developer")


def test_case_sensitive_aliases_avoid_common_words():
    skills = extract_skills("Go above and beyond. We will react quickly and the rest of the team will excel.")
    assert not skills & {"Go", "React", "REST APIs", "Excel"}


def test_case_sensitive_aliases_still_match_real_mentions():
    assert {"Go", "React", "REST APIs"} <= extract_skills("Built REST APIs in Go and a React front end")


def test_c_suite_is_not_c():
    assert "C" not in extract_skills("Presented to the C-suite")


def test_surface_forms_keep_original_spelling():
    assert surface_forms("We use Postgres and K8s")["PostgreSQL"] == "Postgres"


def test_related_present():
    assert related_present("ETL", {"Airflow", "Python"}) == ["Airflow"]
    assert related_present("Kafka", {"Python"}) == []
