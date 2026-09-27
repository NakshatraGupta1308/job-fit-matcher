"""A curated skill vocabulary with aliases, used for keyword matching.

Each canonical skill maps to the surface forms that should count as the same
skill ("k8s" and "Kubernetes", "Postgres" and "PostgreSQL"). Aliases prefixed
with "=" are matched case-sensitively, which keeps short or ambiguous names
("Go", "R") from matching ordinary words.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

SKILLS: dict[str, tuple[str, ...]] = {
    # Programming languages
    "Python": ("python",),
    "Java": ("java",),
    "JavaScript": ("javascript", "ecmascript", "=JS"),
    "TypeScript": ("typescript",),
    "Go": ("golang", "=Go"),
    "Rust": ("rust",),
    "C": ("=C",),
    "C++": ("c++", "cpp"),
    "C#": ("c#", "csharp"),
    "Ruby": ("ruby",),
    "PHP": ("php",),
    "Kotlin": ("kotlin",),
    "Swift": ("=Swift",),
    "Scala": ("scala",),
    "R": ("=R",),
    "MATLAB": ("matlab",),
    "SQL": ("sql",),
    "Bash": ("bash", "shell scripting", "shell scripts"),
    "HTML": ("html", "html5"),
    "CSS": ("css", "css3", "sass", "scss"),
    # Web frameworks and runtimes
    "React": ("=React", "react.js", "reactjs"),
    "Angular": ("angular", "angularjs"),
    "Vue": ("vue", "vue.js", "vuejs"),
    "Next.js": ("next.js", "nextjs"),
    "Node.js": ("node.js", "nodejs", "=Node"),
    "Express": ("express.js", "expressjs"),
    "Django": ("django",),
    "Flask": ("flask",),
    "FastAPI": ("fastapi",),
    "Spring": ("spring boot", "spring framework", "=Spring"),
    "Ruby on Rails": ("ruby on rails", "rails"),
    ".NET": (".net", "asp.net", "dotnet"),
    "GraphQL": ("graphql",),
    "REST APIs": ("=REST", "restful", "rest api", "rest apis", "restful api", "restful apis"),
    "gRPC": ("grpc",),
    "Microservices": ("microservice", "microservices"),
    # Data and machine learning
    "Machine Learning": ("machine learning", "ml"),
    "Deep Learning": ("deep learning",),
    "NLP": ("nlp", "natural language processing"),
    "Computer Vision": ("computer vision",),
    "LLMs": ("llm", "llms", "large language model", "large language models"),
    "PyTorch": ("pytorch",),
    "TensorFlow": ("tensorflow",),
    "Keras": ("keras",),
    "scikit-learn": ("scikit-learn", "sklearn", "scikit learn"),
    "pandas": ("pandas",),
    "NumPy": ("numpy",),
    "Spark": ("=Spark", "pyspark", "apache spark"),
    "Hadoop": ("hadoop",),
    "Airflow": ("airflow", "apache airflow"),
    "dbt": ("dbt",),
    "Kafka": ("kafka", "apache kafka"),
    "ETL": ("etl", "elt", "data pipelines", "data pipeline"),
    "Data Analysis": ("data analysis", "data analytics"),
    "Data Visualization": ("data visualization", "data visualisation"),
    "Tableau": ("tableau",),
    "Power BI": ("power bi", "powerbi"),
    "Statistics": ("statistics", "statistical analysis", "statistical modeling"),
    "A/B Testing": ("a/b testing", "a/b tests", "ab testing", "experimentation"),
    "Excel": ("=Excel", "microsoft excel"),
    "Jupyter": ("jupyter",),
    "MLOps": ("mlops",),
    "CUDA": ("cuda",),
    "OpenCV": ("opencv",),
    "ROS": ("=ROS",),
    "Embedded Systems": ("embedded systems", "embedded software", "firmware"),
    # Databases
    "PostgreSQL": ("postgresql", "postgres"),
    "MySQL": ("mysql",),
    "SQLite": ("sqlite",),
    "MongoDB": ("mongodb", "mongo"),
    "Redis": ("redis",),
    "Elasticsearch": ("elasticsearch", "elastic search", "opensearch"),
    "Cassandra": ("cassandra",),
    "DynamoDB": ("dynamodb",),
    "Snowflake": ("snowflake",),
    "BigQuery": ("bigquery",),
    "Redshift": ("redshift",),
    "NoSQL": ("nosql",),
    # Cloud and infrastructure
    "AWS": ("aws", "amazon web services"),
    "GCP": ("gcp", "google cloud", "google cloud platform"),
    "Azure": ("azure", "microsoft azure"),
    "Docker": ("docker", "containerization", "containers"),
    "Kubernetes": ("kubernetes", "k8s"),
    "Terraform": ("terraform",),
    "Ansible": ("ansible",),
    "CI/CD": ("ci/cd", "continuous integration", "continuous delivery", "continuous deployment", "cicd"),
    "GitHub Actions": ("github actions",),
    "Jenkins": ("jenkins",),
    "Git": ("git",),
    "Linux": ("linux", "unix"),
    "Serverless": ("serverless", "lambda", "cloud functions"),
    "Infrastructure as Code": ("infrastructure as code", "iac"),
    "Monitoring": ("monitoring", "observability", "prometheus", "grafana", "datadog"),
    "Distributed Systems": ("distributed systems", "distributed system"),
    "System Design": ("system design", "systems design"),
    "Networking": ("networking", "tcp/ip"),
    "Security": ("security", "application security", "appsec", "infosec"),
    # Practices
    "Testing": ("unit testing", "unit tests", "integration tests", "integration testing", "test automation", "automated testing", "tdd", "pytest", "jest"),
    "Agile": ("agile", "scrum", "kanban"),
    "Code Review": ("code review", "code reviews"),
    "API Design": ("api design",),
    "Performance Optimization": ("performance optimization", "performance tuning", "latency"),
    "Object-Oriented Design": ("object-oriented", "object oriented", "oop"),
    "Data Structures and Algorithms": ("data structures", "algorithms"),
    # Product, design, and business
    "Product Management": ("product management", "product roadmap", "roadmap", "roadmaps"),
    "Project Management": ("project management",),
    "UX Design": ("ux", "user experience", "ux design"),
    "Figma": ("figma",),
    "SEO": ("seo", "search engine optimization"),
    "Salesforce": ("salesforce",),
    "Jira": ("jira",),
    # Collaboration and soft skills
    "Leadership": ("leadership", "led a team", "team lead", "tech lead"),
    "Mentoring": ("mentor", "mentored", "mentoring", "mentorship", "coached", "coaching"),
    "Communication": ("communication", "communicator", "presentation skills", "written and verbal"),
    "Stakeholder Management": ("stakeholder", "stakeholders", "stakeholder management"),
    "Cross-functional Collaboration": ("cross-functional", "cross functional", "cross-team"),
    "Problem Solving": ("problem solving", "problem-solving"),
}


# Skills that are reasonable evidence for a broader requirement. Used only to point the
# user at bullets worth reframing, never to claim the broader skill automatically.
RELATED_SKILLS: dict[str, tuple[str, ...]] = {
    "ETL": ("Airflow", "dbt", "Spark", "Kafka", "Hadoop", "Snowflake", "BigQuery", "Redshift", "pandas"),
    "Machine Learning": ("PyTorch", "TensorFlow", "Keras", "scikit-learn", "Deep Learning", "NLP", "Computer Vision", "LLMs"),
    "Deep Learning": ("PyTorch", "TensorFlow", "Keras"),
    "Data Analysis": ("pandas", "SQL", "Excel", "Tableau", "Power BI", "Statistics", "Jupyter"),
    "Data Visualization": ("Tableau", "Power BI", "Excel"),
    "Kubernetes": ("Docker",),
    "Docker": ("Kubernetes",),
    "CI/CD": ("Jenkins", "GitHub Actions"),
    "Infrastructure as Code": ("Terraform", "Ansible"),
    "Terraform": ("Infrastructure as Code",),
    "Microservices": ("REST APIs", "gRPC", "Docker", "Kubernetes"),
    "REST APIs": ("Flask", "Django", "FastAPI", "Express", "Spring", "GraphQL"),
    "Distributed Systems": ("Kafka", "Spark", "Microservices", "Cassandra"),
    "NoSQL": ("MongoDB", "Redis", "Cassandra", "DynamoDB", "Elasticsearch"),
    "SQL": ("PostgreSQL", "MySQL", "SQLite", "Snowflake", "BigQuery", "Redshift"),
    "Serverless": ("AWS", "GCP", "Azure"),
    "Leadership": ("Mentoring",),
    "Mentoring": ("Leadership",),
    "JavaScript": ("TypeScript", "React", "Node.js", "Vue", "Angular"),
    "TypeScript": ("JavaScript",),
    "Monitoring": ("Performance Optimization",),
}


def related_present(skill: str, available: set[str]) -> list[str]:
    """Skills the resume has that are reasonable evidence for `skill`."""
    return [s for s in RELATED_SKILLS.get(skill, ()) if s in available]


@dataclass(frozen=True)
class SkillMention:
    skill: str
    surface: str
    start: int
    end: int


def _alias_pattern(alias: str) -> tuple[str, int]:
    flags = re.IGNORECASE
    if alias.startswith("="):
        alias = alias[1:]
        flags = 0
    escaped = re.escape(alias).replace(r"\ ", r"[\s-]+")
    # Boundaries allow symbols like "+", "#", and "." to be part of a skill name.
    pattern = r"(?<![\w+#.])" + escaped + r"(?![\w+#]|\.\w)"
    return pattern, flags


@lru_cache(maxsize=1)
def _compiled() -> list[tuple[str, re.Pattern]]:
    compiled = []
    for skill, aliases in SKILLS.items():
        sensitive = [a for a in aliases if a.startswith("=")]
        insensitive = [a for a in aliases if not a.startswith("=")]
        # Longer aliases first so "rest api" wins over "rest".
        insensitive.sort(key=len, reverse=True)
        if insensitive:
            pattern = "|".join(_alias_pattern(a)[0] for a in insensitive)
            compiled.append((skill, re.compile(pattern, re.IGNORECASE)))
        for alias in sensitive:
            pattern, flags = _alias_pattern(alias)
            compiled.append((skill, re.compile(pattern, flags)))
    return compiled


def find_skill_mentions(text: str) -> list[SkillMention]:
    """Find every skill mention in text, resolving overlaps in favor of longer matches."""
    candidates: list[SkillMention] = []
    for skill, pattern in _compiled():
        for m in pattern.finditer(text):
            candidates.append(SkillMention(skill, m.group(0), m.start(), m.end()))
    candidates.sort(key=lambda s: (s.start, -(s.end - s.start)))
    mentions: list[SkillMention] = []
    last_end = -1
    for cand in candidates:
        if cand.start >= last_end:
            mentions.append(cand)
            last_end = cand.end
    return _drop_false_positives(text, mentions)


def _drop_false_positives(text: str, mentions: list[SkillMention]) -> list[SkillMention]:
    kept = []
    for m in mentions:
        before = text[max(0, m.start - 1) : m.start]
        after = text[m.end : m.end + 3]
        if m.skill == "C" and (after.startswith("-") or after.startswith("/") or before == "/"):
            # "C-suite", "C/C++" (the C++ half is matched separately), "B/C".
            if not after.startswith("/C"):
                continue
        if m.surface in {"Go", "Spring", "Node", "Swift", "Spark", "React", "Excel"} and m.start == _line_start(text, m.start):
            # A capitalized "Go" or "Spring" at the start of a line is usually an ordinary word.
            continue
        kept.append(m)
    return kept


def _line_start(text: str, index: int) -> int:
    nl = text.rfind("\n", 0, index)
    start = nl + 1
    while start < index and text[start] in " -*\t":
        start += 1
    return start


def extract_skills(text: str) -> set[str]:
    """Return the set of canonical skills mentioned in text."""
    return {m.skill for m in find_skill_mentions(text)}


def surface_forms(text: str) -> dict[str, str]:
    """Map each canonical skill found in text to the first surface form used for it."""
    forms: dict[str, str] = {}
    for m in find_skill_mentions(text):
        forms.setdefault(m.skill, m.surface)
    return forms
