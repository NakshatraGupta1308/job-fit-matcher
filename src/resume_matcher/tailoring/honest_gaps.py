"""Honest ways to handle a gap you cannot fill: a cover letter line and a learning plan."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..parser.resume_parser import Resume
from ..skills import RELATED_SKILLS, extract_skills
from ..suggestions.rewrite_suggester import strengthen_opener
from .gap_prompts import GapPrompt

# Small, concrete projects that produce something real enough to put on a resume.
PROJECT_IDEAS: dict[str, str] = {
    "Kafka": "Stream a public event feed (for example GitHub events or a stock ticker) through a Kafka topic into a small consumer that writes aggregates to a database.",
    "Terraform": "Write Terraform for a small cloud setup (a storage bucket, a function, and an IAM role), apply it, then tear it down and bring it back with one command.",
    "Infrastructure as Code": "Describe a small cloud setup in Terraform or Pulumi so it can be recreated from scratch with one command.",
    "Kubernetes": "Deploy a two-service app (API plus worker) to a local cluster with kind or minikube, with health checks and a rolling update.",
    "Docker": "Containerize one of your existing projects with a small Dockerfile and a docker-compose file that also starts its database.",
    "GCP": "Deploy one of your projects to Cloud Run with a managed database, and write down what it costs per month.",
    "AWS": "Deploy one of your projects on AWS (Lambda or ECS plus S3), with a short README on how to deploy it.",
    "Azure": "Deploy one of your projects to Azure App Service or Functions with a managed database.",
    "Airflow": "Build a daily Airflow DAG that pulls a public dataset, validates it, and loads it into Postgres.",
    "ETL": "Build a small pipeline that pulls a public dataset every day, cleans it, and loads it into a database you can query.",
    "dbt": "Model a public dataset in dbt with staging and mart layers, tests, and generated docs.",
    "Spark": "Process a multi-gigabyte public dataset with PySpark and compare the runtime against pandas.",
    "Snowflake": "Load a public dataset into a Snowflake trial account and write a few analytical queries with window functions.",
    "BigQuery": "Answer three real questions with SQL against one of BigQuery's public datasets.",
    "PostgreSQL": "Design a schema for a small app, load realistic data, and speed up a slow query with the right index.",
    "MongoDB": "Build a small API backed by MongoDB, including an aggregation pipeline for a summary endpoint.",
    "Redis": "Add Redis caching to one of your APIs and measure the latency before and after.",
    "GraphQL": "Put a GraphQL layer in front of an existing REST API or database and document the schema.",
    "TypeScript": "Convert a small JavaScript project to TypeScript with strict mode turned on.",
    "React": "Build a small React front end for one of your existing APIs.",
    "Go": "Rewrite a small service or CLI you already have in Go and compare it with the original.",
    "Rust": "Write a small command-line tool in Rust that you would actually use.",
    "Java": "Build a small REST service in Java with Spring Boot and tests.",
    "C++": "Implement a small performance-sensitive tool in C++ and benchmark it.",
    "PyTorch": "Train a small model in PyTorch on a public dataset and write up the results and what you tried.",
    "TensorFlow": "Train and export a small model in TensorFlow or Keras on a public dataset.",
    "Machine Learning": "Take a public dataset end to end: baseline, a better model, an honest evaluation, and a short write-up.",
    "Deep Learning": "Fine-tune a small pretrained model on a public dataset and document the results.",
    "Computer Vision": "Fine-tune a pretrained image model on a small public dataset and report accuracy on a held-out set.",
    "NLP": "Build a text classifier on a public dataset and compare a simple baseline with a pretrained model.",
    "LLMs": "Build a small retrieval app over a set of documents you care about, with a simple evaluation of its answers.",
    "CUDA": "Write a CUDA kernel for a simple operation (for example matrix multiply) and compare it with the CPU version.",
    "CI/CD": "Add a CI pipeline to one of your repos that runs tests and linting on every push and deploys on merge.",
    "GitHub Actions": "Add a GitHub Actions workflow to one of your repos that tests every pull request.",
    "Monitoring": "Add metrics and a dashboard (Prometheus and Grafana, or a hosted tool) to one of your services, plus one alert.",
    "Tableau": "Build a public Tableau dashboard on an open dataset that answers a specific question.",
    "Power BI": "Build a Power BI report on an open dataset with a clear question and takeaway.",
    "A/B Testing": "Analyze a public A/B test dataset: check the setup, compute significance, and write a recommendation.",
    "Statistics": "Write up an analysis of a public dataset with confidence intervals and a clear explanation of the method.",
    "Microservices": "Split a small app into two services that talk over HTTP or a queue, with a docker-compose setup.",
    "REST APIs": "Build and document a small REST API with authentication, pagination, and tests.",
    "Testing": "Add unit and integration tests to one of your projects and set up coverage reporting.",
    "Leadership": "Volunteer to lead a small initiative (a hackathon team, an open source feature, a study group) and note the outcome.",
    "Mentoring": "Mentor someone newer than you (a classmate, a new teammate, an open source newcomer) for a few weeks.",
}


@dataclass
class LearningPlan:
    skill: str
    steps: list[str] = field(default_factory=list)


def _evidence_for(resume: Resume, skills: list[str]) -> Optional[str]:
    for b in resume.evidence_bullets:
        if set(skills) & extract_skills(b.text):
            return b.text
    return None


def cover_letter_line(prompt: GapPrompt, resume: Resume) -> str:
    """One or two honest sentences acknowledging the gap and pointing to real, related experience."""
    target = prompt.target
    if prompt.related:
        evidence = _evidence_for(resume, prompt.related)
        related = _join(prompt.related[:3])
        if evidence:
            return (
                f"I haven't worked with {target} directly yet, but I have hands-on experience with {related}: "
                f"{_evidence_clause(evidence)}. I'm confident that background will help me get productive "
                f"with {target} quickly."
            )
        return (
            f"I haven't worked with {target} directly yet, but my experience with {related} covers many of the "
            f"same ideas, and I'm confident I can get productive with it quickly."
        )
    return (
        f"I haven't worked with {target} directly yet. "
        "[Add one true sentence about how you are closing this gap, such as a course or project you have actually started.]"
    )


def _evidence_clause(bullet_text: str) -> str:
    text = (strengthen_opener(bullet_text) or bullet_text).rstrip(".")
    first = text.split()[0].lower()
    if first.endswith("ed") or first in {"built", "led", "wrote", "ran", "made", "grew", "won", "drove", "taught"}:
        return "most recently, I " + text[:1].lower() + text[1:]
    return f"for example, \"{text}\""


def _join(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + f" and {items[-1]}"


def learning_plan(prompt: GapPrompt) -> list[LearningPlan]:
    skills = prompt.terms or [prompt.phrase]
    canon = prompt.missing_skills or [prompt.phrase]
    plans = []
    tools = set(prompt.tools)
    for name, key in zip(skills, canon):
        # "Infrastructure as Code and Terraform": the Terraform project covers the concept too.
        if name not in tools and any(o in RELATED_SKILLS.get(key, ()) for o, t in zip(canon, skills) if t in tools):
            continue
        idea = PROJECT_IDEAS.get(key)
        if any(ch.isupper() for ch in name):
            steps = [f"Work through the official getting-started guide or tutorial for {name}."]
        else:
            steps = [f"Learn the basics of {name} from a well-reviewed course, book, or tutorial."]
        if idea:
            steps.append(f"Project idea: {idea}")
        else:
            steps.append(f"Build a small project that uses {name} end to end, and put the code on GitHub with a short README.")
        steps.append("Once it works, come back and answer \"Yes\" for this gap to add it to your Projects section.")
        plans.append(LearningPlan(name, steps))
    return plans


def notes_markdown(job_name: str, entries: list[tuple[GapPrompt, str, list[LearningPlan]]]) -> str:
    """Cover letter lines and learning plans for the gaps the user could not fill."""
    out = [f"# Gap notes: {job_name}", ""]
    if not entries:
        out.append("_No open gaps. Nice work._")
        return "\n".join(out) + "\n"
    out += [
        "These are the requirements you do not have experience with yet. Nothing here was added to your resume.",
        "",
    ]
    for prompt, line, plans in entries:
        out += [f"## {prompt.requirement}", "", "**Cover letter line (edit before using):**", "", f"> {line}", ""]
        for plan in plans:
            out.append(f"**How to build real experience with {plan.skill}:**")
            out.append("")
            out += [f"{i}. {step}" for i, step in enumerate(plan.steps, 1)]
            out.append("")
    return "\n".join(out).rstrip() + "\n"
