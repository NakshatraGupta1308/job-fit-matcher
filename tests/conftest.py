from pathlib import Path

import pytest

from resume_matcher.matcher.embeddings import TfidfEmbedder
from resume_matcher.parser import parse_job, parse_resume

DATA = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="session")
def data_dir() -> Path:
    return DATA


@pytest.fixture()
def resume():
    return parse_resume(DATA / "sample_resume.md")


@pytest.fixture()
def job():
    return parse_job(DATA / "sample_job.txt")


@pytest.fixture()
def weak_job():
    return parse_job(DATA / "sample_job_weak_fit.txt")


@pytest.fixture()
def embedder():
    return TfidfEmbedder()
