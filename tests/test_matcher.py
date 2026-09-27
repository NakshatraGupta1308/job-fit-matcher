import numpy as np
import pytest

from resume_matcher.matcher.embeddings import SentenceTransformerEmbedder, TfidfEmbedder, get_embedder, tokenize
from resume_matcher.matcher.scorer import GAP, STRONG, fit_label, is_any_of, rescale, score_match, skill_coverage
from resume_matcher.parser.job_parser import Requirement


def test_tokenize_canonicalizes_skills_and_stems():
    tokens = tokenize("Building services on k8s")
    assert "skill_kubernetes" in tokens
    assert "build" in tokens


def test_tfidf_similarity_prefers_related_text():
    emb = TfidfEmbedder()
    texts = ["Built REST APIs in Flask", "Designed RESTful APIs in Python", "Baked sourdough bread"]
    emb.fit(texts)
    vecs = emb.encode(texts)
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0)
    assert vecs[0] @ vecs[1] > vecs[0] @ vecs[2]


def test_rescale_is_clipped():
    assert rescale(-1, 0.2, 0.6) == 0.0
    assert rescale(0.4, 0.2, 0.6) == pytest.approx(0.5)
    assert rescale(2, 0.2, 0.6) == 1.0


def test_any_of_requirements():
    req = Requirement("Experience with GCP or AWS", "preferred", 0.5, ["AWS", "GCP"])
    assert is_any_of(req)
    coverage, matched, missing = skill_coverage(req, {"AWS"})
    assert coverage == 1.0 and matched == ["AWS"] and missing == []


def test_all_of_requirements():
    req = Requirement("Docker and Kubernetes", "required", 1.0, ["Docker", "Kubernetes"])
    assert skill_coverage(req, {"Docker"})[0] == 0.5


def test_good_fit_scores_higher_than_weak_fit(resume, job, weak_job, embedder):
    good = score_match(resume, job, embedder)
    weak = score_match(resume, weak_job, embedder)
    assert good.overall_score > 55
    assert weak.overall_score < 30
    assert good.overall_score - weak.overall_score > 30


def test_per_requirement_breakdown(resume, job, embedder):
    result = score_match(resume, job, embedder)
    by_text = {m.requirement.text: m for m in result.requirement_matches}
    assert by_text["Strong proficiency in Python and SQL"].label == STRONG
    assert by_text["Familiarity with Terraform and infrastructure as code"].label == GAP
    assert by_text["5+ years of professional software engineering experience"].label == STRONG
    assert result.similarity_matrix.shape == (len(job.requirements), len(resume.matchable_bullets))
    assert 0 <= result.overall_score <= 100


def test_empty_inputs_do_not_crash(embedder):
    from resume_matcher.parser import parse_job_text, parse_resume_text

    result = score_match(parse_resume_text("Just a name"), parse_job_text("Requirements:\n- Python"), embedder)
    assert result.overall_score >= 0


def test_fit_labels():
    assert fit_label(80) == "Strong fit"
    assert fit_label(60) == "Good fit"
    assert fit_label(40) == "Partial fit"
    assert fit_label(10) == "Weak fit"


class FakeModel:
    """Stands in for a SentenceTransformer so the backend can be tested offline."""

    def encode(self, texts, **kwargs):
        vocab = ["python", "api", "bread"]
        return np.array([[t.lower().count(w) + 0.01 for w in vocab] for t in texts], dtype=np.float32)


def test_sentence_transformer_backend_with_fake_model(resume, job):
    emb = SentenceTransformerEmbedder(model=FakeModel())
    vecs = emb.encode(["python api", "bread"])
    assert vecs.shape == (2, 3)
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0)
    result = score_match(resume, job, emb)
    assert result.backend.startswith("sentence-transformers")


def test_auto_backend_falls_back(monkeypatch):
    import resume_matcher.matcher.embeddings as mod

    def boom(name):
        raise OSError("offline")

    monkeypatch.setattr(mod, "_load_sentence_transformer", boom)
    assert isinstance(get_embedder("auto"), TfidfEmbedder)
    with pytest.raises(RuntimeError):
        get_embedder("sentence-transformers")
    with pytest.raises(ValueError):
        get_embedder("nope")
