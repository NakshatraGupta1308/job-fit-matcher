"""Embedding backends.

The default backend runs a sentence-transformers model locally. When that model
is not installed or cannot be downloaded, a TF-IDF backend keeps the tool
working fully offline, with somewhat less semantic understanding.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from functools import lru_cache
from typing import Optional, Protocol, Sequence

import numpy as np

from ..skills import find_skill_mentions

log = logging.getLogger(__name__)

DEFAULT_MODEL = "all-MiniLM-L6-v2"


class Embedder(Protocol):
    name: str
    # Cosine similarities below `low` mean "unrelated", above `high` mean "clearly the same thing".
    low: float
    high: float

    def fit(self, corpus: Sequence[str]) -> "Embedder": ...

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class SentenceTransformerEmbedder:
    low = 0.20
    high = 0.62

    def __init__(self, model_name: str = DEFAULT_MODEL, model=None):
        self.model_name = model_name
        self.name = f"sentence-transformers ({model_name})"
        self._model = model if model is not None else _load_sentence_transformer(model_name)

    def fit(self, corpus: Sequence[str]) -> "SentenceTransformerEmbedder":
        return self

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        vectors = self._model.encode(list(texts), convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        return _l2_normalize(np.asarray(vectors, dtype=np.float32))


@lru_cache(maxsize=4)
def _load_sentence_transformer(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


_STOPWORDS = frozenset(
    """a about above across after again all also am an and any are as at be been being both but by can could did do
    does doing done during each either etc few for from further had has have having he her here hers him his how i if
    in into is it its itself just like may me more most must my no nor not of off on once only or other our ours out
    over own per plus same she should so some such than that the their them then there these they this those through
    to too under until up upon us very via was we well were what when where which while who whom why will with within
    would you your yours years year experience experienced strong solid ability able using use used work working
    worked including new etc eg ie""".split()
)

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+#]*")


def _stem(token: str) -> str:
    """A deliberately light stemmer: good enough to join "building", "builds", and "build"."""
    if len(token) <= 4 or not token.isalpha():
        return token
    for suffix, replacement in (
        ("ational", "ate"), ("ization", "ize"), ("isation", "ize"), ("ations", "ate"), ("ation", "ate"),
        ("ments", ""), ("ment", ""), ("ingly", ""), ("ings", ""), ("ing", ""), ("ied", "y"), ("ies", "y"),
        ("ed", ""), ("ers", "er"), ("es", ""), ("ly", ""), ("s", ""),
    ):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            token = token[: -len(suffix)] + replacement
            break
    if len(token) > 3 and token[-1] == token[-2] and token[-1] not in "aeiouls":
        token = token[:-1]
    if token.endswith("e") and len(token) > 4:
        token = token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    """Lowercase, canonicalize skill aliases, drop stopwords, and stem."""
    mentions = find_skill_mentions(text)
    pieces = []
    cursor = 0
    skill_tokens = []
    for m in mentions:
        pieces.append(text[cursor : m.start])
        pieces.append(" ")
        skill_tokens.append("skill_" + re.sub(r"[^a-z0-9]+", "_", m.skill.lower()).strip("_"))
        cursor = m.end
    pieces.append(text[cursor:])
    remainder = "".join(pieces).lower()
    words = [_stem(t) for t in _TOKEN_RE.findall(remainder) if t not in _STOPWORDS and not t.isdigit()]
    return words + skill_tokens


def _features(text: str) -> Counter:
    tokens = tokenize(text)
    feats = Counter(tokens)
    plain = [t for t in tokens if not t.startswith("skill_")]
    feats.update(f"{a} {b}" for a, b in zip(plain, plain[1:]))
    return feats


class TfidfEmbedder:
    """Offline fallback: TF-IDF over stemmed words, bigrams, and canonical skills."""

    name = "tf-idf (offline)"
    low = 0.04
    high = 0.40

    def __init__(self):
        self.vocab: dict[str, int] = {}
        self.idf: Optional[np.ndarray] = None

    def fit(self, corpus: Sequence[str]) -> "TfidfEmbedder":
        doc_freq: Counter = Counter()
        for text in corpus:
            doc_freq.update(set(_features(text)))
        self.vocab = {term: i for i, term in enumerate(sorted(doc_freq))}
        n_docs = max(len(corpus), 1)
        idf = np.zeros(len(self.vocab), dtype=np.float32)
        for term, i in self.vocab.items():
            idf[i] = math.log((1 + n_docs) / (1 + doc_freq[term])) + 1.0
            if term.startswith("skill_"):
                # Named skills are the strongest signal in a short bullet.
                idf[i] *= 1.5
        self.idf = idf
        return self

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if self.idf is None:
            self.fit(texts)
        matrix = np.zeros((len(texts), max(len(self.vocab), 1)), dtype=np.float32)
        for row, text in enumerate(texts):
            for term, count in _features(text).items():
                col = self.vocab.get(term)
                if col is not None:
                    matrix[row, col] = (1.0 + math.log(count)) * self.idf[col]
        return _l2_normalize(matrix)


BACKENDS = ("auto", "sentence-transformers", "tfidf")


def get_embedder(backend: str = "auto", model_name: str = DEFAULT_MODEL) -> Embedder:
    """Return an embedder. "auto" prefers sentence-transformers and falls back to TF-IDF."""
    if backend not in BACKENDS:
        raise ValueError(f"Unknown backend '{backend}'. Choose one of: {', '.join(BACKENDS)}")
    if backend == "tfidf":
        return TfidfEmbedder()
    try:
        return SentenceTransformerEmbedder(model_name)
    except Exception as exc:  # ImportError, network errors, missing model files
        if backend == "sentence-transformers":
            raise RuntimeError(
                f"Could not load sentence-transformers model '{model_name}': {exc}. "
                "Install it with `pip install sentence-transformers` and make sure the model can be downloaded once."
            ) from exc
        log.warning("Falling back to the offline TF-IDF backend: %s", str(exc).splitlines()[0] if str(exc) else exc)
        return TfidfEmbedder()
