"""Sentence embeddings behind a small interface.

`SentenceTransformerEmbedder` is the production backend. `HashingEmbedder` is a
deterministic bag-of-words hashing model used by the unit tests and CI so they
run without downloading a neural model; it is not meant for real retrieval.
"""

from __future__ import annotations

import hashlib
import logging
import re
import threading
from typing import Protocol

import numpy as np

from app.core.config import EmbeddingSettings

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    model_id: str
    dimension: int

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...
    def count_tokens(self, text: str) -> int: ...


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vectors / norms).astype("float32")


class SentenceTransformerEmbedder:
    def __init__(self, settings: EmbeddingSettings):
        self.settings = settings
        self.model_id = f"sentence-transformers:{settings.model_name}"
        self._model = None
        self._lock = threading.Lock()
        self._dimension: int | None = None

    @property
    def model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer

                    logger.info("Loading embedding model %s (CPU)", self.settings.model_name)
                    model = SentenceTransformer(self.settings.model_name, device="cpu")
                    model.max_seq_length = min(self.settings.max_seq_length, model.max_seq_length or 512)
                    self._model = model
        return self._model

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._dimension = int(self.model.get_sentence_embedding_dimension())
        return self._dimension

    @property
    def max_tokens(self) -> int:
        return int(self.model.max_seq_length)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype="float32")
        vectors = self.model.encode(
            texts, batch_size=self.settings.batch_size, normalize_embeddings=True, show_progress_bar=False
        )
        return np.asarray(vectors, dtype="float32")

    def embed_query(self, text: str) -> np.ndarray:
        vector = self.model.encode(
            [self.settings.query_prefix + text], normalize_embeddings=True, show_progress_bar=False
        )
        return np.asarray(vector, dtype="float32")[0]

    def count_tokens(self, text: str) -> int:
        return len(self.model.tokenizer.encode(text, add_special_tokens=False))


_TOKEN = re.compile(r"[a-z0-9]+")


class HashingEmbedder:
    """Signed feature hashing of unigrams and bigrams (test backend)."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.model_id = f"hashing:{dimension}"

    def _vector(self, text: str) -> np.ndarray:
        tokens = _TOKEN.findall(text.lower())
        features = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
        vec = np.zeros(self.dimension, dtype="float32")
        for feat in features:
            digest = hashlib.md5(feat.encode()).digest()
            idx = int.from_bytes(digest[:4], "little") % self.dimension
            vec[idx] += 1.0 if digest[4] & 1 else -1.0
        return vec

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype="float32")
        return _normalize(np.stack([self._vector(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return _normalize(self._vector(text)[None, :])[0]

    def count_tokens(self, text: str) -> int:
        return int(len(text.split()) * 1.3) + 1


def create_embedder(settings: EmbeddingSettings) -> Embedder:
    if settings.backend == "hashing":
        return HashingEmbedder()
    if settings.backend == "sentence-transformers":
        return SentenceTransformerEmbedder(settings)
    raise ValueError(f"Unknown embedding backend: {settings.backend}")
