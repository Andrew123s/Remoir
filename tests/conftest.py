"""Shared fixtures. Tests use the deterministic hashing embedder and mocked LLMs,
so the suite runs offline without downloading models or needing an API key."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.config import load_settings  # noqa: E402
from app.dependencies import Container, set_container  # noqa: E402
from app.services.embedding_service import HashingEmbedder  # noqa: E402
from scripts import make_fixtures  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures"


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    make_fixtures.main(FIXTURES)
    return FIXTURES


@pytest.fixture
def settings(tmp_path):
    s = load_settings(env={})
    s.app.data_dir = str(tmp_path / "data")
    s.embedding.backend = "hashing"
    # Bag-of-words hashing similarities are lower than neural ones.
    s.evidence.min_dense_score = 0.12
    s.evidence.min_support_score = 0.35
    s.llm.extractive_min_score = 0.1
    s.chunking.chunk_size = 120
    s.chunking.chunk_overlap = 25
    s.chunking.min_chunk_tokens = 15
    return s


class FakeLLM:
    """Deterministic stand-in for an LLM provider."""

    name = "fake"
    model = "fake-model"

    def __init__(self, reply=None):
        self.reply = reply
        self.requests = []

    def status(self):
        return {"provider": self.name, "available": True, "model": self.model}

    def generate(self, request):
        self.requests.append(request)
        if callable(self.reply):
            return self.reply(request)
        return self.reply or "No answer."


@pytest.fixture
def container(settings):
    c = Container(settings, embedder=HashingEmbedder())
    set_container(c)
    yield c
    set_container(None)
    c.store.close()


@pytest.fixture
def indexed(container, fixtures_dir):
    """Container with the three synthetic research PDFs ingested."""
    ids = {}
    for name in ("rainfall_maize_study.pdf", "ml_crop_prediction.pdf", "ecosystem_services_review.pdf"):
        outcome = container.ingestion.ingest_bytes(name, (fixtures_dir / name).read_bytes())
        assert outcome.status == "indexed", outcome.message
        ids[name] = outcome.document_id
    container.retriever.invalidate()
    return container, ids
