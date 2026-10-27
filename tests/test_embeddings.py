import os

import numpy as np
import pytest

from app.core.config import EmbeddingSettings
from app.services.embedding_service import HashingEmbedder, SentenceTransformerEmbedder, create_embedder


def test_hashing_embedder_shapes_and_normalization():
    e = HashingEmbedder(dimension=64)
    vecs = e.embed_documents(["rainfall and maize yields", "soil carbon storage"])
    assert vecs.shape == (2, 64) and vecs.dtype == np.float32
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-5)
    assert e.embed_documents([]).shape == (0, 64)


def test_hashing_embedder_is_deterministic_and_semantic_enough():
    e = HashingEmbedder()
    q = e.embed_query("rainfall variability maize")
    related, unrelated = e.embed_documents(["maize yields fall with rainfall variability", "pollinators need hedgerows"])
    assert np.allclose(q, HashingEmbedder().embed_query("rainfall variability maize"))
    assert float(q @ related) > float(q @ unrelated)


def test_factory_selects_backend():
    assert isinstance(create_embedder(EmbeddingSettings(backend="hashing")), HashingEmbedder)
    assert isinstance(create_embedder(EmbeddingSettings(backend="sentence-transformers")), SentenceTransformerEmbedder)
    with pytest.raises(ValueError):
        create_embedder(EmbeddingSettings(backend="unknown"))


def test_model_id_identifies_model():
    st = SentenceTransformerEmbedder(EmbeddingSettings(model_name="BAAI/bge-small-en-v1.5"))
    assert st.model_id == "sentence-transformers:BAAI/bge-small-en-v1.5"


@pytest.mark.skipif(not os.environ.get("REMOIR_RUN_MODEL_TESTS"), reason="set REMOIR_RUN_MODEL_TESTS=1 to load the real model")
def test_sentence_transformer_embeddings():
    e = SentenceTransformerEmbedder(EmbeddingSettings(query_prefix="Represent this sentence for searching relevant passages: "))
    docs = e.embed_documents(["Rainfall variability reduces maize yields.", "Pollinators depend on hedgerows."])
    q = e.embed_query("How does rainfall affect crop yields?")
    assert docs.shape == (2, e.dimension) == (2, 384)
    assert float(q @ docs[0]) > float(q @ docs[1])
    assert e.count_tokens("maize yields") >= 2
