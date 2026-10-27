"""Application container: wires settings, stores, models and the RAG pipeline."""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

from app.core.config import PROJECT_ROOT, Settings, get_settings
from app.services.citation_service import CitationVerifier
from app.services.document_store import DocumentStore
from app.services.embedding_service import Embedder, create_embedder
from app.services.ingestion_service import IngestionService
from app.services.llm_service import LLMProvider, create_provider
from app.services.rag_pipeline import RAGPipeline
from app.services.retrieval_service import RetrievalService
from app.services.vector_store import FaissVectorStore

logger = logging.getLogger(__name__)

# Embedding models the UI may switch to (all CPU-friendly sentence-transformers models).
ALLOWED_EMBEDDING_MODELS = [
    "BAAI/bge-small-en-v1.5",
    "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI/bge-base-en-v1.5",
]
QUERY_PREFIXES = {
    "BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-base-en-v1.5": "Represent this sentence for searching relevant passages: ",
}


def load_system_prompt() -> str:
    return (PROJECT_ROOT / "config" / "system_prompt.txt").read_text(encoding="utf-8")


class Container:
    def __init__(self, settings: Settings, embedder: Embedder | None = None, provider: LLMProvider | None = None):
        self.lock = threading.RLock()
        self.settings = settings
        self.data_path: Path = settings.data_path
        self._apply_runtime_overrides()
        self.store = DocumentStore(self.data_path / "processed" / "remoir.db")
        self.embedder = embedder or create_embedder(settings.embedding)
        self.vectors = FaissVectorStore(self.data_path / "vector_store", self.embedder.model_id, None)
        self._custom_provider = provider
        self.system_prompt = load_system_prompt()
        self._build_services()

    @property
    def runtime_path(self) -> Path:
        return self.data_path / "runtime_settings.json"

    def _apply_runtime_overrides(self) -> None:
        if not self.runtime_path.exists():
            return
        try:
            data = json.loads(self.runtime_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            logger.warning("Ignoring unreadable runtime settings file")
            return
        s = self.settings
        s.retrieval.top_k = data.get("top_k", s.retrieval.top_k)
        s.retrieval.strategy = data.get("strategy", s.retrieval.strategy)
        s.retrieval.rerank = data.get("rerank", s.retrieval.rerank)
        s.chunking.chunk_size = data.get("chunk_size", s.chunking.chunk_size)
        s.chunking.chunk_overlap = data.get("chunk_overlap", s.chunking.chunk_overlap)
        s.llm.provider = data.get("llm_provider", s.llm.provider)
        s.llm.model = data.get("llm_model", s.llm.model)
        if s.embedding.backend == "sentence-transformers" and data.get("embedding_model") in ALLOWED_EMBEDDING_MODELS:
            s.embedding.model_name = data["embedding_model"]
            s.embedding.query_prefix = QUERY_PREFIXES.get(data["embedding_model"], "")

    def _build_services(self) -> None:
        s = self.settings
        self.retriever = RetrievalService(self.store, self.vectors, self.embedder, s.retrieval)
        self.provider = self._custom_provider or create_provider(s.llm, self.embedder)
        self.verifier = CitationVerifier(self.embedder, s.evidence.min_support_score)
        self.pipeline = RAGPipeline(s, self.store, self.retriever, self.provider, self.verifier, self.system_prompt)
        self.ingestion = IngestionService(s, self.store, self.vectors, self.embedder)

    def runtime_settings(self) -> dict:
        s = self.settings
        return {
            "top_k": s.retrieval.top_k, "strategy": s.retrieval.strategy, "rerank": s.retrieval.rerank,
            "chunk_size": s.chunking.chunk_size, "chunk_overlap": s.chunking.chunk_overlap,
            "embedding_model": s.embedding.model_name, "llm_provider": s.llm.provider, "llm_model": s.llm.model,
        }

    def update_runtime_settings(self, new: dict) -> tuple[bool, list[str]]:
        """Apply UI settings. Returns (reindex_required, messages)."""
        messages: list[str] = []
        with self.lock:
            s = self.settings
            old = self.runtime_settings()
            if new["chunk_overlap"] >= new["chunk_size"]:
                raise ValueError("Chunk overlap must be smaller than chunk size.")
            if new["embedding_model"] != old["embedding_model"] and new["embedding_model"] not in ALLOWED_EMBEDDING_MODELS:
                raise ValueError(f"Embedding model must be one of: {', '.join(ALLOWED_EMBEDDING_MODELS)}")
            s.retrieval.top_k, s.retrieval.strategy, s.retrieval.rerank = new["top_k"], new["strategy"], new["rerank"]
            s.chunking.chunk_size, s.chunking.chunk_overlap = new["chunk_size"], new["chunk_overlap"]
            s.llm.provider, s.llm.model = new["llm_provider"], new["llm_model"]
            if (new["chunk_size"], new["chunk_overlap"]) != (old["chunk_size"], old["chunk_overlap"]):
                messages.append("Chunking changed: applies to new uploads. Rebuild with re-parsing to apply to existing documents.")
            if new["embedding_model"] != old["embedding_model"] and s.embedding.backend == "sentence-transformers":
                s.embedding.model_name = new["embedding_model"]
                s.embedding.query_prefix = QUERY_PREFIXES.get(new["embedding_model"], "")
                self.embedder = create_embedder(s.embedding)
                self.vectors.model_id = self.embedder.model_id
                messages.append("Embedding model changed: the existing index is incompatible and must be rebuilt before search.")
            if new["llm_provider"] != old["llm_provider"] or new["llm_model"] != old["llm_model"]:
                self._custom_provider = None
            self._build_services()
            self.runtime_path.write_text(json.dumps(self.runtime_settings(), indent=2), encoding="utf-8")
        return not self.vectors.compatible, messages

    def index_consistent(self) -> bool:
        return self.vectors.size == self.store.chunk_count()


_container: Container | None = None
_container_lock = threading.Lock()


def get_container() -> Container:
    global _container
    with _container_lock:
        if _container is None:
            _container = Container(get_settings())
        return _container


def set_container(container: Container | None) -> None:
    """Used by tests to inject a container built with fake models."""
    global _container
    with _container_lock:
        _container = container
