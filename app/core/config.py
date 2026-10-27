"""Typed application settings loaded from config/config.yaml plus environment overrides."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


class AppSettings(BaseModel):
    name: str = "Remoir"
    data_dir: str = "data"
    log_level: str = "INFO"


class IngestionSettings(BaseModel):
    max_upload_mb: float = Field(50, gt=0, le=500)
    min_page_chars: int = Field(40, ge=0)
    scanned_page_ratio: float = Field(0.8, gt=0, le=1)


class OCRSettings(BaseModel):
    enabled: bool = False
    language: str = "eng"
    dpi: int = Field(300, ge=72, le=600)


class ChunkingSettings(BaseModel):
    chunk_size: int = Field(220, ge=50, le=1000)
    chunk_overlap: int = Field(40, ge=0, le=300)
    min_chunk_tokens: int = Field(25, ge=1, le=200)
    max_chunk_tokens: int = Field(400, ge=50, le=1000)


class EmbeddingSettings(BaseModel):
    backend: str = "sentence-transformers"
    model_name: str = "BAAI/bge-small-en-v1.5"
    query_prefix: str = ""
    batch_size: int = Field(32, ge=1, le=256)
    max_seq_length: int = Field(512, ge=32, le=8192)


class RetrievalSettings(BaseModel):
    top_k: int = Field(6, ge=1, le=30)
    strategy: str = "hybrid"
    candidate_multiplier: int = Field(4, ge=1, le=20)
    rrf_k: int = Field(60, ge=1)
    rerank: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    dedup_threshold: float = Field(0.85, gt=0, le=1)
    max_context_tokens: int = Field(2500, ge=200, le=16000)


class EvidenceSettings(BaseModel):
    min_dense_score: float = Field(0.45, ge=0, le=1)
    min_term_coverage: float = Field(0.3, ge=0, le=1)
    min_support_score: float = Field(0.5, ge=0, le=1)


class LLMSettings(BaseModel):
    provider: str = "extractive"
    model: str = "llama3.2:3b"
    ollama_url: str = "http://localhost:11434"
    openai_base_url: str = "https://api.openai.com/v1"
    temperature: float = Field(0.1, ge=0, le=2)
    max_tokens: int = Field(700, ge=32, le=8000)
    timeout_s: float = Field(120, gt=0, le=600)
    # Minimum question-sentence relevance for the extractive provider to quote a sentence.
    extractive_min_score: float = Field(0.35, ge=0, le=1)


class ConversationSettings(BaseModel):
    max_turns: int = Field(6, ge=0, le=50)
    max_history_chars: int = Field(2000, ge=0, le=20000)


class Settings(BaseModel):
    app: AppSettings = AppSettings()
    ingestion: IngestionSettings = IngestionSettings()
    ocr: OCRSettings = OCRSettings()
    chunking: ChunkingSettings = ChunkingSettings()
    embedding: EmbeddingSettings = EmbeddingSettings()
    retrieval: RetrievalSettings = RetrievalSettings()
    evidence: EvidenceSettings = EvidenceSettings()
    llm: LLMSettings = LLMSettings()
    conversation: ConversationSettings = ConversationSettings()

    @property
    def data_path(self) -> Path:
        path = Path(self.app.data_dir)
        return path if path.is_absolute() else PROJECT_ROOT / path


# Environment variables that override YAML values: name -> (section, key, type)
_ENV_OVERRIDES = {
    "REMOIR_DATA_DIR": ("app", "data_dir", str),
    "REMOIR_LOG_LEVEL": ("app", "log_level", str),
    "REMOIR_EMBEDDING_BACKEND": ("embedding", "backend", str),
    "REMOIR_EMBEDDING_MODEL": ("embedding", "model_name", str),
    "REMOIR_LLM_PROVIDER": ("llm", "provider", str),
    "REMOIR_LLM_MODEL": ("llm", "model", str),
    "REMOIR_OLLAMA_URL": ("llm", "ollama_url", str),
    "REMOIR_OPENAI_BASE_URL": ("llm", "openai_base_url", str),
    "REMOIR_OCR_ENABLED": ("ocr", "enabled", lambda v: v.lower() in {"1", "true", "yes"}),
    "REMOIR_MAX_UPLOAD_MB": ("ingestion", "max_upload_mb", float),
}


def load_settings(path: Path | str | None = None, env: dict[str, str] | None = None) -> Settings:
    path = Path(path or os.environ.get("REMOIR_CONFIG", DEFAULT_CONFIG_PATH))
    raw: dict = {}
    if path.exists():
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    env = os.environ if env is None else env
    for var, (section, key, cast) in _ENV_OVERRIDES.items():
        if env.get(var):
            raw.setdefault(section, {})[key] = cast(env[var])
    return Settings.model_validate(raw)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()
