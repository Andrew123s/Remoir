"""Pydantic request and response models for the REST API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

DocumentId = Field(pattern=r"^[a-f0-9]{16}$")


class HealthResponse(BaseModel):
    status: str
    version: str
    documents: int
    vectors: int
    embedding_model: str
    index_compatible: bool
    index_consistent: bool
    llm: dict
    ocr_enabled: bool


class PageInfo(BaseModel):
    page: int
    label: str | None = None
    status: str
    chars: int
    source: str


class DocumentInfo(BaseModel):
    id: str
    filename: str
    title: str | None = None
    title_source: str | None = None
    authors: str | None = None
    year: int | None = None
    year_source: str | None = None
    page_count: int = 0
    size_bytes: int = 0
    content_hash: str
    status: str
    error_code: str | None = None
    error: str | None = None
    warnings: list[str] = []
    chunk_count: int = 0
    embedding_model: str | None = None
    ocr_applied: bool = False
    ocr_required: bool = False
    uploaded_at: str
    processing_seconds: float | None = None


class DocumentDetail(DocumentInfo):
    pages: list[PageInfo] = []


class UploadResponse(BaseModel):
    document_id: str
    status: Literal["indexed", "duplicate", "failed", "ocr_required"]
    message: str
    duplicate_of: str | None = None
    document: DocumentInfo | None = None


class RetrievalOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    top_k: int | None = Field(default=None, ge=1, le=20)
    strategy: Literal["dense", "hybrid"] | None = None
    rerank: bool | None = None


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=2000)
    document_ids: list[str] | None = Field(default=None, max_length=100)
    collection_id: str | None = None
    retrieval: RetrievalOptions | None = None


class PassageOut(BaseModel):
    source_id: str | None = None
    chunk_id: str
    document_id: str
    filename: str
    title: str | None = None
    authors: str | None = None
    page_start: int
    page_end: int
    pages: str
    printed_page_labels: list[str] | None = None
    section: str | None = None
    text: str
    extraction_source: str
    score: float
    dense_score: float
    lexical_score: float
    rerank_score: float | None = None


class SearchResponse(BaseModel):
    query: str
    strategy: str
    results: list[PassageOut]
    timings_ms: dict[str, float]


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=1, max_length=2000)
    document_ids: list[str] | None = Field(default=None, max_length=100)
    collection_id: str | None = None
    conversation_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")
    retrieval: RetrievalOptions | None = None


class ClaimOut(BaseModel):
    sentence: str
    citations: list[str]
    support_score: float | None
    supported: bool
    issue: str | None = None


class GroundingOut(BaseModel):
    status: str
    invalid_citations: list[str]
    claims: list[ClaimOut]
    unsupported_claims: int


class EvidenceAssessmentOut(BaseModel):
    sufficient: bool
    top_dense_score: float
    term_coverage: float
    reason: str


class AskResponse(BaseModel):
    conversation_id: str
    answer: str
    status: str
    citations: list[PassageOut]
    evidence: list[PassageOut]
    grounding: GroundingOut | None
    evidence_assessment: EvidenceAssessmentOut
    retrieval_query: str
    scope_document_ids: list[str] | None
    comparison: bool
    provider: str
    model: str
    timings_ms: dict[str, float]
    page_numbering: str = "Page numbers are physical PDF page indices (1-based); printed labels are shown when the PDF defines them."


class ConversationOut(BaseModel):
    conversation_id: str
    turns: list[dict]


class CollectionCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=100)
    document_ids: list[str] = Field(default_factory=list, max_length=500)


class CollectionUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    document_ids: list[str] = Field(max_length=500)


class CollectionOut(BaseModel):
    id: str
    name: str
    created_at: str
    document_ids: list[str]


class RuntimeSettings(BaseModel):
    """Settings that may be changed at runtime from the UI, with safe bounds."""

    model_config = ConfigDict(extra="forbid")
    top_k: int = Field(ge=1, le=20)
    strategy: Literal["dense", "hybrid"]
    rerank: bool
    chunk_size: int = Field(ge=100, le=500)
    chunk_overlap: int = Field(ge=0, le=150)
    embedding_model: str = Field(min_length=1, max_length=100, pattern=r"^[\w.:/\-]+$")
    llm_provider: Literal["extractive", "ollama", "openai"]
    llm_model: str = Field(min_length=1, max_length=100, pattern=r"^[\w.:/\-]+$")


class SettingsResponse(BaseModel):
    settings: RuntimeSettings
    reindex_required: bool
    messages: list[str] = []


class RebuildRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reparse: bool = False


class MessageResponse(BaseModel):
    message: str
