"""Passage retrieval: dense (FAISS) or hybrid (dense + BM25 fused with RRF),
metadata filtering, duplicate suppression, optional cross-encoder reranking and
per-document balancing for comparison questions."""

from __future__ import annotations

import math
import re
import threading
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np

from app.core.config import RetrievalSettings
from app.services import text_utils
from app.services.document_store import DocumentStore
from app.services.embedding_service import Embedder
from app.services.reranking_service import CrossEncoderReranker
from app.services.vector_store import FaissVectorStore

MIN_QUESTION_CHARS = 3
MAX_QUESTION_CHARS = 2000


class InvalidQuestionError(ValueError):
    pass


def normalize_question(question: str) -> str:
    question = re.sub(r"\s+", " ", (question or "")).strip()
    if len(question) < MIN_QUESTION_CHARS:
        raise InvalidQuestionError("The question is too short.")
    if len(question) > MAX_QUESTION_CHARS:
        raise InvalidQuestionError(f"The question exceeds {MAX_QUESTION_CHARS} characters.")
    if not text_utils.tokenize(question):
        raise InvalidQuestionError("The question contains no searchable words.")
    return question


@dataclass
class Passage:
    vector_id: int
    chunk_id: str
    document_id: str
    filename: str
    title: str | None
    authors: str | None
    year: int | None
    page_start: int
    page_end: int
    section: str | None
    text: str
    page_spans: list
    extraction_source: str
    dense_score: float = 0.0
    lexical_score: float = 0.0
    rerank_score: float | None = None
    score: float = 0.0


@dataclass
class RetrievalResult:
    query: str
    passages: list[Passage]
    strategy: str
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def top_dense_score(self) -> float:
        return max((p.dense_score for p in self.passages), default=0.0)


class BM25Index:
    """Okapi BM25 over chunk text, rebuilt lazily when documents change."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.doc_ids: list[int] = []
        self.owner: dict[int, str] = {}
        self.tfs: list[Counter] = []
        self.df: Counter = Counter()
        self.lengths: list[int] = []
        self.avgdl = 0.0

    @staticmethod
    def analyze(text: str) -> list[str]:
        return text_utils.content_terms(text)

    def build(self, rows: list[dict]) -> None:
        self.doc_ids = [r["id"] for r in rows]
        self.owner = {r["id"]: r["document_id"] for r in rows}
        self.tfs = [Counter(self.analyze(r["text"])) for r in rows]
        self.df = Counter(term for tf in self.tfs for term in tf)
        self.lengths = [sum(tf.values()) for tf in self.tfs]
        self.avgdl = sum(self.lengths) / len(self.lengths) if self.lengths else 0.0

    def search(self, query: str, k: int, allowed: set[int] | None = None) -> list[tuple[int, float]]:
        terms = set(self.analyze(query))
        if not terms or not self.doc_ids:
            return []
        n = len(self.doc_ids)
        scores = []
        for i, tf in enumerate(self.tfs):
            vid = self.doc_ids[i]
            if allowed is not None and vid not in allowed:
                continue
            s = 0.0
            for term in terms:
                f = tf.get(term)
                if not f:
                    continue
                idf = math.log(1 + (n - self.df[term] + 0.5) / (self.df[term] + 0.5))
                s += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.lengths[i] / (self.avgdl or 1)))
            if s > 0:
                scores.append((vid, s))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]


class RetrievalService:
    def __init__(self, store: DocumentStore, vectors: FaissVectorStore, embedder: Embedder, settings: RetrievalSettings):
        self.store = store
        self.vectors = vectors
        self.embedder = embedder
        self.settings = settings
        self._bm25: BM25Index | None = None
        self._bm25_lock = threading.Lock()
        self._reranker: CrossEncoderReranker | None = None

    def invalidate(self) -> None:
        self._bm25 = None

    @property
    def bm25(self) -> BM25Index:
        with self._bm25_lock:
            if self._bm25 is None:
                index = BM25Index()
                index.build(self.store.all_chunks())
                self._bm25 = index
            return self._bm25

    @property
    def reranker(self) -> CrossEncoderReranker:
        if self._reranker is None or self._reranker.model_name != self.settings.reranker_model:
            self._reranker = CrossEncoderReranker(self.settings.reranker_model)
        return self._reranker

    def retrieve(
        self,
        question: str,
        document_ids: list[str] | None = None,
        top_k: int | None = None,
        strategy: str | None = None,
        rerank: bool | None = None,
        balance_documents: bool = False,
    ) -> RetrievalResult:
        query = normalize_question(question)
        top_k = top_k or self.settings.top_k
        strategy = strategy or self.settings.strategy
        rerank = self.settings.rerank if rerank is None else rerank
        timings: dict[str, float] = {}

        allowed = self.store.chunk_ids_for_documents(document_ids) if document_ids else None
        if document_ids and not allowed:
            return RetrievalResult(query, [], strategy, timings)

        t = time.perf_counter()
        qvec = self.embedder.embed_query(query)
        timings["embed_query_ms"] = (time.perf_counter() - t) * 1000

        if balance_documents and document_ids and len(document_ids) > 1:
            per_doc = max(2, math.ceil(top_k / len(document_ids)))
            passages = []
            for doc_id in document_ids:
                doc_allowed = self.store.chunk_ids_for_documents([doc_id])
                passages.extend(self._retrieve_once(query, qvec, doc_allowed, per_doc, strategy, rerank, timings))
            passages.sort(key=lambda p: p.score, reverse=True)
            return RetrievalResult(query, passages, strategy, timings)

        passages = self._retrieve_once(query, qvec, allowed, top_k, strategy, rerank, timings)
        return RetrievalResult(query, passages, strategy, timings)

    def _retrieve_once(self, query, qvec, allowed, top_k, strategy, rerank, timings) -> list[Passage]:
        n_candidates = top_k * self.settings.candidate_multiplier
        t = time.perf_counter()
        dense = self.vectors.search(qvec, n_candidates, allowed_ids=allowed)
        timings["dense_search_ms"] = timings.get("dense_search_ms", 0) + (time.perf_counter() - t) * 1000
        dense_scores = dict(dense)
        lexical_scores: dict[int, float] = {}

        if strategy == "hybrid":
            t = time.perf_counter()
            lexical = self.bm25.search(query, n_candidates, set(allowed) if allowed is not None else None)
            timings["lexical_search_ms"] = timings.get("lexical_search_ms", 0) + (time.perf_counter() - t) * 1000
            lexical_scores = dict(lexical)
            fused: dict[int, float] = defaultdict(float)
            for ranking in (dense, lexical):
                for rank, (vid, _) in enumerate(ranking):
                    fused[vid] += 1.0 / (self.settings.rrf_k + rank + 1)
            order = sorted(fused, key=fused.get, reverse=True)
            missing = [vid for vid in order if vid not in dense_scores]
            dense_scores.update(self._dense_scores(qvec, missing))
            scores = dict(fused)
        elif strategy == "dense":
            order = [vid for vid, _ in dense]
            scores = dict(dense)
        else:
            raise ValueError(f"Unknown retrieval strategy: {strategy}")

        rows = self.store.get_chunks(order)
        candidates = []
        for vid in order:
            row = rows.get(vid)
            if row is None:
                continue  # vector without metadata (should not happen); never surface it
            candidates.append(
                Passage(
                    vector_id=vid, chunk_id=row["chunk_id"], document_id=row["document_id"], filename=row["filename"],
                    title=row["title"], authors=row["authors"], year=row["year"], page_start=row["page_start"],
                    page_end=row["page_end"], section=row["section"], text=row["text"], page_spans=row["page_spans"],
                    extraction_source=row["extraction_source"] or "native", dense_score=dense_scores.get(vid, 0.0),
                    lexical_score=lexical_scores.get(vid, 0.0), score=scores[vid],
                )
            )

        candidates = self._deduplicate(candidates)
        if rerank and candidates:
            t = time.perf_counter()
            pool = candidates[: max(top_k * 3, 10)]
            for passage, s in zip(pool, self.reranker.score(query, [p.text for p in pool])):
                passage.rerank_score = s
                passage.score = s
            candidates = sorted(pool, key=lambda p: p.score, reverse=True)
            timings["rerank_ms"] = timings.get("rerank_ms", 0) + (time.perf_counter() - t) * 1000
        return candidates[:top_k]

    def _dense_scores(self, qvec: np.ndarray, ids: list[int]) -> dict[int, float]:
        out = {}
        for vid in ids:
            try:
                out[vid] = float(np.dot(self.vectors.index.reconstruct(int(vid)), qvec))
            except Exception:
                out[vid] = 0.0
        return out

    def _deduplicate(self, passages: list[Passage]) -> list[Passage]:
        kept, kept_shingles = [], []
        for passage in passages:
            sh = text_utils.shingles(passage.text)
            if any(text_utils.jaccard(sh, other) >= self.settings.dedup_threshold for other in kept_shingles):
                continue
            kept.append(passage)
            kept_shingles.append(sh)
        return kept
