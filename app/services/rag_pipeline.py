"""Question -> retrieval -> evidence check -> generation -> citation verification."""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field

from app.core.config import Settings
from app.services import text_utils
from app.services.citation_service import CitationReport, CitationVerifier, format_pages
from app.services.document_store import DocumentStore
from app.services.llm_service import (
    INSUFFICIENT_EVIDENCE,
    GenerationRequest,
    LLMError,
    LLMProvider,
    SourceForPrompt,
    is_abstention,
)
from app.services.retrieval_service import Passage, RetrievalService, normalize_question

logger = logging.getLogger(__name__)

_COMPARISON = re.compile(
    r"\b(compare|comparison|comparing|contrast|differ|differences?|disagree\w*|agree\w*|versus|vs\.?|across|"
    r"each (paper|document|study)|both|which (papers|documents|studies))\b",
    re.IGNORECASE,
)
_FOLLOW_UP = re.compile(
    r"\b(it|its|they|their|them|this|that|these|those|the (paper|study|document|authors|report)|same|also|"
    r"further|more about|what else)\b",
    re.IGNORECASE,
)
_SINGULAR_REFERENCE = re.compile(r"\b(it|its|this (paper|study|document|report)|the (paper|study|document|report))\b", re.IGNORECASE)


@dataclass
class EvidenceAssessment:
    sufficient: bool
    top_dense_score: float
    term_coverage: float
    reason: str


@dataclass
class RAGAnswer:
    answer: str
    status: str  # "answered", "insufficient_evidence", "generation_failed"
    sources: list[dict]
    evidence: list[dict]
    citation_report: CitationReport | None
    assessment: EvidenceAssessment
    retrieval_query: str
    scope_document_ids: list[str] | None
    comparison: bool
    provider: str
    model: str
    timings: dict[str, float] = field(default_factory=dict)


def passage_to_dict(p: Passage, source_id: str | None = None, page_labels: dict | None = None) -> dict:
    labels = page_labels or {}
    printed = [labels.get(n) for n in range(p.page_start, p.page_end + 1) if labels.get(n)]
    return {
        "source_id": source_id,
        "chunk_id": p.chunk_id,
        "document_id": p.document_id,
        "filename": p.filename,
        "title": p.title,
        "authors": p.authors,
        "page_start": p.page_start,
        "page_end": p.page_end,
        "pages": format_pages(p.page_start, p.page_end),
        "printed_page_labels": printed or None,
        "section": p.section,
        "text": p.text,
        "extraction_source": p.extraction_source,
        "score": round(p.score, 4),
        "dense_score": round(p.dense_score, 4),
        "lexical_score": round(p.lexical_score, 4),
        "rerank_score": None if p.rerank_score is None else round(p.rerank_score, 4),
    }


class RAGPipeline:
    def __init__(
        self,
        settings: Settings,
        store: DocumentStore,
        retriever: RetrievalService,
        provider: LLMProvider,
        verifier: CitationVerifier,
        system_prompt: str,
    ):
        self.settings = settings
        self.store = store
        self.retriever = retriever
        self.provider = provider
        self.verifier = verifier
        self.system_prompt = system_prompt

    # Scope and conversation handling ----------------------------------------
    def mentioned_documents(self, question: str) -> list[str]:
        q = question.lower()
        found = []
        for doc in self.store.list_documents():
            if doc["status"] != "indexed":
                continue
            names = {doc["filename"].lower(), doc["filename"].rsplit(".", 1)[0].lower()}
            names.add(doc["filename"].rsplit(".", 1)[0].lower().replace("_", " ").replace("-", " "))
            if doc.get("title") and doc.get("title_source") == "pdf_metadata":
                names.add(doc["title"].lower())
            if any(len(n) >= 5 and n in q for n in names):
                found.append(doc["id"])
        return found

    def contextualize(self, question: str, history: list[dict]) -> tuple[str, list[str] | None]:
        """Return the retrieval query and, for singular references, the previously cited documents."""
        last_user = next((t for t in reversed(history) if t["role"] == "user"), None)
        last_assistant = next((t for t in reversed(history) if t["role"] == "assistant"), None)
        if not last_user:
            return question, None
        is_follow_up = bool(_FOLLOW_UP.search(question)) or len(text_utils.content_terms(question)) <= 2
        if not is_follow_up:
            return question, None
        query = f"{last_user['content']} {question}"
        scope = None
        if _SINGULAR_REFERENCE.search(question) and last_assistant and last_assistant.get("document_ids"):
            scope = last_assistant["document_ids"]
        return query, scope

    def history_for_prompt(self, history: list[dict]) -> list[dict]:
        budget = self.settings.conversation.max_history_chars
        out, used = [], 0
        for turn in reversed(history):
            content = turn["content"]
            if turn["role"] == "assistant":
                content = re.sub(r"\[S\d+\]", "", content)[:400]  # old citations are not valid now
            if used + len(content) > budget:
                break
            out.insert(0, {"role": turn["role"], "content": content})
            used += len(content)
        return out

    # Evidence ---------------------------------------------------------------
    def assess(self, question: str, passages: list[Passage]) -> EvidenceAssessment:
        ev = self.settings.evidence
        if not passages:
            return EvidenceAssessment(False, 0.0, 0.0, "No passages were retrieved from the selected documents.")
        top = max(p.dense_score for p in passages)
        coverage = text_utils.term_coverage(question, [p.text for p in passages[:4]])
        if top < ev.min_dense_score:
            return EvidenceAssessment(False, top, coverage, "Retrieved passages are not semantically close to the question.")
        if coverage < ev.min_term_coverage:
            return EvidenceAssessment(False, top, coverage, "Key terms of the question do not appear in the retrieved passages.")
        return EvidenceAssessment(True, top, coverage, "Relevant passages found.")

    def build_context(self, passages: list[Passage]) -> list[tuple[str, Passage]]:
        budget = self.settings.retrieval.max_context_tokens
        selected, used = [], 0
        for p in passages:
            tokens = int(len(p.text.split()) * 1.3)
            if selected and used + tokens > budget:
                break
            selected.append((f"S{len(selected) + 1}", p))
            used += tokens
        return selected

    def _page_labels(self, doc_ids: set[str]) -> dict[str, dict[int, str]]:
        out = {}
        for doc_id in doc_ids:
            doc = self.store.get_document(doc_id) or {}
            out[doc_id] = {
                pg["page"]: pg["label"] for pg in doc.get("pages", []) if pg.get("label") and pg["label"] != str(pg["page"])
            }
        return out

    # Main entry -------------------------------------------------------------
    def answer(
        self,
        question: str,
        document_ids: list[str] | None = None,
        history: list[dict] | None = None,
        top_k: int | None = None,
        strategy: str | None = None,
        rerank: bool | None = None,
    ) -> RAGAnswer:
        timings: dict[str, float] = {}
        started = time.perf_counter()
        question = normalize_question(question)
        history = history or []

        retrieval_query, follow_up_scope = self.contextualize(question, history)
        scope = document_ids or self.mentioned_documents(question) or follow_up_scope or None
        comparison = bool(_COMPARISON.search(question))
        balance = comparison and scope is not None and len(scope) > 1
        if comparison and scope is None:
            indexed = [d["id"] for d in self.store.list_documents() if d["status"] == "indexed"]
            if 1 < len(indexed) <= 10:
                scope, balance = indexed, True

        result = self.retriever.retrieve(
            retrieval_query, document_ids=scope, top_k=top_k, strategy=strategy, rerank=rerank,
            balance_documents=balance,
        )
        passages = result.passages
        if balance:
            # balanced retrieval can pull weak passages from unrelated documents
            floor = self.settings.evidence.min_dense_score
            passages = [p for p in passages if p.dense_score >= floor or p.lexical_score > 0]
        timings.update(result.timings)
        assessment = self.assess(question if not follow_up_scope else retrieval_query, passages)
        labels = self._page_labels({p.document_id for p in passages})
        evidence = [passage_to_dict(p, page_labels=labels.get(p.document_id)) for p in passages]

        base = dict(
            evidence=evidence, assessment=assessment, retrieval_query=retrieval_query, scope_document_ids=scope,
            comparison=comparison, provider=self.provider.name, model=self.provider.model,
        )
        if not assessment.sufficient:
            timings["total_ms"] = (time.perf_counter() - started) * 1000
            return RAGAnswer(
                answer=f"{INSUFFICIENT_EVIDENCE} {assessment.reason}", status="insufficient_evidence", sources=[],
                citation_report=None, timings=timings, **base,
            )

        context = self.build_context(passages)
        request = GenerationRequest(
            question=question,
            system_prompt=self.system_prompt,
            sources=[
                SourceForPrompt(sid, p.text, p.document_id, p.filename, p.title, format_pages(p.page_start, p.page_end), p.score)
                for sid, p in context
            ],
            history=self.history_for_prompt(history),
            comparison=comparison,
        )
        t = time.perf_counter()
        try:
            raw_answer = self.provider.generate(request)
        except LLMError as exc:
            timings["total_ms"] = (time.perf_counter() - started) * 1000
            return RAGAnswer(answer=str(exc), status="generation_failed", sources=[], citation_report=None,
                             timings=timings, **base)
        timings["generation_ms"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        source_map = {sid: p for sid, p in context}
        report = self.verifier.verify(raw_answer, {sid: p.text for sid, p in context})
        timings["verification_ms"] = (time.perf_counter() - t) * 1000

        if is_abstention(report.answer) and not report.used_source_ids:
            status = "insufficient_evidence"
        else:
            status = "answered"
        answer_text = report.answer
        if status == "answered" and not report.used_source_ids:
            answer_text += "\n\n_Warning: this answer contains no valid citations and is not grounded in the sources._"
        sources = [
            passage_to_dict(source_map[sid], sid, labels.get(source_map[sid].document_id))
            for sid in report.used_source_ids
        ]
        timings["total_ms"] = (time.perf_counter() - started) * 1000
        return RAGAnswer(answer=answer_text, status=status, sources=sources, citation_report=report,
                         timings=timings, **base)
