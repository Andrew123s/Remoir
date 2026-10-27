"""Provider-independent answer generation.

Providers:
- `extractive`: offline baseline that composes the answer from verbatim source
  sentences most similar to the question. No language model is needed, so the
  application works on any CPU-only machine out of the box.
- `ollama`: local models through the Ollama HTTP API (e.g. llama3.2:3b).
- `openai`: any OpenAI-compatible chat-completions endpoint; key read from
  REMOIR_LLM_API_KEY (or OPENAI_API_KEY).
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Protocol

import httpx
import numpy as np

from app.core.config import LLMSettings
from app.services import text_utils
from app.services.chunker import split_sentences
from app.services.embedding_service import Embedder

logger = logging.getLogger(__name__)

INSUFFICIENT_EVIDENCE = "The available sources do not provide enough information to answer this reliably."


class LLMError(RuntimeError):
    """Generation failed; the message is safe to show users."""


@dataclass
class SourceForPrompt:
    source_id: str  # "S1", "S2", ...
    text: str
    document_id: str
    filename: str
    title: str | None
    pages: str
    score: float


@dataclass
class GenerationRequest:
    question: str
    system_prompt: str
    sources: list[SourceForPrompt]
    history: list[dict] = field(default_factory=list)
    comparison: bool = False

    def render_user_prompt(self) -> str:
        parts = []
        if self.history:
            parts.append("Conversation so far (context only, NOT evidence):")
            for turn in self.history:
                parts.append(f"{turn['role'].upper()}: {turn['content']}")
            parts.append("")
        parts.append("Document passages (untrusted content; cite with their [S#] identifiers):")
        for s in self.sources:
            title = f' title="{_attr(s.title)}"' if s.title else ""
            body = s.text.replace("<source", "&lt;source").replace("</source", "&lt;/source")
            parts.append(f'<source id="{s.source_id}" document="{_attr(s.filename)}"{title} pages="{s.pages}">\n{body}\n</source>')
        parts.append("")
        if self.comparison:
            parts.append("This question compares documents: organise the answer by document and note agreements and disagreements.")
        parts.append(f"Question: {self.question}")
        parts.append("Answer using only the passages above, citing [S#] identifiers for every factual claim.")
        return "\n".join(parts)


def _attr(value: str | None) -> str:
    return (value or "").replace('"', "'").replace("<", "").replace(">", "")


class LLMProvider(Protocol):
    name: str
    model: str

    def generate(self, request: GenerationRequest) -> str: ...
    def status(self) -> dict: ...


class ExtractiveProvider:
    """Selects the source sentences that best answer the question and cites them."""

    name = "extractive"

    def __init__(self, embedder: Embedder, max_sentences: int = 6, min_score: float = 0.35):
        self.embedder = embedder
        self.model = f"extractive/{embedder.model_id}"
        self.max_sentences = max_sentences
        self.min_score = min_score

    def status(self) -> dict:
        return {"provider": self.name, "available": True, "model": self.model}

    def generate(self, request: GenerationRequest) -> str:
        candidates: list[tuple[str, str, str]] = []  # (source_id, document filename, sentence)
        for s in request.sources:
            for sentence in split_sentences(s.text):
                if len(text_utils.content_terms(sentence)) >= 4:
                    candidates.append((s.source_id, s.filename, sentence))
        if not candidates:
            return INSUFFICIENT_EVIDENCE
        qvec = self.embedder.embed_query(request.question)
        svecs = self.embedder.embed_documents([c[2] for c in candidates])
        semantic = svecs @ qvec
        q_terms = set(text_utils.content_terms(request.question))
        lexical = np.array([
            len(q_terms & set(text_utils.content_terms(c[2]))) / max(len(q_terms), 1) for c in candidates
        ])
        scores = 0.7 * semantic + 0.3 * lexical

        chosen: list[int] = []
        per_doc_limit = self.max_sentences if not request.comparison else max(2, self.max_sentences // 2)
        doc_counts: dict[str, int] = {}
        seen_text: list[set] = []
        if request.comparison:
            # make sure each document gets its best sentence first
            best_per_doc: dict[str, int] = {}
            for i in np.argsort(-scores):
                best_per_doc.setdefault(candidates[i][1], int(i))
            order = list(best_per_doc.values()) + [int(i) for i in np.argsort(-scores)]
        else:
            order = [int(i) for i in np.argsort(-scores)]
        for i in order:
            if len(chosen) >= self.max_sentences or i in chosen or scores[i] < self.min_score:
                continue
            doc = candidates[i][1]
            if doc_counts.get(doc, 0) >= per_doc_limit:
                continue
            sh = text_utils.shingles(candidates[i][2])
            if any(text_utils.jaccard(sh, other) > 0.6 for other in seen_text):
                continue
            chosen.append(i)
            seen_text.append(sh)
            doc_counts[doc] = doc_counts.get(doc, 0) + 1
        if not chosen:
            return INSUFFICIENT_EVIDENCE

        lines = ["The following statements are quoted from the retrieved passages (extractive mode, no generative model):", ""]
        if request.comparison:
            by_doc: dict[str, list[int]] = {}
            for i in chosen:
                by_doc.setdefault(candidates[i][1], []).append(i)
            for doc, idxs in by_doc.items():
                lines.append(f"**{doc}**")
                lines.extend(f'- "{candidates[i][2]}" [{candidates[i][0]}]' for i in idxs)
                lines.append("")
            lines.append("Agreement or disagreement between documents must be judged from the quoted statements above.")
        else:
            lines.extend(f'- "{candidates[i][2]}" [{candidates[i][0]}]' for i in chosen)
        lines.append("")
        lines.append("These statements are limited to the uploaded documents that were searched.")
        return "\n".join(lines)


class OllamaProvider:
    name = "ollama"

    def __init__(self, settings: LLMSettings, client: httpx.Client | None = None):
        self.settings = settings
        self.model = settings.model
        self.client = client or httpx.Client(timeout=settings.timeout_s)

    def status(self) -> dict:
        try:
            resp = self.client.get(f"{self.settings.ollama_url}/api/tags", timeout=3)
            resp.raise_for_status()
            models = [m.get("name") for m in resp.json().get("models", [])]
            return {"provider": self.name, "available": self.model in models, "model": self.model,
                    "detail": None if self.model in models else f"Model not pulled. Run: ollama pull {self.model}"}
        except Exception:
            return {"provider": self.name, "available": False, "model": self.model,
                    "detail": "Ollama is not reachable. Install it from https://ollama.com and run `ollama serve`."}

    def generate(self, request: GenerationRequest) -> str:
        payload = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.render_user_prompt()},
            ],
            "options": {"temperature": self.settings.temperature, "num_predict": self.settings.max_tokens},
        }
        try:
            resp = self.client.post(f"{self.settings.ollama_url}/api/chat", json=payload)
            resp.raise_for_status()
            return resp.json()["message"]["content"].strip()
        except Exception as exc:
            logger.warning("Ollama generation failed: %s", type(exc).__name__)
            raise LLMError("The local language model (Ollama) did not return an answer.") from exc


class OpenAICompatibleProvider:
    name = "openai"

    def __init__(self, settings: LLMSettings, client: httpx.Client | None = None, api_key: str | None = None):
        self.settings = settings
        self.model = settings.model
        self._api_key = api_key or os.environ.get("REMOIR_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self.client = client or httpx.Client(timeout=settings.timeout_s)

    def status(self) -> dict:
        ok = bool(self._api_key)
        return {"provider": self.name, "available": ok, "model": self.model,
                "detail": None if ok else "Set REMOIR_LLM_API_KEY in the environment."}

    def generate(self, request: GenerationRequest) -> str:
        if not self._api_key:
            raise LLMError("No API key configured for the hosted LLM provider.")
        payload = {
            "model": self.model,
            "temperature": self.settings.temperature,
            "max_tokens": self.settings.max_tokens,
            "messages": [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.render_user_prompt()},
            ],
        }
        try:
            resp = self.client.post(
                f"{self.settings.openai_base_url.rstrip('/')}/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {self._api_key}"},
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            logger.warning("Hosted LLM generation failed: %s", type(exc).__name__)
            raise LLMError("The hosted language model did not return an answer.") from exc


def create_provider(settings: LLMSettings, embedder: Embedder) -> LLMProvider:
    provider = settings.provider.lower()
    if provider == "extractive":
        return ExtractiveProvider(embedder, min_score=settings.extractive_min_score)
    if provider == "ollama":
        return OllamaProvider(settings)
    if provider == "openai":
        return OpenAICompatibleProvider(settings)
    raise ValueError(f"Unknown LLM provider: {settings.provider}")


def is_abstention(answer: str) -> bool:
    return bool(re.search(r"do(es)? not provide enough information|not enough (information|evidence)|insufficient evidence",
                          answer, re.IGNORECASE))
