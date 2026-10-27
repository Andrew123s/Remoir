"""Citation parsing, validation and claim-to-evidence support checks.

Validation (deterministic, complete): every [S#] marker must refer to a source
that was retrieved and supplied for the current question; anything else is
removed from the answer and reported.

Support checking (heuristic, not full semantic entailment): each cited
sentence is compared with its cited passages using embedding similarity to the
best-matching passage sentence and content-term containment. Sentences below
the threshold are flagged as weakly supported; long factual sentences without
any citation are flagged as uncited.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from app.services import text_utils
from app.services.chunker import split_sentences
from app.services.embedding_service import Embedder

_CITATION_GROUP = re.compile(r"\[((?:S?\d+)(?:\s*[,;]\s*S?\d+)*)\]")
_META_LINE = re.compile(
    r"^(the following statements|these statements are limited|agreement or disagreement|interpretation:|"
    r"\*\*[^*]+\*\*$|note:|the available sources do not)",
    re.IGNORECASE,
)


def format_pages(page_start: int, page_end: int) -> str:
    return f"p. {page_start}" if page_start == page_end else f"pp. {page_start}–{page_end}"


@dataclass
class ClaimCheck:
    sentence: str
    citations: list[str]
    support_score: float | None
    supported: bool
    issue: str | None = None  # "weak_support", "uncited", None


@dataclass
class CitationReport:
    answer: str
    used_source_ids: list[str]
    invalid_citations: list[str] = field(default_factory=list)
    claims: list[ClaimCheck] = field(default_factory=list)

    @property
    def unsupported_claims(self) -> list[ClaimCheck]:
        return [c for c in self.claims if not c.supported]

    @property
    def grounding_status(self) -> str:
        cited = [c for c in self.claims if c.citations]
        if not cited:
            return "ungrounded" if self.claims or self.invalid_citations else "no_claims"
        if self.unsupported_claims or self.invalid_citations:
            return "partially_grounded"
        return "grounded"


def extract_citations(text: str) -> list[str]:
    ids = []
    for group in _CITATION_GROUP.findall(text):
        for token in re.split(r"\s*[,;]\s*", group):
            ids.append(token if token.startswith("S") else f"S{token}")
    return ids


def strip_invalid_citations(answer: str, valid_ids: set[str]) -> tuple[str, list[str]]:
    invalid: list[str] = []

    def replace(match: re.Match) -> str:
        tokens = [t if t.startswith("S") else f"S{t}" for t in re.split(r"\s*[,;]\s*", match.group(1))]
        keep = [t for t in tokens if t in valid_ids]
        invalid.extend(t for t in tokens if t not in valid_ids)
        return "".join(f"[{t}]" for t in keep)

    cleaned = _CITATION_GROUP.sub(replace, answer)
    return cleaned, list(dict.fromkeys(invalid))


class CitationVerifier:
    def __init__(self, embedder: Embedder, min_support: float = 0.5, uncited_min_terms: int = 6):
        self.embedder = embedder
        self.min_support = min_support
        self.uncited_min_terms = uncited_min_terms

    def verify(self, answer: str, sources: dict[str, str]) -> CitationReport:
        """`sources` maps source id (S1...) to the passage text supplied for this answer."""
        cleaned, invalid = strip_invalid_citations(answer, set(sources))
        report = CitationReport(answer=cleaned, used_source_ids=[], invalid_citations=invalid)

        claims = []
        for line in cleaned.split("\n"):
            line = line.strip().lstrip("-*• ").strip()
            if not line or _META_LINE.match(line):
                continue
            for sentence in split_sentences(line):
                # "Claim." [S1] splits into a bare citation fragment; attach it to its sentence
                if claims and not _CITATION_GROUP.sub("", sentence).strip(" .;,\"'"):
                    claims[-1] = f"{claims[-1]} {sentence}"
                elif sentence:
                    claims.append(sentence)
        pending: list[tuple[ClaimCheck, str, list[str]]] = []
        for sentence in claims:
            cites = list(dict.fromkeys(extract_citations(sentence)))
            bare = _CITATION_GROUP.sub("", sentence).strip().strip('"')
            if not cites:
                if len(text_utils.content_terms(bare)) >= self.uncited_min_terms:
                    report.claims.append(ClaimCheck(bare, [], None, False, "uncited"))
                continue
            check = ClaimCheck(bare, cites, None, True)
            report.claims.append(check)
            pending.append((check, bare, cites))
            for c in cites:
                if c not in report.used_source_ids:
                    report.used_source_ids.append(c)

        for check, bare, cites in pending:
            score = self._support(bare, [sources[c] for c in cites])
            check.support_score = round(score, 3)
            if score < self.min_support:
                check.supported, check.issue = False, "weak_support"
        return report

    def _support(self, claim: str, passages: list[str]) -> float:
        claim_terms = set(text_utils.content_terms(claim))
        best = 0.0
        claim_vec = self.embedder.embed_documents([claim])[0]
        for passage in passages:
            sentences = split_sentences(passage) or [passage]
            vecs = self.embedder.embed_documents(sentences)
            semantic = float(np.max(vecs @ claim_vec)) if len(sentences) else 0.0
            containment = (
                len(claim_terms & set(text_utils.content_terms(passage))) / len(claim_terms) if claim_terms else 0.0
            )
            best = max(best, 0.5 * semantic + 0.5 * containment)
        return best
