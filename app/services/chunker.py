"""Sentence-aware, token-budgeted chunking that keeps page and section provenance.

Text is split into sentences (headings are kept as their own units), and
sentences are packed into chunks of about `chunk_size` tokens. Consecutive
chunks share roughly `chunk_overlap` tokens of trailing sentences. A heading
closes the current chunk so chunks rarely straddle sections. Every chunk
records the pages it spans and the character range each page contributes,
which is what citations use.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from app.core.config import ChunkingSettings
from app.services.pdf_processor import PageContent

TokenCounter = Callable[[str], int]

_ABBREVIATIONS = r"(?<!\bet al)(?<!\be\.g)(?<!\bi\.e)(?<!\bFig)(?<!\bEq)(?<!\bvs)(?<!\bNo)(?<!\bDr)(?<!\bcf)"
_SENTENCE_END = re.compile(_ABBREVIATIONS + r"(?<=[.!?])[\"')\]]?\s+(?=[A-Z0-9\"'(\[])")


def approx_token_count(text: str) -> int:
    """Approximation used when the embedding tokenizer is unavailable (~1.3 tokens per word)."""
    return int(len(text.split()) * 1.3) + 1


def split_sentences(text: str) -> list[str]:
    parts = []
    for para in text.split("\n"):
        parts.extend(s.strip() for s in _SENTENCE_END.split(para) if s.strip())
    return parts


@dataclass
class _Unit:
    text: str
    page: int
    section: str | None
    tokens: int
    is_heading: bool = False


@dataclass
class Chunk:
    index: int
    text: str
    page_start: int
    page_end: int
    section: str | None
    token_count: int
    # (page, start_char, end_char) segments of `text` that came from each page.
    page_spans: list[tuple[int, int, int]] = field(default_factory=list)

    @property
    def pages(self) -> list[int]:
        return sorted({p for p, _, _ in self.page_spans})


def _split_long(text: str, max_tokens: int, count: TokenCounter) -> list[str]:
    """Split an over-long sentence on word boundaries into pieces under max_tokens."""
    words, pieces, current = text.split(), [], []
    for word in words:
        current.append(word)
        if count(" ".join(current)) > max_tokens and len(current) > 1:
            current.pop()
            pieces.append(" ".join(current))
            current = [word]
    if current:
        pieces.append(" ".join(current))
    return pieces


def _build_units(pages: list[PageContent], settings: ChunkingSettings, count: TokenCounter) -> list[_Unit]:
    units: list[_Unit] = []
    section: str | None = None
    for page in pages:
        if page.status == "failed":
            continue
        for block in page.blocks:
            if block.is_heading:
                section = block.text
                units.append(_Unit(block.text, page.number, section, count(block.text), True))
                continue
            for sentence in split_sentences(block.text):
                n = count(sentence)
                pieces = [sentence] if n <= settings.max_chunk_tokens else _split_long(sentence, settings.chunk_size, count)
                for piece in pieces:
                    units.append(_Unit(piece, page.number, section, count(piece)))
    return units


def _make_chunk(index: int, units: list[_Unit], count: TokenCounter) -> Chunk:
    parts, spans, pos = [], [], 0
    for unit in units:
        if parts:
            pos += 1  # joining space
        start = pos
        parts.append(unit.text)
        pos += len(unit.text)
        if spans and spans[-1][0] == unit.page:
            spans[-1] = (unit.page, spans[-1][1], pos)
        else:
            spans.append((unit.page, start, pos))
    text = " ".join(parts)
    section = next((u.section for u in reversed(units) if u.section), None)
    first_section = next((u.section for u in units if not u.is_heading), section)
    return Chunk(
        index=index,
        text=text,
        page_start=min(u.page for u in units),
        page_end=max(u.page for u in units),
        section=first_section or section,
        token_count=count(text),
        page_spans=spans,
    )


def chunk_pages(pages: list[PageContent], settings: ChunkingSettings, count: TokenCounter | None = None) -> list[Chunk]:
    """Chunk extracted pages. Returns an empty list for documents without text."""
    count = count or approx_token_count
    if settings.chunk_overlap >= settings.chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    units = _build_units(pages, settings, count)
    groups: list[list[_Unit]] = []
    current: list[_Unit] = []
    tokens = 0

    def flush(carry_overlap: bool) -> None:
        nonlocal current, tokens
        if not any(not u.is_heading for u in current):
            return  # a heading alone is carried into the next chunk
        groups.append(current)
        carry: list[_Unit] = []
        if carry_overlap and settings.chunk_overlap > 0:
            acc = 0
            for unit in reversed(current):
                if unit.is_heading or acc + unit.tokens > settings.chunk_overlap:
                    break
                carry.insert(0, unit)
                acc += unit.tokens
        current, tokens = carry, sum(u.tokens for u in carry)

    for unit in units:
        if unit.is_heading and tokens >= settings.min_chunk_tokens:
            flush(carry_overlap=False)
        if current and tokens + unit.tokens > settings.chunk_size:
            flush(carry_overlap=True)
        current.append(unit)
        tokens += unit.tokens
    if current and any(not u.is_heading for u in current):
        tail_tokens = sum(u.tokens for u in current)
        prev = groups[-1] if groups else None
        prev_ids = {id(u) for u in prev} if prev else set()
        new_units = [u for u in current if id(u) not in prev_ids]
        if prev is not None and tail_tokens < settings.min_chunk_tokens:
            if sum(u.tokens for u in prev) + sum(u.tokens for u in new_units) <= settings.max_chunk_tokens:
                prev.extend(new_units)
            else:
                groups.append(current)
        elif new_units:
            groups.append(current)

    return [_make_chunk(i, g, count) for i, g in enumerate(groups)]
