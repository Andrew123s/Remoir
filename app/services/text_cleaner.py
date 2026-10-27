"""Conservative cleanup of PDF extraction artifacts.

The aim is to repair layout damage (hyphenated line breaks, hard-wrapped lines,
running headers/footers, stray page numbers) while keeping headings, tables,
numbers and reference lists intact.
"""

from __future__ import annotations

import re
from collections import Counter

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f￾￿]")
_LIGATURES = {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl", "­": ""}
_HYPHEN_BREAK = re.compile(r"(\w)-\n(?=[a-z])")
_PAGE_NUMBER_LINE = re.compile(r"^\s*(page\s+)?\d{1,4}(\s+of\s+\d{1,4})?\s*$", re.IGNORECASE)


def normalize_chars(text: str) -> str:
    for src, dst in _LIGATURES.items():
        text = text.replace(src, dst)
    text = _CONTROL_CHARS.sub("", text)
    return text.replace(" ", " ").replace("\t", " ")


def clean_block(text: str) -> str:
    """Clean one layout block (roughly a paragraph) of extracted text."""
    text = normalize_chars(text)
    text = _HYPHEN_BREAK.sub(r"\1", text)
    lines = [ln.strip() for ln in text.split("\n")]
    lines = [ln for ln in lines if ln and not _PAGE_NUMBER_LINE.match(ln)]
    # Lines that look like table rows (several runs of 2+ spaces or many numbers)
    # keep their line breaks; prose lines are joined.
    if lines and sum(1 for ln in lines if re.search(r"\S {2,}\S", ln)) >= max(2, len(lines) // 2):
        return "\n".join(re.sub(r" {2,}", " | ", ln) for ln in lines)
    joined = " ".join(lines)
    return re.sub(r" {2,}", " ", joined).strip()


def find_repeated_lines(pages_blocks: list[list[str]], min_pages: int = 3, ratio: float = 0.5) -> set[str]:
    """Short lines repeated on most pages are running headers or footers."""
    if len(pages_blocks) < min_pages:
        return set()
    counts: Counter[str] = Counter()
    for blocks in pages_blocks:
        seen = {b.strip().lower() for b in blocks if 0 < len(b.strip()) <= 80}
        counts.update(seen)
    threshold = max(min_pages, int(len(pages_blocks) * ratio))
    return {line for line, n in counts.items() if n >= threshold}


def is_low_quality(text: str) -> bool:
    """Heuristic for garbled extraction: too few letters relative to length."""
    if not text:
        return True
    letters = sum(ch.isalpha() for ch in text)
    return letters / max(len(text), 1) < 0.4
