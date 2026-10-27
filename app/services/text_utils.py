"""Shared lexical helpers: tokenization, stop words, light stemming, overlap."""

from __future__ import annotations

import re

STOPWORDS = frozenset(
    """a about above after again against all also am an and any are as at be because been before being below
    between both but by can could did do does doing down during each few for from further had has have having he
    her here hers him his how i if in into is it its itself just me more most my no nor not now of off on once
    only or other our ours out over own same she should so some such than that the their theirs them then there
    these they this those through to too under until up very was we were what when where which while who whom why
    will with would you your yours paper papers document documents study studies authors author say says said
    tell describe explain summarize summarise according discuss discusses discussed provide provides main
    what's please using use used uploaded source sources identify find show give list""".split()
)

_WORD = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def stem(word: str) -> str:
    """Very light suffix stripping, enough to match 'yields'/'yield', 'limitations'/'limitation'."""
    for suffix in ("ations", "ation", "ings", "ing", "ies", "ied", "es", "ed", "ly", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)]
    return word


def content_terms(text: str) -> list[str]:
    return [stem(t) for t in tokenize(text) if t not in STOPWORDS and len(t) > 2]


def term_coverage(query: str, passages: list[str]) -> float:
    """Fraction of the query's content terms that appear in any of the passages."""
    terms = set(content_terms(query))
    if not terms:
        return 0.0
    present = set()
    for passage in passages:
        present.update(content_terms(passage))
    return len(terms & present) / len(terms)


def shingles(text: str, n: int = 3) -> set[tuple[str, ...]]:
    tokens = tokenize(text)
    return {tuple(tokens[i : i + n]) for i in range(max(len(tokens) - n + 1, 1))}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0
