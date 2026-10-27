"""Retrieval and answer evaluation metrics.

Relevance is defined by labelled (document, page, evidence phrase) items: a
retrieved passage is relevant to an item when it comes from the item's
document and contains the evidence phrase (case-insensitive).
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def matches(passage: dict, item: dict) -> bool:
    return passage["filename"] == item["document"] and _norm(item["evidence"]) in _norm(passage["text"])


def page_matches(passage: dict, item: dict) -> bool:
    return matches(passage, item) and passage["page_start"] <= item["page"] <= passage["page_end"]


def retrieval_metrics(retrieved: list[dict], relevant: list[dict], k: int) -> dict[str, float]:
    """Precision@k, Recall@k, Hit@k and reciprocal rank for one query."""
    top = retrieved[:k]
    rel_flags = [any(matches(p, item) for item in relevant) for p in top]
    covered = sum(1 for item in relevant if any(matches(p, item) for p in top))
    first = next((i for i, flag in enumerate(rel_flags) if flag), None)
    return {
        "precision": sum(rel_flags) / k,
        "recall": covered / len(relevant) if relevant else 0.0,
        "hit": float(first is not None),
        "rr": 1.0 / (first + 1) if first is not None else 0.0,
    }


def mean_metrics(rows: Iterable[dict[str, float]]) -> dict[str, float]:
    rows = list(rows)
    if not rows:
        return {}
    return {key: round(statistics.fmean(r[key] for r in rows), 4) for key in rows[0]}


def latency_summary(values_ms: list[float]) -> dict[str, float]:
    if not values_ms:
        return {}
    ordered = sorted(values_ms)
    p95 = ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]
    return {"mean_ms": round(statistics.fmean(ordered), 2), "p95_ms": round(p95, 2), "max_ms": round(ordered[-1], 2)}


def key_fact_coverage(answer: str, key_facts: list[str]) -> float | None:
    if not key_facts:
        return None
    text = _norm(answer)
    return sum(_norm(f) in text for f in key_facts) / len(key_facts)
