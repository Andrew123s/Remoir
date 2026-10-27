"""Retrieval evaluation: Precision@k, Recall@k, Hit@k and MRR per strategy.

Run through scripts/run_evaluation.py, which prepares an isolated index.
"""

from __future__ import annotations

import time

from app.dependencies import Container
from app.services.evaluation_service import latency_summary, mean_metrics, retrieval_metrics
from app.services.rag_pipeline import missing_entities, passage_to_dict
from app.services.text_utils import term_coverage

CONFIGS = [
    {"name": "dense", "strategy": "dense", "rerank": False},
    {"name": "hybrid (dense + BM25, RRF)", "strategy": "hybrid", "rerank": False},
    {"name": "dense + cross-encoder rerank", "strategy": "dense", "rerank": True},
    {"name": "hybrid + cross-encoder rerank", "strategy": "hybrid", "rerank": True},
]
KS = [1, 3, 5]


def evaluate_retrieval(container: Container, dataset: dict, configs=CONFIGS, ks=KS) -> dict:
    answerable = [q for q in dataset["questions"] if q["relevant"]]
    results = {}
    for cfg in configs:
        container.retriever.retrieve("warm up", top_k=1, strategy=cfg["strategy"], rerank=cfg["rerank"])
        per_k = {k: [] for k in ks}
        latencies, per_question = [], []
        for q in answerable:
            t = time.perf_counter()
            res = container.retriever.retrieve(q["question"], top_k=max(ks), strategy=cfg["strategy"], rerank=cfg["rerank"])
            latencies.append((time.perf_counter() - t) * 1000)
            passages = [passage_to_dict(p) for p in res.passages]
            for k in ks:
                per_k[k].append(retrieval_metrics(passages, q["relevant"], k))
            m5 = retrieval_metrics(passages, q["relevant"], max(ks))
            per_question.append({"id": q["id"], "type": q["type"], "hit@5": m5["hit"], "rr": m5["rr"],
                                 "recall@5": m5["recall"], "top": passages[0]["filename"] if passages else None})
        results[cfg["name"]] = {
            "config": cfg,
            "metrics": {f"@{k}": mean_metrics(per_k[k]) for k in ks},
            "mrr": mean_metrics(per_k[max(ks)])["rr"],
            "latency": latency_summary(latencies),
            "per_question": per_question,
        }
    return {"n_questions": len(answerable), "ks": ks, "results": results}


def evaluate_abstention_thresholds(container: Container, dataset: dict, thresholds=None, coverage=None) -> dict:
    """Sweep the evidence-sufficiency gate on retrieval signals only (no generation)."""
    thresholds = thresholds or [0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
    coverage = container.settings.evidence.min_term_coverage if coverage is None else coverage
    signals = []
    for q in dataset["questions"]:
        res = container.retriever.retrieve(q["question"])
        top = max((p.dense_score for p in res.passages), default=0.0)
        cov = term_coverage(q["question"], [p.text for p in res.passages[:4]])
        signals.append({"id": q["id"], "answerable": bool(q["relevant"]), "top_dense": round(top, 4), "coverage": round(cov, 3),
                        "missing_entities": missing_entities(q["question"], [p.text for p in res.passages])})
    sweep = []
    for use_entities in (False, True):
        for th in thresholds:
            def passes(s):
                return s["top_dense"] >= th and s["coverage"] >= coverage and not (use_entities and s["missing_entities"])
            n_ans = sum(s["answerable"] for s in signals)
            n_un = len(signals) - n_ans
            tp = sum(1 for s in signals if s["answerable"] and passes(s))
            tn = sum(1 for s in signals if not s["answerable"] and not passes(s))
            sweep.append({"entity_check": use_entities, "min_dense_score": th, "answerable_answered": round(tp / n_ans, 4),
                          "unanswerable_abstained": round(tn / n_un, 4) if n_un else None,
                          "accuracy": round((tp + tn) / len(signals), 4)})
    return {"min_term_coverage": coverage, "signals": signals, "sweep": sweep}
