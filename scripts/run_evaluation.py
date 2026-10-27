"""Run the full evaluation on the synthetic fixtures and write reports.

Builds an isolated index in evaluation/workdir (your real data/ is untouched),
measures ingestion, retrieval and answer quality, and writes
reports/evaluation_results.json and reports/evaluation_report.md.

Usage:  python scripts/run_evaluation.py [--skip-rerank]
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.config import load_settings  # noqa: E402
from app.core.logging import setup_logging  # noqa: E402
from app.dependencies import Container  # noqa: E402
from evaluation.evaluate_answers import (  # noqa: E402
    citation_stress_test,
    evaluate_answers,
    evaluate_follow_ups,
    extractive_floor_sweep,
)  # noqa: E402
from evaluation.evaluate_retrieval import CONFIGS, evaluate_abstention_thresholds, evaluate_retrieval  # noqa: E402
from scripts import make_fixtures  # noqa: E402
from scripts.report_writer import write_markdown  # noqa: E402

WORKDIR = ROOT / "evaluation" / "workdir"
RESULTS_JSON = ROOT / "reports" / "evaluation_results.json"
REPORT_MD = ROOT / "reports" / "evaluation_report.md"


def rss_mb() -> float:
    return round(psutil.Process().memory_info().rss / 1024 / 1024, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-rerank", action="store_true", help="skip cross-encoder configs (avoids a ~90 MB download)")
    args = parser.parse_args()
    setup_logging("WARNING")

    dataset = json.loads((ROOT / "evaluation" / "evaluation_dataset.json").read_text(encoding="utf-8"))
    shutil.rmtree(WORKDIR, ignore_errors=True)
    fixtures = WORKDIR / "fixtures"
    make_fixtures.main(fixtures)

    settings = load_settings()
    settings.app.data_dir = str(WORKDIR / "data")
    settings.llm.provider = "extractive"  # the only provider guaranteed to exist offline
    mem_start = rss_mb()
    t = time.perf_counter()
    container = Container(settings)
    container.embedder.embed_documents(["model warm-up"])
    model_load_s = time.perf_counter() - t
    mem_model = rss_mb()

    ingestion = []
    for name in dataset["documents"]:
        data = (fixtures / name).read_bytes()
        t = time.perf_counter()
        outcome = container.ingestion.ingest_bytes(name, data)
        elapsed = time.perf_counter() - t
        doc = container.store.get_document(outcome.document_id)
        ingestion.append({"document": name, "status": outcome.status, "pages": doc["page_count"],
                          "chunks": doc["chunk_count"], "seconds": round(elapsed, 3), "size_kb": round(len(data) / 1024, 1)})
    container.retriever.invalidate()

    texts = [c["text"] for c in container.store.all_chunks()]
    t = time.perf_counter()
    for _ in range(3):
        container.embedder.embed_documents(texts)
    embed_s = (time.perf_counter() - t) / 3
    token_counts = [c["token_count"] for d in container.store.list_documents() for c in container.store.document_chunks(d["id"])]

    configs = [c for c in CONFIGS if not (args.skip_rerank and c["rerank"])]
    retrieval = evaluate_retrieval(container, dataset, configs)
    abstention = evaluate_abstention_thresholds(container, dataset)
    answers = evaluate_answers(container, dataset)
    follow_ups = evaluate_follow_ups(container, dataset)
    stress = citation_stress_test(container, dataset)
    floor_sweep = extractive_floor_sweep(container, dataset)
    mem_end = rss_mb()

    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": {"name": dataset["name"], "description": dataset["description"],
                    "relevance_procedure": dataset["relevance_procedure"],
                    "n_questions": len(dataset["questions"]), "n_conversations": len(dataset.get("conversations", []))},
        "environment": {
            "python": platform.python_version(), "platform": platform.platform(),
            "processor": platform.processor(), "cpu_count": psutil.cpu_count(),
            "ram_gb": round(psutil.virtual_memory().total / 1024 ** 3, 1), "gpu": "none (CPU only)",
        },
        "configuration": {
            "embedding_model": container.embedder.model_id,
            "embedding_dimension": container.embedder.dimension,
            "chunk_size": settings.chunking.chunk_size, "chunk_overlap": settings.chunking.chunk_overlap,
            "default_strategy": settings.retrieval.strategy, "default_rerank": settings.retrieval.rerank,
            "top_k": settings.retrieval.top_k, "reranker_model": settings.retrieval.reranker_model,
            "llm_provider": container.provider.name, "llm_model": container.provider.model,
            "min_dense_score": settings.evidence.min_dense_score, "min_term_coverage": settings.evidence.min_term_coverage,
            "min_support_score": settings.evidence.min_support_score,
            "extractive_relative_floor": settings.llm.extractive_relative_floor,
        },
        "performance": {
            "model_load_seconds": round(model_load_s, 2),
            "ingestion": ingestion,
            "total_chunks": len(texts),
            "chunk_tokens": {"min": min(token_counts), "max": max(token_counts), "mean": round(sum(token_counts) / len(token_counts), 1)},
            "embedding_seconds_all_chunks": round(embed_s, 3),
            "embedding_chunks_per_second": round(len(texts) / embed_s, 1),
            "memory_mb": {"process_start": mem_start, "after_model_load": mem_model, "end": mem_end},
        },
        "retrieval": retrieval,
        "abstention_thresholds": abstention,
        "answers": answers,
        "follow_ups": follow_ups,
        "citation_stress_test": stress,
        "extractive_floor_sweep": floor_sweep,
    }
    RESULTS_JSON.parent.mkdir(exist_ok=True)
    RESULTS_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")
    REPORT_MD.write_text(write_markdown(results), encoding="utf-8")
    container.store.close()
    print(f"Wrote {RESULTS_JSON.relative_to(ROOT)} and {REPORT_MD.relative_to(ROOT)}")
    for name, r in retrieval["results"].items():
        print(f"{name:34s} " + " ".join(f"R@{k}={r['metrics'][f'@{k}']['recall']:.3f}" for k in retrieval["ks"]) + f" MRR={r['mrr']:.3f}")
    print(json.dumps(answers["summary"], indent=2))


if __name__ == "__main__":
    main()
