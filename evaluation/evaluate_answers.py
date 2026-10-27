"""Answer evaluation: abstention, citation validity/relevance, page accuracy,
key-fact coverage, claim support and a citation-fabrication stress test.

All checks are deterministic string/ID comparisons against the labelled
dataset, plus the pipeline's own heuristic claim-support score (reported as
such, not as ground truth).
"""

from __future__ import annotations

import statistics

from app.dependencies import Container
from app.services.evaluation_service import key_fact_coverage, latency_summary, matches, page_matches
from app.services.rag_pipeline import cited_documents


def _mean(values):
    values = [v for v in values if v is not None]
    return round(statistics.fmean(values), 4) if values else None


def evaluate_answers(container: Container, dataset: dict) -> dict:
    rows, gen_latency, total_latency = [], [], []
    for q in dataset["questions"]:
        r = container.pipeline.answer(q["question"])
        answerable = bool(q["relevant"])
        answered = r.status == "answered"
        report = r.citation_report
        cited = r.sources
        row = {
            "id": q["id"], "type": q["type"], "answerable": answerable, "status": r.status,
            "correct_abstention_decision": answered == answerable,
            "n_citations": len(cited),
            "invalid_citations": len(report.invalid_citations) if report else 0,
            "citation_relevance": _mean([float(any(matches(c, it) for it in q["relevant"])) for c in cited]) if answerable and cited else None,
            "cited_doc_correct": _mean([float(any(c["filename"] == it["document"] for it in q["relevant"])) for c in cited]) if answerable and cited else None,
            "evidence_page_cited": float(any(page_matches(c, it) for c in cited for it in q["relevant"])) if answerable and answered else None,
            "key_fact_coverage": key_fact_coverage(r.answer, q.get("key_facts", [])) if answerable and answered else None,
            "claims": len(report.claims) if report else 0,
            "supported_claims": sum(c.supported for c in report.claims) if report else 0,
            "uncited_claims": sum(c.issue == "uncited" for c in report.claims) if report else 0,
            "grounding": report.grounding_status if report else None,
            "total_ms": round(r.timings.get("total_ms", 0), 1),
        }
        rows.append(row)
        total_latency.append(r.timings.get("total_ms", 0))
        if "generation_ms" in r.timings:
            gen_latency.append(r.timings["generation_ms"])

    ans = [x for x in rows if x["answerable"]]
    unans = [x for x in rows if not x["answerable"]]
    answered = [x for x in ans if x["status"] == "answered"]
    total_claims = sum(x["claims"] for x in rows)
    summary = {
        "provider": container.provider.name,
        "model": container.provider.model,
        "n_answerable": len(ans),
        "n_unanswerable": len(unans),
        "answer_rate_answerable": _mean([float(x["status"] == "answered") for x in ans]),
        "abstention_rate_unanswerable": _mean([float(x["status"] != "answered") for x in unans]),
        "abstention_decision_accuracy": _mean([float(x["correct_abstention_decision"]) for x in rows]),
        "citation_validity": round(1 - sum(x["invalid_citations"] for x in rows) / max(sum(x["n_citations"] + x["invalid_citations"] for x in rows), 1), 4),
        "answers_with_citations": _mean([float(x["n_citations"] > 0) for x in answered]),
        "citation_relevance": _mean([x["citation_relevance"] for x in answered]),
        "cited_document_correct": _mean([x["cited_doc_correct"] for x in answered]),
        "labelled_evidence_cited": _mean([x["evidence_page_cited"] for x in answered]),
        "key_fact_coverage": _mean([x["key_fact_coverage"] for x in answered]),
        "claim_support_rate": round(sum(x["supported_claims"] for x in rows) / total_claims, 4) if total_claims else None,
        "unsupported_claims": sum(x["claims"] - x["supported_claims"] for x in rows),
        "generation_latency": latency_summary(gen_latency),
        "total_latency": latency_summary(total_latency),
    }
    return {"summary": summary, "per_question": rows}


def evaluate_follow_ups(container: Container, dataset: dict) -> dict:
    """Second turns must retrieve the labelled evidence using the conversation context."""
    rows = []
    for conv in dataset.get("conversations", []):
        history = []
        for turn in conv["turns"]:
            r = container.pipeline.answer(turn["question"], history=history)
            if turn.get("relevant"):
                hit = any(matches(e, it) for e in r.evidence for it in turn["relevant"])
                scoped = r.scope_document_ids
                rows.append({"id": conv["id"], "question": turn["question"], "evidence_retrieved": hit,
                             "retrieval_query": r.retrieval_query, "status": r.status,
                             "scoped_to_previous_document": bool(scoped) and len(scoped) == 1})
            history += [{"role": "user", "content": turn["question"]},
                        {"role": "assistant", "content": r.answer, "document_ids": cited_documents(r.sources)}]
    return {"n": len(rows), "evidence_hit_rate": _mean([float(x["evidence_retrieved"]) for x in rows]), "rows": rows}


class _FabricatingLLM:
    """Adversarial stand-in that mixes a valid citation with fabricated ones."""

    name, model = "adversarial-test", "fabricating"

    def status(self):
        return {"provider": self.name, "available": True, "model": self.model}

    def generate(self, request):
        first = request.sources[0]
        sentence = first.text.split(". ")[0].split("\n")[-1].rstrip(".") + "."
        return (f"{sentence} [{first.source_id}]\n"
                f"The authors also proved that pesticides have no environmental impact [S{len(request.sources) + 7}].\n"
                f"Global maize yields will triple by 2030 according to Smith et al. (2019) [{first.source_id}].\n"
                f"Every farmer in the world has adopted drought-tolerant seed since the year 2000.")


def citation_stress_test(container: Container, dataset: dict) -> dict:
    """Checks that fabricated citations are removed and unsupported claims flagged."""
    original = container.pipeline.provider
    container.pipeline.provider = _FabricatingLLM()
    try:
        rows = []
        for q in [q for q in dataset["questions"] if q["relevant"]][:10]:
            r = container.pipeline.answer(q["question"])
            if r.citation_report is None:
                continue
            claims = r.citation_report.claims
            fabricated_valid_cite = [c for c in claims if "triple by 2030" in c.sentence]
            uncited = [c for c in claims if "every farmer" in c.sentence.lower()]
            rows.append({
                "id": q["id"],
                "nonexistent_citation_removed": bool(r.citation_report.invalid_citations)
                and not any(f"[{i}]" in r.answer for i in r.citation_report.invalid_citations),
                "misattributed_claim_flagged": bool(fabricated_valid_cite) and not fabricated_valid_cite[0].supported,
                "uncited_claim_flagged": bool(uncited) and uncited[0].issue == "uncited",
                "genuine_claim_supported": bool(claims) and claims[0].supported,
            })
    finally:
        container.pipeline.provider = original
    keys = ["nonexistent_citation_removed", "misattributed_claim_flagged", "uncited_claim_flagged", "genuine_claim_supported"]
    return {"n": len(rows), **{k: _mean([float(r[k]) for r in rows]) for k in keys}, "rows": rows}


def extractive_floor_sweep(container: Container, dataset: dict, floors=(0.0, 0.7, 0.8, 0.85)) -> list[dict]:
    """Trade-off between citation precision and fact coverage for the extractive provider."""
    provider = container.pipeline.provider
    if not hasattr(provider, "relative_floor"):
        return []
    original, rows = provider.relative_floor, []
    try:
        for floor in floors:
            provider.relative_floor = floor
            s = evaluate_answers(container, dataset)["summary"]
            rows.append({"relative_floor": floor, **{k: s[k] for k in (
                "citation_relevance", "cited_document_correct", "labelled_evidence_cited", "key_fact_coverage")}})
    finally:
        provider.relative_floor = original
    return rows
