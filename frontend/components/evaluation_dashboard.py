"""Evaluation Dashboard page: shows only metrics recorded by scripts/run_evaluation.py."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from frontend import api_client
from frontend.api_client import APIError


def _pct(v) -> str:
    return "n/a" if v is None else f"{v * 100:.1f}%"


def page() -> None:
    st.title("Evaluation Dashboard")
    try:
        r = api_client.get("/evaluation/latest")
    except APIError as exc:
        st.info("No evaluation results yet. Run `python scripts/run_evaluation.py`." if exc.status == 404 else exc.detail)
        return

    st.warning(
        "Measured on four **synthetic** fixture documents (fictional content made for software testing). "
        "These numbers verify the pipeline, not accuracy on real research papers."
    )
    cfg, env = r["configuration"], r["environment"]
    st.caption(
        f"Run {r['generated_at']} · {cfg['embedding_model']} · chunk {cfg['chunk_size']}/{cfg['chunk_overlap']} · "
        f"answers: {cfg['llm_provider']} · {env['processor']} · {env['ram_gb']} GB RAM · {env['gpu']}"
    )

    st.header("Retrieval")
    ret = r["retrieval"]
    rows = []
    for name, v in ret["results"].items():
        row = {"strategy": name, "MRR": v["mrr"], "mean latency ms": v["latency"]["mean_ms"]}
        for k in ret["ks"]:
            row[f"Recall@{k}"] = v["metrics"][f"@{k}"]["recall"]
            row[f"Precision@{k}"] = v["metrics"][f"@{k}"]["precision"]
            row[f"Hit@{k}"] = v["metrics"][f"@{k}"]["hit"]
        rows.append(row)
    df = pd.DataFrame(rows).set_index("strategy")
    c1, c2 = st.columns(2)
    c1.subheader("Recall@k")
    c1.bar_chart(df[[f"Recall@{k}" for k in ret["ks"]]].T, stack=False)
    c2.subheader("MRR and Hit@1")
    c2.bar_chart(df[["MRR", "Hit@1"]], stack=False)
    st.dataframe(df.round(3), use_container_width=True)
    st.caption(f"{ret['n_questions']} answerable questions; relevance = labelled evidence phrase in a chunk from the labelled document.")

    st.header("Answers and citations")
    s = r["answers"]["summary"]
    m = st.columns(4)
    m[0].metric("Abstention decision accuracy", _pct(s["abstention_decision_accuracy"]))
    m[1].metric("Citation validity", _pct(s["citation_validity"]))
    m[2].metric("Citation relevance", _pct(s["citation_relevance"]))
    m[3].metric("Key-fact coverage", _pct(s["key_fact_coverage"]))
    m = st.columns(4)
    m[0].metric("Answered (answerable)", _pct(s["answer_rate_answerable"]))
    m[1].metric("Abstained (unanswerable)", _pct(s["abstention_rate_unanswerable"]))
    m[2].metric("Claims supported (heuristic)", _pct(s["claim_support_rate"]))
    m[3].metric("Follow-up evidence hit rate", _pct(r["follow_ups"]["evidence_hit_rate"]))

    st.subheader("Citation fabrication stress test")
    stress = r["citation_stress_test"]
    st.dataframe(pd.DataFrame([
        {"check": "Non-existent citation removed", "pass rate": _pct(stress["nonexistent_citation_removed"])},
        {"check": "Invented claim with real citation flagged", "pass rate": _pct(stress["misattributed_claim_flagged"])},
        {"check": "Uncited invented claim flagged", "pass rate": _pct(stress["uncited_claim_flagged"])},
        {"check": "Genuine claim accepted", "pass rate": _pct(stress["genuine_claim_supported"])},
    ]), hide_index=True, use_container_width=True)

    st.subheader("Evidence gate threshold sweep")
    sweep = pd.DataFrame(r["abstention_thresholds"]["sweep"])
    st.dataframe(sweep, hide_index=True, use_container_width=True)

    if r.get("extractive_floor_sweep"):
        st.subheader("Extractive sentence floor sweep")
        st.dataframe(pd.DataFrame(r["extractive_floor_sweep"]), hide_index=True, use_container_width=True)

    st.header("Performance and indexing")
    perf = r["performance"]
    m = st.columns(4)
    m[0].metric("Chunks indexed", perf["total_chunks"])
    m[1].metric("Embedding throughput", f"{perf['embedding_chunks_per_second']} chunks/s")
    m[2].metric("Answer latency (mean)", f"{s['total_latency']['mean_ms']:.0f} ms")
    m[3].metric("Memory after model load", f"{perf['memory_mb']['after_model_load']} MB")
    st.dataframe(pd.DataFrame(perf["ingestion"]), hide_index=True, use_container_width=True)

    with st.expander("Per-question answer results"):
        st.dataframe(pd.DataFrame(r["answers"]["per_question"]), hide_index=True, use_container_width=True)
