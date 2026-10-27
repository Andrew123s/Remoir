"""Render evaluation results (dict from run_evaluation.py) as Markdown."""

from __future__ import annotations


def _pct(v) -> str:
    return "n/a" if v is None else f"{v * 100:.1f}%"


def _table(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def write_markdown(r: dict) -> str:
    env, cfg, perf = r["environment"], r["configuration"], r["performance"]
    ret, ans, abst = r["retrieval"], r["answers"]["summary"], r["abstention_thresholds"]
    fu, st = r["follow_ups"], r["citation_stress_test"]
    lines = [
        "# Remoir evaluation report",
        "",
        f"_Generated {r['generated_at']} by `scripts/run_evaluation.py`. All numbers below were measured in that run._",
        "",
        "> **Scope warning.** The evaluation corpus is four **synthetic, fictional** documents created by "
        "`scripts/make_fixtures.py` for software testing. The results show that the pipeline works as designed on "
        "controlled inputs. They are **not** evidence of accuracy on real academic papers, which are longer, noisier "
        "and use more varied language. The thresholds were also calibrated on this same set, so the abstention "
        "figures are optimistic.",
        "",
        "## 1. Environment and configuration",
        "",
        _table(["Item", "Value"], [
            ["Python", env["python"]], ["Platform", env["platform"]], ["CPU", f"{env['processor']} ({env['cpu_count']} logical)"],
            ["RAM", f"{env['ram_gb']} GB"], ["GPU", env["gpu"]],
            ["Embedding model", f"`{cfg['embedding_model']}` ({cfg['embedding_dimension']} dims)"],
            ["Chunking", f"{cfg['chunk_size']} tokens, {cfg['chunk_overlap']} overlap"],
            ["Default retrieval", f"{cfg['default_strategy']}, top-k {cfg['top_k']}, rerank {cfg['default_rerank']}"],
            ["Answer generation", f"`{cfg['llm_provider']}` ({cfg['llm_model']})"],
            ["Evidence gate", f"min dense score {cfg['min_dense_score']}, min term coverage {cfg['min_term_coverage']}"],
            ["Claim support threshold", cfg["min_support_score"]],
        ]),
        "",
        "## 2. Dataset and procedure",
        "",
        f"- {r['dataset']['n_questions']} questions ({ans['n_answerable']} answerable, {ans['n_unanswerable']} unanswerable) "
        f"and {r['dataset']['n_conversations']} two-turn conversations, in `evaluation/evaluation_dataset.json`.",
        f"- Relevance labels: {r['dataset']['relevance_procedure']}",
        "- Retrieval is evaluated separately from generation, using `RetrievalService.retrieve` directly.",
        "- Answer checks are deterministic: abstention decisions, citation ids versus retrieved sources, cited passages "
        "versus relevance labels, and key-fact string matching. Claim support uses the pipeline's own heuristic "
        "(embedding similarity plus term containment). It is a signal, not ground truth.",
        "",
        "## 3. Retrieval quality",
        "",
        _table(
            ["Strategy"] + [f"{m}@{k}" for k in ret["ks"] for m in ("P", "R")] + ["Hit@1", "MRR", "Mean latency", "p95 latency"],
            [
                [name] + [f"{v['metrics'][f'@{k}'][m]:.3f}" for k in ret["ks"] for m in ("precision", "recall")]
                + [f"{v['metrics']['@1']['hit']:.3f}", f"{v['mrr']:.3f}", f"{v['latency']['mean_ms']:.1f} ms", f"{v['latency']['p95_ms']:.1f} ms"]
                for name, v in ret["results"].items()
            ],
        ),
        "",
        f"Computed over {ret['n_questions']} answerable questions. Precision@k is bounded by the number of relevant passages "
        "per question (often one or two), so low P@5 values are expected.",
        "",
        "Retrieval misses at k=5 (default hybrid strategy):",
        "",
    ]
    default_name = next((n for n, v in ret["results"].items() if v["config"]["strategy"] == cfg["default_strategy"]
                         and v["config"]["rerank"] == cfg["default_rerank"]), next(iter(ret["results"])))
    misses = [q for q in ret["results"][default_name]["per_question"] if q["recall@5"] < 1]
    lines += [f"- `{q['id']}` ({q['type']}): recall@5 = {q['recall@5']:.2f}" for q in misses] or ["- none"]
    lines += [
        "",
        "## 4. Evidence-sufficiency gate (abstention)",
        "",
        f"The gate needs top dense similarity ≥ threshold **and** query term coverage ≥ {abst['min_term_coverage']}.",
        "",
        _table(["Entity check", "min_dense_score", "Answerable answered", "Unanswerable abstained", "Accuracy"],
               [["on" if s["entity_check"] else "off", s["min_dense_score"], _pct(s["answerable_answered"]),
                 _pct(s["unanswerable_abstained"]), _pct(s["accuracy"])] for s in abst["sweep"]]),
        "",
        "The entity check (question names such as places that never appear in the retrieved passages) was added after "
        "the first evaluation run showed that topical distractor questions passed the similarity and coverage gates. "
        "It was therefore designed with knowledge of this data.",
        "",
        _table(["Question", "Answerable", "Top dense", "Term coverage", "Missing entities"],
               [[s["id"], s["answerable"], s["top_dense"], s["coverage"], ", ".join(s["missing_entities"]) or "-"]
                for s in abst["signals"] if not s["answerable"]]),
        "",
        "## 5. Answer generation and citations (end to end)",
        "",
        _table(["Metric", "Value"], [
            ["Provider", f"`{ans['provider']}` ({ans['model']})"],
            ["Answer rate on answerable questions", _pct(ans["answer_rate_answerable"])],
            ["Abstention rate on unanswerable questions", _pct(ans["abstention_rate_unanswerable"])],
            ["Abstention decision accuracy", _pct(ans["abstention_decision_accuracy"])],
            ["Citation validity (ids refer to retrieved sources)", _pct(ans["citation_validity"])],
            ["Answers carrying at least one citation", _pct(ans["answers_with_citations"])],
            ["Citation relevance (cited passage contains labelled evidence)", _pct(ans["citation_relevance"])],
            ["Cited document is a labelled source document", _pct(ans["cited_document_correct"])],
            ["Labelled evidence page cited", _pct(ans["labelled_evidence_cited"])],
            ["Key-fact coverage (string match)", _pct(ans["key_fact_coverage"])],
            ["Claims supported by cited passage (heuristic)", _pct(ans["claim_support_rate"])],
            ["Unsupported claims (count)", ans["unsupported_claims"]],
            ["Generation latency (mean / p95)", f"{ans['generation_latency'].get('mean_ms', 0):.0f} / {ans['generation_latency'].get('p95_ms', 0):.0f} ms"],
            ["End-to-end latency (mean / p95)", f"{ans['total_latency']['mean_ms']:.0f} / {ans['total_latency']['p95_ms']:.0f} ms"],
        ]),
        "",
        "The extractive provider quotes source sentences verbatim, so citation validity and claim support are high "
        "by construction. These checks become informative when a generative LLM (Ollama or a hosted API) is used. "
        "Citation relevance and key-fact coverage measure whether the quoted sentences are the right ones.",
        "",
        "### Extractive sentence-selection floor",
        "",
        "The extractive provider quotes sentences that score at least `relative_floor` times the best sentence. "
        f"The sweep below was used to choose the configured value ({cfg.get('extractive_relative_floor')}).",
        "",
        _table(["relative_floor", "Citation relevance", "Cited document correct", "Labelled evidence cited", "Key-fact coverage"],
               [[x["relative_floor"], _pct(x["citation_relevance"]), _pct(x["cited_document_correct"]),
                 _pct(x["labelled_evidence_cited"]), _pct(x["key_fact_coverage"])] for x in r.get("extractive_floor_sweep", [])]),
        "",
        "### Follow-up questions",
        "",
        f"{fu['n']} second-turn questions; labelled evidence retrieved in {_pct(fu['evidence_hit_rate'])} of cases.",
        "",
        _table(["Conversation", "Follow-up", "Evidence retrieved", "Scoped to previous document", "Retrieval query"],
               [[x["id"], x["question"], x["evidence_retrieved"], x["scoped_to_previous_document"], x["retrieval_query"]] for x in fu["rows"]]),
        "",
        "### Citation fabrication stress test",
        "",
        "An adversarial mock LLM returns one genuine quoted claim, one claim citing a non-existent source, one invented "
        "claim attached to a real source id, and one uncited invented claim.",
        "",
        _table(["Check", "Pass rate"], [
            ["Non-existent citation removed from answer", _pct(st["nonexistent_citation_removed"])],
            ["Invented claim with a real citation flagged as weakly supported", _pct(st["misattributed_claim_flagged"])],
            ["Uncited invented claim flagged", _pct(st["uncited_claim_flagged"])],
            ["Genuine quoted claim accepted", _pct(st["genuine_claim_supported"])],
        ]),
        "",
        f"Measured on {st['n']} questions.",
        "",
        "## 6. Performance",
        "",
        _table(["Document", "Status", "Pages", "Chunks", "Size", "Ingestion time"],
               [[i["document"], i["status"], i["pages"], i["chunks"], f"{i['size_kb']} KB", f"{i['seconds']:.2f} s"] for i in perf["ingestion"]]),
        "",
        f"- Embedding model load and warm-up: {perf['model_load_seconds']} s. The first ingestion also includes tokenizer warm-up.",
        f"- Embedding throughput: {perf['embedding_chunks_per_second']} chunks/s ({perf['total_chunks']} chunks, "
        f"tokens per chunk min/mean/max {perf['chunk_tokens']['min']}/{perf['chunk_tokens']['mean']}/{perf['chunk_tokens']['max']}).",
        f"- Process memory (RSS): {perf['memory_mb']['process_start']} MB at start, {perf['memory_mb']['after_model_load']} MB "
        f"after loading the embedding model, {perf['memory_mb']['end']} MB at the end (includes the cross-encoder if it was evaluated).",
        "",
        "## 7. Limitations of this evaluation",
        "",
        "- The corpus is small and synthetic, and the documents are short and cleanly formatted. Real PDFs (two-column "
        "layouts, tables, equations, scanned pages) will produce lower scores.",
        "- The relevance labels and the evidence thresholds were written and tuned by the same author on the same data.",
        "- Key-fact coverage uses exact string matching. It can miss paraphrases and cannot judge answer quality as a whole.",
        "- Claim support is a lexical and embedding heuristic, not natural-language-inference entailment.",
        "- No generative LLM was available in the evaluation environment (Ollama is not installed and no API key is configured). "
        "The Ollama and OpenAI-compatible providers are covered only by mocked unit tests.",
        "",
        "## 8. Opportunities for improvement",
        "",
        "- Evaluate on a permitted set of real open-access papers with independent relevance annotation.",
        "- Add an NLI model (for example a small cross-encoder trained on MNLI) for claim-level entailment.",
        "- Calibrate the evidence gate on held-out questions and report confidence intervals.",
        "- Measure generative-answer faithfulness once a local LLM (for example `llama3.2:3b` through Ollama) is installed.",
        "",
    ]
    return "\n".join(lines)
