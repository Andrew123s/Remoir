# Remoir evaluation report

_Generated 2026-10-27T10:47:18+00:00 by `scripts/run_evaluation.py`. All numbers below were measured in that run._

> **Scope warning.** The evaluation corpus is four **synthetic, fictional** documents created by `scripts/make_fixtures.py` for software testing. The results show that the pipeline works as designed on controlled inputs. They are **not** evidence of accuracy on real academic papers, which are longer, noisier and use more varied language. The thresholds were also calibrated on this same set, so the abstention figures are optimistic.

## 1. Environment and configuration

| Item | Value |
|---|---|
| Python | 3.13.5 |
| Platform | Windows-11-10.0.26300-SP0 |
| CPU | Intel64 Family 6 Model 126 Stepping 5, GenuineIntel (8 logical) |
| RAM | 11.8 GB |
| GPU | none (CPU only) |
| Embedding model | `sentence-transformers:BAAI/bge-small-en-v1.5` (384 dims) |
| Chunking | 220 tokens, 40 overlap |
| Default retrieval | hybrid, top-k 6, rerank False |
| Answer generation | `extractive` (extractive/sentence-transformers:BAAI/bge-small-en-v1.5) |
| Evidence gate | min dense score 0.45, min term coverage 0.3 |
| Claim support threshold | 0.5 |

## 2. Dataset and procedure

- 30 questions (24 answerable, 6 unanswerable) and 2 two-turn conversations, in `evaluation/evaluation_dataset.json`.
- Relevance labels: Labels were written by hand from the fixture source text in scripts/make_fixtures.py. Each relevant item is (document, PDF page, evidence phrase). A retrieved chunk is relevant to an item if it comes from that document and contains the evidence phrase (case-insensitive substring). Labels do not depend on chunk ids, so they stay valid when chunking settings change.
- Retrieval is evaluated separately from generation, using `RetrievalService.retrieve` directly.
- Answer checks are deterministic: abstention decisions, citation ids versus retrieved sources, cited passages versus relevance labels, and key-fact string matching. Claim support uses the pipeline's own heuristic (embedding similarity plus term containment). It is a signal, not ground truth.

## 3. Retrieval quality

| Strategy | P@1 | R@1 | P@3 | R@3 | P@5 | R@5 | Hit@1 | MRR | Mean latency | p95 latency |
|---|---|---|---|---|---|---|---|---|---|---|
| dense | 0.625 | 0.521 | 0.319 | 0.812 | 0.208 | 0.854 | 0.625 | 0.750 | 60.8 ms | 66.7 ms |
| hybrid (dense + BM25, RRF) | 0.625 | 0.521 | 0.347 | 0.875 | 0.233 | 0.958 | 0.625 | 0.788 | 62.1 ms | 71.0 ms |
| dense + cross-encoder rerank | 0.750 | 0.667 | 0.347 | 0.896 | 0.225 | 0.938 | 0.750 | 0.851 | 641.2 ms | 765.8 ms |
| hybrid + cross-encoder rerank | 0.750 | 0.667 | 0.347 | 0.896 | 0.225 | 0.938 | 0.750 | 0.851 | 638.6 ms | 762.8 ms |

Computed over 24 answerable questions. Precision@k is bounded by the number of relevant passages per question (often one or two), so low P@5 values are expected.

Retrieval misses at k=5 (default hybrid strategy):

- `q06` (findings): recall@5 = 0.50
- `q20` (comparison): recall@5 = 0.50

## 4. Evidence-sufficiency gate (abstention)

The gate needs top dense similarity ≥ threshold **and** query term coverage ≥ 0.3.

| Entity check | min_dense_score | Answerable answered | Unanswerable abstained | Accuracy |
|---|---|---|---|---|
| off | 0.4 | 100.0% | 50.0% | 90.0% |
| off | 0.45 | 100.0% | 50.0% | 90.0% |
| off | 0.5 | 100.0% | 50.0% | 90.0% |
| off | 0.55 | 100.0% | 50.0% | 90.0% |
| off | 0.6 | 100.0% | 50.0% | 90.0% |
| off | 0.65 | 87.5% | 83.3% | 86.7% |
| on | 0.4 | 100.0% | 100.0% | 100.0% |
| on | 0.45 | 100.0% | 100.0% | 100.0% |
| on | 0.5 | 100.0% | 100.0% | 100.0% |
| on | 0.55 | 100.0% | 100.0% | 100.0% |
| on | 0.6 | 100.0% | 100.0% | 100.0% |
| on | 0.65 | 87.5% | 100.0% | 90.0% |

The entity check (question names such as places that never appear in the retrieved passages) was added after the first evaluation run showed that topical distractor questions passed the similarity and coverage gates. It was therefore designed with knowledge of this data.

| Question | Answerable | Top dense | Term coverage | Missing entities |
|---|---|---|---|---|
| u01 | False | 0.507 | 0.0 | - |
| u02 | False | 0.6239 | 0.333 | Canadian |
| u03 | False | 0.6047 | 0.667 | Kenya |
| u04 | False | 0.462 | 0.0 | Cup, FIFA, World |
| u05 | False | 0.4837 | 0.25 | - |
| u06 | False | 0.6709 | 0.4 | Bangladesh |

## 5. Answer generation and citations (end to end)

| Metric | Value |
|---|---|
| Provider | `extractive` (extractive/sentence-transformers:BAAI/bge-small-en-v1.5) |
| Answer rate on answerable questions | 100.0% |
| Abstention rate on unanswerable questions | 100.0% |
| Abstention decision accuracy | 100.0% |
| Citation validity (ids refer to retrieved sources) | 100.0% |
| Answers carrying at least one citation | 100.0% |
| Citation relevance (cited passage contains labelled evidence) | 48.9% |
| Cited document is a labelled source document | 84.4% |
| Labelled evidence page cited | 95.8% |
| Key-fact coverage (string match) | 68.8% |
| Claims supported by cited passage (heuristic) | 100.0% |
| Unsupported claims (count) | 0 |
| Generation latency (mean / p95) | 438 / 734 ms |
| End-to-end latency (mean / p95) | 839 / 1452 ms |

The extractive provider quotes source sentences verbatim, so citation validity and claim support are high by construction. These checks become informative when a generative LLM (Ollama or a hosted API) is used. Citation relevance and key-fact coverage measure whether the quoted sentences are the right ones.

### Extractive sentence-selection floor

The extractive provider quotes sentences that score at least `relative_floor` times the best sentence. The sweep below was used to choose the configured value (0.8).

| relative_floor | Citation relevance | Cited document correct | Labelled evidence cited | Key-fact coverage |
|---|---|---|---|---|
| 0.0 | 30.8% | 75.1% | 95.8% | 68.8% |
| 0.7 | 44.0% | 83.1% | 95.8% | 68.8% |
| 0.8 | 48.9% | 84.4% | 95.8% | 68.8% |
| 0.85 | 53.0% | 81.7% | 91.7% | 58.3% |

### Follow-up questions

2 second-turn questions; labelled evidence retrieved in 100.0% of cases.

| Conversation | Follow-up | Evidence retrieved | Scoped to previous document | Retrieval query |
|---|---|---|---|---|
| c01 | What were its main limitations? | True | True | Summarize the methodology of the maize rainfall study. What were its main limitations? |
| c02 | What limitations did they report? | True | False | Which machine learning model performed best for crop yield prediction? What limitations did they report? |

### Citation fabrication stress test

An adversarial mock LLM returns one genuine quoted claim, one claim citing a non-existent source, one invented claim attached to a real source id, and one uncited invented claim.

| Check | Pass rate |
|---|---|
| Non-existent citation removed from answer | 100.0% |
| Invented claim with a real citation flagged as weakly supported | 100.0% |
| Uncited invented claim flagged | 100.0% |
| Genuine quoted claim accepted | 100.0% |

Measured on 10 questions.

## 6. Performance

| Document | Status | Pages | Chunks | Size | Ingestion time |
|---|---|---|---|---|---|
| rainfall_maize_study.pdf | indexed | 3 | 6 | 6.4 KB | 0.41 s |
| ml_crop_prediction.pdf | indexed | 2 | 6 | 4.9 KB | 0.33 s |
| ecosystem_services_review.pdf | indexed | 2 | 5 | 4.9 KB | 0.24 s |
| climate_adaptation_brief.pdf | indexed | 2 | 4 | 4.4 KB | 0.34 s |

- Embedding model load and warm-up: 33.11 s. The first ingestion also includes tokenizer warm-up.
- Embedding throughput: 14.9 chunks/s (21 chunks, tokens per chunk min/mean/max 25/60.2/121).
- Process memory (RSS): 155.4 MB at start, 680.4 MB after loading the embedding model, 801.1 MB at the end (includes the cross-encoder if it was evaluated).

## 7. Limitations of this evaluation

- The corpus is small and synthetic, and the documents are short and cleanly formatted. Real PDFs (two-column layouts, tables, equations, scanned pages) will produce lower scores.
- The relevance labels and the evidence thresholds were written and tuned by the same author on the same data.
- Key-fact coverage uses exact string matching. It can miss paraphrases and cannot judge answer quality as a whole.
- Claim support is a lexical and embedding heuristic, not natural-language-inference entailment.
- No generative LLM was available in the evaluation environment (Ollama is not installed and no API key is configured). The Ollama and OpenAI-compatible providers are covered only by mocked unit tests.

## 8. Opportunities for improvement

- Evaluate on a permitted set of real open-access papers with independent relevance annotation.
- Add an NLI model (for example a small cross-encoder trained on MNLI) for claim-level entailment.
- Calibrate the evidence gate on held-out questions and report confidence intervals.
- Measure generative-answer faithfulness once a local LLM (for example `llama3.2:3b` through Ollama) is installed.
