# BSARD Retrieval Experiments — Project Context

**Author:** Marios Paschalidis · KU Leuven, Master of Artificial Intelligence
**Thesis:** Enhancing Performance and Quality of Context Retrieval in RAG Systems
**Scope of this document:** RQ1 — systematic comparison of retrieval methods over the BSARD Belgian-statutory-law corpus.

This file is the project-context reference: corpus, benchmark, evaluation protocol, and an index of what was actually built. For per-tier implementation and result detail, the canonical sources are the six `TIER*_*_PLAN.md` files at the repo root.

---

## 1. Scope and What This Project Builds

A systematic comparison of retrieval methods over BSARD, producing the empirical results for **RQ1** of the master thesis. Each method is evaluated against the same 222-question BSARD test split using standard IR metrics.

| Tier | Method family | Plan file |
|---|---|---|
| Tier 1 | Sparse — BM25 (Okapi/Plus/L), TF-IDF, FTS5 | [TIER1_SPARSE_RETRIEVAL_PLAN.md](TIER1_SPARSE_RETRIEVAL_PLAN.md) |
| Tier 2 | Dense — bi-encoder over FAISS `IndexFlatIP` | [TIER2_DENSE_RETRIEVAL_PLAN.md](TIER2_DENSE_RETRIEVAL_PLAN.md) |
| Tier 3 | Hybrid — RRF / linear fusion / sparse-gated dense re-ranking | [TIER3_HYBRID_RETRIEVAL_PLAN.md](TIER3_HYBRID_RETRIEVAL_PLAN.md) |
| Tier 4.0 | LLM-as-Judge re-ranking (non-agentic) | [TIER40_LLM_JUDGE_PLAN.md](TIER40_LLM_JUDGE_PLAN.md) |
| Tier 4.1 | Agentic CRAG | [TIER41_AGENTIC_CRAG_PLAN.md](TIER41_AGENTIC_CRAG_PLAN.md) |
| Tier 4.2 | Agentic ReAct | [TIER42_AGENTIC_REACT_PLAN.md](TIER42_AGENTIC_REACT_PLAN.md) |

**Out of scope here.** RQ2 (structure-aware retrieval — chunking, metadata, PageIndex) and RQ3 (autonomous reference-free evaluation) are sibling components of this mono-repo (`RQ2_Structure_Aware_Retrieval`, `RQ3_Autonomous_Evaluation`). The one piece of RQ3 work kept in this component is the evaluator analysis at `analysis/RQ3/`, because it consumes RQ1 result JSONs and the 48-question stratified subset directly; the autonomous evaluation library itself is the sibling `RQ3_Autonomous_Evaluation` component.

---

## 2. Corpus — What Is Available

All corpus data is in **`bsard_corpus.db`** (SQLite, ~100 MB) and its flat exports. These large artefacts are not in git; they live in the companion Hugging Face dataset `mpaschalidis/bsard-rag-thesis-data` and download into a local gitignored data root. The data root is the `BSARD_DATA_DIR` environment variable, defaulting to `<repo>/output`. Pull it with `python scripts/download_data.py` (or the mono-repo root `python data_tooling/download_combined_hf.py --subset rq1`). Paths below are written relative to the data root (shown as `output/...` for the default location).

### 2.1 Primary files

| File | Size | When to use |
|---|---|---|
| `output/bsard_corpus.db` | ~100 MB | All querying, evaluation, citation graph traversal |
| `output/bsard_articles_dedup.parquet` | ~8 MB | **Canonical Phase 1 corpus** — 22,633 unique articles after deduplication |
| `output/bsard_articles_only.parquet` | ~12 MB | Pre-dedup BSARD subset (33,741 rows); use dedup parquet for Phase 1 retrieval |
| `output/bsard_articles.parquet` | ~14 MB | Full corpus including 6,490 distractor articles |
| `output/bsard_articles_only.jsonl` | ~78 MB | JSONL ingestion into vector stores |
| `output/corpus_stats.json` | ~5 KB | Corpus statistics for thesis Chapter 3 |

### 2.2 Corpus tiers

**BSARD articles** (`is_bsard_article = 1`): 33,741 records, 22,633 unique `bsard_id` values (after deduplication).
- These are the articles the benchmark questions were written about.
- They carry ground truth annotations via the `questions` table.
- Sub-paragraph variants share a `bsard_id` (e.g., Art.1.1.1-1, Art.1.1.1-2 → same BSARD ID).
- Article text source: HuggingFace canonical text for 33,741 records (post-2021 language).
- **RQ1 corpus:** the deduplicated corpus (`output/bsard_articles_dedup.parquet`, 22,633 rows) is used for all retrieval experiments, so evaluation metrics are not confounded by duplicate articles.

**Non-BSARD articles** (`is_bsard_article = 0`): 6,490 records.
- Articles found in the 49 Justel PDFs that are not in the BSARD benchmark.
- These are retrieval distractors — the retrieval system must not surface them as answers.
- Text source: PDF-extracted.

### 2.3 Key `articles` table columns for retrieval

| Column | Type | Use |
|---|---|---|
| `article_id` | INTEGER PK | All internal references |
| `bsard_id` | INTEGER | Ground truth linkage to `questions.relevant_article_ids` |
| `is_bsard_article` | INTEGER | Filter: corpus tier |
| `law_code` | TEXT | 34 unique values; metadata-filtered retrieval |
| `article_number` | TEXT | Human-readable article reference |
| `article_text` | TEXT | The text to embed / index |
| `token_count` | INTEGER | Chunking decisions (cl100k_base tokens) |
| `char_count` | INTEGER | Lightweight size proxy |
| `hierarchy_path` | TEXT (JSON) | Hierarchical context for PageIndex (RQ2) |
| `chapter_title`, `section_title` | TEXT | Structural metadata for context-aware chunking |
| `cross_reference_ids` | TEXT (JSON) | Neighbour expansion |
| `cited_by_ids` | TEXT (JSON) | Inverse neighbour expansion |
| `n_outgoing_refs`, `n_cited_by` | INTEGER | Citation degree; failure-condition stratification |
| `has_cross_references` | INTEGER | Stratification flag for §4.4 analysis |
| `amendment_date`, `is_pre_bsard` | TEXT/INTEGER | Temporal filtering (RQ2) |
| `article_status` | TEXT | `ORIGINAL_NEVER_AMENDED`, `PRE_BSARD`, `POST_BSARD` |

### 2.4 FTS5 full-text index (already built)

The database has a pre-built FTS5 virtual table for BM25-style sparse retrieval:

```sql
-- FTS5 search with BM25 ranking and snippet extraction
SELECT a.article_id, a.bsard_id, a.law_code, a.article_number,
       snippet(articles_fts, 0, '>>>', '<<<', '...', 20) AS snippet
FROM articles_fts
JOIN articles a ON a.article_id = articles_fts.rowid
WHERE articles_fts MATCH ?
ORDER BY rank
LIMIT 100;
```

The FTS5 tokenizer is `unicode61` (handles accented French characters correctly).

### 2.5 Citation graph

The `citation_graph` table has 27,712 directed edges. It can be loaded into NetworkX for graph traversal:

```python
import sqlite3, networkx as nx

conn = sqlite3.connect("output/bsard_corpus.db")
edges = conn.execute(
    "SELECT source_id, target_id FROM citation_graph WHERE resolved = 1"
).fetchall()
G = nx.DiGraph()
G.add_edges_from(edges)
```

Per-article neighbour lookups: given a retrieved `article_id`, retrieve its `k`-hop neighbours via the pre-materialised `cross_reference_ids` and `cited_by_ids` columns on the `articles` table.

---

## 3. Benchmark — Questions and Ground Truth

### 3.1 The questions table

1,108 natural language legal questions in French. Each question has one or more ground-truth relevant articles.

```python
import sqlite3, json

conn = sqlite3.connect("output/bsard_corpus.db")

# Load all test questions with ground truth
questions = conn.execute("""
    SELECT question_id, question_text, relevant_article_ids, n_relevant_articles
    FROM questions
    WHERE split = 'test'
""").fetchall()

for q in questions:
    q_id     = q["question_id"]
    text     = q["question_text"]
    gt_ids   = json.loads(q["relevant_article_ids"])  # list of article_id integers
    n_gt     = q["n_relevant_articles"]
```

### 3.2 Ground truth structure

| Field | Description |
|---|---|
| `relevant_article_ids` | JSON array of `article_id` integers — the internal PKs for this corpus |
| `relevant_bsard_ids` | JSON array of BSARD benchmark `id` values — for traceability |
| `n_relevant_articles` | Number of ground-truth articles (mean = 6.18, range 1–many) |
| `split` | Use `test` (222 questions) for all reported experiments; `train` (886) for development |

**Important:** 65.5% of questions require multiple relevant articles. Retrieval must surface *all* of them to score perfectly on Recall@k for those queries.

### 3.3 Benchmark statistics

| Metric | Value |
|---|---|
| Total questions | 1,108 |
| Test set | 222 |
| Train set | 886 |
| Single-article questions | 382 (34.5%) |
| Multi-article questions | 726 (65.5%) |
| Mean relevant articles | 6.18 |
| Median Jaccard (query ↔ relevant article) | 0.045 |

The **low Jaccard overlap (median 0.045)** is the core motivation for dense retrieval: questions rarely share exact vocabulary with their relevant articles.

---

## 4. Evaluation Protocol

### 4.1 Metrics

All methods are evaluated on the **test split (222 questions)** via the `bsard_evaluation` harness from the sibling `RQ3_Autonomous_Evaluation` component (installed with `pip install -e "../RQ3_Autonomous_Evaluation"`). The harness produces a layered metric panel per experiment:

| Layer | Metrics | k values |
|---|---|---|
| **Tier 0 — Efficiency** | Latency (mean/std/p50/p90/p95/p99/min/max), QPS, index build time | — |
| **Tier 1 — BSARD paper** | Recall@k, MRR@k | R@{1, 5, 10, 100}; MRR@{100} |
| **Tier 2 — Full IR** | Recall@k, Precision@k, F1@k, MRR@k, MAP@k, NDCG@k, plus ID-based P@k/R@k | rank-unaware k ∈ {1, 5, 10, 20, 50, 100, 200, 500}; rank-aware k ∈ {10, 100} |
| **Tier 3 — Autonomous (subset only)** | UMBRELA, eRAG, RAGAS-WA, RAGAS-WB → AQS; UMBRELA→Tier 2 bridge (`T2-umbrela/*`) | UMBRELA grades 0–3; k = 10 |

**Primary metric: Recall@100** — matches the BSARD paper, and is the metric on which Tier 1 (TIER1 §3) and Tier 2 (TIER2 §0, §12) results are ranked and significance-tested. R@100 is appropriate for BSARD because 65.5 % of questions have multiple relevant articles (mean 6.18) and R@10 is structurally capped well below 1.0 for high-multiplicity queries.

**Secondary metric: Recall@10** — early-precision focus, used as the within-tier significance-test anchor in Tier 3 (TIER3 §6.5) and Tier 4 (T4.0 / T4.1 / T4.2 plans). It is the natural metric for re-ranking comparisons, where the top-k surface is what matters to a downstream reader / LLM. R@10 numbers are reported alongside R@100 in every per-tier results table.

**Tier 3 autonomous evaluation** (UMBRELA, eRAG, RAGAS-WA, RAGAS-WB + UMBRELA→Tier 2 bridge) is applied to a fixed 48-question stratified subset (`evaluation/data/tier3_subset.json` — 21.6 % of the test set, covering all 12 cells of `article_count × lex_align × cross_ref`). It is run on every Tier 1/2/3/4 result JSON and stored under each result's `subset_metrics.metrics` block.

### 4.2 Significance testing

Two-sided paired t-test on per-query Recall@k at p < 0.05. Tier-specific anchors:

| Comparison | Anchor | Primary k |
|---|---|---|
| Within Tier 1 | `bm25_anchor` (paper-aligned BM25 Okapi, k1=1.5, b=0.75) | 10 |
| Within Tier 2 | inline `bm25 (k1=1.5, b=0.5, lemmatize, text_only)` — intentionally *not* the T1 tuned winner | 100 |
| Within Tier 3 | `dense_me5_large_concat2x_zeroshot` (T2 winner) | 10 |
| Within Tier 4 | T3-A `hybrid_rrf_k60` | 10 |
| Matched-pool agentic-vs-non-agentic | `llm_rerank_binary_top20_hybrid_rrf_k60` (T4.0 at top-20 = matched to T4.1/T4.2 `eval_k`) | 10 |

The matched-pool rule is non-negotiable: comparing a reranker's R@k to a first-stage R@k is only meaningful when k ≤ reranker pool size.

### 4.3 Metric computation

Metric computation is delegated to the `bsard_evaluation` package (the sibling `RQ3_Autonomous_Evaluation` component). `evaluation/runner.py` builds the per-query ranked lists and `contexts_with_ranks` payload, then calls `harness.evaluate(...)`. The harness returns the Tier 0/1/2 panel, the six stratified breakdowns, and the paired-t-test result against the supplied anchor. Tier 3 evaluators are run separately on the 48-question subset and merged into `subset_metrics.metrics` by `scripts/evaluation/compute_subset_metrics.py`.

### 4.4 Stratified analysis

Six strata, persisted in `evaluation/data/query_strata.json`:

- `single_article` vs `multi_article` (`n_relevant_articles == 1` vs > 1)
- `lexically_aligned` vs `semantically_paraphrased` (top vs bottom BM25-score quartile of query against gold article(s); middle quartiles excluded)
- `with_cross_refs` vs `without_cross_refs` (whether any ground-truth article has cross-references)

Every result JSON's `stratified` block reports R@10 / R@100 / MRR@10 / NDCG@10 per stratum.

### 4.5 Train / Validation / Test split

Split persisted in `evaluation/data/split_ids.json` (seed = 42). Test is the BSARD official 222-question test set; val is a 20 % subsample of BSARD train.

| Split key | n | Source | What actually used it |
|---|---|---|---|
| `test` | 222 | BSARD official test | All final-reported results (Tiers 1–4) |
| `val` | 177 | BSARD train, 20 % (seed=42) | Tier 1 BM25 k1/b 4×4 grid only |
| `train` | 709 | BSARD train, remaining 80 % | Corpus statistics and preprocessing decisions; no fine-tuning was performed |
| `train_sample_100` | 100 | BSARD train, seed=42 stratified subsample | Tier 2 EXP-D7 (`concat_2x` field-weighting winner selection across D2/D3/D4/D4c) |

**As-shipped usage:**
- **Tier 1.** Val drives the BM25 k1×b grid (lemmatize, text_only). Winner re-evaluated on test (TIER1 §4).
- **Tier 2.** `train_sample_100` selects the EXP-D7 field-weighting winner across encoder-only models (D2/D3/D4/D4c). The winner's `concat_2x` variant is evaluated on test. All other Tier 2 numbers come from test directly (TIER2 §16.2).
- **Tier 3.** RRF k-grid, linear α-grid, and SGDR K-grid all run on test directly. Per-group canonical results are the test-best entries (TIER3 §3, §4, §5).
- **Tier 4.0 / 4.1 / 4.2.** Hardcoded hyperparameters; no val-driven grid was run end-to-end. T4.0 caches LLM judgments keyed by `(question_id, article_id, prompt_variant, max_article_tokens)`. T4.1 / T4.2 calibration / hyperparameter-tuning scripts under `scripts/evaluation/tier4/` informed the hardcoded values but were not used to gate the final runs.

---

## 5. As-shipped Results

All Recall numbers below are on the 222-question BSARD test split. For full per-experiment tables, latency, and stratified breakdowns, the per-tier plan files are the canonical source.

### 5.1 Headline numbers

| Tier | Experiment | R@10 | R@100 | Notes |
|---|---|---|---|---|
| T1 | `bm25_anchor` (BM25 Okapi, none, text_only, k1=1.5, b=0.75) | 0.2476 | 0.4821 | Tier 1 anchor (paper-aligned) |
| T1 | `bm25_tuned_k11.5_b0.25` (best T1 R@10) | **0.2651** | 0.5210 | k1=1.5, **b=0.25**, lemmatize, text_only — Tier 3 sparse leg + Tier 4 first stage |
| T1 | `bm25_lemmatize_concat_2x` (best T1 R@100) | 0.2572 | **0.5312** | Field-weighted; not used downstream — `concat_2x` leaks metadata into the LLM-Judge surface |
| T2 | `dense_me5_large_concat2x_zeroshot` (Tier 2 winner) | 0.3420 | **0.6215** | mE5-large with `concat_2x` field weighting — EXP-D7 winner |
| T3-A | `hybrid_rrf_k60` (Tier 3 winner) | **0.4021** | **0.6513** | RRF of T1 `bm25_tuned_k11.5_b0.25` + T2 winner; p = 0.0145 vs T2 anchor |
| T4.0 | `llm_rerank_binary_top50_hybrid_rrf_k60` | **0.4451** | 0.5795 | Canonical T4.0; binary LLM-Judge over T3-A top-50; p = 0.016 vs T3-A |
| T4.0 | `llm_rerank_binary_top20_hybrid_rrf_k60` | 0.4347 | 0.4648 | Matched-pool anchor for T4.1 / T4.2 (top-20) |
| T4.0 | `llm_rerank_binary_top50` (BM25 first stage) | 0.3618 | 0.4630 | First-stage ablation (same scorer, weaker pool) |
| T4.0 | `llm_rerank_0to10_top50` (BM25 first stage) | 0.2715 | 0.4630 | Scoring-paradigm ablation (0–10 numeric vs binary) |
| T4.1 | `crag_hybrid_rrf_k60_test_v2` (canonical T4.1) | 0.4263 | 0.6542 | CRAG over T3-A; p = 0.046 vs T3-A |
| T4.1 | `crag_bm25_test_v2` | 0.3012 | 0.5238 | First-stage ablation |
| T4.2 | `react_hybrid_rrf_k60_test_v2` (canonical T4.2) | 0.4256 | 0.4678 | ReAct v2 over T3-A; p = 0.027 vs T3-A |
| T4.2 | `react_hybrid_rrf_k60_test` (v1) | 0.2771 | 0.2922 | v1 design (max_steps=5, top_k=10); under-performed, retained as reference |
| T4.2 | `react_bm25_test` (v1) | 0.2266 | 0.2363 | BM25-backbone v1; under-performed, retained as reference |

### 5.2 Headline thesis finding

At matched candidate-pool size (top-20 on the T3-A hybrid first stage), the agentic methods (T4.1 CRAG R@10 = 0.4263; T4.2 ReAct v2 R@10 = 0.4256) are **not significantly better** than the single-pass non-agentic T4.0 LLM-Judge re-rank (R@10 = 0.4347; Δ = −0.008, p ≈ 0.16 for CRAG; ns for ReAct). **T4.0-hybrid is the Pareto-optimal point on Recall@10 vs latency**: it captures essentially all of the agentic benefit without the iterated LLM-call overhead of CRAG/ReAct.

### 5.3 Source-of-truth pointers

| Tier | Result JSONs | Plan-file section | Notes |
|---|---|---|---|
| T1 | 12 in `output/results/sparse_retrieval/` | [TIER1 §5](TIER1_SPARSE_RETRIEVAL_PLAN.md#5-results) | Tuned BM25 is b=0.25 (not b=0.5 / b=0.75); `concat_2x` excluded from Tier 4 first stage on metadata-leakage grounds (TIER1 §7.2) |
| T2 | 10 in `output/results/dense_retrieval/` (9 zero-shot models + EXP-D7 winner on test) | [TIER2 §5.1](TIER2_DENSE_RETRIEVAL_PLAN.md#51-tier-2-results--222-question-test-set) | Paper checkpoint EXP-D5 (`antoiloui/...`) never released |
| T3 | 15 in `output/results/hybrid/` (3 RRF + 9 linear α + 3 SGDR) | [TIER3 §6.5](TIER3_HYBRID_RETRIEVAL_PLAN.md#65-tier-3-results--222-question-test-set) | T3-A2 (BGE-M3 self-hybrid) skipped — BGE-M3 didn't win T2. T3-D (HyDE / RAG-Fusion) implemented but gated off (`_ENABLE_T3D = False`). Cross-encoder reranker (`retrieval/llm_reranker.py`) implemented + tested but not in canonical T3 pipeline |
| T4.0 | 4 in `output/results/agentic/llm_judge/` (3 binary + 1 numeric) | [TIER40 §Status](TIER40_LLM_JUDGE_PLAN.md#status) | LLaMA 3.1 8B via Ollama; shared with T4.1/T4.2 |
| T4.1 | 2 in `output/results/agentic/CRAG/` (hybrid + BM25 first stages, v2) | [TIER41 §Status](TIER41_AGENTIC_CRAG_PLAN.md#status) | `eval_k = 20` (matched to T4.0 top-20) |
| T4.2 | 3 in `output/results/agentic/ReAct/` (BM25 v1; hybrid v1 + v2) | [TIER42 §Status](TIER42_AGENTIC_REACT_PLAN.md#status) | v1 → v2 design fixes documented in [REACT_ARCHITECTURE_v2.md](REACT_ARCHITECTURE_v2.md) |

### 5.4 Matched-pool comparisons

| Comparison | Δ R@10 | p | Verdict |
|---|---|---|---|
| T3-A `hybrid_rrf_k60` vs T2 anchor | +0.060 | 0.0145 | RRF significantly beats dense-only |
| T4.0 binary top-50 hybrid vs T3-A | +0.043 | 0.016 | LLM-Judge over T3-A pool significantly beats T3-A |
| T4.1 CRAG-hybrid vs T3-A | +0.024 | 0.046 | CRAG significantly beats T3-A |
| T4.2 ReAct-hybrid v2 vs T3-A | +0.024 | 0.027 | ReAct v2 significantly beats T3-A |
| T4.1 CRAG-hybrid vs T4.0 top-20 hybrid | −0.008 | ≈ 0.16 | Not significant (matched pool) |
| T4.2 ReAct-hybrid v2 vs T4.0 top-20 hybrid | −0.009 | ns | Not significant (matched pool) |

### 5.5 RQ2 (structure-aware retrieval) — out of scope for this submission

RQ2 is implemented in the sibling `RQ2_Structure_Aware_Retrieval` component, not here. Its scope is the three structure-aware ablations:

- **Chunking** — fixed-size, sentence-boundary, and structural (hierarchy-path) chunking vs whole-article embedding
- **Metadata filtering** — pre- or post-filtering by `law_code`, `law_type`, `is_pre_bsard`, or `amendment_date`
- **PageIndex / hierarchical context** — prepending `hierarchy_path` + heading titles to article text before embedding

The RQ1 retrieval methods here all use article-level chunking with no structural metadata injected into the embedding text (with the single exception of Tier 1 / Tier 2 `concat_2x` field weighting, which is a BM25-side ablation, not RQ2 structure-aware retrieval). This is the baseline against which RQ2's structure-aware variants are measured in the `RQ2_Structure_Aware_Retrieval` component.

---

## 6. Project Structure (as-shipped)

```
RQ1_Retrieval_Methods/
│
├── retrieval/                          ← Retrieval library
│   ├── preprocessing.py                  Shared: tokenisation, normalisation, field builders
│   ├── sparse.py                         Tier 1: BM25 (Okapi/Plus/L), TF-IDF, FTS5
│   ├── dense.py                          Tier 2: bi-encoder + FAISS IndexFlatIP
│   ├── hybrid.py                         Tier 3: HybridRetriever (RRF / linear), hyde_retrieve, ragfusion_retrieve
│   ├── llm_reranker.py                   Tier 3 cross-encoder reranker (implemented + tested; not in canonical T3 pipeline)
│   └── agentic/                        ← Tier 4 implementations
│       ├── crag.py                       T4.1 CRAG loop
│       ├── react.py                      T4.2 ReAct loop
│       ├── tools.py                      ReAct tool implementations
│       ├── llm_client.py                 Shared OllamaClient (T4.0/T4.1/T4.2)
│       ├── cross_encoder_client.py       Legacy mGTE wrapper — not used by any canonical T4 run; retained for compat
│       ├── prompts.py                    CRAG / ReAct prompts
│       └── llm_eval_prompts.py           T4.0 LLM-Judge prompts (shared with T4.1/T4.2)
│
├── evaluation/                         ← Code + persisted data only — NO result JSONs
│   ├── metrics.py                        Legacy module (runner delegates to bsard_evaluation harness)
│   ├── runner.py                         Experiment runner + significance wrapper
│   ├── split.py                          test / val / train / train_sample_100 loaders
│   ├── stratify.py                       Per-question strata
│   └── data/
│       ├── split_ids.json                  Train/val/test + train_sample_100
│       ├── query_strata.json               Per-question strata
│       ├── tier3_subset.json               48-question stratified subset
│       └── fewshot_examples.json           Locked few-shot pool for T4.0/T4.1/T4.2
│
├── scripts/
│   ├── evaluation/                     ← Experiment orchestration, organised by tier
│   │   ├── tier1/run_sparse_experiments.py
│   │   ├── tier2/run_dense_experiments.py
│   │   ├── tier3/run_hybrid_experiments.py   (deprecated — earlier CE-rerank draft; see note below)
│   │   ├── tier3/run_llm_rerank_experiments.py
│   │   ├── tier4/run_crag_experiments.py
│   │   ├── tier4/run_react_experiments.py
│   │   ├── tier4/calibrate_crag_thresholds.py
│   │   ├── tier4/calibrate_crag_llm.py
│   │   ├── tier4/tune_react_hyperparams.py
│   │   ├── tier4/select_fewshot_examples.py
│   │   ├── tier4/verify_evaluator_accuracy.py
│   │   ├── tier4/round2/                       Late T4.2 diagnostic re-runs
│   │   ├── shared/compute_significance.py
│   │   ├── RQ3/                                RQ3 system runs over the 48-question subset
│   │   ├── create_tier3_subset.py              One-off: build the stratified subset
│   │   └── compute_subset_metrics.py           Post-hoc: recompute Tier 0/1/2 on the subset
│   ├── hybrid/run_hybrid_experiments.py      ← Canonical T3 orchestrator (produced the 15 result JSONs)
│   ├── setup/
│   │   ├── setup_new_device.ps1                New-device bootstrap
│   │   ├── build_hf_id_mapping.py              HF ↔ local article_id mapping
│   │   ├── prepare_dedup_corpus.py             Deprecated — superseded by HF direct load
│   │   ├── download_tier40_hybrid_results.py
│   │   └── download_tier41_hybrid_results.py
│   └── update_nb_exec_times.py
│
├── azure_notebooks/                    ← Notebooks run on Azure VMs (T2 Qwen3 + all of T4)
│   ├── azure_tier2_qwen3.ipynb
│   ├── azure_tier40_llm_rerank.ipynb
│   ├── azure_tier40_hybrid_llm_rerank.ipynb
│   ├── azure_tier40_hybrid_llm_rerank_top20.ipynb
│   ├── azure_tier41_crag.ipynb
│   ├── azure_tier41_crag_hybrid.ipynb
│   ├── azure_tier42_react.ipynb
│   └── azure_tier42_react_hybrid.ipynb
│
├── analysis/                           ← Local analysis notebooks + generated figures
│   ├── sparse_retrieval/tier1_sparse_analysis.ipynb
│   ├── dense_retrieval/tier2_dense_analysis.ipynb
│   ├── hybrid/tier3_hybrid_analysis.ipynb (+ img/)
│   ├── agentic/tier40/                         T4.0 analysis + img/
│   ├── agentic/tier41/                         T4.1 analysis + figures/
│   ├── agentic/tier42/                         T4.2 analysis + figures/ + round2/
│   └── RQ3/                                    RQ3 evaluator analysis (kept here because it consumes RQ1 outputs)
│
├── tests/                              ← pytest
│   ├── sparse_retrieval/test_preprocessing.py
│   ├── dense/test_dense_retriever.py
│   ├── hybrid/test_hybrid_retriever.py
│   ├── hybrid/test_llm_reranker.py
│   └── agentic/test_crag.py, test_react.py
│
├── docs/
│   ├── BSARD_Paper.pdf
│   ├── thesis_proposal_BSARD_RAG_v2.3.pdf
│   └── thesis_presentation_v3.4.pptx
│
├── output/                             ← Data root (BSARD_DATA_DIR; large files, NOT in git)
│   ├── bsard_corpus.db
│   ├── bsard_hf_articles.parquet
│   ├── embeddings/
│   ├── vector_stores/
│   └── results/                          ← ALL experiment result JSONs live here
│       ├── sparse_retrieval/               (12)
│       ├── dense_retrieval/                (10)
│       ├── hybrid/                         (15)
│       └── agentic/
│           ├── llm_judge/llm_rerank/         T4.0 binary
│           ├── llm_judge/0to10/              T4.0 numeric
│           ├── CRAG/                         T4.1
│           └── ReAct/                        T4.2
│
├── README.md                            Repo-level overview
├── RETRIEVAL_PROJECT.md                 This file
├── TIER1_SPARSE_RETRIEVAL_PLAN.md
├── TIER2_DENSE_RETRIEVAL_PLAN.md
├── TIER3_HYBRID_RETRIEVAL_PLAN.md
├── TIER40_LLM_JUDGE_PLAN.md
├── TIER41_AGENTIC_CRAG_PLAN.md
├── TIER42_AGENTIC_REACT_PLAN.md
├── REACT_ARCHITECTURE_v2.md             T4.2 architecture reference
├── requirements.txt
└── .gitignore
```

**Layout notes:**

- **Result JSONs live under `output/results/<tier>/`** inside the gitignored data root. `evaluation/` contains code and persisted runtime data only — never result outputs.
- **Canonical T3 orchestrator is `scripts/hybrid/run_hybrid_experiments.py`** (it produced the 15 result JSONs in `output/results/hybrid/`). `scripts/evaluation/tier3/run_hybrid_experiments.py` is an earlier draft retained for reference; it uses cross-encoder re-ranking for T3-C instead of SGDR.
- **`scripts/evaluation/tier4/round2/`** holds late-stage T4.2 diagnostic re-runs (null-action bucketing, max-token audit) — not part of the canonical T4.2 numbers but cited in the T4.2 analysis notebook.
- **mGTE evaluator** (`retrieval/agentic/cross_encoder_client.py`): file is retained but unused by any canonical Tier 4 run. All canonical Tier 4 runs use LLaMA 3.1 8B via the shared `OllamaClient`.

---

## 7. Data Access Patterns

### Load questions for evaluation

```python
import sqlite3, json

DB = "path/to/output/bsard_corpus.db"
conn = sqlite3.connect(DB)
conn.row_factory = sqlite3.Row

test_questions = {
    row["question_id"]: {
        "text":       row["question_text"],
        "ground_truth": json.loads(row["relevant_article_ids"]),
    }
    for row in conn.execute(
        "SELECT question_id, question_text, relevant_article_ids "
        "FROM questions WHERE split = 'test'"
    )
}
```

### Load articles for indexing

```python
import pandas as pd

# BSARD subset only (for retrieval index)
df = pd.read_parquet("path/to/output/bsard_articles_only.parquet",
                     columns=["article_id", "bsard_id", "law_code",
                               "article_number", "article_text",
                               "token_count", "hierarchy_path",
                               "chapter_title", "section_title"])

df = df[df["article_text"].notna()]   # drop the 1,338 failed extractions
```

### Access citation neighbours

```python
import json

def get_neighbours(article_id: int, conn) -> dict:
    row = conn.execute(
        "SELECT cross_reference_ids, cited_by_ids FROM articles WHERE article_id = ?",
        (article_id,)
    ).fetchone()
    return {
        "outgoing": json.loads(row["cross_reference_ids"] or "[]"),
        "incoming": json.loads(row["cited_by_ids"] or "[]"),
    }
```

---

## 8. Environment and Storage Rules

- **Virtual environment:** always execute scripts inside the project-local `.venv/`
- **Large files:** vector store indices, embedding files, and result dumps live under the gitignored data root, not in the Git repo
- **Database:** treat `output/bsard_corpus.db` as **read-only** — never write to it
- **Parquet/JSONL files:** also read-only source data
- **Result files:** save per-experiment results as JSON in `output/results/<tier>/` under the data root; they are not committed to Git
- **Model weights:** do not commit to Git; download on first use from HuggingFace Hub

---

## 9. Research Question (this repo)

| RQ | Question | Tiers |
|---|---|---|
| **RQ1** | How do lexical, dense, hybrid, and agentic retrieval approaches compare in their ability to retrieve relevant context from regulatory document corpora, and under what query characteristics and document conditions does each approach excel or fail? | Tiers 1–4 (this repo) |

RQ2 (structure-aware retrieval — chunking, metadata, PageIndex) and RQ3 (autonomous reference-free evaluation) are sibling components (`RQ2_Structure_Aware_Retrieval`, `RQ3_Autonomous_Evaluation`). The RQ3 evaluator analysis under `analysis/RQ3/` is the one piece of RQ3 work kept here, because it consumes RQ1 result JSONs and the 48-question stratified subset directly.

---

## 10. Corpus Provenance — Encoding Note

`bsard_full_verify.csv` (the source of `law_code`, metadata) is **UTF-8 encoded**. Always read with `encoding="utf-8"`. An earlier bug (reading as `latin-1`) caused garbled law code names (e.g. "DÃ©mocratie"); this has been corrected in both the pipeline and the database. If you encounter any `Ã` characters in law code strings, apply `s.encode('latin-1').decode('utf-8')` to fix them.
