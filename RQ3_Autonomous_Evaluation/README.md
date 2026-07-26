# RQ3_Autonomous_Evaluation

> Part of the [**bsard-rag-thesis**](../README.md) mono-repo (RQ3). A reusable
> evaluation harness installed into the other components' venvs; it consumes the
> `rq1/` data subset.

**Canonical evaluation metrics service for the BSARD RAG thesis**

| | |
|---|---|
| **Author** | Marios Paschalidis |
| **Affiliation** | KU Leuven, Master of Artificial Intelligence |
| **Thesis** | *Enhancing Performance and Quality of Context Retrieval in RAG Systems* |
| **Research Question** | **RQ3** — Do autonomous (reference-free) evaluation metrics correlate with supervised retrieval metrics, and can they reliably replace manual evaluation for RAG system ranking? |
| **Status** | 🟢 Core evaluation harness implemented (Tiers 0–3) |

---

## 1. Purpose

This project implements **Research Question 3** of the master thesis and serves as the **shared external evaluation service** consumed by all retrieval experiments across the thesis (RQ1, RQ2).

The separation from retrieval projects is deliberate:

- **Single source of truth** — all metric logic lives here; no drift between tiers.
- **Clean interface** — RQ1 produces retrieval results; this project evaluates them.
- **Independent updates** — evaluation methodology can evolve without touching retrieval code.

---

## 2. Four-Tier Evaluation Stack

The evaluation harness implements a four-tier metric hierarchy:

### Tier 0 — Efficiency Metrics (Latency & Throughput)

Measures the computational cost of retrieval — a practical prerequisite for any production RAG deployment.

| Metric | Description |
|---|---|
| **Latency distribution** | mean, std, p50, p90, p95, p99, min, max (ms) |
| **Throughput** | Queries per second (QPS), total retrieval time |
| **Timing breakdown** | Per-stage timing: embedding, search, re-ranking (if provided) |
| **Stage fractions** | % of total time spent in each pipeline stage |
| **Index build time** | One-off cost (seconds) for index construction |

### Tier 1 — BSARD Paper Metrics (Exact Replication)

Reproduces the evaluation protocol of the original BSARD paper for direct comparability.

| Metric | Primary? | Description |
|---|---|---|
| **Recall@k** | ✅ (k=100) | `\|relevant ∩ top-k\| / \|relevant\|` — NOT Hit Rate |
| **MRR@100** | — | Mean reciprocal rank of first relevant article |
| k values | — | {1, 5, 10, 100} (fixed for BSARD comparability) |

### Tier 2 — Full Supervised IR Metrics

Three panels covering complementary evaluation dimensions:

| Panel | Metrics | Requires |
|---|---|---|
| **P1 — Rank-unaware** | Precision@k, Recall@k, F1@k, Hit Rate@k | Binary qrels |
| **P2 — Rank-aware** | MRR@k, MAP@k, NDCG@k | Binary qrels |
| **P3 — Set-utility** | RA-nWG@k, N-Recall4+@k | Graded qrels (1–5) |
| **P3 — ID-based** | IDPrecision@k, IDRecall@k | Binary qrels (always on by default) |

The two ID-based metrics mirror MAP@k and Recall@k but are computed with explicit ID-set logic. Their primary purpose is the **UMBRELA bridge**: when the harness re-runs Tier 2 with UMBRELA-produced graded qrels (output keyed `T2-umbrela/`), they measure how well the retrieval ranking aligns with autonomous relevance grades rather than BSARD ground truth — enabling the core RQ3 system-ranking comparison.

### Tier 3 — Autonomous LLM-Based Metrics (RQ3 Core)

Retrieval-only scope: all components evaluate retrieved contexts without requiring a pre-generated answer.

| Component | AQS weight | Cost (222q × k=10) | Requirement | What It Measures |
|---|---|---|---|---|
| **UMBRELA** | 0.35 | ~$0.17 API | API only | 0–3 relevance grade per (query, doc) pair; produces TREC qrels that bridge into Tier 2 NDCG/MAP |
| **eRAG** | 0.30 | ~$0.25 API / free local | GPU (16 GB bf16 / 4.5 GB 4-bit) or API | Whether retrieved article enables correct reasoning (grounded-answer utility) |
| **ARES** | 0.20 | $0 inference + ~$15 one-time fine-tune | GPU for fine-tune; CPU ok for inference | context_relevance with PPI confidence bounds |
| **RAGAS WA** | 0.15 | ~$0.24 API | API only | LLMContextPrecisionWithoutReference + HyDE synthetic response |
| **RAGAS WB** | — (diagnostic) | ~$0.20 API | API only | Same as WA but query-as-response; zero extra LLM calls — validates HyDE contribution |

> **Pricing note:** Estimates use gpt-4o-mini rates as of 2026-04-05. Update when new pricing is available or when actual run times are measured. See [RQ3_PROJECT_CONTEXT.md §9](RQ3_PROJECT_CONTEXT.md) for the full breakdown.

**AQS (Autonomous Quality Score):** Weighted average of UMBRELA, eRAG, ARES, and RAGAS WA mean scores (weights above). RAGAS WB is excluded from AQS — it runs as a diagnostic comparison against WA. The UMBRELA-based Tier 2 re-run (`T2-umbrela/`) is the primary empirical contribution of RQ3: do autonomous rankings agree with BSARD-supervised rankings?

---

## 3. Interface Contract with RQ1

### 3.1 Input — What RQ1 Sends

**Standard retrieval payload** (Tiers 0–2):

```python
{
    "question_id": int,
    "retrieved_article_ids": list[int],   # ranked, best first
    "ground_truth_article_ids": list[int],
    "latency_ms": float,
    "metadata": {
        "experiment_id": str,
        "tier": str,                       # "tier1" | "tier2" | "tier3" | "tier4"
        "split": str,                      # "test" | "val"
        "hyperparameters": dict,
    }
}
```

**Additional payload for Tier 3** — RQ1 must also provide ranked article texts:

```python
# contexts_with_ranks: per query, the top-k retrieved articles with their texts
{
    question_id: [
        (article_id: int, article_text: str, rank: int),  # rank is 1-based
        ...
    ]
}
```

The `article_text` is the full BSARD article text (concatenation of `article`, `description`, and `reference` fields). The `article_id` must match the BSARD `id` column so it aligns with `qrels`.

For **Tier 4 systems** (agentic), RQ1 must additionally pass `judge_model_tier4` — the name of the LLM used inside the agentic system — so the harness can ensure UMBRELA uses a *different* model (cross-model discipline, prevents self-preference bias).

### 3.2 Output — What This Project Returns

- **Per-query metrics**: Recall@k, MRR@k, NDCG@k, AP@k for each question
- **Aggregate metrics**: Mean over all questions, including `T2/P3/IDPrecision@k` and `T2/P3/IDRecall@k`
- **Significance block**: Paired t-test p-values vs. a named anchor experiment
- **Stratified breakdowns**: Same metric set per stratum (single/multi article, lexical alignment, cross-references)
- **Autonomous Tier 2 block** (`T2-umbrela/...`): full Tier 2 metric set re-computed with UMBRELA graded qrels — present only when Tier 3 runs with the `umbrela` component

### 3.3 Call Mechanism

The harness supports three call mechanisms; RQ1 uses the Python-import path
(installed editable into the RQ1 venv), with the CLI and file-handoff paths
available for decoupled or batch use:

| Option | Method | Pros |
|---|---|---|
| **A — Python import** | `pip install -e .` then `from bsard_evaluation import ...` | Zero overhead, type-safe |
| **B — CLI** | `bsard-eval --input payload.json --output metrics.json` | Fully decoupled |
| **C — File handoff** | RQ1 writes JSON → the harness reads, computes, writes | Inspectable, batch-friendly |

> **Note:** For Tier 0, RQ1 must additionally pass per-query latencies (`{query_id: latency_ms}`) and optionally a timing breakdown dict with keys like `embedding_ms`, `search_ms`, `rerank_ms`, `index_build_s`.

---

## 4. Data

This component ships **no data of its own** — it is a pure evaluation harness.
It consumes the **RQ1 result JSONs**, which are the `rq1/` subset of the
companion Hugging Face dataset
[`mpaschalidis/bsard-rag-thesis-data`](https://huggingface.co/datasets/mpaschalidis/bsard-rag-thesis-data).

Large and binary files are **not** committed to git; they download into a local
gitignored data root:

- **Data root:** env `BSARD_DATA_DIR`, default `<repo>/output` (gitignored).
- **Download** (from the mono-repo root):

  ```bash
  python data_tooling/download_combined_hf.py --subset rq1
  ```

- The Tier-3 analysis helper reads the RQ1 outputs directly from the sibling
  `RQ1_Retrieval_Methods/` folder (see §14 for the `--rq1-root` flow).

No symlink or junction to external/cloud storage is needed — the data downloads
into the local data root and is gitignored there. The published repository
contains only source code, configuration, scripts, notebooks, documentation,
small JSON result fixtures, and the `.gitignore`.

Python is run from the component's local virtual environment (`.venv/`):

```bash
python -m venv .venv
# Windows:        .\.venv\Scripts\Activate.ps1
# macOS / Linux:  source .venv/bin/activate
pip install -e .
```

---

## 5. Project Structure

```
RQ3_Autonomous_Evaluation/
│
├── .gitignore                      # Excludes venv, large/binary files
├── README.md                       # This file
├── requirements.txt                # Python dependencies
├── setup.py                        # Editable install (pip install -e .)
├── RQ3_PROJECT_CONTEXT.md          # Interface contract & full specification
│
├── bsard_evaluation/               # Core library (installable package)
│   ├── __init__.py                 # Package init, version
│   ├── config.py                   # TierConfig, K_PRESETS, factory functions
│   ├── harness.py                  # EvaluationHarness — main entry point
│   ├── tier0_efficiency.py         # Tier 0 — Latency, throughput, timing
│   ├── tier1_bsard.py              # Tier 1 — BSARD paper metrics
│   ├── tier2_supervised.py         # Tier 2 — Full supervised IR (3 panels)
│   ├── tier3_autonomous.py         # Tier 3 — Autonomous LLM-based metrics
│   ├── significance.py             # Statistical significance testing
│   └── stratify.py                 # Stratified analysis helpers
│
├── scripts/
│   ├── evaluate.py                 # CLI: reads retrieval JSON → writes metrics
│   └── compare.py                  # CLI: compare two experiments + significance
│
├── analysis/                       # Jupyter notebooks for RQ3 analysis
│   └── (rq3_analysis.ipynb)        # Cross-tier correlation analysis
│
├── results/                        # Input: RQ1 result JSONs (read-only, small)
│
├── output/                         # Output: computed metrics, reports, plots
│
├── tests/
│   └── test_metrics.py             # Unit tests for all metric functions
│
└── .venv/                          # ⛔ GITIGNORED — local virtual environment
```

---

## 6. Metrics Reference

### 6.1 Recall@k (Primary)

```
Recall@k = |{relevant ∩ top-k}| / |{total relevant}|
```

- **k values:** {1, 5, 10, 20, 50, 100, 200, 500}
- **Primary:** Recall@100 (Tier 2+), Recall@10 (Tier 1 significance testing)
- **Not** Hit Rate — measures evidence coverage, not binary success

### 6.2 MRR@k

```
MRR@k = (1/N) × Σ (1 / rank_of_first_relevant_in_top_k)
```

### 6.3 NDCG@k

```
DCG@k = Σ gain_i / log₂(i + 2)
NDCG@k = DCG@k / IDCG@k
```

Binary relevance (0/1) for Tier 1. Graded (1–5) for Tier 2 Panel 3.

### 6.4 MAP@k

```
AP@k = (1/R) × Σ precision@i × rel(i)
MAP@k = mean(AP@k) over all queries
```

### 6.5 Statistical Significance

- **Primary test:** Two-sided paired t-test (`scipy.stats.ttest_rel`)
- **Threshold:** p < 0.05
- **Anchor:** `bm25_default_test`
- **Optional:** Wilcoxon signed-rank test (non-parametric alternative)

---

## 7. Stratified Analysis

Three stratification axes (labels received from RQ1, not recomputed here):

| Axis | Strata | Rationale |
|---|---|---|
| **Lexical alignment** | `lexically_aligned` / `semantically_paraphrased` / `middle` | Isolates BM25 failure condition (bottom quartile R@10 ≈ 0.01) |
| **Article count** | `single_article` / `multi_article` | Multi-doc queries are harder for Recall@k |
| **Cross-references** | `with_cross_refs` / `without_cross_refs` | Tests citation-aware retrieval benefit |

---

## 8. BSARD Dataset Key Facts

| Property | Value |
|---|---|
| Corpus source | `maastrichtlawtech/bsard` (HuggingFace) |
| Corpus size | ~22,600 articles |
| Article PK | `id` (int32) |
| Test questions | 222 |
| Validation questions | 177 (20% of train, seed=42) |
| Mean relevant articles/question | 6.18 |

---

## 9. RQ1 Experiment Inventory

Experiments whose results this project evaluates:

| Tier | Methods | Primary Metric |
|---|---|---|
| **T1** | BM25 Okapi, TF-IDF, FTS5 | Recall@10 |
| **T2** | Dense bi-encoder (8 models) | Recall@100 |
| **T3** | RRF fusion, Linear fusion, Cross-encoder re-ranking | Recall@100 |
| **T4.0** | LLM-as-a-Judge | Recall@100 |
| **T4.1** | CRAG (BM25/dense/hybrid backbone) | Recall@100 |
| **T4.2** | ReAct | Recall@100 |

---

## 10. Dependencies

### Core (always required)

```
numpy, scipy, pandas
```

### Optional

| Group | Packages | When needed |
|---|---|---|
| Visualisation | `matplotlib`, `seaborn` | Analysis notebooks |
| Tier 3 — UMBRELA & eRAG | `openai` | UMBRELA judge calls; eRAG via API or Ollama endpoint |
| Tier 3 — RAGAS WA / WB | `ragas>=0.2`, `datasets` | RAGAS workarounds |
| Tier 3 — ARES | `ares-ai` | Fine-tuned T5-large judge + PPI |
| Tier 3 — eRAG local GPU | Ollama + `llama3.1:8b` model | eRAG without API cost |
| Development | `pytest`, `jupyter`, `ipykernel` | Testing & notebooks |

Install groups via:

```bash
# Core only (Tiers 0–2)
pip install -e .

# With visualization
pip install -e ".[viz]"

# With Tier 3
pip install -e ".[tier3]"

# Everything
pip install -e ".[viz,tier3,dev]"
```

---

## 11. Quick Start

```powershell
# 1. Activate venv
.\.venv\Scripts\Activate.ps1

# 2. Install in editable mode
pip install -e ".[viz,dev]"

# 3. Run tests
pytest tests/

# 4. Use from another component (e.g., RQ1)
#    In RQ1's venv: pip install -e "../RQ3_Autonomous_Evaluation"
```

```python
# From RQ1 or any other project:
from bsard_evaluation import EvaluationHarness
from bsard_evaluation.config import supervised_standard, full_evaluation

# Tiers 0–2 only (no LLM calls):
harness = EvaluationHarness(supervised_standard())
results = harness.evaluate(
    qrels=qrels,
    run=run,
    latencies=latencies,               # per-query ms for Tier 0
    timing_breakdown=timing_breakdown,  # optional stage timings
)

# Full evaluation including Tier 3 autonomous metrics:
harness = EvaluationHarness(full_evaluation())
results = harness.evaluate(
    qrels=qrels,
    run=run,
    latencies=latencies,
    timing_breakdown=timing_breakdown,
    queries=queries,                        # {question_id: query_text}
    contexts_with_ranks=contexts_with_ranks, # {question_id: [(article_id, text, rank), ...]}
    # judge_model_tier4="gpt-4o",           # set for Tier 4 systems (cross-model discipline)
    # answers=answers,                       # optional: enables ARES answer dimensions
)
# results contains both "T2/..." (supervised) and "T2-umbrela/..." (autonomous)
```

---

## 12. Validation

The harness is validated against the RQ1 reference implementation and unit tests:

- `evaluate()` reproduces the metrics computed by the sibling
  `RQ1_Retrieval_Methods` retrieval code on the same inputs.
- `add_significance()` matches the RQ1 per-query paired-t-test vectors.
- Stratified metrics match the `stratified` blocks in RQ1 result JSONs.
- Unit tests cover edge cases: empty results, all relevant in top-1, no relevant
  in top-k, single vs multi-article questions.
- The interface contract is exercised end-to-end against RQ1 experiments.
- Tier 3 components run independently and produce valid AQS scores.

---

## 13. Related Projects

| Component | Folder | Relationship |
|---|---|---|
| **RQ1 — Retrieval** | `RQ1_Retrieval_Methods` | Producer of retrieval results consumed by this harness |
| **RQ2 — Structure-aware** | `RQ2_Structure_Aware_Retrieval` | Also consumes this evaluation harness |
| **Corpus database** | `bsard2currentlawmatching` | BSARD corpus preparation and verification |

---

## 14. Relationship to RQ1 — and where the RQ3 analysis lives

**What this repository is.** A reusable, pip-installable **evaluation harness**
(`bsard-evaluation`). It defines the metric logic for all tiers and is consumed
by the retrieval projects; it deliberately contains *no* retrieval code.

**Where the RQ3 results and write-up live.** The substantive RQ3 *analysis* — the
thesis chapter, figures, and the validation/aggregation study of the autonomous
evaluators — lives in the **RQ1 repository** under `analysis/RQ3/`, because that
analysis is run against RQ1's retrieval outputs. This repository hosts only the
reusable harness plus a small Tier-3 analysis workspace (`analysis/rq3_tier3/`)
and its aggregate result tables. If you are looking for the RQ3 findings, see
the RQ1 project; if you want the evaluation code, you are in the right place.

**How RQ1 depends on this package.** RQ1 installs this package in editable mode
into its own virtual environment and imports it directly:

```bash
# From within the RQ1 project's venv:
pip install -e "../RQ3_Autonomous_Evaluation"
```

```python
from bsard_evaluation import EvaluationHarness
```

**How this package reaches back into RQ1.** One analysis helper here,
[`analysis/rq3_tier3/scripts/assemble_panel.py`](analysis/rq3_tier3/scripts/assemble_panel.py),
reads RQ1 outputs (the Tier-3 query subset, strata, corpus parquet, and per-system
result sidecars). It assumes RQ1 is a **sibling directory** by default and accepts
an explicit override:

```bash
python analysis/rq3_tier3/scripts/assemble_panel.py \
    --rq1-root "../RQ1_Retrieval_Methods" --families sparse,dense
```

So inside the mono-repo the layout is:

```
bsard-rag-thesis/
├── RQ1_Retrieval_Methods/      # retrieval experiments + RQ3 analysis chapter
└── RQ3_Autonomous_Evaluation/     # this component — the evaluation harness
```

The corpus artefacts the assembler reads (parquet, qrels) are not shipped here;
obtain them from RQ1 — i.e. the **`rq1/` subset** of the combined Hugging Face
dataset [`mpaschalidis/bsard-rag-thesis-data`](https://huggingface.co/datasets/mpaschalidis/bsard-rag-thesis-data)
(pull it with `python data_tooling/download_combined_hf.py --subset rq1` from the
mono-repo root; CC BY-NC-SA 4.0 — see [DATA_LICENSE.md](DATA_LICENSE.md)).

---

## 15. License

- **Source code** — MIT (see [`LICENSE`](LICENSE)).
- **BSARD-derived result tables / artefacts** — CC BY-NC-SA 4.0
  (see [`DATA_LICENSE.md`](DATA_LICENSE.md)).
- API keys for the Tier-3 LLM metrics are read from the environment; copy
  [`.env.example`](.env.example) to `.env.local` and fill in `OPENAI_API_KEY`.

---

*This README is the primary documentation for the RQ3 component. For the detailed interface specification, see [RQ3_PROJECT_CONTEXT.md](RQ3_PROJECT_CONTEXT.md).*
