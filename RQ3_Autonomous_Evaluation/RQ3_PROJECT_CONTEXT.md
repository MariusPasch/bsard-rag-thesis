# RQ3 — Evaluation Harness Specification

**Author:** Marios Paschalidis | KU Leuven, Master of Artificial Intelligence
**Thesis:** Enhancing Performance and Quality of Context Retrieval in RAG Systems
**Component:** `RQ3_Autonomous_Evaluation` — the reusable multi-tier evaluation harness for the thesis. This document specifies its role, its interface contract with the retrieval components, and its metric/tier definitions.

---

## 1. Role in the Thesis Architecture

The thesis consists of three research questions, each a component of the mono-repo:

| RQ | Component | Responsibility |
|---|---|---|
| RQ1 | `RQ1_Retrieval_Methods` | Development and execution of retrieval methods (Tier 1–4 sparse/dense/hybrid/agentic) over the BSARD corpus |
| RQ2 | `RQ2_Structure_Aware_Retrieval` | Structure-aware tooling; extends T4.2 ReAct with document graph/citation awareness |
| RQ3 | `RQ3_Autonomous_Evaluation` | Evaluation metric computation, significance testing, stratified analysis, and reporting — shared by all RQ experiments |

The separation is deliberate: the retrieval components *produce* retrieval results; this component *evaluates* them. This enforces a clean interface, prevents metric logic from drifting independently per tier, and keeps evaluation methodology in one place. The retrieval components install this harness into their own virtual environments and call it directly.

---

## 2. Interface Contract with RQ1

### 2.1 What RQ1 Sends

Each RQ1 experiment produces a **retrieval result JSON** file saved to `evaluation/<tier>/results/<experiment_id>.json`. The schema is:

```json
{
  "experiment_id": "bm25_lemmatize_concat_2x_test",
  "timestamp": "2026-03-21T14:30:00",
  "model_or_method": "bm25",
  "hyperparameters": {
    "k1": 1.5,
    "b": 0.75
  },
  "preprocessing": {
    "normalization": "lemmatize",
    "stopword_list": "spacy_fr_custom",
    "field_weighting": "concat_2x",
    "embedding_prefix": "none"
  },
  "token_length_audit": {
    "fraction_truncated": 0.0,
    "max_tokens_observed": 0
  },
  "training_regime": "zero_shot",
  "latency_ms_mean": 42.3,
  "latency_ms_std": 8.1,
  "metrics": {},
  "significance_vs_anchor": {
    "p_value_recall10": null,
    "significant": null
  },
  "stratified": {}
}
```

The per-query retrieval results are passed separately as a structured payload:

```python
# Per-query retrieval payload sent to the evaluation project:
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

### 2.2 What the Harness Returns

**Per-query metrics** (for each question):
```python
{
    "question_id": int,
    "recall@1": float,
    "recall@5": float,
    "recall@10": float,
    "recall@20": float,
    "recall@50": float,
    "recall@100": float,
    "recall@200": float,
    "recall@500": float,
    "mrr@10": float,
    "mrr@100": float,
    "ndcg@10": float,
    "ap@100": float,
}
```

**Aggregate metrics** (mean over all questions):
```python
{
    "Recall@1": float, "Recall@5": float, "Recall@10": float,
    "Recall@20": float, "Recall@50": float, "Recall@100": float,
    "Recall@200": float, "Recall@500": float,
    "MRR@10": float, "MRR@100": float,
    "NDCG@10": float,
    "MAP": float, "MAP@100": float,
}
```

**Significance block** (vs. a named anchor experiment):
```python
{
    "p_value_recall10": float,
    "p_value_recall100": float,
    "significant": bool,    # based on primary_k (default: Recall@10)
}
```

**Stratified breakdowns** (same metric set per stratum):
```python
{
    "single_article": { ...aggregate metrics... },
    "multi_article": { ...aggregate metrics... },
    "lexically_aligned": { ...aggregate metrics... },
    "semantically_paraphrased": { ...aggregate metrics... },
    "with_cross_refs": { ...aggregate metrics... },
    "without_cross_refs": { ...aggregate metrics... },
}
```

### 2.3 Call Mechanism

The harness supports three call mechanisms. RQ1 uses the Python-import path; the
CLI and file-handoff paths are available for decoupled or batch use:

- **Python import:** The harness is installed as a local package via `pip install -e`. The caller imports it with `from bsard_evaluation import EvaluationHarness`. Zero overhead; the caller's virtual environment shares the package.
- **CLI / subprocess:** The harness exposes a CLI invoked as `subprocess.run(["bsard-eval", "--input", "payload.json", "--output", "metrics.json"])`. Fully decoupled, slower (subprocess launch per call).
- **File handoff:** The caller writes a retrieval result JSON; the harness reads it, computes metrics, and writes a metrics JSON. Decoupled and inspectable — suited to long-running batch jobs.

---

## 3. Metrics

All metrics follow standard IR definitions (trec_eval convention).

### 3.1 Recall@k

```
Recall@k = |{relevant articles} ∩ {top-k retrieved}| / |{total relevant articles}|
```

**Not** Hit Rate (binary "at least one relevant in top-k"). k values: {1, 5, 10, 20, 50, 100, 200, 500}.

**Primary metric:** Recall@100 (matches BSARD paper; used for dense, hybrid, agentic tiers).
**Secondary metric:** Recall@10 (used for significance testing; tuning metric for Tier 3 fusion α).

### 3.2 MRR@k (Mean Reciprocal Rank)

```
MRR@k = (1/N) * Σ (1 / rank_of_first_relevant_in_top_k)
```

k values: {10, 100}.

### 3.3 NDCG@10 (Normalized Discounted Cumulative Gain)

Binary relevance (0/1). Ideal DCG computed from the gain vector sorted descending.

```
DCG@k = Σ gain_i / log2(i + 2)     (i from 0)
NDCG@k = DCG@k / IDCG@k
```

### 3.4 MAP / MAP@100 (Mean Average Precision)

```
AP@k = (1/R) * Σ precision@i * rel(i)     for i=1..k, R = number of relevant articles
MAP@k = mean(AP@k) over all queries
```

`MAP` (backward-compat alias) and `MAP@100` are numerically identical in the current implementation.

### 3.5 Statistical Significance

**Primary test:** Two-sided paired t-test on per-query Recall@k scores (`scipy.stats.ttest_rel`).
**Threshold:** p < 0.05.
**Primary k:** 10 (Tier 1), 100 (Tier 2+).

Each experiment comparison names an **anchor experiment** (e.g., `bm25_default_test`). The test receives two aligned per-query Recall@k vectors (experiment and anchor) over the same question set.

Optional secondary test: Wilcoxon signed-rank test (non-parametric alternative).

---

## 4. Stratified Analysis

Three stratification axes, all derived from the BSARD question set:

### 4.1 Lexical Alignment (`lex_align`)

**Method:** Compute mean BM25 score of each query against its ground-truth articles using the anchor BM25 index (no normalization, text_only field weighting). Quartile thresholds over the full question set.

- Top quartile (≥ Q75) → `lexically_aligned`
- Bottom quartile (≤ Q25) → `semantically_paraphrased`
- Middle → `middle` (not reported in main tables, but present in data)

**Rationale:** The biggest failure mode of sparse retrieval on BSARD is semantically paraphrased queries (R@10 ≈ 0.01 for BM25 on bottom quartile). This stratum isolates that failure condition.

### 4.2 Article Count (`article_count`)

- `n_relevant_articles == 1` → `single_article`
- `n_relevant_articles > 1` → `multi_article`

**Rationale:** Multi-article questions require the retriever to find multiple non-contiguous documents, which is harder and penalised more heavily by Recall@k.

### 4.3 Cross-References (`cross_ref`)

- Any ground-truth article has cross-references → `with_cross_refs`
- None of the ground-truth articles have cross-references → `without_cross_refs`

**Note:** Cross-reference information is stored on articles, not questions. Derived by checking whether any ground-truth article has outgoing legal citations.

### 4.4 Strata Persistence

Strata assignments are computed once (requires the BM25 anchor retriever from RQ1) and persisted to `evaluation/query_strata.json` in RQ1. The harness receives strata labels as part of the per-query payload — it does not recompute them.

```json
// query_strata.json format (integer keys as strings)
{
    "503": {
        "bm25_score": 0.312,
        "lex_align": "lexically_aligned",
        "article_count": "multi_article",
        "cross_ref": "with_cross_refs"
    }
}
```

---

## 5. BSARD Dataset — Key Facts

The harness uses these facts to verify metric inputs:

| Property | Value |
|---|---|
| Corpus | HuggingFace `maastrichtlawtech/bsard`, name=`"corpus"` |
| Corpus size | ~22,600 articles |
| Article primary key | `id` (int32) — this is what appears in `retrieved_article_ids` and `ground_truth_article_ids` |
| Question set (test) | 222 questions — official BSARD test split |
| Question set (val) | 177 questions — 20% of BSARD train, seed=42, persisted in `evaluation/split_ids.json` |
| Question set (train) | 709 questions — remaining 80% of BSARD train |
| Mean relevant articles per test question | 6.18 |
| Ground truth field | `article_ids` (comma-separated int list in HuggingFace questions) |

**HuggingFace column names (corpus):**

| Column | Type | Description |
|---|---|---|
| `id` | int32 | Article primary key — used in all ground-truth lookups |
| `article` | string | Canonical legal article text |
| `code` | string | Law code name |
| `article_no` | string | Article number within code |
| `description` | string | Hierarchical headings joined |
| `reference` | string | Full legal citation string |
| `law_type` | string | "regional" or "national" |

---

## 6. Metric / Significance / Stratify API

The metric, significance, and stratification logic lives in this harness. RQ1
owns only the retrieval-domain pieces (running retrieval, persisting strata); the
boundary below reflects that division of responsibility.

### 6.1 Metrics (`bsard_evaluation`)

```python
def evaluate(
    results: dict[int, list[int]],
    ground_truth: dict[int, list[int]],
    k_values: list[int] | None = None,
) -> dict[str, float]:
    """Compute aggregate IR metrics. k_values default: [1,5,10,20,50,100,200,500]"""

def per_query_recall(
    results: dict[int, list[int]],
    ground_truth: dict[int, list[int]],
    k: int = 10,
) -> list[float]:
    """Per-query Recall@k vector for significance testing."""
```

Key implementation decisions:
- `Recall@k = |relevant ∩ top-k| / |relevant|` — NOT Hit Rate
- MRR@10 and MRR@100 computed in a single pass (break on first relevant)
- NDCG@10 uses binary gains (0/1), ideal sorted descending
- `MAP` and `MAP@100` are identical (AP over top-100 candidates)

### 6.2 Significance

The harness owns the significance logic; RQ1 keeps the retrieval orchestration
(`run_experiment()` measuring latency and calling metrics) and result
serialisation (`save_result()`). Significance is computed by:

```python
def add_significance(
    result: dict,
    anchor_result: dict,
    k_values: list[int] | None = None,
    primary_k: int = 10,
) -> dict:
    """
    Two-sided paired t-test (scipy.stats.ttest_rel) on per-query Recall@k.
    primary_k determines the 'significant' bool flag (threshold p < 0.05).
    Tier 1: primary_k=10. Tier 2+: primary_k=100.
    """
```

### 6.3 Stratification

Ownership splits along the retrieval/evaluation boundary:

**RQ1** (retrieval-domain knowledge):
- `compute_strata()` — requires the BM25 anchor retriever
- `save_strata()` / `load_strata()` — persists to `query_strata.json`

**This harness:**
- `filter_by()` — slices the ground_truth dict by stratum label

---

## 7. Result JSON Schema (Complete)

The canonical schema produced by RQ1 experiments and consumed by this harness:

```json
{
  "experiment_id": "string — unique, used as filename",
  "timestamp": "ISO8601",
  "model_or_method": "string",
  "hyperparameters": {},
  "preprocessing": {
    "normalization": "lemmatize | stem | none",
    "stopword_list": "spacy_fr_custom | none",
    "field_weighting": "concat_2x | text_only | ...",
    "embedding_prefix": "query:/passage: | none"
  },
  "token_length_audit": {
    "fraction_truncated": 0.0,
    "max_tokens_observed": 0
  },
  "training_regime": "zero_shot | fine_tuned_bsard",
  "latency_ms_mean": 0.0,
  "latency_ms_std": 0.0,
  "metrics": {
    "Recall@1": 0.0, "Recall@5": 0.0, "Recall@10": 0.0,
    "Recall@20": 0.0, "Recall@50": 0.0, "Recall@100": 0.0,
    "Recall@200": 0.0, "Recall@500": 0.0,
    "MRR@10": 0.0, "MRR@100": 0.0,
    "NDCG@10": 0.0,
    "MAP": 0.0, "MAP@100": 0.0
  },
  "significance_vs_anchor": {
    "p_value_recall10": null,
    "significant": null
  },
  "stratified": {
    "single_article": {},
    "multi_article": {},
    "lexically_aligned": {},
    "semantically_paraphrased": {},
    "with_cross_refs": {},
    "without_cross_refs": {}
  }
}
```

---

## 8. RQ1 Experiment Inventory

All experiments whose results this harness processes:

| Tier | Method | Split | Experiment IDs (examples) | Primary metric |
|---|---|---|---|---|
| T1 | BM25 Okapi variants | test | `bm25_default_test`, `bm25_lemmatize_concat_2x_test` | R@10 |
| T1 | TF-IDF variants | test | `tfidf_lemmatize_text_only_test` | R@10 |
| T1 | FTS5 | test | `fts5_default_test` | R@10 |
| T2 | Dense bi-encoder (8 models) | test | `dense_mE5_large_instruct_test` | R@100 |
| T3 | RRF fusion | test | `rrf_k60_test` | R@100 |
| T3 | Linear fusion | test | `linear_alpha04_test` | R@100 |
| T3 | Cross-encoder re-ranking | test | `rerank_mMiniLM_top100_test` | R@100 |
| T4.0 | LLM-as-a-Judge | test | `llm_rerank_binary_top50_test` | R@100 |
| T4.1 | CRAG (BM25/dense/hybrid backbone) | test | `crag_bm25_test`, `crag_dense_test`, `crag_hybrid_test` | R@100 |
| T4.2 | ReAct | test | `react_*_test` | R@100 |

Anchor experiment for significance testing: `bm25_default_test` (Tier 1 baseline).

---

## 9. Tier 3 Cost, Time and Requirements Estimates

> **⚠️ These figures were estimated on 2026-04-05 using gpt-4o-mini pricing at that date.**
> Model pricing and GPU performance change frequently. **Update this section whenever new API pricing is published or when actual run times are measured on the target hardware.**

Assumptions: BSARD test set — 222 queries, k = 10 retrieved documents per query = **2,220 (query, doc) pairs**. BSARD legal articles average ~400 tokens; queries average ~25 tokens.

### Per (query, doc) pair

| Component | Exec time / pair | GPU required | API cost / pair | Total (222q × k=10) |
|---|---|---|---|---|
| **UMBRELA** | 0.5–1.5 s (API) | No | ~$0.000075 | **~$0.17** |
| **eRAG** | 1–2.5 s (local GPU) · 0.5–1 s (API) | Yes for local; see VRAM below | ~$0.000112 (API) | **~$0.25 (API)** / free (local) |
| **RAGAS WA** | ~1 s (RAGAS phase) | No | ~$0.000107 (incl. HyDE) | **~$0.24** |
| **RAGAS WB** | ~1 s | No | ~$0.0000885 | **~$0.20** |
| **ARES** | 5–15 ms (GPU) · 80–150 ms (CPU) | GPU for one-time fine-tune only; CPU sufficient for inference | $0 (local inference) | **$0** + one-time fine-tune |

### Detailed breakdown

**UMBRELA** — `gpt-4o-mini` (pricing at time of estimate: $0.15/1M input · $0.60/1M output)
- Token count per pair: ~480 input (system prompt + query + document + scale instructions), ~5 output
- Per-pair cost: `(480 × 0.15 + 5 × 0.60) / 1,000,000 = $0.000075`
- Sequential execution: 2,220 pairs × ~1 s ≈ 37 min. With 10× async parallelism: **~4 min**
- No GPU. Requires `openai` package only.

**eRAG** — LLaMA 3.1 8B via Ollama (local) or Together.ai API ($0.18/1M tokens)
- Token count per pair: ~520 input (system + query + document), ~100 output (answer attempt)
- API cost: `(520 + 100) × 0.18 / 1,000,000 = $0.000112` → **~$0.25 total**
- Local GPU — A100 40 GB (~100 tok/s generation): ~1 s/pair → **~37 min** sequential
- Local GPU — RTX 3090 24 GB (~40 tok/s generation): ~2.5 s/pair → **~92 min** sequential
- VRAM: ~16 GB bf16 (fits RTX 3090 / A100) or ~4.5 GB 4-bit quantised via `ollama`
- Prefix model name with `"ollama/"` to route to local Ollama endpoint.

**RAGAS Workaround A** — `gpt-4o-mini`
- Phase 1 — HyDE (1 call per query, amortised over k=10): ~$0.0000021/pair
- Phase 2 — RAGAS chunk scoring (1 call per pair): ~600 input, ~10 output → $0.000096/pair
- Total per pair: **~$0.000107** → ~$0.24 for 2,220 pairs
- No GPU. Requires `openai`, `ragas>=0.2`, `datasets`.

**RAGAS Workaround B** — `gpt-4o-mini`
- Same as WA Phase 2 only (query text replaces synthetic response — no HyDE call)
- Token count per pair: ~550 input, ~10 output → `$0.0000885`
- **~$0.20 total**. No GPU. Requires `ragas>=0.2`, `datasets`.

**ARES** — T5-large fine-tuned locally
- *One-time fine-tuning:* T5-large (~770M params) on BSARD synthetic training data
  - A100 40 GB: ~2–4 h · RTX 3090 / T4: ~6–8 h
  - Google Colab Pro estimate: ~$12–18 one-time
  - Requires GPU with ≥ 6 GB VRAM plus ~150 human-annotated BSARD examples for PPI calibration
- *Inference per pair:* single T5-large forward pass
  - GPU (4 GB+ VRAM): **5–15 ms/pair** → ~18 s for 2,220 pairs
  - CPU: **80–150 ms/pair** → ~3–7 min for 2,220 pairs
- API cost: **$0** — fully local after fine-tuning

### Full evaluation budget across all RQ1 systems

| Component | Cost per system | 13 systems total |
|---|---|---|
| UMBRELA | ~$0.17 | ~$2.21 |
| eRAG (API) | ~$0.25 | ~$3.25 |
| RAGAS WA | ~$0.24 | ~$3.12 |
| RAGAS WB (diagnostic) | ~$0.20 | ~$2.60 |
| ARES (inference) | $0.00 | $0.00 |
| **Total** | **~$0.86** | **~$11.18** |

Plus the one-time ARES fine-tuning cost (~$12–18).

---

## 10. Dependencies

Minimal dependency footprint for Tiers 0–2 — no ML models, no GPU required:

```
numpy
scipy          # ttest_rel, wilcoxon
pandas         # optional, for result table generation
matplotlib     # optional, for charts
seaborn        # optional, for charts
jupyter        # optional, for analysis notebooks
```

Tier 3 additional dependencies (install only what you run):

```
openai              # UMBRELA, eRAG (API), RAGAS WA
ragas>=0.2          # RAGAS WA, RAGAS WB
datasets            # RAGAS WA, RAGAS WB
ares-ai             # ARES
# For eRAG local GPU: install Ollama + llama3.1:8b model
```

The harness does not import retrieval code from the RQ1 component. All data exchange happens through the interface in §2 (JSON files, CLI, or installed package — see §2.3).

---

## 11. Component Structure

```
RQ3_Autonomous_Evaluation/
│
├── bsard_evaluation/               # Core library (installable package)
│   ├── __init__.py                 # Package init, version
│   ├── config.py                   # TierConfig, K_PRESETS, factory functions
│   ├── harness.py                  # EvaluationHarness — main entry point
│   ├── tier0_efficiency.py         # Tier 0 — latency, throughput, timing
│   ├── tier1_bsard.py              # Tier 1 — BSARD paper metrics
│   ├── tier2_supervised.py         # Tier 2 — full supervised IR (3 panels)
│   ├── tier3_autonomous.py         # Tier 3 — autonomous LLM-based metrics
│   ├── significance.py             # Paired t-test, Wilcoxon
│   └── stratify.py                 # filter_by() stratum slicing
│
├── scripts/
│   ├── evaluate.py                 # CLI: reads retrieval JSON → writes metrics JSON
│   └── compare.py                  # CLI: compare two experiments + significance table
│
├── analysis/                       # Jupyter notebooks for RQ3 analysis
├── results/                        # Input: RQ1 result JSONs (read-only)
├── tests/
│   └── test_metrics.py             # Unit tests for all metric functions
│
├── README.md
└── requirements.txt
```

---

## 12. Validation

The harness is validated against the RQ1 reference implementation and unit tests:

- `evaluate()` reproduces the metrics computed by the RQ1 retrieval code on the same inputs.
- `add_significance()` matches the RQ1 per-query paired-t-test vectors.
- Stratified metrics match the `stratified` blocks in RQ1 result JSONs.
- Unit tests cover edge cases: empty results, all relevant in top-1, no relevant in top-k, single-article vs multi-article questions.
- The interface contract is exercised end-to-end against RQ1 experiments.

---

*This document specifies the RQ3 evaluation harness scope and its interface with the retrieval components.*
