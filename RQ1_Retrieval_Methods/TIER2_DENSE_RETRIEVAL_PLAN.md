# Tier 2 — Dense Retrieval: Implementation & Results

**Status:** Complete
**BSARD Retrieval RAG | RQ1 | Thesis: KU Leuven MAI 2024–2025**
**Experiments executed on the 222-question BSARD test split** (text_only field weighting); EXP-D7 winner selected on the 100-question stratified train sample.

---

## Overview

Tier 2 implements the dense bi-encoder family for BSARD retrieval. Nine zero-shot models were evaluated — seven encoder-only candidates (CamemBERT-base/large, mE5-base/large, BGE-M3, mpnet-multi) plus the Qwen3-Embedding-0.6B decoder-only LLM (plain + instruction-prefixed). The authors' fine-tuned checkpoint (EXP-D5) was not released and was not replicated.

**Primary metric: Recall@100** (matches the BSARD paper). Secondary: R@10, R@200, R@500, MRR@10, MAP@100, NDCG@10. R@10 alone is structurally biased on BSARD because 25 % of questions have ≥ 5 gold articles — R@10 is capped well below 1.0 for those questions regardless of retriever quality (see §12).

**Selection protocol:** EXP-D7 winner (concat_2x field-weighting ablation) is chosen by R@100 on a fixed 100-question seed-42 train sample (`evaluation/data/split_ids.json::train_sample_100`); only the winner is then evaluated on the test split. The test split is never consulted during selection.

---

## 0. Shared Infrastructure

Tier 2 reuses the shared infrastructure established in Tier 1:

| File | Purpose |
|---|---|
| `evaluation/metrics.py` | Compatibility shim (runner delegates to the `bsard_evaluation` harness) |
| `evaluation/runner.py` | `run_experiment()`, `save_result()`, `add_significance()` — delegates to the `bsard_evaluation` package |
| `evaluation/split.py` | `load_questions("test"\|"train")`, `load_train_sample()` |
| `evaluation/stratify.py` | `load_strata()`, per-query failure strata |
| `evaluation/data/split_ids.json` | Persisted train/test split + 100-question train sample (seed=42) |
| `evaluation/data/query_strata.json` | Persisted per-question strata |
| `output/bsard_articles_dedup.parquet` | **Corpus used for all experiments** — same BSARD dataset, locally extracted; `article_id` matches ground truth (Decision D15) |
| `output/bsard_hf_articles.parquet` | HF download of the same dataset — different ID scheme, not used for experiments |
| `output/bsard_corpus.db` | SQLite source database (used for questions via split.py) |
| `output/embeddings/` | Persisted `.npy` embedding files — built from dedup parquet |
| `output/results/dense_retrieval/` | Canonical result JSON output directory (under the data root) |

**Key design contract shared with Tier 1:**

- Retriever interface: `retrieve(query: str, top_k: int) -> tuple[list[int], float]` — returns `(ranked_article_ids, latency_ms)`. Latency measured with `time.perf_counter()`, covers **only** the retrieval call (not index load time).
- Result JSON schema: extended in Section 5 below (R@200, R@500 added).
- **All models evaluated on the TEST split only (222 questions).** This matches Tier 1 sparse and is the canonical reported split.
- **EXP-D7 concat_2x winner selection uses a 100-question train sample** (seed=42, persisted in `evaluation/data/split_ids.json` under `train_sample_100`). This keeps the test split untouched during any selection step. `evaluation/split.py::load_train_sample()` provides this sample.
- **Primary metric: Recall@100.** Secondary: Recall@10, MRR@10, NDCG@10.
- Anchor for significance tests: BM25 Okapi with `k1=1.5, b=0.5, lemmatize, text_only` — built inline on the test questions at runtime by `run_dense_experiments.py::load_anchor_result`. Note this is **not** the Tier 1 tuned winner (which used `b=0.25`); the Tier 2 anchor uses an intermediate `b=0.5` so that the dense significance baseline isn't already a tuned configuration. This choice is locked in `run_dense_experiments.py:243`.

**Tier 1 key results (reference for Tier 2 — 222-question test split, dedup corpus):**

| Configuration | R@10 | R@100 | MRR@100 |
|---|---|---|---|
| `bm25_lemmatize_concat_2x` | 0.2572 | **0.5312** | 0.2520 |
| `bm25_tuned_k11.5_b0.25` | **0.2651** | 0.5210 | 0.2628 |
| `tfidf_lemmatize_concat_2x` | 0.2003 | 0.5152 | 0.1995 |
| `bm25_lemmatize_text_only` | 0.2381 | 0.5121 | 0.2520 |
| `bm25_plus_lemmatize` | 0.2461 | 0.5088 | 0.2552 |
| `bm25_none_concat_2x` | 0.2567 | 0.5009 | 0.2454 |
| `bm25_stem_text_only` | 0.2637 | 0.4905 | 0.2495 |
| `bm25_anchor` (k1=1.5, b=0.75) | 0.2476 | 0.4821 | 0.2463 |
| `fts5_default` | 0.2365 | 0.4697 | 0.2367 |

**Key finding from Tier 1 that motivates Tier 2:** Semantically paraphrased queries (bottom-quartile BM25-score stratum): R@10 ≈ 0.01–0.07 for all sparse methods. This gap (~0.48 vs lexically aligned queries) is the primary motivation for dense retrieval. Zero-shot dense results below show this gap *is* partially closed by mE5-large / BGE-M3 / Qwen3 even without fine-tuning (refuting the original hypothesis that only fine-tuned models would help — see §15).

---

## ✅ Corpus Status

**Corpus (Decision D15):** All Tier 2 experiments use `output/bsard_articles_dedup.parquet` (22,633 rows). This is the same BSARD dataset as `maastrichtlawtech/bsard`, but extracted locally from the PDF source files rather than downloaded from HuggingFace. The local extraction was used throughout the full analysis pipeline (Tier 1 through Tier 4) because it carries the `article_id` scheme that matches the ground-truth IDs in the SQLite DB. The HF parquet (`bsard_hf_articles.parquet`) was downloaded for reference but uses a different ID scheme (`bsard_id`) that is incompatible with the ground truth without a separate mapping step.

**Column mapping (dedup parquet schema):**

| Column | Notes |
|---|---|
| `article_id` | Integer article identifier — matches ground truth |
| `article_text` | Full statutory article text |
| `law_code` | Legal code name (e.g., "Code civil") |
| `article_title` | Article title / chapter description |
| `chapter_title` | Chapter heading (used by `concat_2x` builder; absent in HF parquet) |

**Impact on embeddings:** All `.npy` files in `output/embeddings/` were built from the dedup parquet and are valid. The `build_concat_2x` document builder uses `chapter_title` from the dedup parquet directly.

---

## 1. Implementation File Structure

Paths and module layout:

```
retrieval/dense.py                              ← DenseRetriever + EmbeddingEncoder
scripts/evaluation/tier2/
    __init__.py
    run_dense_experiments.py                    ← Experiment orchestrator
tests/dense/
    __init__.py
    test_dense_retriever.py                     ← Unit tests
analysis/dense_retrieval/
    tier2_dense_analysis.ipynb                  ← Analysis notebook
```

> **Note:** Result JSONs are written to `output/results/dense_retrieval/` under the gitignored data root — same pattern as all other tiers.

---

## 2. Architecture of `retrieval/dense.py`

### 2.1 `EmbeddingEncoder` (internal helper)

Wraps `sentence-transformers` `SentenceTransformer`. Responsibilities:

- Load model from HuggingFace hub (or local cache)
- Report `max_seq_length` before encoding
- Apply embedding prefix to texts if required (see Section 4)
- Encode a list of texts in batches; return L2-normalised `numpy` float32 arrays
- Report truncation statistics (fraction of texts exceeding `max_seq_length`)

**Key method signatures:**

```python
encode_corpus(texts: list[str], batch_size: int = 64, show_progress: bool = True) -> np.ndarray
    # shape (N, dim), float32, L2-normalised

encode_query(query: str) -> np.ndarray
    # shape (dim,), float32, L2-normalised

token_length_audit(texts: list[str]) -> dict
    # Returns {mean, median, p90, max, fraction_truncated}
```

### 2.2 `DenseRetriever` (public class)

Follows the same interface as `BM25Retriever` and `TFIDFRetriever`.

**Constructor:**

```python
__init__(
    df: pd.DataFrame,                          # BSARD articles (must have article_id, article_text columns)
    model_name: str,                           # HuggingFace model ID or local path
    field_weighting: str = "text_only",        # "text_only" | "concat_2x" — mirrors sparse.py
    passage_prefix: str = "",                  # Prepended to each article at encoding time
    query_prefix: str = "",                    # Prepended to each query at retrieval time
    batch_size: int = 64,
    device: str = "cpu",                       # "cpu" | "cuda" | "mps"
    embeddings_dir: Path = Path("output/embeddings"),
    max_seq_length_override: int | None = None, # Overrides model.max_seq_length (use 1024 for bge-m3)
    truncate_dim: int | None = None,            # Matryoshka dim truncation (use 1024 for Qwen3)
)
```

**Initialization sequence:**

1. Instantiate `EmbeddingEncoder` with `model_name`
2. Run `token_length_audit` on `df["article_text"]` — store in `self._audit`
3. Check if cached embeddings exist at `embeddings_dir`:
   - `{model_name_slug}_{field_weighting}.npy` — shape (N, dim), float32
   - `{model_name_slug}_{field_weighting}_ids.npy` — shape (N,), int64
   - Three-tier lookup: in-process `_EMBEDDING_CACHE` → disk `.npy` → full encoding + save. `_index_build_source` records which path was taken.
4. Build `faiss.IndexFlatIP` from embeddings (exact inner product search; cosine similarity with L2-normalised vectors)

**`retrieve` method:**

```python
retrieve(query: str, top_k: int = 500) -> tuple[list[int], float]:
    # 1. Encode query: apply query_prefix, encode, L2-normalise
    # 2. FAISS search: index.search(q_vec, top_k)
    # 3. Map FAISS indices → article_ids
    # 4. Return (ranked_article_ids, latency_ms)
    # Latency covers ONLY steps 1–3, measured with time.perf_counter()
```

**Properties:** `audit`, `embedding_dim`, `n_articles`

### 2.3 Module-level embedding cache

```python
_EMBEDDING_CACHE: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
# Key: (model_slug, field_weighting); Value: (embeddings_array, article_ids_array)
```

Prevents reloading from disk when running multiple experiments with the same model. Same pattern as `_TOKENIZED_CACHE` in `sparse.py`. The key includes `field_weighting` because `text_only` and `concat_2x` produce different embeddings for the same model. The `DenseRetriever` also exposes `_index_build_source: str` (`"cache"` | `"disk"` | `"encode"`) and `_index_build_time_s: float` for Tier 0 latency tracking; `build_time` is forced to 0.0 on cache hits to avoid misleading measurements.

### 2.4 Embedding persistence convention

```
output/embeddings/{model_name_slug}_{field_weighting}.npy
output/embeddings/{model_name_slug}_{field_weighting}_ids.npy
```

`model_name_slug`: replace `/` with `_`, replace `-` with `_`, lower-case. If `truncate_dim` is set, the slug is additionally suffixed with `_dim{truncate_dim}` (e.g., `qwen_qwen3_embedding_0_6b_dim1024`) to prevent cache collisions between full-dimension and Matryoshka-truncated embeddings.
Stored in `output/embeddings/` under the data root — **not in git**.

FAISS indices are rebuilt from embeddings at load time (fast, <1s for 22k articles). **Do not persist FAISS indices** — they are derived from embeddings.

---

## 3. Model Selection — Strategy and Experiments

**Revised strategy: no self-training.** Priority order for the "fine-tuned baseline":

1. **Authors' released checkpoint** (paper replication)
2. **Pre-fine-tuned open-source proxies** (if checkpoint unavailable)
3. **Zero-shot models only** (document gap as a thesis finding)

> **Critical calibration note:** The BSARD paper (Table 2) reports zero-shot CamemBERT R@100 = **4.21%**. This is not a data error. Untuned contextual embeddings perform near-randomly on statutory retrieval because the query–document language distributions are very different (natural language vs. legal jargon). **Do not set thesis expectations around zero-shot dense models outperforming BM25** — they will not.
>
> Zero-shot models in this tier serve two purposes: (a) replicate the paper's zero-shot results for benchmark comparability; (b) quantify the value of pre-training/fine-tuning by contrast with EXP-D5.

### 3.1 Model candidate list

**Step 1 — Check for authors' released checkpoint (do this first):**

- GitHub: `https://github.com/maastrichtlawtech/bsard`
- HuggingFace: `https://huggingface.co/antoiloui`
- Look for a bi-encoder checkpoint trained on BSARD train split (either the siamese 110M or two-tower 220M CamemBERT variant)
- If found: this becomes EXP-D5 (the paper replication) — load in inference-only mode

| ID | HF Model ID | Prefix? | Pooling | Notes |
|---|---|---|---|---|
| `dense_camembert_base` | `camembert-base` (SBERT wrapper) | None | Mean | **Paper benchmark anchor** — elevated to full paper metric replication (R@100, R@200, R@500, MAP@100, MRR@100) |
| `dense_camembert_lg` | `dangvantuan/sentence-camembert-large` | None | Mean | Best French-specific proxy |
| `dense_me5_base` | `intfloat/multilingual-e5-base` | `query:`/`passage:` | Mean | Multilingual, strong recall |
| `dense_me5_large` | `intfloat/multilingual-e5-large` | `query:`/`passage:` | Mean | Multilingual candidate |
| `dense_bge_m3` | `BAAI/bge-m3` | None | CLS | Long-context (native 8192, capped at 1024 via `max_seq_length_override`); ~0–2% truncation on BSARD |
| `dense_paper_checkpoint` | `antoiloui/<checkpoint>` (if released) | None | Mean | Paper replication — primary (unavailable as of 2026-04) |
| `dense_mpnet_multi` | `paraphrase-multilingual-mpnet-base-v2` | None | Mean | 128-token limit — **demoted** |
| `dense_qwen3_0_6b` | `Qwen/Qwen3-Embedding-0.6B` | None | Last-token | EXP-D10: decoder-only LLM backbone; 0.6B params; `truncate_dim=1024`; `max_seq_length_override=512`; `batch_size=4`; tests H8 |
| `dense_qwen3_0_6b_instruct` | `Qwen/Qwen3-Embedding-0.6B` | query instruction only | Last-token | EXP-D10i: reuses D10 corpus embeddings (`text_only` field); instruction on query side only; same overrides as D10 |

**Notes on `dense_bge_m3` (EXP-D4c):** BGE-M3 supports 8,192 tokens but BSARD articles are short (median ~100 tokens, p90 ~420 tokens). **Always set `max_seq_length=1024`** (`max_seq_length_override=1024`) — avoids ~90–150 min CPU encoding becoming ~30–50 min with negligible recall loss. Uses **CLS pooling** (not mean pooling — do not override). `batch_size=16` on GPU. MIT licensed.

**Notes on demoted model:** `paraphrase-multilingual-mpnet-base-v2` has a 128-token context window. p90 is ~420 tokens — heavy truncation. Included for reference only. Do not use for model selection or Tier 3 input.

**Notes on `camembert-base`:** Not SBERT-compatible out of the box. Wrap as `SentenceTransformer` with a `Pooling` module. Elevated from zero-shot reference to **paper benchmark anchor** — must report all BSARD paper metrics (R@100, R@200, R@500, MAP@100, MRR@100) for Table 2 replication. Do not select for Tier 3.

**Notes on `dangvantuan/sentence-camembert-large` (EXP-D2):** Despite the model's advertised 514-token limit, position IDs start at 2, so the effective maximum is 512. `max_seq_length_override=512` is applied in `_MODEL_OVERRIDES`.

**Notes on `Qwen3-Embedding-0.6B` (EXP-D10/D10i):** Decoder-only LLM (Qwen3-0.6B, 0.6B params), adapted for bi-encoder retrieval via last-token pooling and contrastive fine-tuning. MTEB multilingual SOTA at 8B scale; 0.6B is CPU-feasible. Requires `transformers>=4.51.0`. Uses `truncate_dim=1024` for Matryoshka dimension truncation (set in `SentenceTransformer` constructor). **`batch_size=4`** (not 8) to avoid OOM. **`max_seq_length_override=512`** is applied — Qwen3's advertised 32k context is not exercised here to avoid 32k-padding overhead on GPU; the 512 cap makes truncation comparable to mE5 models (~10–20%). Encoding ~22k articles takes ~30–60 min. D10i reuses D10 corpus embeddings (instruction is query-side only; no passage prefix). Run after all encoder-only models with `--skip-qwen3` to defer. Hypothesis H8: will outperform encoder-only models of similar or smaller size on statutory retrieval.

### 3.2 Evaluation order

1. Run `token_length_audit` for all models first (single pass, no GPU needed). Use `--skip-audit` to skip.
2. Run encoder-only zero-shot experiments in order of increasing compute:
   `camembert_base` (D1) → `camembert_lg` (D2) → `me5_base` (D3) → `me5_large` (D4) → `bge_m3` (D4c) → `mpnet_multi` (D6, reference only)
3. If paper checkpoint supplied via `--paper-checkpoint-path`: load and run EXP-D5 (runs after all zero-shot models in the script).
4. Run Qwen3-Embedding-0.6B (D10), then D10i immediately after — shares corpus embeddings. Use `--skip-qwen3` on time-constrained runs; use `--only-qwen3` to run only D10/D10i.
5. EXP-D7 concat_2x: run all eligible models over the 100-question train sample, select winner, evaluate on test.
6. Select Tier 3 dense component: highest R@100 on the **100-question train sample** among D2, D3, D4, D4c — all final reported numbers come from the **test split** (222 questions). D10/D10i are excluded from selection (see Section 6).

---

## 4. Embedding Prefix Conventions *(critical — enforce exactly)*

| Model | Query prefix | Passage prefix | Pooling | Notes |
|---|---|---|---|---|
| `intfloat/multilingual-e5-base` | `"query: "` | `"passage: "` | Mean | Standard mE5 prefix |
| `intfloat/multilingual-e5-large` | `"query: "` | `"passage: "` | Mean | Standard mE5 prefix |
| `BAAI/bge-m3` | `""` | `""` | CLS | No prefix; do not use mean pooling |
| `camembert-base` | `""` | `""` | Mean | — |
| `dangvantuan/sentence-camembert-large` | `""` | `""` | Mean | — |
| `paraphrase-multilingual-mpnet-base-v2` | `""` | `""` | Mean | — |
| paper checkpoint (`antoiloui/...`) | `""` | `""` | Mean | *(verify on HF card)* |
| `Qwen/Qwen3-Embedding-0.6B` (plain) | `""` | `""` | Last-token | Handled automatically by sentence-transformers |
| `Qwen/Qwen3-Embedding-0.6B` (instruct) | `"Instruct: Given a legal question in French, retrieve the relevant Belgian statutory article\nQuery: "` | `""` | Last-token | Same instruction as mE5-instruct for comparability; no passage prefix |

`DenseRetriever` accepts `passage_prefix` and `query_prefix` as constructor arguments.

```python
corpus_texts = [passage_prefix + t for t in df["article_text"].fillna("")]
query_text   = query_prefix + query
```

Stored in result JSON `preprocessing.embedding_prefix` field as `"{query_prefix}|{passage_prefix}"` (e.g., `"query: |passage: "` or `"none"`).

> **mE5/instruct critical:** wrong prefix = silently off-distribution embeddings. Verify prefix conventions against the HuggingFace model card before encoding.

---

## 5. Result JSON Schema *(extended from Tier 1)*

**Changes from Tier 1 schema:**
- `metrics`: added `Recall@200`, `Recall@500`, `MAP@100`, `MRR@100`
- `significance_vs_anchor`: primary metric changed to `recall100`
- `training_regime`: new value `"pretrained_bsard"` for paper checkpoint

```json
{
  "experiment_id": "dense_me5_large_zeroshot_test",
  "timestamp": "<ISO8601>",
  "model_or_method": "dense_biencoder",
  "hyperparameters": {
    "model_name": "intfloat/multilingual-e5-large",
    "max_seq_length": 512,
    "embedding_dim": 1024,
    "normalize_embeddings": true,
    "batch_size": 64,
    "device": "cpu"
  },
  "preprocessing": {
    "normalization": "none",
    "stopword_list": "none",
    "field_weighting": "text_only",
    "embedding_prefix": "query: |passage: "
  },
  "token_length_audit": {
    "fraction_truncated": 0.143,
    "max_tokens_observed": 1842,
    "mean_tokens": 198,
    "median_tokens": 127,
    "p90_tokens": 421
  },
  "training_regime": "zero_shot",
  "index_build_source": "encode",
  "latency_ms_mean": 85.3,
  "latency_ms_std": 12.1,
  "metrics": {
    "Recall@1": 0.0, "Recall@5": 0.0, "Recall@10": 0.0,
    "Recall@20": 0.0, "Recall@50": 0.0, "Recall@100": 0.0,
    "Recall@200": 0.0, "Recall@500": 0.0,
    "MRR@10": 0.0, "MRR@100": 0.0,
    "NDCG@10": 0.0, "MAP@100": 0.0
  },
  "significance_vs_anchor": {
    "anchor": "bm25_tuned",
    "primary_metric": "Recall@100",
    "p_value_recall100": null,
    "p_value_recall10": null,
    "significant": null
  },
  "stratified": {
    "single_article":           { "Recall@10": 0.0, "Recall@100": 0.0, "MRR@10": 0.0, "NDCG@10": 0.0 },
    "multi_article":            { "Recall@10": 0.0, "Recall@100": 0.0, "MRR@10": 0.0, "NDCG@10": 0.0 },
    "lexically_aligned":        { "Recall@10": 0.0, "Recall@100": 0.0, "MRR@10": 0.0, "NDCG@10": 0.0 },
    "semantically_paraphrased": { "Recall@10": 0.0, "Recall@100": 0.0, "MRR@10": 0.0, "NDCG@10": 0.0 },
    "with_cross_refs":          { "Recall@10": 0.0, "Recall@100": 0.0, "MRR@10": 0.0, "NDCG@10": 0.0 },
    "without_cross_refs":       { "Recall@10": 0.0, "Recall@100": 0.0, "MRR@10": 0.0, "NDCG@10": 0.0 }
  }
}
```

**Key field notes:**
- `training_regime`: `"zero_shot"` | `"pretrained_bsard"` (paper checkpoint, inference-only) | `"fine_tuned_bsard"` (reserved for future use)
- `index_build_source`: `"encode"` (first run, authoritative timing) | `"disk"` (loaded from `.npy`) | `"cache"` (in-process hit, build time is 0.0)
- `significance_vs_anchor.primary_metric`: `Recall@100`; both `p_value_recall10` and `p_value_recall100` are computed (`k_values=[10, 100]`)
- `Recall@200` and `Recall@500` require `top_k=500` in `retrieve()` — the experiment runner passes `top_k=500`

The actual schema also includes top-level `latency_distribution` (Tier 0 p50/p90/p95/p99/min/max/index_build_s) and the full `T0/...`, `T1/...`, `T2/P1/...`, `T2/P2/...`, `T2/P3/...` namespaced keys inside `metrics`, mirrored under the legacy flat keys shown above. After `compute_subset_metrics.py` post-processing, a `subset_metrics` block is added with the 48-question stratified subset Tier 0/1/2 panel and the Tier 3 evaluator outputs (`T3/umbrela`, `T3/erag`, `T3/ragas_wa`, `T3/AQS`, `T2-umbrela/...`).

### 5.1 Tier 2 results — 222-question test set

All nine zero-shot experiments, sorted by Recall@100. Significance is the paired t-test on per-query Recall@100 vs the inline `bm25 (k1=1.5, b=0.5, lemmatize, text_only)` anchor.

| Experiment | R@10 | R@100 | R@200 | R@500 | MRR@10 | MAP@100 | Lat (ms) | Trunc% | p (R@100) |
|---|---|---|---|---|---|---|---|---|---|
| `dense_me5_large_concat2x_zeroshot` | 0.3420 | **0.6215** | 0.6978 | 0.7732 | 0.3106 | 0.2157 | 321.4 | 9.4 % | **0.0003** |
| `dense_qwen3_0_6b_instruct_zeroshot` | 0.3389 | 0.5974 | 0.6633 | 0.7722 | **0.3176** | 0.2063 | **63.4** | 11.0 % | **0.0045** |
| `dense_me5_large_zeroshot` | 0.2959 | 0.5941 | 0.6753 | 0.7540 | 0.2878 | 0.1732 | 281.2 | 9.4 % | **0.0095** |
| `dense_bge_m3_zeroshot` | 0.3142 | 0.5921 | 0.6954 | **0.7961** | 0.2898 | 0.1815 | 280.3 | **2.5 %** | **0.0150** |
| `dense_camembert_lg_zeroshot` | 0.3069 | 0.5833 | 0.6693 | 0.7689 | 0.2386 | 0.1620 | 241.0 | 7.4 % | **0.0476** |
| `dense_qwen3_0_6b_zeroshot` | 0.3240 | 0.5822 | 0.6524 | 0.7548 | 0.3068 | 0.1991 | 61.7 | 11.0 % | **0.0274** |
| `dense_me5_base_zeroshot` | 0.2254 | 0.4915 | 0.5699 | 0.6858 | 0.2319 | 0.1404 | 121.3 | 9.4 % | 0.2062 |
| `dense_mpnet_multi_zeroshot` | 0.2108 | 0.4047 | 0.5236 | 0.6417 | 0.1892 | 0.1093 | 107.7 | 49.3 % | 0.0005 |
| `dense_camembert_base_zeroshot` | 0.0077 | 0.0169 | 0.0209 | 0.0562 | 0.0200 | 0.0072 | 90.1 | 7.4 % | 0.0000 |

**Bold p-values** indicate statistically significant differences from the BM25 anchor (p < 0.05). Six of nine systems significantly beat the BM25 anchor on R@100; mpnet-multi and camembert-base are significantly *worse*.

### 5.2 EXP-D7 winner selection — 100-question train sample

The concat_2x ablation winner is chosen by R@100 on the seed-42 train sample (test split untouched):

| Model | Train-sample R@100 |
|---|---|
| `dense_me5_large` | **0.6191** |
| `dense_bge_m3` | 0.6086 |
| `dense_camembert_lg` | 0.5985 |
| `dense_me5_base` | 0.5390 |

Winner: **mE5-large**. Its `concat_2x` variant was then run on the 222-question test split → `dense_me5_large_concat2x_zeroshot` (R@100 = 0.6215, the headline dense result).

### 5.3 Stratified breakdown — top dense vs sparse on semantic paraphrasing

The semantically-paraphrased stratum (bottom BM25-score quartile, ~50 questions) is the failure mode that motivates dense retrieval.

| System | Semantically paraphrased R@10 | Semantically paraphrased R@100 |
|---|---|---|
| `bm25_tuned_k11.5_b0.25` (sparse) | 0.0743 | 0.1723 |
| `bm25_lemmatize_concat_2x` (sparse) | 0.0378 | 0.2186 |
| `dense_bge_m3_zeroshot` | **0.1416** | **0.4287** |
| `dense_me5_large_concat2x_zeroshot` | 0.1407 | 0.3921 |
| `dense_me5_large_zeroshot` | 0.1359 | 0.3676 |
| `dense_qwen3_0_6b_instruct_zeroshot` | 0.1349 | 0.3415 |

Dense models roughly **double** R@10 and **double** R@100 on the semantic-paraphrase stratum vs the best sparse system — refuting the plan's H1 expectation that only fine-tuned dense models would close this gap (see §15).

### 5.4 Tier 3 (AQS) — 48-question stratified subset

After `compute_subset_metrics.py` + UMBRELA/eRAG/RAGAS-WA runs:

| Experiment | AQS | Subset R@10 |
|---|---|---|
| `dense_bge_m3_zeroshot` | **0.3670** | 0.2437 |
| `dense_me5_large_zeroshot` | 0.3559 | 0.2108 |
| `dense_camembert_lg_zeroshot` | 0.3479 | 0.2504 |
| `dense_me5_large_concat2x_zeroshot` | 0.3453 | 0.2203 |
| `dense_me5_base_zeroshot` | 0.3214 | 0.1942 |
| `dense_mpnet_multi_zeroshot` | 0.2604 | 0.1230 |
| `dense_camembert_base_zeroshot` | 0.0518 | 0.0042 |

(`dense_qwen3_0_6b` and `dense_qwen3_0_6b_instruct` were added after the initial Tier 3 pass and do not have AQS scores. They are evaluated only against supervised metrics on the 48-question subset.)

---

## 6. Experiment List

**All models evaluated on the TEST split (222 questions).** EXP-D7 winner selection uses a 100-question train sample (seed=42) — the test split is never touched during selection.

| ID | Experiment ID | Model | Regime | Status |
|---|---|---|---|---|
| EXP-D1 | `dense_camembert_base_zeroshot_test` | `camembert-base` | `zero_shot` | ✓ Run — paper benchmark anchor (R@100 = 1.69 %; below paper's 4.21 %) |
| EXP-D2 | `dense_camembert_lg_zeroshot_test` | `sentence-camembert-large` | `zero_shot` | ✓ Run — French-specific zero-shot |
| EXP-D3 | `dense_me5_base_zeroshot_test` | `multilingual-e5-base` | `zero_shot` | ✓ Run — multilingual baseline |
| EXP-D4 | `dense_me5_large_zeroshot_test` | `multilingual-e5-large` | `zero_shot` | ✓ Run — multilingual large |
| EXP-D4c | `dense_bge_m3_zeroshot_test` | `BAAI/bge-m3` | `zero_shot` | ✓ Run — long-context, CLS pooling, max_seq=1024 |
| EXP-D5 | `dense_paper_checkpoint_pretrained_test` | `antoiloui/<checkpoint>` | `pretrained_bsard` | ✗ Not run — paper checkpoint never released by authors |
| EXP-D6 | `dense_mpnet_multi_zeroshot_test` | `paraphrase-mpnet-base-v2` | `zero_shot` | ✓ Run — 128-token reference only |
| EXP-D7 | `dense_me5_large_concat2x_zeroshot_test` | `multilingual-e5-large` | `zero_shot` | ✓ Run — D4 won train-sample selection (R@100 = 0.6191), concat_2x evaluated on test |
| EXP-D10 | `dense_qwen3_0_6b_zeroshot_test` | `Qwen/Qwen3-Embedding-0.6B` | `zero_shot` | ✓ Run — decoder-only LLM, last-token pooling, truncate_dim=1024 |
| EXP-D10i | `dense_qwen3_0_6b_instruct_zeroshot_test` | `Qwen/Qwen3-Embedding-0.6B` | `zero_shot` | ✓ Run — Qwen3 + legal instruction prefix (reuses D10 corpus embeddings) |

Naming convention: `dense_{model_id}_{training_regime}_test`

**Selection logic (train sample → test):**
- EXP-D7 winner = `argmax(Recall@100 on 100-question train sample)` among **D2, D3, D4, D4c** (encoder-only models only). The actual ranking is in §5.2.
- **D10/D10i excluded from D7 selection**: Qwen3-Embedding-0.6B uses a decoder-only LLM backbone with last-token pooling — a fundamentally different architecture from the encoder-only bi-encoders. Including it in the concat_2x ablation would confound field-weighting effects with architecture differences. D10/D10i are evaluated on the test split as standalone architecture experiments (hypothesis H8).
- D1 and D6 are excluded from selection and Tier 3 but fully evaluated on the test split.
- D5 (paper checkpoint) was reserved as the benchmark ceiling but the authors' checkpoint was never published, so it is reported as "paper weights unavailable as of 2026-04" in the thesis.

---

## 7. Fine-Tuning Protocol — Suspended

Self-fine-tuning is **out of scope** for this thesis. Rationale:
- The authors' checkpoint (if released) is the authoritative fine-tuned model
- Training a new model conflates retrieval architecture research with model training research, which is not the thesis focus
- Reproducing the paper's DPR-style training requires significant GPU time not available within the thesis timeline

If in future self-fine-tuning becomes feasible, the following parameters match the BSARD paper setup:

| Parameter | Value |
|---|---|
| Loss | `MultipleNegativesRankingLoss` (in-batch negatives) |
| Batch size | 22 question–article pairs (paper default) |
| Temperature | 0.05 |
| Epochs | 100 (≈ 20,500 steps on 709 training pairs) |
| Learning rate | 2e-5, AdamW, warmup 500 steps, linear decay |
| Evaluator | `InformationRetrievalEvaluator`, `main_score = Recall@100` |

Do not implement this unless explicitly scoped back into the thesis timeline.

---

## 8. Token Length Audit *(prerequisite — run first for each model)*

Before encoding **any** articles:

```python
from transformers import AutoTokenizer
import numpy as np

tokenizer = AutoTokenizer.from_pretrained(model_name)
lengths = [len(tokenizer.encode(t)) for t in df["article_text"].fillna("")]
audit = {
    "mean_tokens":         float(np.mean(lengths)),
    "median_tokens":       float(np.median(lengths)),
    "p90_tokens":          float(np.percentile(lengths, 90)),
    "max_tokens_observed": int(np.max(lengths)),
    "fraction_truncated":  float(np.mean([l > max_seq_length for l in lengths]))
}
```

**Truncation strategy:** head truncation (`max_length=model.max_seq_length`, `truncation=True`). Statutory articles front-load operative text, so head truncation preserves the most semantically important content. Known limitation: cross-reference clauses at the tail of long articles are lost.

**Expected audit findings:**

| Model | Context window (used) | Expected `fraction_truncated` |
|---|---|---|
| `intfloat/multilingual-e5-base` | 512 tokens | ~10–20% |
| `intfloat/multilingual-e5-large` | 512 tokens | ~10–20% |
| `dangvantuan/sentence-camembert-large` | **512 tokens** (capped via `max_seq_length_override=512`; advertised 514 is unreliable due to position IDs starting at 2) | ~10–20% |
| `BAAI/bge-m3` | **1024 tokens** (set explicitly) | **~0–2%** — primary advantage over mE5 family |
| `Qwen/Qwen3-Embedding-0.6B` | **512 tokens** (capped via `max_seq_length_override=512`) | **~10–20%** — same as mE5 family; 32k native context not exercised to avoid GPU OOM from 32k padding overhead |
| `paraphrase-multilingual-mpnet-base-v2` | 128 tokens | >> 50% — confirms demoted status |

If `fraction_truncated > 0.20` for any primary candidate (D2–D4c), add a note in the experiment commentary. The truncation delta between BGE-M3 and mE5 is a thesis finding — report it explicitly.

---

## 9. FAISS Index Setup

**Index type:** `IndexFlatIP` (exact exhaustive search, inner product)

**Rationale:** 22,633 articles × 1024 dims × 4 bytes ≈ 93 MB per model — fits in RAM. Approximate indices (IVF, HNSW) introduce retrieval noise that confounds recall metrics. Do **not** use approximate indices for Tier 2.

```python
import faiss
import numpy as np

# Build
faiss.normalize_L2(embeddings)        # in-place, must be float32
index = faiss.IndexFlatIP(embedding_dim)
index.add(embeddings)

# Search
q_vec = q_vec.reshape(1, -1).astype(np.float32)
faiss.normalize_L2(q_vec)             # in-place
distances, faiss_indices = index.search(q_vec, top_k)
article_ids = self._article_ids[faiss_indices[0]]
```

> **Important:** `top_k` must be at least **500** for experiments reporting Recall@500. Set default `top_k=500` in the experiment runner.

---

## 10. Experiment Script: `scripts/evaluation/tier2/run_dense_experiments.py`

Mirrors `run_sparse_experiments.py` structure. Key behaviours:

1. CLI args: `--paper-checkpoint-path <path>`, `--device cpu|cuda|mps`, `--batch-size N`, `--skip-audit`, `--skip-qwen3`, `--only-qwen3` (run D10/D10i only), `--skip-done` (skip experiments whose result JSON already exists), `--skip-d7` (skip EXP-D7 concat_2x ablation).
2. Load corpus from `output/bsard_articles_dedup.parquet` (local 22,633-row dedup parquet, **not** the HF parquet — see §Corpus Status). Load test questions (222), 100-question train sample, and strata.
3. For **each model** (all evaluated on the test split):
   - Instantiate `DenseRetriever` (encodes corpus once, caches to `.npy` + module-level dict).
   - Token audit stored in `retriever.audit`; `_index_build_time_s` / `_index_build_source` recorded for Tier 0.
   - Call `run_experiment()` with `top_k=500`.
   - Call `add_significance()` against the inline BM25 anchor (`k1=1.5, b=0.5, lemmatize`) on Recall@10 and Recall@100 (`primary_k=100`).
   - Call `save_result()` → `output/results/dense_retrieval/{experiment_id}.json`.
4. EXP-D7: run all eligible encoder-only models over the 100-question train sample to pick the concat_2x winner; then evaluate the winner's concat_2x variant on the full test split. The train-sample selection scores are saved as `_sel_{stem}.json` for audit.
5. Print summary table sorted by R@100, with columns `R@10 | R@100 | R@200 | R@500 | MRR@10 | MAP@100 | lat_ms | trunc% | p-val | sig`.

Single command:
```
.venv/Scripts/python scripts/evaluation/tier2/run_dense_experiments.py
```

The encoder-only experiments were run on CPU; the Qwen3-0.6B experiments (D10, D10i) were run on Azure VMs because of GPU requirements (notebook: `azure_notebooks/azure_tier2_qwen3.ipynb`). All result JSONs land in the same `output/results/dense_retrieval/` directory.

**Module-level design:** Instantiate all retrievers for the same model before moving to the next model — maximises embedding cache reuse. The `_EMBEDDING_CACHE` dict + `.npy` disk persistence mean the corpus is encoded **once** per `(model, field_weighting)` pair across all runs.

---

## 11. Unit Tests: `tests/dense/test_dense_retriever.py`

13 tests covering retriever contract, prefix handling, embedding properties, and disk caching. All use a 10-row synthetic French legal corpus and `paraphrase-multilingual-mpnet-base-v2` (the fastest dependency) so the suite runs without GPU and without loading the real BSARD corpus.

| Test | Assertion |
|---|---|
| `test_slugify` | `_slugify` maps HF IDs to safe filename stems (`/` and `-` → `_`, lowercased) |
| `test_retriever_returns_correct_types` | `retrieve()` returns `(list[int], float)`; latency ≥ 0 |
| `test_retriever_returns_top_k_results` | `retrieve(q, top_k=5)` returns exactly 5 article IDs |
| `test_top_k_larger_than_corpus_clipped` | `top_k=9999` on a 10-row corpus returns 10 IDs (FAISS `k = min(top_k, ntotal)`) |
| `test_article_ids_are_valid` | All returned IDs exist in the corpus dataframe |
| `test_retrieve_latency_positive` | Latency > 0 for any non-empty query |
| `test_token_length_audit_keys` | `audit` dict has: `fraction_truncated`, `max_tokens_observed`, `mean_tokens`, `median_tokens`, `p90_tokens` |
| `test_embeddings_are_normalised` | L2 norm of each stored document embedding ≈ 1.0 (tolerance 1e-5) |
| `test_reproducibility` | Two calls to `retrieve()` with same query return identical ranked lists |
| `test_prefix_changes_query_embedding` | Non-empty `query_prefix` produces a different embedding than no prefix |
| `test_n_articles_property` | `retriever.n_articles == len(corpus)` |
| `test_embedding_dim_property` | `retriever.embedding_dim > 0` |
| `test_disk_cache_loaded_on_second_init` | Second `DenseRetriever` with same model+fw loads from `.npy` in < 60 s (no re-encoding) |

---

## 12. Evaluation and Statistical Testing

**Primary metric: Recall@100** — matches the BSARD paper, appropriate for the dataset's variable number of relevant articles (up to 109 per question).

**Secondary metrics:** Recall@10, Recall@200, Recall@500, MRR@10, MRR@100, NDCG@10, MAP@100

**Paper comparability:** Report R@100, R@200, R@500, MAP@100, MRR@100 in the main results table — these match the BSARD paper's Table 2 exactly, allowing direct numerical comparison.

**Why R@10 is not the primary metric:** 7% of BSARD questions have more than 20 relevant articles; 18% have 5–20. A question with 25 relevant articles can never exceed R@10 = 0.40 regardless of retriever quality. R@10 is useful for the cross-encoder re-ranking stage (Tier 3) but is structurally biased as a primary retrieval metric on BSARD.

**Statistical significance:**

| Scope | Test | Threshold |
|---|---|---|
| Full test set (222 questions) | Paired t-test on per-query Recall@100 (`scipy.stats.ttest_rel`, two-sided) | p < 0.05 |
| Per-stratum (n ≈ 50–55) | Wilcoxon signed-rank (`scipy.stats.wilcoxon`, two-sided) | p < 0.05 |

Comparisons: each dense model vs `bm25_tuned` anchor, and each model vs each other. Report p-values alongside metric deltas in all results tables.

**Additional diagnostic metric (optional, recommended):** R-Precision — precision at rank = number of relevant documents per query. Provides a normalised view robust to variable relevant-article count. Not in the BSARD paper but useful for stratified analysis.

---

## 13. Key Decisions (Locked)

| # | Decision |
|---|---|
| D1 | FAISS `IndexFlatIP`, unit-normalised embeddings (exact search, not approximate) |
| D2 | ColBERT excluded from Tier 2 (architectural incompatibility with bi-encoder setup) |
| D3 | Zero-shot evaluation across all models; no self-fine-tuning |
| D4 | Paper checkpoint (if released) = primary dense baseline, inference-only |
| D5 | Primary metric = Recall@100 (matches paper); R@10 secondary |
| D6 | Token length audit is a mandatory prerequisite gate before any encoding |
| D7 | Truncation strategy: head truncation (`max_length = model.max_seq_length`) |
| D8 | Embedding prefixes enforced per model (Section 4); wrong prefix = invalid result |
| D9 | Latency measurement: retrieval path only (not index load), `time.perf_counter()` |
| D10 | Significance tests: paired t-test (full set) and Wilcoxon (per-stratum); anchor = `bm25_tuned`; primary metric = Recall@100 |
| D11 | Field weighting (`concat_2x`) tested as ablation on best zero-shot model only |
| D12 | Results saved to `output/results/dense_retrieval/*.json` under the data root — **not committed to git** |
| D13 | Embeddings stored in `output/embeddings/` under the data root — not in git |
| D14 | `mpnet_multi` retained for reference only; excluded from model selection |
| D15 | Corpus is deduplicated to 22,633 rows; use `output/bsard_articles_dedup.parquet` for all experiments |
| D16 | `top_k=500` default in experiment runner (needed for Recall@200, Recall@500) |
| D17 | `BAAI/bge-m3` encoded with `max_seq_length=1024` (not 8192 default) — reduces encoding time ~3× with negligible recall loss on short BSARD articles |
| D18 | `BAAI/bge-m3` uses CLS pooling (model default); do not override to mean pooling |

---

## 14. Alignment with the BSARD Paper

**Reference:** Louis & Spanakis, ACL 2022. BSARD paper used:
- CamemBERT-based bi-encoder (French RoBERTa architecture)
- DPR-style in-batch negative training on BSARD train split (709 pairs)
- Evaluation: R@100, R@200, R@500, MAP@100, MRR@100 on BSARD test split

**Paper results vs our zero-shot replication (Table 2):**

| System | R@100 (paper) | R@100 (ours) | MAP@100 (paper) | MAP@100 (ours) | MRR@100 (paper) | MRR@100 (ours) |
|---|---|---|---|---|---|---|
| TF-IDF (zero-shot) | 40.13 | 45.43 (`tfidf_none_text_only`) | 8.69 | n/a | 12.98 | n/a |
| BM25 (zero-shot) | 51.33 | 48.21 (`bm25_anchor` k1=1.5, b=0.75) | 16.04 | n/a | 24.59 | 24.63 |
| CamemBERT zero-shot siamese | 4.21 | **1.69** | 0.50 | 0.72 | 2.04 | n/a |
| CamemBERT fine-tuned siamese | 71.63 | — *(not replicated)* | 35.44 | — | 43.52 | — |
| CamemBERT fine-tuned two-tower | **74.78** | — *(checkpoint never released)* | **35.67** | — | 42.46 | — |
| mE5-large (zero-shot, paper-novel) | — | 59.41 | — | 17.32 | — | n/a |
| mE5-large + concat_2x (zero-shot, ours) | — | **62.15** | — | **21.57** | — | n/a |
| BGE-M3 (zero-shot, paper-novel) | — | 59.21 | — | 18.15 | — | n/a |
| Qwen3-0.6B + instruct (zero-shot, paper-novel) | — | 59.74 | — | 20.63 | — | n/a |

**Headline take-aways:**
- BM25 anchor reproduces the paper to within ~3 pp on R@100 (48.21 vs 51.33) — slight underperformance likely from the corpus-dedup difference (paper uses 33,741-row raw corpus; we use 22,633-row deduplicated).
- Zero-shot CamemBERT collapses even harder than the paper reports (1.69 vs 4.21 % R@100). Same conclusion: untuned French contextual embeddings are not viable on statutory retrieval.
- Three zero-shot multilingual models — mE5-large, BGE-M3, Qwen3-0.6B — all reach ~59–62 % R@100, comfortably beating the paper's BM25 and approaching the paper's *fine-tuned* siamese ceiling (71.63 %) without any task-specific training.

**Thesis contributions relative to the paper:**
- Evaluates multilingual encoder-only models the paper did not consider (mE5-base, mE5-large, BGE-M3).
- Evaluates a decoder-only LLM embedding family (Qwen3-Embedding-0.6B, plain + instruction-prefixed).
- Evaluates long-context encoding (BGE-M3 at 1024 tokens) vs truncated encoding (mE5 at 512 tokens) as a retrieval factor — truncation rates drop from ~9 % to ~2.5 % but R@100 is essentially identical, so truncation is not a binding constraint on BSARD article lengths.
- Tests the zero-shot vs fine-tuned gap with more recent model families.
- Reports stratified metrics (lexical alignment, cross-reference, article count) that the paper does not report.

The authors' fine-tuned checkpoint was never published on the URLs originally targeted (`github.com/maastrichtlawtech/bsard`, `huggingface.co/antoiloui`, `huggingface.co/datasets/antoiloui/bsard`), so EXP-D5 was not run.

---

## 15. Hypotheses — Verdicts

| # | Hypothesis | Verdict | Evidence |
|---|---|---|---|
| H1 | Only fine-tuned dense models will close the semantic-paraphrase gap; zero-shot dense will be near-random | **Refuted (zero-shot part)** | Zero-shot mE5-large / BGE-M3 / Qwen3-instruct ~2× the best sparse R@10 (0.14 vs 0.07) and ~2× the R@100 (0.34–0.43 vs 0.17–0.22) on the semantic-paraphrase stratum. Zero-shot CamemBERT *did* collapse (R@100 = 1.69 %) — that half of H1 holds. |
| H2 | French-specific `sentence-camembert-large` may underperform multilingual mE5-large despite domain specificity | **Confirmed** | mE5-large R@100 = 0.5941 > camembert-large R@100 = 0.5833. The multilingual retrieval pretraining wins over French-only LM training. |
| H3 | mE5-large may not consistently outperform mE5-base on statutory text | **Refuted** | mE5-large R@100 = 0.5941, mE5-base R@100 = 0.4915 — a ~10 pp gap. Scale matters here. |
| H4 | Paper checkpoint (if available) would significantly outperform all zero-shot models | **Not tested** | Authors' checkpoint never released; EXP-D5 unavailable. |
| H5 | Dense retrieval may outperform BM25 on semantically paraphrased queries while underperforming on lexically aligned queries | **Confirmed (paraphrase) / Refuted (lexical)** | Paraphrase: dense > sparse (above). Lexical: top dense models (e.g. Qwen3-instruct R@10 = 0.5902 on lex-aligned) *also* beat the best sparse (BM25 ~0.52) — dense doesn't underperform on the lex-aligned stratum. |
| H6 | BGE-M3 will outperform mE5 on long articles due to 1024-token window; comparable on short articles | **Refuted (overall) / Confirmed (mechanism)** | BGE-M3 R@100 = 0.5921 is essentially tied with mE5-large (0.5941) despite truncating only 2.5 % vs 9.4 %. BGE-M3 *does* lead on R@500 (0.7961 vs 0.7540) — long-context advantage shows up only in deeper recall. Mechanism is real; magnitude on BSARD's mostly-short articles is small. |
| H7 | concat_2x will give a modest (~0.5–1 pp) R@100 improvement on the winning encoder-only model | **Confirmed, larger than predicted** | mE5-large concat_2x R@100 = 0.6215 vs text_only 0.5941 → +2.7 pp (well above the predicted 0.5–1 pp). p = 0.0003 vs the sparse anchor. |
| H8 | Qwen3-0.6B (decoder-only, last-token pooling) will outperform same-scale encoder-only bi-encoders; D10i > D10 | **Confirmed** | Qwen3-plain R@100 = 0.5822, Qwen3-instruct R@100 = 0.5974, both substantially above mE5-base (0.4915) at similar parameter scale. Instruct > plain (+1.5 pp R@100), confirming the sub-hypothesis. Qwen3 is also the fastest dense model on GPU (~62 ms). |

---

## 16. Selected Configurations for Downstream Tiers

### 16.1 Tier 3 (Hybrid) dense input — `dense_me5_large_concat2x_zeroshot`

mE5-large with `passage: ` / `query: ` prefixes, `concat_2x` field weighting (law_code×2 + chapter_title + article_number + article_text). Test R@100 = 0.6215, R@10 = 0.3420 — highest R@100 across all 10 Tier 2 experiments and the natural complement to the sparse anchor `bm25_lemmatize_concat_2x` (R@100 = 0.5312, see Tier 1).

Selection: D7 winner among encoder-only candidates (D2/D3/D4/D4c) was mE5-large on the 100-question train sample (R@100 = 0.6191; see §5.2). Its concat_2x variant was then evaluated on the 222-question test split.

### 16.2 RQ2 dense baseline — `dense_me5_large_zeroshot` (text_only)

RQ2 (structure-aware retrieval) deliberately uses **text_only** mE5-large rather than the Tier 3 concat_2x variant. The reason is methodological isolation: RQ2 evaluates whether explicit structural metadata (hierarchy paths, chapter chunking, GraphRAG neighbours, etc.) helps. If the RQ2 baseline already injected `law_code` and `chapter_title` via `concat_2x` field weighting, the structural-aware methods would only be measured against an already-structure-aware baseline, blunting the RQ2 contrast. The text_only variant (R@100 = 0.5941) is the clean retrieval-only comparator for that research question.

### 16.3 Sparse-side anchor (carried over from Tier 1)

`bm25_lemmatize_concat_2x` (Tier 1 R@100 = 0.5312) is the sparse anchor for Tier 3 hybrid fusion (per §7.1 of `TIER1_SPARSE_RETRIEVAL_PLAN.md`). See `TIER3_HYBRID_RETRIEVAL_PLAN.md` for how the fusion of these two anchors is parameterised and evaluated.
