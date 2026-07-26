# T06 Source Map — Where to Find Arm Results, Parameters & Prior Analysis

> Working notes for **RQ2_T06_ARM_RESULTS**. Objective: consolidate and compare the
> retrieval results of **T03 (ARM1 Naive)**, **T04 (ARM2 Metadata)** and **T05 (ARM2 PageIndex)**
> on the curated 5-PDF set, plus the **T07** metric layer that all three are scored through.
> All paths are relative to the shared data root (see below).

## 0. Orientation

| | T03 ARM1 Naive | T04 ARM2 Metadata | T05 ARM2 PageIndex | T07 Evaluation |
|---|---|---|---|---|
| **Approach** | Flat chunking → FAISS+BM25→RRF | AzureDI node/article + enrichment + boost; FAISS+BM25→RRF | Vectorless LLM navigation over Law→Chapter→Article tree | Metric hub (R@K, MRR, nDCG, weighted/effective) |
| **Embedding** | `intfloat/multilingual-e5-large-instruct` (1024d) | same | none (LLaMA 3.1 8B via Ollama, num_ctx 16384, temp 0) | n/a |
| **Headline result** | R@10 ~0.6, R@100 ~0.94 (doc 1804) | best variant `node_summary` beats T03 on R@10 (4/5), MRR/nDCG (5/5) | R@10 ~0.17; strictly dominated by T04 on recall AND cost | — |

**Curated 5-PDF set** (doc 2 / `2004A27101` excluded for GT drift):

| PDF stem | AzureDI doc_id | Code |
|---|---|---|
| `1804_03_21_1804032150` | 9 | Code Civil |
| `1867_06_08_1867060850` | 6 | Code Pénal |
| `1967_10_10_1967101055` | 7 | Code Judiciaire (larger) |
| `1967_10_10_1967101056` | 5 | Code Judiciaire (smaller) |
| `2003_07_17_2013A31614` | 8 | Code du Logement |

Registry: `RQ2_T00_ORCHESTRATOR/data/selected_pdfs.json`.

**Storage convention:** each project's local `data/` is wired (via
`scripts/setup/link_data.py`) to `<data root>/<PROJECT>/`, where the data root
(env `RQ2_DATA_DIR`, default `<repo>/data`) is populated from the companion
Hugging Face dataset `mpaschalidis/bsard-rag-thesis-data` (subset `rq2`). Git
tracks source + docs; the data root holds indices/results/large artefacts. T03
and T07 share the same `pdf_cache/` backing. T06's own `data/` wires to
`<data root>/RQ2_T06_ARM_RESULTS/`.

---

## 1. T03 — ARM1 Naive

Repo: `RQ2_T03_ARM1_NAIVE`  •  Data backing: `<data root>/pdf_cache/` (shared with T07)

### Results files
- **Per-PDF summary CSV (local):** `RQ2_T03_ARM1_NAIVE/notebook_runs/_per_pdf_status_table.csv`
  — doc_id, n_relevant_q, n_chunks, n_unique_bsard, pool_frac, weight_coverage, fused/lenient R@10/R@100/MRR.
- **Per-question retrieval pools (data root):**
  `pdf_cache/<doc_id>/results/<config_hash>/retrieval_pool/<qset_hash>.jsonl`
  (one row/question: candidates[] with rank, chunk_id, fused/dense/sparse scores, `bsard_ids`, char spans)
  + sibling `.manifest.json` (config_hash, qset_hash, qset_size, top_k=200, rrf_k=60, timestamp).
- **Chunk text:** `pdf_cache/<doc_id>/configs/<config_hash>/chunks.json`.
- **T08 bundles:** `pdf_cache/<doc_id>/t08_bundles/<qset_hash>/q<qid>.json`.
- **Analysis notebooks (local):** `RQ2_T03_ARM1_NAIVE/notebook_runs/<doc_id>.ipynb` (5) + root `analyze_retrieval_1804032150.ipynb`.
- **Chunking inspection (local):** `RQ2_T03_ARM1_NAIVE/inspect_output/{article_spans,chunks_recursive,chunks_sliding_window}.csv`, `report.html`.

### Parameters
- Embedding/tokenizer: `intfloat/multilingual-e5-large-instruct`. Chunking: sliding window
  `window_size=512`, `stride=256` (alt: recursive, `max_tokens=512`). RRF `k=60`. top_k 100–200.
- Recorded per run in `pdf_cache/<doc_id>/configs/<config_hash>/manifest.json` (config_hash =
  first 12 hex of sha256 over {pdf_sha256, tokenizer, embedding_model, chunking}).
- Defaults live in `scripts/precompute_pdf.py` / `precompute_retrieval.py` argparse.

### Entry points (`scripts/`)
`precompute_pdf.py` (extract→chunk→embed→FAISS+BM25), `precompute_retrieval.py` (gen top-K pools),
`query_cached.py`, `build_t08_bundle.py`, `compute_drift_status.py`, `apply_drift_filter.py`.

### Docs
`README.md`, `PROJECT_CONTEXT.md`, `VERIFICATION_CHECKPOINT.md`, `CHANGE_NOTES.md`. PROJECT_CONTEXT notes the doc-2 GT drift caveat.

---

## 2. T04 — ARM2 Metadata

Repo: `RQ2_T04_ARM2_METADATA`  •  Data backing: `<data root>/RQ2_T04_ARM2_METADATA/`

### Results files
- **Aggregate 5-stem CSV (data root):** `comparison_t03_vs_t04.csv`
  (cols: method, R@10, R@100, MRR@10, nDCG@10, Cw/R@10, Cw/R@100, latency_ms, n_queries; T03 + 6 T04 variants).
- **Aggregate per-query JSON:** `comparison_per_query.json` (query_id, query_text, ground_truth[], per_method{...}, cosine_ground_truth on doc 9 only).
- **Per-stem CSV/JSON:** `comparison_t03_vs_t04_<doc_id>.csv` + `comparison_per_query_<doc_id>.json` (5 each).
- **Raw per-query runs:** `<doc_id>/results/<config_hash>/compare_<timestamp>.jsonl` (RetrievalResult, ranked_items[] w/ metadata+scores+bsard_ids).
- **Indices/config:** `<doc_id>/configs/<config_hash>/{manifest.json, enrichment_stats.json, faiss.index, faiss_meta.json, bm25.pkl}`.

### Prior analysis (local `analysis/`, v4 sweep 2026-05-27, `LINKER_VERSION=4`)
- `comparison_T03_vs_T04_v4.md` — thesis-ready tables + interpretation.
- `comparison_summary_v4.md` + `comparison_summary_v4.csv` (wide) + `comparison_all_stems_v4.csv` (long: stem,method,metric,value — best for T06 pivots).
- `error_analysis_doc6_v4.md` — doc 6 R@10 deficit (multi-GT siblings rank 11–100).
- `boost_ablation_v4.md` — boost stage inert under default regex-only signal extraction.
- `v3_vs_v4_doc8_retro.md` — linker fix gave doc 8 R@10 +0.55–0.65.

### Parameters
- 6 variants × unit {node, article}: `raw, enriched, summary (node-only), filtered, full, terms`
  (defined in `src/arm2_metadata/enricher.py`). Embedding same e5-large-instruct (1024d).
  `max_tokens=512`, `retrieval_top_k=100`, `rerank_top_k=10`, RRF (reuses T03's).
  Boost multipliers `term_match=1.3, used_in=1.5, jurisdiction=1.1`, `drop_non_effective=true`.
  `LINKER_VERSION=4` (`retriever.py:63`). Persisted in `<doc_id>/configs/<hash>/manifest.json`.
- Run plans in `scripts/compare_t03_vs_t04.py`: full (11 variants) vs `--smoke` (5).

### Entry points (`scripts/`)
`precompute_t04_indices.py` (build indices per unit/variant; GPU), `compare_t03_vs_t04.py`
(eval driver → per-stem/per-query CSVs), `aggregate_comparison.py` (→ `analysis/`),
`boost_ablation.py`, `check_precompute_requirements.py`, `prepare_azure_bundle.py`.
Per-stem cache-hit smoke uses `precompute_t04_indices.py --doc-id <stem>`.

### Docs
`README.md`, `PROJECT_CONTEXT.md` (§9 = v4 results), `CHANGE_NOTES.md` (2026-05-27 v4 entry), `notebooks/README.md` (Azure precompute workflow + timings).

---

## 3. T05 — ARM2 PageIndex

Repo: `RQ2_T05_ARM2_PAGEINDEX`  •  Data backing: `<data root>/RQ2_T05_ARM2_PAGEINDEX/`

### Results files (all under the data root)
- **Per-query JSON:** `<PDF_STEM>/results/q<qid>.json` (ranked_items[] w/ metadata.bsard_id, method `2B-pageindex`, `cost{llm_calls,tokens,latency_ms}`, `trace[]` navigation log — unique T05 asset).
- **Snapshots:** `results/` (current best = post-fix + chapter+law padding), `results_baseline_post_fix_2026_05_29/`, `results_pre_fix_2026_05_29/`.
- **Cross-PDF long CSV:** `cross_pdf_cross_arm_long.csv` (R@1–R@100, MRR, NDCG across 5 PDFs × arms).
- **Per-PDF cross-arm CSV:** `<PDF_STEM>/experiments/cross_arm_comparison.csv` (T03+T04+T05).
- **Post-process ablations:** `<PDF_STEM>/experiments/local_postprocess_experiments.csv` (14 variants: tie-break/padding/backfill/RRF).
- **Tree cache:** `<PDF_STEM>/tree.json`.

### Prior analysis
- **`CROSS_PDF_SYNTHESIS.md`** (data root) — the key doc. Post-mortem of 4 bugs (Ollama num_ctx default 4096 truncation; hard-coded example IDs copied by 8B model; list[dict] JSON dropped; over-conservative article prompt) and before→after R@10 (e.g. 1804 0.000→0.203, 2003 0.029→0.323). Cross-arm: T05 R@10≈0.174, R@100≈0.408, 21–50% of T04 per-PDF, no PDF where T05 beats T04 on R@10; niche R@20 win on 1867. Cost: 5–6.3 LLM calls/query, ~30s p50, ~7.5h T4 wall — strictly dominated by T04.
- **Per-PDF `<PDF_STEM>/experiments/REPORT.md`** — success rates, per-variant R@K, cost, single-vs-multi-article stratification.

### Parameters (`src/arm2_pageindex/navigator.py:NavigatorConfig`)
`max_laws=3, max_chapters_per_law=5, max_articles_per_chapter=6, max_iterations=3, pad_to_k=100,
score_threshold=1, skip_law_selection_if_single_doc=True`. LLM = LLaMA 3.1 8B (Ollama, num_ctx 16384, temp 0). tree_builder v1 (pinned in cache hash). No YAML config — params in code + CLI args.

### Entry points (`scripts/`)
`build_tree.py`, `prepare_azure_bundle.py`, `upload_to_blob.py`, `rederive_padding.py` (in-place re-pad),
`experiment_local_postprocess.py` (14-variant ablation), `compare_arms_1804.py`, `combine_cross_arm_csvs.py`,
`smoke_navigator.py`, `smoke_pipeline.py` (bsard_id non-null check). Notebooks: `azure_t05_pageindex_run.ipynb`, `local_t05_eval_and_compare.ipynb`.

### Docs
`README.md`, `PROJECT_CONTEXT.md`, `RQ2_CONTEXT_FOR_T05.md`, `CHANGE_NOTES.md`; in the data root `CROSS_PDF_SYNTHESIS.md`, `INITIAL_FAILURE_AND_FIXES.md`.

---

## 4. T07 — Evaluation (how every arm is scored)

Repo: `RQ2_T07_EVALUATION`  •  Data backing: shares `pdf_cache/` with T03.

### Ground truth
- Canonical (committed): `ground_truth/{bsard_train,bsard_test}.json` — `{question_id_str: [bsard_id,...]}`.
- Per-run curated (gitignored): `ground_truth/runs/<run>.json`, e.g. `t04_1804_03_21_1804032150.json`.
  Selected via `data.ground_truth_file` (single) or `data.ground_truth_dir` (union of `*.json`).
- Stratification: `ground_truth/question_extraction_status/questions_by_extraction_status.jsonl`
  (buckets exact/partial/not_present/mixed; carries `extraction_cosine`).

### Metric code & definitions
- `src/evaluation/metrics.py`: `recall_at_k`, `mrr_at_k`, `ndcg_at_k` (binary gain), `cosine_weighted_recall_at_k`.
  Standard cutoffs via `compute_all_binary_metrics`: **R@10, R@100, MRR@10, nDCG@10**.
  **Effective (drift-aware)** variants score against per-PDF effective GT (GT ∩ bsard_ids in PDF);
  return `None` (not-evaluable) when effective GT empty — distinct from 0-recall.
- `src/evaluation/weighted_metrics.py` (Arm1 chunk partial relevance): weight
  `w(c,B)=|tokens of B in c| / |tokens of B|` capped at 1.0; `weighted_recall/precision/mrr/ndcg_at_k`.
- `src/evaluation/adapter.py`: ingests `list[RetrievalResult]`; bsard resolution priority
  `metadata.bsard_ids` (T03 chunks) → `metadata.bsard_id` (T04/T05 articles) → mapping → raw id.
  Arm1 article score = max overlapping-chunk score. Emits TREC run via `to_bsard_run()`.
- `src/evaluation/comparator.py`: `evaluate(...)` master entry; significance (paired t / Wilcoxon)
  on `T1/R@10, T1/R@100, W/recall@10, W/recall@100`; stratified by single/multi-article + extraction status. Returns `EvalReport`.

### Metric outputs
- Reports (local `analysis/`): e.g. `dense_retrieval_5pdf_<ts>.csv` + `.md`.
- Weights cache (data root): `data/<doc_id>/eval/<arm1_config_hash>/{chunk_bsard_weights.csv|.pkl, article_token_ranges.json, manifest.json}`.
- GT projections: `data/<doc_id>/question_projections/<qset_hash>.json` (gt_bsard_ids, gt_in_pdf, recall_ceiling).

### Docs
`README.md`, `PROJECT_CONTEXT.md` (binary + weighted + effective metric contract), `CHANGE_NOTES.md`.

---

## 5. Implications for T06

- **Most consumable, already-aligned source:** T04's `analysis/comparison_all_stems_v4.csv` (long format)
  and T05's `cross_pdf_cross_arm_long.csv` already contain T03+T04 and T03+T04+T05 cross-arm metrics
  respectively. These are the fastest starting point before re-deriving anything.
- **Per-query reconciliation:** join on `(doc_id, query_id, bsard_id)` using T04 `comparison_per_query_<doc_id>.json`
  and T05 `<stem>/results/q<qid>.json`; T03 candidates carry `bsard_ids` per chunk (aggregate via T07 adapter's max-score rule).
- **Metric consistency caveat:** pool depths differ (T03 top_k=200, T04=100, T05 pad_to_k=100) — not strictly
  comparable at high K; note when reporting R@100. Prefer T07's `comparator.evaluate` for apples-to-apples re-scoring.
- **Use effective/drift-aware metrics** for per-PDF ceilings; doc 2 is out of scope.
- **T06 modules:** `paths.py`, `loaders.py`, `consolidate.py`, `tables.py`,
  `figures.py`, `errors.py`. Run `python -m arm_results.consolidate` (or `scripts/run_consolidate.py`)
  to regenerate the comparable cross-arm tables. See README "Cross-arm consolidation".
