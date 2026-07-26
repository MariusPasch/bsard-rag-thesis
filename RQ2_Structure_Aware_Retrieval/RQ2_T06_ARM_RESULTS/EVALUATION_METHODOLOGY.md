# Evaluation Methodology — Designed (T07) vs Actually Used (per Arm)

> Companion to [SOURCE_MAP.md](SOURCE_MAP.md). Establishes how retrieval evaluation
> actually worked for each experiment, so T06 can reconcile numbers honestly.
> **Headline finding at the bottom — read §4 before trusting any cross-arm table.**

---

## 1. What T07 DESIGNED (the full framework)

T07 (`src/evaluation/`) is a comprehensive hub that delegates the tiered metrics to the
RQ3 `bsard_evaluation` package and adds RQ2-specific layers. Public API:
`from evaluation import evaluate, evaluate_partial_views` → returns an `EvalReport`.

**Metric tiers (delegated to `RQ3_Autonomous_Evaluation/bsard_evaluation/`, via `comparator.evaluate`):**
- **T0/** efficiency — latency mean/p50/p90/p95/p99, throughput QPS, timing breakdown.
- **T1/** BSARD-paper — Recall@{1,5,10,100}, MRR@100. (`tier1_bsard.py`)
- **T2/** supervised IR — P1 (Precision/Recall/F1/HitRate@k), P2 (MRR/MAP/nDCG@k), P3 graded (RA-nWG, N-Recall4+). (`tier2_supervised.py`)
- **T3/** autonomous LLM-judge — RAGAS (faithfulness, answer_relevancy) + G-Eval (completeness, factual_accuracy, coherence). Implemented in `autonomous_eval.py`, not stubbed.

**RQ2-specific layers (stay in T07):**
- **`adapter.py`** — converts `RetrievalResult` → TREC qrels/run. bsard-id resolution priority:
  `metadata.bsard_ids` (chunks, T03) → `metadata.bsard_id` (articles, T04/T05) → caller mapping → raw id.
  Arm1 chunk→article aggregation = **max chunk score** per bsard_id.
- **`metrics.py`** — `recall_at_k`, `mrr_at_k`, `ndcg_at_k` (binary gain, log2(i+2) discount),
  `cosine_weighted_recall_at_k` (`Cw/`), and **effective/drift-aware** `effective_*_at_k` (`E/`)
  that score against `gt_in_pdf` and return `None` (not-evaluable) when no GT reaches the PDF.
  `compute_all_binary_metrics` cutoffs = **{R@10, R@100, MRR@10, nDCG@10}**.
- **`weighted_metrics.py`** (`W/`, Arm1-only) — chunk-article partial relevance.
  Weight `w(c,B)=|tokens of B in c| / |tokens of B|`, capped at 1.0 per article.
  `score_chunks_for_question(mode="strict"|"lenient")`.
- **`weight_precomputer.py`** — precomputes/caches `chunk_bsard_weights.csv` per `(doc_id, arm1_config_hash)`.
- **`projection.py`** — `ProjectionRow{gt_bsard_ids, gt_in_pdf, gt_missing_from_pdf, recall_ceiling}` (Layer-3 effective GT).
- **`comparator.py`** — significance (paired t + Wilcoxon) on `{T1/R@10, T1/R@100, W/recall@10, W/recall@100}`;
  stratified by single/multi-article and extraction-status buckets; `evaluate_partial_views` → strict/lenient/delta.
- **`EvalReport.to_dict()`** = `{results, significance, stratified, costs, autonomous}`.

**Ground truth (canonical):** `ground_truth/{bsard_test,bsard_train}.json` = `{qid: [bsard_id,...]}`;
per-run curated subsets in `ground_truth/runs/<run>.json`; extraction-status (FOUND/PARTIAL/NOT_FOUND
+ `extraction_cosine`) in `ground_truth/question_extraction_status/questions_by_extraction_status.jsonl`.

> **Key fact:** Almost none of this full framework was used end-to-end by the arms. Each arm
> consumed only the pieces it needed, three different ways. See §2–§3.

---

## 2. What each arm ACTUALLY used

| | **T03 ARM1 Naive** | **T04 ARM2 Metadata** | **T05 ARM2 PageIndex** |
|---|---|---|---|
| **Mechanism** | **Inline** in notebooks. No T07 metric import. | **Imports T07 `metrics.py`** funcs (not the full comparator). | **Inline** reimplementation of T07 formulas. No T07 import. |
| **Where** | `notebook_runs/<doc>.ipynb` cell `406e0cf7` → `summary_overall.csv`; hand-rolled into `_per_pdf_status_table.csv` | `scripts/compare_t03_vs_t04.py` (+ `boost_ablation.py`); `aggregate_comparison.py` only reshapes | `scripts/compare_arms_1804.py`, `experiment_local_postprocess.py`, `notebooks/_build_local_eval_notebook.py` (3 copies of the formulas) |
| **Imports** | T07 `evaluation.cache` only (paths), **not metrics** | `from evaluation.metrics import recall_at_k, mrr_at_k, ndcg_at_k, cosine_weighted_recall_at_k` | none — verbatim re-impl |
| **Metrics reported** | R@10, R@100, MRR (first-rank) | R@10, R@100, MRR@10, nDCG@10, (opt) Cw/R@10, Cw/R@100, latency_ms | R@1,5,10,20,100, MRR@10, NDCG@10, cost |
| **K set** | table reports {10,100}; nb computes {1,5,10,20,50,100} | {10,100} | {1,5,10,20,100} |
| **Granularity / aggregation to article** | chunk pool; bsard_ids deduped in rank order; binary | one bsard_id per ranked row, dedup keep-first | `metadata.bsard_id` per ranked item, dedup |
| **Fusion at retrieval** | RRF dense+BM25, k=60, **top-200 pool** | RRF dense+BM25 (per variant), **top-100** | none (LLM nav), **pad_to_k=100** |
| **Per-query agg** | mean over questions | arithmetic mean | `np.mean` |
| **Weighted (`W/`) metrics** | **No** (notebook caveat flags it as missing; weight_coverage mostly missing) | No (uses binary + optional cosine `Cw/`) | No |
| **Effective/drift (`E/`)** | Not as `E/`; instead uses **lenient GT projection** + recall_ceiling | **GT clipped to reachable bsard_ids** (≈ effective recall by construction) | **No** — full GT used directly |
| **Significance / stratified / T0 / T3** | none | none | none |

---

## 3. Ground-truth definition differs per arm (the subtle trap)

All three ultimately derive from BSARD GT, but the **set each scores against is different**:

- **T03** — per-PDF `data/<doc>/question_relevance.json` projection; reports the **lenient** GT
  (`verification_status ∈ {FOUND, PARTIAL}`). "lenient" in the column name = this GT choice;
  "fused" = RRF ranking. Strict (FOUND-only) also available but not in the headline table.
- **T04** — loads `bsard_test.json` + `bsard_train.json`, then **clips each question's GT to the
  set of bsard_ids reachable in that PDF's T04 index** (`build_subset`, compare_t03_vs_t04.py:160-178),
  and only keeps questions with non-empty intersection. This is "recoverable recall" — it removes the
  out-of-corpus floor, effectively an effective-GT measure.
- **T05** — loads `ground_truth/runs/t04_<stem>.json` (the curated per-run files) and uses them
  **as-is, full GT, no clipping**.

Consequence: even at the same K, T03/T04/T05 headline numbers are computed against different
denominators (lenient projection vs reachable-clipped vs full-run GT) and different pool depths.

---

## 4. HEADLINE FINDING — comparability & what T06 should do

**The arms' own published CSVs are NOT apples-to-apples.** They differ in GT definition (§3),
pool depth (200/100/100), K sets, and evaluation code path (inline×2 vs T07-funcs×1).

**The one internally-consistent cross-arm artifact already exists:** T05's
`compare_arms_1804.py` (output `<stem>/experiments/cross_arm_comparison.csv`, combined into
`cross_pdf_cross_arm_long.csv`) **recomputes all three arms from their raw per-query result files
with a single inline method** — binary R@K, same `t04_<stem>.json` GT, same dedup. So within that
CSV, T03 vs T04 vs T05 ARE comparable (caveat: pool depths still differ, so R@100 favours T03's 200-pool;
and it omits MRR/nDCG for the cross-arm rows). This is the best existing starting point for T06's tables.

**Recommended path for T06 (to produce a defensible cross-arm comparison):**
1. **Re-evaluate all three arms through one path** rather than stitching the three native CSVs.
   Two options:
   - **(a) Reuse T05's inline recompute approach** (already reads all three raw result formats) — fastest,
     binary metrics only. Extend it to also emit MRR@10/nDCG@10 for all arms.
   - **(b) Route all three through `evaluation.comparator.evaluate`** (the designed T07 path) — gives
     T0/T1/T2 + `W/` + `E/` + significance + stratified in one `EvalReport`, the most rigorous and
     thesis-defensible. Requires feeding each arm's `list[RetrievalResult]` + one GT choice + config.
     T06 already lists `loaders.py` for exactly this.
2. **Fix one GT definition** for the comparison and state it (recommend effective/drift-aware `E/`,
   since doc-2 drift is the known confound and T04 already clips this way). Report lenient vs strict as a sensitivity view.
3. **Normalise pool depth** or footnote it — truncate all to a common K (e.g. 100) before R@100, or
   clearly label T03 as 200-pool.
4. **Carry cost (T0) explicitly** — T05's dominance story is recall-vs-cost; pull `cost{llm_calls,
   tokens, latency_ms}` from T05 result JSONs and latency_ms from T04 CSVs.

**Verified equivalence note:** T03's and T05's inline `recall_at_k`/`mrr_at_k`/`ndcg_at_k` are
mathematically identical to T07's `metrics.py` definitions (binary gain, log2 discount, 1/(rank) MRR).
So the *formulas* agree across all three; only the *inputs* (GT set, pool, K) diverge. That means a
single re-evaluation through one path will reproduce each arm's intent without changing metric semantics.

---

## 5. Exact code references

- **T07 framework:** `RQ2_T07_EVALUATION/src/evaluation/{metrics,weighted_metrics,adapter,comparator,models,projection,weight_precomputer,autonomous_eval}.py`; RQ3 tiers at `RQ3_Autonomous_Evaluation/bsard_evaluation/{harness,tier0_efficiency,tier1_bsard,tier2_supervised,config}.py`.
- **T03 metrics:** `RQ2_T03_ARM1_NAIVE/notebook_runs/<doc>.ipynb` cell `406e0cf7` (`metrics_for_ranking`); outputs `data/<doc>/analysis/summary_overall.csv` → `notebook_runs/_per_pdf_status_table.csv`.
- **T04 metrics:** `RQ2_T04_ARM2_METADATA/scripts/compare_t03_vs_t04.py` (imports L59-74; GT clip L160-178; per-query L251-293; `_bsard_ranking_from_result` L226-248); `boost_ablation.py:52`; `aggregate_comparison.py` (reshape only).
- **T05 metrics:** `RQ2_T05_ARM2_PAGEINDEX/scripts/compare_arms_1804.py` (`recall_at_k`/`score_arm` L26-49, three-arm recompute L65-136), `experiment_local_postprocess.py` (L48,205-218), `notebooks/_build_local_eval_notebook.py` (L190-211).
- **Sibling install trick:** T04/T05 add `RQ2_T0*/src` to `sys.path` at runtime; T07 `evaluation` is installed via `pip install -e ../RQ2_T07_EVALUATION` (not pinned in requirements.txt).
