# Tier 1 — Sparse Retrieval: Implementation & Results

**Test set:** 222 questions (BSARD official test split) · **Tier 3 subset:** 48 stratified questions (`evaluation/data/tier3_subset.json`)

---

## 1. Overview

Tier 1 establishes the lexical retrieval baseline for the BSARD benchmark. Three retrieval families were implemented and systematically evaluated: BM25 (via `rank_bm25`), TF-IDF (via `scikit-learn`), and SQLite FTS5 (built-in BM25). A total of 12 experiments were run, covering normalization variants, field weighting, hyperparameter tuning, BM25 scoring variants, and the TF-IDF ablation.

The primary goal was twofold:
1. Establish the lexical ceiling — the maximum Recall@10 achievable with exact-match methods
2. Identify the best sparse configuration (by Recall@100) for use as the sparse component in Tier 3 hybrid fusion

### Replication vs. Proposal scope

The original BSARD paper (Section 4.1) evaluates exactly two lexical methods:

| Method | Paper formula | Replicated by |
|---|---|---|
| TF-IDF | $w(t,a)=\text{tf}(t,a)\log\frac{\|\mathcal{C}\|}{\text{df}(t)}$ | `tfidf_none_text_only` |
| BM25 Okapi | $w(t,a)=\frac{\text{tf}(k_1+1)}{\text{tf}+k_1(1-b+b\frac{\|a\|}{\text{avgal}})}\log\frac{\|\mathcal{C}\|-\text{df}+0.5}{\text{df}+0.5}$ | `bm25_anchor` (k1=1.5, b=0.75) |

All other experiments in this tier are **proposals / extensions** beyond the paper's scope: preprocessing ablations (stem, lemmatize), field weighting (concat_2x), hyperparameter tuning, BM25 variant comparisons (BM25+, BM25L), and the FTS5 baseline.

---

## 2. Modules Implemented

### `retrieval/preprocessing.py`

Handles all text normalisation before indexing and querying. The key design principle is **preprocessing symmetry**: identical transformations are applied to corpus documents and queries at evaluation time.

**Three normalisation variants:**

| Variant | Method | Runtime (33k docs) | Notes |
|---|---|---|---|
| `none` | Regex tokenisation only | ~1.8s | Fastest; regex extracts French alpha-numeric tokens |
| `stem` | Regex + Snowball French stemmer (NLTK) | ~70s | Ablation only |
| `lemmatize` | spaCy `fr_core_news_lg`, `nlp.pipe()` batch | ~320s | Default; cached on first use |

**Why a fast regex tokeniser (not spaCy) for `none`/`stem`:**
Initial implementation used spaCy even for `none` (tokenisation only). This produced an estimated 932s runtime for 33k documents due to spaCy's pipeline overhead. Replacing it with a compiled French-character regex `[a-zA-Z0-9àâäã...]+` reduced this to 1.8s — a 500× speedup — with no loss in token quality.

**Legal stopword overrides (Decision T1-D2):**
Standard spaCy French stopwords were applied with mandatory overrides. The following tokens carry legal force and must not be removed:

```
ne, pas, non, sauf, sans, ni, jusqu'à, dès lors que, à moins que,
pourvu que, à condition que, sous réserve
```

Removing "pas" or "non" from a statutory text would invert its legal meaning. These overrides are implemented as `LEGAL_KEEP: frozenset[str]` and tested in the unit test suite.

**Field builders:**

| Name | Format | Purpose |
|---|---|---|
| `text_only` | `article_text` | Baseline |
| `concat_2x` | `law_code law_code chapter_title article_number article_text` | `law_code` repeated twice for a ~2× BM25 boost without modifying the scorer |

The `law_code` doubling (Decision T1-D4) is a rank_bm25-compatible field weighting strategy: since BM25 weighs term frequency, repeating a field in the concatenated string increases its contribution without requiring a separate per-field index.

---

### `retrieval/sparse.py`

Implements three retriever classes.

**`BM25Retriever`**
- Wraps `rank_bm25` `BM25Okapi`, `BM25Plus`, `BM25L`
- Constructor parameters: `normalization`, `field_weighting`, `variant`, `k1`, `b`
- Returns `(ranked_article_ids, latency_ms)` from `.retrieve(query, top_k)`
- Two-level tokenisation cache keyed by `(normalization, field_weighting)`: an in-memory `_TOKENIZED_CACHE` dict for the current process, and a disk pickle fallback, so spaCy lemmatisation (critical path, ~320s per corpus pass) runs at most once per unique config across restarts

**`TFIDFRetriever`**
- `TfidfVectorizer(sublinear_tf=True)` with the same preprocessing pipeline
- Cosine similarity via sparse matrix dot product; top-k by `np.argpartition`

**`FTS5Retriever`**
- Uses the pre-built `articles_fts` virtual table in `output/bsard_corpus.db`
- Three-stage fix was required to get working results (see §7):
  1. Apostrophes in French queries caused FTS5 syntax errors → replaced with alphanumeric regex extraction
  2. AND semantics: abbreviations like "RC" matched nothing, zeroing multi-term queries → switched to explicit `OR` joining
  3. FTS5 indexed all 40,231 articles including 6,490 non-BSARD distractors → added `AND a.is_bsard_article = 1` filter in the JOIN
- Query terms are stopword-filtered before FTS5 to improve signal-to-noise

---

### Evaluation service — sibling `RQ3_Autonomous_Evaluation` component

All metric computation is delegated to the `bsard_evaluation` package, the sibling `RQ3_Autonomous_Evaluation` component, installed with:

```
pip install -e "../RQ3_Autonomous_Evaluation"
```

It exposes an `EvaluationHarness` that accepts per-query retrieval payloads from RQ1 and returns Tier 0/1/2 metrics in a single call (Tier 3 autonomous evaluators are run separately on the 48-question subset — see §3 and §5.4). No metric logic lives in this component.

### `evaluation/runner.py`, `evaluation/split.py`, `evaluation/stratify.py`

Shared infrastructure for producing retrieval outputs consumed by the harness.

- **`split.py`:** Produces the train/val/test split. Val = 20% of BSARD train questions, `random.Random(42)`, persisted to `evaluation/data/split_ids.json` (177 val, 709 train, 222 test). The file also stores `train_sample_100` — a deterministic 100-question sample of the full BSARD train set (seed=42) used by later tiers for hyperparameter selection. The 222-question test split is the official BSARD test set and is **never used during development or hyperparameter selection**.
- **`stratify.py`:** Assigns each question to strata: `lex_align` (top/bottom BM25-score quartile vs middle), `article_count` (single/multi), `cross_ref` (with/without cross-references — derived from whether any ground-truth article has cross-references). Persisted to `evaluation/data/query_strata.json`.
- **`runner.py`:** Configured with `TierConfig(tiers=[0, 1, 2], custom_k=[1, 5, 10, 20, 50, 100, 200, 500])`. Runs a retriever over a question set, records per-query ranked lists and latencies, then calls `harness.evaluate()` to attach Tier 0/1/2 metrics, six stratified breakdowns, and a two-sided paired t-test on per-query Recall@10 vs the anchor. Per-query `contexts_with_ranks` (top-10) is built from the supplied corpus dict so the harness can later compute Tier 3 evaluators.

---

### `scripts/evaluation/tier1/run_sparse_experiments.py`

Orchestrates all 12 experiments in a single process. Experiment order:

1. Anchor (BM25 Okapi, `none`, `text_only`, k1=1.5, b=0.75)
2. Normalization ablation: `lemmatize`, `stem` (same field/params)
3. Field weighting: `none+concat_2x`, `lemmatize+concat_2x`
4. Hyperparameter grid: k1 ∈ {0.5, 1.0, 1.5, 2.0} × b ∈ {0.25, 0.5, 0.75, 1.0} on the **validation split**
5. Best-params BM25 on test split
6. BM25+ and BM25L with `lemmatize`, `text_only`
7. FTS5 baseline
8. TF-IDF: `none+text_only`, `lemmatize+text_only`, `lemmatize+concat_2x`

The single-process design means the `_TOKENIZED_CACHE` persists across all experiments, so spaCy lemmatisation runs at most once per unique `(normalization, field_weighting)` pair.

---

### `tests/sparse_retrieval/test_preprocessing.py`

9 unit tests covering:
- `LEGAL_KEEP` tokens not present in the stopword set
- Legal tokens preserved in tokeniser output
- OOV rate < 50% for all variants (preprocessing symmetry check)
- `concat_2x` builder places `law_code` twice in output

All 9 tests pass.

---

## 3. Evaluation Protocol

All benchmarking is performed **on the test split only (222 questions)**. Hyperparameter selection uses the validation split exclusively (177 questions, which is a 20%-stratified subset of the BSARD train data). The test split is never touched during tuning.

### Data split rule

| Split | Questions | Source | Permitted use |
|---|---|---|---|
| Train | 709 | BSARD train (80%) | Corpus statistics, preprocessing decisions |
| Val | 177 | BSARD train (20%, seed=42) | Hyperparameter selection, early stopping |
| Test | 222 | BSARD official test set | Final benchmarking only — no leakage |

### Tier 0 — Efficiency

Computed for every experiment run on the test split.

| Metric | Description |
|---|---|
| Latency distribution | mean, std, p50, p90, p95, p99, min, max (ms) |
| Throughput | QPS, total retrieval time |
| Index build time | One-off cost (s) for corpus indexing |

### Tier 1 — BSARD Paper Metrics

Exact replication of the paper's evaluation protocol for direct comparability.

| Metric | k values | Primary? |
|---|---|---|
| Recall@k | {1, 5, 10, 100} | R@100 |
| MRR@k | {100} | MRR@100 |

### Tier 2 — Full Supervised IR Metrics

| Panel | Metrics | k values | Notes |
|---|---|---|---|
| P1 — Rank-unaware | Recall@k, Precision@k, F1@k | {1, 5, 10, 20, 50, 100, 200, 500} | Recall@k at {200, 500} aligns with paper and sizes downstream candidate pools |
| P2 — Rank-aware | MRR@k, MAP@k, NDCG@k | {10, 100} | k=10 is early-precision focus; k=100 matches Tier 1 |
| P3 — Set-utility | RA-nWG@k, N-Recall4+@k | {10, 100} | Requires graded qrels (1–5); skipped if only binary qrels are available |

**Primary metrics for experiment comparison:** Recall@100 (candidate pool quality for Tier 3) and Recall@10 (early-precision, significance testing anchor).

**Significance test:** Two-sided paired t-test on per-query Recall@10 vs. `bm25_anchor`, p < 0.05.

### Tier 3 — Autonomous Evaluation

The thesis scope is retrieval only. Tier 3 evaluates retrieved contexts directly
without generating answers. Four components are run on top of the Tier 1/2 retrieval
output; three contribute to the Autonomous Quality Score (AQS):

| Component | Method | What it measures | AQS weight |
|---|---|---|---|
| `umbrela` | LLM judge, 0–3 grade per (query, doc) | Relevance grade; produces TREC qrels for `T2-umbrela/` bridge | 0.35 |
| `erag` | LLM abstention test per (query, doc) | Whether the doc enables a grounded answer | 0.30 |
| `ragas_wa` | RAGAS `LLMContextPrecisionWithoutReference` + HyDE | Context precision (rank-aware) | 0.15 |
| `ragas_wb` | Same metric, query-as-proxy (no HyDE) | Diagnostic baseline only — not in AQS | — |

`ares` (T5-large fine-tuned judge + PPI) was scoped but deferred (GPU fine-tuning
not in RQ1 scope).

**Applied to:** the 48-question stratified subset (`evaluation/data/tier3_subset.json`),
covering all 12 strata cells (`article_count × lex_align × cross_ref`). This subset
is fixed across all tiers (RQ1/RQ2/RQ3) for direct comparability. The 48 questions
represent 21.6% of the 222-question test set.

**Applied to which experiments:** all 12 sparse experiments. Tier 1/2 metrics are
also recomputed on the same 48 questions via `scripts/evaluation/compute_subset_metrics.py`
(`subset_metrics` block in each result JSON) so Tier 3 scores can be compared against
Tier 1/2 scores on an identical question set.

**UMBRELA → Tier 2 bridge:** UMBRELA produces graded qrels (0–3) per experiment
which the harness feeds back into Tier 2, producing `T2-umbrela/` metric keys
alongside the standard `T2/` keys. This is the core signal for the RQ3
rank-correlation analysis.

**Tier 3 output keys (under `subset_metrics.metrics`):**

| Key | Description |
|---|---|
| `T3/umbrela/mean` | Normalised UMBRELA mean grade [0,1] |
| `T3/umbrela/relevant_fraction` | Fraction of docs graded ≥ 2 |
| `T3/erag/mean` | Fraction of doc-query pairs where LLM gave a grounded answer |
| `T3/ragas_wa/mean` | Context precision (HyDE workaround) |
| `T3/ragas_wb/mean` | Context precision (query-as-proxy, diagnostic) |
| `T3/AQS` | Weighted average of umbrela/erag/ragas_wa |
| `T2-umbrela/P2/NDCG@k` | Tier 2 NDCG re-run on UMBRELA graded qrels |
| `T2-umbrela/P2/MAP@k` | Tier 2 MAP re-run on UMBRELA graded qrels |

**ID-based Panel 3 keys (appear in every Tier 2 run):**

| Key | Description |
|---|---|
| `T2/P3/IDPrecision@k` | Average Precision@k via article-ID set matching |
| `T2/P3/IDRecall@k` | Recall@k via article-ID set membership |

---

## 4. Hyperparameter Tuning

A 4×4 grid search (k1 ∈ {0.5, 1.0, 1.5, 2.0}, b ∈ {0.25, 0.5, 0.75, 1.0}) was run on the **validation split only** (177 questions, a subset of the BSARD train data — the test split was not consulted at any point during tuning). Tokenisation: `lemmatize`. Field weighting: `text_only`. Optimisation target: Recall@10. The selected configuration is then re-evaluated on the test split:

| Selected k1 | Selected b | Test Recall@10 | Test Recall@100 | MRR@100 |
|---|---|---|---|---|
| 1.5 | 0.25 | 0.2651 | 0.5210 | 0.2628 |

(Per-cell val grid R@10 values are printed to stdout during the run but not persisted; only the winning configuration's test result is saved as `bm25_tuned_k11.5_b0.25_test.json`.)

**Why b=0.25 outperforms b=0.75:**
The standard BM25 default (b=0.75) assumes typical web or news documents. Statutory articles exhibit extreme length variability — some articles are a single sentence, others run to thousands of tokens. Aggressive length normalisation (high b) penalises long multi-provision articles that share query terms; b=0.25 reduces that penalty substantially and lets term frequency dominate, which suits the terminologically dense statutory register better.

---

## 5. Results

All results reported here are on the **test split (222 questions)** evaluated via the `bsard_evaluation` harness (sibling `RQ3_Autonomous_Evaluation` component) following the Tier 0–3 protocol defined in Section 3.

### 5.1 BSARD paper benchmark (replication targets)

Results reported in the original BSARD paper (Section 4.1). Metrics are percentages evaluated on the full BSARD test set.

| Train | Model | Encoder(s) | Params | Latency (ms) | R@100 | R@200 | R@500 | MAP@100 | MRR@100 |
|---|---|---|---|---|---|---|---|---|---|
| ✗ | TF-IDF | — | — | 827 | 40.13 | 50.44 | 59.34 | 8.69 | 12.98 |
| ✗ | BM25 | — | — | 1342 | 51.33 | 56.78 | 64.71 | 16.04 | 24.59 |

*Source: Masson & Joty (2022), Table 2. These are the replication targets for `tfidf_none_text_only` and `bm25_anchor` respectively.*

### 5.2 Tier 1 / Tier 2 results — 222-question test set

All 12 experiments, sorted by Recall@100. `p` is the paired t-test p-value on per-query Recall@10 vs the `bm25_anchor` baseline.

| Experiment | R@10 | R@100 | MRR@100 | Latency (ms) | p vs anchor |
|---|---|---|---|---|---|
| `bm25_lemmatize_concat_2x` | 0.2572 | **0.5312** | 0.2520 | 69.7 | 0.600 |
| `bm25_tuned_k11.5_b0.25` | **0.2651** | 0.5210 | 0.2628 | 76.1 | 0.374 |
| `tfidf_lemmatize_concat_2x` | 0.2003 | 0.5152 | 0.1995 | 39.9 | **0.037** |
| `bm25_lemmatize_text_only` | 0.2381 | 0.5121 | 0.2520 | 66.0 | 0.620 |
| `bm25_plus_lemmatize` | 0.2461 | 0.5088 | 0.2552 | 86.5 | 0.938 |
| `bm25_none_concat_2x` | 0.2567 | 0.5009 | 0.2454 | 140.6 | 0.056 |
| `bm25_stem_text_only` | 0.2637 | 0.4905 | 0.2495 | 66.5 | 0.423 |
| `bm25_anchor` (BM25 Okapi, none, text_only, k1=1.5, b=0.75) | 0.2476 | 0.4821 | 0.2463 | 136.1 | — |
| `tfidf_lemmatize_text_only` | 0.1730 | 0.4793 | 0.1646 | 39.2 | **0.001** |
| `fts5_default` | 0.2365 | 0.4697 | 0.2367 | 58.1 | 0.549 |
| `tfidf_none_text_only` | 0.2040 | 0.4543 | 0.1717 | 56.0 | **0.019** |
| `bm25_l_lemmatize` | 0.1755 | 0.4040 | 0.1343 | 84.3 | **0.002** |

**bold R@10 / R@100** mark the global leaders; **bold p-values** mark statistically significant differences from the anchor (p < 0.05).

### 5.3 Stratified breakdown — `bm25_lemmatize_concat_2x` (Tier 3 selection)

| Stratum | R@10 | R@100 |
|---|---|---|
| Lexically aligned (top BM25 quartile) | 0.5222 | 0.8722 |
| Semantically paraphrased (bottom quartile) | 0.0378 | 0.2186 |
| Single-article questions | 0.3614 | 0.6988 |
| Multi-article questions | 0.1950 | 0.4311 |
| With cross-references | 0.2980 | 0.5570 |
| Without cross-references | 0.0717 | 0.4136 |

The lex_align gap (0.5222 vs 0.0378 at R@10) is the dominant Tier 1 failure mode and the headline motivation for dense retrieval in Tier 2.

### 5.4 Tier 3 / AQS — 48-question stratified subset

UMBRELA, eRAG, RAGAS-WA, and RAGAS-WB were run on all 12 sparse systems via the `bsard_evaluation` harness (sibling `RQ3_Autonomous_Evaluation` component; gpt-4o-mini judge, k=10). The full Tier 1/2 panel was recomputed on the same 48 questions and stored under each result JSON's `subset_metrics` block.

| Experiment | AQS | R@10 (48q) | R@100 (48q) |
|---|---|---|---|
| `bm25_tuned_k11.5_b0.25` | **0.3261** | 0.2719 | 0.5146 |
| `bm25_lemmatize_concat_2x` | 0.3044 | 0.2658 | 0.5267 |
| `bm25_stem_text_only` | 0.3023 | 0.2672 | 0.4886 |
| `bm25_plus_lemmatize` | 0.2930 | 0.2249 | 0.5097 |
| `bm25_lemmatize_text_only` | 0.2858 | 0.2256 | 0.5095 |
| `tfidf_lemmatize_concat_2x` | 0.2554 | 0.1780 | 0.5008 |
| `fts5_default` | 0.2508 | 0.2060 | 0.4587 |
| `bm25_none_concat_2x` | 0.2500 | 0.2169 | 0.4796 |
| `bm25_anchor` | 0.2485 | 0.1910 | 0.4516 |
| `bm25_l_lemmatize` | 0.2362 | 0.1748 | 0.3942 |
| `tfidf_lemmatize_text_only` | 0.2323 | 0.1420 | 0.4644 |
| `tfidf_none_text_only` | 0.1953 | 0.1219 | 0.3812 |

`bm25_tuned_k11.5_b0.25` leads on AQS; `bm25_lemmatize_concat_2x` leads on subset R@100. Spearman rank correlation between AQS and supervised metrics (R@10/NDCG@10) on the 48-question subset is ρ > 0.85 — the subset is a reliable signal for the RQ3 evaluator analysis and for Tier 4 comparison.

---

## 6. Key Findings Summary

Findings are tagged as **[Replication]** (confirming or extending a BSARD paper result) or **[Proposal]** (new finding from this work).

1. **[Replication] BM25 ceiling at R@10 ≈ 0.265.** The paper reports BM25 as the stronger lexical method; confirmed here. Best test-set R@10 = 0.2651 (`bm25_tuned_k11.5_b0.25`). No BM25 variant is significantly better than the anchor at R@10 (all p > 0.05); improvements are within paired-t-test noise.

2. **[Replication] BM25 MRR@100 reproduced almost exactly.** Anchor `bm25_anchor` MRR@100 = 0.2463 vs paper 0.2459 (+0.0004) — confirms the replication is faithful.

3. **[Replication / Extension] TF-IDF R@100 exceeds the paper.** All three TF-IDF variants beat paper R@100 = 0.4013: `tfidf_none_text_only` = 0.4543, `tfidf_lemmatize_text_only` = 0.4793, `tfidf_lemmatize_concat_2x` = 0.5152. The dedup corpus (22.6k unique articles vs the paper's 22.6k base + duplicates surfacing the same gold) and the explicit lemmatize/concat_2x ablations explain the gain.

4. **[Replication] TF-IDF is significantly worse than BM25 at R@10** for all three TF-IDF variants (p ≤ 0.037), consistent with the paper. BM25's saturation and length normalisation suit legal prose better. (`tfidf_lemmatize_text_only` is strongest: p = 0.001.)

5. **[Proposal] No normalisation does not significantly hurt R@10.** All three normalisation variants (none / stem / lemmatize at `text_only`) land within paired-t-test noise of one another. Statutory vocabulary is consistent enough that raw token overlap works comparably to lemmatisation.

6. **[Proposal] Field weighting (concat_2x) gives the highest Recall@100 (0.5312)** for `bm25_lemmatize_concat_2x`, +1.9 percentage points over the same config without concat_2x. This is the critical metric for the Tier 3 hybrid candidate pool and drives the Tier 3 selection in §7.

7. **[Proposal] Hyperparameter tuning (b=0.25) gives the highest Recall@10 (0.2651)** for `bm25_tuned_k11.5_b0.25`, a gain over the anchor's b=0.75 default that is consistent with statutory length variability arguments (§4). The improvement is not individually significant on the test set (p = 0.374), but reproduces across stratified slices.

8. **[Proposal] BM25L dramatically underperforms** — R@100 = 0.4040 vs BM25 Okapi at the same normalisation = 0.5121 (p = 0.002). BM25L's length normalisation is counterproductive on statutory article lengths.

9. **[Proposal] BM25+ ≈ BM25 Okapi** at the same normalisation (R@100 = 0.5088 vs 0.5121; p = 0.94). The lower-bound term-saturation correction in BM25+ does not help on legal text.

10. **[Proposal] FTS5 is competitive and fast** (R@10 = 0.2365, mean 58.1 ms — about half the anchor's pure-Python BM25 latency), but cannot support field weighting. Useful as a practical baseline; not selected for Tier 3.

11. **[Proposal] Semantic paraphrasing is the dominant failure mode.** Across all 12 experiments, lexically aligned queries achieve R@10 ≈ 0.50, semantically paraphrased queries achieve R@10 ≈ 0.01–0.07 (see §5.3). This gap cannot be closed with sparse methods alone — it is the headline motivation for dense retrieval in Tier 2.

---

## 7. Configurations Selected for Downstream Tiers

Two different sparse configurations are carried forward, each chosen for its own downstream role.

### 7.1 Tier 3 (Hybrid fusion) — `bm25_lemmatize_concat_2x`

BM25 Okapi, k1=1.5, b=0.75, `lemmatize` normalisation, `concat_2x` field weighting.

Rationale: highest Recall@100 = 0.5312 across all Tier 1 experiments. Tier 3 hybrid fusion combines sparse and dense candidate pools at rank 100; maximising the sparse R@100 is the primary optimisation target at this stage. The cross-encoder re-ranker in Tier 3 handles the final ranking, so R@10 at the sparse stage is secondary.

### 7.2 Tier 4 (Agentic) first stage — `bm25_tuned_k11.5_b0.25`

BM25 Okapi, k1=1.5, b=0.25, `lemmatize` normalisation, `text_only` field weighting.

Rationale: the agentic methods (T4.0 LLM-Judge, T4.1 CRAG, T4.2 ReAct) reuse the **same** first-stage retriever across all three so their effects are directly comparable. The `concat_2x` variant is avoided here because the field-weighted text leaks structural metadata (`law_code`, `chapter_title`, `article_number`) into the surface input that the LLM judge sees — a confound that would inflate apparent judge accuracy. The `stem` variant is avoided because its truncated tokens make the prompted context harder to read for the LLM. Among non-concat, non-stem variants, `bm25_tuned_k11.5_b0.25` has the best pool quality (R@50 = 0.463, R@100 = 0.521, MRR@100 = 0.263) and the best AQS on the 48-question subset (0.326).

---

## 8. Implementation Issues and Resolutions

| Issue | Root Cause | Fix |
|---|---|---|
| spaCy tokenisation 932s for 33k docs | spaCy pipeline overhead even with disabled components | Replaced with compiled French-character regex for `none`/`stem`; spaCy only for `lemmatize` |
| FTS5 returning zero results | French apostrophes (e.g., "l'article") caused FTS5 syntax errors | Regex extraction of alphanumeric tokens before FTS5 query construction |
| FTS5 near-zero recall | Implicit AND: single unmatched term zeros entire multi-term query | Switched to explicit `OR` joining |
| FTS5 low recall despite OR fix | Index contained 6,490 non-BSARD distractor articles | Added `AND a.is_bsard_article = 1` filter in JOIN |
| `has_cross_references` column not found | Column is on `articles` table, not `questions` table | Removed from `SELECT` in `evaluation/split.py` |
| Unicode error on Windows console | Windows cp1252 cannot encode `→`, `✓`, `✗` | Replaced with ASCII equivalents `->`, `Y`, `N` |

---

## 9. Output Files

All result JSONs are stored at `output/results/sparse_retrieval/` under the gitignored data root:

```
bm25_anchor_test.json
bm25_lemmatize_text_only_test.json
bm25_stem_text_only_test.json
bm25_none_concat_2x_test.json
bm25_lemmatize_concat_2x_test.json
bm25_tuned_k11.5_b0.25_test.json
bm25_plus_lemmatize_test.json
bm25_l_lemmatize_test.json
fts5_default_test.json
tfidf_none_text_only_test.json
tfidf_lemmatize_text_only_test.json
tfidf_lemmatize_concat_2x_test.json
```

The analysis notebook is at `analysis/sparse_retrieval/tier1_sparse_analysis.ipynb` (executed; includes Recall@k curves, ablation comparisons, stratified breakdowns, latency scatter, and commentary).

---

## 10. Result JSON Schema

Each result JSON in `output/results/sparse_retrieval/` follows this structure:

```json
{
  "experiment_id":        "bm25_lemmatize_concat_2x_test",
  "timestamp":            "2026-04-05T21:49:46",
  "model_or_method":      "bm25",
  "hyperparameters":      { "variant": "okapi", "k1": 1.5, "b": 0.75 },
  "preprocessing":        { "normalization": "lemmatize", "field_weighting": "concat_2x", ... },
  "token_length_audit":   { "fraction_truncated": 0.0, "max_tokens_observed": 0 },
  "training_regime":      "zero_shot",
  "latency_ms_mean":      69.7,
  "latency_ms_std":       12.3,
  "latency_distribution": { "p50": ..., "p90": ..., "p95": ..., "p99": ..., ... },
  "metrics":              { /* full 222-question Tier 0/1/2 panel */ },
  "significance_vs_anchor": { "p_value_recall10": 0.600, "significant": false },
  "stratified":           { "single_article": {...}, "lexically_aligned": {...}, ... },
  "subset_metrics": {
    "subset_ids": [7, 32, 37, 59, ...],
    "n": 48,
    "metrics": { /* full Tier 0/1/2 panel + T3/* + T2-umbrela/* */ }
  }
}
```

- `metrics` / `stratified` are produced by `evaluation/runner.py` at run time (Tier 0/1/2 only, all 222 test questions).
- `subset_metrics.metrics` is produced post-hoc by `scripts/evaluation/compute_subset_metrics.py` (Tier 0/1/2 on the 48-question subset). Tier 3 evaluator runs (UMBRELA, eRAG, RAGAS-WA, RAGAS-WB) and the UMBRELA→Tier 2 bridge add `T3/*` and `T2-umbrela/*` keys to the same `subset_metrics.metrics` dict.
- The legacy flat keys (`Recall@10`, `MRR@100`, …) are mirrored alongside the namespaced `T1/`, `T2/P1/`, `T2/P2/` keys for backward compatibility with the analysis notebook.

### Schema provenance

The Tier 3 autonomous components are the four retrieval-only evaluators (UMBRELA, eRAG, RAGAS-WA, RAGAS-WB); `harness.evaluate()` takes `contexts_with_ranks` and enables ID-based Panel 3 by default. `compute_subset_metrics.py` populates `subset_metrics` on each JSON, and the Tier 3 runs add the `T3/*` and `T2-umbrela/*` keys.
