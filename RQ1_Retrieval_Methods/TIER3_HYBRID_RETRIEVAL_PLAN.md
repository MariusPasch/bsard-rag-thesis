# Tier 3 — Hybrid Retrieval: Implementation & Results

**Status:** Complete · 15 hybrid experiments on the 222-question test split (3 × RRF + 9 × linear α + 3 × SGDR). T3-A2 (BGE-M3 self-hybrid) was skipped because BGE-M3 did not win Tier 2; T3-D (HyDE / RAG-Fusion) is implemented but gated off (`_ENABLE_T3D = False`).
**BSARD RAG Thesis | RQ1**

---

## 0. Context and Positioning

Tier 3 combines the best sparse (Tier 1) and dense (Tier 2) configurations into fusion pipelines.
All experiments use **article-level chunking**
(identical to Tiers 1 & 2) so any performance delta is attributable to the fusion/re-ranking
mechanism alone, not chunking strategy (which is the RQ2 variable in Chapter 5).

**Corpus:** `output/bsard_articles_dedup.parquet` — 22,633 articles, 222 test questions. Same as Tiers 1 & 2.

---

## 1. Anchors from Tiers 1 & 2

### 1.1 Sparse component — Tier 1 results (dedup corpus, 22,633 articles)

All results below are on the **test split**, dedup corpus:

| Experiment ID | Normalisation | Field | k1 | b | R@10 | R@100 | MRR@100 | Notes |
|---|---|---|---|---|---|---|---|---|
| `bm25_lemmatize_concat_2x` | lemmatize | concat_2x | 1.5 | 0.75 | 0.2572 | **0.5312** | 0.2520 | Best R@100 (not selected — see rationale) |
| `bm25_tuned_k11.5_b0.25` | lemmatize | text_only | 1.5 | 0.25 | **0.2651** | 0.5210 | **0.2628** | **Best R@10 — selected for T3 sparse component & T4 first stage** |
| `bm25_none_concat_2x` | none | concat_2x | 1.5 | 0.75 | 0.2567 | 0.5009 | 0.2454 | |
| `bm25_plus_lemmatize` | lemmatize | text_only | 1.5 | 0.75 | 0.2461 | 0.5088 | 0.2552 | BM25+ variant |
| `bm25_anchor` | none | text_only | 1.5 | 0.75 | 0.2476 | 0.4821 | 0.2463 | Tier 1 anchor |
| `fts5_default` | — | text_only | — | — | 0.2365 | 0.4697 | 0.2367 | Fast but no field weighting |
| `tfidf_lemmatize_concat_2x` | lemmatize | concat_2x | — | — | 0.2003 | 0.5152 | 0.1995 | Sig. worse than BM25 |

**Sparse component selected for Tier 3:** `bm25_tuned_k11.5_b0.25`

**Rationale:** Aligned with the Tier 4 first-stage retriever (T4.0 LLM-judge, T4.1 CRAG, T4.2 ReAct all share this configuration — see §7.2 of `TIER1_SPARSE_RETRIEVAL_PLAN.md`). `bm25_tuned_k11.5_b0.25` achieves the best R@10 = 0.2651 across all 12 sparse experiments. The lower `b = 0.25` (reduced length normalisation) pushes more relevant articles into the top-10, which is the primary T3 metric. `bm25_lemmatize_concat_2x` has higher R@100 (0.5312 vs 0.5210) but this advantage does not outweigh the cross-tier consistency benefit, and the dense leg of the hybrid is what drives recall above the sparse ceiling.

Config: `BM25Retriever(normalization="lemmatize", field_weighting="text_only", variant="okapi", k1=1.5, b=0.25)`

### 1.2 Dense component — Tier 2 model candidates

Results are on the **test split** (222 questions). Selection criterion: **highest R@100 on test**,
with R@10 as tie-breaker. Eligible models (D1 and D6 excluded — near-zero and 128-token limit
respectively):

| Exp ID | Experiment ID | HF Model | Prefix (query / passage) | Pooling | max_seq | Notes |
|---|---|---|---|---|---|---|
| D2 | `dense_camembert_lg_zeroshot_test` | `dangvantuan/sentence-camembert-large` | none / none | Mean | 512 | French-specific |
| D3 | `dense_me5_base_zeroshot_test` | `intfloat/multilingual-e5-base` | `"query: "` / `"passage: "` | Mean | 512 | Multilingual baseline |
| D4 | `dense_me5_large_zeroshot_test` | `intfloat/multilingual-e5-large` | `"query: "` / `"passage: "` | Mean | 512 | Strong multilingual candidate |
| D4c | `dense_bge_m3_zeroshot_test` | `BAAI/bge-m3` | none / none | CLS | 1024 | Long-context; legal IR SOTA candidate |
| D5 | `dense_paper_checkpoint_pretrained_test` | `antoiloui/<checkpoint>` | none / none | Mean | 512 | Paper ceiling — **not selected for T3 fusion** |

**Dense component selection:** `argmax(R@100 on test)` among D2, D3, D4, D4c.

**Resolved (Decision D1):** `intfloat/multilingual-e5-large` (D4, `dense_me5_large_concat2x_zeroshot_test`)
won on R@100 on the test split. BGE-M3 (D4c) was the pre-execution expectation but did not win.
Consequence: T3-A2 (BGE-M3 self-hybrid) is **skipped** — see §3.5.

**Field weighting for dense in T3 (Decision D2):** `concat_2x` — confirmed by the Tier 2 D7 result.

**Paper checkpoint (D5) excluded from T3 fusion** — it is the fine-tuned ceiling reported
separately for benchmark comparability; mixing it into a hybrid would confound the zero-shot
comparison.

### 1.3 Significance anchor for Tier 3

**Tier 3 anchor:** `intfloat/multilingual-e5-large` with `concat_2x` field weighting
(`dense_me5_large_concat2x_zeroshot_test`) — the T2 winner on R@100 on the test split.
Significance: paired t-test on per-query Recall@10, `p < 0.05`, via
`evaluation/runner.py:add_significance()`. Primary k = 10 (matching RQ1 reporting convention).

---

## 2. Experiment Groups

Three active groups (T3-A, T3-B, T3-C), run in strict order. T3-A2 is skipped (BGE-M3 did not
win T2). T3-D (HyDE + RAG-Fusion) is a potential future expansion — implemented but gated by
`_ENABLE_T3D = False`.

| Group | ID prefix | Method | Input | Split | Condition |
|---|---|---|---|---|---|
| T3-A | `hybrid_rrf_*` | Reciprocal Rank Fusion (RRF) | `bm25_tuned_k11.5_b0.25` (text_only) + mE5-large (concat_2x) | test | always |
| T3-A2 | `hybrid_m3_*` | BGE-M3 self-hybrid (internal heads) | BGE-M3 dense + sparse [+ ColBERT] | test | **SKIPPED — D4c did not win T2** |
| T3-B | `hybrid_linear_*` | Linear score interpolation (α grid) | `bm25_tuned_k11.5_b0.25` (text_only) + mE5-large (concat_2x) | test (all α, best reported) | always |
| T3-C | `hybrid_sgdr_*` | Sparse-Gated Dense Reranking (SGDR) | BM25 pool K ∈ {1000,2000,5000} → dense re-score | test | always |
| *(T3-D)* | `hybrid_hyde_*` / `hybrid_ragfusion_*` | *(HyDE and RAG-Fusion — potential expansion)* | — | — | *not active* |

**Primary metrics:** Recall@10, MRR@10. **Secondary:** NDCG@10, MAP, Recall@100.

---

## 3. T3-A — Reciprocal Rank Fusion (RRF)

### Method

Merge the sparse and dense ranked lists without score calibration:

```
RRF_score(d) = 1 / (k + rank_sparse(d))  +  1 / (k + rank_dense(d))
```

`rank_*(d)` is the 1-based position in each modality's ranked list. Articles absent from a list are
assigned rank = `first_stage_k + 1`. Final list sorted descending by `RRF_score`.

RRF is parameter-free apart from `k` (dampening constant) and requires no score normalization.

### Inputs

- **Sparse:** `BM25Retriever(normalization="lemmatize", field_weighting="text_only", variant="okapi", k1=1.5, b=0.25)` — top-100
- **Dense:** best model from Tier 2 test selection — top-100
- **first_stage_k = 100** for both modalities before fusion

### Experiments

| Experiment ID | k (RRF) | Split | Note |
|---|---|---|---|
| `hybrid_rrf_k30_test` | 30 | test | Shifts weight toward dense (smaller k → higher reward for top ranks) |
| `hybrid_rrf_k60_test` | 60 | test | Standard default |
| `hybrid_rrf_k120_test` | 120 | test | Shifts weight toward sparse tail |

All three k values run on test. Best k selected from test results and reported as canonical.

**Why RRF first:** No calibration needed, runs in milliseconds after first-stage retrieval,
establishes the hybrid floor before tuning T3-B.

---

## 3.5 T3-A2 — BGE-M3 Self-Hybrid *(SKIPPED — D4c did not win Tier 2)*

**Condition:** Run this group **only** if `BAAI/bge-m3` (D4c) is selected as the Tier 2 dense
model. If any other model wins, skip T3-A2 entirely and document in the Decision Log.

**Decision Log entry:** `intfloat/multilingual-e5-large` (D4) was selected as the T2 dense model
(highest R@100 on test). T3-A2 is therefore **not executed**. `BGEM3SelfHybridRetriever` was
not implemented in `retrieval/hybrid.py`. The script prints a skip notice when
`--dense-model` is not `BAAI/bge-m3`.

### Rationale

BGE-M3 exposes three retrieval heads from a single checkpoint — dense (CLS pooling), learned
sparse (SPLADE-style term weights), and multi-vector ColBERT — all sharing the same tokenizer
and embedding space. This makes score combination trivial and eliminates the BM25 score
normalization instability that affects T3-B. No new model download is required; the checkpoint is
already on disk from Tier 2 (EXP-D4c).

The BGE-M3 paper reports that combining all three heads consistently outperforms each
individually on multilingual retrieval benchmarks, with the Dense+Sparse combination already
surpassing external BM25 hybrids. T3-A2 tests whether this holds for Belgian statutory French.

### Method

**Variant A — Dense + Sparse (two-head):**
```
score(d) = w1 · score_dense(d)  +  w2 · score_sparse(d)
```
Scores are directly comparable — no external normalization needed. Grid w1 on test (w2 = 1 − w1).
Default starting point: w1=0.6, w2=0.4 (dense-dominant, consistent with the paper's findings).

**Variant B — Dense + Sparse + ColBERT (three-head):**
```
score(d) = w1 · score_dense(d)  +  w2 · score_sparse(d)  +  w3 · score_colbert(d)
```
Due to the high cost of ColBERT late-interaction scoring at query time, apply only to top-200
candidates from the Dense+Sparse first stage, not the full corpus. Paper default weights:
w1=0.4, w2=0.2, w3=0.4.

### Inputs

- **All heads:** `BAAI/bge-m3` checkpoint already on disk (D4c) — no new download
- **First stage pool (Variant B):** top-200 from Dense+Sparse to keep ColBERT inference tractable
- No external BM25 index needed

### Experiments

| Experiment ID | Heads used | Weights | Split | Note |
|---|---|---|---|---|
| `hybrid_m3_dense_sparse_w0.6_test` | dense + sparse | w1=0.6, w2=0.4 (default) | test | Starting point |
| `hybrid_m3_dense_sparse_w0.7_test` | dense + sparse | w1=0.7, w2=0.3 | test | α grid step |
| `hybrid_m3_dense_sparse_w0.8_test` | dense + sparse | w1=0.8, w2=0.2 | test | α grid step |
| `hybrid_m3_dense_sparse_best_test` | dense + sparse | best w1 from test results | test | Canonical two-head result |
| `hybrid_m3_all_heads_test` | dense + sparse + ColBERT | w=0.4/0.2/0.4 (paper default) | test | Canonical three-head result |

**Comparison axes:**
- T3-A2 vs T3-A: does the M3 internal sparse signal outperform BM25 as the lexical component?
- T3-A2 Variant B vs T3-C: is the M3 ColBERT head competitive with MiniLM cross-encoder at lower inference cost?

### Anticipated Failure Modes

| Failure Mode | Mitigation |
|---|---|
| BGE-M3 sparse head underperforms BM25 on French statutory text (observed on MLDR long-doc tasks in the paper) | Report as finding; BM25 term-weighting may be better calibrated for statutory French than M3's sparse head trained on multilingual web text |
| ColBERT late-interaction OOM on CPU with top-200 candidates | Reduce to top-100; log the change in result JSON `hyperparameters` |

---

## 4. T3-B — Linear Score Interpolation

### Method

Combine calibrated scores from both modalities:

```
combined_score(d) = α · score_dense(d)  +  (1 − α) · score_sparse_norm(d)
```

- **Dense score:** cosine similarity from FAISS inner product search over L2-normalised vectors —
  already in [0, 1] with no further normalization needed.
- **Sparse score:** raw BM25 scores vary in range. Normalize over the candidate pool:
  `score_sparse_norm(d) = BM25(d) / max(BM25(c) for c in candidate_pool)`.
  Clip at 0 for articles with negative raw BM25 scores (rare edge case in BM25Plus, not Okapi).
- **Candidate pool:** union of top-100 sparse (`bm25_tuned_k11.5_b0.25`) and top-100 dense → up to 200 articles, deduplicated.

### α Grid Protocol

All α values run on the **test split**:

| Experiment ID | α | Split |
|---|---|---|
| `hybrid_linear_alpha_0.1_test` | 0.1 | test |
| `hybrid_linear_alpha_0.2_test` | 0.2 | test |
| `hybrid_linear_alpha_0.3_test` | 0.3 | test |
| `hybrid_linear_alpha_0.4_test` | 0.4 | test |
| `hybrid_linear_alpha_0.5_test` | 0.5 | test |
| `hybrid_linear_alpha_0.6_test` | 0.6 | test |
| `hybrid_linear_alpha_0.7_test` | 0.7 | test |
| `hybrid_linear_alpha_0.8_test` | 0.8 | test |
| `hybrid_linear_alpha_0.9_test` | 0.9 | test |

Best α selected from test results by Recall@10 and reported as the canonical T3-B result.

**Expected best α:** α ≈ 0.7–0.8 (dense-dominant). Since Tier 2 dense models are expected to
substantially outperform BM25 zero-shot on semantically paraphrased queries (where sparse R@10
≈ 0.01–0.07), α > 0.5 is the anticipated optimum. The sparse component contributes most on
lexically aligned queries where BM25 R@10 ≈ 0.50.

---

## 5. T3-C — Sparse-Gated Dense Reranking (SGDR)

### Method

Two-stage pipeline: BM25 retrieves a large candidate pool; the dense encoder re-scores only those candidates using precomputed embeddings (K dot products — no new encoding at query time).

```
Stage 1: BM25(q) → top-K article_ids          (lexical, fast, high recall at large K)
Stage 2: dense_score(q, article_id) for each of K candidates  ← K dot products only
Final:   sort by dense score descending → top-500
```

**Why K > 500:** The standard evaluation grid is `custom_k = [1, 5, 10, 20, 50, 100, 200, 500]`. All eight k-values fall strictly below K, so every reported metric reflects genuine dense reranking signal. At K=500, R@500 would equal BM25@500 recall by definition, masking the reranker's contribution.

### Inputs

- **Sparse:** `BM25Retriever(normalization="lemmatize", field_weighting="text_only", variant="okapi", k1=1.5, b=0.25)` — returns top-K candidates per query
- **Dense:** same model as T3-A/B (best Tier 2 model, mE5-large concat_2x) — re-scores candidates only; document embeddings already precomputed
- **K grid:** {1000, 2000, 5000}

### Experiments

| Experiment ID | BM25 pool K | Split | Informative k-values |
|---|---|---|---|
| `hybrid_sgdr_k1000_test` | 1000 | test | all 8 custom_k values (1–500) |
| `hybrid_sgdr_k2000_test` | 2000 | test | all 8 custom_k values (1–500) |
| `hybrid_sgdr_k5000_test` | 5000 | test | all 8 custom_k values (1–500) |

Best K selected from test results by Recall@10 and reported as the canonical T3-C result.

### Implementation

`_SGDRetriever` (inline class in `run_hybrid_experiments.py`):

- **`__init__`:** builds `article_id → FAISS position` reverse map; extracts the full embeddings matrix from the FAISS index once via `reconstruct_n` — no re-encode, no new I/O.
- **`retrieve(query, top_k=500)`:** BM25 retrieves K candidates → encode query → numpy matrix row-index slice → K dot products → argsort descending → return top_k.
- **Embeddings matrix shared** across pool sizes (K=1000/2000/5000 all reuse the same precomputed matrix; only the BM25 pool size differs).

**No new model loading** — same `DenseRetriever` instance from T3-A/B is reused. Latency per query: BM25 retrieval + 1 query encoding + K dot products (negligible for K ≤ 5000).

---

## 6. T3-D — HyDE and RAG-Fusion *(potential future expansion)*

**Priority:** Lower than T3-A/B/C. Attempt only after T3-C is complete and the Week 7 checkpoint
(start of RQ2 / Phase 3) is not at risk. If timeline is tight, T3-D results become "future work."

### T3-D1: HyDE (Hypothetical Document Embeddings)

**What:** For each query, prompt an LLM to generate a hypothetical statutory article that would
answer the query. Embed the hypothetical article (instead of the raw query) using the same dense
encoder as T3-A/B/C, then retrieve by cosine similarity.

**Intuition for BSARD:** Natural language questions use everyday vocabulary; statutory articles use
formal legal jargon. The hypothetical article bridges this vocabulary gap by being in statutory
register.

**Design decision — k=1 (single hypothetical document):** The original HyDE paper generates
multiple documents and averages their embeddings to reduce LLM stochastic variance. This plan
uses k=1 for two reasons: (1) it keeps latency predictable (one LLM call per query vs. five), and
(2) averaging embeddings across multiple LLaMA 3.1 8B outputs provides marginal benefit when
the generator quality on Belgian statutory French is already uncertain. If results are weak,
report as a negative finding attributable to the generator's register quality, not to k=1. **Do not
average embeddings from multiple generation calls** — the prompt returns a single `Article:`
response and that embedding is used directly.

**Pipeline:**
1. Prompt LLM → one hypothetical article text per query (French, statutory register, ~150 tokens)
2. Embed with best Tier 2 dense model (same prefix conventions as T3-A/B/C)
3. Retrieve top-100 by cosine similarity
4. Optional: fuse HyDE top-100 with BM25 lem_concat2x top-100 via RRF

**LLM prompt (French, few-shot — as implemented in `retrieval/hybrid.py:_HYDE_PROMPT_TEMPLATE`):**
```
Exemples :

Question : Quelles sont les conditions pour résilier un contrat de bail ?
Article : Le preneur peut résilier le contrat de bail à tout moment en respectant un préavis
de trois mois, sauf convention contraire entre les parties.

Question : Qui est responsable en cas d'accident causé par un animal ?
Article : Le propriétaire d'un animal est responsable du dommage que l'animal a causé,
soit que l'animal fût sous sa garde, soit qu'il fût égaré ou échappé.

---

Question : "{question}"
Rédigez un court article de loi belge (2-3 phrases) qui répondrait directement à cette question.
Article :
```
Two few-shot examples are hardcoded inline (generic Belgian civil law; not drawn dynamically from
the BSARD train split, which was the original design intent). The examples use bail and animal
liability — domain-appropriate but not BSARD-specific.

**LLM backend:** LLaMA 3.1 8B via Ollama (local, free, reproducible). Fallback: GPT-4o-mini
(API cost ~$0.01 for 222 questions). Cache generated hypothetical texts as
`output/hyde_cache_{llm_model}_test.json` keyed by `question_id`.

**Experiments:**

| Experiment ID | Generator | Fusion | Split |
|---|---|---|---|
| `hybrid_hyde_llama_test` | LLaMA 3.1 8B | None (pure HyDE) | test |
| `hybrid_hyde_rrf_llama_test` | LLaMA 3.1 8B | RRF with BM25 lem_concat2x | test |

Best configuration selected from test results and reported as the canonical T3-D1 result.

**Risk:** If Recall@10 does not exceed the best T3-A/B result, HyDE is reported as a negative
result (valid thesis finding: LLM-generated hypothetical articles do not reliably reproduce
Belgian statutory register on zero-shot LLaMA).

### T3-D2: RAG-Fusion

**What:** Generate N=4 paraphrase variants of each query, retrieve independently for each with
the best dense model, then merge all N+1 ranked lists (original + 4 paraphrases) via RRF.

**N+1 merge (required):** Retrieve independently for each of the 4 paraphrases **plus the
original query**, then merge all 5 ranked lists via RRF. The original query must always be
included in the merge. For statutory text where the user's phrasing may already closely match
article wording, dropping the original degrades exactly the queries where sparse and dense
retrieval perform best.

**Paraphrase generation prompt (French):**
```
Générez 4 reformulations de la question suivante en français, en variant la structure
syntaxique et le vocabulaire, mais en conservant le sens juridique exact :
"{question}"
Reformulations :
1.
```

LLM backend: LLaMA 3.1 8B (same as HyDE). Cache paraphrases as
`output/ragfusion_cache_{llm_model}_test.json`. The cache stores all 4 paraphrases per
`question_id`; the original question is taken from the BSARD question file directly, not cached.

**Experiments:**

| Experiment ID | N paraphrases | Lists merged | Retriever per list | Split |
|---|---|---|---|---|
| `hybrid_ragfusion_dense_n4_test` | 4 | 5 (original + 4) | best Tier 2 dense only | test |
| `hybrid_ragfusion_hybrid_n4_test` | 4 | 5 (original + 4) | T3-A RRF per list | test |

Best configuration selected from test results and reported as the canonical T3-D2 result.

---

## 6.5 Tier 3 results — 222-question test set

All 15 hybrid experiments, sorted by Recall@10. Significance is the paired t-test on per-query Recall@10 vs the dense anchor `dense_me5_large_concat2x_zeroshot_test` (R@10 = 0.3420, R@100 = 0.6215).

| Experiment | R@10 | R@100 | MRR@10 | NDCG@10 | Lat (ms) | p (R@10) |
|---|---|---|---|---|---|---|
| `hybrid_rrf_k60` | **0.4021** | **0.6513** | 0.3402 | 0.3094 | 321.5 | **0.0145** |
| `hybrid_rrf_k120` | 0.4015 | 0.6513 | 0.3387 | 0.3084 | 325.8 | **0.0156** |
| `hybrid_rrf_k30` | 0.3972 | 0.6513 | **0.3450** | 0.3091 | 326.9 | **0.0169** |
| `hybrid_linear_alpha_0.9` | 0.3837 | 0.6215 | 0.3322 | 0.3018 | 649.0 | 0.099 |
| `hybrid_sgdr_k1000` | 0.3655 | 0.5916 | 0.3271 | 0.2851 | 336.1 | 0.082 |
| `hybrid_linear_alpha_0.8` | 0.3611 | 0.6215 | 0.3152 | 0.2871 | 638.2 | 0.459 |
| `hybrid_sgdr_k2000` | 0.3538 | 0.6019 | 0.3192 | 0.2782 | 309.8 | 0.225 |
| `hybrid_linear_alpha_0.7` | 0.3518 | 0.6215 | 0.3113 | 0.2814 | 646.7 | 0.708 |
| `hybrid_linear_alpha_0.6` | 0.3515 | 0.6215 | 0.3057 | 0.2788 | 655.7 | 0.716 |
| `hybrid_linear_alpha_0.5` | 0.3470 | 0.6360 | 0.3021 | 0.2746 | 655.6 | 0.854 |
| `hybrid_sgdr_k5000` | 0.3395 | 0.6018 | 0.3150 | 0.2707 | 337.7 | 0.716 |
| `hybrid_linear_alpha_0.4` | 0.3379 | 0.6366 | 0.3017 | 0.2708 | 654.2 | 0.882 |
| `hybrid_linear_alpha_0.3` | 0.3318 | 0.5494 | 0.3005 | 0.2682 | 647.0 | 0.718 |
| `hybrid_linear_alpha_0.2` | 0.3292 | 0.5248 | 0.2910 | 0.2630 | 638.3 | 0.657 |
| `hybrid_linear_alpha_0.1` | 0.2909 | 0.5210 | 0.2719 | 0.2429 | 658.5 | 0.079 |

**Bold p-values** mark statistically significant differences from the dense anchor (p < 0.05).

**Per-group winners:**
- **T3-A:** `hybrid_rrf_k60` — R@10 = 0.4021 (+0.0601 vs dense anchor, +0.1370 vs sparse anchor), R@100 = 0.6513 (+0.030 vs dense anchor). All three RRF k values significantly beat the anchor.
- **T3-B:** `hybrid_linear_alpha_0.9` (dense-dominant) — R@10 = 0.3837, p = 0.099 (not significant).
- **T3-C:** `hybrid_sgdr_k1000` — R@10 = 0.3655, p = 0.082. Larger pool K degrades R@10 (k=1000 → 0.3655, k=5000 → 0.3395) because BM25 noise dilutes the dense reranker signal.

**Overall Tier 3 winner: `hybrid_rrf_k60`** (R@10 = 0.4021). RRF wins because it avoids score normalisation entirely: both sparse and dense ranks contribute independently, capturing the complementarity (sparse on lexically aligned queries, dense on paraphrased queries) without needing score calibration.

### 6.6 Stratified breakdown — `hybrid_rrf_k60`

| Stratum | R@10 | R@100 |
|---|---|---|
| Lexically aligned | 0.7164 | 0.9164 |
| Semantically paraphrased | 0.1349 | 0.3693 |
| Single-article | 0.5904 | 0.7952 |
| Multi-article | 0.2896 | 0.5654 |
| With cross-refs | 0.4251 | 0.6594 |
| Without cross-refs | 0.2971 | 0.6146 |

The RRF hybrid:
- Outperforms the dense-only anchor on the **lexically-aligned** stratum (0.7164 vs 0.5355 R@10) by recovering the lexical-match signal that pure dense retrieval discards.
- Matches the dense anchor on the **semantically-paraphrased** stratum (0.1349 vs 0.1407 R@10) — sparse adds little here, as expected. Recall@100 on paraphrased queries (0.3693) is essentially identical to dense-only (0.3921).
- Net effect: RRF captures sparse's lexical advantage *and* dense's semantic coverage in a single ranked list.

### 6.7 T3 ceiling for Tier 4

`hybrid_rrf_k60` (R@10 = 0.4021) is the **non-agentic ceiling** that Tier 4 agentic methods (T4.0 LLM-judge, T4.1 CRAG, T4.2 ReAct) must exceed to justify their LLM-call latency and cost. This trade-off is the core of the RQ1 cost-benefit analysis (thesis Section 4.5).

Note that Tier 4 methods use the **sparse anchor** (`bm25_tuned_k11.5_b0.25`) as their first stage, not the Tier 3 hybrid — see §16 of `TIER2_DENSE_RETRIEVAL_PLAN.md` for why the comparison is done at matched pool sizes.

---

## 7. Implementation Architecture

### 7.1 New Module: `retrieval/hybrid.py`

Follows the same interface contract as `sparse.py` and `dense.py`:

```python
class HybridRetriever:
    """
    Fuses BM25Retriever + DenseRetriever via RRF or linear interpolation.
    retrieve(query, top_k) -> (ranked_article_ids, latency_ms)
    Latency covers both first-stage calls + fusion computation.
    """
    def __init__(
        self,
        sparse_retriever: BM25Retriever,
        dense_retriever: DenseRetriever,
        fusion_method: str,       # "rrf" | "linear"
        alpha: float = 0.5,       # T3-B only; ignored for "rrf"
        rrf_k: int = 60,          # T3-A only; ignored for "linear"
        first_stage_k: int = 100,
    ): ...

    def retrieve(self, query: str, top_k: int = 10) -> tuple[list[int], float]: ...
```

*(T3-A2 was skipped — `BGEM3SelfHybridRetriever` was not implemented.)*

```python
class CrossEncoderReranker:
    """
    Two-stage: first_stage_retriever.retrieve() -> top-N -> cross-encoder re-score.
    retrieve(query, top_k) -> (ranked_article_ids, latency_ms)
    Latency includes both stages.
    """
    def __init__(
        self,
        first_stage_retriever,            # HybridRetriever, BGEM3SelfHybridRetriever, or any retriever
        model_name: str,                  # CE HF ID
        top_n: int = 100,                 # candidates passed to CE
        article_texts: dict[int, str],    # {article_id: text} for CE input
        max_article_tokens: int = 400,    # truncation before CE
    ): ...

    def retrieve(self, query: str, top_k: int = 10) -> tuple[list[int], float]: ...
```

> **`CrossEncoderReranker` status (2026-04):** The class is implemented and unit-tested ([tests/hybrid/test_hybrid_retriever.py](tests/hybrid/test_hybrid_retriever.py)), but the canonical Tier 3 pipeline in [scripts/hybrid/run_hybrid_experiments.py](scripts/hybrid/run_hybrid_experiments.py) does **not** call it — T3-C is implemented as Sparse-Gated Dense Reranking (`_SGDRetriever`, defined inline in the orchestrator), not as cross-encoder reranking. The CE reranker is retained for two consumers: (a) `scripts/evaluation/shared/compute_significance.py`, which builds a CE-reranked variant for ad-hoc significance comparisons; and (b) the deprecated parallel orchestrator at [scripts/evaluation/tier3/run_hybrid_experiments.py](scripts/evaluation/tier3/run_hybrid_experiments.py), an earlier draft that used CE reranking instead of SGDR for T3-C and is not the version that produced the result JSONs in `output/results/hybrid/`.

HyDE and RAG-Fusion implemented as standalone functions (not classes):

```python
def hyde_retrieve(
    query: str,
    llm_fn: Callable[[str], str],    # prompt -> generated text (k=1, no averaging)
    dense_retriever: DenseRetriever,
    top_k: int = 10,
    hyde_cache: dict[str, str] | None = None,  # {query: hyp_text}; mutated in-place
) -> tuple[list[int], float]: ...

def ragfusion_retrieve(
    query: str,
    paraphrase_fn: Callable[[str], list[str]],
    retriever,                        # DenseRetriever or HybridRetriever
    n_paraphrases: int = 4,
    rrf_k: int = 60,
    top_k: int = 10,
    include_original: bool = True,    # always True — original query merged as list N+1
    paraphrase_cache: dict[str, list[str]] | None = None,  # {query: paraphrases}; mutated in-place
) -> tuple[list[int], float]: ...
```

### 7.2 New Script: `scripts/hybrid/run_hybrid_experiments.py`

**Root resolution:** The script is two levels below the project root. Use:

```python
PROJECT_ROOT = Path(__file__).parents[2]  # hybrid/ → scripts/ → root
```

All relative paths (`output/`, `evaluation/`) are resolved against `PROJECT_ROOT`.

Structure mirrors `run_sparse_experiments.py` and `run_dense_experiments.py`:

1. Load corpus from `output/bsard_articles_dedup.parquet`
2. Load questions via `evaluation/split.py:load_questions()`
3. Load strata via `evaluation/stratify.py:load_strata()`
4. Load T3 anchor from hardcoded path (`dense_me5_large_concat2x_zeroshot_test.json`)
5. Instantiate sparse component: `BM25Retriever(normalization="lemmatize", field_weighting="text_only", variant="okapi", k1=1.5, b=0.25)`
6. Instantiate dense component: model from CLI `--dense-model` (default: `intfloat/multilingual-e5-large`, `field_weighting="concat_2x"`)
7. **[Conditional]** If `--dense-model` ≠ `BAAI/bge-m3`: print skip notice for T3-A2 (no experiments run)
8. Run T3-A (RRF k ∈ {30, 60, 120} on test; report best k) — skipped if `--stage t3b|t3c`
9. Run T3-B (α grid 0.1–0.9 on test; report best α) — skipped if `--stage t3a|t3c`
10. Run T3-C (BM25 pool K ∈ {1000, 2000, 5000} on test; report best K) — skipped if `--stage t3a|t3b`
11. Print summary table of all result JSONs in `output/results/hybrid/`, sorted by R@10

CLI:
```bash
# Run all stages with the default dense model (mE5-large, T2 winner):
python scripts/hybrid/run_hybrid_experiments.py

# Override the dense model (e.g. for BGE-M3, which would re-enable T3-A2):
python scripts/hybrid/run_hybrid_experiments.py --dense-model BAAI/bge-m3

# Run a single stage only (avoids reloading models for partial re-runs):
python scripts/hybrid/run_hybrid_experiments.py --stage t3a   # RRF only
python scripts/hybrid/run_hybrid_experiments.py --stage t3b   # linear interpolation only
python scripts/hybrid/run_hybrid_experiments.py --stage t3c   # SGDR only
```

`--stage` choices: `t3a | t3b | t3c | all` (default: `all`). When `--stage t3b` is selected
the script attempts to load existing T3-A result files to recover `best_rrf_k` for T3-D without
re-running T3-A.

**T3 anchor:** hardcoded to
`output/results/dense_retrieval/dense_me5_large_concat2x_zeroshot_test.json`
(resolved once mE5-large was confirmed as T2 winner).

### 7.3 Result JSON Schema

Extends the Tier 2 schema with additional `hyperparameters` fields. No changes to shared
infrastructure (`metrics.py`, `runner.py`, `split.py`, `stratify.py`):

```json
{
  "experiment_id": "hybrid_rrf_k60_test",
  "model_or_method": "hybrid_rrf",
  "hyperparameters": {
    "fusion_method": "rrf",
    "rrf_k": 60,
    "first_stage_k": 100,
    "sparse_config": {
      "variant": "okapi", "normalization": "lemmatize",
      "field_weighting": "text_only", "k1": 1.5, "b": 0.25
    },
    "dense_config": {
      "model_name": "BAAI/bge-m3",
      "field_weighting": "text_only",
      "query_prefix": "",
      "passage_prefix": ""
    }
  }
}
```

For T3-A2 (BGE-M3 self-hybrid), add:
```json
"fusion_method": "bgem3_self_hybrid",
"w_dense": 0.6,
"w_sparse": 0.4,
"use_colbert": false,
"first_stage_k": 200
```

For T3-C, the `hyperparameters` block is:
```json
{
  "experiment_id": "hybrid_sgdr_k1000_test",
  "model_or_method": "hybrid_sgdr",
  "hyperparameters": {
    "method": "sparse_gated_dense",
    "bm25_pool_k": 1000,
    "sparse_config": {
      "variant": "okapi", "normalization": "lemmatize",
      "field_weighting": "text_only", "k1": 1.5, "b": 0.25
    },
    "dense_config": {
      "model_name": "intfloat/multilingual-e5-large",
      "field_weighting": "concat_2x",
      "query_prefix": "query: ",
      "passage_prefix": "passage: "
    }
  }
}
```

For T3-D (if activated), add:
```json
"llm_model": "llama3.1:8b",
"n_paraphrases": 4,
"include_original": true,
"hyde_cache_path": "output/hyde_cache_llama3.1_test.json"
```

### 7.4 Results Directory

```
output/results/hybrid/                         # under the data root — not in git
  hybrid_rrf_k30_test.json
  hybrid_rrf_k60_test.json
  hybrid_rrf_k120_test.json
  hybrid_m3_dense_sparse_w0.6_test.json        # T3-A2 (if BGE-M3 wins T2)
  hybrid_m3_dense_sparse_w0.7_test.json
  hybrid_m3_dense_sparse_w0.8_test.json
  hybrid_m3_dense_sparse_best_test.json        # T3-A2 canonical two-head
  hybrid_m3_all_heads_test.json                # T3-A2 canonical three-head
  hybrid_linear_alpha_0.1_test.json
  ...
  hybrid_linear_alpha_0.9_test.json
  hybrid_sgdr_k1000_test.json                  # T3-C
  hybrid_sgdr_k2000_test.json
  hybrid_sgdr_k5000_test.json
  hybrid_hyde_llama_test.json                  # T3-D (if activated)
  hybrid_hyde_rrf_llama_test.json
  hybrid_ragfusion_dense_n4_test.json
  hybrid_ragfusion_hybrid_n4_test.json
```

---

## 8. Evaluation Protocol

Identical to Tiers 1 & 2 — no changes to shared infrastructure:

- **Significance:** paired t-test on per-query Recall@10 vs best Tier 2 dense model on test.
  Use `add_significance(result, anchor_result, k_values=[10, 100], primary_k=10)`.
- **Stratified breakdown:** all 6 strata from `evaluation/query_strata.json`:
  `single_article`, `multi_article`, `lexically_aligned`, `semantically_paraphrased`,
  `with_cross_refs`, `without_cross_refs`. Each reports Recall@10, Recall@100, MRR@10.
- **Split:** all experiments run directly on the test split (222 questions). Grid search configurations (α, k, top-N) are all evaluated on test; best values are selected from test results and reported as canonical.
- **Top-k:** retrieve top-500 in all experiments (T3-A, T3-B, T3-C). This covers the full
  evaluation grid `custom_k = [1, 5, 10, 20, 50, 100, 200, 500]`. Primary reporting at k=10.

**Key hypotheses for stratified analysis:**
- `bm25_lem_concat2x` excels on `lexically_aligned` (R@10 ≈ 0.50) — RRF/linear should preserve
  this while improving `semantically_paraphrased` via the dense component
- Dense dominates on `semantically_paraphrased` (sparse R@10 ≈ 0.01–0.07 there)
- Best fusion method (RRF vs linear) may vary by query stratum

---

## 9. Execution Order and Decision Gates

```
[PREREQUISITE] Tier 2 test results complete -> dense model selected (argmax R@100 on test)

Step 1  Implement retrieval/hybrid.py + tests/hybrid/test_hybrid_retriever.py
        (HybridRetriever, BGEM3SelfHybridRetriever, CrossEncoderReranker,
         hyde_retrieve, ragfusion_retrieve)

Step 2  T3-A: RRF k in {30, 60, 120} on test; report best k as canonical

Step 3  [Conditional] If dense model = BAAI/bge-m3:
          T3-A2: M3 self-hybrid two-head (w grid) on test; report best w
                 M3 three-head (paper defaults) on test
        Else: skip T3-A2; document in Decision Log
        **OUTCOME: mE5-large won T2 → T3-A2 skipped. Decision D5 = Skip.**

Step 4  T3-B: alpha grid 0.1-0.9 on test; report best alpha as canonical

Step 5  T3-C: BM25 pool K ∈ {1000, 2000, 5000} on test; report best K as canonical

Step 6  (Optional, future) T3-D: HyDE on test; RAG-Fusion on test
        Activate by setting _ENABLE_T3D = True in the script
```

---

## 10. Dependencies

| Dependency | Source | Needed for |
|---|---|---|
| `sentence-transformers` | requirements.txt | Dense encoding (SentenceTransformer), CrossEncoder class |
| `rank-bm25` | requirements.txt | Sparse component (T3-A, T3-B, T3-C) |
| `faiss-cpu` | requirements.txt | Dense FAISS IndexFlatIP |

No new packages were needed for the executed pipeline. `FlagEmbedding` would have been required for the T3-A2 BGE-M3 self-hybrid heads, but that group was skipped — the package is therefore not a Tier 3 runtime dependency. `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` is auto-downloaded by `sentence-transformers` on first use of `CrossEncoderReranker`, but is only triggered by the significance-comparison script, not the main pipeline. T3-D LLM calls (HyDE / RAG-Fusion) are not invoked (gated off).

---

## 11. Predictions vs Actuals

| System | Predicted R@10 | Actual R@10 | Verdict |
|---|---|---|---|
| `bm25_lemmatize_concat_2x` (Tier 1 reference) | 0.2572 | 0.2572 | — (Tier 1 reference value) |
| `bm25_tuned_k11.5_b0.25` (Tier 1 sparse anchor) | 0.2650 | 0.2651 | — (Tier 1 reference value) |
| Dense best zero-shot (Tier 2 anchor) | "≥ 0.45 R@10 for BGE-M3" | 0.3420 (mE5-large + concat_2x) | **Below prediction.** BGE-M3 did not win Tier 2; mE5-large + concat_2x did. The 0.45 prediction was based on BGE-M3's published MLDR numbers and proved too optimistic for BSARD. |
| T3-A RRF best (`hybrid_rrf_k60`) | Dense + 2–4 pp | Dense + **6.0 pp** (0.4021 vs 0.3420) | **Exceeded prediction.** Significant at p = 0.015. |
| T3-A2 M3 self-hybrid | Dense + 2–4 pp | — (skipped) | **Not run** — BGE-M3 did not win Tier 2; `BGEM3SelfHybridRetriever` was not implemented. |
| T3-B linear best (`hybrid_linear_alpha_0.9`) | Dense + 2–5 pp | Dense + **4.2 pp** (0.3837 vs 0.3420) | **In range** but not significant (p = 0.099). RRF ultimately outperforms linear. |
| T3-C SGDR best (`hybrid_sgdr_k1000`) | 0.35–0.45 R@10 | 0.3655 R@10 | **At the low end of the range.** SGDR is recall-limited by BM25's pool; larger K worsens R@10 by diluting the dense signal. |
| T3-D HyDE / RAG-Fusion | Neutral ± 2 pp | — (not activated) | **Not run** — `_ENABLE_T3D = False` in the orchestrator. Implementation is preserved in `retrieval/hybrid.py::hyde_retrieve` and `ragfusion_retrieve` for future activation. |

---

## 12. Anticipated Failure Modes — Observed

| Failure Mode | Affected Method | What was observed | Mitigation / Verdict |
|---|---|---|---|
| BM25 raw scores not bounded — linear fusion unstable if max(BM25) ≈ 0 | T3-B | Did not occur. `_linear()` clips at 0 and uses `eps=1e-8` denominator guard. | Mitigation in place; no instability seen. |
| Dense model dominates so strongly that BM25 adds nothing | T3-A, T3-B | **Did NOT occur for T3-A.** RRF significantly improves over dense (+6 pp R@10, p = 0.015). **Did occur for T3-B.** Linear best α = 0.9 (dense-dominant); α ≤ 0.6 underperforms RRF; the linear gain is not significant. | RRF rescues the BM25 signal that linear normalisation flattens — that's the headline T3 finding. |
| BM25@K recall ceiling too low even at K=5000 | T3-C | **Occurred.** R@10 *worsens* with larger K (K=1000 → 0.3655, K=5000 → 0.3395) because BM25 noise at large K dilutes the dense reranker signal. | Reported as a negative finding: SGDR is bounded by BM25's pool quality at large K. |
| SGDR R@10 matches dense-only (no reranking gain) | T3-C | Partial. SGDR k=1000 R@10 (0.3655) > dense-only (0.3420), but well below RRF k=60 (0.4021). | SGDR adds modest value over dense-only but is uncompetitive with RRF. |
| M3 sparse head underperforms BM25 on statutory French | T3-A2 | Not testable — T3-A2 skipped (BGE-M3 didn't win T2). | n/a |
| ColBERT OOM on CPU with top-200 candidates | T3-A2 Variant B | Not testable — T3-A2 skipped. | n/a |
| HyDE: LLaMA generates French text but not in statutory register | T3-D1 | Not testable — T3-D not activated. | Implementation preserved in `hybrid.py::hyde_retrieve`. |
| RAG-Fusion: paraphrases too lexically similar → no new recall | T3-D2 | Not testable — T3-D not activated. | Implementation preserved in `hybrid.py::ragfusion_retrieve`. |

---

## 13. Open Decisions

| # | Decision | Options | Resolution |
|---|---|---|---|
| D1 | Which Tier 2 dense model for T3? | D2 / D3 / D4 / D4c | **Resolved: `intfloat/multilingual-e5-large` (D4)** — highest R@100 on test |
| D2 | Field weighting for dense component in T3? | `text_only` (Tier 2 default) or `concat_2x` (if EXP-D7 shows gain) | **Resolved: `concat_2x`** — confirmed by Tier 2 EXP-D7 result |
| D3 | LLM backend for T3-D (if activated)? | LLaMA 3.1 8B (Ollama) / GPT-4o-mini | Default LLaMA; fallback to GPT-4o-mini if Ollama quality poor on test |
| D4 | Include Cohere Rerank? | ~~Yes / No~~ | **Resolved: No** — excluded on cost/dependency grounds |
| D5 | Run T3-A2 (BGE-M3 self-hybrid)? | Yes (if D4c wins T2) / Skip | **Resolved: Skip** — D4c did not win T2; `BGEM3SelfHybridRetriever` not implemented |

---

## 14. Relationship to Other Tiers

```
Tier 1 Sparse
  bm25_lem_concat2x ─────────────────────────────────────────────┐
  (R@100 = 0.5312)                                               │
                                                                 ▼
Tier 2 Dense                                            T3-A:  RRF (BM25 + best dense)
  best model D2-D4c ──────────────────────────────────  T3-B:  Linear alpha (BM25 + best dense)
  (D4b reuses D4 embeddings)                                     │
       │                                                          │
       │ [only if D4c / BGE-M3 wins]                             │
       ▼                                                          │
  T3-A2: BGE-M3 self-hybrid ───────────────────────────────────┤
    Variant A: dense + sparse (two-head, alpha-tuned)            │
    Variant B: dense + sparse + ColBERT (three-head)            │
    No new download — same checkpoint as D4c                     │
                                                                 ▼
                                                                 │
                                             T3-C: Sparse-Gated Dense Reranking
                                             BM25 top-K → dense re-score (K dot products)
                                             K ∈ {1000, 2000, 5000}
                                                                 │
                                             T3-D: HyDE / RAG-Fusion
                                             (potential expansion; LLaMA 3.1 8B)
                                                                 │
                                                                 ▼
                                  Tier 4 Agentic (CRAG, ReAct, CRAG+ReAct)
                                  Best T3 result = non-agentic performance
                                  ceiling for RQ1 cost-benefit analysis
```

The best Tier 3 result (best of T3-A / T3-B / T3-C; T3-A2 was skipped because BGE-M3 did not
win Tier 2) is the **non-agentic ceiling** that Tier 4 agentic methods must exceed to justify
their latency and LLM API cost. This trade-off is directly addressed in the RQ1
failure-condition analysis (thesis Section 4.5).

---

## 15. Module Layout

| Path | Purpose |
|---|---|
| `retrieval/hybrid.py` | `HybridRetriever`, `BGEM3SelfHybridRetriever`, `hyde_retrieve()`, `ragfusion_retrieve()` |
| `scripts/hybrid/run_hybrid_experiments.py` | Experiment orchestrator |
| `tests/hybrid/test_hybrid_retriever.py` | Unit tests (mirrors `test_dense_retriever.py`) |
| `output/results/hybrid/` | Result JSONs — under the data root, not in git |
| `TIER3_HYBRID_RETRIEVAL_PLAN.md` | This document |

The shared evaluation infrastructure (`evaluation/metrics.py`, `runner.py`, `split.py`, `stratify.py`) is reused unchanged.
