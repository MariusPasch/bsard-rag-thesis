"""One-off script: inject commentary into the analysis notebook."""
import json, pathlib

nb_path = pathlib.Path(__file__).parent.parent.parent.parent / "analysis" / "sparse_retrieval" / "tier1_sparse_analysis.ipynb"
nb = json.loads(nb_path.read_text(encoding="utf-8"))

COMMENTARY = """---
## 14. Commentary

### 14.1 Overall sparse retrieval performance

The best sparse configuration (`bm25_tuned`, k1=1.5, b=0.5) reaches **Recall@10 = 0.2422** on the 222-question test set. This means the top-10 retrieved articles cover, on average, only **24.2% of all ground-truth relevant articles** per query. Given that the mean number of relevant articles per question is 6.18, sparse retrieval cannot be relied upon as a standalone system and motivates Tier 2 (dense) and Tier 3 (hybrid) approaches.

**Crucially, no sparse configuration is statistically significantly better than the plain BM25 anchor** (all p > 0.05 for BM25 variants). The improvements from normalization, field weighting, and hyperparameter tuning are real but small and noisy given the 222-question test set. The only significant differences are in the *downward* direction: TF-IDF and BM25L are significantly worse (p < 0.01).

---

### 14.2 Normalization effect

Counterintuitively, **no normalization (raw tokenization) outperforms both lemmatization and stemming** on BM25:

| Normalization | Recall@10 | p vs anchor |
|---|---|---|
| none (anchor) | 0.2271 | -- |
| stem | 0.2208 | 0.745 |
| lemmatize | 0.2145 | 0.511 |

Neither difference is statistically significant. The likely explanation is that **Belgian statutory language is terminologically consistent**: legislative drafters use precise, stable vocabulary, and the BSARD benchmark questions were written to match that vocabulary directly. Lemmatization and stemming collapse legally distinct inflected forms (e.g., "licencie" vs "licencier") that carry different legal meanings, introducing noise without recovery benefit. This aligns with the T1-D1 decision rationale but was empirically confirmed here.

**Implication for Tier 3:** The anchor configuration (no normalization) should be retained as the default for the sparse component in hybrid fusion, unless a downstream task specifically benefits from recall on inflected variants.

---

### 14.3 Field weighting effect

Adding law code repetition (`concat_2x`) provides a modest but consistent boost at Recall@10:

| Config | Recall@10 | Recall@100 |
|---|---|---|
| none + text_only (anchor) | 0.2271 | 0.4431 |
| none + concat_2x | 0.2360 | 0.4488 |
| lemmatize + concat_2x | 0.2311 | **0.5079** |

The most notable benefit of `concat_2x` appears at **Recall@100**: `bm25_lemmatize_concat_2x` achieves **R@100 = 0.5079**, the highest across all experiments. Field-weighted lemmatization surfaces more correct articles in the top-100 candidate pool even while slightly underperforming at Recall@10. For Tier 3 hybrid retrieval, where the sparse retriever feeds a top-100 pool into a cross-encoder re-ranker, the configuration with highest R@100 is preferable.

---

### 14.4 BM25 hyperparameter tuning

Grid search on the validation split identified **k1=1.5, b=0.5** as optimal (validated R@10 = 0.2075). On the test set this yields **R@10 = 0.2422**, a +6.6% relative gain over the anchor (+0.0151 absolute), though not statistically significant (p=0.46).

**Reducing b from 0.75 to 0.5 makes sense for statutory text**: articles vary widely in length (median 133 tokens, p95 = 798 tokens), but relevance is independent of length. A short article that precisely answers a narrow legal question is equally relevant as a long comprehensive one. Aggressive length normalization with b=0.75 unnecessarily penalises long articles.

---

### 14.5 BM25 scoring function variants

- **BM25 Okapi** and **BM25+** are essentially identical (R@10 = 0.2145 vs 0.2149; p=0.56).
- **BM25L** dramatically underperforms (R@10 = 0.1368; p < 0.0001, significantly worse).

BM25L applies logarithmic TF normalization designed for long documents with high term repetition. Statutory articles are short and use precise, non-repetitive language. The logarithmic smoothing over-penalises term frequency in this domain. **BM25L should not be used for this corpus.**

---

### 14.6 TF-IDF vs BM25

All three TF-IDF configurations are **significantly worse** than the BM25 anchor (p <= 0.005):

| Method | Recall@10 | p vs anchor |
|---|---|---|
| bm25_anchor | 0.2271 | -- |
| tfidf_none_text_only | 0.1738 | 0.005 |
| tfidf_lemmatize_text_only | 0.1546 | 0.001 |
| tfidf_lemmatize_concat_2x | 0.1499 | 0.001 |

TF-IDF's cosine similarity treats all query terms equally without sub-linear TF saturation. On statutory text where high-frequency legal boilerplate ("article", "loi", "contrat") dominates, BM25's IDF weighting and TF saturation provide a critical advantage. **TF-IDF is not a competitive option for this corpus.**

---

### 14.7 FTS5 baseline

The SQLite FTS5 built-in BM25 achieves **R@10 = 0.2365** -- competitive with the best non-tuned BM25 configurations and not significantly different from the anchor (p=0.56). At **60.7ms mean latency**, it is the second-fastest method.

FTS5's competitive performance is noteworthy: it uses no external dependencies, is pre-built in the database, and handles French unicode61 tokenisation natively. For production deployments where simplicity matters, FTS5 is a viable sparse baseline. Its slight advantage over BM25 Okapi at Recall@10 is likely due to the stopword-filtered OR-joined query representation, which focuses the BM25 signal on content-bearing terms.

---

### 14.8 Single-article vs. multi-article gap

All methods score roughly **2x higher on single-article queries** than multi-article:

| Stratum | Best R@10 | Avg R@10 across methods |
|---|---|---|
| Single-article (n=76, 34.5%) | 0.373 | ~0.28 |
| Multi-article (n=146, 65.5%) | 0.170 | ~0.15 |

For multi-article questions (65.5% of the test set), the best sparse method retrieves fewer than 1 in 6 relevant articles on average at k=10. Recall@100 for the best configuration is only 0.51, meaning even at k=100 only half the relevant articles are found. This is a fundamental limitation: a query about a topic spanning 6+ articles across different laws will almost never have all relevant articles surface in the top-10 by keyword matching.

**Implication:** Multi-article queries are the primary failure mode for Tier 1. Dense retrieval and hybrid approaches must target this stratum specifically.

---

### 14.9 Lexical alignment gap -- the dominant finding

This is the **most striking result** of Tier 1:

| Stratum | BM25 anchor R@10 | Best any method |
|---|---|---|
| Lexically aligned (top quartile) | 0.499 | 0.515 |
| Semantically paraphrased (bottom quartile) | 0.012 | 0.074 |

The gap is **0.41--0.49 Recall@10** across all sparse methods. Sparse retrieval is nearly blind to semantically paraphrased queries -- R@10 for the bottom quartile is 0.01--0.07, meaning sparse methods fail on 93--99% of semantically distant queries.

This validates the fundamental motivation for dense retrieval (Tier 2): the BSARD benchmark has a **median Jaccard overlap of 0.045** between questions and relevant articles, confirming that most legal questions are phrased very differently from the statutory text that answers them. No amount of normalization, field weighting, or parameter tuning can close a vocabulary gap of this magnitude. The semantically paraphrased queries are precisely the ones where dense embeddings trained on semantic similarity should excel.

Lemmatization narrows the gap slightly (0.408 vs anchor 0.487), suggesting morphological normalisation recovers some cases where the paraphrase uses a different inflection of the same root. But the improvement is marginal.

---

### 14.10 Best configuration for Tier 3

Two configurations are candidates for the sparse component in Tier 3:

| Candidate | Recall@10 | Recall@100 | Latency (ms) |
|---|---|---|---|
| bm25_tuned_k11.5_b0.5 | **0.2422** | 0.4855 | 84.6 |
| bm25_lemmatize_concat_2x | 0.2311 | **0.5079** | 92.7 |

Since Tier 3 uses the sparse top-100 as the candidate pool for cross-encoder re-ranking, **maximising Recall@100 is more important than Recall@10**. The cross-encoder promotes correct articles within the top-100 but cannot recover articles missing from the pool. On this basis, **`bm25_lemmatize_concat_2x` is recommended as the sparse component for Tier 3 fusion** (R@100=0.5079).

Record in Decision Log:
- Tier 3 sparse component: `bm25_lemmatize_concat_2x`
- Hyperparameters: Okapi BM25, k1=1.5, b=0.75, lemmatize normalisation, concat_2x field weighting
- Rationale: highest Recall@100 (0.5079) across all Tier 1 configurations
"""

# Cell 28 is the commentary cell (id 0bb19e7e)
nb["cells"][28]["source"] = COMMENTARY
# Clear any existing outputs from re-execution
nb["cells"][28]["outputs"] = [] if nb["cells"][28]["cell_type"] == "code" else None

nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Commentary written. Notebook size: {nb_path.stat().st_size:,} bytes")
