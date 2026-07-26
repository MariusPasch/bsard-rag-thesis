# 07 — EVALUATION CONTEXT
## Binary Metrics, Weighted Partial-Relevance Metrics, Autonomous Eval, Cost Tracking

---

## 1. PURPOSE

Evaluates all retrieval methods on a common basis. Handles the Arm 1 vs Arm 2 comparability problem (chunks vs articles) through weighted partial-relevance metrics. Provides autonomous evaluation when no ground truth exists. Tracks computational cost for all methods. Generates cross-method comparison tables and visualizations.

## 2. DIRECTORY STRUCTURE

```
evaluation/
├── __init__.py
├── adapter.py               # RetrievalResult → TREC qrels/run; bsard_id resolution
├── metrics.py               # Binary Recall@k, NDCG, MRR
├── weight_precomputer.py    # Chunk-article overlap weight precomputation (+ cached wrapper)
├── weighted_metrics.py      # Weighted Recall@k, Precision@k, nDCG; score_chunks_for_question
├── cache.py                 # T07's view of the shared PDF cache (Phase 2)
├── projection.py            # Question→PDF GT projections (Phase 2)
├── ground_truth_loader.py   # load_ground_truth / ground_truth_exists (file-or-dir GT)
├── question_subsets.py      # Build curated GT files from extraction-status metadata
├── autonomous_eval.py       # RAGAS / G-Eval (reference-free) wrappers
├── cost_tracker.py          # LLM calls, latency, tokens tracking
├── comparator.py            # evaluate() / evaluate_partial_views() + significance + stratified
├── eval_stamp.py            # Provenance stamp embedded in eval outputs
└── models.py                # EvalReport
```

## 3. DEPENDENCIES

- Uses: `shared.llm` (for autonomous eval judge), `shared.utils`; reads T03's PDF cache
  (`arm1_naive.cache`)
- **Supervised IR metrics (T0/T1/T2) are delegated to the RQ3 `bsard_evaluation` package**
  (install the sibling `RQ3_Autonomous_Evaluation` to call `evaluate(...)`). T07 adds the `W/*`
  weighted partial-relevance metrics for Arm 1 plus the caching/projection layer.
- Libraries: `scipy` (statistical tests), `pandas`, `numpy`
- Input: `dict[str, list[RetrievalResult]]` from all sub-projects
- Output: `EvalReport` (flat-dict, RQ3-format-compatible)

## 4. BINARY METRICS (metrics.py)

Standard IR metrics computed against ground-truth article IDs.

```python
def recall_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of relevant items found in top-k."""
    retrieved = set(ranked_ids[:k])
    if not relevant_ids:
        return 0.0
    return len(retrieved & relevant_ids) / len(relevant_ids)

def mrr_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Reciprocal rank of first relevant item in top-k."""
    for i, rid in enumerate(ranked_ids[:k]):
        if rid in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0

def ndcg_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Normalized DCG with binary relevance."""
    gains = [1.0 if rid in relevant_ids else 0.0 for rid in ranked_ids[:k]]
    dcg = sum(g / np.log2(i + 2) for i, g in enumerate(gains))
    ideal_gains = sorted(gains, reverse=True)
    idcg = sum(g / np.log2(i + 2) for i, g in enumerate(ideal_gains))
    return dcg / idcg if idcg > 0 else 0.0

def compute_all_binary_metrics(ranked_ids: list[str], relevant_ids: set[str]) -> dict:
    """Compute all binary metrics at standard cutoffs."""
    return {
        'recall@10': recall_at_k(ranked_ids, relevant_ids, 10),
        'recall@100': recall_at_k(ranked_ids, relevant_ids, 100),
        'mrr@10': mrr_at_k(ranked_ids, relevant_ids, 10),
        'ndcg@10': ndcg_at_k(ranked_ids, relevant_ids, 10),
    }
```

## 5. CHUNK-ARTICLE OVERLAP WEIGHTS (weight_precomputer.py)

### 5.1 Problem

Arm 1 retrieves chunks; Arm 2 retrieves articles. A chunk may cover only part of a ground-truth article. Binary metrics give full credit for partial coverage. Weighted metrics assign fractional credit proportional to the overlap.

Reference: Kekäläinen (2005), "Using graded relevance assessments in IR evaluation."

### 5.2 Weight Definition

For each chunk-article pair, compute:

```
w(c, D) = len(tokens of D contained in c) / len(tokens of D)
```

### 5.3 Implementation

```python
from transformers import AutoTokenizer

@dataclass
class ChunkArticleWeight:
    chunk_id: str
    article_id: str
    weight: float

def precompute_weights(
    chunks: list[Chunk],
    articles: list[Article],
    tokenizer_name: str
) -> list[ChunkArticleWeight]:
    """
    Compute overlap weights between all chunk-article pairs.
    
    Uses token positions: each Chunk has start_token, end_token (set during chunking).
    Articles need corresponding token ranges computed from the same raw text.
    
    Returns list of ChunkArticleWeight (only non-zero weights).
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Build article token ranges from the same raw text used for chunking
    # Articles must be mapped to the raw text character/token positions
    article_ranges = compute_article_token_ranges(articles, tokenizer)
    # Returns: {article_id: (start_token, end_token)}
    
    weights = []
    for chunk in chunks:
        chunk_range = range(chunk.start_token, chunk.end_token)
        
        for article_id, (art_start, art_end) in article_ranges.items():
            art_range = range(art_start, art_end)
            art_length = art_end - art_start
            
            # Compute overlap
            overlap_start = max(chunk.start_token, art_start)
            overlap_end = min(chunk.end_token, art_end)
            overlap = max(0, overlap_end - overlap_start)
            
            if overlap > 0 and art_length > 0:
                weight = overlap / art_length
                weights.append(ChunkArticleWeight(
                    chunk_id=chunk.chunk_id,
                    article_id=article_id,
                    weight=weight
                ))
    
    # Validate: sum of weights per article should ≈ 1.0
    validate_weight_coverage(weights, articles)
    
    return weights

def validate_weight_coverage(weights: list[ChunkArticleWeight], articles: list[Article]):
    """Verify that chunks fully cover each article (sum of weights ≈ 1.0)."""
    from collections import defaultdict
    coverage = defaultdict(float)
    for w in weights:
        coverage[w.article_id] += w.weight
    
    for article in articles:
        total = coverage.get(article.article_id, 0.0)
        if abs(total - 1.0) > 0.05:  # 5% tolerance for float rounding
            logger.warning(f"Article {article.article_id}: weight sum = {total:.4f} (expected ~1.0)")

def weights_to_lookup(weights: list[ChunkArticleWeight]) -> dict[tuple[str, str], float]:
    """Convert to dict for O(1) lookup during metric computation."""
    return {(w.chunk_id, w.article_id): w.weight for w in weights}
```

### 5.4 Storage

Store weights as a CSV or pickle for reuse:

```
chunk_id,article_id,weight
chunk_0,art_35,0.45
chunk_0,art_36,0.12
chunk_1,art_35,0.55
...
```

For ~30,000 chunks covering ~22,000 articles, the table has ~30,000-40,000 rows (most chunks overlap exactly one article). Storage: <1 MB.

### 5.5 Computing Article Token Ranges

The articles must be located within the same raw text that was chunked. This requires either:

1. **Character-offset matching:** Find each article's text as a substring of the raw PDF text, then convert character offsets to token offsets using the tokenizer.

2. **Sequential assumption:** If articles appear in document order (which they do in statutory law), compute cumulative token positions sequentially.

```python
def compute_article_token_ranges(articles: list[Article], raw_text: str, 
                                  tokenizer) -> dict[str, tuple[int, int]]:
    """
    Locate each article within the raw text and compute token ranges.
    Assumes articles appear in order within the raw text.
    """
    ranges = {}
    search_start = 0
    
    raw_tokens = tokenizer.tokenize(raw_text)
    
    for article in articles:
        # Find article text in raw text (fuzzy match for OCR variations)
        article_snippet = article.text[:100]  # first 100 chars as anchor
        char_pos = raw_text.find(article_snippet, search_start)
        
        if char_pos == -1:
            # Fuzzy fallback: try with normalized whitespace
            char_pos = fuzzy_find(raw_text, article_snippet, search_start)
        
        if char_pos >= 0:
            # Convert character position to token position
            prefix_tokens = len(tokenizer.tokenize(raw_text[:char_pos]))
            article_tokens = len(tokenizer.tokenize(article.text))
            ranges[article.article_id] = (prefix_tokens, prefix_tokens + article_tokens)
            search_start = char_pos + len(article.text)
        else:
            logger.warning(f"Could not locate article {article.article_id} in raw text")
    
    return ranges
```

## 6. WEIGHTED METRICS (weighted_metrics.py)

```python
import numpy as np

def weighted_recall_at_k(
    ranked_chunk_ids: list[str],
    weights_lookup: dict[tuple[str, str], float],
    gt_article_ids: set[str],
    k: int
) -> float:
    """
    Weighted Recall@K: for each relevant article, sum overlap weights
    of top-K chunks, capped at 1.0. Average across relevant articles.
    """
    if not gt_article_ids:
        return 0.0
    total = 0.0
    for article_id in gt_article_ids:
        w_sum = sum(
            weights_lookup.get((c, article_id), 0.0)
            for c in ranked_chunk_ids[:k]
        )
        total += min(1.0, w_sum)
    return total / len(gt_article_ids)

def weighted_precision_at_k(
    ranked_chunk_ids: list[str],
    weights_lookup: dict[tuple[str, str], float],
    gt_article_ids: set[str],
    k: int
) -> float:
    """Sum of relevant coverage in top-K, divided by K."""
    total_w = sum(
        sum(weights_lookup.get((c, a), 0.0) for a in gt_article_ids)
        for c in ranked_chunk_ids[:k]
    )
    return total_w / k if k > 0 else 0.0

def weighted_mrr(
    ranked_chunk_ids: list[str],
    weights_lookup: dict[tuple[str, str], float],
    gt_article_ids: set[str],
    k: int
) -> float:
    """
    Weighted MRR: effective rank of each relevant article is the rank
    of the first chunk that overlaps with any portion of it.
    """
    if not gt_article_ids:
        return 0.0
    total = 0.0
    for article_id in gt_article_ids:
        for rank, chunk_id in enumerate(ranked_chunk_ids[:k]):
            if weights_lookup.get((chunk_id, article_id), 0.0) > 0:
                total += 1.0 / (rank + 1)
                break
    return total / len(gt_article_ids)

def weighted_ndcg_at_k(
    ranked_chunk_ids: list[str],
    weights_lookup: dict[tuple[str, str], float],
    gt_article_ids: set[str],
    k: int
) -> float:
    """Weighted nDCG: chunk gain = sum of overlap weights with relevant articles."""
    gains = [
        sum(weights_lookup.get((c, a), 0.0) for a in gt_article_ids)
        for c in ranked_chunk_ids[:k]
    ]
    dcg = sum(g / np.log2(i + 2) for i, g in enumerate(gains))
    ideal_gains = sorted(gains, reverse=True)
    idcg = sum(g / np.log2(i + 2) for i, g in enumerate(ideal_gains))
    return dcg / idcg if idcg > 0 else 0.0

def compute_all_weighted_metrics(
    ranked_chunk_ids: list[str],
    weights_lookup: dict[tuple[str, str], float],
    gt_article_ids: set[str]
) -> dict:
    """Compute all weighted metrics at standard cutoffs."""
    return {
        'w_recall@10': weighted_recall_at_k(ranked_chunk_ids, weights_lookup, gt_article_ids, 10),
        'w_recall@100': weighted_recall_at_k(ranked_chunk_ids, weights_lookup, gt_article_ids, 100),
        'w_precision@10': weighted_precision_at_k(ranked_chunk_ids, weights_lookup, gt_article_ids, 10),
        'w_mrr@10': weighted_mrr(ranked_chunk_ids, weights_lookup, gt_article_ids, 10),
        'w_ndcg@10': weighted_ndcg_at_k(ranked_chunk_ids, weights_lookup, gt_article_ids, 10),
    }
```

### 6.1 When to Apply Which Metrics

| Retrieval Unit | Metrics Applied | Rationale |
|---|---|---|
| Arm 1 (chunks) | Binary + Weighted | Chunks give partial article coverage; both perspectives reported |
| Arm 2 (articles) | Binary only | Articles retrieved as whole units; weighted = binary when coverage is 100% |

For cross-arm comparison tables, Arm 1 rows show both binary and weighted columns. Arm 2 rows show binary only (weighted columns are identical, can be filled for completeness).

## 7. AUTONOMOUS EVALUATION (autonomous_eval.py)

Used when no ground truth annotations are available.

```python
def run_autonomous_evaluation(
    results: dict[str, list[RetrievalResult]],
    llm_client,
    config: dict
) -> dict:
    """
    Score all retrieval systems using reference-free metrics.
    Requires a generation step: for each query, generate an answer
    from the retrieved context, then score the answer.
    """
    scores = {}
    for method_name, method_results in results.items():
        method_scores = []
        for result in method_results:
            # Generate answer from retrieved context
            context = "\n\n".join(item['text'] for item in result.ranked_items[:10])
            answer = llm_client.generate(
                prompt=f"Based on the following legal articles:\n{context}\n\n"
                       f"Answer this question: {result.query_text}"
            )
            
            # Score with autonomous metrics
            score = {
                'ragas': compute_ragas_scores(result.query_text, answer.text, context),
                'ragchecker': compute_ragchecker_scores(answer.text, context),
                'geval': compute_geval_scores(result.query_text, answer.text, context, llm_client),
            }
            method_scores.append(score)
        
        scores[method_name] = aggregate_scores(method_scores)
    
    return scores
```

### RAGAS Dimensions (reference-free only):
- **Faithfulness:** Is the answer consistent with the retrieved context?
- **Answer Relevance:** Does the answer address the question?
- **Context Precision:** Are relevant articles ranked first?
- **Context Recall (unsupervised proxy):** Is sufficient context present?

### G-Eval Custom Rubrics (1-5 scale):
- **Completeness:** Does the answer address all dimensions of the question?
- **Factual Accuracy:** No hallucinated article numbers, dates, or provision references?
- **Coherence:** Internally consistent, no conflation of distinct provisions?

## 8. COST TRACKING (cost_tracker.py)

```python
@dataclass
class CostRecord:
    method: str
    query_id: str
    llm_calls: int
    tokens_in: int
    tokens_out: int
    latency_ms: float
    
def aggregate_costs(records: list[CostRecord]) -> dict:
    """Compute per-method cost statistics."""
    by_method = defaultdict(list)
    for r in records:
        by_method[r.method].append(r)
    
    stats = {}
    for method, records in by_method.items():
        stats[method] = {
            'avg_llm_calls': np.mean([r.llm_calls for r in records]),
            'avg_tokens': np.mean([r.tokens_in + r.tokens_out for r in records]),
            'avg_latency_ms': np.mean([r.latency_ms for r in records]),
            'median_latency_ms': np.median([r.latency_ms for r in records]),
            'total_cost_estimate': estimate_dollar_cost(records),
        }
    return stats
```

## 9. COMPARISON AND VISUALIZATION (comparator.py)

```python
def generate_comparison_report(
    all_results: dict[str, list[RetrievalResult]],
    ground_truth: dict | None,
    config: dict
) -> EvalReport:
    """
    Master comparison across all methods.
    Outputs:
    - Main results table (method × metric)
    - Ablation delta tables
    - Stratified analysis (if ground truth available)
    - Cost-performance tradeoff plot
    - Statistical significance tests
    """
```

### Output Artifacts:

1. **Main results table** — all methods × all metrics
2. **Ablation deltas** — Arm1 vs 2A-raw, 2A-raw vs 2A-full, 2A-full vs 2B
3. **Stratified breakdown** — metrics per query stratum (single-article, multi-article, cross-reference, etc.)
4. **Cost-performance plot** — Recall@100 vs latency per method
5. **Statistical tests** — paired t-test or Wilcoxon signed-rank for key comparisons, bootstrap CIs

### Statistical Testing:

```python
from scipy import stats

def significance_test(scores_a: list[float], scores_b: list[float]) -> dict:
    """Paired comparison between two methods across queries."""
    t_stat, t_pval = stats.ttest_rel(scores_a, scores_b)
    w_stat, w_pval = stats.wilcoxon(scores_a, scores_b, alternative='two-sided')
    diff = np.array(scores_b) - np.array(scores_a)
    
    return {
        'mean_diff': np.mean(diff),
        'std_diff': np.std(diff),
        'cohens_d': np.mean(diff) / np.std(diff) if np.std(diff) > 0 else 0,
        'paired_t_pval': t_pval,
        'wilcoxon_pval': w_pval,
        'significant_at_005': min(t_pval, w_pval) < 0.05,
    }
```

## 10. INTERFACE FOR ORCHESTRATOR

The real `comparator.evaluate(...)` is **cache-aware**: alongside the legacy in-memory
`chunks=`/`articles=` path, it accepts a keyword-only cache path
(`use_cache=True, doc_id=…, arm1_config_hash=…, t07_cache_root=…, t03_cache_root=…,
db_conn=…, pdf_filename=…`) that reads/builds chunk-bsard weights via the shared PDF cache.
See the README **Public API** for the full call. A sibling `evaluate_partial_views(...)`
runs strict/lenient PARTIAL regimes. (The simplified signature below is illustrative of the
binary + weighted + cost flow only.)

```python
def evaluate(
    all_results: dict[str, list[RetrievalResult]],
    ground_truth: dict | None,
    config: dict,
    *,                                 # plus cache kwargs OR legacy chunks=/articles=
    # use_cache, doc_id, arm1_config_hash, t07_cache_root, t03_cache_root, db_conn, pdf_filename
) -> EvalReport:
    """
    Full evaluation pipeline (illustrative).
    """
    report = EvalReport()
    
    if ground_truth:
        # Binary metrics for all methods
        for method, results in all_results.items():
            report.binary_metrics[method] = compute_binary_metrics(results, ground_truth)
        
        # Weighted metrics for Arm 1 only
        if chunks and articles and any('arm1' in m for m in all_results):
            weights = weight_precomputer.precompute_weights(chunks, articles, config)
            weights_lookup = weight_precomputer.weights_to_lookup(weights)
            for method, results in all_results.items():
                if 'arm1' in method:
                    report.weighted_metrics[method] = compute_weighted_metrics(
                        results, ground_truth, weights_lookup)
        
        # Statistical tests
        report.significance = run_significance_tests(report.binary_metrics)
        
        # Stratified analysis
        report.stratified = run_stratified_analysis(all_results, ground_truth)
    
    # Autonomous evaluation (always, but especially when no ground truth)
    if config['evaluation'].get('use_autonomous_eval', False):
        report.autonomous = run_autonomous_evaluation(all_results, llm_client, config)
    
    # Cost analysis
    report.costs = cost_tracker.aggregate_costs_from_results(all_results)
    
    return report
```
