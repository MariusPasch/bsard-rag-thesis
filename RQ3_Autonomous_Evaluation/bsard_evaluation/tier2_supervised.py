"""
evaluation/tier2_supervised.py
───────────────────────────────
Tier 2 — Full supervised retrieval evaluation metrics.

Three panels matching retrieval_evaluation_metrics.md:

  Panel 1 — Rank-unaware  : Precision@K, Recall@K, F1@K, Hit Rate@K
  Panel 2 — Rank-aware    : MRR@K, MAP@K, NDCG@K
  Panel 3 — Set-utility   : RA-nWG@K, N-Recall4+@K   (graded qrels only)
                            IDPrecision@K, IDRecall@K  (binary qrels, always)

Input format (TREC standard)
────────────────────────────
qrels : dict[query_id, dict[doc_id, relevance_grade]]
    Binary labels (0/1) suffice for Panels 1 and 2.
    Graded labels (1–5 integer scale) are required for Panel 3.
    If Panel 3 is requested but all labels are binary, it is skipped
    with a warning rather than raising an error.

run   : dict[query_id, dict[doc_id, score]]
    Higher score = ranked higher.

All results are prefixed "T2/" and include the k value, e.g.
    "T2/P1/Precision@10", "T2/P2/NDCG@10", "T2/P3/RA-nWG@10"
"""

from __future__ import annotations

import math
import warnings
from typing import Dict, List, Optional

import numpy as np

# ════════════════════════════════════════════════════════════════════════
# Panel 1 — Rank-unaware metrics
# ════════════════════════════════════════════════════════════════════════


def precision_at_k(
    qrels: Dict[str, Dict[str, int]],
    run: Dict[str, Dict[str, float]],
    k: int,
) -> float:
    """
    Precision@K — fraction of retrieved documents that are relevant.

    P@K = |{relevant ∩ top-K}| / K

    Measures context-window purity: a low P@K means the generator
    receives noisy passages that may introduce hallucinations.
    Uses binary relevance (grade > 0 = relevant).
    """
    scores: List[float] = []
    for qid, relevant in qrels.items():
        rel_docs = {d for d, r in relevant.items() if r > 0}
        if not rel_docs:
            continue
        ranked = _top_k(run.get(qid, {}), k)
        hits = sum(1 for d in ranked if d in rel_docs)
        scores.append(hits / k if k > 0 else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def recall_at_k(
    qrels: Dict[str, Dict[str, int]],
    run: Dict[str, Dict[str, float]],
    k: int,
) -> float:
    """
    Recall@K — fraction of all relevant documents retrieved in top-K.

    R@K = |{relevant ∩ top-K}| / |{all relevant}|

    Measures evidence coverage.  Critical for multi-hop queries where
    the generator needs all relevant passages to answer correctly.
    Uses binary relevance.
    """
    scores: List[float] = []
    for qid, relevant in qrels.items():
        rel_docs = {d for d, r in relevant.items() if r > 0}
        if not rel_docs:
            continue
        ranked = _top_k(run.get(qid, {}), k)
        hits = len(rel_docs & set(ranked))
        scores.append(hits / len(rel_docs))
    return float(np.mean(scores)) if scores else 0.0


def f1_at_k(
    qrels: Dict[str, Dict[str, int]],
    run: Dict[str, Dict[str, float]],
    k: int,
) -> float:
    """
    F1@K — harmonic mean of Precision@K and Recall@K.

    F1@K = 2 · P@K · R@K / (P@K + R@K)

    Single balanced summary of set quality.  Penalises systems that
    excel at only precision or only recall.  Uses binary relevance.
    """
    p = precision_at_k(qrels, run, k)
    r = recall_at_k(qrels, run, k)
    denom = p + r
    return (2 * p * r / denom) if denom > 0 else 0.0


def hit_rate_at_k(
    qrels: Dict[str, Dict[str, int]],
    run: Dict[str, Dict[str, float]],
    k: int,
) -> float:
    """
    Hit Rate@K — fraction of queries with at least one relevant doc in top-K.

    HR@K = (1/|Q|) · Σ_q  𝟙[any relevant doc in top-K for q]

    Coarsest sanity check: did the retriever find *anything* useful?
    A low Hit Rate means all other metrics will necessarily suffer.
    Uses binary relevance.
    """
    hits = 0
    total = 0
    for qid, relevant in qrels.items():
        rel_docs = {d for d, r in relevant.items() if r > 0}
        if not rel_docs:
            continue
        total += 1
        ranked = _top_k(run.get(qid, {}), k)
        if any(d in rel_docs for d in ranked):
            hits += 1
    return hits / total if total > 0 else 0.0


# ════════════════════════════════════════════════════════════════════════
# Panel 2 — Rank-aware metrics
# ════════════════════════════════════════════════════════════════════════


def mrr_at_k(
    qrels: Dict[str, Dict[str, int]],
    run: Dict[str, Dict[str, float]],
    k: int,
) -> float:
    """
    MRR@K — mean reciprocal rank of the first relevant document.

    MRR@K = (1/|Q|) · Σ_q  1 / rank_q(first relevant in top-K)

    If no relevant document appears in top-K, reciprocal rank = 0.

    Measures how quickly the retriever surfaces at least one relevant
    passage.  Best suited to single-answer queries or agentic systems
    that read the top result first (T4.0, T4.1 CRAG, T4.2 ReAct).
    Only scores the FIRST relevant hit; ignores all others.
    Uses binary relevance.
    """
    scores: List[float] = []
    for qid, relevant in qrels.items():
        rel_docs = {d for d, r in relevant.items() if r > 0}
        if not rel_docs:
            continue
        ranked = _top_k(run.get(qid, {}), k)
        rr = 0.0
        for rank, doc_id in enumerate(ranked, start=1):
            if doc_id in rel_docs:
                rr = 1.0 / rank
                break
        scores.append(rr)
    return float(np.mean(scores)) if scores else 0.0


def map_at_k(
    qrels: Dict[str, Dict[str, int]],
    run: Dict[str, Dict[str, float]],
    k: int,
) -> float:
    """
    MAP@K — mean average precision across all queries.

    AP@K_q = (1 / min(R_q, K)) · Σ_{j=1}^{K}  P@j · 𝟙[doc_j is relevant]
    MAP@K  = (1/|Q|) · Σ_q  AP@K_q

    Evaluates ranking quality for ALL relevant documents, not just the
    first.  Complements MRR: while MRR only cares about the first hit,
    MAP rewards systems that rank every relevant passage highly.
    Essential for multi-document BSARD queries.
    Uses binary relevance.
    """
    scores: List[float] = []
    for qid, relevant in qrels.items():
        rel_docs = {d for d, r in relevant.items() if r > 0}
        if not rel_docs:
            continue
        ranked = _top_k(run.get(qid, {}), k)
        hits = 0
        precision_sum = 0.0
        for rank, doc_id in enumerate(ranked, start=1):
            if doc_id in rel_docs:
                hits += 1
                precision_sum += hits / rank
        num_relevant_in_k = min(len(rel_docs), k)
        ap = precision_sum / num_relevant_in_k if num_relevant_in_k > 0 else 0.0
        scores.append(ap)
    return float(np.mean(scores)) if scores else 0.0


def ndcg_at_k(
    qrels: Dict[str, Dict[str, int]],
    run: Dict[str, Dict[str, float]],
    k: int,
) -> float:
    """
    NDCG@K — normalised discounted cumulative gain.

    DCG@K  = Σ_{i=1}^{K}  rel(i) / log₂(i+1)
    IDCG@K = DCG of ideal ranking (all relevant docs placed first)
    NDCG@K = DCG@K / IDCG@K

    Applies a logarithmic position discount, reflecting that top-ranked
    results contribute more to generation quality.  The standard metric
    on the MTEB Leaderboard for retrieval evaluation.

    Supports graded relevance: if qrels contain grades > 1, those grades
    are used directly (giving partial credit to marginally relevant docs).
    With binary labels (0/1) it reduces to position-discounted recall.

    Range [0, 1] where 1 = perfect ranking.
    """
    scores: List[float] = []
    for qid, relevant in qrels.items():
        rel_docs = {d: r for d, r in relevant.items() if r > 0}
        if not rel_docs:
            continue
        ranked = _top_k(run.get(qid, {}), k)

        # DCG
        dcg = sum(
            rel_docs.get(doc_id, 0) / math.log2(rank + 1)
            for rank, doc_id in enumerate(ranked, start=1)
        )

        # IDCG — ideal ordering of all grades in the qrels for this query
        ideal_grades = sorted(rel_docs.values(), reverse=True)[:k]
        idcg = sum(
            grade / math.log2(rank + 1)
            for rank, grade in enumerate(ideal_grades, start=1)
        )

        scores.append(dcg / idcg if idcg > 0 else 0.0)
    return float(np.mean(scores)) if scores else 0.0


# ════════════════════════════════════════════════════════════════════════
# Panel 3 — Set-utility-aware metrics (graded qrels required)
# ════════════════════════════════════════════════════════════════════════


def _check_graded(qrels: Dict[str, Dict[str, int]]) -> bool:
    """Return True if any document has a grade > 1 (i.e. not purely binary)."""
    return any(
        grade > 1
        for rel in qrels.values()
        for grade in rel.values()
    )


def _rarity_weight(grade: int, grade_counts: Dict[int, int], total: int) -> float:
    """
    Compute per-query rarity weight for a given grade level.

    Rare high-utility evidence receives a higher weight, aligning
    evaluation with real-world impact on generation quality.

    w_g = max_grade_weight / n_g  (capped at 5.0 to prevent extreme values)

    where n_g is the number of documents with exactly grade g for this query.

    References
    ----------
    Trappolini et al. (2025). Practical RAG Evaluation: A Rarity-Aware
    Set-Based Metric. arXiv:2511.09545.
    """
    n_g = grade_counts.get(grade, 1)
    # Base weight scales with grade (grade 5 > grade 4 > ... > grade 1)
    grade_scale = grade / 5.0
    rarity = total / max(n_g, 1)
    return min(grade_scale * rarity, 5.0)


def ra_nwg_at_k(
    qrels: Dict[str, Dict[str, int]],
    run: Dict[str, Dict[str, float]],
    k: int,
) -> Optional[float]:
    """
    RA-nWG@K — Rarity-Aware Normalised Weighted Gain.

    Measures how close the actual retrieved set is to the oracle best-K
    set, with per-query rarity weighting that amplifies the importance
    of scarce high-quality evidence.

    RA-nWG@K = G_actual(K) / G_oracle(K)

    where G(set) = Σ_{d ∈ set}  w_{g(d)} · g(d)

    Key differences from NDCG@K:
    - Order-agnostic: no positional discount (LLMs process passages as a
      flat set, not sequentially).
    - Per-query normalised: accounts for evidence landscape heterogeneity.
    - Rarity-weighted: missing the one decisive passage for a hard query
      is penalised more than missing one of many easy passages.

    Returns None if qrels do not contain graded labels.

    References
    ----------
    Trappolini et al. (2025). Practical RAG Evaluation: A Rarity-Aware
    Set-Based Metric. arXiv:2511.09545.
    """
    if not _check_graded(qrels):
        return None

    scores: List[float] = []
    for qid, relevant in qrels.items():
        graded = {d: r for d, r in relevant.items() if r > 0}
        if not graded:
            continue

        total_docs = len(graded)
        grade_counts: Dict[int, int] = {}
        for g in graded.values():
            grade_counts[g] = grade_counts.get(g, 0) + 1

        # Compute weighted gain for the actual retrieved top-K set
        ranked = _top_k(run.get(qid, {}), k)
        actual_gain = sum(
            _rarity_weight(graded[d], grade_counts, total_docs) * graded[d]
            for d in ranked
            if d in graded
        )

        # Oracle: take the top-K docs by (rarity_weight * grade) descending
        oracle_candidates = [
            (d, _rarity_weight(g, grade_counts, total_docs) * g)
            for d, g in graded.items()
        ]
        oracle_candidates.sort(key=lambda x: -x[1])
        oracle_gain = sum(score for _, score in oracle_candidates[:k])

        scores.append(actual_gain / oracle_gain if oracle_gain > 0 else 0.0)

    return float(np.mean(scores)) if scores else None


def n_recall4plus_at_k(
    qrels: Dict[str, Dict[str, int]],
    run: Dict[str, Dict[str, float]],
    k: int,
) -> Optional[float]:
    """
    N-Recall4+@K — normalised recall of high-utility passages (grade ≥ 4).

    N-Recall4+@K = |{grade ≥ 4 docs in top-K}| / min(K, R_{4+})

    where R_{4+} = total grade-≥-4 passages available for the query.

    Measures coverage of decisive evidence specifically — the passages
    most likely to enable correct generation.  Reported alongside
    RA-nWG@K for a complete picture of set utility vs. evidence coverage.

    Returns None if qrels do not contain graded labels (grade ≥ 4).
    """
    if not _check_graded(qrels):
        return None

    has_high_grade = any(
        r >= 4
        for rel in qrels.values()
        for r in rel.values()
    )
    if not has_high_grade:
        return None

    scores: List[float] = []
    for qid, relevant in qrels.items():
        high_docs = {d for d, r in relevant.items() if r >= 4}
        if not high_docs:
            continue
        ranked = _top_k(run.get(qid, {}), k)
        hits = sum(1 for d in ranked if d in high_docs)
        denom = min(k, len(high_docs))
        scores.append(hits / denom if denom > 0 else 0.0)

    return float(np.mean(scores)) if scores else None


# ════════════════════════════════════════════════════════════════════════
# Panel 3 — ID-based metrics (binary qrels, no graded labels required)
# ════════════════════════════════════════════════════════════════════════


def id_based_context_precision(
    qrels: Dict[str, Dict[str, int]],
    run: Dict[str, Dict[str, float]],
    k: int,
) -> float:
    """
    IDPrecision@K — average precision at K using article-ID matching.

    For each query, computes the average of precision@i values at every
    rank i (1..K) where the document at position i is relevant, divided
    by min(K, |relevant|).  This is the AP@K formulation, equivalent to
    MAP@K from Panel 2 when run on the same supervised qrels.

    The metric's purpose becomes distinct from MAP@K when called with
    UMBRELA-produced graded qrels (via the T2-umbrela/ harness bridge):
    in that context it measures how well the retrieval ranking aligns with
    the autonomous relevance grades rather than BSARD ground truth.

    Uses binary relevance (grade > 0 = relevant).
    """
    scores: List[float] = []
    for qid, relevant in qrels.items():
        rel_docs = {d for d, r in relevant.items() if r > 0}
        if not rel_docs:
            continue
        ranked = _top_k(run.get(qid, {}), k)
        hits = 0
        precision_sum = 0.0
        for rank, doc_id in enumerate(ranked, start=1):
            if doc_id in rel_docs:
                hits += 1
                precision_sum += hits / rank
        num_relevant_in_k = min(len(rel_docs), k)
        ap = precision_sum / num_relevant_in_k if num_relevant_in_k > 0 else 0.0
        scores.append(ap)
    return float(np.mean(scores)) if scores else 0.0


def id_based_context_recall(
    qrels: Dict[str, Dict[str, int]],
    run: Dict[str, Dict[str, float]],
    k: int,
) -> float:
    """
    IDRecall@K — set-membership recall using article-ID matching.

    R@K = |{relevant article IDs} ∩ {top-K retrieved IDs}| / |{all relevant}|

    Numerically identical to Recall@K from Panel 1 when applied to the same
    supervised qrels.  The distinction arises in two situations:

    1. When called with UMBRELA qrels (T2-umbrela/ bridge): measures what
       fraction of UMBRELA-judged relevant articles appear in the top-K,
       providing the autonomous counterpart to supervised Recall@K.

    2. Score tie-breaking: if multiple documents share the same retrieval
       score, _top_k resolves ties by dict insertion order, which is
       arbitrary.  Comparing IDRecall@K against Recall@K on the same run
       isolates queries where this artifact changes the result.

    Uses binary relevance (grade > 0 = relevant).
    """
    scores: List[float] = []
    for qid, relevant in qrels.items():
        rel_docs = {d for d, r in relevant.items() if r > 0}
        if not rel_docs:
            continue
        ranked = _top_k(run.get(qid, {}), k)
        hits = len(rel_docs & set(ranked))
        scores.append(hits / len(rel_docs))
    return float(np.mean(scores)) if scores else 0.0


# ════════════════════════════════════════════════════════════════════════
# Main Tier 2 entry point
# ════════════════════════════════════════════════════════════════════════


def evaluate_tier2(
    qrels: Dict[str, Dict[str, int]],
    run: Dict[str, Dict[str, float]],
    k_values: Optional[List[int]] = None,
    include_graded: bool = False,
    include_id_based: bool = True,
) -> Dict[str, float]:
    """
    Run all Tier 2 metrics across all requested k values.

    Parameters
    ----------
    qrels            : TREC qrels dict
    run              : TREC run dict
    k_values         : list of k cutoffs (use TierConfig.k_values)
    include_graded   : if True, also compute Panel 3 graded metrics
                       (RA-nWG, N-Recall4+).  Silently skips if labels
                       are binary.
    include_id_based : if True, compute Panel 3 ID-based metrics
                       (IDPrecision@K, IDRecall@K) using binary qrels.
                       Set False when article IDs are not tracked in the
                       run dict (e.g. chunk-level retrieval).

    Returns
    -------
    Flat dict with keys like:
        "T2/P1/Precision@10"
        "T2/P1/Recall@10"
        "T2/P1/F1@10"
        "T2/P1/HitRate@10"
        "T2/P2/MRR@10"
        "T2/P2/MAP@100"
        "T2/P2/NDCG@10"
        "T2/P3/RA-nWG@10"         (if include_graded and graded qrels)
        "T2/P3/N-Recall4+@10"     (if include_graded and grade ≥ 4 exists)
        "T2/P3/IDPrecision@10"    (if include_id_based)
        "T2/P3/IDRecall@10"       (if include_id_based)
    """
    ks = k_values or [5, 10, 20, 100]
    results: Dict[str, float] = {}

    for k in ks:
        # Panel 1 — rank-unaware
        results[f"T2/P1/Precision@{k}"]  = precision_at_k(qrels, run, k)
        results[f"T2/P1/Recall@{k}"]     = recall_at_k(qrels, run, k)
        results[f"T2/P1/F1@{k}"]         = f1_at_k(qrels, run, k)
        results[f"T2/P1/HitRate@{k}"]    = hit_rate_at_k(qrels, run, k)

        # Panel 2 — rank-aware
        results[f"T2/P2/MRR@{k}"]        = mrr_at_k(qrels, run, k)
        results[f"T2/P2/MAP@{k}"]        = map_at_k(qrels, run, k)
        results[f"T2/P2/NDCG@{k}"]       = ndcg_at_k(qrels, run, k)

        # Panel 3 — set-utility (graded labels required)
        if include_graded:
            raw_nwg  = ra_nwg_at_k(qrels, run, k)
            raw_nrec = n_recall4plus_at_k(qrels, run, k)
            if raw_nwg is not None:
                results[f"T2/P3/RA-nWG@{k}"]     = raw_nwg
            if raw_nrec is not None:
                results[f"T2/P3/N-Recall4+@{k}"] = raw_nrec

        # Panel 3 — ID-based (binary qrels sufficient)
        if include_id_based:
            results[f"T2/P3/IDPrecision@{k}"] = id_based_context_precision(qrels, run, k)
            results[f"T2/P3/IDRecall@{k}"]    = id_based_context_recall(qrels, run, k)

    if include_graded and not _check_graded(qrels):
        warnings.warn(
            "Panel 3 metrics (RA-nWG, N-Recall4+) were requested but all "
            "qrels labels are binary (0/1).  Panel 3 requires a 1–5 graded "
            "annotation scheme.  Panel 3 graded results omitted.",
            stacklevel=2,
        )

    return results


# ════════════════════════════════════════════════════════════════════════
# Internal helpers
# ════════════════════════════════════════════════════════════════════════


def _top_k(doc_scores: Dict[str, float], k: int) -> List[str]:
    """Return doc_ids ranked by score descending, limited to top-k."""
    return [
        doc_id
        for doc_id, _ in sorted(doc_scores.items(), key=lambda x: -x[1])[:k]
    ]
