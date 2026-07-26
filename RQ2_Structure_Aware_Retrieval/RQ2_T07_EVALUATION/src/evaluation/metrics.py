"""Binary IR metrics: Recall@k, MRR@k, nDCG@k computed against ground-truth article IDs."""

from __future__ import annotations

import numpy as np


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


def cosine_weighted_recall_at_k(
    ranked_ids: list[str],
    relevant_with_cosine: dict[int, float | None],
    k: int,
) -> float:
    """Recall@k weighted by per-article extraction_cosine.

    Cw_recall = Σ (hit * cosine) / Σ cosine over GT items with non-null cosine.
    GT items with cosine=None are dropped from both numerator and denominator —
    they're informationless under this metric. When no GT item has a usable
    cosine (all None, or denom == 0), falls back to plain binary recall@k so
    the column degrades gracefully on un-regenerated upstream data.
    """
    weights = {bid: c for bid, c in relevant_with_cosine.items() if c is not None}
    sum_w = sum(weights.values())
    if not weights or sum_w <= 0:
        return recall_at_k(ranked_ids, {str(b) for b in relevant_with_cosine}, k)

    retrieved = set(ranked_ids[:k])
    num = sum(c for bid, c in weights.items() if str(bid) in retrieved)
    return num / sum_w


def compute_all_binary_metrics(ranked_ids: list[str], relevant_ids: set[str]) -> dict[str, float]:
    """Compute all binary metrics at standard cutoffs."""
    return {
        "recall@10": recall_at_k(ranked_ids, relevant_ids, 10),
        "recall@100": recall_at_k(ranked_ids, relevant_ids, 100),
        "mrr@10": mrr_at_k(ranked_ids, relevant_ids, 10),
        "ndcg@10": ndcg_at_k(ranked_ids, relevant_ids, 10),
    }


# ---------------------------------------------------------------------------
# Effective (drift-aware) metrics — Layer 3
# ---------------------------------------------------------------------------
#
# Same definitions as recall_at_k / mrr_at_k / ndcg_at_k, but the relevant set
# is the per-PDF "effective GT" (full GT ∩ bsard_ids actually present in this
# PDF; see RQ2_T07_EVALUATION/src/evaluation/projection.py and
# RQ2_T03_ARM1_NAIVE/data/<doc_id>/drift_status.json).
#
# Returning None when the effective GT is empty distinguishes a not-evaluable
# question (no GT article reachable in this PDF) from a 0-recall outcome
# (relevant article exists but the retriever missed it). Aggregators must
# handle the None / not-evaluable case explicitly — see
# comparator._compute_effective_metrics.


def effective_recall_at_k(
    ranked_bsard_ids: list[str],
    gt_in_pdf: set[str],
    k: int,
) -> float | None:
    """Recall@k against the per-PDF effective GT, or None if not-evaluable."""
    if not gt_in_pdf:
        return None
    return recall_at_k(ranked_bsard_ids, gt_in_pdf, k)


def effective_mrr_at_k(
    ranked_bsard_ids: list[str],
    gt_in_pdf: set[str],
    k: int,
) -> float | None:
    if not gt_in_pdf:
        return None
    return mrr_at_k(ranked_bsard_ids, gt_in_pdf, k)


def effective_ndcg_at_k(
    ranked_bsard_ids: list[str],
    gt_in_pdf: set[str],
    k: int,
) -> float | None:
    if not gt_in_pdf:
        return None
    return ndcg_at_k(ranked_bsard_ids, gt_in_pdf, k)


def aggregate_binary_metrics(
    per_query: list[dict[str, float]],
) -> dict[str, float]:
    """Mean across queries for each metric."""
    if not per_query:
        return {}
    keys = per_query[0].keys()
    return {k: float(np.mean([q[k] for q in per_query])) for k in keys}
