"""Weighted partial-relevance IR metrics for Arm 1 (chunk retrieval).

Chunk gain = fractional overlap with ground-truth BSARD articles, so a chunk
covering half a BSARD article contributes 0.5 instead of binary 0 or 1.

Keys throughout: weights_lookup is keyed by (chunk_id, bsard_id);
gt_bsard_ids is a set of int bsard_ids.
"""

from __future__ import annotations

from typing import Iterable, Literal

import numpy as np


def weighted_recall_at_k(
    ranked_chunk_ids: list[str],
    weights_lookup: dict[tuple[str, int], float],
    gt_bsard_ids: set[int],
    k: int,
) -> float:
    """For each relevant bsard article, sum overlap weights of top-k chunks,
    capped at 1.0. Average across relevant bsard articles.
    """
    if not gt_bsard_ids:
        return 0.0
    total = 0.0
    for bsard_id in gt_bsard_ids:
        w_sum = sum(
            weights_lookup.get((c, bsard_id), 0.0)
            for c in ranked_chunk_ids[:k]
        )
        total += min(1.0, w_sum)
    return total / len(gt_bsard_ids)


def weighted_precision_at_k(
    ranked_chunk_ids: list[str],
    weights_lookup: dict[tuple[str, int], float],
    gt_bsard_ids: set[int],
    k: int,
) -> float:
    """Sum of relevant coverage in top-k, divided by k."""
    total_w = sum(
        sum(weights_lookup.get((c, b), 0.0) for b in gt_bsard_ids)
        for c in ranked_chunk_ids[:k]
    )
    return total_w / k if k > 0 else 0.0


def weighted_mrr(
    ranked_chunk_ids: list[str],
    weights_lookup: dict[tuple[str, int], float],
    gt_bsard_ids: set[int],
    k: int,
) -> float:
    """Effective rank of each relevant bsard = rank of its first overlapping chunk."""
    if not gt_bsard_ids:
        return 0.0
    total = 0.0
    for bsard_id in gt_bsard_ids:
        for rank, chunk_id in enumerate(ranked_chunk_ids[:k]):
            if weights_lookup.get((chunk_id, bsard_id), 0.0) > 0:
                total += 1.0 / (rank + 1)
                break
    return total / len(gt_bsard_ids)


def weighted_ndcg_at_k(
    ranked_chunk_ids: list[str],
    weights_lookup: dict[tuple[str, int], float],
    gt_bsard_ids: set[int],
    k: int,
) -> float:
    """Weighted nDCG: chunk gain = sum of overlap weights with all relevant bsard articles."""
    gains = [
        sum(weights_lookup.get((c, b), 0.0) for b in gt_bsard_ids)
        for c in ranked_chunk_ids[:k]
    ]
    dcg = sum(g / np.log2(i + 2) for i, g in enumerate(gains))
    ideal_gains = sorted(gains, reverse=True)
    idcg = sum(g / np.log2(i + 2) for i, g in enumerate(ideal_gains))
    return dcg / idcg if idcg > 0 else 0.0


def score_chunks_for_question(
    gt_bsard_ids: set[int],
    weights_lookup: dict[tuple[str, int], float],
    chunk_ids: Iterable[str],
    mode: Literal["strict", "lenient"] = "strict",
) -> dict[str, float]:
    """Per-chunk relevance score for a single question.

    strict
        Σ_{a ∈ gt} weights_lookup[(c, a)] — fractional coverage in [0, |gt|].
        Same gain function used by ``weighted_recall_at_k`` /
        ``weighted_ndcg_at_k`` (no per-chunk capping).
    lenient
        1.0 if any (c, a) with a ∈ gt has weight > 0, else 0.0. Equivalent to
        ``chunk.bsard_ids ∩ gt ≠ ∅`` (binary contact).

    Pure derivation from the cached weight table — never materialised to disk
    (CACHE_DESIGN.md §9).
    """
    if mode == "strict":
        return {
            c: float(sum(weights_lookup.get((c, a), 0.0) for a in gt_bsard_ids))
            for c in chunk_ids
        }
    if mode == "lenient":
        return {
            c: 1.0 if any(weights_lookup.get((c, a), 0.0) > 0 for a in gt_bsard_ids) else 0.0
            for c in chunk_ids
        }
    raise ValueError(f"Unknown mode={mode!r} (expected 'strict' or 'lenient')")
