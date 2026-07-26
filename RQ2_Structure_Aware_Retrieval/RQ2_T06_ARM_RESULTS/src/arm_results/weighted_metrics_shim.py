"""Single-article content-coverage, factored out of T07's weighted_metrics.

``coverage_for_article`` is exactly the per-article inner term of T07's
``evaluation.weighted_metrics.weighted_recall_at_k`` (sum of chunk->article
token-overlap weights over a set of chunks, capped at 1.0). Factored here so the
Arm-1 deep dive can read the *per-gold-article* coverage distribution (Group B),
not just the question-averaged weighted recall. Weight semantics are owned by T07
(``w(c,B) = |tokens of B in c| / |tokens of B|``); we only sum the cached values.
"""

from __future__ import annotations


def coverage_for_article(
    ranked_chunk_ids: list[str],
    weights_lookup: dict[tuple[str, int], float],
    bsard_id: int,
) -> float:
    """Fraction of gold article ``bsard_id`` covered by ``ranked_chunk_ids``.

    Capped at 1.0, matching ``weighted_recall_at_k``'s per-article cap.
    """
    w_sum = sum(weights_lookup.get((c, bsard_id), 0.0) for c in ranked_chunk_ids)
    return min(1.0, w_sum)
