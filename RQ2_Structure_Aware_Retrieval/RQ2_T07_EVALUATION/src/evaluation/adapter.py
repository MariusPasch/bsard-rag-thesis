"""Adapts RQ2 data structures to TREC-standard format consumed by bsard_evaluation.

TREC format:
  qrels : dict[query_id, dict[doc_id, int]]   — relevance grades (1 = relevant)
  run   : dict[query_id, dict[doc_id, float]] — retrieval scores (higher = better rank)

doc_id throughout this module is a stringified `bsard_id` (the canonical BSARD
article identifier from `maastrichtlawtech/bsard`). Ground truth files in this
repo are keyed by bsard_id.

For Arm 1 (chunks), one chunk may overlap multiple BSARD articles, so
resolution returns a list. Resolution priority for each ranked item:
  1. item["metadata"]["bsard_ids"]   — list[int], chunk-level (T03)
  2. item["metadata"]["bsard_id"]    — int, article-level (T04/T05)
  3. chunk_to_bsard[raw_id]           — explicit caller-supplied mapping
  4. Raw item ID treated as bsard_id  — last-resort fallback (warns once)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------

def to_qrels(
    ground_truth: dict[str, list[int]],
) -> dict[str, dict[str, int]]:
    """Convert {query_id: [bsard_id, ...]} → TREC qrels {query_id: {bsard_id_str: 1}}.

    bsard_ids are stringified for TREC compatibility; bsard_evaluation expects
    string doc_ids.
    """
    return {
        qid: {str(bid): 1 for bid in bids}
        for qid, bids in ground_truth.items()
    }


# ---------------------------------------------------------------------------
# Run (bsard-id-level)
# ---------------------------------------------------------------------------

def to_bsard_run(
    results: list,  # list[RetrievalResult]
    chunk_to_bsard: dict[str, int] | None = None,
) -> dict[str, dict[str, float]]:
    """Convert list[RetrievalResult] → TREC run {query_id: {bsard_id_str: score}}.

    For chunk-based results (Arm 1): bsard score = max chunk score across all
    chunks that overlap that bsard article in the ranked list.

    For article-based results (Arm 2): each item maps to one bsard_id (typically
    via metadata["bsard_id"]).
    """
    run: dict[str, dict[str, float]] = {}
    unresolved_warned = False

    for result in results:
        bsard_scores: dict[str, float] = defaultdict(lambda: float("-inf"))

        for item in result.ranked_items:
            raw_id: str = item["id"]
            score: float = float(item.get("score", 0.0))

            bsard_ids = _resolve_bsard_ids(raw_id, item, chunk_to_bsard)

            if not bsard_ids:
                if not unresolved_warned:
                    logger.warning(
                        "Could not resolve item id=%s to any bsard_id. "
                        "Provide chunk_to_bsard or store bsard_ids in metadata. "
                        "Binary metrics will be under-estimated.",
                        raw_id,
                    )
                    unresolved_warned = True
                bsard_ids = [raw_id]  # last-resort fallback

            for bsard_id in bsard_ids:
                key = str(bsard_id)
                if score > bsard_scores[key]:
                    bsard_scores[key] = score

        run[result.query_id] = {
            k: v for k, v in bsard_scores.items() if v > float("-inf")
        }

    return run


def _resolve_bsard_ids(
    raw_id: str,
    item: dict,
    chunk_to_bsard: dict[str, int] | None,
) -> list[str]:
    """Resolve a raw item ID to zero or more bsard_ids (returned as strings).

    Returns an empty list only if every resolution path fails.
    """
    metadata = item.get("metadata") or {}

    # 1. Chunks
    ids = metadata.get("bsard_ids")
    if ids:
        return [str(int(b)) for b in ids if b is not None]

    # 2. Articles
    bid = metadata.get("bsard_id")
    if bid is not None:
        return [str(int(bid))]

    # 3. Explicit caller-supplied chunk → bsard mapping
    if chunk_to_bsard and raw_id in chunk_to_bsard:
        return [str(chunk_to_bsard[raw_id])]

    return []


# ---------------------------------------------------------------------------
# Latencies
# ---------------------------------------------------------------------------

def to_latencies(results: list) -> dict[str, float]:
    """Extract {query_id: latency_ms} from RetrievalResult.cost dicts."""
    return {
        result.query_id: float((result.cost or {}).get("latency_ms", 0.0))
        for result in results
    }


# ---------------------------------------------------------------------------
# Chunk → bsard mapping utility
# ---------------------------------------------------------------------------

def build_chunk_to_bsard_from_weights(
    weights_lookup: dict[tuple[str, int], float],
) -> dict[str, int]:
    """For each chunk, return the bsard_id it most overlaps with (highest weight).

    Used as fallback when metadata["bsard_ids"] is not available.
    """
    best: dict[str, tuple[int, float]] = {}
    for (chunk_id, bsard_id), weight in weights_lookup.items():
        if chunk_id not in best or weight > best[chunk_id][1]:
            best[chunk_id] = (bsard_id, weight)
    return {cid: bid for cid, (bid, _) in best.items()}


def is_chunk_based(method_name: str) -> bool:
    """True if method_name indicates a chunk-level retriever (Arm 1)."""
    return "arm1" in method_name.lower()
