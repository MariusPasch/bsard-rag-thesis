"""Apply hard filters and soft boosts to fused retrieval results.

Hard filter: drop items whose ``regulatory_status`` is non-Effective; when the
query has a high-confidence ``jurisdiction`` signal (from the regex method),
drop items whose metadata jurisdiction disagrees (items without metadata are
kept — defensive default).

Soft boosts (multiplicative):
  - Term match           (defined_terms ∩ matched_terms)        × term_match
  - UsedIn membership    (id ∈ signals.term_node_ids)            × used_in
  - Jurisdiction match   (metadata.jurisdiction == signal)       × jurisdiction
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

# Hoisted from inside `apply_filters_and_boosts` (review #15-B1).
from shared.faiss_store import SearchResult as _SR

if TYPE_CHECKING:
    from shared.faiss_store import SearchResult  # noqa: F401 — kept for callers

    from .query_extractor import QuerySignals

logger = logging.getLogger(__name__)


@dataclass
class BoostConfig:
    term_match: float = 1.3
    used_in: float = 1.5
    jurisdiction: float = 1.1
    hard_filter_jurisdiction: bool = False  # only flip on with high-precision signals
    drop_non_effective: bool = True


def normalise_boost(bc: "BoostConfig | dict | None") -> dict:
    """Canonicalise a BoostConfig into the dict shape stored in chunking_params.

    Single source of truth — both `arm2_metadata.retriever._chunking_params`
    and `scripts/precompute_t04_indices.precompute_one` need this exact shape
    so cache hashes line up across callers (review #15-B11 dedup).
    """
    if bc is None:
        bc = BoostConfig()
    elif isinstance(bc, dict):
        bc = BoostConfig(**bc)
    return {
        "term_match": bc.term_match,
        "used_in": bc.used_in,
        "jurisdiction": bc.jurisdiction,
        "hard_filter_jurisdiction": bool(bc.hard_filter_jurisdiction),
        "drop_non_effective": bool(bc.drop_non_effective),
    }


def apply_filters_and_boosts(
    results: list["SearchResult"],
    signals: "QuerySignals",
    boost_config: BoostConfig | dict | None = None,
) -> list["SearchResult"]:
    """Filter + re-score ``results`` according to ``signals`` and ``boost_config``.

    The input list is left untouched; a new list is returned (sorted by the
    boosted score). Items with no ``metadata`` survive every filter — we never
    drop on absence of evidence.
    """
    if boost_config is None:
        boost_config = BoostConfig()
    elif isinstance(boost_config, dict):
        boost_config = BoostConfig(**boost_config)

    filtered: list["SearchResult"] = []
    for r in results:
        meta = r.metadata or {}
        if boost_config.drop_non_effective:
            status = (meta.get("regulatory_status") or "Effective")
            doc_status = meta.get("document_status") or status
            if status and status != "Effective":
                continue
            if doc_status and doc_status != "Effective":
                continue
        if (
            boost_config.hard_filter_jurisdiction
            and signals.jurisdiction
            and meta.get("jurisdiction")
            and meta["jurisdiction"].lower() != signals.jurisdiction.lower()
        ):
            continue
        filtered.append(r)

    matched_terms = set(signals.matched_terms)
    term_node_ids = signals.term_node_ids

    boosted: list["SearchResult"] = []
    for r in filtered:
        meta = r.metadata or {}
        score = r.score
        meta_terms = set(meta.get("defined_terms") or [])
        if matched_terms and meta_terms & matched_terms:
            score *= boost_config.term_match
        if term_node_ids and r.id in term_node_ids:
            score *= boost_config.used_in
        if (
            signals.jurisdiction
            and meta.get("jurisdiction")
            and meta["jurisdiction"].lower() == signals.jurisdiction.lower()
        ):
            score *= boost_config.jurisdiction
        # SearchResult dataclass preserves metadata so downstream consumers (viewer,
        # T07 adapter) see the same fields after boosting.
        boosted.append(_SR(id=r.id, score=score, metadata=meta))

    return sorted(boosted, key=lambda r: r.score, reverse=True)
