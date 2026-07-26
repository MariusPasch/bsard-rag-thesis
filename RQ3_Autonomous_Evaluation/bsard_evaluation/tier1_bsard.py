"""
evaluation/tier1_bsard.py
─────────────────────────
Tier 1 — BSARD paper metrics (exact replication).

Reproduces the evaluation protocol of:
    Bench, S. et al. (2022). BSARD: A Belgian Statutory Article Retrieval
    Dataset. ACM SIGIR 2022.

Primary metric : Recall@100
Secondary      : Recall@1, Recall@5, Recall@10
Comparability  : MRR@100

These values are kept fixed so results are directly comparable to the
published CamemBERT bi-encoder baseline (R@100 ≈ 0.62 for the best
mE5-large model in the thesis experiments).

Input format (TREC standard)
────────────────────────────
qrels : dict[query_id, dict[doc_id, relevance_grade]]
    Ground truth.  relevance_grade > 0 means relevant.
    BSARD uses binary labels (0 / 1).

run   : dict[query_id, dict[doc_id, score]]
    Retrieval results.  Higher score = ranked higher.
    Ties are broken arbitrarily (stable sort).
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional

# ── BSARD paper constants ──────────────────────────────────────────────────────

BSARD_K_VALUES: List[int] = [1, 5, 10, 100]
BSARD_PRIMARY_K: int = 100

# ── Core metric functions ──────────────────────────────────────────────────────


def recall_at_k(
    qrels: Dict[str, Dict[str, int]],
    run: Dict[str, Dict[str, float]],
    k: int,
) -> float:
    """
    Recall@k — fraction of relevant articles retrieved in top-k.

    R@k = |{relevant ∩ top-k}| / |{all relevant}|

    Macro-averaged across all queries that have at least one relevant article.
    Queries with no relevant articles in qrels are skipped.

    Notes
    -----
    - Order-unaware: a relevant hit at rank 1 = rank 100.
    - BSARD primary metric at k=100.
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


def mrr_at_k(
    qrels: Dict[str, Dict[str, int]],
    run: Dict[str, Dict[str, float]],
    k: int,
) -> float:
    """
    MRR@k — mean reciprocal rank of the first relevant article.

    MRR = (1/|Q|) · Σ_q  1 / rank_q(first relevant)

    If no relevant article appears in the top-k for a query,
    the reciprocal rank is 0 for that query.

    Notes
    -----
    - Only scores the FIRST relevant hit; ignores all others.
    - Best suited to single-answer queries.
    - For multi-answer queries, prefer MAP@k.
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


# ── Main Tier 1 entry point ────────────────────────────────────────────────────


def evaluate_tier1(
    qrels: Dict[str, Dict[str, int]],
    run: Dict[str, Dict[str, float]],
    k_values: Optional[List[int]] = None,
) -> Dict[str, float]:
    """
    Run all BSARD paper metrics and return a flat results dict.

    R@100 and MRR@100 are always included regardless of k_values
    to guarantee benchmark comparability.

    Parameters
    ----------
    qrels     : TREC qrels dict
    run       : TREC run dict
    k_values  : list of k cutoffs to evaluate (defaults to BSARD paper values)

    Returns
    -------
    dict, e.g.:
        {
            "T1/R@1":     0.210,
            "T1/R@5":     0.465,
            "T1/R@10":    0.550,
            "T1/R@100":   0.621,   ← primary BSARD metric
            "T1/MRR@100": 0.312,
        }

    Keys are prefixed with "T1/" so they are unambiguous when merged
    with Tier 0 / Tier 2 / Tier 3 results.
    """
    ks = sorted(set((k_values or BSARD_K_VALUES) + [BSARD_PRIMARY_K]))
    results: Dict[str, float] = {}

    for k in ks:
        results[f"T1/R@{k}"] = recall_at_k(qrels, run, k)

    # Always add MRR@100 for paper comparability
    results["T1/MRR@100"] = mrr_at_k(qrels, run, 100)

    return results


# ── Internal helpers ───────────────────────────────────────────────────────────


def _top_k(doc_scores: Dict[str, float], k: int) -> List[str]:
    """Return doc_ids ranked by score descending, limited to top-k."""
    return [
        doc_id
        for doc_id, _ in sorted(doc_scores.items(), key=lambda x: -x[1])[:k]
    ]

def per_query_recall(
    qrels: dict[str, dict[str, int]],
    run:   dict[str, dict[str, float]],
    k: int,
) -> list[float]:
    """
    Return per-query Recall@k as a list of floats (one per query in qrels).
    Used for paired significance testing.
    """
    scores = []
    for qid, relevant in qrels.items():
        if not relevant:
            scores.append(0.0)
            continue
        ranked = sorted(run.get(qid, {}).items(), key=lambda x: -x[1])
        top_k  = {doc_id for doc_id, _ in ranked[:k]}
        hits   = sum(1 for doc_id in relevant if doc_id in top_k)
        scores.append(hits / len(relevant))
    return scores
