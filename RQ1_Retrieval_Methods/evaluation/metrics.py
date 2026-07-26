"""
Standard IR evaluation metrics for the BSARD retrieval experiments.

╔══════════════════════════════════════════════════════════════════╗
║  Compatibility shim.                                            ║
║                                                                  ║
║  All metric logic lives in the bsard_evaluation package, the    ║
║  sibling RQ3_Autonomous_Evaluation component.                   ║
║                                                                  ║
║  Prefer importing directly:                                      ║
║    from bsard_evaluation import EvaluationHarness                ║
║    from bsard_evaluation import per_query_recall                 ║
╚══════════════════════════════════════════════════════════════════╝

Decisions implemented:
  E-D1  Recall@k = |relevant ∩ top-k| / |relevant|  (trec_eval definition, not Hit Rate)
  E-D1  MAP included alongside Recall@k, MRR@10, NDCG@10

Public API (preserved for backward compatibility)
----------
evaluate(results, ground_truth, k_values) -> dict
per_query_recall(results, ground_truth, k)  -> list[float]
"""

from __future__ import annotations

import warnings

import numpy as np

from bsard_evaluation import EvaluationHarness, per_query_recall as _bsard_pqr
from bsard_evaluation.config import TierConfig

# Re-export the runner's legacy wrapper for per_query_recall
from evaluation.runner import (
    per_query_recall,
    _to_trec_run,
    _to_trec_qrels,
    _trec_metrics_to_legacy,
)


def evaluate(
    results: dict[int, list[int]],
    ground_truth: dict[int, list[int]],
    k_values: list[int] | None = None,
) -> dict[str, float]:
    """
    Compute aggregate IR metrics over all queries.

    DEPRECATED: Use bsard_evaluation.EvaluationHarness instead.

    Parameters
    ----------
    results      : {question_id: [article_id, ...]} — ranked list, best first
    ground_truth : {question_id: [article_id, ...]} — from questions table
    k_values     : cutoffs for Recall@k (default: [1, 5, 10, 20, 50, 100])

    Returns
    -------
    dict with keys: Recall@1…Recall@500, MRR@10, MRR@100, NDCG@10, MAP, MAP@100
    """
    warnings.warn(
        "evaluation.metrics.evaluate() is deprecated. "
        "Use bsard_evaluation.EvaluationHarness instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    if k_values is None:
        k_values = [1, 5, 10, 20, 50, 100, 200, 500]

    # Convert to TREC format
    trec_run = _to_trec_run(results)
    trec_qrels = _to_trec_qrels(ground_truth)

    # Use bsard_evaluation harness with BSARD benchmark config + custom k
    cfg = TierConfig(tiers=[1], custom_k=k_values)
    harness = EvaluationHarness(cfg)
    trec_metrics = harness.evaluate(
        qrels=trec_qrels,
        run=trec_run,
        verbose=False,
    )

    # Convert back to legacy keys
    legacy = _trec_metrics_to_legacy(trec_metrics)

    # Ensure backward-compat keys exist
    if "MAP@100" not in legacy:
        # Compute MAP@100 from the harness directly if missing
        cfg_map = TierConfig(tiers=[2], custom_k=[100])
        harness_map = EvaluationHarness(cfg_map)
        t2 = harness_map.evaluate(qrels=trec_qrels, run=trec_run, verbose=False)
        map_key = "T2/P2/MAP@100"
        if map_key in t2:
            legacy["MAP@100"] = t2[map_key]
            legacy["MAP"] = t2[map_key]

    if "NDCG@10" not in legacy:
        cfg_ndcg = TierConfig(tiers=[2], custom_k=[10])
        harness_ndcg = EvaluationHarness(cfg_ndcg)
        t2 = harness_ndcg.evaluate(qrels=trec_qrels, run=trec_run, verbose=False)
        ndcg_key = "T2/P2/NDCG@10"
        if ndcg_key in t2:
            legacy["NDCG@10"] = t2[ndcg_key]

    # Ensure MAP alias
    if "MAP" not in legacy and "MAP@100" in legacy:
        legacy["MAP"] = legacy["MAP@100"]

    return legacy
