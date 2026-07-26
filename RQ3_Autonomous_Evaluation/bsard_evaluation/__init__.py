"""
bsard_evaluation — Canonical evaluation metrics for the BSARD RAG thesis.

Four-tier evaluation stack:

  Tier 0 : Efficiency metrics (latency distribution, throughput, timing)
  Tier 1 : BSARD paper metrics (Recall@k, MRR@100)
  Tier 2 : Full supervised IR metrics — three panels
             P1 rank-unaware : Precision, Recall, F1, Hit Rate
             P2 rank-aware   : MRR, MAP, NDCG
             P3 set-utility  : RA-nWG, N-Recall4+ (graded labels)
  Tier 3 : Autonomous LLM-based metrics (RAGAS, RAGChecker, G-Eval, ARES)
             + AQS aggregate score for RQ3 rank-correlation analysis

Quick start
───────────
    from bsard_evaluation import EvaluationHarness
    from bsard_evaluation.config import supervised_standard, bsard_benchmark

    # BSARD paper metrics only
    h = EvaluationHarness(bsard_benchmark())
    results = h.evaluate(qrels=qrels, run=run, latencies=latencies)

    # Tier 0 + 1 + 2, standard k values
    h = EvaluationHarness(supervised_standard())
    results = h.evaluate(qrels=qrels, run=run, latencies=latencies)

    # Efficiency only (no quality metrics)
    from bsard_evaluation.config import efficiency_only
    h = EvaluationHarness(efficiency_only())
    results = h.evaluate(latencies=latencies, timing_breakdown=breakdown)

See bsard_evaluation/harness.py for full usage documentation.
"""

from .harness import EvaluationHarness
from .config import (
    TierConfig,
    K_PRESETS,
    bsard_benchmark,
    supervised_standard,
    supervised_full,
    fast_debug,
    efficiency_only,
    full_evaluation,
    generation_window_eval,
)
from .tier1_bsard import per_query_recall
from .stratify import filter_by

__all__ = [
    "EvaluationHarness",
    "TierConfig",
    "K_PRESETS",
    # Config factories
    "bsard_benchmark",
    "supervised_standard",
    "supervised_full",
    "fast_debug",
    "efficiency_only",
    "full_evaluation",
    "generation_window_eval",
    "per_query_recall",
    "filter_by",
]

__version__ = "0.1.0"
