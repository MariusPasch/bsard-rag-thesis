"""Evaluation harness for RQ2.

Delegates T0/T1/T2 supervised metrics to bsard_evaluation (RQ3).
Adds W/* weighted partial-relevance metrics for Arm 1 (chunk retrieval).
Outputs a single flat-dict EvalReport format-compatible with RQ3.

Ground truth and run docs are bsard-id-keyed.
"""

from .models import EvalReport
from .comparator import (
    evaluate,
    evaluate_partial_views,
    significance_test,
    save_metrics_table_csv,
    save_cost_performance_plot,
)
from .ground_truth_loader import load_ground_truth, ground_truth_exists

__all__ = [
    "EvalReport",
    "evaluate",
    "evaluate_partial_views",
    "significance_test",
    "save_metrics_table_csv",
    "save_cost_performance_plot",
    "load_ground_truth",
    "ground_truth_exists",
]
