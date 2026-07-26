"""EvalReport — combined output container for RQ2 evaluation.

Output structure aligns with bsard_evaluation (RQ3):
  results[method] is a flat dict with keys:
    T0/*          — efficiency metrics (latency, throughput)
    T1/*          — BSARD paper metrics (R@k, MRR@100)
    T2/P1/*       — rank-unaware supervised metrics (Precision, Recall, F1, HitRate)
    T2/P2/*       — rank-aware supervised metrics (MRR, MAP, NDCG)
    T2/P3/*       — set-utility graded metrics (RA-nWG, N-Recall4+), if applicable
    W/*           — weighted partial-relevance metrics (Arm 1 only)
                    W/recall@k, W/precision@k, W/mrr@k, W/ndcg@k
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EvalReport:
    """Aggregated evaluation results across all retrieval methods.

    results            : per-method flat metric dicts (T0/T1/T2/W prefixed keys).
    per_query_scores   : per-query vectors used for significance/stratified analysis
                         (not written to disk — large and re-computable).
    significance       : pairwise paired t-test + Wilcoxon results.
    stratified         : per-stratum aggregate metrics (single/multi-article).
    costs              : per-method LLM call and latency statistics.
    autonomous         : reference-free evaluation scores (populated separately).
    """

    results: dict[str, dict[str, float]] = field(default_factory=dict)
    per_query_scores: dict[str, dict[str, list[float]]] = field(default_factory=dict)
    significance: dict[str, dict[str, dict]] = field(default_factory=dict)
    stratified: dict[str, dict[str, dict]] = field(default_factory=dict)
    costs: dict[str, dict] = field(default_factory=dict)
    autonomous: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialisable dict.  Omits per_query_scores (large, re-computable)."""
        return {
            "results": self.results,
            "significance": self.significance,
            "stratified": self.stratified,
            "costs": self.costs,
            "autonomous": self.autonomous,
        }

    def save(self, path: str | Path) -> None:
        """Write the full report to a single JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path) -> "EvalReport":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        report = cls()
        for key, value in data.items():
            if hasattr(report, key):
                setattr(report, key, value)
        return report
