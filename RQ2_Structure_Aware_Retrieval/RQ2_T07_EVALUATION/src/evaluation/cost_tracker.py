"""LLM call cost tracking and aggregation across retrieval methods."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np


# Approximate pricing per 1M tokens (Ollama local = $0, but we track for reference)
_COST_PER_1M_TOKENS_IN = 0.0
_COST_PER_1M_TOKENS_OUT = 0.0


@dataclass
class CostRecord:
    method: str
    query_id: str
    llm_calls: int
    tokens_in: int
    tokens_out: int
    latency_ms: float


def aggregate_costs(records: list[CostRecord]) -> dict[str, dict]:
    """Compute per-method cost statistics."""
    by_method: dict[str, list[CostRecord]] = defaultdict(list)
    for r in records:
        by_method[r.method].append(r)

    stats: dict[str, dict] = {}
    for method, method_records in by_method.items():
        stats[method] = _method_stats(method_records)
    return stats


def _method_stats(records: list[CostRecord]) -> dict:
    llm_calls = [r.llm_calls for r in records]
    tokens = [r.tokens_in + r.tokens_out for r in records]
    latencies = [r.latency_ms for r in records]
    return {
        "n_queries": len(records),
        "avg_llm_calls": float(np.mean(llm_calls)),
        "total_llm_calls": int(np.sum(llm_calls)),
        "avg_tokens": float(np.mean(tokens)),
        "total_tokens": int(np.sum(tokens)),
        "avg_tokens_in": float(np.mean([r.tokens_in for r in records])),
        "avg_tokens_out": float(np.mean([r.tokens_out for r in records])),
        "avg_latency_ms": float(np.mean(latencies)),
        "median_latency_ms": float(np.median(latencies)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "total_cost_estimate": _estimate_dollar_cost(records),
    }


def _estimate_dollar_cost(records: list[CostRecord]) -> float:
    total_in = sum(r.tokens_in for r in records)
    total_out = sum(r.tokens_out for r in records)
    return (
        total_in * _COST_PER_1M_TOKENS_IN / 1_000_000
        + total_out * _COST_PER_1M_TOKENS_OUT / 1_000_000
    )


def aggregate_costs_from_results(
    all_results: dict[str, list],  # dict[str, list[RetrievalResult]]
) -> dict[str, dict]:
    """Extract cost records from RetrievalResult.cost dicts and aggregate."""
    records: list[CostRecord] = []
    for method, results in all_results.items():
        for result in results:
            cost = result.cost or {}
            records.append(CostRecord(
                method=method,
                query_id=result.query_id,
                llm_calls=cost.get("llm_calls", 0),
                tokens_in=cost.get("tokens_in", 0),
                tokens_out=cost.get("tokens_out", 0),
                latency_ms=cost.get("latency_ms", 0.0),
            ))
    return aggregate_costs(records)
