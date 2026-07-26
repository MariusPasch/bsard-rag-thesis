"""
evaluation/tier0_efficiency.py
──────────────────────────────
Tier 0 — Efficiency and execution-time metrics.

Evaluates the computational cost of a retrieval system.  While Tiers 1–3
measure *quality*, Tier 0 measures *speed* — a practical prerequisite for
any production RAG deployment.

Metrics produced
────────────────
  Latency distribution : mean, std, median (p50), p90, p95, p99, min, max
  Throughput           : queries per second (QPS)
  Timing breakdown     : embedding, search, re-ranking, total (if provided)
  Index build time     : one-off cost amortised over queries (if provided)

Input format
────────────
  latencies : dict[str, float]
      Per-query retrieval latency in milliseconds.
      {query_id: latency_ms}

  timing_breakdown : dict[str, float] | None
      Optional aggregate timing for pipeline stages (in milliseconds
      unless noted).  Expected keys (all optional):
        - "embedding_ms"    : query embedding time (dense/hybrid only)
        - "search_ms"       : index search time
        - "rerank_ms"       : re-ranking / cross-encoder time
        - "index_build_s"   : one-off index construction (in SECONDS)
        - "total_ms"        : end-to-end pipeline time
      Any additional keys are passed through with "T0/breakdown/" prefix.

Output
──────
  Flat dict with keys prefixed "T0/", e.g.:
      {
          "T0/latency_mean_ms":  42.3,
          "T0/latency_std_ms":    8.1,
          "T0/latency_p50_ms":   40.5,
          "T0/latency_p90_ms":   52.1,
          "T0/latency_p95_ms":   58.7,
          "T0/latency_p99_ms":   71.2,
          "T0/latency_min_ms":   28.0,
          "T0/latency_max_ms":   85.3,
          "T0/throughput_qps":   23.6,
          "T0/total_retrieval_s": 9.41,
          "T0/breakdown/embedding_ms":  12.0,
          "T0/breakdown/search_ms":     28.5,
          "T0/breakdown/rerank_ms":      0.0,
          "T0/breakdown/index_build_s": 12.5,
      }
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


# ── Percentile cutoffs for latency distribution ──────────────────────────────

LATENCY_PERCENTILES: List[int] = [50, 90, 95, 99]


# ════════════════════════════════════════════════════════════════════════
# Core metric functions
# ════════════════════════════════════════════════════════════════════════


def latency_distribution(
    latencies_ms: List[float],
) -> Dict[str, float]:
    """
    Compute descriptive statistics for a per-query latency vector.

    Parameters
    ----------
    latencies_ms : list of per-query latencies in milliseconds

    Returns
    -------
    Dict with mean, std, percentiles (p50/p90/p95/p99), min, max.
    All values in milliseconds.
    """
    if not latencies_ms:
        return {}

    arr = np.array(latencies_ms, dtype=np.float64)

    stats: Dict[str, float] = {
        "T0/latency_mean_ms": float(np.mean(arr)),
        "T0/latency_std_ms":  float(np.std(arr)),
        "T0/latency_min_ms":  float(np.min(arr)),
        "T0/latency_max_ms":  float(np.max(arr)),
    }

    for p in LATENCY_PERCENTILES:
        stats[f"T0/latency_p{p}_ms"] = float(np.percentile(arr, p))

    return stats


def throughput(
    latencies_ms: List[float],
) -> Dict[str, float]:
    """
    Compute throughput and total retrieval time.

    Throughput (QPS) is computed as ``n_queries / total_time_seconds``,
    assuming queries are executed sequentially (worst-case throughput).

    Parameters
    ----------
    latencies_ms : list of per-query latencies in milliseconds

    Returns
    -------
    Dict with QPS and total retrieval time in seconds.
    """
    if not latencies_ms:
        return {}

    total_ms = sum(latencies_ms)
    total_s  = total_ms / 1000.0
    n        = len(latencies_ms)

    return {
        "T0/throughput_qps":    float(n / total_s) if total_s > 0 else 0.0,
        "T0/total_retrieval_s": float(total_s),
        "T0/n_queries":         float(n),
    }


def timing_breakdown_metrics(
    timing_breakdown: Dict[str, float],
) -> Dict[str, float]:
    """
    Pass through pipeline stage timings with a standardised prefix.

    Known keys are normalised; unknown keys are passed through verbatim.
    If both component timings and a total are present, also computes
    the fraction of time spent in each stage.

    Parameters
    ----------
    timing_breakdown : dict of stage_name → time (ms or s, see key suffix)

    Returns
    -------
    Dict with "T0/breakdown/" prefixed keys.
    """
    if not timing_breakdown:
        return {}

    results: Dict[str, float] = {}

    for key, value in timing_breakdown.items():
        results[f"T0/breakdown/{key}"] = float(value)

    # ── Compute stage fractions if total is available ─────────────────
    total_key = None
    for candidate in ("total_ms", "total_retrieval_ms"):
        if candidate in timing_breakdown and timing_breakdown[candidate] > 0:
            total_key = candidate
            break

    if total_key:
        total_val = timing_breakdown[total_key]
        ms_keys = [k for k in timing_breakdown if k.endswith("_ms") and k != total_key]
        for k in ms_keys:
            fraction = timing_breakdown[k] / total_val
            stage_name = k.replace("_ms", "")
            results[f"T0/fraction/{stage_name}"] = round(float(fraction), 4)

    return results


# ════════════════════════════════════════════════════════════════════════
# Main Tier 0 entry point
# ════════════════════════════════════════════════════════════════════════


def evaluate_tier0(
    latencies: Dict[str, float],
    timing_breakdown: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Run all Tier 0 efficiency metrics and return a flat results dict.

    Parameters
    ----------
    latencies : {query_id: latency_ms}
        Per-query retrieval latency in milliseconds.
        Keys are query IDs (str); values are float milliseconds.

    timing_breakdown : optional dict of pipeline stage timings
        Expected keys (all optional):
          - "embedding_ms"    : mean query embedding time
          - "search_ms"       : mean index search time
          - "rerank_ms"       : mean re-ranking time
          - "index_build_s"   : one-off index build time (in seconds)
          - "total_ms"        : mean end-to-end pipeline time per query
        Any extra keys are passed through with "T0/breakdown/" prefix.

    Returns
    -------
    Flat dict with all "T0/" prefixed metric keys.

    Examples
    --------
    >>> latencies = {"q1": 42.0, "q2": 38.5, "q3": 55.0}
    >>> results = evaluate_tier0(latencies)
    >>> results["T0/latency_mean_ms"]
    45.166...
    >>> results["T0/throughput_qps"]
    22.16...

    >>> results = evaluate_tier0(
    ...     latencies,
    ...     timing_breakdown={
    ...         "embedding_ms": 12.0,
    ...         "search_ms": 28.5,
    ...         "rerank_ms": 5.2,
    ...         "total_ms": 45.7,
    ...         "index_build_s": 12.5,
    ...     },
    ... )
    >>> results["T0/breakdown/embedding_ms"]
    12.0
    >>> results["T0/fraction/embedding"]
    0.2626
    """
    if not latencies:
        return {}

    lat_list = list(latencies.values())
    results: Dict[str, float] = {}

    # ── Latency distribution ──────────────────────────────────────────
    results.update(latency_distribution(lat_list))

    # ── Throughput ────────────────────────────────────────────────────
    results.update(throughput(lat_list))

    # ── Pipeline timing breakdown ─────────────────────────────────────
    if timing_breakdown:
        results.update(timing_breakdown_metrics(timing_breakdown))

    return results
