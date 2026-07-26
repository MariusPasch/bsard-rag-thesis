"""
evaluation/harness.py
─────────────────────
EvaluationHarness — single entry point for the full evaluation stack.

Usage examples
──────────────

1. Exact BSARD paper metrics only (Tier 0 + 1):

    from bsard_evaluation import EvaluationHarness
    from bsard_evaluation.config import bsard_benchmark

    harness = EvaluationHarness(bsard_benchmark())
    results = harness.evaluate(qrels=qrels, run=run, latencies=latencies)

2. Tier 0 + 1 + 2 with standard k values (default):

    from bsard_evaluation import EvaluationHarness
    from bsard_evaluation.config import supervised_standard

    harness = EvaluationHarness(supervised_standard())
    results = harness.evaluate(qrels=qrels, run=run, latencies=latencies)

3. Full evaluation including autonomous Tier 3:

    from bsard_evaluation import EvaluationHarness
    from bsard_evaluation.config import full_evaluation

    harness = EvaluationHarness(full_evaluation())
    results = harness.evaluate(
        qrels=qrels,
        run=run,
        latencies=latencies,
        timing_breakdown=timing_breakdown,
        queries=queries,                       # required for Tier 3
        contexts_with_ranks=contexts_with_ranks,  # required for Tier 3
        # answers=answers,                     # optional: enables ARES answer dims
        # judge_model_tier4="gpt-4o",          # for Tier 4 cross-model discipline
    )

4. Efficiency-only (latency analysis without quality metrics):

    from bsard_evaluation import EvaluationHarness
    from bsard_evaluation.config import efficiency_only

    harness = EvaluationHarness(efficiency_only())
    results = harness.evaluate(latencies=latencies, timing_breakdown=breakdown)

5. Custom k values, supervised only:

    from bsard_evaluation import EvaluationHarness
    from bsard_evaluation.config import TierConfig

    cfg = TierConfig(tiers=[0, 1, 2], custom_k=[5, 10, 50, 100, 200])
    harness = EvaluationHarness(cfg)
    results = harness.evaluate(qrels=qrels, run=run, latencies=latencies)

6. Only a subset of Tier 3 components (e.g., skip ARES):

    from bsard_evaluation.config import TierConfig

    cfg = TierConfig(
        tiers=[0, 1, 2, 3],
        k_preset="standard",
        tier3_components=["umbrela", "erag", "ragas_wa"],   # no ares or ragas_wb
        tier3_sample_size=150,
        tier3_use_api=True,
        tier3_model="gpt-4o-mini",
        umbrela_judge_model="gpt-4o-mini",
    )
    harness = EvaluationHarness(cfg)
    results = harness.evaluate(
        qrels=qrels, run=run, latencies=latencies,
        queries=queries, contexts_with_ranks=contexts_with_ranks,
    )

Input format
────────────
qrels    : dict[query_id, dict[doc_id, relevance_grade]]
           Binary (0/1) works for Tier 1 + 2 Panels 1–2.
           Graded (1–5) additionally enables Panel 3 and NDCG graded.

run      : dict[query_id, dict[doc_id, score]]
           Higher score = ranked higher.

latencies : dict[query_id, float]             (Tier 0)
            Per-query retrieval latency in milliseconds.

timing_breakdown : dict[str, float] | None    (Tier 0, optional)
            Pipeline stage timings, e.g.:
            {"embedding_ms": 12.0, "search_ms": 28.5, "index_build_s": 12.5}

queries             : dict[query_id, str]                          (Tier 3)
contexts_with_ranks : dict[query_id, list[tuple[doc_id,           (Tier 3)
                      doc_text, rank]]]   — 1-based rank
answers             : dict[query_id, str]           (Tier 3, optional)
                      Only needed for ARES answer_faithfulness /
                      answer_relevance dimensions.

Output
──────
Flat dict of metric_name → float, e.g.:
    {
      "T0/latency_mean_ms":    42.3,
      "T0/latency_p95_ms":     58.7,
      "T0/throughput_qps":     23.6,
      "T0/breakdown/embedding_ms": 12.0,
      "T1/R@1":                0.210,
      "T1/R@5":                0.465,
      "T1/R@10":               0.550,
      "T1/R@100":              0.621,
      "T1/MRR@100":            0.312,
      "T2/P1/Precision@10":    0.089,
      "T2/P1/Recall@10":       0.550,
      "T2/P1/F1@10":           0.152,
      "T2/P1/HitRate@10":      0.842,
      "T2/P2/MRR@10":          0.442,
      "T2/P2/MAP@100":         0.498,
      "T2/P2/NDCG@10":         0.471,
      "T2/P3/RA-nWG@10":         0.614,   # only if graded qrels
      "T2/P3/N-Recall4+@10":   0.731,   # only if grade ≥ 4 labels exist
      "T2/P3/IDPrecision@10":  0.412,   # if include_id_based (default True)
      "T2/P3/IDRecall@10":     0.550,   # if include_id_based (default True)
      "T2-umbrela/P2/NDCG@10": 0.461,   # only if umbrela ran (Tier 3)
      "T2-umbrela/P3/IDPrecision@10": 0.389,
      "T3/umbrela/mean":        0.71,
      "T3/erag/mean":           0.68,
      "T3/ares/mean":           0.74,
      "T3/ragas_wa/mean":       0.66,
      "T3/ragas_wb/mean":       0.61,   # diagnostic only
      "T3/AQS":                 0.70,
    }
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config import TierConfig, supervised_standard
from .tier0_efficiency import evaluate_tier0
from .tier1_bsard import evaluate_tier1
from .tier2_supervised import evaluate_tier2
from .tier3_autonomous import Tier3Result, evaluate_tier3


class EvaluationHarness:
    """
    Orchestrates all four evaluation tiers for a single retrieval run.

    Parameters
    ----------
    config : TierConfig
        Controls which tiers run, at which k values, and Tier 3 settings.
        Use the factory functions in config.py for common scenarios.
    """

    def __init__(self, config: Optional[TierConfig] = None) -> None:
        self.config = config or supervised_standard()
        # Stores the most recent Tier 3 result so callers can persist
        # per-(q,d) and per-query data (UMBRELA qrels, eRAG scores,
        # RAGAS-WA per-query scores, HyDE responses) to sidecar files.
        # Re-set on every evaluate() call when Tier 3 is configured.
        self.last_tier3_result: Optional[Tier3Result] = None

    # ── Main evaluation entry point ────────────────────────────────────────────

    def evaluate(
        self,
        qrels: Optional[Dict[str, Dict[str, int]]] = None,
        run: Optional[Dict[str, Dict[str, float]]] = None,
        latencies: Optional[Dict[str, float]] = None,
        timing_breakdown: Optional[Dict[str, float]] = None,
        queries: Optional[Dict[str, str]] = None,
        contexts_with_ranks: Optional[Dict[str, List[Tuple[str, str, int]]]] = None,
        answers: Optional[Dict[str, str]] = None,
        judge_model_tier4: Optional[str] = None,
        ares_paths: Optional[Dict[str, str]] = None,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """
        Run the configured evaluation tiers and return a flat results dict.

        Parameters
        ----------
        qrels                : ground truth {query_id: {doc_id: relevance}}
        run                  : retrieval results {query_id: {doc_id: score}}
        latencies            : per-query latencies {query_id: ms}    — Tier 0
        timing_breakdown     : pipeline stage timings {stage: ms/s}  — Tier 0
        queries              : query texts {query_id: str}           — Tier 3
        contexts_with_ranks  : ranked docs {query_id:
                               [(doc_id, doc_text, rank), ...]}      — Tier 3
        answers              : optional generated answers {query_id: str};
                               if provided, passed to ARES for
                               answer_faithfulness / answer_relevance
        judge_model_tier4    : override UMBRELA judge model when evaluating
                               Tier 4 systems (cross-model discipline)
        ares_paths           : paths for ARES input files
        verbose              : print tier timings and result summaries

        Returns
        -------
        Merged flat dict of all metric_name → float across all tiers.
        """
        if verbose:
            print(f"\n{'='*60}")
            print("EvaluationHarness — run summary")
            print(self.config.summary())
            if qrels:
                print(f"Queries in qrels          : {len(qrels)}")
            if run:
                print(f"Queries in run            : {len(run)}")
            if latencies:
                print(f"Queries w/ latency        : {len(latencies)}")
            if contexts_with_ranks:
                print(f"Queries w/ ranked contexts: {len(contexts_with_ranks)}")
            print(f"{'='*60}\n")

        all_results: Dict[str, float] = {}
        # Clear any prior Tier 3 result so callers don't read stale data
        # when this evaluate() call doesn't run Tier 3.
        self.last_tier3_result = None

        # ── Tier 0 ────────────────────────────────────────────────────────────
        if self.config.run_tier0:
            if latencies:
                t0_start = time.time()
                t0 = evaluate_tier0(
                    latencies=latencies,
                    timing_breakdown=timing_breakdown,
                )
                all_results.update(t0)
                if verbose:
                    _print_tier("Tier 0 — Efficiency metrics", t0, time.time() - t0_start)
            elif verbose:
                print("  Tier 0 — SKIPPED (no latencies provided)\n")

        # ── Tier 1 ────────────────────────────────────────────────────────────
        if self.config.run_tier1:
            _validate_quality_inputs(qrels, run, "Tier 1")
            t0_start = time.time()
            t1 = evaluate_tier1(
                qrels=qrels,
                run=run,
                k_values=self.config.k_values,
            )
            all_results.update(t1)
            if verbose:
                _print_tier("Tier 1 — BSARD paper metrics", t1, time.time() - t0_start)

        # ── Tier 2 ────────────────────────────────────────────────────────────
        if self.config.run_tier2:
            _validate_quality_inputs(qrels, run, "Tier 2")
            t0_start = time.time()
            t2 = evaluate_tier2(
                qrels=qrels,
                run=run,
                k_values=self.config.k_values,
                include_graded=self.config.tier2_graded,
                include_id_based=self.config.include_id_based,
            )
            all_results.update(t2)
            if verbose:
                _print_tier("Tier 2 — Supervised metrics", t2, time.time() - t0_start)

        # ── Tier 3 ────────────────────────────────────────────────────────────
        if self.config.run_tier3:
            _validate_tier3_inputs(queries, contexts_with_ranks)
            t0_start = time.time()
            t3_result: Tier3Result = evaluate_tier3(
                queries=queries,
                contexts_with_ranks=contexts_with_ranks,
                answers=answers,
                components=self.config.tier3_components,
                sample_size=self.config.tier3_sample_size,
                llm_model=self.config.tier3_model,
                judge_model_tier4=judge_model_tier4,
                ares_paths=ares_paths,
            )
            self.last_tier3_result = t3_result
            all_results.update(t3_result.metrics)
            if verbose:
                _print_tier(
                    "Tier 3 — Autonomous metrics",
                    t3_result.metrics,
                    time.time() - t0_start,
                )

            # ── UMBRELA → Tier 2 bridge ───────────────────────────────────────
            # Re-run Tier 2 with UMBRELA-produced graded qrels so that the
            # output contains both:
            #   T2/...          — supervised (BSARD ground truth)
            #   T2-umbrela/...  — autonomous (UMBRELA grades)
            # Comparing the two rankings is the core empirical contribution
            # of RQ3.
            if t3_result.umbrela_qrels and run:
                t2u_start = time.time()
                t2_umbrela = evaluate_tier2(
                    qrels=t3_result.umbrela_qrels,
                    run=run,
                    k_values=self.config.k_values,
                    include_graded=True,   # UMBRELA grades are 0–3
                    include_id_based=False, # ID matching is already the vehicle
                )
                # Re-key "T2/..." → "T2-umbrela/..."
                all_results.update(
                    {k.replace("T2/", "T2-umbrela/", 1): v
                     for k, v in t2_umbrela.items()}
                )
                if verbose:
                    _print_tier(
                        "Tier 2 (UMBRELA qrels) — autonomous Tier 2 bridge",
                        {k.replace("T2/", "T2-umbrela/", 1): v
                         for k, v in t2_umbrela.items()},
                        time.time() - t2u_start,
                    )

        if verbose:
            print(f"\n{'='*60}")
            print(f"Total metrics computed: {len(all_results)}")
            if "T0/latency_mean_ms" in all_results:
                print(f"  T0 latency mean  : {all_results['T0/latency_mean_ms']:.1f} ms")
            if "T0/throughput_qps" in all_results:
                print(f"  T0 throughput     : {all_results['T0/throughput_qps']:.1f} QPS")
            if "T1/R@100" in all_results:
                print(f"  Primary — T1/R@100 : {all_results['T1/R@100']:.4f}")
            if "T2/P2/NDCG@10" in all_results:
                print(f"  T2/P2/NDCG@10      : {all_results['T2/P2/NDCG@10']:.4f}")
            if "T3/AQS" in all_results:
                print(f"  T3/AQS                    : {all_results['T3/AQS']:.4f}")
            if "T2-umbrela/P2/NDCG@10" in all_results:
                print(f"  T2-umbrela/P2/NDCG@10     : {all_results['T2-umbrela/P2/NDCG@10']:.4f}")
            print(f"{'='*60}\n")

        return all_results

    # ── Convenience: compare multiple systems ─────────────────────────────────

    def compare(
        self,
        qrels: Dict[str, Dict[str, int]],
        runs: Dict[str, Dict[str, Dict[str, float]]],
        all_latencies: Optional[Dict[str, Dict[str, float]]] = None,
        all_timing_breakdowns: Optional[Dict[str, Dict[str, float]]] = None,
        queries: Optional[Dict[str, str]] = None,
        all_contexts_with_ranks: Optional[
            Dict[str, Dict[str, List[Tuple[str, str, int]]]]
        ] = None,
        all_answers: Optional[Dict[str, Dict[str, str]]] = None,
        all_judge_models_tier4: Optional[Dict[str, str]] = None,
        verbose: bool = True,
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate multiple retrieval systems and return per-system result dicts.

        Parameters
        ----------
        runs                    : {system_name: run_dict}
        all_latencies           : {system_name: {query_id: ms}}       — Tier 0
        all_timing_breakdowns   : {system_name: timing_dict}           — Tier 0
        all_contexts_with_ranks : {system_name: contexts_with_ranks}   — Tier 3
        all_answers             : {system_name: answers_dict}          — Tier 3
        all_judge_models_tier4  : {system_name: model_name}            — Tier 3
                                  per-system UMBRELA judge model override

        Returns
        -------
        {system_name: {metric_name: float}}
        """
        all_system_results: Dict[str, Dict[str, float]] = {}
        for system_name, run in runs.items():
            if verbose:
                print(f"\n>>> Evaluating system: {system_name}")
            all_system_results[system_name] = self.evaluate(
                qrels=qrels,
                run=run,
                latencies=(all_latencies or {}).get(system_name),
                timing_breakdown=(all_timing_breakdowns or {}).get(system_name),
                queries=queries,
                contexts_with_ranks=(all_contexts_with_ranks or {}).get(system_name),
                answers=(all_answers or {}).get(system_name),
                judge_model_tier4=(all_judge_models_tier4 or {}).get(system_name),
                verbose=verbose,
            )
        return all_system_results

    # ── Persistence helpers ───────────────────────────────────────────────────

    def save_results(
        self,
        results: Dict[str, float],
        path: str,
        system_name: Optional[str] = None,
    ) -> None:
        """
        Save results dict to a JSON file.

        The file is appended to (not overwritten) so multiple system
        results can accumulate in a single artefact, matching the
        evaluation/*/results/ directory convention from the thesis proposal.

        Parameters
        ----------
        results     : output of evaluate()
        path        : target .json file path
        system_name : optional label attached to the results object
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        existing: List[dict] = []
        if p.exists():
            try:
                with p.open() as f:
                    existing = json.load(f)
                if not isinstance(existing, list):
                    existing = [existing]
            except (json.JSONDecodeError, ValueError):
                existing = []

        entry = {
            "system":    system_name,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "config": {
                "tiers":    self.config.tiers,
                "k_values": self.config.k_values,
            },
            "results": results,
        }
        existing.append(entry)

        with p.open("w") as f:
            json.dump(existing, f, indent=2)

        print(f"Results saved → {path}")


# ── Internal helpers ───────────────────────────────────────────────────────────


def _validate_quality_inputs(
    qrels: Optional[Dict],
    run: Optional[Dict],
    tier_name: str,
) -> None:
    missing = [
        name
        for name, obj in [("qrels", qrels), ("run", run)]
        if obj is None
    ]
    if missing:
        raise ValueError(
            f"{tier_name} requires: {missing}.  "
            f"Pass them as keyword arguments to harness.evaluate()."
        )


def _validate_tier3_inputs(
    queries: Optional[Dict],
    contexts_with_ranks: Optional[Dict],
) -> None:
    missing = [
        name
        for name, obj in [
            ("queries", queries),
            ("contexts_with_ranks", contexts_with_ranks),
        ]
        if obj is None
    ]
    if missing:
        raise ValueError(
            f"Tier 3 requires: {missing}.  "
            f"Pass them as keyword arguments to harness.evaluate().  "
            f"'answers' is optional — only needed for ARES "
            f"answer_faithfulness / answer_relevance dimensions."
        )


def _print_tier(label: str, results: Dict[str, float], elapsed: float) -> None:
    print(f"  {label}  [{elapsed:.1f}s]")
    for k, v in sorted(results.items()):
        print(f"    {k:42s} {v:.4f}")
    print()
