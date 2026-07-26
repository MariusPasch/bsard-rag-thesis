"""RQ2 evaluation pipeline.

Delegation map
──────────────
CALLS bsard_evaluation (RQ3):
  • evaluate_tier0  — T0/* latency + throughput (takes {query_id: ms})
  • evaluate_tier1  — T1/* BSARD paper Recall@k + MRR@100 (TREC format)
  • evaluate_tier2  — T2/P1/* P2/* P3/* supervised IR (TREC format)

STAYS IN T07 (RQ2-specific or needs per-query access):
  • adapter.py           — converts list[RetrievalResult] → TREC qrels / run / latencies
  • Arm 1 chunk→article  — max-score article aggregation for binary metrics
  • W/* weighted metrics — chunk-article fractional relevance (not in RQ3)
  • per_query_scores     — needed for significance tests and stratified analysis
  • significance_test()  — all-pairwise paired t-test + Wilcoxon
  • stratified analysis  — single/multi-article breakdown

Output
──────
EvalReport.results[method] = flat dict, e.g.:
  {
    "T0/latency_mean_ms": 42.3,
    "T1/R@10":            0.550,
    "T2/P2/NDCG@10":      0.471,
    "W/recall@10":        0.523,   ← Arm 1 only
    "W/ndcg@10":          0.489,   ← Arm 1 only
  }
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

import numpy as np
from scipy import stats

from . import adapter, weight_precomputer
from .cost_tracker import aggregate_costs_from_results
from .metrics import recall_at_k, mrr_at_k, ndcg_at_k, cosine_weighted_recall_at_k
from .models import EvalReport
from .projection import ProjectionRow
from .weighted_metrics import (
    weighted_recall_at_k,
    weighted_precision_at_k,
    weighted_mrr,
    weighted_ndcg_at_k,
)

# Default location of the vendored question extraction-status JSONL — used to
# auto-load per-question buckets for stratification and to power the
# strict/lenient PARTIAL-handling views.
_QUESTION_STATUS_JSONL = (
    Path(__file__).resolve().parents[2]
    / "ground_truth"
    / "question_extraction_status"
    / "questions_by_extraction_status.jsonl"
)

logger = logging.getLogger(__name__)

# Metrics tracked per-query for significance tests and stratified analysis.
# Must be a subset of what bsard_evaluation computes (T1/ T2/ keys) + W/.
_SIG_METRICS = ("T1/R@10", "T1/R@100")
_SIG_METRICS_W = ("W/recall@10", "W/recall@100")


# ---------------------------------------------------------------------------
# Master entry point
# ---------------------------------------------------------------------------

def evaluate(
    all_results: dict[str, list],         # {method: list[RetrievalResult]}
    ground_truth: dict[str, list[int]],
    config=None,                           # bsard_evaluation.config.TierConfig | None
    # Arm 1 weighted metrics — provide ONE of:
    weights_lookup: dict[tuple[str, int], float] | None = None,
    chunks: list | None = None,            # list[Chunk] — compute weights on the fly
    articles: list | None = None,          # list[Article]
    # Arm 1 binary metrics — chunk→bsard mapping (optional; tries metadata first)
    chunk_to_bsard: dict[str, int] | None = None,
    # Cosine-weighted recall (Cw/recall@k). Pass the output of
    # question_subsets.build_ground_truth_with_cosines — same key set as
    # ground_truth, values are {bsard_id: extraction_cosine | None}. When None or
    # all-cosines-missing, the Cw/ columns are skipped silently.
    cosine_ground_truth: dict[str, dict[int, float | None]] | None = None,
    # Weight precomputation settings (used only when chunks/articles provided)
    weights_cache_path: str | None = None,
    tokenizer_name: str = "intfloat/multilingual-e5-large-instruct",
    # Stratification — {query_id: bucket} from the extraction-status JSONL.
    # When None, auto-loaded from the vendored JSONL; pass {} to disable.
    question_buckets: dict[str, str] | None = None,
    # PDF-centric experiment cache (CACHE_DESIGN.md §5.2). When use_cache is
    # True AND all of (doc_id, arm1_config_hash, t07_cache_root, t03_cache_root,
    # db_conn, pdf_filename) are set, weights are loaded/built via the cache;
    # otherwise the legacy chunks+articles or weights_cache_path paths apply.
    *,
    use_cache: bool = True,
    doc_id: str | None = None,
    arm1_config_hash: str | None = None,
    t07_cache_root: "Path | None" = None,
    t03_cache_root: "Path | None" = None,
    db_conn: "sqlite3.Connection | None" = None,
    pdf_filename: str | None = None,
    # Layer 3 — Effective (drift-aware) metrics. When provided, emit E/ keys
    # alongside T1/W/Cw with the recall denominator restricted to gt_in_pdf
    # (full GT ∩ bsard_ids reachable in this PDF). Questions with empty
    # gt_in_pdf are excluded from the E/ aggregate and counted in
    # E/n_not_evaluable. Backwards-compatible: when None, no E/ keys emitted.
    projection: list[ProjectionRow] | None = None,
) -> EvalReport:
    """Run the full RQ2 evaluation pipeline.

    Ground truth is bsard-id-keyed: {query_id: [bsard_id, ...]} with int values.

    For each method:
      1. Convert to TREC format (via adapter).
      2. Call bsard_evaluation.EvaluationHarness for T0/T1/T2 aggregate metrics.
      3. Add W/ weighted metrics (Arm 1 methods only).
      4. Track per-query scores for significance + stratified analysis.
    """
    try:
        from bsard_evaluation import EvaluationHarness
        from bsard_evaluation.config import supervised_standard
    except ImportError as exc:
        raise ImportError(
            "bsard_evaluation (RQ3) is not installed.  "
            "Run: pip install -e ../../RQ3_Autonomous_Evaluation"
        ) from exc

    cfg = config if config is not None else supervised_standard()
    harness = EvaluationHarness(cfg)

    report = EvalReport()

    # ── Resolve weights for Arm 1 ─────────────────────────────────────────────
    has_arm1 = any(adapter.is_chunk_based(m) for m in all_results)
    if has_arm1:
        weights_lookup = _resolve_weights(
            weights_lookup, chunks, articles,
            weights_cache_path, tokenizer_name,
            use_cache=use_cache,
            doc_id=doc_id,
            arm1_config_hash=arm1_config_hash,
            t07_cache_root=t07_cache_root,
            t03_cache_root=t03_cache_root,
            db_conn=db_conn,
            pdf_filename=pdf_filename,
        )
        if weights_lookup and not chunk_to_bsard:
            chunk_to_bsard = adapter.build_chunk_to_bsard_from_weights(weights_lookup)

    qrels = adapter.to_qrels(ground_truth)

    # ── Per-method evaluation ─────────────────────────────────────────────────
    for method, results in all_results.items():
        is_chunks = adapter.is_chunk_based(method)
        run = adapter.to_bsard_run(
            results,
            chunk_to_bsard=chunk_to_bsard if is_chunks else None,
        )
        latencies = adapter.to_latencies(results)

        # T0 / T1 / T2 via bsard_evaluation
        method_metrics: dict[str, float] = harness.evaluate(
            qrels=qrels,
            run=run,
            latencies=latencies,
            verbose=False,
        )

        # W/ weighted metrics (Arm 1 only)
        if is_chunks and weights_lookup:
            w_metrics = _compute_weighted_metrics(
                results, weights_lookup, ground_truth, cfg.k_values
            )
            method_metrics.update(w_metrics)

        # Cw/ cosine-weighted recall (any method, requires cosine_ground_truth)
        if cosine_ground_truth:
            cw_metrics = _compute_cosine_weighted_metrics(
                results, cosine_ground_truth, cfg.k_values
            )
            method_metrics.update(cw_metrics)

        # E/ effective (drift-aware) metrics — Layer 3
        if projection is not None:
            e_metrics = _compute_effective_metrics(run, projection, cfg.k_values)
            method_metrics.update(e_metrics)

        report.results[method] = method_metrics
        logger.info(
            "Evaluated method=%s  metrics=%d  arm1=%s",
            method, len(method_metrics), is_chunks,
        )

    # ── Per-query scores for significance + stratified ────────────────────────
    _collect_per_query_scores(
        report, all_results, ground_truth, weights_lookup, has_arm1,
        projection=projection, chunk_to_bsard=chunk_to_bsard,
    )

    # ── Significance tests ────────────────────────────────────────────────────
    report.significance = _run_significance_tests(report.per_query_scores)

    # ── Stratified analysis ───────────────────────────────────────────────────
    if question_buckets is None:
        question_buckets = _load_question_buckets()
    report.stratified = _run_stratified_analysis(
        report.per_query_scores, all_results, ground_truth, question_buckets,
    )

    # ── Cost summary ──────────────────────────────────────────────────────────
    report.costs = aggregate_costs_from_results(all_results)

    return report


# ---------------------------------------------------------------------------
# Weighted metrics (W/ keys — Arm 1 only)
# ---------------------------------------------------------------------------

def _compute_weighted_metrics(
    results: list,
    weights_lookup: dict[tuple[str, int], float],
    ground_truth: dict[str, list[int]],
    k_values: list[int],
) -> dict[str, float]:
    """Compute mean weighted metrics across all queries. Returns W/ keyed dict."""
    per_query: dict[str, list[float]] = {}
    for k in k_values:
        per_query[f"W/recall@{k}"] = []
        per_query[f"W/precision@{k}"] = []
        per_query[f"W/mrr@{k}"] = []
        per_query[f"W/ndcg@{k}"] = []

    for result in results:
        relevant = {int(b) for b in ground_truth.get(result.query_id, [])}
        ranked_ids = [item["id"] for item in result.ranked_items]
        for k in k_values:
            per_query[f"W/recall@{k}"].append(
                weighted_recall_at_k(ranked_ids, weights_lookup, relevant, k)
            )
            per_query[f"W/precision@{k}"].append(
                weighted_precision_at_k(ranked_ids, weights_lookup, relevant, k)
            )
            per_query[f"W/mrr@{k}"].append(
                weighted_mrr(ranked_ids, weights_lookup, relevant, k)
            )
            per_query[f"W/ndcg@{k}"].append(
                weighted_ndcg_at_k(ranked_ids, weights_lookup, relevant, k)
            )

    return {key: float(np.mean(vals)) for key, vals in per_query.items() if vals}


# ---------------------------------------------------------------------------
# Cosine-weighted recall (Cw/ keys — any method)
# ---------------------------------------------------------------------------

def _compute_cosine_weighted_metrics(
    results: list,
    cosine_ground_truth: dict[str, dict[int, float | None]],
    k_values: list[int],
) -> dict[str, float]:
    """Mean cosine-weighted recall across queries. Returns Cw/ keyed dict.

    Per-query: degrades to plain recall when no GT item has a usable cosine
    (see metrics.cosine_weighted_recall_at_k). Aggregate is mean over queries
    that appear in cosine_ground_truth — queries absent from it are skipped.
    """
    per_query: dict[str, list[float]] = {f"Cw/recall@{k}": [] for k in k_values}

    for result in results:
        gt = cosine_ground_truth.get(result.query_id)
        if not gt:
            continue
        ranked_ids = [item["id"] for item in result.ranked_items]
        for k in k_values:
            per_query[f"Cw/recall@{k}"].append(
                cosine_weighted_recall_at_k(ranked_ids, gt, k)
            )

    return {key: float(np.mean(vals)) for key, vals in per_query.items() if vals}


# ---------------------------------------------------------------------------
# Effective (drift-aware) metrics — Layer 3 (E/ keys — any method)
# ---------------------------------------------------------------------------

def _compute_effective_metrics(
    run: dict[str, dict[str, float]],
    projection: list[ProjectionRow],
    k_values: list[int],
) -> dict[str, float | int]:
    """Mean E/ binary metrics across evaluable queries.

    ``run`` is the bsard-id-keyed TREC run produced by
    :func:`adapter.to_bsard_run`, so semantics match T1/ exactly — same scores,
    same ranking — only the recall denominator changes from full GT to
    ``gt_in_pdf``. Queries with empty ``gt_in_pdf`` are skipped from the
    aggregate (drift makes them not-evaluable on this PDF) and counted in
    ``E/n_not_evaluable``.
    """
    proj_by_qid = {str(row.question_id): row for row in projection}

    per_query: dict[str, list[float]] = {}
    for k in k_values:
        per_query[f"E/recall@{k}"] = []
        per_query[f"E/mrr@{k}"] = []
        per_query[f"E/ndcg@{k}"] = []
    n_evaluable = 0
    n_not_evaluable = 0

    for qid, bsard_scores in run.items():
        proj_row = proj_by_qid.get(qid)
        if proj_row is None or not proj_row.gt_in_pdf:
            n_not_evaluable += 1
            continue
        n_evaluable += 1
        gt_in_pdf_str = {str(b) for b in proj_row.gt_in_pdf}
        ranked = sorted(bsard_scores, key=lambda b: -bsard_scores[b])
        for k in k_values:
            per_query[f"E/recall@{k}"].append(recall_at_k(ranked, gt_in_pdf_str, k))
            per_query[f"E/mrr@{k}"].append(mrr_at_k(ranked, gt_in_pdf_str, k))
            per_query[f"E/ndcg@{k}"].append(ndcg_at_k(ranked, gt_in_pdf_str, k))

    out: dict[str, float | int] = {
        key: float(np.mean(vals)) for key, vals in per_query.items() if vals
    }
    out["E/n_evaluable"] = n_evaluable
    out["E/n_not_evaluable"] = n_not_evaluable
    return out


# ---------------------------------------------------------------------------
# Per-query scores (for significance + stratified)
# ---------------------------------------------------------------------------

def _collect_per_query_scores(
    report: EvalReport,
    all_results: dict[str, list],
    ground_truth: dict[str, list[int]],
    weights_lookup: dict[tuple[str, int], float] | None,
    has_arm1: bool,
    *,
    projection: list[ProjectionRow] | None = None,
    chunk_to_bsard: dict[str, int] | None = None,
) -> None:
    """Populate report.per_query_scores with T1/R@k and W/recall@k vectors.

    Binary recall is computed against bsard-id ground truth using the run's
    *resolved* doc_ids (string-form bsard_ids), so this matches what
    bsard_evaluation does. Weighted recall consumes the int set directly.

    When ``projection`` is provided, also collects E/recall@10 and E/recall@100
    per-query for evaluable queries (those with non-empty gt_in_pdf). Drifted
    queries are skipped from the E/ vectors so significance tests run only
    over the comparable, evaluable subset.
    """
    proj_by_qid = (
        {str(row.question_id): row for row in projection}
        if projection is not None
        else None
    )

    for method, results in all_results.items():
        is_chunks = adapter.is_chunk_based(method)
        pqs: dict[str, list[float]] = {
            "T1/R@10": [], "T1/R@100": [],
        }
        if is_chunks and weights_lookup:
            pqs["W/recall@10"] = []
            pqs["W/recall@100"] = []
        if proj_by_qid is not None:
            pqs["E/recall@10"] = []
            pqs["E/recall@100"] = []

        # Build a bsard-id-level run once per method so E/ semantics match T1/.
        run = (
            adapter.to_bsard_run(
                results,
                chunk_to_bsard=chunk_to_bsard if is_chunks else None,
            )
            if proj_by_qid is not None
            else None
        )

        for result in results:
            relevant_int = {int(b) for b in ground_truth.get(result.query_id, [])}
            relevant_str = {str(b) for b in relevant_int}
            ranked_ids = [item["id"] for item in result.ranked_items]

            pqs["T1/R@10"].append(recall_at_k(ranked_ids, relevant_str, 10))
            pqs["T1/R@100"].append(recall_at_k(ranked_ids, relevant_str, 100))

            if is_chunks and weights_lookup:
                pqs["W/recall@10"].append(
                    weighted_recall_at_k(ranked_ids, weights_lookup, relevant_int, 10)
                )
                pqs["W/recall@100"].append(
                    weighted_recall_at_k(ranked_ids, weights_lookup, relevant_int, 100)
                )

            if proj_by_qid is not None and run is not None:
                proj_row = proj_by_qid.get(result.query_id)
                if proj_row is not None and proj_row.gt_in_pdf:
                    gt_in_pdf_str = {str(b) for b in proj_row.gt_in_pdf}
                    bsard_scores = run.get(result.query_id, {})
                    ranked_bsard = sorted(
                        bsard_scores, key=lambda b: -bsard_scores[b]
                    )
                    pqs["E/recall@10"].append(
                        recall_at_k(ranked_bsard, gt_in_pdf_str, 10)
                    )
                    pqs["E/recall@100"].append(
                        recall_at_k(ranked_bsard, gt_in_pdf_str, 100)
                    )

        report.per_query_scores[method] = pqs


# ---------------------------------------------------------------------------
# Significance tests
# ---------------------------------------------------------------------------

def significance_test(scores_a: list[float], scores_b: list[float]) -> dict:
    """Paired comparison between two methods across queries."""
    a = np.array(scores_a)
    b = np.array(scores_b)
    diff = b - a

    _, t_pval = stats.ttest_rel(a, b)
    try:
        _, w_pval = stats.wilcoxon(a, b, alternative="two-sided")
    except ValueError:
        w_pval = 1.0

    return {
        "mean_diff": float(np.mean(diff)),
        "std_diff": float(np.std(diff)),
        "cohens_d": float(np.mean(diff) / np.std(diff)) if np.std(diff) > 0 else 0.0,
        "paired_t_pval": float(t_pval),
        "wilcoxon_pval": float(w_pval),
        "significant_at_005": bool(min(t_pval, w_pval) < 0.05),
    }


def _run_significance_tests(
    per_query_scores: dict[str, dict[str, list[float]]],
) -> dict[str, dict[str, dict]]:
    """All-pairwise significance tests for T1/R@10, T1/R@100, and W/recall@k."""
    methods = list(per_query_scores.keys())
    results: dict[str, dict[str, dict]] = {}

    for i, method_a in enumerate(methods):
        for method_b in methods[i + 1:]:
            pair_key = f"{method_a}_vs_{method_b}"
            pair_results: dict[str, dict] = {}

            shared = set(per_query_scores[method_a]) & set(per_query_scores[method_b])
            for metric in sorted(shared):
                va = per_query_scores[method_a][metric]
                vb = per_query_scores[method_b][metric]
                if len(va) == len(vb) and len(va) > 1:
                    pair_results[metric] = significance_test(va, vb)

            if pair_results:
                results[pair_key] = pair_results

    return results


# ---------------------------------------------------------------------------
# Stratified analysis
# ---------------------------------------------------------------------------

def _run_stratified_analysis(
    per_query_scores: dict[str, dict[str, list[float]]],
    all_results: dict[str, list],
    ground_truth: dict[str, list[int]],
    question_buckets: dict[str, str] | None = None,
) -> dict[str, dict[str, dict]]:
    """Per-stratum mean T1/R@10 and T1/R@100 (+ W/recall@ for Arm 1).

    Strata always emitted:
      single_article  — queries with exactly one ground-truth article
      multi_article   — queries with two or more ground-truth articles

    Additional strata when question_buckets is provided (one per
    extraction_status bucket present in the eval set):
      exact / partial / mixed / not_present
    """
    # Build per-query strata across both axes. Each query gets:
    #   - one cardinality stratum (single/multi)
    #   - optionally one bucket stratum (extraction_status)
    cardinality_stratum: dict[str, str] = {}
    first_results = next(iter(all_results.values()), [])
    for result in first_results:
        n = len(ground_truth.get(result.query_id, []))
        cardinality_stratum[result.query_id] = (
            "single_article" if n == 1 else "multi_article"
        )

    # Build per-query index: method → metric → {query_id: score}
    per_query_indexed: dict[str, dict[str, dict[str, float]]] = {}
    for method, results in all_results.items():
        per_query_indexed[method] = {}
        pqs = per_query_scores.get(method, {})
        for metric, scores in pqs.items():
            per_query_indexed[method][metric] = {
                results[i].query_id: scores[i]
                for i in range(len(scores))
                if i < len(results)
            }

    aggregated: dict[str, dict[str, dict]] = {}

    def _aggregate(stratum_name: str, stratum_qids: set[str]) -> None:
        if not stratum_qids:
            return
        aggregated[stratum_name] = {}
        for method in all_results:
            method_entry: dict[str, object] = {"n": len(stratum_qids)}
            for metric, qid_scores in per_query_indexed.get(method, {}).items():
                stratum_scores = [
                    score for qid, score in qid_scores.items() if qid in stratum_qids
                ]
                if stratum_scores:
                    method_entry[metric] = float(np.mean(stratum_scores))
            aggregated[stratum_name][method] = method_entry

    # Cardinality axis
    for stratum in ("single_article", "multi_article"):
        qids = {qid for qid, s in cardinality_stratum.items() if s == stratum}
        _aggregate(stratum, qids)

    # Bucket axis (only for queries actually evaluated, i.e. present in GT)
    if question_buckets:
        eval_qids = set(cardinality_stratum)  # queries that ran
        gt_qids = set(ground_truth)            # queries with non-empty GT
        present_qids = eval_qids & gt_qids
        for bucket in ("exact", "partial", "mixed", "not_present"):
            qids = {qid for qid in present_qids
                    if question_buckets.get(qid) == bucket}
            _aggregate(bucket, qids)

    return aggregated


# ---------------------------------------------------------------------------
# Question-bucket helper
# ---------------------------------------------------------------------------

def _load_question_buckets(
    jsonl_path: str | Path | None = None,
) -> dict[str, str]:
    """Load {query_id_str: extraction_status} from the vendored JSONL.

    Returns {} if the file is missing — bucket stratification is then skipped
    silently (a gentler failure mode than raising during evaluation).
    """
    p = Path(jsonl_path) if jsonl_path else _QUESTION_STATUS_JSONL
    if not p.exists():
        logger.warning(
            "question extraction-status JSONL not found at %s — "
            "bucket stratification will be skipped.", p,
        )
        return {}
    buckets: dict[str, str] = {}
    with open(p, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            buckets[str(row["question_id"])] = row["extraction_status"]
    return buckets


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def save_metrics_table_csv(report: EvalReport, path: str | Path) -> None:
    """Write a method × metric CSV from report.results."""
    import csv

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not report.results:
        return

    methods = sorted(report.results)
    all_keys = sorted({k for m in report.results.values() for k in m})

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["method"] + all_keys)
        for method in methods:
            row = [method] + [report.results[method].get(k, "") for k in all_keys]
            writer.writerow(row)

    logger.info("Metrics table saved → %s", path)


def save_cost_performance_plot(
    report: EvalReport,
    output_path: str | Path,
    x_metric: str = "T1/R@100",
    y_metric: str = "T0/latency_mean_ms",
) -> None:
    """Scatter: x = Recall@100, y = mean latency, one point per method."""
    import matplotlib.pyplot as plt

    methods = sorted(set(report.results) & set(report.costs))
    if not methods:
        return

    x_vals = [report.results[m].get(x_metric, 0.0) for m in methods]
    y_vals = [report.costs[m].get("avg_latency_ms", 0.0) for m in methods]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x_vals, y_vals)
    for method, x, y in zip(methods, x_vals, y_vals):
        ax.annotate(method, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)

    ax.set_xlabel(x_metric)
    ax.set_ylabel(y_metric)
    ax.set_title("Cost–Performance Tradeoff")
    fig.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Cost-performance plot saved → %s", output_path)


# ---------------------------------------------------------------------------
# Weight resolution helper
# ---------------------------------------------------------------------------

def _resolve_weights(
    weights_lookup: dict | None,
    chunks: list | None,
    articles: list | None,
    weights_cache_path: str | None,
    tokenizer_name: str,
    *,
    use_cache: bool = True,
    doc_id: str | None = None,
    arm1_config_hash: str | None = None,
    t07_cache_root: Path | None = None,
    t03_cache_root: Path | None = None,
    db_conn: sqlite3.Connection | None = None,
    pdf_filename: str | None = None,
) -> dict | None:
    """Return weights_lookup, resolving via the cache when wired, otherwise legacy.

    Resolution order: explicit ``weights_lookup`` → PDF-centric cache →
    legacy ``weights_cache_path`` CSV → in-memory ``chunks + articles`` build.
    """
    if weights_lookup is not None:
        return weights_lookup

    cache_args_complete = bool(
        use_cache
        and doc_id and arm1_config_hash
        and t07_cache_root is not None and t03_cache_root is not None
        and db_conn is not None and pdf_filename
    )
    if cache_args_complete:
        from arm1_naive.cache import CacheRoot
        from .cache import T07CacheRoot

        logger.info(
            "Resolving chunk-article weights via PDF cache (doc=%s arm1_config=%s)",
            doc_id, arm1_config_hash,
        )
        weights = weight_precomputer.precompute_weights_cached(
            doc_id=doc_id,
            arm1_config_hash=arm1_config_hash,
            t07_cache_root=T07CacheRoot(t07_cache_root),
            t03_cache_root=CacheRoot(t03_cache_root),
            db_conn=db_conn,
            tokenizer_name=tokenizer_name,
            pdf_filename=pdf_filename,
        )
        return weight_precomputer.weights_to_lookup(weights)

    if weights_cache_path and Path(weights_cache_path).exists():
        logger.info("Loading cached chunk-article weights from %s", weights_cache_path)
        weights = weight_precomputer.load_weights_csv(weights_cache_path)
        return weight_precomputer.weights_to_lookup(weights)

    if chunks and articles:
        logger.info(
            "Precomputing chunk-article weights (tokenizer=%s)", tokenizer_name
        )
        weights = weight_precomputer.precompute_weights(chunks, articles, tokenizer_name)
        if weights_cache_path:
            weight_precomputer.save_weights_csv(weights, weights_cache_path)
            logger.info("Weights cached → %s", weights_cache_path)
        return weight_precomputer.weights_to_lookup(weights)

    logger.warning(
        "Arm 1 detected but no weights_lookup, cache wiring, weights_cache_path, "
        "or chunks/articles provided.  W/ metrics will be skipped."
    )
    return None


# ---------------------------------------------------------------------------
# PARTIAL-handling wrapper: strict / lenient / delta
# ---------------------------------------------------------------------------

def evaluate_partial_views(
    all_results: dict[str, list],
    *,
    extraction_status_jsonl: str | Path | None = None,
    split: str | None = None,
    extraction_status_buckets: list[str] | None = None,
    config=None,
    weights_lookup: dict[tuple[str, int], float] | None = None,
    chunks: list | None = None,
    articles: list | None = None,
    chunk_to_bsard: dict[str, int] | None = None,
    weights_cache_path: str | None = None,
    tokenizer_name: str = "intfloat/multilingual-e5-large-instruct",
    # PDF-centric experiment cache (forwarded verbatim to ``evaluate``).
    use_cache: bool = True,
    doc_id: str | None = None,
    arm1_config_hash: str | None = None,
    t07_cache_root: "Path | None" = None,
    t03_cache_root: "Path | None" = None,
    db_conn: "sqlite3.Connection | None" = None,
    pdf_filename: str | None = None,
) -> dict[str, EvalReport]:
    """Run evaluate() under two PARTIAL-handling regimes and return both reports + delta.

    Regimes
    -------
    strict   GT keeps only FOUND bsard_articles per question.  PARTIAL counts as
             a miss.  Questions whose entire GT becomes empty (partial-only,
             not_present) are dropped from this view.
    lenient  GT keeps FOUND + PARTIAL bsard_articles per question.  PARTIAL
             counts as a hit.  Only not_present-only questions are dropped.

    NOT FOUND articles are always dropped from GT under both regimes — they are
    not retrievable from the PDF corpus and would conflate retrieval quality
    with corpus completeness.

    Filters
    -------
    extraction_status_jsonl   Path to questions_by_extraction_status.jsonl.
                              Defaults to the vendored copy.
    split                     'train' / 'test' / None.
    extraction_status_buckets Subset of {exact, partial, mixed, not_present}.
                              Pre-filters questions before strict/lenient GT
                              assembly.

    Returns
    -------
    dict with three keys:
      'strict'   — EvalReport from the strict regime
      'lenient'  — EvalReport from the lenient regime
      'delta'    — EvalReport whose .results[m][k] = lenient[m][k] − strict[m][k]
                   and likewise for .stratified.  Confounds two effects:
                   (i) PARTIAL credit on shared questions; (ii) lenient pool is
                   larger.  For pure PARTIAL-sensitivity, read the two reports
                   independently — the delta is a quick directional summary.
    """
    # Lazy import to avoid a runtime cycle and to keep pandas dependency local.
    from . import question_subsets

    df = question_subsets.load_extraction_status(extraction_status_jsonl)
    qids_pool = question_subsets.select_question_ids(
        df,
        extraction_status=extraction_status_buckets,
        split=split,
    )

    strict_gt = question_subsets.build_ground_truth(
        df, qids_pool, restrict_per_question_to_status=["FOUND"],
    )
    lenient_gt = question_subsets.build_ground_truth(
        df, qids_pool, restrict_per_question_to_status=["FOUND", "PARTIAL"],
    )
    strict_cos_gt = question_subsets.build_ground_truth_with_cosines(
        df, qids_pool, restrict_per_question_to_status=["FOUND"],
    )
    lenient_cos_gt = question_subsets.build_ground_truth_with_cosines(
        df, qids_pool, restrict_per_question_to_status=["FOUND", "PARTIAL"],
    )

    logger.info(
        "evaluate_partial_views: pool=%d questions; strict=%d (PARTIAL→miss); lenient=%d (PARTIAL→hit)",
        len(qids_pool), len(strict_gt), len(lenient_gt),
    )

    # bsard_evaluation can mark queries-with-empty-qrels as recall=0 and skew
    # the average; safer to filter results to each regime's question pool.
    def _filter(results: list, allowed: set[str]) -> list:
        return [r for r in results if r.query_id in allowed]

    strict_results = {
        m: _filter(rs, set(strict_gt)) for m, rs in all_results.items()
    }
    lenient_results = {
        m: _filter(rs, set(lenient_gt)) for m, rs in all_results.items()
    }

    # Auto-load buckets once and share between runs.
    buckets = _load_question_buckets(extraction_status_jsonl)

    common_kwargs = dict(
        config=config,
        weights_lookup=weights_lookup,
        chunks=chunks,
        articles=articles,
        chunk_to_bsard=chunk_to_bsard,
        weights_cache_path=weights_cache_path,
        tokenizer_name=tokenizer_name,
        question_buckets=buckets,
        use_cache=use_cache,
        doc_id=doc_id,
        arm1_config_hash=arm1_config_hash,
        t07_cache_root=t07_cache_root,
        t03_cache_root=t03_cache_root,
        db_conn=db_conn,
        pdf_filename=pdf_filename,
    )

    strict_report = evaluate(
        strict_results, strict_gt, cosine_ground_truth=strict_cos_gt, **common_kwargs,
    )
    lenient_report = evaluate(
        lenient_results, lenient_gt, cosine_ground_truth=lenient_cos_gt, **common_kwargs,
    )
    delta_report = _compute_delta_report(strict_report, lenient_report)

    return {"strict": strict_report, "lenient": lenient_report, "delta": delta_report}


def _compute_delta_report(strict: EvalReport, lenient: EvalReport) -> EvalReport:
    """Build a delta EvalReport: lenient − strict per method × metric.

    Populates .results and .stratified only.  significance/costs/autonomous
    are not meaningful as differences across regimes and are left empty.

    Stratum entries get auxiliary 'n_strict' and 'n_lenient' counts so the
    reader can see how the question pool shifted per stratum.
    """
    delta = EvalReport()

    for method in sorted(set(strict.results) & set(lenient.results)):
        s = strict.results[method]
        l = lenient.results[method]
        keys = sorted(set(s) & set(l))
        delta.results[method] = {
            k: float(l[k] - s[k])
            for k in keys
            if isinstance(s[k], (int, float)) and isinstance(l[k], (int, float))
        }

    for stratum in sorted(set(strict.stratified) & set(lenient.stratified)):
        delta.stratified[stratum] = {}
        for method in sorted(
            set(strict.stratified[stratum]) & set(lenient.stratified[stratum])
        ):
            s_entry = strict.stratified[stratum][method]
            l_entry = lenient.stratified[stratum][method]
            metric_keys = (set(s_entry) & set(l_entry)) - {"n"}
            d: dict[str, object] = {
                "n_strict": s_entry.get("n"),
                "n_lenient": l_entry.get("n"),
            }
            for k in metric_keys:
                if isinstance(s_entry[k], (int, float)) and isinstance(
                    l_entry[k], (int, float)
                ):
                    d[k] = float(l_entry[k] - s_entry[k])
            delta.stratified[stratum][method] = d

    return delta
