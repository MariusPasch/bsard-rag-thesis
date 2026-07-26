"""
Experiment runner — produces a fully-populated result JSON for one retriever run.

Decisions implemented:
  E-D4  Per-query latency logged with time.perf_counter()
  E-D3  Two-sided paired t-test vs anchor on per-query Recall@10
  X-D2  Full result JSON schema with all required fields

Metric computation, significance testing, and stratified evaluation are
delegated to the external bsard_evaluation package (RQ3).

Evaluation tiers (per TIER1_SPARSE_RETRIEVAL_PLAN.md §3):
  Tier 0 — latency distribution (mean, std, p50, p90, p95, p99, min, max) + index build time
  Tier 1 — BSARD paper metrics: Recall@{1,5,10,100}, MRR@100
  Tier 2 — Full IR: P1 Recall/Precision/F1/HitRate@{1,5,10,20,50,100,200,500},
                    P2 MRR/MAP/NDCG@{10,100}
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from bsard_evaluation import (
    EvaluationHarness,
    per_query_recall as _bsard_per_query_recall,
    filter_by as _bsard_filter_by,
)
from bsard_evaluation.config import TierConfig
from bsard_evaluation.significance import _paired_test

# ---------------------------------------------------------------------------
# Evaluation configuration (Tier 0 + 1 + 2, per plan §3)
# ---------------------------------------------------------------------------
_K_FULL = [1, 5, 10, 20, 50, 100, 200, 500]  # Tier 2 P1 Recall@k + Tier 1 R@k
_TIER_CFG = TierConfig(tiers=[0, 1, 2], custom_k=_K_FULL)

# Results and large artefacts live under the data root resolved by
# evaluation.paths (BSARD_DATA_DIR env var, or <repo>/output by default).
# See scripts/download_data.py to fetch the data bundle from Hugging Face.
from evaluation.paths import RESULTS_DIR  # noqa: E402

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Format adapters: RQ1 list-based → TREC dict-based
# ---------------------------------------------------------------------------

def _to_trec_run(results: dict[int, list[int]]) -> dict[str, dict[str, float]]:
    """
    Convert RQ1 ranked-list results to TREC run format.

    RQ1 format : {question_id: [article_id, ...]}  (ranked, best first)
    TREC format: {query_id: {doc_id: score}}        (higher = better)
    """
    trec_run: dict[str, dict[str, float]] = {}
    for qid, ranked in results.items():
        trec_run[str(qid)] = {
            str(doc_id): float(len(ranked) - rank)
            for rank, doc_id in enumerate(ranked)
        }
    return trec_run


def build_contexts_with_ranks(
    run: dict[int, list[int]],
    corpus: dict[int, str],
    k: int = 10,
) -> dict[str, list[tuple[str, str, int]]]:
    """
    Convert ranked retrieval output to contexts_with_ranks format for Tier 3.

    Parameters
    ----------
    run    : {question_id: [article_id, ...]} ranked best-first
    corpus : {article_id: article_text} — built once at experiment startup
    k      : number of top documents to include

    Returns
    -------
    {str(question_id): [(str(article_id), corpus.get(article_id, ""), rank), ...]}
    """
    out = {}
    for qid, ranked_ids in run.items():
        out[str(qid)] = [
            (str(article_id), corpus.get(article_id, ""), rank)
            for rank, article_id in enumerate(ranked_ids[:k], start=1)
        ]
    return out


def _to_trec_qrels(ground_truth: dict[int, list[int]]) -> dict[str, dict[str, int]]:
    """
    Convert RQ1 ground-truth lists to TREC qrels format.

    RQ1 format : {question_id: [article_id, ...]}
    TREC format: {query_id: {doc_id: relevance_grade}}  (binary: 1)
    """
    trec_qrels: dict[str, dict[str, int]] = {}
    for qid, relevant in ground_truth.items():
        trec_qrels[str(qid)] = {str(doc_id): 1 for doc_id in relevant}
    return trec_qrels


def _trec_metrics_to_legacy(trec_metrics: dict[str, float]) -> dict[str, float]:
    """
    Convert bsard_evaluation namespaced metric keys to legacy RQ1 flat keys.

    This ensures backward compatibility with existing result JSON consumers
    (analysis notebooks, significance scripts, etc.).

    Tier precedence for Recall@k: T1/R@k takes priority over T2/P1/Recall@k
    so the BSARD-paper-compatible value is always used.

    Example: "T1/R@10" → "Recall@10", "T1/MRR@100" → "MRR@100"
    """
    legacy: dict[str, float] = {}
    for key, value in trec_metrics.items():
        # Tier 1 keys (BSARD paper — highest priority for Recall/MRR)
        if key.startswith("T1/R@"):
            k = key.split("@")[1]
            legacy[f"Recall@{k}"] = value
        elif key.startswith("T1/MRR@"):
            k = key.split("@")[1]
            legacy[f"MRR@{k}"] = value
        # Tier 2 Panel 1 keys
        elif key.startswith("T2/P1/Recall@"):
            k = key.split("@")[1]
            recall_key = f"Recall@{k}"
            if recall_key not in legacy:  # T1 takes precedence
                legacy[recall_key] = value
        elif key.startswith("T2/P1/Precision@"):
            k = key.split("@")[1]
            legacy[f"Precision@{k}"] = value
        elif key.startswith("T2/P1/F1@"):
            k = key.split("@")[1]
            legacy[f"F1@{k}"] = value
        elif key.startswith("T2/P1/HitRate@"):
            k = key.split("@")[1]
            legacy[f"HitRate@{k}"] = value
        # Tier 2 Panel 2 keys
        elif key.startswith("T2/P2/NDCG@"):
            k = key.split("@")[1]
            legacy[f"NDCG@{k}"] = value
        elif key.startswith("T2/P2/MAP@"):
            k = key.split("@")[1]
            legacy[f"MAP@{k}"] = value
        elif key.startswith("T2/P2/MRR@"):
            k = key.split("@")[1]
            mrr_key = f"MRR@{k}"
            if mrr_key not in legacy:  # T1 takes precedence
                legacy[mrr_key] = value
        elif key.startswith("T2/P3/IDPrecision@"):
            k = key.split("@")[1]
            legacy[f"IDPrecision@{k}"] = value
        elif key.startswith("T2/P3/IDRecall@"):
            k = key.split("@")[1]
            legacy[f"IDRecall@{k}"] = value
        # Tier 0 keys — pass through with cleaner names
        elif key.startswith("T0/"):
            legacy[key] = value
    return legacy


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_experiment(
    retriever,
    questions: list[dict],
    experiment_id: str,
    hyperparameters: dict,
    preprocessing: dict,
    strata: dict[int, dict],
    corpus: dict[int, str] = None,
    top_k: int = 100,
    index_build_time_s: float = 0.0,
) -> dict:
    """
    Run one retrieval experiment over *questions* and return a result dict.

    Parameters
    ----------
    retriever           : any object with .retrieve(query, top_k) -> (list[int], float)
    questions           : list of question dicts (from evaluation.split.load_questions)
    experiment_id       : unique string identifier, used as the output filename
    hyperparameters     : dict of model/index hyperparameters to record
    preprocessing       : dict matching the schema's preprocessing block
    strata              : {question_id: strata_dict} from evaluation.stratify
    top_k               : candidate pool size
    index_build_time_s  : one-off cost to build the retrieval index (seconds), for Tier 0
    """
    from tqdm import tqdm
    results: dict[int, list[int]] = {}
    latencies: list[float] = []

    for q in tqdm(questions, desc="  queries", unit="q"):
        ranked, latency_ms = retriever.retrieve(q["question_text"], top_k=top_k)
        results[q["question_id"]] = ranked
        latencies.append(latency_ms)

    ground_truth = {q["question_id"]: q["relevant_article_ids"] for q in questions}

    # ── Convert to TREC format for bsard_evaluation ─────────────────────────
    trec_run   = _to_trec_run(results)
    trec_qrels = _to_trec_qrels(ground_truth)

    # ── Aggregate metrics via bsard_evaluation harness (Tier 0 + 1 + 2) ─────
    harness = EvaluationHarness(_TIER_CFG)
    latency_dict = {str(q["question_id"]): lat for q, lat in zip(questions, latencies)}
    timing_breakdown = {"index_build_s": index_build_time_s} if index_build_time_s else {}
    
    contexts_with_ranks = None
    if corpus is not None:
        contexts_with_ranks = build_contexts_with_ranks(results, corpus, k=10)

    trec_metrics = harness.evaluate(
        qrels=trec_qrels,
        run=trec_run,
        latencies=latency_dict,
        timing_breakdown=timing_breakdown,
        queries={str(q["question_id"]): q["question_text"] for q in questions},
        contexts_with_ranks=contexts_with_ranks,
        verbose=False,
    )
    metrics = _trec_metrics_to_legacy(trec_metrics)

    # ── Ensure MAP alias exists for backward compat ─────────────────────────
    if "MAP@100" in metrics and "MAP" not in metrics:
        metrics["MAP"] = metrics["MAP@100"]

    # ── Tier 0: full latency distribution ───────────────────────────────────
    arr = np.array(latencies)
    latency_distribution = {
        "mean":  float(arr.mean()),
        "std":   float(arr.std()),
        "p50":   float(np.percentile(arr, 50)),
        "p90":   float(np.percentile(arr, 90)),
        "p95":   float(np.percentile(arr, 95)),
        "p99":   float(np.percentile(arr, 99)),
        "min":   float(arr.min()),
        "max":   float(arr.max()),
        "index_build_s": index_build_time_s,
    }

    # ── Stratified metrics via bsard_evaluation ─────────────────────────────
    str_strata = {str(k): v for k, v in strata.items()}

    def eval_stratum(field: str, value: str) -> dict:
        sub_qrels = _bsard_filter_by(trec_qrels, str_strata, field, value)
        if not sub_qrels:
            return {}
        sub_run = {qid: trec_run[qid] for qid in sub_qrels if qid in trec_run}
        sub_metrics = harness.evaluate(
            qrels=sub_qrels, run=sub_run, verbose=False,
        )
        return _trec_metrics_to_legacy(sub_metrics)

    stratified = {
        "single_article":           eval_stratum("article_count", "single_article"),
        "multi_article":            eval_stratum("article_count", "multi_article"),
        "lexically_aligned":        eval_stratum("lex_align", "lexically_aligned"),
        "semantically_paraphrased": eval_stratum("lex_align", "semantically_paraphrased"),
        "with_cross_refs":          eval_stratum("cross_ref", "with_cross_refs"),
        "without_cross_refs":       eval_stratum("cross_ref", "without_cross_refs"),
    }

    result = {
        "experiment_id":    experiment_id,
        "timestamp":        time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model_or_method":  _infer_method(retriever),
        "hyperparameters":  hyperparameters,
        "preprocessing":    preprocessing,
        "token_length_audit": getattr(retriever, "audit", {"fraction_truncated": 0.0, "max_tokens_observed": 0}),
        "training_regime":  "zero_shot",
        # Tier 0: top-level aliases kept for backward compat; full block added
        "latency_ms_mean":  latency_distribution["mean"],
        "latency_ms_std":   latency_distribution["std"],
        "latency_distribution": latency_distribution,
        "metrics":          metrics,
        "significance_vs_anchor": {"p_value_recall10": None, "significant": None},
        "stratified":       stratified,
        # Internal — stripped before final save; used by add_significance()
        "_raw_results":     results,
        "_raw_gt":          ground_truth,
        "_trec_run":        trec_run,
        "_trec_qrels":      trec_qrels,
    }
    return result


def add_significance(
    result: dict,
    anchor_result: dict,
    k_values: list[int] | None = None,
    primary_k: int = 10,
) -> dict:
    """
    Compute two-sided paired t-tests on per-query Recall@k vs the anchor for
    each k in k_values. primary_k determines the 'significant' flag.

    Tier 1 default: primary_k=10.  Tier 2 default: primary_k=100.

    Delegates to bsard_evaluation.significance for metric computation.
    Supports both new (_trec_*) and legacy (_raw_*) internal keys.
    """
    if k_values is None:
        k_values = [primary_k]

    # Resolve TREC-format data (prefer new keys, fall back to legacy)
    if "_trec_qrels" in result:
        qrels = result["_trec_qrels"]
        exp_run = result["_trec_run"]
        anc_run = anchor_result["_trec_run"]
    else:
        # Legacy path: convert on the fly
        qrels = _to_trec_qrels(result["_raw_gt"])
        exp_run = _to_trec_run(result["_raw_results"])
        anc_run = _to_trec_run(anchor_result["_raw_results"])

    p_values = {}
    for k in k_values:
        scores_exp = _bsard_per_query_recall(qrels, exp_run, k)
        scores_anc = _bsard_per_query_recall(qrels, anc_run, k)
        p_values[k] = round(_paired_test(scores_exp, scores_anc, "ttest"), 4)

    sig_block = {f"p_value_recall{k}": p_values[k] for k in k_values}
    sig_block["significant"] = bool(p_values[primary_k] < 0.05)
    # Always keep p_value_recall10 for Tier 1 backward compatibility
    if 10 not in p_values:
        sig_block["p_value_recall10"] = None
    result["significance_vs_anchor"] = sig_block
    return result


def save_result(result: dict, path: Path | None = None, results_dir: Path | None = None) -> Path:
    """
    Strip internal keys and write the result JSON to disk.
    Returns the path written.
    """
    clean = {k: v for k, v in result.items() if not k.startswith("_") or k in ("_trec_run", "_trec_qrels")}
    if path is None:
        out_dir = results_dir if results_dir is not None else RESULTS_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{result['experiment_id']}.json"
    path.write_text(json.dumps(clean, indent=2, ensure_ascii=False))
    return path


# ---------------------------------------------------------------------------
# Legacy compatibility: per_query_recall re-export
# ---------------------------------------------------------------------------

def per_query_recall(
    results: dict[int, list[int]],
    ground_truth: dict[int, list[int]],
    k: int = 10,
) -> list[float]:
    """
    Legacy wrapper: accepts RQ1 list-based format and delegates to
    bsard_evaluation.per_query_recall (TREC format) internally.

    Kept for backward compatibility with scripts that import from
    evaluation.runner or evaluation.metrics.
    """
    trec_qrels = _to_trec_qrels(ground_truth)
    trec_run = _to_trec_run(results)
    return _bsard_per_query_recall(trec_qrels, trec_run, k)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _infer_method(retriever) -> str:
    name = type(retriever).__name__
    if "BM25" in name:
        return "bm25"
    if "TFIDF" in name:
        return "tfidf"
    if "FTS5" in name:
        return "fts5"
    if "Dense" in name:
        return "dense_biencoder"
    return name.lower()
