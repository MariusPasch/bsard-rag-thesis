"""
ReAct Experiments — Tier 4.2.

Runs the ReAct experiment on a given split (val / test).
Hyperparameters must be locked in output/results/agentic/ReAct/react_hyperparams.json
before calling this script — run tune_react_hyperparams.py first.

Checkpointing
-------------
Per-question results are saved to {experiment_id}_checkpoint.json after every
retrieve() call.  On interrupt, re-running the script with the same arguments
resumes from the last completed question — no work is lost.  The checkpoint
records accumulated wall-clock time across sessions so the final JSON reports
the true total runtime.

Backbones
---------
react_bm25          : bm25_tuned_k11.5_b0.25 — same first-stage as T4.0 LLM reranker
                      (BM25 Okapi, k1=1.5, b=0.25, normalization=lemmatize, field_weighting=text_only)
react_hybrid_rrf_k60: hybrid_rrf_k60 — BM25 (same) + mE5-large concat_2x, RRF k=60,
                      first_stage_k=100. Pool quality ablation vs react_bm25.
                      Requires CUDA and output/embeddings/intfloat_multilingual_e5_large_concat_2x.npy

Output
------
output/results/agentic/ReAct/react_{variant}_{split}.json
  Standard result schema + agent_loop_stats + latency_breakdown_ms_mean
  + total_experiment_wall_clock_s.

Usage
-----
    python scripts/evaluation/tier4/run_react_experiments.py \\
        --split test --variant bm25 \\
        --max-steps 5 --top-k-shown 10 --overlap-threshold 1.1

    python scripts/evaluation/tier4/run_react_experiments.py \\
        --split test --variant hybrid_rrf_k60 \\
        --max-steps 5 --top-k-shown 10 --overlap-threshold 1.1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

from bsard_evaluation import EvaluationHarness, filter_by as _bsard_filter_by
from bsard_evaluation.config import TierConfig
from evaluation.runner import (
    _to_trec_run,
    _to_trec_qrels,
    _trec_metrics_to_legacy,
    save_result,
)
from evaluation.split import load_questions
from evaluation.stratify import load_strata
from retrieval.agentic.llm_client import OllamaClient
from retrieval.agentic.llm_eval_prompts import load_fewshot_examples
from retrieval.agentic.react import ReActRetriever
from retrieval.sparse import BM25Retriever

RESULTS_DIR  = ROOT / "output" / "results" / "agentic" / "ReAct"
HPARAMS_FILE = RESULTS_DIR / "react_hyperparams.json"

EMB_DIR  = ROOT / "output" / "embeddings"
EMB_SLUG = "intfloat_multilingual_e5_large_concat_2x"

_K_FULL = [1, 5, 10, 20, 50, 100, 200, 500]


# ---------------------------------------------------------------------------
# Backbone factories
# ---------------------------------------------------------------------------

def build_bm25(df: pd.DataFrame) -> BM25Retriever:
    """BM25 Okapi k1=1.5 b=0.25 lemmatize text_only — same backbone as T4.0 LLM reranker."""
    return BM25Retriever(
        df,
        normalization="lemmatize",
        field_weighting="text_only",
        variant="okapi",
        k1=1.5,
        b=0.25,
    )


def build_hybrid(df: pd.DataFrame):
    """BM25 + mE5-large concat_2x, RRF k=60 — pool quality ablation vs react_bm25."""
    from retrieval.dense import DenseRetriever
    from retrieval.hybrid import HybridRetriever

    bm25 = build_bm25(df)
    print(f"  Building DenseRetriever (mE5-large, concat_2x, loading from {EMB_DIR})...")
    dense = DenseRetriever(
        df,
        model_name="intfloat/multilingual-e5-large",
        field_weighting="concat_2x",
        passage_prefix="passage: ",
        query_prefix="query: ",
        device="cuda",
        embeddings_dir=EMB_DIR,
    )
    return HybridRetriever(
        sparse_retriever=bm25,
        dense_retriever=dense,
        fusion_method="rrf",
        rrf_k=60,
        first_stage_k=100,
    )


BACKBONE_FACTORIES = {
    "bm25":           (build_bm25,   "bm25_tuned_k11.5_b0.25"),
    "hybrid_rrf_k60": (build_hybrid, "hybrid_rrf_k60"),
}

PREPROCESSING = {
    "bm25":           {"normalization": "lemmatize", "field_weighting": "text_only",  "embedding_prefix": "none"},
    "hybrid_rrf_k60": {"normalization": "lemmatize", "field_weighting": "concat_2x",  "embedding_prefix": "query/passage"},
}


# ---------------------------------------------------------------------------
# Single experiment runner (with per-question checkpointing)
# ---------------------------------------------------------------------------

def run_react_experiment(
    variant: str,
    split: str,
    df: pd.DataFrame,
    fewshot: dict,
    max_steps: int,
    top_k_shown: int,
    overlap_threshold: float,
    max_article_tokens: int = 1000,
) -> None:
    experiment_id = f"react_{variant}_{split}"
    out_path  = RESULTS_DIR / f"{experiment_id}.json"
    ckpt_path = RESULTS_DIR / f"{experiment_id}_checkpoint.json"

    if out_path.exists():
        r10 = json.loads(out_path.read_text()).get("metrics", {}).get("Recall@10", 0.0)
        print(f"  [{experiment_id}] Already exists (R@10={r10:.4f}) — skipping.")
        return

    print(f"\n{'='*60}")
    print(f"Experiment : {experiment_id}")
    print(f"  max_steps={max_steps}  top_k_shown={top_k_shown}  overlap_threshold={overlap_threshold}")
    print(f"{'='*60}")

    # Build backbone
    factory_fn, backbone_label = BACKBONE_FACTORIES[variant]
    print(f"Building {variant} backbone...")
    backbone = factory_fn(df)

    # Build ReAct retriever
    llm     = OllamaClient()
    lat_pre = llm.preflight()
    print(f"  Ollama preflight OK ({lat_pre:.0f} ms)")

    react = ReActRetriever(
        retriever=backbone,
        df_articles=df,
        llm_client=llm,
        top_k_shown=top_k_shown,
        max_steps=max_steps,
        overlap_threshold=overlap_threshold,
        fewshot_examples=fewshot,
        max_article_tokens=max_article_tokens,
    )

    print("  Warming up LLM (loading into GPU)...", end=" ", flush=True)
    llm.generate("Prêt.", max_tokens=1)
    print("done")

    questions = load_questions(subset=split)
    strata    = load_strata()

    # ── Load checkpoint ────────────────────────────────────────────────────────
    # Each entry: {question_id, ranked, latency_ms, trace, latency_breakdown_ms}
    checkpoint: dict[int, dict] = {}
    accumulated_wall_s: float = 0.0

    if ckpt_path.exists():
        ckpt_data = json.loads(ckpt_path.read_text())
        checkpoint = {e["question_id"]: e for e in ckpt_data.get("completed", [])}
        accumulated_wall_s = float(ckpt_data.get("accumulated_wall_s", 0.0))
        print(f"  Resuming from checkpoint: {len(checkpoint)}/{len(questions)} done "
              f"({accumulated_wall_s / 60:.1f} min accumulated).")

    remaining = [q for q in questions if q["question_id"] not in checkpoint]
    print(f"Running {len(remaining)} remaining questions (of {len(questions)} total)...")

    # ── Main question loop ─────────────────────────────────────────────────────
    t_session = time.perf_counter()

    for q in tqdm(remaining, desc="  queries", unit="q",
                  initial=len(checkpoint), total=len(questions)):
        ranked, latency_ms = react.retrieve(q["question_text"], top_k=100)
        trace  = react.get_all_traces()[-1]
        lat_bd = react.get_latency_breakdown()

        checkpoint[q["question_id"]] = {
            "question_id":          q["question_id"],
            "ranked":               ranked,
            "latency_ms":           latency_ms,
            "trace":                trace,
            "latency_breakdown_ms": lat_bd,
        }

        session_wall = time.perf_counter() - t_session
        ckpt_path.write_text(json.dumps(
            {
                "experiment_id":      experiment_id,
                "accumulated_wall_s": round(accumulated_wall_s + session_wall, 1),
                "completed":          list(checkpoint.values()),
            },
            ensure_ascii=False,
        ))

    session_wall = time.perf_counter() - t_session
    total_wall   = accumulated_wall_s + session_wall
    print(f"\nAll {len(questions)} questions done. Total wall clock: {total_wall / 60:.1f} min")

    # ── Reconstruct ordered arrays ─────────────────────────────────────────────
    results_dict = {q["question_id"]: checkpoint[q["question_id"]]["ranked"]            for q in questions}
    latencies    = [checkpoint[q["question_id"]]["latency_ms"]                          for q in questions]
    all_traces_  = [checkpoint[q["question_id"]]["trace"]                               for q in questions]
    all_bds_     = [checkpoint[q["question_id"]]["latency_breakdown_ms"]                for q in questions]
    ground_truth = {q["question_id"]: q["relevant_article_ids"]                        for q in questions}

    # ── Metrics ────────────────────────────────────────────────────────────────
    trec_run   = _to_trec_run(results_dict)
    trec_qrels = _to_trec_qrels(ground_truth)
    harness    = EvaluationHarness(TierConfig(tiers=[0, 1, 2], custom_k=_K_FULL))
    lat_dict   = {str(q["question_id"]): checkpoint[q["question_id"]]["latency_ms"] for q in questions}

    trec_metrics = harness.evaluate(
        qrels=trec_qrels,
        run=trec_run,
        latencies=lat_dict,
        queries={str(q["question_id"]): q["question_text"] for q in questions},
        verbose=False,
    )
    metrics = _trec_metrics_to_legacy(trec_metrics)
    if "MAP@100" in metrics and "MAP" not in metrics:
        metrics["MAP"] = metrics["MAP@100"]

    arr = np.array(latencies)
    latency_distribution = {
        "mean": float(arr.mean()), "std":  float(arr.std()),
        "p50":  float(np.percentile(arr, 50)), "p90": float(np.percentile(arr, 90)),
        "p95":  float(np.percentile(arr, 95)), "p99": float(np.percentile(arr, 99)),
        "min":  float(arr.min()),              "max": float(arr.max()),
        "index_build_s": 0.0,
    }

    str_strata = {str(k): v for k, v in strata.items()}

    def eval_stratum(field: str, value: str) -> dict:
        sub_qrels = _bsard_filter_by(trec_qrels, str_strata, field, value)
        if not sub_qrels:
            return {}
        sub_run = {qid: trec_run[qid] for qid in sub_qrels if qid in trec_run}
        return _trec_metrics_to_legacy(
            harness.evaluate(qrels=sub_qrels, run=sub_run, verbose=False)
        )

    stratified = {
        "single_article":           eval_stratum("article_count", "single_article"),
        "multi_article":            eval_stratum("article_count", "multi_article"),
        "lexically_aligned":        eval_stratum("lex_align", "lexically_aligned"),
        "semantically_paraphrased": eval_stratum("lex_align", "semantically_paraphrased"),
        "with_cross_refs":          eval_stratum("cross_ref", "with_cross_refs"),
        "without_cross_refs":       eval_stratum("cross_ref", "without_cross_refs"),
    }

    # ── Loop stats + latency breakdown from checkpoint data ────────────────────
    n = len(questions)
    steps_per_query     = [len(t) for t in all_traces_]
    finished            = sum(1 for t in all_traces_ if any(e.get("tool_name") == "finish" for e in t))
    terminated_by_limit = sum(
        1 for t in all_traces_
        if t and not any(e.get("tool_name") == "finish" for e in t) and len(t) >= max_steps
    )
    total_overlap   = sum(sum(1 for e in t if e.get("overlap_guard_fired", False)) for t in all_traces_)
    total_preflight = sum(sum(1 for e in t if e.get("preflight_failed",   False)) for t in all_traces_)

    loop_stats = {
        "n_queries":                            n,
        "mean_steps_per_query":                 float(np.mean(steps_per_query)),
        "fraction_queries_converged_finish":    finished / n,
        "fraction_queries_terminated_by_limit": terminated_by_limit / n,
        "total_overlap_guard_fires":            total_overlap,
        "total_preflight_failures":             total_preflight,
    }

    keys = ("llm_generate", "retrieval", "rerank_posthoc")
    latency_breakdown_mean = {
        k: round(float(np.mean([b.get(k, 0.0) for b in all_bds_])), 1)
        for k in keys
    }

    # ── Assemble result dict ───────────────────────────────────────────────────
    result = {
        "experiment_id":    experiment_id,
        "timestamp":        time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model_or_method":  "ReActRetriever",
        "hyperparameters": {
            "retrieval_backbone": backbone_label,
            "llm_backbone":       "llama3.1:8b",
            "max_steps":          max_steps,
            "top_k_shown":        top_k_shown,
            "overlap_threshold":  overlap_threshold,
            "max_article_tokens": max_article_tokens,
            "d1_reranking":       "llama3.1:8b_binary",
            "d2_overlap_guard":   f"jaccard>={overlap_threshold}",
        },
        "preprocessing":    PREPROCESSING[variant],
        "token_length_audit": {"fraction_truncated": 0.0, "max_tokens_observed": 0},
        "training_regime":  "zero_shot",
        "latency_ms_mean":  latency_distribution["mean"],
        "latency_ms_std":   latency_distribution["std"],
        "latency_distribution":           latency_distribution,
        "metrics":                        metrics,
        "significance_vs_anchor":         {"p_value_recall10": None, "significant": None},
        "stratified":                     stratified,
        "agent_loop_stats":               loop_stats,
        "latency_breakdown_ms_mean":      latency_breakdown_mean,
        "total_experiment_wall_clock_s":  round(total_wall, 1),
        "_trec_run":   trec_run,
        "_trec_qrels": trec_qrels,
    }

    # ── Print summary ──────────────────────────────────────────────────────────
    m = result["metrics"]
    print(f"\n  R@10   = {m.get('Recall@10', 0.0):.4f}")
    print(f"  R@100  = {m.get('Recall@100', 0.0):.4f}")
    print(f"  MRR@10 = {m.get('MRR@10', 0.0):.4f}")
    print(f"  Latency mean = {result['latency_ms_mean']:.0f} ms/query")
    print(f"  Wall clock   = {total_wall / 60:.1f} min")

    if loop_stats:
        print(f"\n  Loop stats:")
        print(f"    mean_steps_per_query               : {loop_stats.get('mean_steps_per_query', 0):.1f}")
        print(f"    fraction_queries_converged_finish  : {loop_stats.get('fraction_queries_converged_finish', 0):.1%}")
        print(f"    fraction_queries_terminated_by_limit: {loop_stats.get('fraction_queries_terminated_by_limit', 0):.1%}")

        frac_lim = loop_stats.get("fraction_queries_terminated_by_limit", 0)
        if frac_lim > 0.30:
            print(
                f"\n  *** WARNING: {frac_lim:.0%} of {split} queries hit max_steps={max_steps}. "
                "Consider re-tuning with a higher max_steps."
            )

    # ── Save result ────────────────────────────────────────────────────────────
    saved = save_result(result, results_dir=RESULTS_DIR)
    print(f"\n  Saved: {saved}")

    # ── Save traces ────────────────────────────────────────────────────────────
    traces_with_ids = [
        {"question_id": q["question_id"], "trace": checkpoint[q["question_id"]]["trace"]}
        for q in questions
    ]
    traces_path = RESULTS_DIR / f"{experiment_id}_traces.json"
    traces_path.write_text(json.dumps(traces_with_ids, indent=2, ensure_ascii=False))
    print(f"  Traces: {traces_path}")

    # ── Clean up checkpoint (clean completion) ─────────────────────────────────
    if ckpt_path.exists():
        ckpt_path.unlink()
    print(f"  Checkpoint cleaned up.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split", choices=["val", "test"], required=True,
        help="Evaluation split to run.",
    )
    parser.add_argument(
        "--variant", choices=["bm25", "hybrid_rrf_k60"], required=True,
        help="Backbone variant. bm25 = T4.0 first-stage; hybrid_rrf_k60 = pool quality ablation.",
    )
    parser.add_argument("--max-steps",          type=int,   required=True)
    parser.add_argument("--top-k-shown",        type=int,   required=True)
    parser.add_argument("--overlap-threshold",  type=float, required=True)
    parser.add_argument("--max-article-tokens", type=int,   default=1000)
    args = parser.parse_args()

    if args.split == "test":
        print("*** TEST SPLIT — ensure val results have been reviewed first! ***\n")

    if not HPARAMS_FILE.exists():
        sys.exit(
            "ERROR: react_hyperparams.json not found.\n"
            "Run: python scripts/evaluation/tier4/tune_react_hyperparams.py"
        )

    fewshot = load_fewshot_examples()
    df      = pd.read_parquet(ROOT / "output" / "bsard_articles_dedup.parquet")

    run_react_experiment(
        variant=args.variant,
        split=args.split,
        df=df,
        fewshot=fewshot,
        max_steps=args.max_steps,
        top_k_shown=args.top_k_shown,
        overlap_threshold=args.overlap_threshold,
        max_article_tokens=args.max_article_tokens,
    )


if __name__ == "__main__":
    main()
