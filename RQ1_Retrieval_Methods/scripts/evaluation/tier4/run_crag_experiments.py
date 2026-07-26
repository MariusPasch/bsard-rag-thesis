"""
CRAG Experiments — Tier 4.1 Step 7 (collective LLM evaluator, v2 run).

Runs three CRAG variants on the test split: bm25, dense, hybrid.
BM25 backbone aligned with T4.0 first stage: bm25_tuned_k11.5_b0.25
(BM25 Okapi, k1=1.5, b=0.25, lemmatize, text_only).

eval_k selection
----------------
If --eval-k is given, that value is used directly (no probe).
Otherwise, auto_select_eval_k() probes --eval-k-candidates (largest first)
on a small val sample and picks the largest one whose estimated per-variant
wall-clock fits within --max-minutes. Falls back to the smallest candidate
if none fit.

Usage:
    # Standard: auto-select eval_k, then run all test variants:
    python scripts/evaluation/tier4/run_crag_experiments.py --split test

    # Lock eval_k (skips probe):
    python scripts/evaluation/tier4/run_crag_experiments.py --split test --eval-k 10

    # Single variant:
    python scripts/evaluation/tier4/run_crag_experiments.py --split test --variant hybrid --eval-k 10

    # Custom budget / wider candidate ladder:
    python scripts/evaluation/tier4/run_crag_experiments.py --split test --max-minutes 120 --eval-k-candidates 50,20,10,5

    # T4.0-aligned ablation (backbone_top_k=20, eval_k=20, skips probe):
    python scripts/evaluation/tier4/run_crag_experiments.py --split test --aligned
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.runner import run_experiment, save_result
from evaluation.split import load_questions, make_ground_truth
from evaluation.stratify import load_strata
from retrieval.agentic.crag import CRAGRetriever
from retrieval.agentic.llm_client import OllamaClient
from retrieval.agentic.llm_eval_prompts import (
    LLM_JUDGE_COLLECTIVE_STRICT_PROMPT,
    format_articles_block,
    load_fewshot_examples,
    parse_binary_judgment,
)
from retrieval.dense import DenseRetriever
from retrieval.hybrid import HybridRetriever
from retrieval.sparse import BM25Retriever

RESULTS_DIR = ROOT / "output" / "results" / "agentic" / "CRAG"

# Fixed estimates for latency components other than the eval call (ms).
# Derived from the aligned run (Azure T4 GPU, empirical).
_RETRIEVAL_MS   = 2_000   # backbone retrieval per iteration
_REWRITE_STD_MS = 2_800   # standard rewrite (iter 0, CRAG_REWRITE_PROMPT)
_REWRITE_ASP_MS = 10_000  # aspect extract + focused rewrite (iter 1+, two calls)


# ---------------------------------------------------------------------------
# Backbone factories
# ---------------------------------------------------------------------------

def build_bm25(df: pd.DataFrame) -> BM25Retriever:
    """
    T4.0-aligned BM25 backbone: bm25_tuned_k11.5_b0.25.
    Matches T4.0 first-stage retriever (k1=1.5, b=0.25, lemmatize, text_only).
    R@10=0.2651, R@100=0.5210, MRR@10=0.2628 on test split.
    """
    return BM25Retriever(
        df,
        normalization="lemmatize",
        field_weighting="text_only",
        variant="okapi",
        k1=1.5,
        b=0.25,
    )


def build_dense(df: pd.DataFrame) -> DenseRetriever:
    return DenseRetriever(
        df,
        model_name="intfloat/multilingual-e5-large-instruct",
        field_weighting="text_only",
    )


def build_hybrid(df: pd.DataFrame) -> HybridRetriever:
    """
    Best Tier 3 config: linear fusion alpha=0.9, mE5-large-instruct concat_2x.
    hybrid_linear_best_test: R@10=0.3941, R@100=0.6454 (best across Tier 3).
    Pre-computed embeddings from output/embeddings/
      intfloat_multilingual_e5_large_instruct_concat_2x.npy.
    """
    sparse = BM25Retriever(
        df,
        normalization="lemmatize",
        field_weighting="concat_2x",
        variant="okapi",
        k1=1.5,
        b=0.75,
    )
    dense = DenseRetriever(
        df,
        model_name="intfloat/multilingual-e5-large-instruct",
        field_weighting="concat_2x",
        query_prefix=(
            "Instruct: Given a legal question in French, retrieve the relevant "
            "Belgian statutory article\nQuery: "
        ),
        passage_prefix="",
    )
    return HybridRetriever(
        sparse_retriever=sparse,
        dense_retriever=dense,
        fusion_method="linear",
        alpha=0.9,
        first_stage_k=100,
    )


def build_hybrid_rrf_k60(df: pd.DataFrame) -> HybridRetriever:
    """
    T4.0-aligned hybrid backbone: hybrid_rrf_k60.
    Identical first stage to llm_rerank_binary_top50_hybrid_rrf_k60_test (T4.0-hybrid).
    BM25 (k1=1.5, b=0.25, lemmatize, text_only) + mE5-large concat_2x, RRF k=60.
    Pre-computed embeddings: intfloat_multilingual_e5_large_concat_2x.npy.
    """
    sparse = BM25Retriever(
        df,
        normalization="lemmatize",
        field_weighting="text_only",
        variant="okapi",
        k1=1.5,
        b=0.25,
    )
    dense = DenseRetriever(
        df,
        model_name="intfloat/multilingual-e5-large",
        field_weighting="concat_2x",
        passage_prefix="passage: ",
        query_prefix="query: ",
        embeddings_dir=ROOT / "output" / "embeddings",
    )
    return HybridRetriever(
        sparse_retriever=sparse,
        dense_retriever=dense,
        fusion_method="rrf",
        rrf_k=60,
        first_stage_k=100,
    )


BACKBONE_FACTORIES = {
    "bm25":            (build_bm25,            "bm25_tuned_k11.5_b0.25"),
    "dense":           (build_dense,           "dense_me5_large_instruct"),
    "hybrid":          (build_hybrid,          "hybrid_linear_alpha0.9_me5_large_instruct_concat2x"),
    "hybrid_rrf_k60":  (build_hybrid_rrf_k60,  "hybrid_rrf_k60"),
}

PREPROCESSING = {
    "bm25":           {"normalization": "lemmatize", "field_weighting": "text_only",
                       "embedding_prefix": "none"},
    "dense":          {"normalization": "none",      "field_weighting": "text_only",
                       "embedding_prefix": "query:/passage:"},
    "hybrid":         {"normalization": "lemmatize", "field_weighting": "concat_2x",
                       "embedding_prefix": "query:/passage:"},
    "hybrid_rrf_k60": {"normalization": "lemmatize", "field_weighting": "concat_2x",
                       "embedding_prefix": "query: /passage: "},
}


# ---------------------------------------------------------------------------
# eval_k probe and auto-selection
# ---------------------------------------------------------------------------

def probe_eval_k(
    eval_k: int,
    df: pd.DataFrame,
    llm_client: OllamaClient,
    probe_questions: list[dict],
    backbone_top_k: int = 100,
    max_article_tokens: int = 300,
) -> dict:
    """
    Run one collective-eval pass per probe question using the T4.0-aligned BM25
    backbone (bm25_tuned_k11.5_b0.25).  No rewrites — only retrieval + one LLM
    call per question.

    Returns
    -------
    {
        "eval_k":        int,
        "n_probed":      int,
        "mean_eval_ms":  float,   # mean LLM call time at this eval_k
        "oui_rate":      float,   # fraction of Oui (Correct first-pass) responses
    }
    """
    bm25   = build_bm25(df)
    corpus = _build_corpus(df)

    eval_times: list[float] = []
    oui_count = 0

    for q in probe_questions:
        ranked, _ = bm25.retrieve(q["question_text"], top_k=backbone_top_k)
        texts = [corpus.get(aid, "") for aid in ranked[:eval_k]]
        articles_block = format_articles_block(texts, max_article_tokens)
        prompt = LLM_JUDGE_COLLECTIVE_STRICT_PROMPT.format(
            question=q["question_text"],
            articles_block=articles_block,
        )
        response, lat_ms = llm_client.generate(prompt, temperature=0.0, max_tokens=80)
        eval_times.append(lat_ms)
        is_correct, _ = parse_binary_judgment(response)
        if is_correct:
            oui_count += 1

    return {
        "eval_k":       eval_k,
        "n_probed":     len(probe_questions),
        "mean_eval_ms": sum(eval_times) / len(eval_times),
        "oui_rate":     oui_count / len(probe_questions),
    }


def estimate_total_minutes(
    mean_eval_ms: float,
    oui_rate: float,
    n_questions: int = 222,
) -> float:
    """
    Estimate per-variant wall-clock time (minutes) from probe results.

    Three-scenario model
    --------------------
    A  First-pass Correct (oui_rate fraction):
         retrieval + eval
    B  One standard rewrite then Correct (85% of remaining):
         2×retrieval + eval + std_rewrite + eval
    C  Two rewrites / limit-hit (15% of remaining):
         3×retrieval + eval + std_rewrite + eval + aspect_rewrite + eval

    Fixed latency constants (_RETRIEVAL_MS, _REWRITE_STD_MS, _REWRITE_ASP_MS)
    are empirical estimates from the aligned run on Azure T4.
    """
    p_a = oui_rate
    p_b = (1.0 - oui_rate) * 0.85
    p_c = (1.0 - oui_rate) * 0.15

    lat_a = _RETRIEVAL_MS + mean_eval_ms
    lat_b = 2 * _RETRIEVAL_MS + mean_eval_ms + _REWRITE_STD_MS + mean_eval_ms
    lat_c = (3 * _RETRIEVAL_MS + mean_eval_ms + _REWRITE_STD_MS
             + mean_eval_ms + _REWRITE_ASP_MS + mean_eval_ms)

    mean_query_ms = p_a * lat_a + p_b * lat_b + p_c * lat_c
    return (n_questions * mean_query_ms / 1_000) / 60.0


def auto_select_eval_k(
    df: pd.DataFrame,
    llm_client: OllamaClient,
    candidates: list[int],
    max_minutes: float,
    probe_n: int,
    backbone_top_k: int,
    max_article_tokens: int,
    n_full_questions: int = 222,
) -> int:
    """
    Probe each eval_k candidate (largest first) on a small val sample and
    return the largest one whose estimated per-variant wall-clock ≤ max_minutes.
    Falls back to the smallest candidate if none fit.

    Prints a selection table during probing.
    """
    probe_questions = load_questions(subset="val")[:probe_n]
    selected        = candidates[-1]   # fallback = smallest

    print(f"\nProbing eval_k candidates {candidates} "
          f"(budget: {max_minutes:.0f} min/variant, "
          f"probe n={probe_n} val questions)")
    print(f"\n  {'eval_k':>7}  {'eval_ms':>9}  {'Oui%':>6}  "
          f"{'Est.(min)':>10}  {'Budget':>10}  Decision")
    print("  " + "-" * 60)

    for candidate in candidates:
        stats   = probe_eval_k(
            eval_k=candidate,
            df=df,
            llm_client=llm_client,
            probe_questions=probe_questions,
            backbone_top_k=backbone_top_k,
            max_article_tokens=max_article_tokens,
        )
        est_min = estimate_total_minutes(
            mean_eval_ms=stats["mean_eval_ms"],
            oui_rate=stats["oui_rate"],
            n_questions=n_full_questions,
        )
        fits     = est_min <= max_minutes
        decision = "SELECTED ✓" if fits else "too slow  ✗"
        print(f"  {candidate:>7}  {stats['mean_eval_ms']:>9.0f}  "
              f"{stats['oui_rate']:>5.1%}  "
              f"{est_min:>9.1f}  "
              f"{max_minutes:>9.0f}  "
              f"  {decision}")

        if fits:
            selected = candidate
            break   # largest that fits — stop here

    print(f"\n  → Selected eval_k = {selected}")
    return selected


# ---------------------------------------------------------------------------
# Backbone builder (handles pre-computed embeddings for dense/hybrid)
# ---------------------------------------------------------------------------

def build_backbone_with_embeddings(variant: str, df: pd.DataFrame, factory_fn):
    """Build backbone, loading pre-computed embeddings for dense/hybrid variants."""
    if variant in ("bm25", "hybrid_rrf_k60"):
        # hybrid_rrf_k60 factory uses embeddings_dir internally (DenseRetriever auto-loads)
        return factory_fn(df)

    emb_dir  = ROOT / "output" / "embeddings"
    emb_path = emb_dir / "intfloat_multilingual_e5_large_instruct_concat_2x.npy"
    ids_path = emb_dir / "intfloat_multilingual_e5_large_instruct_concat_2x_ids.npy"

    if emb_path.exists() and ids_path.exists():
        import numpy as np
        emb_matrix = np.load(str(emb_path))
        emb_ids    = np.load(str(ids_path)).tolist()
        print(f"  Loaded pre-computed embeddings: {emb_matrix.shape}")

        if variant == "dense":
            retriever = DenseRetriever(
                df,
                model_name="intfloat/multilingual-e5-large-instruct",
                field_weighting="text_only",
            )
            retriever.load_embeddings(emb_matrix, emb_ids)
            return retriever
        else:   # hybrid (instruct/linear) — factory handles embedding loading internally
            return factory_fn(df)
    else:
        print(f"  WARNING: Pre-computed embeddings not found at {emb_dir}")
        print("  Building embeddings from scratch (slow)...")
        return factory_fn(df)


# ---------------------------------------------------------------------------
# Single experiment runner
# ---------------------------------------------------------------------------

def run_crag_experiment(
    variant: str,
    split: str,
    df: pd.DataFrame,
    eval_k: int,
    fewshot: dict,
    max_iterations: int = 2,
    max_article_tokens: int = 300,
    backbone_top_k: int = 100,
    aligned: bool = False,
):
    suffix        = "_aligned" if aligned else ""
    experiment_id = f"crag_{variant}{suffix}_{split}_v2"
    out_path      = RESULTS_DIR / f"{experiment_id}.json"

    if out_path.exists():
        print(f"  [{experiment_id}] Already exists — skipping. Delete to re-run.")
        return

    print(f"\n{'='*60}")
    print(f"Experiment : {experiment_id}")
    print(f"  eval_k             = {eval_k}")
    print(f"  max_iterations     = {max_iterations}")
    print(f"  max_article_tokens = {max_article_tokens}")
    print(f"  backbone_top_k     = {backbone_top_k}")
    print(f"{'='*60}")

    factory_fn, backbone_label = BACKBONE_FACTORIES[variant]
    print(f"Building {variant} backbone...")
    backbone = build_backbone_with_embeddings(variant, df, factory_fn)

    llm     = OllamaClient()
    llm_lat = llm.preflight()
    print(f"  Ollama preflight OK ({llm_lat:.0f} ms)")

    crag = CRAGRetriever(
        retriever=backbone,
        df_articles=df,
        llm_client=llm,
        eval_k=eval_k,
        max_iterations=max_iterations,
        backbone_top_k=backbone_top_k,
        fewshot_examples=fewshot,
        max_article_tokens=max_article_tokens,
    )

    questions = load_questions(subset=split)
    strata    = load_strata()
    gt        = make_ground_truth(questions)
    qid_map   = {q["question_text"]: q["question_id"] for q in questions}

    # ── Per-question checkpoint (interrupt resume) ─────────────────────────────
    ckpt_path = RESULTS_DIR / f"{experiment_id}.ckpt.json"
    per_q_checkpoint: dict[str, list[int]] = {}
    if ckpt_path.exists():
        per_q_checkpoint = json.loads(ckpt_path.read_text(encoding="utf-8"))
        n_ckpt   = len(per_q_checkpoint)
        n_remain = len(questions) - n_ckpt
        print(f"  Checkpoint loaded: {n_ckpt}/{len(questions)} done"
              f" — {n_remain} remaining")
    else:
        print(f"  No checkpoint — starting from question 1/{len(questions)}")

    CKPT_EVERY   = 5      # save checkpoint every N fresh questions
    _fresh_count = [0]
    _running_r10 = []
    _fresh_lats  = []
    _t0_run      = time.perf_counter()
    _orig_retrieve = crag.retrieve

    def _retrieve_checkpointed(query: str, top_k: int = 100) -> tuple[list[int], float]:
        qid     = qid_map.get(query, hash(query) & 0x7FFFFFFF)
        qid_str = str(qid)
        rel     = gt.get(qid, [])

        # Fast path: already processed — serve from checkpoint instantly
        if qid_str in per_q_checkpoint:
            ids = per_q_checkpoint[qid_str]
            if rel:
                _running_r10.append(len(set(ids[:10]) & set(rel)) / len(rel))
            # Inject dummy trace to keep crag._all_traces aligned with questions
            if hasattr(crag, "_all_traces"):
                crag._all_traces.append([])
            return ids[:top_k], 0.0

        # Slow path: full CRAG loop
        ids, lat = _orig_retrieve(query, top_k=top_k)
        per_q_checkpoint[qid_str] = ids
        _fresh_count[0] += 1
        _fresh_lats.append(lat)
        if rel:
            _running_r10.append(len(set(ids[:10]) & set(rel)) / len(rel))

        n_done    = len(per_q_checkpoint)
        elapsed_s = time.perf_counter() - _t0_run
        avg_lat_s = sum(_fresh_lats) / len(_fresh_lats) / 1000
        eta_s     = avg_lat_s * max(0, len(questions) - n_done)
        r10_str   = f"{np.mean(_running_r10):.4f}" if _running_r10 else "---"
        print(
            f"  [{n_done:3d}/{len(questions)}]"
            f"  R@10={r10_str}"
            f"  lat={lat/1000:.1f}s  avg={avg_lat_s:.1f}s/q"
            f"  elapsed={elapsed_s/60:.1f}m  ETA≈{eta_s/60:.0f}m",
            flush=True,
        )

        if _fresh_count[0] % CKPT_EVERY == 0:
            ckpt_path.write_text(
                json.dumps(per_q_checkpoint, ensure_ascii=False), encoding="utf-8"
            )

        return ids, lat

    crag.retrieve = _retrieve_checkpointed

    print(f"Running {len(questions)} {split} questions"
          f" ({len(per_q_checkpoint)} from checkpoint,"
          f" {len(questions) - len(per_q_checkpoint)} fresh)...")

    t_exp = time.perf_counter()
    try:
        result = run_experiment(
            retriever=crag,
            questions=questions,
            experiment_id=experiment_id,
            hyperparameters={
                "retrieval_backbone":         backbone_label,
                "evaluator":                  "llama3.1:8b",
                "llm_backbone":               "llama3.1:8b",
                "eval_k":                     eval_k,
                "backbone_top_k":             backbone_top_k,
                "max_iterations":             max_iterations,
                "max_article_tokens":         max_article_tokens,
                "routing":                    "binary_collective_llm",
                "evaluator_prompt_iter0":     "collective_strict_variant_b",
                "evaluator_prompt_iter1plus": "collective_variant_d",
            },
            preprocessing=PREPROCESSING[variant],
            strata=strata,
            top_k=100,
        )
    finally:
        crag.retrieve = _orig_retrieve
        # Always save checkpoint on exit — even on KeyboardInterrupt
        ckpt_path.write_text(
            json.dumps(per_q_checkpoint, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\n[exit] Checkpoint saved"
              f" ({len(per_q_checkpoint)}/{len(questions)} done)", flush=True)

    exp_wall = time.perf_counter() - t_exp

    loop_stats = crag.get_loop_stats()
    result["crag_loop_stats"]               = loop_stats
    result["latency_breakdown_ms_mean"]     = crag.get_latency_breakdown_mean()
    result["total_experiment_wall_clock_s"] = round(exp_wall, 1)

    # Patch latency mean: checkpointed questions return lat=0 and skew the mean.
    # Overwrite with fresh-only values when a partial resume occurred.
    if _fresh_lats and len(_fresh_lats) < len(questions):
        arr = np.array(_fresh_lats)
        result["latency_ms_mean"] = float(arr.mean())
        result["latency_ms_std"]  = float(arr.std())
        result["latency_distribution_fresh_only"] = {
            "n_fresh_questions": len(arr),
            "mean_ms":           float(arr.mean()),
            "std_ms":            float(arr.std()),
            "p50_ms":            float(np.percentile(arr, 50)),
            "p90_ms":            float(np.percentile(arr, 90)),
            "note":              "Latency for questions not loaded from per-question checkpoint",
        }

    # ── Per-iteration recall tracking ─────────────────────────────────────────
    # Empty traces ([]) are injected for checkpointed questions — they are skipped
    # by the `not trace` guard so only fresh questions contribute to per-iter stats.
    iter_recalls: dict[int, list[float]] = {}
    rewrite_deltas: list[float]          = []

    for question, trace in zip(questions, crag._all_traces):
        relevant = set(gt.get(question["question_id"], []))
        if not relevant or not trace:   # skip no-relevance and checkpoint placeholders
            continue
        iter_recall_for_q: list[float] = []
        for entry in trace:
            idx          = entry["iteration"]
            article_ids  = entry.get("retrieved_article_ids", [])
            recall       = len(set(article_ids) & relevant) / len(relevant)
            iter_recalls.setdefault(idx, []).append(recall)
            iter_recall_for_q.append(recall)
        for i in range(1, len(iter_recall_for_q)):
            rewrite_deltas.append(iter_recall_for_q[i] - iter_recall_for_q[i - 1])

    n_rw     = len(rewrite_deltas)
    per_iter = {
        f"mean_recall_at_eval_k_iter_{k}": round(float(np.mean(v)), 4)
        for k, v in sorted(iter_recalls.items())
    }
    per_iter["n_queries_with_rewrites"] = n_rw
    if n_rw > 0:
        per_iter["fraction_improved_after_rewrite"]  = round(
            sum(1 for d in rewrite_deltas if d > 0)  / n_rw, 4)
        per_iter["fraction_unchanged_after_rewrite"] = round(
            sum(1 for d in rewrite_deltas if d == 0) / n_rw, 4)
        per_iter["fraction_regressed_after_rewrite"] = round(
            sum(1 for d in rewrite_deltas if d < 0)  / n_rw, 4)
        per_iter["mean_recall_delta_after_rewrite"]  = round(
            float(np.mean(rewrite_deltas)), 4)
    result["per_iteration_recall"] = per_iter

    print(f"\n  R@10   = {result['metrics'].get('Recall@10', 0.0):.4f}")
    print(f"  R@100  = {result['metrics'].get('Recall@100', 0.0):.4f}")
    print(f"  MRR@10 = {result['metrics'].get('MRR@10', 0.0):.4f}")
    print(f"  Latency mean = {result['latency_ms_mean']:.0f} ms/query")
    print(f"  Wall clock: {exp_wall / 60:.1f} min")

    if loop_stats:
        print(f"\n  Loop stats:")
        print(f"    Correct first-pass       : "
              f"{loop_stats.get('fraction_queries_correct_first_pass', 0):.1%}")
        print(f"    Rewritten                : "
              f"{loop_stats.get('fraction_queries_rewritten', 0):.1%}")
        print(f"    Used aspect rewrite      : "
              f"{loop_stats.get('fraction_queries_used_aspect_rewrite', 0):.1%}")
        print(f"    Terminated by limit      : "
              f"{loop_stats.get('fraction_queries_terminated_by_limit', 0):.1%}")

        frac_limit = loop_stats.get("fraction_queries_terminated_by_limit", 0)
        if frac_limit > 0.30:
            print(f"\n  *** WARNING: {frac_limit:.0%} queries terminated by "
                  f"max_iterations limit.")
            print("  Consider --max-iterations 3 before running remaining variants.")

    saved = save_result(result, results_dir=RESULTS_DIR)

    # Clean up checkpoint only after the result JSON is successfully written
    if ckpt_path.exists():
        ckpt_path.unlink()
        print("  Per-question checkpoint cleaned up.")

    print(f"\n  Saved: {saved}")
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_corpus(df: pd.DataFrame) -> dict[int, str]:
    """Build {article_id: article_text} lookup from the articles DataFrame."""
    id_col   = "article_id"   if "article_id"   in df.columns else df.columns[0]
    text_col = "article_text" if "article_text" in df.columns else df.columns[-1]
    return dict(zip(df[id_col], df[text_col]))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="CRAG Tier 4.1 experiments — collective LLM evaluator, v2 run."
    )
    parser.add_argument(
        "--split", choices=["val", "test", "all"], default="test",
        help="Which split(s) to run (default: test).",
    )
    parser.add_argument(
        "--variant", choices=["bm25", "dense", "hybrid", "hybrid_rrf_k60", "all"], default="all",
        help="Which backbone variant to run (default: all).",
    )
    parser.add_argument(
        "--eval-k", type=int, default=None,
        help=(
            "Lock eval_k directly and skip the probe entirely. "
            "If omitted, eval_k is auto-selected via --eval-k-candidates."
        ),
    )
    parser.add_argument(
        "--eval-k-candidates", type=str, default="50,20,10",
        help=(
            "Comma-separated eval_k values to probe, largest first. "
            "Used only when --eval-k is not set (default: 50,20,10)."
        ),
    )
    parser.add_argument(
        "--max-minutes", type=float, default=90.0,
        help=(
            "Maximum acceptable wall-clock minutes per variant. "
            "Auto-selection stops at the largest eval_k whose estimate fits "
            "(default: 90)."
        ),
    )
    parser.add_argument(
        "--probe-n", type=int, default=20,
        help="Val questions to probe per eval_k candidate (default: 20).",
    )
    parser.add_argument(
        "--max-iterations", type=int, default=2,
        help="Hard loop limit per query (default: 2).",
    )
    parser.add_argument(
        "--max-article-tokens", type=int, default=300,
        help="Per-article token budget for LLM evaluation (default: 300).",
    )
    parser.add_argument(
        "--aligned", action="store_true",
        help=(
            "T4.0-aligned ablation: backbone_top_k=20, eval_k=20, skips probe. "
            "Outputs crag_{variant}_aligned_{split}_v2.json."
        ),
    )
    args = parser.parse_args()

    print("CRAG Experiments — Tier 4.1 (collective LLM evaluator, v2 run)")
    print(f"  max_iterations     = {args.max_iterations}")
    print(f"  max_article_tokens = {args.max_article_tokens}")

    df = pd.read_parquet(ROOT / "output" / "bsard_articles_dedup.parquet")

    # ── Determine eval_k and backbone_top_k ─────────────────────────────────
    if args.aligned:
        eval_k         = 20
        backbone_top_k = 20
        fewshot        = load_fewshot_examples()
        print(f"\nALIGNED mode — backbone_top_k={backbone_top_k}, eval_k={eval_k}")
        print("Probe skipped (eval_k fixed to match T4.0 top_n=20).")

    elif args.eval_k is not None:
        eval_k         = args.eval_k
        backbone_top_k = 100
        fewshot        = load_fewshot_examples()
        print(f"\neval_k locked by --eval-k: {eval_k}  (probe skipped)")

    else:
        backbone_top_k = 100
        fewshot        = load_fewshot_examples()
        candidates     = sorted(
            [int(x.strip()) for x in args.eval_k_candidates.split(",")],
            reverse=True,
        )
        llm = OllamaClient()
        llm.preflight()
        eval_k = auto_select_eval_k(
            df=df,
            llm_client=llm,
            candidates=candidates,
            max_minutes=args.max_minutes,
            probe_n=args.probe_n,
            backbone_top_k=backbone_top_k,
            max_article_tokens=args.max_article_tokens,
        )

    print(f"\nRunning with eval_k = {eval_k},  backbone_top_k = {backbone_top_k}")

    # ── Run experiments ──────────────────────────────────────────────────────
    variants = ["bm25", "dense", "hybrid"] if args.variant == "all" else [args.variant]
    # hybrid_rrf_k60 is excluded from "all" — run explicitly via --variant hybrid_rrf_k60
    splits   = ["val", "test"]             if args.split   == "all" else [args.split]

    for split in splits:
        if split == "test":
            print("\n*** TEST SPLIT — results are final. Do not re-run to tune. ***")
        for variant in variants:
            try:
                run_crag_experiment(
                    variant=variant,
                    split=split,
                    df=df,
                    eval_k=eval_k,
                    fewshot=fewshot,
                    max_iterations=args.max_iterations,
                    max_article_tokens=args.max_article_tokens,
                    backbone_top_k=backbone_top_k,
                    aligned=args.aligned,
                )
            except NotImplementedError as e:
                print(f"\n  SKIPPED {variant}: {e}")


if __name__ == "__main__":
    main()
