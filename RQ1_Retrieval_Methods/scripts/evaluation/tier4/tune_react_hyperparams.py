"""
Hyperparameter tuning for the ReAct agent — Tier 4.2.

Runs a two-phase grid search on the VAL split using the react_bm25 backbone.
The locked hyperparameters are then applied to ALL variants without re-tuning.

Phase 1 — Joint grid: max_steps × top_k_shown  (overlap_threshold fixed at 0.7)
  max_steps   ∈ {3, 4, 5, 6, 8}
  top_k_shown ∈ {3, 5, 10}

Phase 2 — Threshold sweep: overlap_threshold  (best max_steps + top_k_shown from Phase 1)
  overlap_threshold ∈ {0.5, 0.7, 0.9}

Primary metric : Recall@10 on val split
Secondary check: fraction_queries_terminated_by_limit > 0.30 → warning only

Output
------
output/results/agentic/ReAct/react_hyperparams.json
  {
    "max_steps": int,
    "top_k_shown": int,
    "overlap_threshold": float,
    "recall10_val": float,
    "fraction_queries_terminated_by_limit_best": float,
    "grid_results": [
      {"max_steps": int, "top_k_shown": int, "overlap_threshold": float,
       "recall_10": float, "mean_latency_s": float,
       "fraction_terminated_by_limit": float},
      ...
    ],
    "tuning_variant": "react_bm25",
    "val_questions": int
  }

Usage
-----
    python scripts/evaluation/tier4/tune_react_hyperparams.py --split val --backbone bm25
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

from evaluation.metrics import evaluate
from evaluation.split import load_questions
from retrieval.agentic.llm_client import OllamaClient
from retrieval.agentic.llm_eval_prompts import load_fewshot_examples
from retrieval.agentic.react import ReActRetriever
from retrieval.sparse import BM25Retriever

RESULTS_DIR = ROOT / "output" / "results" / "agentic" / "ReAct"
OUTPUT_PATH = RESULTS_DIR / "react_hyperparams.json"

# ---------------------------------------------------------------------------
# Backbone factory
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


# ---------------------------------------------------------------------------
# Single grid-point evaluation
# ---------------------------------------------------------------------------

def evaluate_config(
    questions: list[dict],
    df: pd.DataFrame,
    llm: OllamaClient,
    fewshot: dict,
    max_steps: int,
    top_k_shown: int,
    overlap_threshold: float,
    label: str = "",
) -> dict:
    """
    Run ReAct with the given hyperparameters and return a results dict.
    """
    backbone = build_bm25(df)
    react    = ReActRetriever(
        retriever=backbone,
        df_articles=df,
        llm_client=llm,
        top_k_shown=top_k_shown,
        max_steps=max_steps,
        overlap_threshold=overlap_threshold,
        fewshot_examples=fewshot,
        max_article_tokens=300,
    )

    results:   dict[int, list[int]] = {}
    latencies: list[float]          = []

    n = len(questions)
    for i, q in enumerate(questions, start=1):
        ranked, lat = react.retrieve(q["question_text"], top_k=100)
        results[q["question_id"]] = ranked
        latencies.append(lat)
        if i % 20 == 0 or i == n:
            print(f"    {label}  {i}/{n}  last={lat/1000:.1f}s", flush=True)

    ground_truth = {q["question_id"]: q["relevant_article_ids"] for q in questions}
    metrics      = evaluate(results, ground_truth)
    loop_stats   = react.get_loop_stats()

    return {
        "max_steps":                   max_steps,
        "top_k_shown":                 top_k_shown,
        "overlap_threshold":           overlap_threshold,
        "recall_10":                   metrics.get("Recall@10", 0.0),
        "mean_latency_s":              round(float(np.mean(latencies)) / 1000.0, 2),
        "fraction_terminated_by_limit": loop_stats.get(
            "fraction_queries_terminated_by_limit", 0.0
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split",    default="val", choices=["val"])
    parser.add_argument("--backbone", default="bm25", choices=["bm25"])
    args = parser.parse_args()

    if OUTPUT_PATH.exists():
        tune = json.loads(OUTPUT_PATH.read_text())
        print("react_hyperparams.json already exists — skipping tuning.")
        print(f"  max_steps         = {tune['max_steps']}")
        print(f"  top_k_shown       = {tune['top_k_shown']}")
        print(f"  overlap_threshold = {tune['overlap_threshold']}")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("ReAct Hyperparameter Tuning — Tier 4.2")
    print(f"  backbone : {args.backbone}")
    print(f"  split    : {args.split}")

    # ── Load data ────────────────────────────────────────────────────────────
    df        = pd.read_parquet(ROOT / "output" / "bsard_articles_dedup.parquet")
    questions = load_questions(subset=args.split)
    fewshot   = load_fewshot_examples()
    print(f"  {len(questions)} questions loaded")
    print(f"  {len(fewshot.get('examples', []))} few-shot examples loaded")

    # ── Ollama preflight ──────────────────────────────────────────────────────
    llm     = OllamaClient()
    lat_pre = llm.preflight()
    print(f"  Ollama preflight OK ({lat_pre:.0f} ms)\n")

    grid_results: list[dict] = []

    # ── Phase 1: max_steps × top_k_shown grid (overlap_threshold = 0.7) ─────
    PHASE1_OVERLAP = 0.7
    phase1_grid = [
        (max_steps, top_k_shown)
        for max_steps   in [3, 4, 5, 6, 8]
        for top_k_shown in [3, 5, 10]
    ]
    print(f"Phase 1: {len(phase1_grid)} configs  (overlap_threshold={PHASE1_OVERLAP})")

    best_p1_recall   = -1.0
    best_p1_config   = (6, 5)

    for max_steps, top_k_shown in phase1_grid:
        label = f"steps={max_steps} top_k={top_k_shown} ov={PHASE1_OVERLAP}"
        print(f"\n  [{label}]")
        t0  = time.time()
        row = evaluate_config(
            questions, df, llm, fewshot,
            max_steps, top_k_shown, PHASE1_OVERLAP,
            label=label,
        )
        elapsed = time.time() - t0
        print(
            f"  → R@10={row['recall_10']:.4f}  "
            f"lat={row['mean_latency_s']:.1f}s  "
            f"term%={row['fraction_terminated_by_limit']:.1%}  "
            f"wall={elapsed/60:.1f}min"
        )
        grid_results.append(row)

        if row["recall_10"] > best_p1_recall:
            best_p1_recall = row["recall_10"]
            best_p1_config = (max_steps, top_k_shown)

    best_max_steps, best_top_k_shown = best_p1_config
    print(
        f"\nPhase 1 best: max_steps={best_max_steps}  "
        f"top_k_shown={best_top_k_shown}  "
        f"R@10={best_p1_recall:.4f}"
    )

    # ── Phase 2: overlap_threshold sweep ─────────────────────────────────────
    print(f"\nPhase 2: overlap_threshold sweep  (max_steps={best_max_steps}, top_k_shown={best_top_k_shown})")

    best_ov_recall = -1.0
    best_ov        = PHASE1_OVERLAP
    best_ov_stats  = {}

    for ov in [0.5, 0.7, 0.9]:
        if ov == PHASE1_OVERLAP:
            # Reuse Phase 1 result for this exact config
            match = next(
                (r for r in grid_results
                 if r["max_steps"] == best_max_steps
                 and r["top_k_shown"] == best_top_k_shown
                 and r["overlap_threshold"] == PHASE1_OVERLAP),
                None,
            )
            if match:
                row = match
                print(f"  [ov={ov}] reusing Phase 1 result: R@10={row['recall_10']:.4f}")
            else:
                label = f"steps={best_max_steps} top_k={best_top_k_shown} ov={ov}"
                print(f"\n  [{label}]")
                row = evaluate_config(
                    questions, df, llm, fewshot,
                    best_max_steps, best_top_k_shown, ov,
                    label=label,
                )
                grid_results.append(row)
                print(f"  → R@10={row['recall_10']:.4f}")
        else:
            label = f"steps={best_max_steps} top_k={best_top_k_shown} ov={ov}"
            print(f"\n  [{label}]")
            row = evaluate_config(
                questions, df, llm, fewshot,
                best_max_steps, best_top_k_shown, ov,
                label=label,
            )
            grid_results.append(row)
            print(
                f"  → R@10={row['recall_10']:.4f}  "
                f"term%={row['fraction_terminated_by_limit']:.1%}"
            )

        if row["recall_10"] > best_ov_recall:
            best_ov_recall = row["recall_10"]
            best_ov        = ov
            best_ov_stats  = row

    print(
        f"\nPhase 2 best: overlap_threshold={best_ov}  "
        f"R@10={best_ov_recall:.4f}"
    )

    # ── Write output ──────────────────────────────────────────────────────────
    frac_lim = best_ov_stats.get("fraction_terminated_by_limit", 0.0)
    output = {
        "max_steps":         best_max_steps,
        "top_k_shown":       best_top_k_shown,
        "overlap_threshold": best_ov,
        "recall10_val":      round(best_ov_recall, 6),
        "fraction_queries_terminated_by_limit_best": round(frac_lim, 4),
        "grid_results":      grid_results,
        "tuning_variant":    "react_bm25",
        "val_questions":     len(questions),
    }

    OUTPUT_PATH.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\nSaved: {OUTPUT_PATH}")
    print(f"  max_steps         = {best_max_steps}")
    print(f"  top_k_shown       = {best_top_k_shown}")
    print(f"  overlap_threshold = {best_ov}")

    if frac_lim > 0.30:
        print(
            f"\nWARNING: {frac_lim:.0%} of val queries terminated by step limit. "
            "Consider increasing max_steps before running test split."
        )


if __name__ == "__main__":
    main()
