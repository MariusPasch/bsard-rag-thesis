"""
CRAG Threshold Calibration — Tier 4.1 Step 4.1.

Runs on the val split (177 questions) to calibrate tau+ (binary routing).
Writes evaluation/agentic/results/crag_threshold_calibration.json.

Ground-truth labels (binary):
  Correct   : precision@eval_k > 0  (at least one relevant article in top-k)
  Incorrect : recall@eval_k == 0    (no relevant article in top-k)

  Questions that are neither (i.e., partial recall but precision@k=0 by
  multi-article spread) are rare edge cases and are excluded from accuracy
  computation (they receive no GT label).

Grid search: tau+ in [0.3, 0.9] step 0.05 → 13 combinations.
  tau- removed — binary routing uses a single threshold τ+.

Objective: maximise binary classification accuracy subject to >= 10% Incorrect
           (ensures the rewrite path fires in experiments).

Evaluator: Alibaba-NLP/gte-multilingual-reranker-base (mGTE)
           Outputs sigmoid scores in [0, 1] after the wrapper's sigmoid layer.

Usage:
    python scripts/agentic/calibrate_crag_thresholds.py [--backbone bm25|hybrid]
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

from evaluation.split import load_questions
from retrieval.agentic.cross_encoder_client import CrossEncoderEvaluator
from retrieval.sparse import BM25Retriever

RESULTS_DIR = ROOT / "evaluation" / "agentic" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

EVAL_K = 5  # number of docs scored by CE (mirrors CRAGRetriever.eval_k)


# ---------------------------------------------------------------------------
# Label generation (binary)
# ---------------------------------------------------------------------------

def generate_labels(
    questions: list[dict],
    retriever,
    eval_k: int,
) -> list[dict]:
    """
    For each question retrieve top-eval_k docs and assign a binary GT label.

    Correct   = precision@eval_k > 0  (≥1 relevant article in top-k)
    Incorrect = recall@eval_k == 0    (no relevant article in top-k)

    Returns a list of dicts:
        {question_id, question_text, relevant_ids, retrieved_ids,
         precision_k, recall_k, label, max_ce_score (placeholder)}
    """
    records = []
    n = len(questions)
    for i, q in enumerate(questions, 1):
        qid       = q["question_id"]
        qtext     = q["question_text"]
        relevant  = set(q["relevant_article_ids"])
        retrieved, _ = retriever.retrieve(qtext, top_k=eval_k)

        hits        = len(relevant & set(retrieved))
        precision_k = hits / eval_k if eval_k > 0 else 0.0
        recall_k    = hits / len(relevant) if relevant else 0.0

        if precision_k > 0:
            label = "Correct"
        elif recall_k == 0.0:
            label = "Incorrect"
        else:
            label = None  # excluded from accuracy (pathological edge case)

        records.append({
            "question_id":   qid,
            "question_text": qtext,
            "relevant_ids":  list(relevant),
            "retrieved_ids": retrieved,
            "precision_k":   round(precision_k, 4),
            "recall_k":      round(recall_k, 4),
            "label":         label,
            "max_ce_score":  None,  # filled below
        })

        if i % 20 == 0 or i == n:
            dist = {l: sum(r["label"] == l for r in records)
                    for l in ("Correct", "Incorrect")}
            n_excl = sum(r["label"] is None for r in records)
            print(f"  [Q {i:3d}/{n}] labels so far: {dist}  excluded={n_excl}")

    return records


# ---------------------------------------------------------------------------
# CE scoring
# ---------------------------------------------------------------------------

def score_with_ce(
    records: list[dict],
    ce: CrossEncoderEvaluator,
    id_to_text: dict[int, str],
    eval_k: int,
) -> list[dict]:
    """Add max_ce_score to each record by scoring top-eval_k docs."""
    n = len(records)
    t0 = time.perf_counter()

    for i, rec in enumerate(records, 1):
        pairs = [
            (rec["question_text"], id_to_text.get(aid, ""))
            for aid in rec["retrieved_ids"][:eval_k]
        ]
        if not pairs:
            rec["max_ce_score"] = 0.0
            continue

        scores, lat = ce.predict(pairs)
        rec["max_ce_score"] = float(scores.max())

        if i % 20 == 0 or i == n:
            elapsed = time.perf_counter() - t0
            eta = (elapsed / i) * (n - i)
            print(f"  [CE {i:3d}/{n}] max_ce={rec['max_ce_score']:.3f} | "
                  f"elapsed={elapsed:.0f}s ETA={eta:.0f}s")

    return records


# ---------------------------------------------------------------------------
# Grid search (binary — single tau+)
# ---------------------------------------------------------------------------

def grid_search(
    records: list[dict],
    tau_plus_range: list[float],
) -> tuple[dict, list[dict]]:
    """
    Find tau+ maximising binary label accuracy subject to >= 10% Incorrect.
    Excludes records with label=None from accuracy computation.

    Returns (best_params, grid_results).
    """
    # Only use labelled records for accuracy
    labelled  = [r for r in records if r["label"] is not None]
    all_recs  = records
    n_labelled = len(labelled)
    n_all     = len(all_recs)

    labels    = [r["label"]       for r in labelled]
    ce_scores = [r["max_ce_score"] for r in labelled]
    all_scores = [r["max_ce_score"] for r in all_recs]

    grid_results = []
    t0 = time.perf_counter()

    for tau_p in tau_plus_range:
        predicted = ["Correct" if s >= tau_p else "Incorrect" for s in ce_scores]
        accuracy   = sum(p == l for p, l in zip(predicted, labels)) / n_labelled

        # Fraction Incorrect over ALL records (not just labelled) — for
        # the >=10% gate that ensures rewrite path fires in experiments.
        all_predicted   = ["Correct" if s >= tau_p else "Incorrect" for s in all_scores]
        frac_incorr     = all_predicted.count("Incorrect") / n_all
        frac_correct    = all_predicted.count("Correct") / n_all

        grid_results.append({
            "tau_plus":                round(tau_p, 4),
            "classification_accuracy": round(accuracy, 4),
            "fraction_incorrect":      round(frac_incorr, 4),
            "fraction_correct":        round(frac_correct, 4),
            "n_labelled":              n_labelled,
        })

    elapsed_grid = time.perf_counter() - t0
    print(f"\n  Grid search: {len(grid_results)} combinations in {elapsed_grid:.1f} s")

    # Filter: must have >= 10% Incorrect
    valid = [g for g in grid_results if g["fraction_incorrect"] >= 0.10]
    if not valid:
        print("  WARNING: No combination satisfies >= 10% Incorrect. Relaxing constraint.")
        valid = grid_results

    best = max(valid, key=lambda g: g["classification_accuracy"])
    return best, grid_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backbone", choices=["bm25", "hybrid"], default="bm25",
        help="Retrieval backbone for calibration (default: bm25)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("CRAG Threshold Calibration (binary routing)")
    print(f"Backbone : {args.backbone}")
    print(f"Val split: 177 questions")
    print(f"Evaluator: Alibaba-NLP/gte-multilingual-reranker-base")
    print("=" * 60)

    t_total = time.perf_counter()

    # ── Load data ────────────────────────────────────────────────────────────
    print("\n[1/5] Loading corpus and val split...")
    corpus_path = ROOT / "output" / "bsard_articles_dedup.parquet"
    df = pd.read_parquet(corpus_path)
    id_to_text = dict(zip(df["article_id"], df["article_text"]))

    questions = load_questions(subset="val")
    print(f"  Val questions: {len(questions)}")

    # ── Build retriever ──────────────────────────────────────────────────────
    print(f"\n[2/5] Building {args.backbone} retriever...")
    if args.backbone == "bm25":
        retriever = BM25Retriever(
            df,
            normalization="lemmatize",
            field_weighting="concat_2x",
            variant="okapi",
            k1=1.5,
            b=0.75,
        )
        backbone_label = "bm25_lemmatize_concat2x"
    else:
        raise NotImplementedError(
            "Hybrid backbone not yet configured. "
            "Run with --backbone bm25 first, then fill in the hybrid config "
            "once the best Tier 3 result is locked."
        )

    # ── Generate ground-truth labels ─────────────────────────────────────────
    print(f"\n[3/5] Generating binary GT labels (top-{EVAL_K} retrieval)...")
    t_labels = time.perf_counter()
    records = generate_labels(questions, retriever, eval_k=EVAL_K)
    elapsed_labels = time.perf_counter() - t_labels

    n_correct   = sum(r["label"] == "Correct"   for r in records)
    n_incorrect = sum(r["label"] == "Incorrect" for r in records)
    n_excluded  = sum(r["label"] is None         for r in records)
    label_dist_gt = {
        "Correct":   round(n_correct   / len(records), 4),
        "Incorrect": round(n_incorrect / len(records), 4),
        "excluded":  round(n_excluded  / len(records), 4),
    }
    print(f"  GT label distribution: {label_dist_gt}")
    print(f"  Label generation: {elapsed_labels:.1f} s")

    # ── CE scoring ───────────────────────────────────────────────────────────
    print(f"\n[4/5] Scoring top-{EVAL_K} docs with mGTE CE evaluator...")
    ce = CrossEncoderEvaluator()
    t_ce = time.perf_counter()
    records = score_with_ce(records, ce, id_to_text, eval_k=EVAL_K)
    elapsed_ce = time.perf_counter() - t_ce

    ce_scores = [r["max_ce_score"] for r in records]
    print(f"  CE score range: [{min(ce_scores):.3f}, {max(ce_scores):.3f}]")
    print(f"  CE scoring: {elapsed_ce:.1f} s")

    # ── Grid search ──────────────────────────────────────────────────────────
    print("\n[5/5] Running binary threshold grid search...")
    tau_plus_range = [round(v, 2) for v in np.arange(0.30, 0.95, 0.05)]  # 13 values
    print(f"  tau+ candidates: {tau_plus_range}")

    best, grid_results = grid_search(records, tau_plus_range)

    total_elapsed = time.perf_counter() - t_total

    # ── Write output ─────────────────────────────────────────────────────────
    output = {
        "tau_plus":                best["tau_plus"],
        "classification_accuracy": best["classification_accuracy"],
        "distribution": {
            "Correct":   best["fraction_correct"],
            "Incorrect": best["fraction_incorrect"],
        },
        "search_grid_size":              len(grid_results),
        "backbone_used_for_calibration": backbone_label,
        "evaluator":                     CrossEncoderEvaluator.DEFAULT_MODEL,
        "eval_k":                        EVAL_K,
        "n_val_questions":               len(questions),
        "n_labelled_questions":          best["n_labelled"],
        "calibration_time_s":            round(total_elapsed, 1),
        "label_distribution_gt":         label_dist_gt,
        "grid_results":                  grid_results,
    }

    out_path = RESULTS_DIR / "crag_threshold_calibration.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))

    print("\n" + "=" * 60)
    print(f"CALIBRATION COMPLETE in {total_elapsed:.0f} s")
    print(f"  tau+     = {best['tau_plus']}")
    print(f"  Accuracy = {best['classification_accuracy']:.3f}")
    print(f"  Distribution: {best['fraction_correct']:.0%} Correct, "
          f"{best['fraction_incorrect']:.0%} Incorrect")
    print(f"  Output: {out_path}")
    print("=" * 60)
    print("\nNEXT STEP: Lock tau+ in the Decision Log, then run")
    print("  python scripts/agentic/verify_evaluator_accuracy.py")


if __name__ == "__main__":
    main()
