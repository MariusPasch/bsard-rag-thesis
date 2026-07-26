"""
CRAG LLM Calibration — Tier 4.1 Step 3.

Runs on the val split (177 questions) to calibrate eval_k (binary routing).
Replaces calibrate_crag_thresholds.py — tau+ grid search removed; eval_k
grid search substituted (LLaMA 3.1 8B evaluator replaces mGTE CE).

Ground-truth labels (binary):
  Correct   : precision@5 > 0  (at least one relevant article in top-k)
  Incorrect : recall@5 == 0    (no relevant article in top-k)

  Questions that are neither (partial recall but precision@k=0 via multi-article
  spread) are rare edge cases and excluded from accuracy computation.

Grid search: eval_k ∈ {3, 5, 10}.
  Routing rule: relevant_count >= 1 → Correct, else → Incorrect.
  Score caching: each (question, article) pair is LLM-scored once (top-10 docs
  retrieved once per question), then subsets of size eval_k are evaluated.
  This avoids redundant LLM calls across grid combinations.

Objective: select eval_k maximising routing accuracy vs ground-truth labels.
  No hard gate — result reported for thesis disclosure only.

Backbone for calibration: BM25 (bm25_lemmatize_concat_2x) — lightweight and
  reproducible. Results transfer to other backbones since the evaluator is
  query/article-pair based, not backbone-specific.

Expected runtime (T4 GPU, ~1-2 s/call):
  177 questions × 10 docs × ~1.5 s  ≈  45 min
  Grid search (3 combinations): < 1 s

Usage:
    python scripts/agentic/calibrate_crag_llm.py [--backbone bm25]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.split import load_questions
from retrieval.agentic.llm_client import OllamaClient
from retrieval.agentic.llm_eval_prompts import (
    LLM_JUDGE_BINARY_PROMPT,
    format_fewshot_block,
    load_fewshot_examples,
    parse_binary_judgment,
)
from retrieval.sparse import BM25Retriever

RESULTS_DIR = ROOT / "evaluation" / "agentic" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MAX_RETRIEVE  = 10   # retrieve top-10 once; sub-sample for each eval_k candidate
EVAL_K_GRID   = [3, 5, 10]
MAX_ARTICLE_TOKENS = 300


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _truncate(text: str, max_tokens: int = MAX_ARTICLE_TOKENS) -> str:
    words = text.split()
    if len(words) <= max_tokens:
        return text
    return " ".join(words[:max_tokens]) + " …"


def generate_gt_labels(
    questions: list[dict],
    retriever,
    eval_k: int,
) -> list[dict]:
    """
    Retrieve top-MAX_RETRIEVE docs for each question and assign binary GT label.

    Correct   = precision@eval_k > 0  (≥1 relevant article in top-k)
    Incorrect = recall@eval_k == 0    (no relevant article in top-k)
    None      = excluded (edge case: precision@k=0 but recall@k>0)

    Returns list of dicts with keys:
        question_id, question_text, relevant_ids, retrieved_ids,
        precision_k, recall_k, label
    """
    records = []
    n = len(questions)
    for i, q in enumerate(questions, 1):
        relevant  = set(q["relevant_article_ids"])
        retrieved, _ = retriever.retrieve(q["question_text"], top_k=MAX_RETRIEVE)

        top_k_retrieved = retrieved[:eval_k]
        hits_k      = len(relevant & set(top_k_retrieved))
        precision_k = hits_k / eval_k if eval_k > 0 else 0.0
        recall_k    = hits_k / len(relevant) if relevant else 0.0

        if precision_k > 0:
            label = "Correct"
        elif recall_k == 0.0:
            label = "Incorrect"
        else:
            label = None

        records.append({
            "question_id":   q["question_id"],
            "question_text": q["question_text"],
            "relevant_ids":  list(relevant),
            "retrieved_ids": retrieved,   # full top-MAX_RETRIEVE list
            "precision_k":   round(precision_k, 4),
            "recall_k":      round(recall_k, 4),
            "label":         label,
        })

        if i % 20 == 0 or i == n:
            dist = {lbl: sum(r["label"] == lbl for r in records)
                    for lbl in ("Correct", "Incorrect")}
            n_excl = sum(r["label"] is None for r in records)
            print(f"  [GT {i:3d}/{n}] labels so far: {dist}  excluded={n_excl}")

    return records


def score_all_pairs(
    records: list[dict],
    llm: OllamaClient,
    id_to_text: dict[int, str],
    fewshot_block: str,
) -> list[dict]:
    """
    For each question, LLM-score the top-MAX_RETRIEVE retrieved docs.
    Stores 'llm_judgments' list (bool) in each record.

    Uses LLM_JUDGE_BINARY_PROMPT with few-shot block — identical to the
    evaluate node in crag.py. Caches scores so the eval_k grid search is free.

    Expected wall-clock: 177 × 10 × ~1.5 s ≈ 45 min on T4 GPU.
    """
    n   = len(records)
    t0  = time.perf_counter()

    for i, rec in enumerate(records, 1):
        judgments = []
        for aid in rec["retrieved_ids"][:MAX_RETRIEVE]:
            article_text = _truncate(id_to_text.get(aid, ""))
            prompt = LLM_JUDGE_BINARY_PROMPT.format(
                fewshot_block=fewshot_block,
                question=rec["question_text"],
                article_text_truncated=article_text,
            )
            response, _ = llm.generate(prompt, temperature=0.0, max_tokens=5)
            is_relevant, _ = parse_binary_judgment(response)
            judgments.append(is_relevant)

        rec["llm_judgments"] = judgments  # list[bool], len = min(MAX_RETRIEVE, retrieved)

        if i % 20 == 0 or i == n:
            elapsed = time.perf_counter() - t0
            eta     = (elapsed / i) * (n - i)
            n_oui   = sum(judgments)
            print(
                f"  [Q {i:3d}/{n}] relevant_count={n_oui}/{len(judgments)} "
                f"label={rec['label']} | elapsed={elapsed/60:.1f}min ETA={eta/60:.1f}min"
            )

    return records


def grid_search(records: list[dict]) -> tuple[dict, list[dict]]:
    """
    For each eval_k in EVAL_K_GRID, compute routing accuracy vs GT labels.

    Routing rule: relevant_count (= #Oui in top-eval_k) >= 1 → Correct.
    Only uses records with non-None GT labels for accuracy.

    Returns (best_result, all_grid_results).
    """
    labelled = [r for r in records if r["label"] is not None]
    n_all    = len(records)

    t0 = time.perf_counter()
    grid_results = []

    for ek in EVAL_K_GRID:
        correct = 0
        dist_pred = {"Correct": 0, "Incorrect": 0}
        dist_all  = {"Correct": 0, "Incorrect": 0}

        for rec in labelled:
            judgments = rec.get("llm_judgments", [])
            relevant_count = sum(judgments[:ek])
            predicted = "Correct" if relevant_count >= 1 else "Incorrect"
            if predicted == rec["label"]:
                correct += 1
            dist_pred[predicted] += 1

        for rec in records:
            judgments = rec.get("llm_judgments", [])
            relevant_count = sum(judgments[:ek])
            predicted = "Correct" if relevant_count >= 1 else "Incorrect"
            dist_all[predicted] += 1

        accuracy = correct / len(labelled) if labelled else 0.0
        grid_results.append({
            "eval_k":                ek,
            "accuracy":              round(accuracy, 4),
            "fraction_correct":      round(dist_all["Correct"]   / n_all, 4),
            "fraction_incorrect":    round(dist_all["Incorrect"] / n_all, 4),
            "n_labelled":            len(labelled),
        })

    elapsed_grid = time.perf_counter() - t0
    print(f"\n  Grid search: {len(grid_results)} combinations in {elapsed_grid:.1f} s")

    best = max(grid_results, key=lambda g: g["accuracy"])
    return best, grid_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backbone", choices=["bm25"], default="bm25",
        help="Retrieval backbone for calibration (default: bm25)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("CRAG LLM Calibration (binary routing, eval_k grid)")
    print(f"Backbone : {args.backbone}")
    print(f"Val split: 177 questions")
    print(f"Evaluator: llama3.1:8b (LLM binary judgment)")
    print(f"eval_k grid: {EVAL_K_GRID}")
    print("=" * 60)

    t_total = time.perf_counter()

    # ── Load data ────────────────────────────────────────────────────────────
    print("\n[1/5] Loading corpus and val split...")
    corpus_path = ROOT / "output" / "bsard_articles_dedup.parquet"
    df = pd.read_parquet(corpus_path)
    id_to_text = dict(zip(df["article_id"], df["article_text"]))
    questions  = load_questions(subset="val")
    print(f"  Val questions: {len(questions)}")

    # ── Load few-shot examples ───────────────────────────────────────────────
    print("\n[2/5] Loading few-shot examples...")
    fewshot = load_fewshot_examples()
    fewshot_block = format_fewshot_block(fewshot, "binary")
    print(f"  Few-shot examples: {len(fewshot.get('examples', []))}")

    # ── Build retriever ──────────────────────────────────────────────────────
    print(f"\n[3/5] Building {args.backbone} retriever and generating GT labels...")
    if args.backbone == "bm25":
        retriever = BM25Retriever(
            df,
            normalization="lemmatize",
            field_weighting="concat_2x",
            variant="okapi",
            k1=1.5,
            b=0.75,
        )
        backbone_label = "bm25_lemmatize_concat_2x"
    else:
        raise NotImplementedError(f"Backbone {args.backbone!r} not configured.")

    # GT labels use eval_k=5 for the ground-truth precision/recall definition
    records = generate_gt_labels(questions, retriever, eval_k=5)
    n_correct   = sum(r["label"] == "Correct"   for r in records)
    n_incorrect = sum(r["label"] == "Incorrect" for r in records)
    n_excluded  = sum(r["label"] is None         for r in records)
    print(f"  GT distribution: Correct={n_correct}, Incorrect={n_incorrect}, "
          f"excluded={n_excluded}")

    # ── LLM scoring (cached across all eval_k) ───────────────────────────────
    print(f"\n[4/5] LLM scoring (top-{MAX_RETRIEVE} docs per question, cached)...")
    print(f"  Total LLM calls: {len(questions)} × {MAX_RETRIEVE} = "
          f"{len(questions) * MAX_RETRIEVE}")
    print(f"  Expected time on T4 GPU: ~{len(questions)*MAX_RETRIEVE*1.5/60:.0f} min")

    llm = OllamaClient()
    llm_lat = llm.preflight()
    print(f"  Ollama preflight OK ({llm_lat:.0f} ms)")

    t_llm = time.perf_counter()
    records = score_all_pairs(records, llm, id_to_text, fewshot_block)
    elapsed_llm = time.perf_counter() - t_llm
    print(f"  LLM scoring done in {elapsed_llm/60:.1f} min")

    # ── Grid search ──────────────────────────────────────────────────────────
    print(f"\n[5/5] Grid search over eval_k ∈ {EVAL_K_GRID}...")
    best, grid_results = grid_search(records)

    total_elapsed = time.perf_counter() - t_total

    # ── Print grid ───────────────────────────────────────────────────────────
    print("\n  eval_k | accuracy | frac_correct | frac_incorrect")
    print("  " + "-" * 50)
    for g in grid_results:
        marker = " ← best" if g["eval_k"] == best["eval_k"] else ""
        print(f"  k={g['eval_k']:2d}   | {g['accuracy']:.4f}   | "
              f"{g['fraction_correct']:.3f}        | {g['fraction_incorrect']:.3f}{marker}")

    # ── Write output ─────────────────────────────────────────────────────────
    output = {
        "eval_k":             best["eval_k"],
        "routing_accuracy":   best["accuracy"],
        "distribution": {
            "Correct":   best["fraction_correct"],
            "Incorrect": best["fraction_incorrect"],
        },
        "grid_results":              grid_results,
        "backbone_used_for_calibration": backbone_label,
        "evaluator_model":           "llama3.1:8b",
        "max_article_tokens":        MAX_ARTICLE_TOKENS,
        "n_val_questions":           len(questions),
        "n_labelled_questions":      best["n_labelled"],
        "label_distribution_gt": {
            "Correct":   round(n_correct   / len(questions), 4),
            "Incorrect": round(n_incorrect / len(questions), 4),
            "excluded":  round(n_excluded  / len(questions), 4),
        },
        "calibration_time_s": round(total_elapsed, 1),
    }

    out_path = RESULTS_DIR / "crag_llm_calibration.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))

    print("\n" + "=" * 60)
    print(f"CALIBRATION COMPLETE in {total_elapsed/60:.1f} min")
    print(f"  Best eval_k  = {best['eval_k']}")
    print(f"  Accuracy     = {best['accuracy']:.4f}")
    print(f"  Distribution : {best['fraction_correct']:.1%} Correct, "
          f"{best['fraction_incorrect']:.1%} Incorrect")
    print(f"  Output: {out_path}")
    print("=" * 60)
    print("\nNEXT STEPS:")
    print("  1. Lock eval_k in the Decision Log (TIER41_AGENTIC_CRAG_PLAN.md)")
    print("  2. python scripts/agentic/verify_evaluator_accuracy.py")
    print("  3. python scripts/agentic/run_crag_experiments.py --split val")


if __name__ == "__main__":
    main()
