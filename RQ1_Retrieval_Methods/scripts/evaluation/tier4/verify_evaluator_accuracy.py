"""
CRAG Evaluator Accuracy Check — Tier 4.1 Step 4.

Measures how accurately LLaMA 3.1 8B binary judgment classifies retrieval
quality on a held-out val sample. Results inform the thesis; no hard gate is
applied — the script always exits 0.

Protocol (mirrors the CRAG evaluate node exactly):
  For each question, retrieve top eval_k docs with BM25, then run per-doc LLM
  binary judgment using LLM_JUDGE_BINARY_PROMPT + few-shot examples.
  Routing: relevant_count (= #Oui) >= 1 → Correct, else → Incorrect.
  This is identical to the evaluate node in retrieval/agentic/crag.py.

Ground-truth labels (binary):
  Correct   = precision@eval_k > 0  (≥1 relevant article in top-k)
  Incorrect = recall@eval_k  == 0   (no relevant article in top-k)

Sample construction (81 questions, RANDOM_SEED=42):
  - Retrieve top eval_k BM25 docs for all 177 val questions
  - Assign binary GT label per question (exclude pathological edge cases)
  - Keep ALL binary-Correct questions (~35–40 expected)
  - Randomly sample Incorrect questions to reach 81 total

Writes: evaluation/agentic/results/crag_evaluator_accuracy.json

Usage:
    python scripts/agentic/verify_evaluator_accuracy.py [--eval-k 5]
"""

from __future__ import annotations

import argparse
import json
import random
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

SAMPLE_SIZE  = 81
RANDOM_SEED  = 42
MAX_ARTICLE_TOKENS = 300


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _truncate(text: str, max_tokens: int = MAX_ARTICLE_TOKENS) -> str:
    words = text.split()
    if len(words) <= max_tokens:
        return text
    return " ".join(words[:max_tokens]) + " …"


def build_gt_labels(
    questions: list[dict],
    retriever,
    eval_k: int,
) -> list[dict]:
    """
    Retrieve top-eval_k docs for every val question and assign binary GT label.

    Correct   = precision@eval_k > 0
    Incorrect = recall@eval_k  == 0
    None      = excluded (edge case: precision=0 but recall>0)
    """
    records = []
    for q in questions:
        relevant = set(q["relevant_article_ids"])
        retrieved, _ = retriever.retrieve(q["question_text"], top_k=eval_k)

        hits        = len(relevant & set(retrieved))
        precision_k = hits / eval_k if eval_k > 0 else 0.0
        recall_k    = hits / len(relevant) if relevant else 0.0

        if precision_k > 0:
            gt_label = "Correct"
        elif recall_k == 0.0:
            gt_label = "Incorrect"
        else:
            gt_label = None

        records.append({
            "question_id":   q["question_id"],
            "question_text": q["question_text"],
            "relevant_ids":  list(relevant),
            "retrieved_ids": retrieved,
            "precision_k":   round(precision_k, 4),
            "recall_k":      round(recall_k, 4),
            "gt_label":      gt_label,
        })
    return records


def run_llm_evaluator(
    sample: list[dict],
    llm: OllamaClient,
    id_to_text: dict[int, str],
    eval_k: int,
    fewshot_block: str,
) -> list[dict]:
    """
    Run per-doc LLM binary judgment on each question's top eval_k docs.
    Routing: relevant_count >= 1 → Correct, else → Incorrect.
    This exactly mirrors the CRAG evaluate node.
    """
    n   = len(sample)
    t0  = time.perf_counter()
    results = []

    for i, rec in enumerate(sample, 1):
        relevant_count = 0
        doc_judgments  = []

        for aid in rec["retrieved_ids"][:eval_k]:
            article_text = _truncate(id_to_text.get(aid, ""))
            prompt = LLM_JUDGE_BINARY_PROMPT.format(
                fewshot_block=fewshot_block,
                question=rec["question_text"],
                article_text_truncated=article_text,
            )
            response, lat = llm.generate(prompt, temperature=0.0, max_tokens=5)
            is_relevant, _ = parse_binary_judgment(response)
            if is_relevant:
                relevant_count += 1
            doc_judgments.append({"article_id": aid, "response": response.strip(),
                                   "is_relevant": is_relevant})

        predicted = "Correct" if relevant_count >= 1 else "Incorrect"

        results.append({
            **rec,
            "relevant_count": relevant_count,
            "llm_predicted":  predicted,
            "doc_judgments":  doc_judgments,
        })

        if i % 10 == 0 or i == n:
            elapsed = time.perf_counter() - t0
            eta     = (elapsed / i) * (n - i)
            print(
                f"  [Q {i:3d}/{n}] rel_count={relevant_count}/{eval_k} "
                f"pred={predicted} gt={rec['gt_label']} | "
                f"elapsed={elapsed/60:.1f}min ETA={eta/60:.1f}min"
            )

    return results


def compute_metrics(records: list[dict]) -> dict:
    """Compute overall accuracy, precision, recall (binary Correct/Incorrect)."""
    gt_all   = [r["gt_label"]    for r in records]
    pred_all = [r["llm_predicted"] for r in records]
    n        = len(records)

    overall = sum(g == p for g, p in zip(gt_all, pred_all)) / n

    tp = sum(g == "Correct"   and p == "Correct"   for g, p in zip(gt_all, pred_all))
    fp = sum(g != "Correct"   and p == "Correct"   for g, p in zip(gt_all, pred_all))
    fn = sum(g == "Correct"   and p != "Correct"   for g, p in zip(gt_all, pred_all))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return {
        "accuracy":  round(overall, 4),
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-k", type=int, default=5,
                        help="Docs evaluated per question (default: 5; lock after calibration)")
    args = parser.parse_args()

    eval_k = args.eval_k

    print("=" * 60)
    print("CRAG LLM Evaluator Accuracy Check (binary)")
    print(f"Evaluator: llama3.1:8b via Ollama")
    print(f"eval_k   : {eval_k}")
    print(f"Sample   : {SAMPLE_SIZE} questions (all Correct + sampled Incorrect)")
    print("=" * 60)

    # ── Load data ────────────────────────────────────────────────────────────
    print("\n[1/4] Loading corpus and val split...")
    df = pd.read_parquet(ROOT / "output" / "bsard_articles_dedup.parquet")
    id_to_text = dict(zip(df["article_id"], df["article_text"]))
    questions  = load_questions(subset="val")
    print(f"  Val questions: {len(questions)}")

    # ── Few-shot examples ────────────────────────────────────────────────────
    fewshot       = load_fewshot_examples()
    fewshot_block = format_fewshot_block(fewshot, "binary")
    print(f"  Few-shot examples: {len(fewshot.get('examples', []))}")

    # ── Build retriever and generate GT labels ───────────────────────────────
    print("\n[2/4] Building BM25 retriever and generating binary GT labels...")
    retriever = BM25Retriever(
        df, normalization="lemmatize", field_weighting="concat_2x",
        variant="okapi", k1=1.5, b=0.75,
    )
    all_records = build_gt_labels(questions, retriever, eval_k=eval_k)

    n_correct   = sum(r["gt_label"] == "Correct"   for r in all_records)
    n_incorrect = sum(r["gt_label"] == "Incorrect" for r in all_records)
    n_excluded  = sum(r["gt_label"] is None         for r in all_records)
    print(f"  GT distribution: Correct={n_correct}, Incorrect={n_incorrect}, "
          f"excluded={n_excluded}")

    # ── Build stratified sample ──────────────────────────────────────────────
    rng = random.Random(RANDOM_SEED)
    correct_pool   = [r for r in all_records if r["gt_label"] == "Correct"]
    incorrect_pool = [r for r in all_records if r["gt_label"] == "Incorrect"]

    n_keep_correct   = len(correct_pool)
    n_keep_incorrect = min(SAMPLE_SIZE - n_keep_correct, len(incorrect_pool))

    sample = correct_pool + rng.sample(incorrect_pool, n_keep_incorrect)
    rng.shuffle(sample)
    print(f"  Sample: {len(sample)} questions "
          f"({n_keep_correct} Correct / {n_keep_incorrect} Incorrect)")

    # ── LLM evaluator ────────────────────────────────────────────────────────
    print(f"\n[3/4] Running LLM evaluator on {len(sample)} questions...")
    print(f"  Total LLM calls: {len(sample)} × {eval_k} = {len(sample) * eval_k}")
    print(f"  Expected time on T4 GPU: ~{len(sample)*eval_k*1.5/60:.0f} min")

    llm = OllamaClient()
    llm_lat = llm.preflight()
    print(f"  Ollama preflight OK ({llm_lat:.0f} ms)")

    t_llm = time.perf_counter()
    sample_scored = run_llm_evaluator(sample, llm, id_to_text, eval_k, fewshot_block)
    elapsed_llm   = time.perf_counter() - t_llm
    print(f"  LLM evaluation done in {elapsed_llm/60:.1f} min")

    # ── Compute metrics ──────────────────────────────────────────────────────
    print("\n[4/4] Computing metrics...")
    metrics = compute_metrics(sample_scored)

    dist_pred = {
        "Correct":   sum(r["llm_predicted"] == "Correct"   for r in sample_scored) / len(sample_scored),
        "Incorrect": sum(r["llm_predicted"] == "Incorrect" for r in sample_scored) / len(sample_scored),
    }
    dist_gt = {
        "Correct":   n_keep_correct   / len(sample),
        "Incorrect": n_keep_incorrect / len(sample),
    }

    # ── Write output ─────────────────────────────────────────────────────────
    output = {
        "n_questions":     len(sample_scored),
        "eval_k":          eval_k,
        "evaluator_model": "llama3.1:8b",
        "accuracy":        metrics["accuracy"],
        "precision":       metrics["precision"],
        "recall":          metrics["recall"],
        "distribution_pred": {k: round(v, 4) for k, v in dist_pred.items()},
        "distribution_gt":   {k: round(v, 4) for k, v in dist_gt.items()},
        "sample_composition": {
            "Correct":   n_keep_correct,
            "Incorrect": n_keep_incorrect,
        },
    }

    out_path = RESULTS_DIR / "crag_evaluator_accuracy.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))

    print("\n" + "=" * 60)
    print("EVALUATOR ACCURACY REPORT")
    print(f"  LLM accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Precision     : {metrics['precision']:.4f}")
    print(f"  Recall        : {metrics['recall']:.4f}")
    print(f"  Pred dist     : {dist_pred}")
    print(f"\n  Output: {out_path}")
    print("=" * 60)
    print("\nScript exits 0 regardless of accuracy — results are for thesis disclosure.")
    print("Proceed to: python scripts/agentic/run_crag_experiments.py --split val")


if __name__ == "__main__":
    main()
