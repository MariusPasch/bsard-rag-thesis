"""
Create a stratified 50-question Tier 3 evaluation subset from the 222-question
BSARD test set.

The subset is used for all Tier 3 (autonomous) evaluations across RQ1, RQ2, and
RQ3 so that results are directly comparable across tiers without re-running on
different question sets.

Stratification axes
-------------------
1. article_count  : single_article / multi_article
2. lex_align      : lexically_aligned / middle / semantically_paraphrased
3. cross_ref      : with_cross_refs / without_cross_refs

This gives 12 cells (2 × 3 × 2). Each cell is sampled proportionally to its
share of the 222-question test set, with a minimum of 1 per cell. Cells with
pool size < 1 (none exist in the test set) are skipped.

Random seed: 42 (fixed for reproducibility across all tiers).

Output
------
evaluation/tier3_subset.json

Schema:
{
  "meta": {
    "created":      "<ISO date>",
    "seed":         42,
    "target_n":     50,
    "actual_n":     <int>,
    "total_test_n": 222,
    "pct_of_test":  <float>,
    "strategy":     "proportional_stratified",
    "axes":         ["article_count", "lex_align", "cross_ref"]
  },
  "cell_summary": {
    "<cell_key>": {"pool": <int>, "sampled": <int>}
    ...
  },
  "questions": [
    {
      "question_id": <int>,
      "question_text": "<str>",
      "relevant_article_ids": [<int>, ...],
      "n_relevant_articles": <int>,
      "strata": {
        "article_count": "<str>",
        "lex_align": "<str>",
        "cross_ref": "<str>",
        "bm25_score": <float>
      }
    },
    ...
  ]
}

Usage (from project root):
    .venv/Scripts/python scripts/evaluation/create_tier3_subset.py
    .venv/Scripts/python scripts/evaluation/create_tier3_subset.py --target 40
    .venv/Scripts/python scripts/evaluation/create_tier3_subset.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sqlite3
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

# Make project root importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_DB_PATH      = _PROJECT_ROOT / "output" / "bsard_corpus.db"
_STRATA_FILE  = _PROJECT_ROOT / "evaluation" / "data" / "query_strata.json"
_OUTPUT_FILE  = _PROJECT_ROOT / "evaluation" / "data" / "tier3_subset.json"

SEED   = 42
AXES   = ["article_count", "lex_align", "cross_ref"]


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_test_questions(db_path: Path) -> list[dict]:
    """Load all test-split questions from the SQLite corpus DB."""
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT question_id, question_text, relevant_article_ids, n_relevant_articles "
        "FROM questions WHERE split='test'"
    ).fetchall()
    conn.close()

    questions = []
    for row in rows:
        qid, text, rel_ids_json, n_rel = row
        rel_ids = json.loads(rel_ids_json) if isinstance(rel_ids_json, str) else list(rel_ids_json)
        questions.append({
            "question_id":          qid,
            "question_text":        text,
            "relevant_article_ids": rel_ids,
            "n_relevant_articles":  n_rel,
        })
    return questions


def load_strata(strata_file: Path) -> dict[int, dict]:
    """Load pre-computed query strata (from evaluation/data/query_strata.json)."""
    raw = json.loads(strata_file.read_text())
    return {int(k): v for k, v in raw.items()}


# ---------------------------------------------------------------------------
# Stratified sampling
# ---------------------------------------------------------------------------

def _cell_key(strata_entry: dict) -> tuple[str, str, str]:
    return (
        strata_entry["article_count"],
        strata_entry["lex_align"],
        strata_entry["cross_ref"],
    )


def stratified_sample(
    questions: list[dict],
    strata: dict[int, dict],
    target: int,
    seed: int = SEED,
) -> tuple[list[dict], dict]:
    """
    Proportional stratified sample across the 12 (article_count × lex_align × cross_ref)
    cells. Returns (sampled_questions, cell_summary_dict).
    """
    # Group question IDs by cell
    cells: dict[tuple, list[int]] = defaultdict(list)
    for q in questions:
        qid = q["question_id"]
        if qid in strata:
            cells[_cell_key(strata[qid])].append(qid)

    total = sum(len(v) for v in cells.values())
    rate  = target / total

    rng   = random.Random(seed)
    qid_to_q = {q["question_id"]: q for q in questions}

    sampled_ids: list[int] = []
    cell_summary: dict[str, dict] = {}

    for cell_key in sorted(cells.keys()):
        pool  = cells[cell_key]
        n     = max(1, math.floor(len(pool) * rate))
        n     = min(n, len(pool))
        chosen = rng.sample(pool, n)
        sampled_ids.extend(chosen)
        cell_summary[str(cell_key)] = {"pool": len(pool), "sampled": n}

    # Trim to target if rounding pushed us over
    rng.shuffle(sampled_ids)
    if len(sampled_ids) > target:
        sampled_ids = sampled_ids[:target]

    sampled = [qid_to_q[qid] for qid in sorted(sampled_ids)]
    return sampled, cell_summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Create stratified Tier 3 evaluation subset."
    )
    parser.add_argument(
        "--target", type=int, default=50,
        help="Target subset size (default: 50, must be < 25%% of test set = <56)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print plan without writing output file"
    )
    parser.add_argument(
        "--output", type=Path, default=_OUTPUT_FILE,
        help=f"Output path (default: {_OUTPUT_FILE})"
    )
    args = parser.parse_args()

    if args.target >= 56:
        print(f"WARNING: target={args.target} >= 25% of 222 test questions (56).")
        print("Consider using --target 50 or lower to stay within budget.")

    print(f"Loading test questions from {_DB_PATH} …")
    questions = load_test_questions(_DB_PATH)
    print(f"  Test questions: {len(questions)}")

    print(f"Loading strata from {_STRATA_FILE} …")
    strata = load_strata(_STRATA_FILE)
    print(f"  Strata entries: {len(strata)}")

    print(f"\nBuilding proportional stratified sample (target={args.target}, seed={SEED}) …")
    sampled, cell_summary = stratified_sample(questions, strata, args.target, SEED)
    actual_n = len(sampled)
    pct      = actual_n / len(questions) * 100

    # ── Print cell breakdown ─────────────────────────────────────────────────
    print(f"\n{'Cell':<72} {'Pool':>5} {'Sampled':>8}")
    print("-" * 88)
    for cell_str, counts in sorted(cell_summary.items()):
        print(f"  {cell_str:<70} {counts['pool']:>5} {counts['sampled']:>8}")
    print("-" * 88)
    print(f"  {'TOTAL':<70} {len(questions):>5} {actual_n:>8}")
    print(f"\n  Coverage: {actual_n}/{len(questions)} = {pct:.1f}% of test set")

    # ── Print stratum coverage check ─────────────────────────────────────────
    from collections import Counter
    sampled_strata = [strata.get(q["question_id"], {}) for q in sampled]
    print(f"\nStratum coverage in selected subset:")
    for axis in AXES:
        counts = Counter(s.get(axis, "?") for s in sampled_strata)
        print(f"  {axis:20s}: {dict(sorted(counts.items()))}")

    if args.dry_run:
        print("\n[dry-run] No file written.")
        return

    # ── Build output document ────────────────────────────────────────────────
    output = {
        "meta": {
            "created":      str(date.today()),
            "seed":         SEED,
            "target_n":     args.target,
            "actual_n":     actual_n,
            "total_test_n": len(questions),
            "pct_of_test":  round(pct, 2),
            "strategy":     "proportional_stratified",
            "axes":         AXES,
            "note": (
                "Fixed subset for all Tier 3 evaluations across RQ1/RQ2/RQ3. "
                "Do not resample — use this file as the canonical T3 question set."
            ),
        },
        "cell_summary": cell_summary,
        "questions": [
            {
                "question_id":          q["question_id"],
                "question_text":        q["question_text"],
                "relevant_article_ids": q["relevant_article_ids"],
                "n_relevant_articles":  q["n_relevant_articles"],
                "strata": {
                    k: strata.get(q["question_id"], {}).get(k)
                    for k in ["article_count", "lex_align", "cross_ref", "bm25_score"]
                },
            }
            for q in sampled
        ],
    }

    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSubset written to: {args.output}")
    print(f"  {actual_n} questions, {pct:.1f}% of test set")


if __name__ == "__main__":
    main()
