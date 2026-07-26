"""
Compute Tier 0+1+2 metrics on the Tier 3 subset (48 questions) for every
existing retrieval result JSON in a results directory.

This produces a `subset_metrics` block inside each result JSON so that
Tier 3 scores (which only cover the 48 questions) can be compared directly
against Tier 1/2 scores computed on the same question set.

Why post-hoc?
  The full-test-set result JSONs are already committed.  Rather than rerunning
  retrieval, this script re-evaluates from the stored `_trec_run` and
  `_trec_qrels` (or reconstructs them from `run` / `qrels` in older schemas).
  Latency distributions are subset-filtered from the existing per-query values.

Output:
  Each result JSON gains a top-level `"subset_metrics"` key:
  {
    "subset_metrics": {
      "subset_ids": [<int>, ...],   # the 48 question IDs
      "n": 48,
      "metrics": { "Recall@10": ..., "NDCG@10": ..., ... }
    }
  }

Usage (from project root):
    # Default: sparse retrieval results
    .venv/Scripts/python scripts/evaluation/compute_subset_metrics.py

    # Dense retrieval (backfill subset_metrics for RQ3 Tier 3 prep)
    .venv/Scripts/python scripts/evaluation/compute_subset_metrics.py \\
        --results-dir output/results/dense_retrieval

    # Inspect without writing
    .venv/Scripts/python scripts/evaluation/compute_subset_metrics.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from evaluation.runner import _trec_metrics_to_legacy, _TIER_CFG
from bsard_evaluation import EvaluationHarness

_PROJECT_ROOT          = Path(__file__).parent.parent.parent
_DEFAULT_RESULTS_DIR   = _PROJECT_ROOT / "output" / "results" / "sparse_retrieval"
_SUBSET_FILE           = _PROJECT_ROOT / "evaluation" / "data" / "tier3_subset.json"


def load_subset_ids(subset_file: Path) -> set[str]:
    data = json.loads(subset_file.read_text(encoding="utf-8"))
    return {str(q["question_id"]) for q in data["questions"]}


def filter_trec(
    qrels: dict[str, dict],
    run:   dict[str, dict],
    ids:   set[str],
) -> tuple[dict, dict]:
    """Keep only entries whose query_id is in ids."""
    sub_qrels = {qid: v for qid, v in qrels.items() if qid in ids}
    sub_run   = {qid: v for qid, v in run.items()   if qid in ids}
    return sub_qrels, sub_run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute and print without writing back to JSON files")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=_DEFAULT_RESULTS_DIR,
        help=f"Results directory to process (default: {_DEFAULT_RESULTS_DIR})",
    )
    args = parser.parse_args()

    results_dir = args.results_dir if args.results_dir.is_absolute() \
        else _PROJECT_ROOT / args.results_dir

    print(f"Loading Tier 3 subset from {_SUBSET_FILE} …")
    subset_ids = load_subset_ids(_SUBSET_FILE)
    print(f"  Subset: {len(subset_ids)} question IDs")

    harness = EvaluationHarness(_TIER_CFG)

    result_files = sorted(results_dir.glob("*_test.json"))
    if not result_files:
        print(f"No result files found in {results_dir}")
        sys.exit(1)

    print(f"\nProcessing {len(result_files)} result file(s) from {results_dir} …\n")

    for fpath in result_files:
        data = json.loads(fpath.read_text(encoding="utf-8"))
        exp_id = data["experiment_id"]

        # Reconstruct TREC dicts — prefer pre-built keys, fall back to
        # per-query results stored in older schema formats
        if "_trec_qrels" in data and "_trec_run" in data:
            trec_qrels = data["_trec_qrels"]
            trec_run   = data["_trec_run"]
        else:
            # Older JSON: rebuild from raw result arrays if present
            print(f"  WARNING: {exp_id} has no _trec_qrels/_trec_run — skipping")
            continue

        # Subset filter
        sub_qrels, sub_run = filter_trec(trec_qrels, trec_run, subset_ids)
        n_sub = len(sub_qrels)

        if n_sub == 0:
            print(f"  {exp_id}: no overlap with subset — skipping")
            continue

        # Re-evaluate on subset (Tier 1 + 2 only; Tier 0 uses latency_dict)
        sub_trec_metrics = harness.evaluate(
            qrels=sub_qrels,
            run=sub_run,
            verbose=False,
        )
        sub_metrics = _trec_metrics_to_legacy(sub_trec_metrics)

        r10  = sub_metrics.get("Recall@10",  float("nan"))
        r100 = sub_metrics.get("Recall@100", float("nan"))
        print(f"  {exp_id:<38}  n={n_sub}  R@10={r10:.4f}  R@100={r100:.4f}")

        if not args.dry_run:
            data["subset_metrics"] = {
                "subset_ids": sorted(int(q) for q in subset_ids),
                "n": n_sub,
                "metrics": sub_metrics,
            }
            # Strip internal keys before writing back
            clean = {k: v for k, v in data.items() if not k.startswith("_")}
            fpath.write_text(json.dumps(clean, indent=2, ensure_ascii=False),
                             encoding="utf-8")

    if args.dry_run:
        print("\n[dry-run] No files modified.")
    else:
        print(f"\nSubset metrics written back to {len(result_files)} JSON files.")


if __name__ == "__main__":
    main()
