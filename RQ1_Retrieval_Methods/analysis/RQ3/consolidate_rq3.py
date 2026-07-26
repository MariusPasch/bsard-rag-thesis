"""
Consolidate RQ3 (48-question autonomous evaluation) results into dedicated,
easy-to-organize files separate from the bulky per-experiment JSONs.

For every sparse/dense experiment whose ``subset_metrics.metrics`` block
contains autonomous T3/* keys, this writes:

    output/results/RQ3/{exp_id}.json

Each consolidated file carries:
    - experiment_id, family ("sparse" | "dense"), model_or_method,
      hyperparameters, preprocessing
    - the full subset_metrics block (supervised + T3 autonomous + T0 latency
      on the 48-q subset)
    - per_query_sidecar: relative path to the matching tier3_per_query JSON

A flat summary table is also emitted at:

    output/results/RQ3/summary.csv

with one row per experiment and the headline columns used for cross-family
ranking (T3/AQS, T3/umbrela/mean, T3/erag/mean, T3/ragas_wa/mean, Recall@10,
NDCG@10, T0/latency_p50_ms).

Usage (from project root):
    .venv/Scripts/python analysis/RQ3/consolidate_rq3.py
    .venv/Scripts/python analysis/RQ3/consolidate_rq3.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

_ROOT      = Path(__file__).parent.parent.parent
_RESULTS   = _ROOT / "output" / "results"
_OUT_DIR   = _RESULTS / "RQ3"
_FAMILIES  = {
    "sparse": _RESULTS / "sparse_retrieval",
    "dense":  _RESULTS / "dense_retrieval",
}

_SUMMARY_COLUMNS = [
    "experiment_id",
    "family",
    "model_or_method",
    "n",
    "T3/AQS",
    "T3/umbrela/mean",
    "T3/erag/mean",
    "T3/ragas_wa/mean",
    "Recall@10",
    "NDCG@10",
    "T0/latency_p50_ms",
]


def consolidate_one(src_path: Path, family: str) -> tuple[dict, dict] | None:
    """Return (consolidated_record, summary_row) or None to skip."""
    data = json.loads(src_path.read_text(encoding="utf-8"))
    sm = data.get("subset_metrics") or {}
    metrics = sm.get("metrics") or {}

    if not any(k.startswith("T3/") for k in metrics):
        return None

    exp_id = data["experiment_id"]
    sidecar = _RESULTS / f"{family}_retrieval" / "tier3_per_query" / f"{exp_id}.json"
    sidecar_rel = sidecar.relative_to(_ROOT).as_posix() if sidecar.exists() else None

    record = {
        "experiment_id":    exp_id,
        "family":           family,
        "model_or_method":  data.get("model_or_method"),
        "hyperparameters":  data.get("hyperparameters"),
        "preprocessing":    data.get("preprocessing"),
        "source_result":    src_path.relative_to(_ROOT).as_posix(),
        "per_query_sidecar": sidecar_rel,
        "subset_metrics":   sm,
    }

    summary = {
        "experiment_id":      exp_id,
        "family":             family,
        "model_or_method":    data.get("model_or_method"),
        "n":                  sm.get("n"),
        "T3/AQS":             metrics.get("T3/AQS"),
        "T3/umbrela/mean":    metrics.get("T3/umbrela/mean"),
        "T3/erag/mean":       metrics.get("T3/erag/mean"),
        "T3/ragas_wa/mean":   metrics.get("T3/ragas_wa/mean"),
        "Recall@10":          metrics.get("Recall@10"),
        "NDCG@10":            metrics.get("NDCG@10"),
        "T0/latency_p50_ms":  metrics.get("T0/latency_p50_ms"),
    }

    return record, summary


def consolidate_gold_ceiling() -> dict | None:
    """
    Pick up output/results/RQ3/gold_ceiling.json (produced directly by
    scripts/evaluation/RQ3/run_gold_ceiling.py — it is already in the
    consolidated schema, marked family="oracle") and emit a summary row.

    Returns the summary row, or None if the file does not exist.
    """
    src = _OUT_DIR / "gold_ceiling.json"
    if not src.exists():
        return None
    data = json.loads(src.read_text(encoding="utf-8"))
    sm = data.get("subset_metrics") or {}
    metrics = sm.get("metrics") or {}
    return {
        "experiment_id":      data["experiment_id"],
        "family":             data.get("family", "oracle"),
        "model_or_method":    data.get("model_or_method"),
        "n":                  sm.get("n"),
        "T3/AQS":             metrics.get("T3/AQS"),
        "T3/umbrela/mean":    metrics.get("T3/umbrela/mean"),
        "T3/erag/mean":       metrics.get("T3/erag/mean"),
        "T3/ragas_wa/mean":   metrics.get("T3/ragas_wa/mean"),
        "Recall@10":          None,  # oracle has no first-stage retrieval
        "NDCG@10":            None,
        "T0/latency_p50_ms":  None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would be written without touching disk.")
    args = parser.parse_args()

    if not args.dry_run:
        _OUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    written = 0
    skipped: list[str] = []

    for family, fam_dir in _FAMILIES.items():
        for src in sorted(fam_dir.glob("*_test.json")):
            result = consolidate_one(src, family)
            if result is None:
                skipped.append(f"{family}/{src.stem} (no T3/* metrics)")
                continue

            record, summary = result
            out_path = _OUT_DIR / f"{record['experiment_id']}.json"

            print(f"  [{family:6}] {record['experiment_id']:40}  "
                  f"AQS={summary['T3/AQS']}  R@10={summary['Recall@10']}")

            if not args.dry_run:
                out_path.write_text(
                    json.dumps(record, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
            summary_rows.append(summary)
            written += 1

    # Gold-ceiling row (family="oracle") — file is produced directly by
    # run_gold_ceiling.py in the consolidated schema, so we just load and
    # add its summary row alongside the per-system rows.
    oracle_row = consolidate_gold_ceiling()
    if oracle_row is not None:
        print(f"  [oracle] gold_ceiling                                "
              f"AQS={oracle_row['T3/AQS']}  R@10=—")
        summary_rows.append(oracle_row)
        written += 1
    else:
        skipped.append("oracle/gold_ceiling.json (file not present yet)")

    summary_path = _OUT_DIR / "summary.csv"
    if not args.dry_run:
        # Preserve any extra columns already in the existing CSV (e.g. the
        # T3/AQS_dd column added by analysis/RQ3/derive_aqs_weights.py).
        # Strategy: write the canonical columns first; if the existing file
        # has extra columns, re-merge them by experiment_id afterwards.
        existing_extras: dict[str, dict] = {}
        if summary_path.exists():
            try:
                import pandas as pd  # local import; pandas is a heavy dep
                _df = pd.read_csv(summary_path)
                extras = [c for c in _df.columns if c not in _SUMMARY_COLUMNS]
                if extras:
                    existing_extras = {
                        row["experiment_id"]: {c: row[c] for c in extras}
                        for _, row in _df.iterrows()
                    }
            except Exception:
                existing_extras = {}

        all_columns = _SUMMARY_COLUMNS + (
            sorted({c for v in existing_extras.values() for c in v})
            if existing_extras else []
        )
        with summary_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=all_columns)
            writer.writeheader()
            for row in summary_rows:
                merged = dict(row)
                for k, v in existing_extras.get(row["experiment_id"], {}).items():
                    merged[k] = v
                writer.writerow(merged)

    print()
    print(f"Consolidated experiments: {written}")
    if skipped:
        print(f"Skipped ({len(skipped)}):")
        for s in skipped:
            print(f"  - {s}")
    if args.dry_run:
        print("\n[dry-run] No files written.")
    else:
        print(f"\nWrote {written} consolidated JSON(s) + summary.csv to "
              f"{_OUT_DIR.relative_to(_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
