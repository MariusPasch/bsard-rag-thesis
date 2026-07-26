"""Per-question side-by-side comparison + failure categorisation across arms.

Resolution of ranked items to bsard_ids reuses T07's ``evaluation.adapter`` so
no metric/resolution logic is duplicated here. Hit/miss bookkeeping below is
presentation-only error analysis (PROJECT_CONTEXT §5), not thesis metrics.
"""

from __future__ import annotations

import json
from pathlib import Path

from . import loaders, paths


def _ranked_bsard(result) -> list[int]:
    """Resolved bsard_id ranking for one query, dedup keep-first, via the adapter."""
    from evaluation.adapter import to_bsard_run

    run = to_bsard_run([result])  # {qid: {bsard_str: score}}
    scores = run.get(result.query_id, {})
    # Skip the adapter's raw-id fallback keys for chunks that map to no article
    # (those keys are chunk ids like "..._sw_625", not numeric bsard ids).
    ranked = sorted(scores, key=lambda b: -scores[b])
    return [int(b) for b in ranked if b.isdigit()]


def export_per_query(stem: str, *, k: int = 10, top: int = 20) -> Path:
    """Write data/per_query/<stem>.json: GT + each arm's ranking & hit@k per query."""
    paths.ensure_output_dirs()
    gt = loaders.load_ground_truth(stem)
    arms = loaders.load_all_arms(stem)

    by_arm: dict[str, dict[str, list[int]]] = {}
    for method, results in arms.items():
        by_arm[method] = {r.query_id: _ranked_bsard(r) for r in results}

    out: dict[str, dict] = {}
    for qid, gt_ids in gt.items():
        gt_set = set(gt_ids)
        rec = {"query_id": qid, "ground_truth": gt_ids, "arms": {}}
        for method, ranked_by_q in by_arm.items():
            ranked = ranked_by_q.get(qid, [])
            hit_rank = next((i + 1 for i, b in enumerate(ranked) if b in gt_set), None)
            rec["arms"][method] = {
                "ranked_top": ranked[:top],
                f"hit@{k}": bool(gt_set & set(ranked[:k])),
                "first_hit_rank": hit_rank,
            }
        out[qid] = rec

    fp = paths.PER_QUERY_DIR / f"{stem}.json"
    fp.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return fp


def categorise_failures(stem: str, method: str, *, k: int = 10) -> dict:
    """Summarise where ``method`` misses on ``stem``.

    Returns counts of: total evaluable, hit@k, missed@k (no GT in top-k), and
    near_miss (GT present in ranking but below k).
    """
    gt = loaders.load_ground_truth(stem)
    arms = loaders.load_all_arms(stem)
    if method not in arms:
        raise KeyError(f"{method} not found for {stem}; have {sorted(arms)}")

    ranked_by_q = {r.query_id: _ranked_bsard(r) for r in arms[method]}
    hit = missed = near_miss = 0
    for qid, gt_ids in gt.items():
        gt_set = set(gt_ids)
        ranked = ranked_by_q.get(qid, [])
        if gt_set & set(ranked[:k]):
            hit += 1
        elif gt_set & set(ranked):
            near_miss += 1
        else:
            missed += 1
    return {
        "stem": stem,
        "method": method,
        f"hit@{k}": hit,
        f"near_miss_below_{k}": near_miss,
        "missed_entirely": missed,
        "n_questions": len(gt),
    }


def export_all(stems: list[str] | None = None, *, k: int = 10):
    """Export per-query records for every stem + a failure-summary CSV/MD.

    Writes data/per_query/<stem>.json (one per PDF) and
    data/error_analysis/failure_summary.{csv,md} (hit / near-miss / missed
    counts per stem × method). Returns the summary DataFrame.
    """
    import pandas as pd

    stems = stems or paths.STEMS
    paths.ensure_output_dirs()
    rows = []
    for stem in stems:
        fp = export_per_query(stem, k=k)
        print(f"  wrote {fp}")
        arms = loaders.load_all_arms(stem)
        for method in arms:
            rows.append(categorise_failures(stem, method, k=k))

    df = pd.DataFrame(rows)
    df["stem_label"] = df["stem"].map(paths.stem_label)
    csv_fp = paths.ERROR_DIR / "failure_summary.csv"
    df.to_csv(csv_fp, index=False)
    print(f"  wrote {csv_fp}  ({len(df)} rows)")
    return df


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Per-query side-by-side + failure summary.")
    ap.add_argument("--stems", nargs="*", default=None)
    ap.add_argument("-k", type=int, default=10)
    args = ap.parse_args()
    export_all(args.stems, k=args.k)


if __name__ == "__main__":
    main()
