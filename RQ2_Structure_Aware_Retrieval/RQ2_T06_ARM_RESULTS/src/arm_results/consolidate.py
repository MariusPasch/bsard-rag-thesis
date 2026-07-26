"""Route every arm's persisted rankings through T07's evaluation engine.

This is the fix the cross-arm comparison needs: T03, T04 (each variant) and T05
are scored through a *single* code path (``evaluation.comparator.evaluate``)
against a *single* per-PDF ground truth, so the resulting numbers are finally
apples-to-apples. T06 computes no metrics of its own — T07 is the metric
authority (PROJECT_CONTEXT §2/§7); we only load, invoke, and reshape.

Tiers 1+2 are requested (binary IR: R@k, MRR@k, MAP@k, nDCG@k, P/F1/HitRate).
Tier 0 (latency/throughput) is intentionally skipped: only T05 recorded real
per-query latency, so a cross-arm T0 table would be misleading — T05 cost is
surfaced separately from its rich ``cost`` dicts.

CLI:
    python -m arm_results.consolidate                 # all 5 curated PDFs → tables
    python -m arm_results.consolidate --stems 1804_03_21_1804032150
    python -m arm_results.consolidate --include-ablations
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from . import loaders, paths


# ---------------------------------------------------------------------------
# T07 engine wiring
# ---------------------------------------------------------------------------

def _evaluate():
    """Return T07's evaluate fn, or raise a clear install hint."""
    try:
        from evaluation.comparator import evaluate
    except Exception as exc:  # noqa: BLE001
        raise ImportError(
            "Could not import T07's `evaluation` package. From this project's "
            "venv run:\n"
            "  pip install -e ..\\RQ2_T01_SHARED\n"
            "  pip install -e ..\\RQ2_T02_DATA_LOADER\n"
            "  pip install -e ..\\RQ2_T07_EVALUATION\n"
            "  pip install -e ..\\..\\RQ3_Autonomous_Evaluation   (provides bsard_evaluation)"
        ) from exc
    return evaluate


def _config():
    """TierConfig for binary IR only (tiers 1+2, standard k=[5,10,20,100]).

    Falls back to None (T07 default = supervised_standard, tiers 0+1+2) if the
    RQ3 config import is unavailable; the default still works but emits T0 keys.
    """
    try:
        from bsard_evaluation.config import TierConfig

        return TierConfig(tiers=[1, 2], k_preset="standard")
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Per-stem evaluation
# ---------------------------------------------------------------------------

def _build_projection(stem: str, gt: dict[str, list[int]]):
    """Layer-3 effective-GT projection for ``stem`` (for E/ drift-aware metrics).

    Uses T07's ``projection.build_projection`` over T03's per-PDF article-span
    cache + the BSARD db. gt_in_pdf = full GT ∩ bsard_ids locatable in the PDF.
    Returns None (with a note) if the inputs are missing.
    """
    import sqlite3

    from arm1_naive.cache import CacheRoot
    from evaluation.projection import build_projection

    if not paths.BSARD_DB.exists():
        print(f"  (BSARD db not found at {paths.BSARD_DB}; E/ skipped for {stem})")
        return None
    cache = CacheRoot(paths.T03_PDF_CACHE).pdf(stem)
    conn = sqlite3.connect(str(paths.BSARD_DB))
    try:
        return build_projection([int(q) for q in gt], stem, cache, conn)
    finally:
        conn.close()


def evaluate_stem(
    stem: str,
    *,
    include_ablations: bool = False,
    effective: bool = False,
    weighted: bool = False,
):
    """Evaluate all arms for one PDF. Returns (EvalReport, arms, ground_truth).

    ``effective`` adds drift-aware E/ metrics (recall denominator = GT in PDF).
    ``weighted`` adds Arm-1 W/ partial-relevance metrics for T03 (chunk overlap)
    by loading T07's cached weights and renaming T03 so it is treated as
    chunk-based; T04/T05 are article-level and unaffected.
    """
    gt = loaders.load_ground_truth(stem)
    arms = loaders.load_all_arms(stem, t04_include_ablations=include_ablations)
    if not arms:
        raise FileNotFoundError(f"No arm results found on disk for stem {stem}")

    # Align every arm to the GT question set. Binary T1/T2 already iterate qrels
    # (so extra queries are ignored), but T07's weighted-metric aggregator means
    # over the *result list* — T03's pool carries all 1108 qset questions, which
    # would dilute W/ by the non-GT queries. Filtering keeps all arms (and all
    # metrics) averaged over the same questions; binary results are unchanged.
    gt_keys = set(gt)
    arms = {m: [r for r in res if r.query_id in gt_keys] for m, res in arms.items()}

    weights_lookup = None
    if weighted and "T03_naive" in arms:
        wl = loaders.load_t03_weights(stem)
        if wl:
            arms["T03_arm1_naive"] = arms.pop("T03_naive")
            weights_lookup = wl
        else:
            print(f"  (no weights cache for {stem}; W/ skipped)")

    projection = _build_projection(stem, gt) if effective else None

    evaluate = _evaluate()
    report = evaluate(
        arms, gt, config=_config(), use_cache=False,
        weights_lookup=weights_lookup, projection=projection,
    )
    return report, arms, gt


# ---------------------------------------------------------------------------
# Long dataframe (one row per stem × method × metric)
# ---------------------------------------------------------------------------

def build_long_dataframe(
    stems: list[str] | None = None,
    *,
    include_ablations: bool = False,
    effective: bool = False,
    weighted: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Evaluate the given stems and return (long_df, costs_by_stem).

    long_df columns: stem, stem_label, method, n_method_q, n_gt_q, metric, value.
    """
    stems = stems or paths.STEMS
    rows: list[dict] = []
    costs: dict[str, dict] = {}

    for stem in stems:
        report, arms, gt = evaluate_stem(
            stem, include_ablations=include_ablations,
            effective=effective, weighted=weighted,
        )
        n_gt = len(gt)
        gt_keys = set(gt)
        costs[stem] = report.costs
        for method, metrics in report.results.items():
            n_method = len({r.query_id for r in arms[method]} & gt_keys)
            for metric, value in metrics.items():
                rows.append(
                    {
                        "stem": stem,
                        "stem_label": paths.stem_label(stem),
                        "method": method,
                        "n_method_q": n_method,
                        "n_gt_q": n_gt,
                        "metric": metric,
                        "value": value,
                    }
                )
        print(f"  [{stem}] {paths.stem_label(stem)}: "
              f"{len(report.results)} methods, {n_gt} GT questions")

    return pd.DataFrame(rows), costs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stems", nargs="*", default=None,
                    help="PDF stems to evaluate (default: all 5 curated).")
    ap.add_argument("--include-ablations", action="store_true",
                    help="Also load T04 boost-ablation runs.")
    ap.add_argument("--effective", action="store_true",
                    help="Add drift-aware E/ metrics (recall denominator = GT in PDF).")
    ap.add_argument("--weighted", action="store_true",
                    help="Add Arm-1 W/ partial-relevance metrics for T03 (chunk overlap).")
    ap.add_argument("--no-tables", action="store_true",
                    help="Write only the long CSV, skip rendering tables.")
    args = ap.parse_args()

    paths.ensure_output_dirs()
    extras = [n for n, on in (("E/", args.effective), ("W/", args.weighted)) if on]
    print(f"Evaluating arms through T07 (tiers 1+2{', ' + '+'.join(extras) if extras else ''})…")
    long_df, costs = build_long_dataframe(
        args.stems, include_ablations=args.include_ablations,
        effective=args.effective, weighted=args.weighted,
    )

    long_csv = paths.TABLES_DIR / "cross_arm_long.csv"
    long_df.to_csv(long_csv, index=False)
    (paths.TABLES_DIR / "cross_arm_costs.json").write_text(
        json.dumps(costs, indent=2, default=str), encoding="utf-8"
    )
    print(f"\nWrote {long_csv}  ({len(long_df)} rows)")

    if not args.no_tables:
        from . import tables

        tables.emit_all(long_df, out_dir=paths.TABLES_DIR)
        print(f"Wrote headline + full tables to {paths.TABLES_DIR}")


if __name__ == "__main__":
    main()
