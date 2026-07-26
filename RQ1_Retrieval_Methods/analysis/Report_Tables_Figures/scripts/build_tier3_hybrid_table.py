"""
Build the RQ1 Tier 3 (hybrid) results table -- Chapter 5, Table 5.3.

Reads the fifteen canonical Tier 3 hybrid result records (222-question BSARD
test split) plus the dense anchor as a context row, and emits one
GitHub-flavoured **markdown** results table for the thesis report. Same metric
columns as the Tier 1 / Tier 2 tables.

Output:
  - printed to stdout
  - written to analysis/tables/tier3_hybrid_results.md

Columns (in order):
  System | Fusion | Setting | R@1 | R@5 | R@10 | R@50 | R@100
         | MRR@100 | NDCG@10 | MAP@10 | Hit@10

Config columns:
  - Fusion  : RRF / Linear / SGDR (the fusion algorithm), read from the record.
              "--" for the dense reference row.
  - Setting : the swept hyperparameter -- k for RRF, alpha for Linear, K for
              SGDR -- read from the record's hyperparameters.

Every hybrid row fuses the tuned BM25 sparse stage with the E5-large+metadata
dense stage; the fusion algorithm and its setting are what vary, so the System
label is constant and Fusion / Setting carry the distinction.

Conventions:
  - Hybrid systems are first-stage / fusion retrievers, so every cut-off is
    genuine -- no cells are masked.
  - Metrics rounded to 3 dp; the best value in each metric column is **bolded**,
    computed over the HYBRID rows only -- the dense anchor is context and is
    excluded from the bolding comparison.
  - HitRate@10 maps to the Hit@10 column.

Row order: RRF (best R@100 first; ties broken by R@10) -> Linear (alpha
ascending) -> SGDR (K ascending) -> dense anchor last. The carried-forward RRF
winner is marked with a dagger.

Usage:
  python analysis/Report_Tables_Figures/scripts/build_tier3_hybrid_table.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPORT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPORT_DIR))

from load_results import load_records  # noqa: E402

OUT_DIR = REPORT_DIR.parent / "tables"          # <project>/analysis/tables
OUT_FILE = OUT_DIR / "tier3_hybrid_results.md"

HYBRID_SYSTEM = "Hybrid (BM25 + E5-large)"
DAGGER = "†"

# Dense reference row (context only, excluded from bolding).
DENSE_ANCHOR_ID = "dense_me5_large_concat2x_zeroshot_test"
DENSE_ANCHOR_LABEL = "Dense anchor (E5-large + metadata)"

# (column header, record metrics key). All are higher-is-better.
METRIC_COLS: list[tuple[str, str]] = [
    ("R@1", "Recall@1"),
    ("R@5", "Recall@5"),
    ("R@10", "Recall@10"),
    ("R@50", "Recall@50"),
    ("R@100", "Recall@100"),
    ("MRR@100", "MRR@100"),
    ("NDCG@10", "NDCG@10"),
    ("MAP@10", "MAP@10"),
    ("Hit@10", "HitRate@10"),
]

# fusion family -> group rank for the top-level row ordering.
_GROUP_RANK = {"rrf": 0, "linear": 1, "sgdr": 2}


def _records_by_id() -> dict:
    return {rec.experiment_id: rec for _tier, rec in load_records()}


def _fusion_and_setting(rec) -> tuple[str, str, str, float]:
    """Return (family_key, Fusion display, Setting display, sweep value)."""
    hp = rec.hyperparameters
    family = rec.payload.get("model_or_method", "").split("_")[-1]  # rrf/linear/sgdr
    if family == "rrf":
        return "rrf", "RRF", f"k={hp['rrf_k']}", float(hp["rrf_k"])
    if family == "linear":
        a = hp["alpha"]
        return "linear", "Linear", f"α={a:g}", float(a)
    if family == "sgdr":
        return "sgdr", "SGDR", f"K={hp['bm25_pool_k']}", float(hp["bm25_pool_k"])
    raise SystemExit(f"Unrecognised hybrid family for {rec.experiment_id!r}")


def build() -> str:
    recs = _records_by_id()
    if DENSE_ANCHOR_ID not in recs:
        raise SystemExit(f"Missing dense anchor record {DENSE_ANCHOR_ID!r}")

    # Collect every hybrid record (model_or_method starts with 'hybrid_').
    hyb = [rec for rec in recs.values()
           if str(rec.payload.get("model_or_method", "")).startswith("hybrid_")]
    if len(hyb) != 15:
        raise SystemExit(
            f"Expected 15 hybrid records, found {len(hyb)} -- is the data bundle "
            "present under output/results/hybrid/?")

    rows = []   # [group_rank, sort_key, family, system, fusion, setting, values]
    for rec in hyb:
        family, fusion, setting, sweep = _fusion_and_setting(rec)
        values = [float(rec.metrics[key]) for _hdr, key in METRIC_COLS]
        rec_r100 = float(rec.metrics["Recall@100"])
        rec_r10 = float(rec.metrics["Recall@10"])
        # RRF: best R@100 first, ties broken by R@10 (both descending).
        # Linear / SGDR: ascending sweep value (monotonic trend).
        if family == "rrf":
            sort_key = (-rec_r100, -rec_r10)
        else:
            sort_key = (sweep,)
        rows.append([_GROUP_RANK[family], sort_key, family, HYBRID_SYSTEM,
                     fusion, setting, values])

    rows.sort(key=lambda r: (r[0], r[1]))

    # Carried-forward RRF winner = top-ranked RRF row after the sort.
    winner = next(r for r in rows if r[2] == "rrf")
    winner[3] = HYBRID_SYSTEM + " " + DAGGER

    # Column maxima over hybrid rows only (dense anchor excluded).
    col_max = [max(round(r[6][c], 3) for r in rows) for c in range(len(METRIC_COLS))]

    def cell(value: float, col: int, bold: bool) -> str:
        s = f"{value:.3f}"
        return f"**{s}**" if bold and round(value, 3) == col_max[col] else s

    headers = ["System", "Fusion", "Setting"] + [h for h, _ in METRIC_COLS]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for _gr, _sk, _fam, system, fusion, setting, values in rows:
        cells = [system, fusion, setting] + [cell(v, c, True) for c, v in enumerate(values)]
        lines.append("| " + " | ".join(cells) + " |")

    # Dense anchor reference row, last; never bolded.
    da = recs[DENSE_ANCHOR_ID]
    da_vals = [float(da.metrics[key]) for _hdr, key in METRIC_COLS]
    da_cells = [DENSE_ANCHOR_LABEL, "—", "—"] + [cell(v, c, False) for c, v in enumerate(da_vals)]
    lines.append("| " + " | ".join(da_cells) + " |")

    table = "\n".join(lines) + "\n"
    note = f"\n_{DAGGER} carried forward as the Tier 4 first stage._\n"
    return table + note


def main() -> None:
    # The table contains non-cp1252 glyphs (α); make stdout UTF-8 on Windows.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    table = build()
    print(table)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(table, encoding="utf-8")
    print(f"[done] Wrote {OUT_FILE.relative_to(REPORT_DIR.parent.parent)}")


if __name__ == "__main__":
    main()
