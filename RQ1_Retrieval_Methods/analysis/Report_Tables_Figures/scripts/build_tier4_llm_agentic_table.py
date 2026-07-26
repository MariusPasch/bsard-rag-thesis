"""
Build the RQ1 Tier 4 (LLM-Judge + agentic) results table -- Chapter 5, Table 5.4.

Reads the matched Tier 4 records (all on the hybrid RRF k=60 first stage, v2 /
matched design points) plus the hybrid first stage as a context row, and emits
one GitHub-flavoured **markdown** results table for the thesis report.

Tier 4 flexes the recall columns relative to Tiers 1-3: it ADDS R@20 (the
matched-pool cut-off) and conditionally OMITS the saturated R@100 (see below).

Output:
  - printed to stdout (with a saturation-audit block)
  - written to analysis/tables/tier4_llm_agentic_results.md

Columns (in order):
  Sub-tier | Mechanism | Pool | R@1 | R@5 | R@10 | R@20 | R@50 | R@100
           | MRR@100 | NDCG@10 | MAP@10 | Hit@10

Saturation rule (A):
  Pool-limited re-rankers cannot retrieve beyond their pool, so their Recall@k
  plateaus once k reaches the pool size. Where Recall@100 == Recall@50 the R@100
  cell is rendered "—" (it carries no information beyond R@50): this hits the
  LLM-Judge top-50 / top-20 and ReAct v2 rows. CRAG's loop expands the pool, so
  R@100 (0.654) is a genuine gain over R@50 and is kept; the hybrid reference
  keeps R@100 too. R@50 is kept for every row. The script also prints, per
  system, the smallest k where Recall@k == Recall@500, so the omissions can be
  audited.

Other conventions:
  - MRR@100 is used (for the re-rankers it equals the within-pool MRR, since the
    first relevant rank lies inside the pool).
  - HitRate@10 maps to the Hit@10 column. Metrics rounded to 3 dp.
  - Bold: best in each metric column among the four Tier-4 sub-tier rows only;
    the hybrid reference row is context and is excluded from the comparison.

Row order: hybrid reference, then T4.0 top-50, T4.0 top-20 (matched), T4.1 CRAG,
T4.2 ReAct v2.

Usage:
  python analysis/Report_Tables_Figures/scripts/build_tier4_llm_agentic_table.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPORT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPORT_DIR))

from load_results import load_records  # noqa: E402

OUT_DIR = REPORT_DIR.parent / "tables"          # <project>/analysis/tables
OUT_FILE = OUT_DIR / "tier4_llm_agentic_results.md"

# Ordered rows: (experiment_id, sub_tier, mechanism, pool, is_reference).
ROWS: list[tuple[str, str, str, str, bool]] = [
    ("hybrid_rrf_k60_test",                              "—",    "Hybrid (RRF) first stage", "full",              True),
    ("llm_rerank_binary_top50_hybrid_rrf_k60_test",      "T4.0", "LLM-Judge (binary)",       "top-50",            False),
    ("llm_rerank_binary_top20_hybrid_rrf_k60_test",      "T4.0", "LLM-Judge (binary)",       "top-20 (matched)",  False),
    ("crag_hybrid_rrf_k60_test_v2",                      "T4.1", "CRAG",                     "top-20 + loop",     False),
    ("react_hybrid_rrf_k60_test_v2",                     "T4.2", "ReAct v2",                 "top-20 + loop",     False),
]

# (column header, record metrics key). All are higher-is-better.
METRIC_COLS: list[tuple[str, str]] = [
    ("R@1", "Recall@1"),
    ("R@5", "Recall@5"),
    ("R@10", "Recall@10"),
    ("R@20", "Recall@20"),
    ("R@50", "Recall@50"),
    ("R@100", "Recall@100"),
    ("MRR@100", "MRR@100"),
    ("NDCG@10", "NDCG@10"),
    ("MAP@10", "MAP@10"),
    ("Hit@10", "HitRate@10"),
]
R100_COL = [h for h, _ in METRIC_COLS].index("R@100")
EMDASH = "—"


def _records_by_id() -> dict:
    return {rec.experiment_id: rec for _tier, rec in load_records()}


def _saturation_k(rec) -> int | None:
    """Smallest k where Recall@k == Recall@500 (the plateau point)."""
    ladder = sorted(int(key.split("@")[1]) for key in rec.metrics
                    if key.startswith("Recall@"))
    r500 = rec.metrics.get("Recall@500")
    if r500 is None:
        return None
    for k in ladder:
        v = rec.metrics.get(f"Recall@{k}")
        if v is not None and abs(v - r500) < 1e-9:
            return k
    return None


def build() -> tuple[str, list[tuple[str, int | None]]]:
    recs = _records_by_id()
    missing = [eid for eid, *_ in ROWS if eid not in recs]
    if missing:
        raise SystemExit(
            "Missing Tier 4 records (is the data bundle present under "
            f"output/results/?):\n  " + "\n  ".join(missing))

    rows = []      # dicts with display fields + numeric values + flags
    audit = []     # (label, saturation_k)
    for eid, sub, mech, pool, is_ref in ROWS:
        rec = recs[eid]
        vals = [float(rec.metrics[key]) for _hdr, key in METRIC_COLS]
        # Omit R@100 where it carries no information beyond R@50 (pool plateau).
        omit_r100 = round(vals[R100_COL], 3) == round(rec.metrics["Recall@50"], 3)
        rows.append(dict(sub=sub, mech=mech, pool=pool, ref=is_ref,
                         vals=vals, omit_r100=omit_r100))
        audit.append((f"{mech} ({pool})", _saturation_k(rec)))

    # Column maxima over the four Tier-4 rows only (reference excluded), counting
    # only cells that are actually displayed (omitted R@100 cells don't compete).
    col_max: list[float | None] = []
    for c, (hdr, _key) in enumerate(METRIC_COLS):
        cands = [r["vals"][c] for r in rows
                 if not r["ref"] and not (c == R100_COL and r["omit_r100"])]
        col_max.append(max(cands) if cands else None)

    def cell(row: dict, c: int) -> str:
        if c == R100_COL and row["omit_r100"]:
            return EMDASH
        v = row["vals"][c]
        s = f"{v:.3f}"
        if (not row["ref"]) and col_max[c] is not None and round(v, 3) == round(col_max[c], 3):
            return f"**{s}**"
        return s

    headers = ["Sub-tier", "Mechanism", "Pool"] + [h for h, _ in METRIC_COLS]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for r in rows:
        cells = [r["sub"], r["mech"], r["pool"]] + [cell(r, c) for c in range(len(METRIC_COLS))]
        lines.append("| " + " | ".join(cells) + " |")

    table = "\n".join(lines) + "\n"
    note = ("\n_R@100 is omitted (—) for pool-limited re-rankers where it equals "
            "R@50; kept for CRAG (loop expands the pool) and the hybrid "
            "reference. R@50 is shown for all rows._\n")
    return table + note, audit


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    table, audit = build()
    print("Saturation audit (smallest k with Recall@k == Recall@500):")
    for label, k in audit:
        print(f"  {label:38s} k = {k}")
    print()
    print(table)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(table, encoding="utf-8")
    print(f"[done] Wrote {OUT_FILE.relative_to(REPORT_DIR.parent.parent)}")


if __name__ == "__main__":
    main()
