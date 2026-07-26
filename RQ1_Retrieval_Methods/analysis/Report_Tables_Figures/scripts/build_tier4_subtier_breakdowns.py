"""
Build the Tier 4 sub-tier breakdown "story" tables -- Chapter 5, Tables 5.4a/b/c.

Three small ablation tables, one per agentic sub-tier, written together to
analysis/tables/tier4_subtier_breakdowns.md:

  5.4a  LLM-Judge   -- scoring paradigm x first stage x pool size
  5.4b  CRAG        -- first-stage ablation, both at the final v2 config
  5.4c  ReAct       -- v2 (final) vs v1, same RRF-k60 first stage

Output:
  - printed to stdout (with a saturation-audit block)
  - written to analysis/tables/tier4_subtier_breakdowns.md

Common conventions:
  - HitRate@10 maps to the Hit@10 column; MRR@100 is used. Metrics to 3 dp.
  - Best value in each metric column is **bolded** (over all rows of that table).
  - The headline configuration carried into the main Tier 4 table (5.4) is
    marked with a star.
  - The script prints each record's saturation point (smallest k where
    Recall@k == Recall@500) for audit. 5.4a / 5.4c omit R@100 because it is
    saturated for these pool-limited re-rankers; 5.4b keeps R@100 (CRAG's loop
    expands the pool).

Usage:
  python analysis/Report_Tables_Figures/scripts/build_tier4_subtier_breakdowns.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPORT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPORT_DIR))

from load_results import load_records  # noqa: E402

OUT_DIR = REPORT_DIR.parent / "tables"          # <project>/analysis/tables
OUT_FILE = OUT_DIR / "tier4_subtier_breakdowns.md"

STAR = "★"

# Metric column menus (header, record metrics key). All higher-is-better.
_M = {
    "R@10": "Recall@10", "R@20": "Recall@20", "R@50": "Recall@50",
    "R@100": "Recall@100", "MRR@100": "MRR@100", "NDCG@10": "NDCG@10",
    "MAP@10": "MAP@10", "Hit@10": "HitRate@10",
}


def _mcols(*headers: str) -> list[tuple[str, str]]:
    return [(h, _M[h]) for h in headers]


# Each table: title, optional caption, config column headers, metric columns,
# and rows. A row = (experiment_id, [config cells], star_col index or None).
TABLES = [
    dict(
        title="### Table 5.4a — LLM-Judge: scoring paradigm × first stage × pool",
        caption=None,
        config_headers=["Scoring", "First stage", "Pool"],
        metric_cols=_mcols("R@10", "R@20", "R@50", "MRR@100", "NDCG@10", "Hit@10"),
        rows=[
            ("llm_rerank_binary_top50_hybrid_rrf_k60_test", ["Binary", "Hybrid", "top-50"], 2),
            ("llm_rerank_binary_top20_hybrid_rrf_k60_test", ["Binary", "Hybrid", "top-20"], None),
            ("llm_rerank_binary_top50_test",                ["Binary", "Sparse", "top-50"], None),
            ("llm_rerank_0to10_top50_test",                 ["0–10", "Hybrid", "top-50"], None),
            ("llm_rerank_0to10_top20_test",                 ["0–10", "Hybrid", "top-20"], None),
        ],
    ),
    dict(
        title="### Table 5.4b — CRAG: first-stage ablation",
        caption=("_Both at the final v2 config (LLaMA collective evaluator, "
                 "eval_k 20, ≤6 iters); only the first stage differs._"),
        config_headers=["First stage"],
        metric_cols=_mcols("R@10", "R@20", "R@50", "R@100", "MRR@100", "NDCG@10", "MAP@10", "Hit@10"),
        rows=[
            ("crag_hybrid_rrf_k60_test_v2", ["Hybrid RRF-k60"], 0),
            ("crag_bm25_test_v2",           ["BM25"], None),
        ],
    ),
    dict(
        title="### Table 5.4c — ReAct: v2 (final) vs v1",
        caption="_Same RRF-k60 first stage; the agent configuration differs._",
        config_headers=["Version", "Key configuration"],
        metric_cols=_mcols("R@10", "R@20", "R@50", "MRR@100", "NDCG@10", "MAP@10", "Hit@10"),
        rows=[
            ("react_hybrid_rrf_k60_test_v2", ["v2 (final)",
             "obs. window 3 · overlap guard 0.6 · function-calling · 8 steps · top-up + re-rank"], 0),
            ("react_hybrid_rrf_k60_test", ["v1",
             "obs. window 10 · no overlap guard · free-text actions · 5 steps"], None),
        ],
    ),
]


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


def _render(spec: dict, recs: dict) -> tuple[str, list[tuple[str, int | None]]]:
    metric_cols = spec["metric_cols"]
    cfg_headers = spec["config_headers"]

    body = []       # (config cells with star applied, [metric values])
    audit = []
    for eid, cfg, star_col in spec["rows"]:
        if eid not in recs:
            raise SystemExit(f"Missing record {eid!r} for {spec['title']}")
        rec = recs[eid]
        cells = list(cfg)
        if star_col is not None:
            cells[star_col] = f"{cells[star_col]} {STAR}"
        vals = [float(rec.metrics[key]) for _h, key in metric_cols]
        body.append((cells, vals))
        audit.append((eid, _saturation_k(rec)))

    col_max = [max(round(v[1][c], 3) for v in body) for c in range(len(metric_cols))]

    def cell(value: float, c: int) -> str:
        s = f"{value:.3f}"
        return f"**{s}**" if round(value, 3) == col_max[c] else s

    headers = cfg_headers + [h for h, _ in metric_cols]
    lines = [spec["title"], ""]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for cells, vals in body:
        row = cells + [cell(v, c) for c, v in enumerate(vals)]
        lines.append("| " + " | ".join(row) + " |")
    if spec["caption"]:
        lines += ["", spec["caption"]]
    return "\n".join(lines) + "\n", audit


def build() -> tuple[str, list[tuple[str, str, int | None]]]:
    recs = _records_by_id()
    blocks = []
    all_audit: list[tuple[str, str, int | None]] = []
    for spec in TABLES:
        md, audit = _render(spec, recs)
        blocks.append(md)
        tag = spec["title"].split("—")[0].replace("###", "").strip()
        all_audit += [(tag, eid, k) for eid, k in audit]
    doc = "\n".join(blocks)
    doc += f"\n{STAR} headline configuration (carried into Table 5.4).\n"
    return doc, all_audit


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    doc, audit = build()
    print("Saturation audit (smallest k with Recall@k == Recall@500):")
    for tag, eid, k in audit:
        print(f"  {tag:11s} {eid:48s} k = {k}")
    print()
    print(doc)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(doc, encoding="utf-8")
    print(f"[done] Wrote {OUT_FILE.relative_to(REPORT_DIR.parent.parent)}")


if __name__ == "__main__":
    main()
