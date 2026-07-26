"""
Build the Chapter 15 cross-tier headline results table.

Output: Report/tables/tab_rq1_headline.tex

Columns: System | Tier | R@10 | R@20 | R@100 | Hit@10 | n_zero@10 | MRR@10 | NDCG@10

Seven systems -- one canonical per tier, plus T4.0 top-20 so the matched-pool
comparison is readable from the table itself:

  T1     bm25_tuned             (R@10 = 0.265)
  T2     dense_me5_large        (R@10 = 0.342)
  T3     hybrid_rrf_k60         (R@10 = 0.402)
  T4.0   llm_rerank_top50       (R@10 = 0.445, canonical T4.0)
  T4.0   llm_rerank_top20       (R@10 = 0.435, matched-pool anchor)
  T4.1   crag_hybrid_v2         (R@10 = 0.426)
  T4.2   react_hybrid_v2        (R@10 = 0.426)

- R@k cells where k > the system's pool are masked "--" (consistent with
  per-tier tables; pool definitions: T1/T2/T3 = first-stage = 100; T4.0 =
  top_n; T4.1/T4.2 = eval_k = 20). For CRAG and ReAct, R@100 in the JSON
  technically has a value but it reflects first-stage fallback, not the
  agentic mechanism's reach -- masking is the honest move.
- Hit@10 ("at least one relevant in top-10") and n_zero@10 (count of queries
  with no relevant in top-10, out of 222) are headline-only columns per
  STYLEGUIDE.md sec 6; they expose failure-mode information that the
  averaged Recall@k hides.
- No p-value column (per RQ1_STORY.md): significance lives in the per-tier
  tables (Ch. 10-13).

Usage:
  .venv/Scripts/python Report/scripts/build_headline_table.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPORT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPORT_DIR))

from load_results import load_records, to_wide
from thesis_style import (
    SYSTEMS,
    fmt_count,
    fmt_metric,
    get_system,
    save_table,
)

# Headline scope: 7 systems, one per tier + T4.0 matched-pool anchor.
HEADLINE_KEYS = [
    "bm25_tuned",
    "dense_me5_large",
    "hybrid_rrf_k60",
    "llm_rerank_top50",
    "llm_rerank_top20",
    "crag_hybrid_v2",
    "react_hybrid_v2",
]

# Tier label per system, displayed in the second column.
TIER_LABEL = {
    "bm25_tuned":       "T1",
    "dense_me5_large":  "T2",
    "hybrid_rrf_k60":   "T3",
    "llm_rerank_top50": "T4.0",
    "llm_rerank_top20": "T4.0",
    "crag_hybrid_v2":   "T4.1",
    "react_hybrid_v2":  "T4.2",
}

# Pool size per system (used to mask Recall@k cells where k > pool).
# T1/T2/T3 are first-stage retrievers -> pool = 100 (max k we display).
# T4 systems use their reranker top_n / eval_k.
POOL_SIZE = {
    "bm25_tuned":       100,
    "dense_me5_large":  100,
    "hybrid_rrf_k60":   100,
    "llm_rerank_top50":  50,
    "llm_rerank_top20":  20,
    "crag_hybrid_v2":    20,   # eval_k
    "react_hybrid_v2":   20,   # effective pool (R@k plateau at ~20-50)
}

# Recall@k columns (other than R@10) need masking based on pool.
RECALL_COLUMNS_K = {"Recall@10": 10, "Recall@20": 20, "Recall@100": 100}

METRICS = [
    "Recall@10", "Recall@20", "Recall@100",
    "Hit@10", "n_zero@10",
    "MRR@10", "NDCG@10",
]

# Visual group breaks: which keys begin a new \midrule block.
GROUP_HEADS = {
    "dense_me5_large":   True,   # break before T2
    "hybrid_rrf_k60":    True,   # break before T3
    "llm_rerank_top50":  True,   # break before T4 block
}


def _render_table(rows: list[dict], best: dict[str, float]) -> str:
    n_recall_cols = 3   # R@10 / R@20 / R@100 -- siunitx-formatted
    # System | Tier | 3 recall | Hit@10 | n_zero@10 | MRR@10 | NDCG@10
    col_spec = (
        "l l "
        + " ".join(["S[table-format=1.3]"] * n_recall_cols)
        + " S[table-format=1.3]"   # Hit@10
        + " S[table-format=3]"     # n_zero@10 (integer)
        + " S[table-format=1.3]"   # MRR@10
        + " S[table-format=1.3]"   # NDCG@10
    )

    out: list[str] = []
    out.append(r"\begin{tabular}{" + col_spec + "}")
    out.append(r"\toprule")
    out.append(
        r"System & Tier & "
        r"{R@10} & {R@20} & {R@100} & "
        r"{Hit@10} & {$n_{0}$@10} & "
        r"{MRR@10} & {NDCG@10} \\"
    )
    out.append(r"\midrule")

    for i, r in enumerate(rows):
        if i > 0 and GROUP_HEADS.get(r["key"], False):
            out.append(r"\midrule")

        cells: list[str] = [r["display_short"], r["tier"]]
        pool = POOL_SIZE.get(r["key"])

        for m in METRICS:
            val = r.get(m)
            # Recall@k masking for k > pool.
            k = RECALL_COLUMNS_K.get(m)
            if k is not None and pool is not None and k > pool:
                cells.append("{--}")
                continue

            # Integer formatting for n_zero@10.
            if m == "n_zero@10":
                text = fmt_count(val)
                if val is not None and best.get(m) is not None \
                   and int(round(val)) == int(round(best[m])):
                    # Best = lowest n_zero (fewest catastrophic failures).
                    text = rf"\textbf{{{text}}}"
                cells.append(text)
                continue

            text = fmt_metric(m, val)
            if val is not None and best.get(m) is not None \
               and abs(val - best[m]) < 1e-12:
                text = rf"\textbf{{{text}}}"
            if "\\" in text:
                text = "{" + text + "}"
            cells.append(text)

        out.append(" & ".join(cells) + r" \\")

    out.append(r"\bottomrule")
    out.append(r"\end{tabular}")
    return "\n".join(out) + "\n"


def main() -> None:
    records = load_records()
    wide = to_wide(records)

    rows: list[dict] = []
    for key in HEADLINE_KEYS:
        s = SYSTEMS.get(key)
        if s is None:
            print(f"[warn] unknown SYSTEMS key in headline scope: {key!r}")
            continue
        match = wide[wide["experiment_id"] == s.result_id]
        if match.empty:
            print(f"[warn] no result row for {key} (result_id={s.result_id!r})")
            continue
        w = match.iloc[0]
        rows.append({
            "key":            s.key,
            "display_short":  s.display_short,
            "tier":           TIER_LABEL[s.key],
            **{m: w.get(m) for m in METRICS},
        })

    if not rows:
        raise SystemExit("No headline rows produced; check SYSTEMS result_ids.")

    # Best per column, excluding masked cells (k > pool).
    best: dict[str, float] = {}
    for m in METRICS:
        vals = []
        for r in rows:
            v = r.get(m)
            if v is None:
                continue
            k = RECALL_COLUMNS_K.get(m)
            pool = POOL_SIZE.get(r["key"])
            if k is not None and pool is not None and k > pool:
                continue
            vals.append(v)
        if not vals:
            continue
        # n_zero@10: lower is better (fewer catastrophic failures).
        best[m] = min(vals) if m == "n_zero@10" else max(vals)

    tex = _render_table(rows, best)
    out_path = save_table(tex, "tab_rq1_headline")
    print(f"[done] Wrote {out_path.relative_to(REPORT_DIR.parent)}")
    print(f"[done] {len(rows)} rows. Best per column:")
    for m, v in best.items():
        if m == "n_zero@10":
            print(f"        {m:12s} = {int(round(v)):d}  (lower is better)")
        else:
            print(f"        {m:12s} = {v:.4f}")


if __name__ == "__main__":
    main()
