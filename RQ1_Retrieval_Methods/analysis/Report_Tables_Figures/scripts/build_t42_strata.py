"""
Build the Tier 4.2 ReAct stratified small-multiples figure.

Output: Report/figures/fig_rq1_t42_strata.{pdf,png}

Same 2x3 layout as the other tier strata figures. Shows the three canonical
ReAct systems plus the T4.0 top-20 matched-pool anchor, the T3-A upstream
reference, and the T4.1 CRAG (hyb.) peer as faded reference bars -- so the
per-stratum agentic-vs-non-agentic comparison reads directly.

Usage:
  .venv/Scripts/python Report/scripts/build_t42_strata.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPORT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPORT_DIR))

from load_results import load_records, to_long
from thesis_style import (
    SYSTEMS,
    rcparams_report,
    save_figure,
)

STRATA_LAYOUT = [
    ("single_article",            0, 0, "Single-article queries"),
    ("multi_article",             0, 1, "Multi-article queries"),
    ("lexically_aligned",         1, 0, "Lexically aligned"),
    ("semantically_paraphrased",  1, 1, "Semantically paraphrased"),
    ("with_cross_refs",           2, 0, "With cross-references"),
    ("without_cross_refs",        2, 1, "Without cross-references"),
]

PLOT_KEYS = [
    "react_hybrid_v2",      # T4.2 canonical
    "react_hybrid_v1",      # v1 reference
    "react_bm25_v1",        # v1 reference
    "crag_hybrid_v2",       # T4.1 agentic peer (faded)
    "llm_rerank_top20",     # T4.0 matched-pool anchor (faded)
    "hybrid_rrf_k60",       # T3-A upstream (faded)
]
REFERENCE_KEYS = {"hybrid_rrf_k60", "llm_rerank_top20", "crag_hybrid_v2"}

OUTPUT_NAME = "fig_rq1_t42_strata"


def main() -> None:
    records = load_records()
    recs = [(t, r) for t, r in records
            if t in ("T3", "T4.0", "T4.1", "T4.2")]
    if not recs:
        raise SystemExit("No T3/T4.* result records found.")

    long = to_long(recs)
    r10 = long[(long["metric"] == "Recall") & (long["k"] == 10)]

    overall_by_exp: dict[str, float] = {
        row["experiment_id"]: float(row["value"])
        for _, row in r10[r10["stratum"] == "overall"].iterrows()
    }

    sys_order: list[dict] = []
    for key in PLOT_KEYS:
        s = SYSTEMS.get(key)
        if s is None:
            continue
        sys_order.append({
            "key":           s.key,
            "display_short": s.display_short,
            "color":         s.color,
            "is_reference":  s.key in REFERENCE_KEYS,
            "overall":       overall_by_exp.get(s.result_id, float("nan")),
        })
    sys_order.sort(key=lambda r: r["overall"] if r["overall"] == r["overall"] else -1)
    n_sys = len(sys_order)
    if n_sys == 0:
        raise SystemExit("Loader produced no matching SYSTEMS entries.")

    by_strat: dict[tuple[str, str], float] = {}
    for _, row in r10.iterrows():
        by_strat[(row["experiment_id"], row["stratum"])] = float(row["value"])

    sys_result_id = {s.key: s.result_id for s in SYSTEMS.values()}

    rcparams_report()
    fig, axes = plt.subplots(
        2, 3,
        figsize=(7.8, 4.6),
        sharey=True,
        gridspec_kw={"wspace": 0.25, "hspace": 0.45},
    )

    y = np.arange(n_sys)
    colors = [s["color"] for s in sys_order]
    alphas = [0.5 if s["is_reference"] else 1.0 for s in sys_order]
    hatches = ["//" if s["is_reference"] else None for s in sys_order]

    for stratum, col, row, title in STRATA_LAYOUT:
        ax = axes[row, col]
        values = []
        for s in sys_order:
            rid = sys_result_id.get(s["key"])
            values.append(by_strat.get((rid, stratum), float("nan")))
        bars = ax.barh(y, values, color=colors, edgecolor="black",
                       linewidth=0.4)
        for bar, a, h in zip(bars, alphas, hatches):
            bar.set_alpha(a)
            if h:
                bar.set_hatch(h)
        ax.set_yticks(y, [s["display_short"] for s in sys_order])

        finite = [v for v in values if v == v]
        xmax = max(finite) if finite else 1.0
        ax.set_xlim(0, max(0.05, xmax * 1.25))
        for yi, v in zip(y, values):
            if v != v:
                continue
            ax.text(v + xmax * 0.04, yi, f"{v:.3f}",
                    va="center", ha="left", fontsize=7)

        ax.set_title(title, fontsize=9, pad=4)
        ax.tick_params(axis="x", labelsize=8)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="x", color="#DDDDDD", linewidth=0.6)
        ax.set_axisbelow(True)

    for ax in axes[1, :]:
        ax.set_xlabel(r"Recall@10")

    fig.tight_layout()
    paths = save_figure(fig, OUTPUT_NAME)
    plt.close(fig)
    for p in paths:
        print(f"[done] Wrote {p.relative_to(REPORT_DIR.parent)}")


if __name__ == "__main__":
    main()
