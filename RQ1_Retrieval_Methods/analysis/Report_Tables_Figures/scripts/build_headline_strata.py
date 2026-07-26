"""
Build the Chapter 15 cross-tier headline stratified small-multiples figure.

Output: Report/figures/fig_rq1_strata.{pdf,png}

Same 2 x 3 layout as the per-tier strata figures, populated with the seven
headline systems (one canonical per tier + T4.0 matched-pool anchor).

  Cols:  by # relevant articles  |  by lexical alignment  |  by cross-references
  Row 1: single_article          |  lexically_aligned     |  with_cross_refs
  Row 2: multi_article           |  semantically_paraphr. |  without_cross_refs

Per-stratum colors come from the locked SYSTEMS map; no reference fading
here because every row is a headline system in its own right.

This is the "under what conditions" closer for RQ1 -- the second half of the
research question is answered by reading down each column.

Usage:
  .venv/Scripts/python Report/scripts/build_headline_strata.py
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

HEADLINE_KEYS = [
    "bm25_tuned",
    "dense_me5_large",
    "hybrid_rrf_k60",
    "llm_rerank_top50",
    "llm_rerank_top20",
    "crag_hybrid_v2",
    "react_hybrid_v2",
]

OUTPUT_NAME = "fig_rq1_strata"


def main() -> None:
    records = load_records()
    long = to_long(records)
    r10 = long[(long["metric"] == "Recall") & (long["k"] == 10)]

    overall_by_exp: dict[str, float] = {
        row["experiment_id"]: float(row["value"])
        for _, row in r10[r10["stratum"] == "overall"].iterrows()
    }

    sys_order: list[dict] = []
    for key in HEADLINE_KEYS:
        s = SYSTEMS.get(key)
        if s is None:
            continue
        sys_order.append({
            "key":           s.key,
            "display_short": s.display_short,
            "color":         s.color,
            "overall":       overall_by_exp.get(s.result_id, float("nan")),
        })
    # Sort ASC by overall R@10 so best ends up at the top of each panel
    # (matplotlib default + barh: index 0 -> bottom).
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
        figsize=(7.5, 4.4),
        sharey=True,
        gridspec_kw={"wspace": 0.25, "hspace": 0.45},
    )

    y = np.arange(n_sys)
    colors = [s["color"] for s in sys_order]

    for stratum, col, row, title in STRATA_LAYOUT:
        ax = axes[row, col]
        values = []
        for s in sys_order:
            rid = sys_result_id.get(s["key"])
            values.append(by_strat.get((rid, stratum), float("nan")))
        ax.barh(y, values, color=colors, edgecolor="black", linewidth=0.4)
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
