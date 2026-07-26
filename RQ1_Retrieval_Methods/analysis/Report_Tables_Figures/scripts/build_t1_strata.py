"""
Build the Tier 1 stratified small-multiples figure.

Output: Report/figures/fig_rq1_t1_strata.{pdf,png}

What's drawn
------------
2 x 3 grid of horizontal-bar panels:
  Cols:  by # relevant articles  |  by lexical alignment  |  by cross-references
  Row 1: single_article          |  lexically_aligned     |  with_cross_refs
  Row 2: multi_article           |  semantically_paraphr. |  without_cross_refs

Each panel:
  - Horizontal bars of R@10 within that stratum
  - One bar per Tier 1 system (12 bars)
  - Same y-axis (system order) across all panels: sorted by overall R@10 desc
  - Bar colors from the locked thesis_style.SYSTEMS map (per-system colors)
  - Per-panel x-axis (strata span wildly different R@10 ranges: lex ~ 0.5,
    paraphrased ~ 0.05; a shared x would hide the bars in the right panels)
  - Each bar annotated with its value in 3 dp
  - Panel title = stratum name

For Tier 1, this figure makes the chapter's *behavior* story visible:
  - col 1: gap between single- and multi-article performance
  - col 2: sparse retrieval's reliance on lexical overlap (the paraphrase gap
    that motivates Tier 2)
  - col 3: cross-reference presence correlates with retrievability

Replicates at T2 / T3 / T4 with the tier's system list.

Usage:
  .venv/Scripts/python Report/scripts/build_t1_strata.py
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
    get_system,
    rcparams_report,
    save_figure,
)

# Six strata in three pairs. Each cell = (column_index, row_index, label).
STRATA_LAYOUT = [
    # col 0 -- by # relevant articles
    ("single_article",            0, 0, "Single-article queries"),
    ("multi_article",             0, 1, "Multi-article queries"),
    # col 1 -- by lexical alignment
    ("lexically_aligned",         1, 0, "Lexically aligned"),
    ("semantically_paraphrased",  1, 1, "Semantically paraphrased"),
    # col 2 -- by cross-references
    ("with_cross_refs",           2, 0, "With cross-references"),
    ("without_cross_refs",        2, 1, "Without cross-references"),
]


def main() -> None:
    records = load_records()
    t1 = [(t, r) for t, r in records if t == "T1"]
    if not t1:
        raise SystemExit("No Tier 1 result records found.")

    long = to_long(t1)
    r10 = long[(long["metric"] == "Recall") & (long["k"] == 10)]

    # ---- system order: descending by overall R@10 ------------------------
    overall = r10[r10["stratum"] == "overall"]
    sys_order: list[dict] = []
    for _, row in overall.iterrows():
        s = get_system(row["experiment_id"])
        if s is None:
            continue
        sys_order.append({
            "key":           s.key,
            "display_short": s.display_short,
            "color":         s.color,
            "overall":       float(row["value"]),
        })
    # Sort ASCENDING (worst at index 0). Combined with matplotlib's default
    # y-axis direction (y=0 at bottom), this puts the best system at the top
    # of each panel without needing invert_yaxis -- which would toggle 6 times
    # across the sharey panels and cancel out.
    sys_order.sort(key=lambda r: r["overall"])
    n_sys = len(sys_order)
    if n_sys == 0:
        raise SystemExit("Loader produced no matching SYSTEMS entries.")

    # Build a {(experiment_id, stratum): R@10} index for quick lookup.
    by_strat: dict[tuple[str, str], float] = {}
    for _, row in r10.iterrows():
        by_strat[(row["experiment_id"], row["stratum"])] = float(row["value"])

    # Map system key -> result_id for the lookup.
    sys_result_id = {s.key: s.result_id for s in SYSTEMS.values()}

    # ---- plot -----------------------------------------------------------
    rcparams_report()
    fig, axes = plt.subplots(
        2, 3,
        figsize=(7.5, 5.4),
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

        # Per-panel x-axis. Annotate each bar with its value.
        finite = [v for v in values if v == v]   # filters NaN
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

    # Bottom-row x-axis label.
    for ax in axes[1, :]:
        ax.set_xlabel(r"Recall@10")

    fig.tight_layout()
    paths = save_figure(fig, "fig_rq1_t1_strata")
    plt.close(fig)
    for p in paths:
        print(f"[done] Wrote {p.relative_to(REPORT_DIR.parent)}")


if __name__ == "__main__":
    main()
