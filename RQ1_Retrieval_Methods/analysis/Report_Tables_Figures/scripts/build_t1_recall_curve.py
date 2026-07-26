"""
Build the Tier 1 Recall@k line plot.

Output: Report/figures/fig_rq1_t1_recall_curve.{pdf,png}

What's drawn
------------
- One line per Tier 1 system (12 total)
- x-axis: retrieval depth k in {1, 5, 10, 20, 50, 100}, log-scale
- y-axis: mean Recall@k over the 222 test queries
- Colors / markers / linestyles from the locked thesis_style.SYSTEMS map.
  BM25 family = blue / sky variants, TF-IDF = dark grey, FTS5 = mid grey.
  Anchor uses a darker shade; winners use solid lines; ablations use dotted.
- Legend outside the axes, grouped by family.

This is the per-tier figure that replicates verbatim at T2, T3, T4 (and as the
cross-tier headline in Chapter 15) with the relevant SYSTEMS subset.

Usage:
  .venv/Scripts/python Report/scripts/build_t1_recall_curve.py
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
    FIGSIZE,
    SYSTEMS,
    get_system,
    rcparams_report,
    save_figure,
)

# x-axis cutoffs to show. 200/500 are in the harness output too but R@200 ==
# R@500 == R@100 for any first-stage retriever that returns top-100, so we
# stop at 100.
K_VALUES = [1, 5, 10, 20, 50, 100]


def main() -> None:
    records = load_records()
    t1 = [(t, r) for t, r in records if t == "T1"]
    if not t1:
        raise SystemExit("No Tier 1 result records found.")

    # Pull Recall rows from the long view, restricted to (overall, Recall, k in K_VALUES).
    long = to_long(t1)
    rec = long[(long["metric"] == "Recall")
               & (long["stratum"] == "overall")
               & (long["k"].isin(K_VALUES))]

    # One series per system, keyed by canonical SYSTEMS order.
    order_index = {k: i for i, k in enumerate(SYSTEMS)}
    series: dict[str, dict] = {}
    for _, row in rec.iterrows():
        s = get_system(row["experiment_id"])
        if s is None:
            continue
        d = series.setdefault(s.key, {
            "style": s,
            "k": [], "y": [],
        })
        d["k"].append(int(row["k"]))
        d["y"].append(float(row["value"]))

    # Stable plotting order: canonical SYSTEMS declaration.
    keys_sorted = sorted(series.keys(), key=lambda k: order_index.get(k, 10_000))

    rcparams_report()
    # Slightly larger than report_wide; the 12-system legend wants room.
    fig, ax = plt.subplots(figsize=(7.4, 4.0))

    for key in keys_sorted:
        d = series[key]
        s = d["style"]
        # Sort points by k so the line draws monotonically.
        ks, ys = zip(*sorted(zip(d["k"], d["y"])))
        ax.plot(
            ks, ys,
            color=s.color, marker=s.marker, linestyle=s.linestyle,
            linewidth=1.4, markersize=4.5, markeredgewidth=0.6,
            label=s.display_short,
        )

    ax.set_xscale("log")
    ax.set_xticks(K_VALUES, [str(k) for k in K_VALUES])
    ax.minorticks_off()
    ax.set_xlabel(r"Retrieval depth $k$ (log scale)")
    ax.set_ylabel(r"Recall@$k$ (test, $n=222$)")
    ax.set_xlim(K_VALUES[0] * 0.9, K_VALUES[-1] * 1.1)
    ax.set_ylim(0, None)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)

    # Family-grouped legend on the right.
    ax.legend(
        loc="center left", bbox_to_anchor=(1.02, 0.5),
        fontsize=8, handlelength=2.6, handletextpad=0.6,
        frameon=False, borderaxespad=0.0,
        labelspacing=0.4,
    )

    fig.subplots_adjust(left=0.08, right=0.74, top=0.95, bottom=0.13)
    paths = save_figure(fig, "fig_rq1_t1_recall_curve")
    plt.close(fig)
    for p in paths:
        print(f"[done] Wrote {p.relative_to(REPORT_DIR.parent)}")


if __name__ == "__main__":
    main()
