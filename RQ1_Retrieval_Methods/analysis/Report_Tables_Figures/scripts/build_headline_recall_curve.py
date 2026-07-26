"""
Build the Chapter 15 cross-tier headline Recall@k line plot.

Output: Report/figures/fig_rq1_recall_curve.{pdf,png}

Seven systems on one log-x cutoff axis -- one canonical per tier plus T4.0
top-20 (matched-pool anchor). Each line uses its locked color/marker/style
from thesis_style.SYSTEMS:

  T1     bm25_tuned           sky blue solid diamond
  T2     dense_me5_large      orange   solid diamond
  T3     hybrid_rrf_k60       green    solid diamond
  T4.0   llm_rerank_top50     vermillion solid diamond  (pool = 50)
  T4.0   llm_rerank_top20     vermillion dash-dot square (pool = 20)
  T4.1   crag_hybrid_v2       purple   solid diamond  (pool = 20, eval_k)
  T4.2   react_hybrid_v2      teal     solid diamond  (pool = 20)

Pool truncation: each T4 line ends *one K_VALUES cutoff past* its pool so
the plateau is visually confirmed (the rule established in the per-tier
recall curves). T1/T2/T3 systems span the full k range. Pool annotations on
the truncated lines.

This figure is the one-glance synthesis of RQ1's performance ladder.

Usage:
  .venv/Scripts/python Report/scripts/build_headline_recall_curve.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

REPORT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPORT_DIR))

from load_results import load_records, to_long
from thesis_style import (
    get_system,
    rcparams_report,
    save_figure,
)

K_VALUES = [1, 5, 10, 20, 50, 100]
OUTPUT_NAME = "fig_rq1_recall_curve"

HEADLINE_KEYS = [
    "bm25_tuned",
    "dense_me5_large",
    "hybrid_rrf_k60",
    "llm_rerank_top50",
    "llm_rerank_top20",
    "crag_hybrid_v2",
    "react_hybrid_v2",
]

# T1/T2/T3 systems span the full curve (first-stage, pool == 100); T4 systems
# truncate at one K_VALUES step past their effective pool.
POOL_SIZE = {
    "llm_rerank_top50":  50,
    "llm_rerank_top20":  20,
    "crag_hybrid_v2":    20,
    "react_hybrid_v2":   20,
}


def _truncate_k(pool: int, k_values: list[int]) -> int:
    """Largest k to plot for a system with this pool: smallest k_value > pool,
    so the line ends one step past the pool ceiling (confirms plateau)."""
    larger = [k for k in k_values if k > pool]
    return min(larger) if larger else pool


def main() -> None:
    records = load_records()
    long = to_long(records)
    rec = long[(long["metric"] == "Recall")
               & (long["stratum"] == "overall")
               & (long["k"].isin(K_VALUES))]

    series: dict[str, dict] = {}
    for _, row in rec.iterrows():
        s = get_system(row["experiment_id"])
        if s is None or s.key not in HEADLINE_KEYS:
            continue
        d = series.setdefault(s.key, {"style": s, "k": [], "y": []})
        d["k"].append(int(row["k"]))
        d["y"].append(float(row["value"]))

    rcparams_report()
    fig, ax = plt.subplots(figsize=(7.6, 4.4))

    # Plot in canonical headline order so the legend reads T1 -> T4.2.
    for key in HEADLINE_KEYS:
        d = series.get(key)
        if d is None:
            print(f"[warn] missing series for {key}")
            continue
        s = d["style"]
        pool = POOL_SIZE.get(key)

        pairs = sorted(zip(d["k"], d["y"]))
        if pool is not None:
            end_k = _truncate_k(pool, K_VALUES)
            pairs = [(k, y) for k, y in pairs if k <= end_k]
        if not pairs:
            continue
        ks, ys = zip(*pairs)

        ax.plot(
            ks, ys,
            color=s.color, marker=s.marker, linestyle=s.linestyle,
            linewidth=1.6, markersize=5.0, markeredgewidth=0.6,
            label=s.display_short,
        )

        # Pool annotation at the terminal marker for truncated lines.
        if pool is not None:
            ax.annotate(
                f"pool = {pool}",
                xy=(ks[-1], ys[-1]),
                xytext=(8, 0), textcoords="offset points",
                fontsize=7, color=s.color,
                va="center", ha="left",
            )

    ax.set_xscale("log")
    ax.set_xticks(K_VALUES, [str(k) for k in K_VALUES])
    ax.minorticks_off()
    ax.set_xlabel(r"Retrieval depth $k$ (log scale)")
    ax.set_ylabel(r"Recall@$k$ (test, $n=222$)")
    ax.set_xlim(K_VALUES[0] * 0.9, K_VALUES[-1] * 1.1)
    ax.set_ylim(0, None)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)

    ax.legend(
        loc="center left", bbox_to_anchor=(1.02, 0.5),
        fontsize=8, handlelength=2.6, handletextpad=0.6,
        frameon=False, labelspacing=0.4, borderaxespad=0.0,
    )

    fig.subplots_adjust(left=0.08, right=0.66, top=0.95, bottom=0.13)
    paths = save_figure(fig, OUTPUT_NAME)
    plt.close(fig)
    for p in paths:
        print(f"[done] Wrote {p.relative_to(REPORT_DIR.parent)}")


if __name__ == "__main__":
    main()
