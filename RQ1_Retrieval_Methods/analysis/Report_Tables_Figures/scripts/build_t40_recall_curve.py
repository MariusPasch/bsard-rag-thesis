"""
Build the Tier 4.0 Recall@k line plot.

Output: Report/figures/fig_rq1_t40_recall_curve.{pdf,png}

What's drawn
------------
Four T4.0 systems (the canonical proposal scope) plus the T3-A hybrid winner
as a faded green reference line so the lift from "best non-LLM pipeline" to
"LLM-Judge re-rank" is visible:

  - LLM-Judge (hybrid, binary, top-50)  -- T4.0 canonical, R@10 = 0.445
  - LLM-Judge (hybrid, binary, top-20)  -- matched-pool anchor for T4.1/T4.2
  - LLM-Judge (BM25, binary, top-50)    -- first-stage ablation
  - LLM-Judge (BM25, numeric 0-10, top-50) -- scoring-paradigm ablation
  - Hybrid RRF (k=60), T3-A reference (faded green dashed)

Pool ceilings.
  T4.0 reranker R@k plateaus at k > pool_size (no candidates beyond the pool
  to surface). Vertical grey dotted lines mark each system's pool ceiling so
  the plateau is a *visible feature* of the figure rather than a footnote.
  See STYLEGUIDE.md sec 6 -- the matched-pool rule.

Usage:
  .venv/Scripts/python Report/scripts/build_t40_recall_curve.py
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
OUTPUT_NAME = "fig_rq1_t40_recall_curve"

PLOT_KEYS = [
    "llm_rerank_top50",            # canonical
    "llm_rerank_top20",            # matched-pool anchor
    "llm_rerank_top50_bm25",       # first-stage ablation
    "llm_rerank_0to10_top50_bm25", # scoring-paradigm ablation
]
REFERENCE_KEY = "hybrid_rrf_k60"   # T3-A winner

# Per-system pool size. Each line is plotted up to *one K_VALUES cutoff past*
# this value, so the plateau (a single point that doesn't move past R@pool)
# is visually confirmed in the figure.
POOL_SIZE: dict[str, int] = {
    "llm_rerank_top50":            50,
    "llm_rerank_top20":            20,
    "llm_rerank_top50_bm25":       50,
    "llm_rerank_0to10_top50_bm25": 50,
}


def _truncate_k(pool: int, k_values: list[int]) -> int:
    """Largest k in K_VALUES that we should plot for a system with this pool.

    Rule: include the smallest k_value strictly greater than `pool`, so the
    line ends one step past the pool ceiling. If no such larger value exists,
    cap at pool itself.
    """
    larger = [k for k in k_values if k > pool]
    return min(larger) if larger else pool


def main() -> None:
    records = load_records()
    recs = [(t, r) for t, r in records if t in ("T3", "T4.0")]
    if not recs:
        raise SystemExit("No T3/T4.0 result records found.")

    long = to_long(recs)
    rec = long[(long["metric"] == "Recall")
               & (long["stratum"] == "overall")
               & (long["k"].isin(K_VALUES))]

    series: dict[str, dict] = {}
    for _, row in rec.iterrows():
        s = get_system(row["experiment_id"])
        if s is None or s.key not in (PLOT_KEYS + [REFERENCE_KEY]):
            continue
        d = series.setdefault(s.key, {"style": s, "k": [], "y": []})
        d["k"].append(int(row["k"]))
        d["y"].append(float(row["value"]))

    rcparams_report()
    fig, ax = plt.subplots(figsize=(7.6, 4.0))

    # Reference first (sits behind the foreground lines).
    ref = series.get(REFERENCE_KEY)
    if ref is not None:
        ks, ys = zip(*sorted(zip(ref["k"], ref["y"])))
        ax.plot(
            ks, ys,
            color=ref["style"].color, marker=ref["style"].marker,
            linestyle=(0, (3, 3)), linewidth=1.0, markersize=4.0,
            alpha=0.55,
            label=f"{ref['style'].display_short} (T3-A, reference)",
        )

    # T4.0 systems on top. Each line ends *one K_VALUES cutoff past* its
    # pool, so the reader sees a single plateau point that confirms the
    # ceiling -- e.g. top-20 lines extend to k=50 (R@50 == R@20 because
    # the reranker has no candidates beyond position 20 to surface).
    for key in PLOT_KEYS:
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
            linewidth=1.5, markersize=5.0, markeredgewidth=0.6,
            label=s.display_short,
        )

        # Annotate the terminal point with the pool size so the reader sees
        # *why* the line stops moving where it does.
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
