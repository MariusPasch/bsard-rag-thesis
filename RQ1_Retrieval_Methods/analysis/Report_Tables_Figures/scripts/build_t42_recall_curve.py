"""
Build the Tier 4.2 ReAct Recall@k line plot.

Output: Report/figures/fig_rq1_t42_recall_curve.{pdf,png}

Shows up to six lines (three T4.2 + three references):
  - ReAct (hybrid, v2)        -- T4.2 canonical, max_steps=8 + redesign
  - ReAct (hybrid, v1)        -- under-performed v1 reference (max_steps=5)
  - ReAct (BM25, v1)          -- under-performed v1 reference (max_steps=5)
  - Hybrid RRF (k=60)         -- T3-A upstream (faded green dashed)
  - LLM-Judge top-20 (hyb.)   -- T4.0 matched-pool anchor (faded vermillion)
  - CRAG (hyb., v2)           -- T4.1 agentic peer (faded purple)

Pool truncation.
  All ReAct systems and the T4.0 / CRAG references use pool = 20 (the
  LLM-touched window). Per the styleguide one-past-pool rule, each line
  extends to k = 50 so the plateau is visually confirmed. T3-A is
  first-stage (pool = 100), no truncation.

Usage:
  .venv/Scripts/python Report/scripts/build_t42_recall_curve.py
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
OUTPUT_NAME = "fig_rq1_t42_recall_curve"

# Main T4.2 lines (full-weight)
REACT_KEYS = ["react_hybrid_v2", "react_hybrid_v1", "react_bm25_v1"]
# Reference lines (faded)
REFERENCE_KEYS = [
    "hybrid_rrf_k60",      # T3-A upstream
    "llm_rerank_top20",    # T4.0 matched-pool anchor
    "crag_hybrid_v2",      # T4.1 agentic peer
]

# Pool ceilings (LLM-touched window). T3-A has no ceiling.
POOL_SIZE: dict[str, int] = {
    "react_hybrid_v2":  20,
    "react_hybrid_v1":  20,
    "react_bm25_v1":    20,
    "llm_rerank_top20": 20,
    "crag_hybrid_v2":   20,
}


def _truncate_k(pool: int, k_values: list[int]) -> int:
    larger = [k for k in k_values if k > pool]
    return min(larger) if larger else pool


def main() -> None:
    records = load_records()
    recs = [(t, r) for t, r in records
            if t in ("T3", "T4.0", "T4.1", "T4.2")]
    if not recs:
        raise SystemExit("No T3/T4.* result records found.")

    long = to_long(recs)
    rec = long[(long["metric"] == "Recall")
               & (long["stratum"] == "overall")
               & (long["k"].isin(K_VALUES))]

    plot_keys = set(REACT_KEYS) | set(REFERENCE_KEYS)
    series: dict[str, dict] = {}
    for _, row in rec.iterrows():
        s = get_system(row["experiment_id"])
        if s is None or s.key not in plot_keys:
            continue
        d = series.setdefault(s.key, {"style": s, "k": [], "y": []})
        d["k"].append(int(row["k"]))
        d["y"].append(float(row["value"]))

    rcparams_report()
    fig, ax = plt.subplots(figsize=(7.6, 4.2))

    def _plot(key: str, *, faded: bool, label_suffix: str = "") -> None:
        d = series.get(key)
        if d is None:
            print(f"[warn] missing series for {key}")
            return
        s = d["style"]
        pool = POOL_SIZE.get(key)
        pairs = sorted(zip(d["k"], d["y"]))
        if pool is not None:
            end_k = _truncate_k(pool, K_VALUES)
            pairs = [(k, y) for k, y in pairs if k <= end_k]
        if not pairs:
            return
        ks, ys = zip(*pairs)

        if faded:
            ax.plot(
                ks, ys,
                color=s.color, marker=s.marker,
                linestyle=(0, (3, 3)), linewidth=1.0, markersize=4.0,
                alpha=0.55,
                label=f"{s.display_short}{label_suffix}",
            )
        else:
            ax.plot(
                ks, ys,
                color=s.color, marker=s.marker, linestyle=s.linestyle,
                linewidth=1.6, markersize=5.5, markeredgewidth=0.6,
                label=s.display_short,
            )

        if pool is not None:
            # Pool annotation only on the canonical T4.2 line and on the
            # references; the v1 ReAct lines stack near each other and an
            # extra annotation each would clutter -- keep just the v2.
            if key not in ("react_hybrid_v1", "react_bm25_v1"):
                ax.annotate(
                    f"pool = {pool}",
                    xy=(ks[-1], ys[-1]),
                    xytext=(8, 0), textcoords="offset points",
                    fontsize=7, color=s.color,
                    alpha=0.7 if faded else 1.0,
                    va="center", ha="left",
                )

    # References first (drawn behind main lines).
    _plot("hybrid_rrf_k60",   faded=True, label_suffix=" (T3-A, ref.)")
    _plot("llm_rerank_top20", faded=True, label_suffix=" (T4.0, matched-pool)")
    _plot("crag_hybrid_v2",   faded=True, label_suffix=" (T4.1, agentic peer)")

    # T4.2 systems on top.
    for key in REACT_KEYS:
        _plot(key, faded=False)

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

    fig.subplots_adjust(left=0.08, right=0.60, top=0.95, bottom=0.13)
    paths = save_figure(fig, OUTPUT_NAME)
    plt.close(fig)
    for p in paths:
        print(f"[done] Wrote {p.relative_to(REPORT_DIR.parent)}")


if __name__ == "__main__":
    main()
