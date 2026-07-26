"""
Build the Tier 4.1 CRAG Recall@k line plot.

Output: Report/figures/fig_rq1_t41_recall_curve.{pdf,png}

Shows four lines:
  - CRAG (hybrid, v2)         -- T4.1 canonical
  - CRAG (BM25, v2)           -- first-stage ablation
  - Hybrid RRF (k=60)         -- T3-A upstream reference (faded green dashed)
  - LLM-Judge top-20 (hyb.)   -- T4.0 matched-pool anchor (faded vermillion dash-dot)

Pool truncation.
  Every line that has a pool ceiling ends *one K_VALUES cutoff past* its pool
  so the plateau is visually confirmed.
    - T4.0 top-20: pool = 20 -> line ends at k = 50.
    - CRAG (hyb./BM25): pool = eval_k = 20 -> line ends at k = 50.
      CRAG technically returns up to backbone_top_k = 100 items, but
      positions 21-100 are unchanged first-stage ranking (CRAG's mechanism
      did not touch them). Treating eval_k as the pool reflects the
      *agentic* contribution honestly.
  The T3-A reference is first-stage retrieval (pool == 100) so it spans the
  full K_VALUES range.

The matched-pool comparison reads at k = 20 -- where CRAG (hyb.) sits
roughly on top of T4.0 top-20 (the agentic-vs-non-agentic null result of
Act 3).

Usage:
  .venv/Scripts/python Report/scripts/build_t41_recall_curve.py
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
OUTPUT_NAME = "fig_rq1_t41_recall_curve"

# Main series (full curve)
CRAG_KEYS = ["crag_hybrid_v2", "crag_bm25_v2"]
# References, drawn faded
T3_REF_KEY = "hybrid_rrf_k60"      # T3-A upstream
T40_REF_KEY = "llm_rerank_top20"   # T4.0 matched-pool anchor; truncated at k=20

# Pool ceilings. CRAG's pool is eval_k = 20 (the LLM-evaluator window) --
# positions past eval_k are unchanged first-stage ranking, so reporting them
# as "CRAG output" would overstate the agentic mechanism's reach.
POOL_SIZE = {
    "llm_rerank_top20": 20,
    "crag_hybrid_v2":   20,   # eval_k
    "crag_bm25_v2":     20,   # eval_k
}


def _truncate_k(pool: int, k_values: list[int]) -> int:
    """Largest k in K_VALUES we should plot for a system with this pool.

    Rule: include the smallest k_value strictly greater than `pool`, so the
    line ends one step past the pool ceiling (confirms the plateau visually).
    """
    larger = [k for k in k_values if k > pool]
    return min(larger) if larger else pool


def main() -> None:
    records = load_records()
    recs = [(t, r) for t, r in records if t in ("T3", "T4.0", "T4.1")]
    if not recs:
        raise SystemExit("No T3/T4.0/T4.1 result records found.")

    long = to_long(recs)
    rec = long[(long["metric"] == "Recall")
               & (long["stratum"] == "overall")
               & (long["k"].isin(K_VALUES))]

    series: dict[str, dict] = {}
    plot_keys = CRAG_KEYS + [T3_REF_KEY, T40_REF_KEY]
    for _, row in rec.iterrows():
        s = get_system(row["experiment_id"])
        if s is None or s.key not in plot_keys:
            continue
        d = series.setdefault(s.key, {"style": s, "k": [], "y": []})
        d["k"].append(int(row["k"]))
        d["y"].append(float(row["value"]))

    rcparams_report()
    fig, ax = plt.subplots(figsize=(7.6, 4.0))

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

        # Pool annotation for systems with a ceiling. Faded references use a
        # muted color (consistent with the line); CRAG uses its own color.
        if pool is not None:
            ax.annotate(
                f"pool = {pool}",
                xy=(ks[-1], ys[-1]),
                xytext=(8, 0), textcoords="offset points",
                fontsize=7, color=s.color,
                alpha=0.7 if faded else 1.0,
                va="center", ha="left",
            )

    # References first (sit behind main series).
    _plot(T3_REF_KEY,  faded=True, label_suffix=" (T3-A, reference)")
    _plot(T40_REF_KEY, faded=True, label_suffix=" (T4.0, matched-pool anchor)")
    # Main CRAG series.
    for key in CRAG_KEYS:
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

    fig.subplots_adjust(left=0.08, right=0.62, top=0.95, bottom=0.13)
    paths = save_figure(fig, OUTPUT_NAME)
    plt.close(fig)
    for p in paths:
        print(f"[done] Wrote {p.relative_to(REPORT_DIR.parent)}")


if __name__ == "__main__":
    main()
