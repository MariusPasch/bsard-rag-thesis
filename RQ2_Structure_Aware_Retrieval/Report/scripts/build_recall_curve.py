"""Build the RQ2 cross-arm Recall@k curve over retrieval depth.

Output: Report/figures/fig_rq2_headline_recall_curve.{pdf,png}

Recall@k vs k (log-x) for the three canonical arms, markers at the measured
cutoffs k in {5, 10, 20, 100}. 95% paired-bootstrap CI whiskers (seed 42,
2000 resamples, pooled over the 725 questions) are drawn at k = 10 and k = 100 --
the two depths the per-question recall file records. A vertical dotted line at
k = 100 marks the Arm-2 first-stage pool ceiling.

Reading (STYLEGUIDE §5 pool caveat): Arm 1 and Arm 2A track together through
k = 20, then Arm 1 pulls ahead at k = 100 -- a pool-depth effect (Arm 1 pools
200 candidates, Arm 2 only 100, so Arm 1's top-100 still gains relevants where
Arm 2 has exhausted its pool). Arm 2B trails at every depth.

Usage:
  .venv/Scripts/python Report/scripts/build_recall_curve.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPORT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import load_arm2c as la
import load_results as lr
from thesis_style import (
    POOL_DEPTHS,
    get_variant,
    rcparams_report,
    save_figure,
)

OUTPUT_NAME = "fig_rq2_headline_recall_curve"
ARMS = ["T03_arm1_naive", "T04_summary_node_hybrid", "T05_pageindex", "ARM2C_agentic"]
K_CURVE = [5, 10, 20, 100]            # measured cutoffs (long frame)
CI_K = [10, 100]                      # depths with per-question recall
N_BOOT = 2000
SEED = 42


def _bootstrap_ci(vals: np.ndarray, n: int = N_BOOT, seed: int = SEED):
    """Percentile 95% CI of the mean over a pooled per-question recall array."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(vals), size=(n, len(vals)))
    means = vals[idx].mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def main() -> None:
    long = lr.load_long(include_arm2c=True)
    pq = lr.load_per_query(include_arm2c=True)

    # Line point estimates: micro R@k over all PDFs, all four cutoffs.
    metric_names = [f"R@{k}" for k in K_CURVE]
    wide = lr.to_wide(long, metric_names, weighting="micro").set_index("method")

    rcparams_report()
    fig, ax = plt.subplots(figsize=(6.7, 3.8))

    for method in ARMS:
        v = get_variant(method)
        ys = [float(wide.loc[method, f"R@{k}"]) for k in K_CURVE]
        label = "Arm 2C (simple)" if method == "ARM2C_agentic" else v.display_short
        ax.plot(K_CURVE, ys, color=v.color, marker=v.marker, linestyle=v.linestyle,
                linewidth=1.6, markersize=6, markeredgewidth=0.6,
                label=label, zorder=3)
        # CI whiskers where per-question recall exists.
        for k in CI_K:
            vals = pq[(pq["method"] == method) & (pq["k"] == k)]["recall"].to_numpy(float)
            if len(vals) == 0:
                continue
            lo, hi = _bootstrap_ci(vals)
            y = float(wide.loc[method, f"R@{k}"])
            ax.errorbar([k], [y], yerr=[[y - lo], [hi - y]], fmt="none",
                        ecolor=v.color, elinewidth=1.2, capsize=2.5, zorder=2)

    # Arm-2C CNR: full curve (k=5/10/20/100) -- recovered by re-running the e5 rerank
    # over the saved corrective pools and capturing the ordered list (see
    # corrective/e2_metrics_full.py). Dashed + hollow squares to distinguish it from
    # the solid Arm-2C simple-nav curve (same teal series).
    cnr = la.arm2c_cnr_metrics()
    teal = get_variant("ARM2C_agentic").color
    ax.plot(K_CURVE, [cnr[f"R@{k}"] for k in K_CURVE], color=teal, marker="s",
            linestyle="--", linewidth=1.6, markersize=6, markeredgewidth=1.0,
            markerfacecolor="white", label="Arm 2C — CNR", zorder=4)

    # Arm-2 pool ceiling (Arm 1 pools 200, beyond the measured range).
    pool2 = POOL_DEPTHS["arm2a"]
    ax.axvline(pool2, color="#888888", linestyle=":", linewidth=1.0, zorder=1)
    ax.text(pool2, 0.02, f" Arm 2 pool ceiling ({pool2})\n (Arm 1 pool {POOL_DEPTHS['arm1']})",
            rotation=90, va="bottom", ha="right", fontsize=7.5, color="#666666")

    ax.set_xscale("log")
    ax.set_xticks(K_CURVE, [str(k) for k in K_CURVE])
    ax.minorticks_off()
    ax.set_xlabel(r"Retrieval depth / pool size $k$ (log scale)")
    ax.set_ylabel(r"Recall@$k$ (micro, 5 PDFs, $n=725$)")
    ax.set_xlim(K_CURVE[0] * 0.9, K_CURVE[-1] * 1.15)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9, frameon=False,
              handlelength=2.4, handletextpad=0.6)

    fig.tight_layout()
    paths = save_figure(fig, OUTPUT_NAME)
    plt.close(fig)
    for p in paths:
        print(f"[done] Wrote {p.relative_to(REPORT_DIR.parent)}")


if __name__ == "__main__":
    main()
