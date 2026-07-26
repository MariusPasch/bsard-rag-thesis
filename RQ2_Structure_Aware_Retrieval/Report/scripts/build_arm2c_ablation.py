"""Build the Arm-2C component ablation as a Recall@k curve (Code Civil).

Output: Report/figures/fig_rq2_arm2c_ablation.{pdf,png}

Recall@k vs retrieval depth k (log-x, k in {5, 10, 20, 100}) for the three
cumulative ablation steps on the Code Civil -- deep tree alone (labels only) ->
+ native metadata -> + 8B re-rank (= the simple-navigation Arm-2C, the shipped
endpoint that §4e's CNR then builds on) -- with the Arm-2B (PageIndex) curve on
the same document as the reference the approach refines.

Reading: metadata lifts the whole curve over PageIndex; the re-rank step is a
*shape* change, not a *level* change -- it lifts early recall (R@5/R@10) toward
the pool that navigation already reached (R@100 ~unchanged at ~0.52), i.e. it
re-orders the reached pool rather than reaching more.

Run via the RQ2 project venv:
  .venv/Scripts/python Report/scripts/build_arm2c_ablation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

REPORT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import load_arm2c as la  # noqa: E402
import load_results as lr  # noqa: E402
from thesis_style import (  # noqa: E402
    ARMS, PALETTE, get_variant, rcparams_report, save_figure,
)

OUTPUT_NAME = "fig_rq2_arm2c_ablation"
K_CURVE = la.CURVE_K                       # [5, 10, 20, 100]

# cumulative ablation steps -> (colour, marker) ; teal = the simple-navigation
# Arm-2C config (the shipped endpoint that §4e's CNR then builds on).
STEP_STYLE = {
    "bare":            (PALETTE["grey_mid"], "o", "Deep tree (labels only)"),
    "enriched":        (PALETTE["orange"],   "s", "+ metadata (enriched)"),
    "enriched_rerank": (ARMS["arm2c"].color, "D", "+ 8B re-rank (Simple-nav Arm-2C)"),
}


def _arm2b_civil_curve() -> dict[int, float]:
    """Arm-2B (PageIndex) Recall@k on the Code Civil, from the T06 long frame."""
    long = lr.load_long()
    ps = lr.per_stem_wide(long, [f"R@{k}" for k in K_CURVE])
    row = ps[(ps["stem"] == la.ABLATION_STEM) & (ps["method"] == "T05_pageindex")].iloc[0]
    return {k: float(row[f"R@{k}"]) for k in K_CURVE}


def main() -> None:
    curve = la.arm2c_ablation_curve()
    ref = _arm2b_civil_curve()

    rcparams_report()
    fig, ax = plt.subplots(figsize=(5.0, 3.8))

    # Arm-2B reference (the baseline the approach refines).
    v2b = get_variant("T05_pageindex")
    ax.plot(K_CURVE, [ref[k] for k in K_CURVE], color=v2b.color, marker="d",
            linestyle=(0, (4, 3)), linewidth=1.3, markersize=5, alpha=0.9,
            label="Arm 2B (PageIndex)", zorder=2)

    # the three ablation steps.
    for reg in ("bare", "enriched", "enriched_rerank"):
        color, marker, label = STEP_STYLE[reg]
        ys = [curve[reg][f"R@{k}"] for k in K_CURVE]
        ax.plot(K_CURVE, ys, color=color, marker=marker, linestyle="-",
                linewidth=1.6, markersize=6, markeredgewidth=0.6, label=label,
                zorder=3)

    ax.set_xscale("log")
    ax.set_xticks(K_CURVE, [str(k) for k in K_CURVE])
    ax.minorticks_off()
    ax.set_xlabel(r"Retrieval depth / pool size $k$ (log scale)")
    ax.set_ylabel(r"Recall@$k$ (Code Civil, $n=252$)")
    ax.set_xlim(K_CURVE[0] * 0.9, K_CURVE[-1] * 1.15)
    ax.set_ylim(0, 0.6)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=8, frameon=False,
              handlelength=2.2, handletextpad=0.6)
    fig.tight_layout()

    for p in save_figure(fig, OUTPUT_NAME):
        print(f"[done] Wrote {p.relative_to(REPORT_DIR.parent)}")
    plt.close(fig)


if __name__ == "__main__":
    main()
