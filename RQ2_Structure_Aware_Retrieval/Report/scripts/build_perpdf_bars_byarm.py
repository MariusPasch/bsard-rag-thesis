"""Build the RQ2 per-document bar chart, split into one panel per arm.

Output: Report/figures/fig_rq2_perpdf_bars_byarm.{pdf,png}

Companion to build_perpdf_bars.py. Instead of one panel with the three arms
grouped per PDF, this gives each arm its own panel, and within each panel shows
Effective Recall at two retrieval depths -- @10 and @100 -- per PDF. This makes
the depth gain visible: how much recall each arm recovers when the pool widens
from 10 to 100.

Depth note: the T06 evaluation measured {5, 10, 20, 100} (thesis_style
RECALL_K_CURVE). @50 was never a cutoff, so it cannot be shown without a
pipeline re-eval; @10 and @100 are both present for every (PDF, arm).

Usage:
  .venv/Scripts/python Report/scripts/build_perpdf_bars_byarm.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

REPORT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import load_arm2c as la
import load_results as lr
from thesis_style import get_variant, rcparams_report, save_figure

OUTPUT_NAME = "fig_rq2_perpdf_bars_byarm"
METRICS = ["E/R@10", "E/R@100"]
# Arms, one panel each (left-to-right). Arm 2C's E/ cells are its binary recall
# (its gold is already reachable-clipped, so binary recall is drift-aware). The
# agentic arm is split into its two configs: simple navigation and CNR.
ARMS = ["T03_arm1_naive", "T04_summary_node_hybrid", "T05_pageindex",
        "ARM2C_agentic", "ARM2C_CNR"]
# Title / colour overrides for the synthetic agentic-config panels.
TITLE = {"ARM2C_agentic": "Arm 2C (simple)", "ARM2C_CNR": "Arm 2C (CNR)"}


def _lighten(hex_color: str, amount: float = 0.55) -> tuple[float, float, float]:
    """Blend a hex colour toward white by ``amount`` (0=unchanged, 1=white)."""
    h = hex_color.lstrip("#")
    rgb = tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    return tuple(c + (1.0 - c) * amount for c in rgb)


def main() -> None:
    long = lr.load_long(include_arm2c=True)
    ps = lr.per_stem_wide(long, METRICS)
    ps = ps[ps["method"].isin(ARMS)]

    # PDFs ordered by GT-question count, largest first (shared across panels).
    order = (ps[["stem", "stem_label", "n_gt_q"]].drop_duplicates()
             .sort_values("n_gt_q", ascending=False))
    stems = order["stem"].tolist()
    labels = [f"{lbl}\n($n={int(n)}$)"
              for lbl, n in zip(order["stem_label"], order["n_gt_q"])]

    val = {(r["stem"], r["method"], m): float(r[m])
           for _, r in ps.iterrows() for m in METRICS}

    # CNR per-stem E/R@10 (e5_corr) and E/R@100 (corrective pool@100).
    ea = la.arm2c_e2_all().set_index("stem")
    rp = la.arm2c_e2_reach_padding().set_index("stem")
    for s in stems:
        val[(s, "ARM2C_CNR", "E/R@10")] = float(ea.loc[s, "r10_e5_corr"])
        val[(s, "ARM2C_CNR", "E/R@100")] = float(rp.loc[s, "pad100_cnr"])

    rcparams_report()
    fig, axes = plt.subplots(1, len(ARMS), figsize=(14.8, 4.6), sharey=True)

    n_pdfs = len(stems)
    ys = np.arange(n_pdfs)
    bar_h = 0.8 / len(METRICS)

    for ax, arm in zip(axes, ARMS):
        base = (get_variant("ARM2C_agentic").color if arm == "ARM2C_CNR"
                else get_variant(arm).color)
        title = TITLE.get(arm) or get_variant(arm).display_short
        depth_colors = [base, _lighten(base, 0.55)]
        for k, (metric, color) in enumerate(zip(METRICS, depth_colors)):
            offset = (k - (len(METRICS) - 1) / 2) * bar_h  # @10 on top (y inverted)
            bar_y = ys + offset
            vals = [val.get((s, arm, metric), float("nan")) for s in stems]
            ax.barh(bar_y, vals, height=bar_h, color=color,
                    edgecolor="black", linewidth=0.4,
                    label=metric.replace("E/R@", "@"))
            for yy, xv in zip(bar_y, vals):
                if xv == xv:
                    ax.text(xv + 0.012, yy, f"{xv:.2f}", va="center", ha="left",
                            fontsize=6.5, color="#333333")
        ax.set_title(title, fontsize=10)
        ax.set_xlim(0, 1.0)
        ax.grid(axis="x", color="#DDDDDD", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)

    axes[0].set_yticks(ys, labels)
    axes[0].invert_yaxis()  # largest PDF on top

    # One shared legend: lightness encodes depth identically in every panel, so
    # neutral grey swatches read across all three (darker = @10, lighter = @100).
    handles = [Patch(facecolor="#555555", edgecolor="black", linewidth=0.4, label="@10"),
               Patch(facecolor=_lighten("#555555", 0.55), edgecolor="black",
                     linewidth=0.4, label="@100")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               title="retrieval depth", fontsize=9, title_fontsize=9,
               bbox_to_anchor=(0.5, 0.0), handlelength=1.2, handletextpad=0.4,
               columnspacing=1.4)
    fig.supxlabel(r"Effective Recall (E/R) per document", y=0.085)

    fig.tight_layout(rect=(0, 0.10, 1, 1))
    paths = save_figure(fig, OUTPUT_NAME)
    plt.close(fig)
    for p in paths:
        print(f"[done] Wrote {p.relative_to(REPORT_DIR.parent)}")


if __name__ == "__main__":
    main()
