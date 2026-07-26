"""Build the Arm-2C navigated-vs-padded figure (Simple vs CNR, grouped per PDF).

Output: Report/figures/fig_rq2_arm2c_navigated_vs_padded.{pdf,png}

Single axes, figure title = Arm-2C. Per document (+ corpus) two horizontal bars
-- Simple navigation on top (light hatched), CNR below (solid teal) -- each =
padded recall@100 (the candidate pool) with a tick (|) at the final Recall@10
(value labelled). The before/after thus sits in one frame per document.

  Simple navigation = shipped Arm-2C (single descent, 8B re-rank).
  CNR               = Corrective Navigation and Re-Ranking (corrective loop +
                      e5 re-rank). Both from the controlled E2 output, so the
                      contrast is confound-free.

Reading: per document, CNR's bar (candidate pool) extends past Simple's and its
R@10 tick sits to the right -- the corrective loop surfaces more gold into the
pool and the e5 re-rank converts more of it into the top-10 (corpus R@10 .27->.38,
padded .46->.58).

Run via the RQ2 project venv:
  .venv/Scripts/python Report/scripts/build_arm2c_navigated_vs_padded.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

REPORT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import load_arm2c as la  # noqa: E402
from thesis_style import ARMS, PALETTE, rcparams_report, save_figure  # noqa: E402

OUTPUT_NAME = "fig_rq2_arm2c_navigated_vs_padded"
TEAL = ARMS["arm2c"].color


def main() -> None:
    d = la.arm2c_e2_reach_padding()           # large-first PDFs, corpus last
    labels = [f"{s} ($n={int(n)}$)" for s, n in zip(d["short"], d["n"])]
    y = np.arange(len(d))
    off = 0.21
    h = 0.36

    rcparams_report()
    fig, ax = plt.subplots(figsize=(6.8, 4.0))

    # Simple navigation (TOP sub-bar of each group; y-off sits higher once the
    # y-axis is inverted): light hatched.
    ax.barh(y - off, d["pad100_single"], height=h, color=TEAL, alpha=0.32,
            hatch="///", edgecolor="white", zorder=2)
    ax.scatter(d["r10_single"], y - off, color=PALETTE["black"], s=78, zorder=5,
               marker="|", linewidth=1.5)
    # CNR (BOTTOM sub-bar): solid.
    ax.barh(y + off, d["pad100_cnr"], height=h, color=TEAL, alpha=0.92,
            edgecolor="white", zorder=2)
    ax.scatter(d["r10_cnr"], y + off, color=PALETTE["black"], s=78, zorder=5,
               marker="|", linewidth=1.5)

    # R@10 value text just past each tick.
    for yy, v in zip(y - off, d["r10_single"]):
        ax.annotate(f"{v:.2f}".lstrip("0"), (v, yy), fontsize=5.6, color="#333333",
                    ha="center", va="bottom", xytext=(0, 4), textcoords="offset points")
    for yy, v in zip(y + off, d["r10_cnr"]):
        ax.annotate(f"{v:.2f}".lstrip("0"), (v, yy), fontsize=5.6, color="#333333",
                    ha="center", va="bottom", xytext=(0, 4), textcoords="offset points")

    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Article recall")
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    handles = [
        Patch(facecolor=TEAL, alpha=0.32, hatch="///", edgecolor="white",
              label="Simple nav. (padded@100)"),
        Patch(facecolor=TEAL, alpha=0.92, edgecolor="white",
              label="CNR (padded@100)"),
        Line2D([0], [0], color=PALETTE["black"], marker="|", linestyle="None",
               markersize=9, markeredgewidth=1.5, label="Recall@10"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.10),
              ncol=3, fontsize=7.5, frameon=False, columnspacing=1.3,
              handlelength=1.4, handletextpad=0.4)
    fig.suptitle("Arm-2C", fontsize=11, fontweight="bold", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.92))

    for p in save_figure(fig, OUTPUT_NAME):
        print(f"[done] Wrote {p.relative_to(REPORT_DIR.parent)}")
    plt.close(fig)


if __name__ == "__main__":
    main()
