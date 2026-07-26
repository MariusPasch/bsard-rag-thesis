"""Padding-free version of fig_rq2_arm2c_why: a pure 2B-vs-2C navigator comparison.

Output: Report/figures/fig_rq2_arm2c_why_nopadding.{pdf,png}

Padding removed, three series per PDF (PageIndex / Arm-2C simple nav / Arm-2C
CNR), isolating the two things the navigators actually do -- descend the tree and
commit to a top-10 -- on equal terms:

  LEFT  -- Navigation reach: gold the descent surfaced before padding.
           2C simple reached_recall and 2C CNR reach_corr (section expanded) vs
           2B nav_recall (navigated candidate set).
  RIGHT -- Selection / ranking head: gold committed to the top-10 pre-padding.
           2C simple selected_recall (the agent's selected_bsard_ids) and 2C CNR
           e5 top-10 (r10_e5_corr; CNR has no agent-selection step -- its e5
           ranking over the reached pool IS the commitment) vs 2B exposed_recall
           (the LLM's score>0 picks).

The story across both levers: the rebuilt deep tree *reaches* far more gold than
PageIndex (left, both 2C variants), and CNR reaches the most. On the ranking head
(right) the *simple* agent under-commits -- it selects even less than PageIndex's
exposed picks -- but CNR's e5 ranking recovers and clears both: the selection
collapse, and its fix, in one frame. (Question-weighted; Arm-2B from its deep-dive
per-PDF summary; CNR from the controlled E2 run.)

Run via the RQ2 project venv:
  .venv/Scripts/python Report/scripts/build_arm2c_why_nopadding.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPORT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import load_arm2c as la  # noqa: E402
from thesis_style import ARMS, rcparams_report, save_figure  # noqa: E402

OUTPUT_NAME = "fig_rq2_arm2c_why_nopadding"
C2B, C2C = ARMS["arm2b"].color, ARMS["arm2c"].color

# (offset, colour, alpha, hatch, label) -- top to bottom within each group.
SERIES = [
    ("b",  -0.27, C2B, 0.95, None,  "Arm 2B (PageIndex)"),
    ("c",   0.0,  C2C, 0.34, "///", "Arm 2C — simple nav."),
    ("cc",  0.27, C2C, 0.95, None,  "Arm 2C — CNR"),
]


def _panel(ax, y, df, reach_or_select: str, title, xmax):
    h = 0.24
    col = {"b": f"b_{reach_or_select}", "c": f"c_{reach_or_select}",
           "cc": f"cc_{reach_or_select}"}
    for key, off, color, alpha, hatch, _label in SERIES:
        vals = df[col[key]].to_numpy(float)
        ax.barh(y + off, vals, height=h, color=color, alpha=alpha, hatch=hatch,
                edgecolor="white", linewidth=0.4, zorder=3)
        for yy, v in zip(y + off, vals):
            ax.text(v + xmax * 0.012, yy, f"{v:.2f}".lstrip("0"), va="center",
                    ha="left", fontsize=6.2, color="#333333")
    ax.set_xlim(0, xmax)
    ax.set_title(title, fontsize=9.5)
    ax.grid(axis="x", color="#EEEEEE", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)


def main() -> None:
    df = la.pure_comparison()
    y = np.arange(len(df))

    rcparams_report()
    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(10.2, 4.2), sharey=True,
        gridspec_kw={"wspace": 0.06})

    _panel(axA, y, df, "reach",
           "Navigation reach\n(gold the descent surfaced, pre-padding)", 0.72)
    _panel(axB, y, df, "select",
           "Selection / ranking head\n(gold in the committed picks, pre-padding)", 0.55)

    axA.set_yticks(y)
    axA.set_yticklabels([f"{s}  ($n={int(n)}$)"
                         for s, n in zip(df["short"], df["n"])], fontsize=8.5)
    axA.invert_yaxis()

    from matplotlib.patches import Patch
    handles = [Patch(facecolor=c, alpha=a, hatch=hc, edgecolor="white", label=lab)
               for _k, _o, c, a, hc, lab in SERIES]
    fig.legend(handles, [h.get_label() for h in handles],
               loc="lower center", ncol=3, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.02), handlelength=1.3, columnspacing=1.8)
    fig.suptitle("Arm-2C, padding removed: navigation reach vs ranking head",
                 fontsize=11, y=1.0)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))

    for p in save_figure(fig, OUTPUT_NAME):
        print(f"[done] Wrote {p.relative_to(REPORT_DIR.parent)}")
    plt.close(fig)


if __name__ == "__main__":
    main()
