"""Build the Arm-2C E2 'ladder' figure — both levers compound.

Output: Report/figures/fig_rq2_arm2c_e2_ladder.{pdf,png}

Four R@10 configs off one corrective navigation per query, as an ascending ladder:

  8B/single (shipped 2C) -> 8B/corrective -> e5/single -> e5/corrective (synthesis)

The story in the gaps: corrective navigation raises reach but the 8B ranker barely
converts it (bar 1->2 flat); the e5 ranker converts (bar 1->3); and the corrective
reach stacks on top of the e5 ranker (bar 3->4). Dashed lines mark the embedding
(Arm-2A) and naive (Arm-1) full-set Civil R@10 the synthesis is closing toward.

Run via the RQ2 project venv:
  .venv/Scripts/python Report/scripts/build_arm2c_e2_ladder.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPORT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import load_arm2c as la  # noqa: E402
from thesis_style import ARMS, PALETTE, rcparams_report, save_figure  # noqa: E402

OUTPUT_NAME = "fig_rq2_arm2c_e2_ladder"
TEAL = ARMS["arm2c"].color
GREY = PALETTE["grey_light"]
GREY_D = PALETTE["grey_mid"]
CIVIL_STEM = "1804_03_21_1804032150"


def main() -> None:
    s = la.arm2c_e2_summary()
    n = int(s.get("n", 0))
    bars = [
        ("8B · single\n(shipped 2C)", s["r10_8B_single"], GREY),
        ("8B · corrective",           s["r10_8B_corr"],   GREY),
        ("e5 · single",               s["r10_e5_single"], TEAL),
        ("e5 · corrective\n(synthesis)", s["r10_e5_corr"], TEAL),
    ]
    # reference lines: full-set Civil Arm-1 / Arm-2A from the T06 one-path frame.
    cross = la.cross_arm_recall10().set_index("stem")
    arm1 = float(cross.loc[CIVIL_STEM, "arm1"])
    arm2a = float(cross.loc[CIVIL_STEM, "arm2a"])

    rcparams_report()
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    x = np.arange(len(bars))
    for i, (_, v, c) in enumerate(bars):
        ax.bar(i, v, 0.62, color=c, edgecolor=(TEAL if c == TEAL else GREY_D),
               linewidth=(1.2 if i == 3 else 0), zorder=3)
        ax.text(i, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=8.5,
                fontweight=("bold" if i == 3 else "normal"))

    # compounding deltas, stated as a caption (clearer than arrows fighting the lines)
    d_rank = bars[2][1] - bars[0][1]
    d_reach = bars[3][1] - bars[2][1]
    d_tot = bars[3][1] - bars[0][1]
    ax.text(1.85, arm2a + 0.035,
            f"ranking +{d_rank:.2f}  ·  reach +{d_reach:.2f}  →  +{d_tot:.2f} vs shipped 2C",
            ha="center", va="bottom", fontsize=8.5, color=TEAL, fontweight="bold")

    # reference lines for the arms it is closing toward (labels top-LEFT, clear of bars)
    for val, lab, col in [(arm2a, f"Arm 2A (embedding) {arm2a:.2f}", ARMS["arm2a"].color),
                          (arm1, f"Arm 1 (naive) {arm1:.2f}", ARMS["arm1"].color)]:
        ax.axhline(val, ls="--", lw=1.0, color=col, zorder=1)
        ax.text(-0.42, val + 0.006, lab, ha="left", va="bottom", fontsize=7, color=col)

    ax.set_xticks(x)
    ax.set_xticklabels([b[0] for b in bars])
    ax.set_ylabel("Recall@10 (question-weighted)")
    ax.set_ylim(0, max(arm2a, bars[-1][1]) + 0.16)
    ax.grid(axis="y", color="#EEEEEE", lw=0.8, zorder=0)
    ax.set_axisbelow(True)

    fig.suptitle("Both levers compound: navigation reach + embedding ranking",
                 fontsize=10.5, y=0.99)
    ax.set_title(f"Code Civil — {n} questions   ·   "
                 f"reach {s['reach_single']:.2f}→{s['reach_corr']:.2f}",
                 fontsize=8, color=GREY_D, pad=6)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for p in save_figure(fig, OUTPUT_NAME):
        print(f"[done] Wrote {p.relative_to(REPORT_DIR.parent)}")
    plt.close(fig)


if __name__ == "__main__":
    main()
