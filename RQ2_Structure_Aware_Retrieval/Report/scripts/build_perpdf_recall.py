"""Build the cross-arm per-document Recall@10 bars (with the Arm-2C CNR result).

Output: Report/figures/fig_rq2_perpdf_recall.{pdf,png}

Per-document Recall@10 for all arms, grouped per PDF, plus a corpus-weighted
group. Five series: the three baselines (Naive / Metadata / PageIndex) and the
agentic arm shown as BOTH of its configs -- Arm-2C simple navigation (light teal,
the shipped single-pass + 8B re-rank) and Arm-2C CNR (solid teal, corrective
navigation + e5 re-rank). The point: CNR lifts the agentic arm from below
PageIndex-or-just-above to near the vector arms on the structurally hard codes
(and past Metadata on Code Pénal), while still trailing on the lexically trivial
Code du Logement.

Arms 1/2A/2B and Arm-2C simple come from the T06 single-re-eval long frame
(cross_arm_recall10); Arm-2C CNR from the controlled E2 run (arm2c_e2_all). Run
via the RQ2 project venv:
  .venv/Scripts/python Report/scripts/build_perpdf_recall.py
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

OUTPUT_NAME = "fig_rq2_perpdf_recall"
TEAL = ARMS["arm2c"].color

# (column, colour, alpha, legend label) -- the two 2C configs share teal (light
# = simple, solid = CNR) so they read as one arm, two configs.
SERIES = [
    ("arm1",         ARMS["arm1"].color,  1.00, ARMS["arm1"].display_short + " (Naive)"),
    ("arm2a",        ARMS["arm2a"].color, 1.00, ARMS["arm2a"].display_short + " (Metadata)"),
    ("arm2b",        ARMS["arm2b"].color, 1.00, ARMS["arm2b"].display_short + " (PageIndex)"),
    ("arm2c_simple", TEAL,                0.38, "Arm 2C — simple nav."),
    ("arm2c_cnr",    TEAL,                1.00, "Arm 2C — CNR"),
]


def main() -> None:
    df = la.cross_arm_recall10().rename(columns={"arm2c": "arm2c_simple"})
    ea = la.arm2c_e2_all().set_index("stem")
    df["arm2c_cnr"] = df["stem"].map(lambda s: float(ea.loc[s, "r10_e5_corr"]))

    groups = [s.replace("Code ", "Code\n").replace("Corpus", "Corpus\n(n=725)")
              for s in df["short"]]

    rcparams_report()
    fig, ax = plt.subplots(figsize=(7.8, 3.8))

    x = np.arange(len(df))
    n_arms = len(SERIES)
    width = 0.84 / n_arms

    for i, (col, color, alpha, label) in enumerate(SERIES):
        offset = (i - (n_arms - 1) / 2) * width
        vals = df[col].to_numpy(dtype=float)
        bars = ax.bar(x + offset, vals, width, label=label, color=color,
                      alpha=alpha, edgecolor="white", linewidth=0.3)
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.2f}".lstrip("0"),
                        (b.get_x() + b.get_width() / 2, v),
                        ha="center", va="bottom", fontsize=4.2, color="#333333",
                        xytext=(0, 1), textcoords="offset points")

    # divider before the corpus-weighted group.
    ax.axvline(len(df) - 1.5, color="#CCCCCC", linewidth=0.8, linestyle=(0, (3, 3)))

    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel("Recall@10")
    ax.set_ylim(0, 0.9)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.16),
              columnspacing=1.2, handlelength=1.3, fontsize=8)
    fig.tight_layout()

    for p in save_figure(fig, OUTPUT_NAME):
        print(f"[done] Wrote {p.relative_to(REPORT_DIR.parent)}")
    plt.close(fig)


if __name__ == "__main__":
    main()
