"""Arm 2A recall vs retrieval-pool depth, split by GT cardinality.

Output: Report/figures/fig_rq2_arm2a_recall_curve_cardinality.{pdf,png}

Replaces the Arm-2A strata small-multiples. Single panel, recall@k as a curve
over the retrieval pool depth k in {5, 10, 20, 100} (log x), one line per GT
cardinality level (single- vs multi-gold-article questions), canonical
summary_node variant. Shows how the single/multi recall gap evolves as the pool
widens.

Reads the persisted Arm-2A per-question table (one-path; never re-evaluates).

Usage:
  .venv/Scripts/python Report/scripts/build_arm2a_recall_curve_cardinality.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPORT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import thesis_style as ts
from thesis_style import rcparams_report, save_figure

OUTPUT_NAME = "fig_rq2_arm2a_recall_curve_cardinality"
PQ = ts.T06_DATA_DIR / "tables" / "arm2a_per_question.csv"
VARIANT = "T04_summary_node_hybrid"
KS = [5, 10, 20, 100]
# (is_multi value, label, colour, marker, linestyle)
SERIES = [
    (False, "single", ts.PALETTE["orange"],   "o", "-"),
    (True,  "multi",  ts.PALETTE["grey_mid"], "s", "--"),
]


def main() -> None:
    df = pd.read_csv(PQ)
    df = df[df["variant"] == VARIANT]

    rcparams_report()
    fig, ax = plt.subplots(figsize=ts.FIGSIZE["report_wide"])
    for is_multi, label, color, marker, ls in SERIES:
        g = df[df["is_multi"] == is_multi]
        ys = [g[f"recall@{k}"].mean() for k in KS]
        ax.plot(KS, ys, color=color, marker=marker, ls=ls, lw=1.8, ms=6,
                label=f"{label} ($n={len(g)}$)")

    ax.set_xscale("log")
    ax.set_xticks(KS)
    ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
    ax.set_xlim(KS[0] * 0.9, KS[-1] * 1.1)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel(r"Retrieval pool depth $k$ (log scale)")
    ax.set_ylabel(r"Recall@$k$ (binary, full GT)")
    ax.grid(color="#DDDDDD", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9, frameon=False, title="GT cardinality")

    fig.tight_layout()
    paths = save_figure(fig, OUTPUT_NAME)
    plt.close(fig)
    for p in paths:
        print(f"[done] Wrote {p.relative_to(REPORT_DIR.parent)}")


if __name__ == "__main__":
    main()
