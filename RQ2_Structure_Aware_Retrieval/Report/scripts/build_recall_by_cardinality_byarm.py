"""Cross-arm recall@10 split by GT cardinality (single- vs multi-gold questions).

Output: Report/figures/fig_rq2_recall_by_cardinality_byarm.{pdf,png}

Replaces the Arm-1-only strata small-multiples. Single panel, grouped bars:
two clusters (single / multi GT) with one bar per canonical arm. Shows that
every arm loses recall on multi-gold questions, and by how much each.

Reads the persisted per-arm strata summaries (one-path; never re-evaluates).
Arm 2A is the canonical summary_node variant.

Usage:
  .venv/Scripts/python Report/scripts/build_recall_by_cardinality_byarm.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPORT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import load_arm2c as la
import thesis_style as ts
from thesis_style import get_variant, rcparams_report, save_figure

OUTPUT_NAME = "fig_rq2_recall_by_cardinality_byarm"
TABLES = ts.T06_DATA_DIR / "tables"
METRIC = "recall@10"
LEVELS = ["single", "multi"]

# (strata csv, row filter, key, display label, colour, alpha) per arm, L-to-R.
# The two agentic configs (simple nav / CNR) share teal: light = simple, solid =
# CNR. Arm 2C carries no T06 strata CSV (run outside the consolidation): its
# cardinality split is computed from the runs by load_arm2c.
_TEAL = get_variant(la.ARM2C_METHOD).color
ARMS = [
    ("arm1_strata_summary.csv",  None, "T03_arm1_naive",          None, None, 1.0),
    ("arm2a_strata_summary.csv", ("variant", "T04_summary_node_hybrid"),
     "T04_summary_node_hybrid", None, None, 1.0),
    ("arm2b_strata_summary.csv", None, "T05_pageindex",           None, None, 1.0),
    (None, None, la.ARM2C_METHOD, "Arm 2C (simple)", _TEAL, 0.38),
    ("CNR", None, "ARM2C_CNR",    "Arm 2C (CNR)",    _TEAL, 1.0),
]


def _card_rows(fname: str, filt) -> dict:
    if fname == "CNR":                      # Arm 2C CNR: e5 over the corrective pool
        return la.arm2c_cardinality_cnr()
    if fname is None:                       # Arm 2C simple: from the experiment runs
        return la.arm2c_cardinality()
    df = pd.read_csv(TABLES / fname)
    if filt is not None:
        df = df[df[filt[0]] == filt[1]]
    df = df[df["stratum"] == "cardinality"]
    return {r["level"]: (float(r[METRIC]), int(r["n_q"])) for _, r in df.iterrows()}


def main() -> None:
    data = {key: _card_rows(f, filt) for f, filt, key, *_ in ARMS}
    # n per level is identical across arms (same questions) -> read from Arm 1.
    n_by_level = {lv: data["T03_arm1_naive"][lv][1] for lv in LEVELS}

    rcparams_report()
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    x = np.arange(len(LEVELS))
    n_arms = len(ARMS)
    width = 0.84 / n_arms

    for j, (_f, _filt, key, lab_o, col_o, alpha) in enumerate(ARMS):
        color = col_o or get_variant(key).color
        label = lab_o or get_variant(key).display_short
        offset = (j - (n_arms - 1) / 2) * width
        vals = [data[key].get(lv, (np.nan, 0))[0] for lv in LEVELS]
        bx = x + offset
        ax.bar(bx, vals, width=width, color=color, alpha=alpha, edgecolor="black",
               linewidth=0.4, label=label)
        for xx, yv in zip(bx, vals):
            if yv == yv:
                ax.text(xx, yv + 0.012, f"{yv:.2f}", ha="center", va="bottom",
                        fontsize=6.8, color="#333333")

    ax.set_xticks(x, [f"{lv}\n($n={n_by_level[lv]}$)" for lv in LEVELS])
    ax.set_xlabel("GT cardinality (gold articles per question)")
    ax.set_ylabel(r"Recall@10 (binary, full GT)")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", fontsize=8, frameon=False,
              handlelength=1.4, handletextpad=0.5)

    fig.tight_layout()
    paths = save_figure(fig, OUTPUT_NAME)
    plt.close(fig)
    for p in paths:
        print(f"[done] Wrote {p.relative_to(REPORT_DIR.parent)}")


if __name__ == "__main__":
    main()
