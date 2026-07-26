"""Build the Arm-2C miss decomposition, BEFORE vs AFTER (the two CNR levers).

Output: Report/figures/fig_rq2_arm2c_miss_anatomy.{pdf,png}

Two panels sharing the document axis, every gold article classified into one of
three fates (gold-article-weighted -- the x-axis is 'share of gold articles'):
  - HIT                 : retrieved in the final top-10.
  - Reached, not top-10 : the agent's descent expanded the section that holds it,
                          but the re-ranker left it out of the top-10
                          (a ranking problem -- the gold WAS reachable).
  - Not reached         : the branch was never expanded, so the article was never
                          seen (the navigation ceiling; orphan catch-all folded
                          in here -- e2 does not split it out).

LEFT  = Simple navigation  = shipped Arm-2C (8B re-rank, single navigation pass).
RIGHT = CNR                = Corrective Navigation and Re-Ranking (e5 re-rank over
                             the corrective-loop reached pool).
Both panels come from the SAME controlled E2 per-query output, so the before/after
is confound-free. The point: corrective navigation shrinks 'Not reached' (the
navigation lever) and the e5 re-ranker converts reach into HIT (the ranking
lever) -- the two §4e levers made visible on the coverage axis.

Run via the RQ2 project venv:
  .venv/Scripts/python Report/scripts/build_arm2c_miss_anatomy.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPORT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import load_arm2c as la  # noqa: E402
from thesis_style import PALETTE, rcparams_report, save_figure  # noqa: E402

OUTPUT_NAME = "fig_rq2_arm2c_miss_anatomy"

# (legend label, colour). Order = stack order, left to right.
CLASSES = [
    ("HIT (top-10)",          PALETTE["green"]),
    ("Reached, not top-10",   PALETTE["orange"]),
    ("Not reached",           PALETTE["vermillion"]),
]

# (panel title, hit column, reached column)
PANELS = [
    ("Simple navigation (Arm-2C)", "hit_single", "reached_single"),
    ("CNR (corrective nav + re-rank)", "hit_cnr", "reached_cnr"),
]


def _stack(ax: plt.Axes, df, hit_col: str, reached_col: str, y) -> None:
    hit = df[hit_col].to_numpy(dtype=float)
    reached = df[reached_col].to_numpy(dtype=float)
    parts = [hit, reached - hit, 1.0 - reached]  # HIT / reached-miss / not-reached
    left = np.zeros(len(df))
    for (label, color), frac in zip(CLASSES, parts):
        ax.barh(y, frac, left=left, height=0.66, label=label,
                color=color, edgecolor="white", linewidth=0.4)
        for yy, f, l in zip(y, frac, left):
            if f >= 0.08:
                ax.text(l + f / 2, yy, f"{100 * f:.0f}", ha="center", va="center",
                        fontsize=6.5, color="white")
        left += frac


def main() -> None:
    df = la.arm2c_e2_fate()
    # plot order: corpus at the bottom, then PDFs small-first so largest is on top.
    df = df.iloc[::-1].reset_index(drop=True)
    labels = [f"{s}\n($n={int(n)}$)" for s, n in zip(df["short"], df["n"])]
    y = np.arange(len(df))

    rcparams_report()
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4), sharey=True)

    for ax, (title, hit_col, reached_col) in zip(axes, PANELS):
        _stack(ax, df, hit_col, reached_col, y)
        ax.set_title(title, fontsize=8.5, pad=4)
        ax.set_xlim(0, 1.0)
        ax.set_xlabel("Share of gold articles")
        ax.grid(axis="x", color="#DDDDDD", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)

    axes[0].set_yticks(y, labels)
    handles, leg = axes[0].get_legend_handles_labels()
    fig.legend(handles, leg, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.04),
               columnspacing=1.2, handlelength=1.1, handletextpad=0.4, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    for p in save_figure(fig, OUTPUT_NAME):
        print(f"[done] Wrote {p.relative_to(REPORT_DIR.parent)}")
    plt.close(fig)


if __name__ == "__main__":
    main()
