"""Build the Arm-2C 'how to fix the tree' figure: reachability barriers per PDF.

Output: Report/figures/fig_rq2_arm2c_tree_barriers.{pdf,png}

An article is *blocked from being picked* when the agent never reaches its menu.
This figure separates the two kinds of barrier (and the tree fix each implies),
over the distinct gold articles of each document:

  LEFT  -- HARD blocks (structurally hard/impossible to reach), fix = repair tree:
             * orphaned in the catch-all branch  -> reattach by page proximity
             * in an oversized (>12-article) section -> split the section
  RIGHT -- SOFT barrier = descent depth: how many correct sequential branch
           choices are needed to reach the article (>=4 = a long descent whose
           per-step error compounds), fix = flatten / collapse sub-levels.

Reading: the Code du Logement (housing) is dominated by HARD blocks (19% orphaned)
-> reattach; the smaller Code Judiciaire by an oversized 63-article section ->
split; Code Pénal has ~no hard blocks but ~95% of its gold sits behind >=4 choices
-> flatten the deep offence taxonomy. (Depth limits even the winners, so the soft
fix helps every code.)

Run via the RQ2 project venv:
  .venv/Scripts/python Report/scripts/build_arm2c_tree_barriers.py
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


def main() -> None:
    df = la.arm2c_reachability_barriers().sort_values("mean_dec", ascending=False)
    df = df.reset_index(drop=True)
    y = np.arange(len(df))

    rcparams_report()
    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(10.6, 3.9), sharey=True,
        gridspec_kw={"wspace": 0.07})

    # ---- LEFT: hard blocks (orphan + oversized) ----------------------------
    orph = df["orphan_pct"].to_numpy(float)
    over = df["oversized_pct"].to_numpy(float)
    axA.barh(y, orph, color=PALETTE["grey_mid"], edgecolor="white", linewidth=0.4,
             label="Orphaned (catch-all)  →  reattach by proximity")
    axA.barh(y, over, left=orph, color=PALETTE["orange"], edgecolor="white",
             linewidth=0.4, label="Oversized section (>12 art.)  →  split section")
    for i, (o, v) in enumerate(zip(orph, over)):
        if o >= 0.04:
            axA.text(o / 2, i, f"{o*100:.0f}", ha="center", va="center",
                     fontsize=6.5, color="white")
        if v >= 0.04:
            axA.text(o + v / 2, i, f"{v*100:.0f}", ha="center", va="center",
                     fontsize=6.5, color="white")
    axA.set_xlim(0, 0.8)
    axA.set_xlabel("Share of gold articles")
    axA.set_title("Hard blocks — repair the tree", fontsize=10)
    axA.grid(axis="x", color="#EEEEEE", linewidth=0.6)
    axA.set_axisbelow(True)
    axA.spines["left"].set_visible(False)
    axA.tick_params(axis="y", length=0)
    axA.legend(loc="lower right", fontsize=7.0, frameon=True, framealpha=0.92,
               facecolor="white", edgecolor="#DDDDDD", handlelength=1.1,
               handletextpad=0.4)

    # ---- RIGHT: descent depth (soft barrier) -------------------------------
    shallow = (df["dec1"] + df["dec2"]).to_numpy(float)
    mid = df["dec3"].to_numpy(float)
    deep = df["dec4plus"].to_numpy(float)
    segs = [(shallow, PALETTE["green"], "≤ 2 choices"),
            (mid, PALETTE["orange"], "3 choices"),
            (deep, PALETTE["vermillion"], "≥ 4 choices (long descent)")]
    left = np.zeros(len(df))
    for vals, color, label in segs:
        axB.barh(y, vals, left=left, color=color, edgecolor="white",
                 linewidth=0.4, label=label)
        for i, (v, l) in enumerate(zip(vals, left)):
            if v >= 0.10:
                axB.text(l + v / 2, i, f"{v*100:.0f}", ha="center", va="center",
                         fontsize=6.5, color="white")
        left += vals
    axB.set_xlim(0, 1.0)
    axB.set_xlabel("Share of gold articles")
    axB.set_title("Descent depth — flatten the tree", fontsize=10)
    axB.grid(axis="x", color="#EEEEEE", linewidth=0.6)
    axB.set_axisbelow(True)
    axB.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3,
               fontsize=7.0, frameon=False, columnspacing=1.1, handlelength=1.0,
               handletextpad=0.4)

    axA.set_yticks(y)
    axA.set_yticklabels([f"{s}  ($n={int(n)}$)"
                         for s, n in zip(df["short"], df["n_gold"])], fontsize=8.5)
    axA.invert_yaxis()

    fig.suptitle("Arm-2C (simple navigation) — structural barriers the single greedy "
                 "pass faces; CNR's corrective loop targets the depth barrier\n"
                 "Tree fix implied: Code Logement → reattach orphans · smaller Code "
                 "Judiciaire → split a 63-art section · Code Pénal → flatten a deep "
                 "taxonomy", fontsize=9.0, y=1.05)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))

    for p in save_figure(fig, "fig_rq2_arm2c_tree_barriers"):
        print(f"[done] Wrote {p.relative_to(REPORT_DIR.parent)}")
    plt.close(fig)


if __name__ == "__main__":
    main()
