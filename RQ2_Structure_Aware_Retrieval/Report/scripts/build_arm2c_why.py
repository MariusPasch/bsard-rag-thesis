"""Build the Arm-2C 'why it wins/loses vs PageIndex' diagnostic figure.

Output: Report/figures/fig_rq2_arm2c_why.{pdf,png}

The story: the rebuilt AzureDI deep tree is a *multi-level structure bet*. Its
descent succeeds only if every level is right, so it can break at a different
level on each document. It beats PageIndex sharply where the hierarchy is clean
and the coarse branch is reliably localised (Code Civil, larger Code Judiciaire),
and ties/loses where the descent breaks -- at the coarse level (smaller Code
Judiciaire: wrong top-branch), structurally (housing code: 25% orphan, fragmented
14-way root), or at the fine level (Code Penal: right book, wrong chapter in a
deep offence taxonomy). PageIndex flattens the tree to one chapter pick, so it
has a single uniform failure mode -- never exploiting depth, never wrecked by it.

The headline config is CNR (Corrective Navigation and Re-Ranking); simple
navigation is shown as an intermediate reference so the lever's contribution and
the residual gap to embedding are both visible.

Two panels, PDF rows shared and ordered by CNR minus PageIndex (winners top):
  LEFT  -- outcome: Recall@10 lollipop, PageIndex -> simple nav -> CNR, with the
           embedding arm (Arm-2A) as the ceiling reference; CNR-vs-2B delta labelled.
  RIGHT -- mechanism: per-question CNR gold-fate decomposition (sums to 1); the
           green HIT segment equals CNR's recall@10, so the two panels align. The
           corrective loop shrinks the fine/coarse/orphan misses (vs simple nav)
           by re-reaching pruned branches.

Run via the RQ2 project venv:
  .venv/Scripts/python Report/scripts/build_arm2c_why.py
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

OUTPUT_NAME = "fig_rq2_arm2c_why"

# gold-fate segments, left->right = success then by descent level the miss hit.
FATE = [
    ("hit",     PALETTE["green"],      "Hit@10 (= Arm-2C CNR recall@10)"),
    ("misrank", PALETTE["sky"],        "Reached, mis-ranked"),
    ("fine",    PALETTE["orange"],     "Fine-prune (wrong section)"),
    ("coarse",  PALETTE["vermillion"], "Coarse-prune (wrong branch)"),
    ("orphan",  PALETTE["grey_mid"],   "Orphan (broken hierarchy)"),
]

# curated dominant cause per document (from the trace+tree analysis).
CAUSE = {
    "1967_10_10_1967101055": "clean, wide tree — coarse localises",
    "1804_03_21_1804032150": "clean tree — residual deep-section misses",
    "2003_07_17_2013A31614": "broken tree (19% orphan, 14-way root)",
    "1967_10_10_1967101056": "coarse mis-localisation (wrong branch)",
    "1867_06_08_1867060850": "deep offence taxonomy (right book, wrong chapter)",
}


def main() -> None:
    df = la.arm2c_failure_decomp_cnr()
    df["rel"] = 100.0 * (df["arm2c_cnr"] - df["arm2b"]) / df["arm2b"]
    df = df.sort_values("rel", ascending=False).reset_index(drop=True)
    y = np.arange(len(df))

    rcparams_report()
    fig, (ax0, ax1) = plt.subplots(
        1, 2, figsize=(10.6, 4.3), sharey=True,
        gridspec_kw={"width_ratios": [1.0, 1.35], "wspace": 0.06})

    # ---- LEFT: outcome lollipop (PageIndex -> simple-nav -> CNR, + ceiling) --
    for i, r in df.iterrows():
        ax0.plot([r["arm2b"], r["arm2c_cnr"]], [i, i], color="#BBBBBB",
                 lw=1.4, zorder=1)
    ax0.scatter(df["arm2a"], y, s=34, marker="d", facecolor="white",
                edgecolor=ARMS["arm2a"].color, linewidth=1.2, zorder=3,
                label="Embedding (Arm 2A)")
    ax0.scatter(df["arm2b"], y, s=46, color=ARMS["arm2b"].color, zorder=4,
                label="PageIndex (Arm 2B)")
    ax0.scatter(df["arm2c_simple"], y, s=42, marker="o", facecolor="white",
                edgecolor=ARMS["arm2c"].color, linewidth=1.2, zorder=4,
                label="Arm-2C simple nav.")
    ax0.scatter(df["arm2c_cnr"], y, s=62, color=ARMS["arm2c"].color, zorder=5,
                label="Arm-2C CNR")
    for i, r in df.iterrows():
        win = r["arm2c_cnr"] >= r["arm2b"]
        ax0.annotate(f"{r['rel']:+.0f}%",
                     (max(r["arm2b"], r["arm2c_cnr"], r["arm2a"]) + 0.015, i),
                     va="center", ha="left", fontsize=7.5,
                     color=PALETTE["green"] if win else PALETTE["vermillion"])
    ax0.set_xlim(0, 0.92)
    ax0.set_xlabel("Recall@10")
    ax0.set_title("Outcome: PageIndex → simple → CNR", fontsize=10)
    ax0.legend(loc="upper right", fontsize=7.0, frameon=True, framealpha=0.92,
               facecolor="white", edgecolor="#DDDDDD", handletextpad=0.3)
    ax0.grid(axis="x", color="#EEEEEE", linewidth=0.6)
    ax0.set_axisbelow(True)

    # ---- RIGHT: CNR gold-fate decomposition (mechanism) --------------------
    left = np.zeros(len(df))
    for key, color, label in FATE:
        ax1.barh(y, df[key], left=left, height=0.62, color=color,
                 edgecolor="white", linewidth=0.4, label=label)
        for i, (v, l) in enumerate(zip(df[key], left)):
            if v >= 0.10:
                ax1.text(l + v / 2, i, f"{v*100:.0f}", ha="center", va="center",
                         fontsize=6.2, color="white")
        left += df[key].to_numpy()
    ax1.set_xlim(0, 1.0)
    ax1.set_xlabel("Share of gold articles (per-question mean)")
    ax1.set_title("Where Arm-2C (CNR)'s gold goes (why)", fontsize=10)
    ax1.grid(axis="x", color="#EEEEEE", linewidth=0.6)
    ax1.set_axisbelow(True)
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3,
               fontsize=7.3, frameon=False, columnspacing=1.2,
               handlelength=1.1, handletextpad=0.4)

    # ---- shared 2-line y labels: document + curated cause ------------------
    labels = [f"{r['short']}  ($n={int(r['n'])}$)\n{CAUSE[r['stem']]}"
              for _, r in df.iterrows()]
    ax0.set_yticks(y)
    ax0.set_yticklabels(labels, fontsize=8)
    ax0.invert_yaxis()   # biggest winner on top

    fig.suptitle("The deep tree is a structure bet: CNR beats PageIndex where the "
                 "hierarchy is clean, and breaks at a\ndifferent descent level on "
                 "each document where it loses", fontsize=10.5, y=1.02)
    fig.tight_layout(rect=(0, 0.02, 1, 1))

    for p in save_figure(fig, OUTPUT_NAME):
        print(f"[done] Wrote {p.relative_to(REPORT_DIR.parent)}")
    plt.close(fig)


if __name__ == "__main__":
    main()
