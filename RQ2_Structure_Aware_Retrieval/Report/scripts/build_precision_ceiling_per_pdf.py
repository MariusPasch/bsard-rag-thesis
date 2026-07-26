"""Build the cross-arm precision@10 vs achievable-ceiling figure (per PDF).

Output: Report/figures/fig_rq2_precision_ceiling_per_pdf.{pdf,png}

Article precision@10 (gold articles / distinct articles seen in the top-10) is
structurally low for every arm because a BSARD question has only ~1-3 gold
articles -- you cannot fill 10 slots with gold. This figure puts each arm's
achieved precision next to that per-question ceiling:

    achieved_q = n_gold_articles@10 / distinct_articles@10
    ceiling_q  = min(n_gold, distinct_articles@10) / distinct_articles@10

(both averaged over a PDF's questions). The bar is what the arm got; the open
diamond is the best any retriever could get given how many distinct articles it
surfaced. Gap bar->diamond = precision actually left on the table.

Reads the persisted one-path per-question deep-dive tables (never re-evaluates).

Usage:
  .venv/Scripts/python Report/scripts/build_precision_ceiling_per_pdf.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

REPORT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import load_arm2c as la
import thesis_style as ts
from thesis_style import get_variant, rcparams_report, save_figure

OUTPUT_NAME = "fig_rq2_precision_ceiling_per_pdf"
TABLES = ts.T06_DATA_DIR / "tables"

_TEAL = get_variant(la.ARM2C_METHOD).color
# (per-question csv, row filter, key, label override, colour override, alpha).
# The agentic arm has no T06 per_question.csv (run outside the consolidation);
# its precision/ceiling come from load_arm2c. Note: article-level arms (2B/2C)
# sit at distinct@10=10, so their ceiling just tracks gold density (n_gold/10).
ARMS = [
    ("arm1_per_question.csv",  None, "T03_arm1_naive",          None, None, 1.0),
    ("arm2a_per_question.csv", ("variant", "T04_summary_node_hybrid"),
     "T04_summary_node_hybrid", None, None, 1.0),
    ("arm2b_per_question.csv", None, "T05_pageindex",           None, None, 1.0),
    ("ARM2C_SIMPLE", None, "ARM2C_simple", "Arm 2C (simple)", _TEAL, 0.42),
    ("ARM2C_CNR",    None, "ARM2C_cnr",    "Arm 2C (CNR)",    _TEAL, 1.0),
]


def _arm_frame(fname: str, filt) -> pd.DataFrame:
    if fname == "ARM2C_SIMPLE":
        return la.arm2c_precision_ceiling("simple")
    if fname == "ARM2C_CNR":
        return la.arm2c_precision_ceiling("cnr")
    df = pd.read_csv(TABLES / fname)
    if filt is not None:
        df = df[df[filt[0]] == filt[1]]
    d10 = df["distinct_articles@10"].astype(float)
    df = df[d10 > 0].copy()
    d10 = df["distinct_articles@10"].astype(float)
    df["achieved"] = df["n_gold_articles@10"].astype(float) / d10
    df["ceiling"] = np.minimum(df["n_gold"].astype(float), d10) / d10
    return df[["stem", "stem_label", "achieved", "ceiling"]]


def main() -> None:
    frames = {key: _arm_frame(f, filt) for f, filt, key, *_ in ARMS}

    # PDF order: by GT-question count, largest first (use Arm 1's counts).
    base = frames["T03_arm1_naive"]
    order = (base.groupby(["stem", "stem_label"]).size()
             .reset_index(name="n").sort_values("n", ascending=False))
    stems = order["stem"].tolist()
    labels = [f"{lbl}\n($n={int(n)}$)"
              for lbl, n in zip(order["stem_label"], order["n"])]

    rcparams_report()
    fig, ax = plt.subplots(figsize=(7.0, 5.8))
    n_pdfs, n_arms = len(stems), len(ARMS)
    ys = np.arange(n_pdfs)
    bar_h = 0.8 / n_arms

    for j, (_f, _filt, key, lab_o, col_o, alpha) in enumerate(ARMS):
        color = col_o or get_variant(key).color
        label = lab_o or get_variant(key).display_short
        g = frames[key].groupby("stem")[["achieved", "ceiling"]].mean()
        ach = [g.loc[s, "achieved"] if s in g.index else np.nan for s in stems]
        ceil = [g.loc[s, "ceiling"] if s in g.index else np.nan for s in stems]
        offset = (j - (n_arms - 1) / 2) * bar_h     # j=0 (Arm 1) on top (y inverted)
        bar_y = ys + offset
        ax.barh(bar_y, ach, height=bar_h, color=color, alpha=alpha,
                edgecolor="black", linewidth=0.4, label=label)
        # dotted connector bar-tip -> ceiling = precision headroom, in arm colour
        for yy, a, c in zip(bar_y, ach, ceil):
            if a == a and c == c:
                ax.plot([a, c], [yy, yy], color=color, ls=":", lw=0.9, zorder=4)
        ax.scatter(ceil, bar_y, s=30, marker="D", facecolor="white",
                   edgecolor=color, linewidth=1.2, zorder=6)

    ax.set_yticks(ys, labels)
    ax.invert_yaxis()
    xmax = max(frames[k]["ceiling"].mean() for _f, _filt, k, *_ in ARMS) * 1.4
    ax.set_xlim(0, max(0.30, xmax))
    ax.set_xlabel(r"Article precision@10 (gold / distinct articles seen)")
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    handles, lbls = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], marker="D", color="none", markerfacecolor="white",
                          markeredgecolor="black", markersize=7,
                          label="achievable ceiling"))
    ax.legend(handles=handles, loc="lower right", fontsize=8, frameon=True,
              framealpha=0.92, facecolor="white", edgecolor="#DDDDDD",
              handlelength=1.4, handletextpad=0.5)

    fig.tight_layout()
    paths = save_figure(fig, OUTPUT_NAME)
    plt.close(fig)
    for p in paths:
        print(f"[done] Wrote {p.relative_to(REPORT_DIR.parent)}")


if __name__ == "__main__":
    main()
