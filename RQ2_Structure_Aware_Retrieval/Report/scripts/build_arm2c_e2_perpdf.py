"""Build the Arm-2C E2 per-document figure — does the synthesis generalise?

Output: Report/figures/fig_rq2_arm2c_e2_perpdf.{pdf,png}

Per PDF (+ corpus-weighted), two bars — simple navigation (8B over the single-pass
pool) vs CNR (e5 over the corrective pool) — with the embedding-retrieval ceiling
(Arm-2A) marked as a diamond. Shows CNR lifts every document, by how much, and how
much of the gap to pure embedding it closes.

Run via the RQ2 project venv:
  .venv/Scripts/python Report/scripts/build_arm2c_e2_perpdf.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPORT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import load_arm2c as la  # noqa: E402
from thesis_style import ARMS, PALETTE, FIGSIZE, rcparams_report, save_figure  # noqa: E402

OUTPUT_NAME = "fig_rq2_arm2c_e2_perpdf"
TEAL = ARMS["arm2c"].color
GREY = PALETTE["grey_light"]
GREY_D = PALETTE["grey_dark"]
A2A = ARMS["arm2a"].color


def main() -> None:
    df = la.arm2c_e2_all()
    # PDFs large-first (DOCS order), corpus row last; plot top-to-bottom in that order.
    df = df.reset_index(drop=True)
    rcparams_report()
    y = np.arange(len(df))[::-1]                 # first row at top
    h = 0.38
    fig, ax = plt.subplots(figsize=FIGSIZE["report_wide"])

    ax.barh(y + h / 2, df["r10_8B_single"], height=h, color=GREY, zorder=3,
            label="Simple navigation (8B · single)")
    ax.barh(y - h / 2, df["r10_e5_corr"], height=h, color=TEAL, zorder=3,
            label="CNR (e5 · corrective)")
    ax.scatter(df["arm2a"], y, marker="D", s=34, facecolor="white",
               edgecolor=A2A, linewidth=1.4, zorder=5,
               label="Embedding ceiling (Arm-2A)")

    for yi, r in zip(y, df.itertuples()):
        ax.text(r.r10_8B_single + 0.008, yi + h / 2, f"{r.r10_8B_single:.2f}",
                va="center", ha="left", fontsize=7, color=GREY_D)
        bold = (r.short == "Corpus")
        ax.text(r.r10_e5_corr + 0.008, yi - h / 2, f"{r.r10_e5_corr:.2f}",
                va="center", ha="left", fontsize=7.5,
                fontweight=("bold" if bold else "normal"), color=TEAL)

    labels = [f"{r.short}  (n={int(r.n)})" for r in df.itertuples()]
    ax.set_yticks(y); ax.set_yticklabels(labels)
    # visually separate the corpus row
    corpus_y = y[df.index[df["short"] == "Corpus"][0]]
    ax.axhline(corpus_y + 0.62, color="#DDDDDD", lw=0.8, zorder=0)

    ax.set_xlabel("Recall@10 (question-weighted)")
    ax.set_xlim(0, 0.88)
    ax.grid(axis="x", color="#EEEEEE", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=7.5, frameon=True, framealpha=0.92,
              facecolor="white", edgecolor="#DDDDDD")

    fig.suptitle("CNR lifts every document — unevenly", fontsize=10.5, y=0.99)
    ax.set_title("CNR (corrective navigation reach + embedding ranking) vs simple "
                 "navigation; diamond = pure-embedding ceiling", fontsize=8,
                 color=GREY_D, pad=6)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for p in save_figure(fig, OUTPUT_NAME):
        print(f"[done] Wrote {p.relative_to(REPORT_DIR.parent)}")
    plt.close(fig)


if __name__ == "__main__":
    main()
