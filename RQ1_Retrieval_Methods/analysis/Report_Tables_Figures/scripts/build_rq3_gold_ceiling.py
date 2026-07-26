"""
Build the RQ3 gold-ceiling artifacts (Chapter 14, Act 2 opener).

Outputs:
  Report/figures/fig_rq3_gold_ceiling.{pdf,png}
      Per evaluator, grouped horizontal bars: the gold ceiling (evaluator score
      on perfect / gold-only retrieval) vs the best retriever's achieved score
      (max over the 19 systems) and the 19-system mean. The ceiling-to-best gap
      is the headline; where best > ceiling the evaluator over-credits
      retrieved-but-not-gold passages.
  Report/tables/tab_rq3_gold_ceiling.tex
      Evaluators x {gold ceiling, best retriever, subset mean, headroom}.

Source: output/results/RQ3/gold_ceiling.json + summary.csv.

Usage:
  .venv/Scripts/python Report/scripts/build_rq3_gold_ceiling.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPORT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPORT_DIR))

from thesis_style import (  # noqa: E402
    EVALUATORS,
    PALETTE,
    rcparams_report,
    save_figure,
    save_table,
)

# REPORT_DIR is <project>/analysis/Report_Tables_Figures; the project root
# (holding output/) is two parents up.
REPO_ROOT = REPORT_DIR.parent.parent
RQ3_DIR = REPO_ROOT / "output" / "results" / "RQ3"

EVALS = [("umbrela", "T3/umbrela/mean"), ("erag", "T3/erag/mean"),
         ("ragas_wa", "T3/ragas_wa/mean")]


def _data():
    summary = pd.read_csv(RQ3_DIR / "summary.csv")
    sysrows = summary[summary["family"] != "oracle"]
    ceil = json.loads((RQ3_DIR / "gold_ceiling.json").read_text(encoding="utf-8"))
    m = ceil["subset_metrics"]["metrics"]
    rows = []
    for key, col in EVALS:
        rows.append({
            "key": key,
            "display": EVALUATORS[key].display.split(" (")[0],
            "ceiling": m[f"T3/{key}/mean"],
            "best": float(sysrows[col].max()),
            "mean": float(sysrows[col].mean()),
        })
    return pd.DataFrame(rows)


def build_figure(d: pd.DataFrame) -> None:
    rcparams_report()
    fig, ax = plt.subplots(figsize=(6.7, 3.0))
    y = np.arange(len(d))
    h = 0.26
    ax.barh(y + h, d["ceiling"], h, color=PALETTE["grey_dark"],
            edgecolor="white", label="gold ceiling (oracle)")
    ax.barh(y, d["best"], h, color=[EVALUATORS[k].color for k in d["key"]],
            edgecolor="white", label="best retriever")
    ax.barh(y - h, d["mean"], h, color=PALETTE["grey_light"],
            edgecolor="white", label="19-system mean")
    for yi, r in zip(y, d.itertuples()):
        gap = r.ceiling - r.best
        ax.annotate(f"headroom {gap:+.2f}", (max(r.ceiling, r.best) + 0.01, yi),
                    va="center", ha="left", fontsize=7.5,
                    color=(PALETTE["grey_dark"] if gap >= 0 else PALETTE["vermillion"]))
    ax.set_yticks(y, d["display"])
    ax.set_xlim(0, 0.95)
    ax.set_xlabel(r"autonomous score on the 48-q subset ($k{=}10$)")
    ax.invert_yaxis()
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    for p in save_figure(fig, "fig_rq3_gold_ceiling"):
        print(f"[done] {p.relative_to(REPO_ROOT)}")
    plt.close(fig)


def build_table(d: pd.DataFrame) -> None:
    out = pd.DataFrame({
        "Evaluator": d["display"],
        "Gold ceiling": d["ceiling"].map(lambda v: f"{v:.3f}"),
        "Best retriever": d["best"].map(lambda v: f"{v:.3f}"),
        "Subset mean": d["mean"].map(lambda v: f"{v:.3f}"),
        "Headroom": (d["ceiling"] - d["best"]).map(lambda v: f"{v:+.3f}"),
    })
    tex = out.to_latex(index=False, escape=False, column_format="lcccc")
    p = save_table(tex, "tab_rq3_gold_ceiling")
    print(f"[done] {p.relative_to(REPO_ROOT)}")
    print(out.to_string(index=False))


def main() -> None:
    d = _data()
    build_figure(d)
    build_table(d)


if __name__ == "__main__":
    main()
