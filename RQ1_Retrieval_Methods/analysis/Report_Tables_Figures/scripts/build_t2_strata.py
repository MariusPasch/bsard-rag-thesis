"""
Build the Tier 2 stratified small-multiples figure.

Output: Report/figures/fig_rq1_t2_strata.{pdf,png}

Same shape as fig_rq1_t1_strata, different system list.

When read alongside the Tier 1 strata figure, this figure tells the Act 2
"the paraphrase gap closes" story:
  - col 2, row 2 (semantically_paraphrased): every sparse system collapsed to
    R@10 < 0.08 at Tier 1; dense systems lift it to R@10 ~ 0.14 - 0.20.
  - col 2, row 1 (lexically_aligned): sparse still wins this column, so dense
    is not strictly better -- it is complementary, which sets up Tier 3.

Usage:
  .venv/Scripts/python Report/scripts/build_t2_strata.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPORT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPORT_DIR))

from load_results import load_records, to_long
from thesis_style import (
    SYSTEMS,
    get_system,
    rcparams_report,
    save_figure,
)

STRATA_LAYOUT = [
    ("single_article",            0, 0, "Single-article queries"),
    ("multi_article",             0, 1, "Multi-article queries"),
    ("lexically_aligned",         1, 0, "Lexically aligned"),
    ("semantically_paraphrased",  1, 1, "Semantically paraphrased"),
    ("with_cross_refs",           2, 0, "With cross-references"),
    ("without_cross_refs",        2, 1, "Without cross-references"),
]

TIER = "T2"
OUTPUT_NAME = "fig_rq1_t2_strata"


def main() -> None:
    records = load_records()
    recs = [(t, r) for t, r in records if t == TIER]
    if not recs:
        raise SystemExit(f"No {TIER} result records found.")

    long = to_long(recs)
    r10 = long[(long["metric"] == "Recall") & (long["k"] == 10)]

    # System order: ascending by overall R@10 so best ends at top of plot.
    overall = r10[r10["stratum"] == "overall"]
    sys_order: list[dict] = []
    for _, row in overall.iterrows():
        s = get_system(row["experiment_id"])
        if s is None:
            continue
        sys_order.append({
            "key":           s.key,
            "display_short": s.display_short,
            "color":         s.color,
            "overall":       float(row["value"]),
        })
    sys_order.sort(key=lambda r: r["overall"])
    n_sys = len(sys_order)
    if n_sys == 0:
        raise SystemExit("Loader produced no matching SYSTEMS entries.")

    by_strat: dict[tuple[str, str], float] = {}
    for _, row in r10.iterrows():
        by_strat[(row["experiment_id"], row["stratum"])] = float(row["value"])

    sys_result_id = {s.key: s.result_id for s in SYSTEMS.values()}

    rcparams_report()
    fig, axes = plt.subplots(
        2, 3,
        figsize=(7.5, 5.0),
        sharey=True,
        gridspec_kw={"wspace": 0.25, "hspace": 0.45},
    )

    y = np.arange(n_sys)
    colors = [s["color"] for s in sys_order]

    for stratum, col, row, title in STRATA_LAYOUT:
        ax = axes[row, col]
        values = []
        for s in sys_order:
            rid = sys_result_id.get(s["key"])
            values.append(by_strat.get((rid, stratum), float("nan")))
        ax.barh(y, values, color=colors, edgecolor="black", linewidth=0.4)
        ax.set_yticks(y, [s["display_short"] for s in sys_order])

        finite = [v for v in values if v == v]
        xmax = max(finite) if finite else 1.0
        ax.set_xlim(0, max(0.05, xmax * 1.25))
        for yi, v in zip(y, values):
            if v != v:
                continue
            ax.text(v + xmax * 0.04, yi, f"{v:.3f}",
                    va="center", ha="left", fontsize=7)

        ax.set_title(title, fontsize=9, pad=4)
        ax.tick_params(axis="x", labelsize=8)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="x", color="#DDDDDD", linewidth=0.6)
        ax.set_axisbelow(True)

    for ax in axes[1, :]:
        ax.set_xlabel(r"Recall@10")

    fig.tight_layout()
    paths = save_figure(fig, OUTPUT_NAME)
    plt.close(fig)
    for p in paths:
        print(f"[done] Wrote {p.relative_to(REPORT_DIR.parent)}")


if __name__ == "__main__":
    main()
