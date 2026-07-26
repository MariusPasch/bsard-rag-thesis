"""
Build the Tier 3 stratified small-multiples figure.

Output: Report/figures/fig_rq1_t3_strata.{pdf,png}

Same shape as fig_rq1_t1_strata / fig_rq1_t2_strata. To stay within the
"family canonicals only" convention used by fig_rq1_t3_recall_curve, this
figure shows three hybrid systems plus the T2 winner as a reference bar:

  - Hybrid RRF (k=60)         -- T3 canonical
  - Hybrid linear (alpha=0.9) -- family canonical
  - Hybrid SGDR (K=1000)      -- family canonical
  - mE5-large concat-2x       -- T2 winner, faded reference bar

Read alongside Tier 1's and Tier 2's strata figures, this closes Act 2 -- the
hybrid winner is the only system that leads on both `lexically_aligned` and
`semantically_paraphrased` columns.

Usage:
  .venv/Scripts/python Report/scripts/build_t3_strata.py
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

PLOT_KEYS = [
    "hybrid_rrf_k60",
    "hybrid_linear_alpha_0.9",
    "hybrid_sgdr_k1000",
    "dense_me5_large",   # T2 winner reference (rendered faded)
]
REFERENCE_KEY = "dense_me5_large"

OUTPUT_NAME = "fig_rq1_t3_strata"


def main() -> None:
    records = load_records()
    recs = [(t, r) for t, r in records if t in ("T2", "T3")]
    if not recs:
        raise SystemExit("No T2/T3 result records found.")

    long = to_long(recs)
    r10 = long[(long["metric"] == "Recall") & (long["k"] == 10)]

    # Filter the systems we want to plot, in declaration order from PLOT_KEYS.
    sys_order: list[dict] = []
    overall_by_exp: dict[str, float] = {
        row["experiment_id"]: float(row["value"])
        for _, row in r10[r10["stratum"] == "overall"].iterrows()
    }
    for key in PLOT_KEYS:
        s = SYSTEMS.get(key)
        if s is None:
            print(f"[warn] PLOT_KEYS contains unknown key {key!r}")
            continue
        sys_order.append({
            "key":           s.key,
            "display_short": s.display_short,
            "color":         s.color,
            "is_reference":  s.key == REFERENCE_KEY,
            "overall":       overall_by_exp.get(s.result_id, float("nan")),
        })
    # Sort ASC by overall R@10 so best is at the top of each panel.
    sys_order.sort(key=lambda r: r["overall"] if r["overall"] == r["overall"] else -1)
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
        figsize=(7.5, 3.6),
        sharey=True,
        gridspec_kw={"wspace": 0.25, "hspace": 0.45},
    )

    y = np.arange(n_sys)
    colors = [s["color"] for s in sys_order]
    # Reference bar (T2 winner) drawn with lower alpha + hatch.
    alphas = [0.5 if s["is_reference"] else 1.0 for s in sys_order]
    hatches = ["//" if s["is_reference"] else None for s in sys_order]

    for stratum, col, row, title in STRATA_LAYOUT:
        ax = axes[row, col]
        values = []
        for s in sys_order:
            rid = sys_result_id.get(s["key"])
            values.append(by_strat.get((rid, stratum), float("nan")))
        bars = ax.barh(y, values, color=colors, edgecolor="black",
                       linewidth=0.4, alpha=1.0)
        for bar, a, h in zip(bars, alphas, hatches):
            bar.set_alpha(a)
            if h:
                bar.set_hatch(h)
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
