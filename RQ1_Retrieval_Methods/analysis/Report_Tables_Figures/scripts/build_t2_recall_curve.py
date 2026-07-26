"""
Build the Tier 2 Recall@k line plot.

Output: Report/figures/fig_rq1_t2_recall_curve.{pdf,png}

Same shape as fig_rq1_t1_recall_curve, different system list:
  - One line per Tier 2 dense system (9 total)
  - mE5 family shares marker D, varies linestyle
  - Qwen3 shares marker P
  - Singletons (BGE-M3, CamemBERT-lg, mpnet-multi) use unique markers
  - CamemBERT-base (paper replication anchor) rendered in grey to signal
    "baseline expected to collapse" rather than competing encoder

Usage:
  .venv/Scripts/python Report/scripts/build_t2_recall_curve.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

REPORT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPORT_DIR))

from load_results import load_records, to_long
from thesis_style import (
    SYSTEMS,
    get_system,
    rcparams_report,
    save_figure,
)

K_VALUES = [1, 5, 10, 20, 50, 100]
TIER = "T2"
OUTPUT_NAME = "fig_rq1_t2_recall_curve"


def main() -> None:
    records = load_records()
    recs = [(t, r) for t, r in records if t == TIER]
    if not recs:
        raise SystemExit(f"No {TIER} result records found.")

    long = to_long(recs)
    rec = long[(long["metric"] == "Recall")
               & (long["stratum"] == "overall")
               & (long["k"].isin(K_VALUES))]

    order_index = {k: i for i, k in enumerate(SYSTEMS)}
    series: dict[str, dict] = {}
    for _, row in rec.iterrows():
        s = get_system(row["experiment_id"])
        if s is None:
            continue
        d = series.setdefault(s.key, {"style": s, "k": [], "y": []})
        d["k"].append(int(row["k"]))
        d["y"].append(float(row["value"]))

    keys_sorted = sorted(series.keys(), key=lambda k: order_index.get(k, 10_000))

    rcparams_report()
    fig, ax = plt.subplots(figsize=(7.4, 4.0))

    for key in keys_sorted:
        d = series[key]
        s = d["style"]
        ks, ys = zip(*sorted(zip(d["k"], d["y"])))
        ax.plot(
            ks, ys,
            color=s.color, marker=s.marker, linestyle=s.linestyle,
            linewidth=1.4, markersize=4.5, markeredgewidth=0.6,
            label=s.display_short,
        )

    ax.set_xscale("log")
    ax.set_xticks(K_VALUES, [str(k) for k in K_VALUES])
    ax.minorticks_off()
    ax.set_xlabel(r"Retrieval depth $k$ (log scale)")
    ax.set_ylabel(r"Recall@$k$ (test, $n=222$)")
    ax.set_xlim(K_VALUES[0] * 0.9, K_VALUES[-1] * 1.1)
    ax.set_ylim(0, None)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)

    ax.legend(
        loc="center left", bbox_to_anchor=(1.02, 0.5),
        fontsize=8, handlelength=2.6, handletextpad=0.6,
        frameon=False, borderaxespad=0.0,
        labelspacing=0.4,
    )

    fig.subplots_adjust(left=0.08, right=0.72, top=0.95, bottom=0.13)
    paths = save_figure(fig, OUTPUT_NAME)
    plt.close(fig)
    for p in paths:
        print(f"[done] Wrote {p.relative_to(REPORT_DIR.parent)}")


if __name__ == "__main__":
    main()
