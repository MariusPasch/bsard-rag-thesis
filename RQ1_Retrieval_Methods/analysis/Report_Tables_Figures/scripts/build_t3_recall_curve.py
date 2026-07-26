"""
Build the Tier 3 Recall@k line plot.

Output: Report/figures/fig_rq1_t3_recall_curve.{pdf,png}

T3 has 15 systems but only 3 distinct fusion mechanisms (RRF, linear, SGDR);
the other 12 are parameter variants within each family. Showing all 15 would
overplot. This figure shows one canonical per family (best R@10 within that
family) plus the T2 winner as a faded reference line, so the lift from
dense-only to hybrid is visible:

  - Hybrid RRF (k=60)     -- T3 canonical (and the cross-tier headline)
  - Hybrid linear (alpha=0.9) -- family canonical
  - Hybrid SGDR (K=1000)  -- family canonical
  - mE5-large concat-2x   -- T2 winner, faded orange dashed reference

The within-family design space is covered by fig_rq1_t3_param_sweeps.

Usage:
  .venv/Scripts/python Report/scripts/build_t3_recall_curve.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

REPORT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPORT_DIR))

from load_results import load_records, to_long
from thesis_style import (
    get_system,
    rcparams_report,
    save_figure,
)

K_VALUES = [1, 5, 10, 20, 50, 100]
OUTPUT_NAME = "fig_rq1_t3_recall_curve"

# Family canonicals + T2 winner reference.
CANONICAL_KEYS = [
    "hybrid_rrf_k60",
    "hybrid_linear_alpha_0.9",
    "hybrid_sgdr_k1000",
]
REFERENCE_KEY = "dense_me5_large"   # T2 winner (mE5-large concat-2x)


def main() -> None:
    records = load_records()
    # Pull both T2 and T3 so the reference line is available.
    recs = [(t, r) for t, r in records if t in ("T2", "T3")]
    if not recs:
        raise SystemExit("No T2/T3 result records found.")

    long = to_long(recs)
    rec = long[(long["metric"] == "Recall")
               & (long["stratum"] == "overall")
               & (long["k"].isin(K_VALUES))]

    plot_keys = CANONICAL_KEYS + [REFERENCE_KEY]
    series: dict[str, dict] = {}
    for _, row in rec.iterrows():
        s = get_system(row["experiment_id"])
        if s is None or s.key not in plot_keys:
            continue
        d = series.setdefault(s.key, {"style": s, "k": [], "y": []})
        d["k"].append(int(row["k"]))
        d["y"].append(float(row["value"]))

    rcparams_report()
    fig, ax = plt.subplots(figsize=(6.5, 3.6))

    # Draw the reference line first so the canonical lines sit on top of it.
    ref = series.get(REFERENCE_KEY)
    if ref is not None:
        ks, ys = zip(*sorted(zip(ref["k"], ref["y"])))
        ax.plot(
            ks, ys,
            color=ref["style"].color, marker=ref["style"].marker,
            linestyle=(0, (3, 3)), linewidth=1.0, markersize=4.0,
            alpha=0.5,
            label=f"{ref['style'].display_short} (T2 winner, reference)",
        )

    # Then the three T3 family canonicals.
    for key in CANONICAL_KEYS:
        d = series.get(key)
        if d is None:
            print(f"[warn] missing series for {key}")
            continue
        s = d["style"]
        ks, ys = zip(*sorted(zip(d["k"], d["y"])))
        ax.plot(
            ks, ys,
            color=s.color, marker=s.marker, linestyle=s.linestyle,
            linewidth=1.6, markersize=5.5, markeredgewidth=0.6,
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
        loc="lower right", fontsize=9,
        handlelength=2.6, handletextpad=0.6,
        frameon=False,
    )

    fig.tight_layout()
    paths = save_figure(fig, OUTPUT_NAME)
    plt.close(fig)
    for p in paths:
        print(f"[done] Wrote {p.relative_to(REPORT_DIR.parent)}")


if __name__ == "__main__":
    main()
