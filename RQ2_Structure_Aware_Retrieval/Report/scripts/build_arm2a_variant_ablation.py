"""Arm 2A within-arm variant ablation — recall@k curves, split by PDF, vs Arm 1.

Output: Report/figures/fig_rq2_arm2a_variant_ablation.{pdf,png}

Per-PDF small multiples (2x3: the five curated codes + an aggregate panel).
Each panel plots Recall@k vs retrieval pool size k (log-x) for:
  * Arm 1 (naive)                          — the structure-blind baseline
  * 2A summary · node  (canonical)         — the only Arm-2A lift
  * 2A raw == enriched == full · node      — boost/filter + FR headers inert
  * 2A raw == full · article               — boost/filter inert
The six executed Arm-2A variants collapse to three curves (the query-time
boost/filter and the FR doc-context headers are byte-identical to raw; only the
index-time English gpt-4o summary moves a number). Adding Arm 1 shows the
metadata enrichment essentially tracks the structure-blind baseline per PDF.

Reads the consolidated cross-arm long frame (one path); the aggregate panel
micro-averages over the 5 PDFs. Never re-evaluates.

Caveat: pool depth differs (Arm 1 = 200, Arm 2A = 100); curves are shown over
the common k <= 100.

Usage:
  .venv/Scripts/python Report/scripts/build_arm2a_variant_ablation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPORT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import thesis_style as ts
from thesis_style import load_long_dataframe, rcparams_report, save_figure

OUTPUT_NAME = "fig_rq2_arm2a_variant_ablation"
KS = [5, 10, 20, 100]

# (method, label, color, linestyle, marker, linewidth, zorder)
PROFILES = [
    ("T03_arm1_naive",
     "Arm 1 (naive)", ts.PALETTE["green"], "-", "D", 1.7, 5),
    ("T04_summary_node_hybrid",
     "2A  summary · node  (canonical)", ts.PALETTE["orange"], "-", "*", 2.2, 6),
    ("T04_raw_node_hybrid",
     "2A  raw ≡ enriched ≡ full · node", "#8C510A", "--", "s", 1.5, 4),
    ("T04_raw_article_hybrid",
     "2A  raw ≡ full · article", "#C9A227", ":", "^", 1.5, 4),
]


def main() -> None:
    df = load_long_dataframe()

    def series(method: str, stem: str | None) -> list[float]:
        d = df[df.method == method]
        out = []
        for k in KS:
            dk = d[d.metric == f"T1/R@{k}"]
            if stem is None:  # micro over the 5 PDFs
                out.append((dk.value * dk.n_method_q).sum() / dk.n_method_q.sum())
            else:
                out.append(float(dk[dk.stem == stem].value.iloc[0]))
        return out

    # PDFs ordered by question count (desc), then an aggregate panel
    nq = (df[(df.method == "T03_arm1_naive") & (df.metric == "T1/R@10")]
          .set_index("stem")[["stem_label", "n_method_q"]])
    stems = nq.sort_values("n_method_q", ascending=False).index.tolist()
    panels = [(s, nq.loc[s, "stem_label"], int(nq.loc[s, "n_method_q"])) for s in stems]
    panels.append((None, "All PDFs (micro)", int(nq.n_method_q.sum())))

    rcparams_report()
    fig, axes = plt.subplots(2, 3, figsize=(10.0, 6.3), sharex=True, sharey=True)
    axes = axes.ravel()

    handles = None
    for ax, (stem, label, n) in zip(axes, panels):
        for method, lab, color, ls, marker, lw, z in PROFILES:
            ys = series(method, stem)
            ax.plot(KS, ys, ls=ls, color=color, marker=marker,
                    markersize=8 if marker == "*" else 4.5, linewidth=lw,
                    label=lab, zorder=z,
                    markerfacecolor=color if marker == "*" else "white",
                    markeredgecolor=color, markeredgewidth=1.1)
        if handles is None:
            handles, labels = ax.get_legend_handles_labels()
        ax.set_title(f"{label}  ($n={n}$)", fontsize=9.5)
        ax.set_xscale("log")
        ax.set_xticks(KS, [str(k) for k in KS])
        ax.set_xlim(4.4, 135)
        ax.set_ylim(0, 1.0)
        ax.grid(color="#DDDDDD", linewidth=0.5)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # lay axes out in the upper ~84%; reserve the bottom band for label + legend
    fig.tight_layout(rect=(0.05, 0.15, 1.0, 1.0))
    fig.supylabel("Recall@$k$  (binary, full GT)", x=0.008, fontsize=10)
    fig.text(0.52, 0.105, "Retrieval pool size  $k$  (log scale)",
             ha="center", va="center", fontsize=10)
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=8.5,
               frameon=False, handlelength=2.4, handletextpad=0.5,
               columnspacing=1.8, bbox_to_anchor=(0.52, 0.0))

    paths = save_figure(fig, OUTPUT_NAME)
    plt.close(fig)
    for p in paths:
        print(f"[done] Wrote {p.relative_to(REPORT_DIR.parent)}")


if __name__ == "__main__":
    main()
