"""Build the RQ2 per-document Recall@k small-multiples figure.

Output: Report/figures/fig_rq2_perpdf_recall_curve.{pdf,png}

The per-PDF version of fig_rq2_headline_recall_curve: one panel per law, each a
Recall@k curve (log-x, k in {5, 10, 20, 100}) for the three canonical arms, with
95% bootstrap CI whiskers at k = 10 and k = 100 (per that law's questions) and a
dotted Arm-2 pool-ceiling line at k = 100. Panels share axes; the sixth grid
slot carries the legend. PDFs ordered by question count.

Reading: the pool-depth crossover (Arm 1 overtaking Arm 2A by k = 100) is not
uniform -- it is pronounced on Code Civil and the Codes Judiciaires but absent
on the Code du Logement (where both saturate near 0.85). Arm 2B trails in every
panel.

Usage:
  .venv/Scripts/python Report/scripts/build_perpdf_recall_curve.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPORT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import load_arm2c as la
import load_results as lr
from thesis_style import POOL_DEPTHS, get_variant, rcparams_report, save_figure

OUTPUT_NAME = "fig_rq2_perpdf_recall_curve"
ARMS = ["T03_arm1_naive", "T04_summary_node_hybrid", "T05_pageindex", "ARM2C_agentic"]
K_CURVE = [5, 10, 20, 100]
CI_K = [10, 100]
N_BOOT = 2000
SEED = 42


def _bootstrap_ci(vals: np.ndarray, n: int = N_BOOT, seed: int = SEED):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(vals), size=(n, len(vals)))
    means = vals[idx].mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def main() -> None:
    long = lr.load_long(include_arm2c=True)
    pq = lr.load_per_query(include_arm2c=True)
    ps = lr.per_stem_wide(long, [f"R@{k}" for k in K_CURVE])
    ps = ps[ps["method"].isin(ARMS)]

    order = (ps[["stem", "stem_label", "n_gt_q"]].drop_duplicates()
             .sort_values("n_gt_q", ascending=False))
    stems = order["stem"].tolist()
    label = dict(zip(order["stem"], order["stem_label"]))
    n_by_stem = dict(zip(order["stem"], order["n_gt_q"]))
    val = {(r["stem"], r["method"]): {k: float(r[f"R@{k}"]) for k in K_CURVE}
           for _, r in ps.iterrows()}

    # Arm-2C CNR per stem: full curve (k=5/10/20/100), recovered by re-running the
    # e5 rerank over the saved corrective pools and capturing the ordered list
    # (corrective/e2_metrics_full.py -> arm2c_cnr_metrics_per_pdf).
    cpp = la.arm2c_cnr_metrics_per_pdf().set_index("stem")
    cnr = {s: {k: float(cpp.loc[s, f"R@{k}"]) for k in K_CURVE} for s in stems}
    teal = get_variant("ARM2C_agentic").color

    rcparams_report()
    fig, axes = plt.subplots(2, 3, figsize=(7.5, 4.6), sharex=True, sharey=True,
                             gridspec_kw={"wspace": 0.12, "hspace": 0.38})
    axf = axes.ravel()

    handles = labels_ = None
    for i, stem in enumerate(stems):
        ax = axf[i]
        for method in ARMS:
            v = get_variant(method)
            ys = [val[(stem, method)][k] for k in K_CURVE]
            lab = "Arm 2C (simple)" if method == "ARM2C_agentic" else v.display_short
            ax.plot(K_CURVE, ys, color=v.color, marker=v.marker,
                    linestyle=v.linestyle, linewidth=1.3, markersize=4.5,
                    markeredgewidth=0.5, label=lab, zorder=3)
            for k in CI_K:
                vals = pq[(pq["method"] == method) & (pq["k"] == k)
                          & (pq["stem"] == stem)]["recall"].to_numpy(float)
                if len(vals) == 0:
                    continue
                lo, hi = _bootstrap_ci(vals)
                y = val[(stem, method)][k]
                ax.errorbar([k], [y], yerr=[[y - lo], [hi - y]], fmt="none",
                            ecolor=v.color, elinewidth=1.0, capsize=2.0, zorder=2)
        # Arm-2C CNR: full dashed curve (k=5/10/20/100).
        ax.plot(K_CURVE, [cnr[stem][k] for k in K_CURVE], color=teal, marker="s",
                linestyle="--", linewidth=1.3, markersize=4.5, markeredgewidth=0.9,
                markerfacecolor="white", label="Arm 2C — CNR", zorder=4)
        ax.axvline(POOL_DEPTHS["arm2a"], color="#AAAAAA", linestyle=":",
                   linewidth=0.9, zorder=1)
        ax.set_xscale("log")
        ax.set_xticks(K_CURVE, [str(k) for k in K_CURVE])
        ax.minorticks_off()
        ax.set_xlim(K_CURVE[0] * 0.85, K_CURVE[-1] * 1.2)
        ax.set_ylim(0, 1.0)
        ax.tick_params(labelbottom=True, labelsize=8)
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.set_title(rf"{label[stem]}  ($n={int(n_by_stem[stem])}$)",
                     fontsize=8.5, pad=3)
        if handles is None:
            handles, labels_ = ax.get_legend_handles_labels()

    # Legend in the empty sixth slot.
    legend_ax = axf[len(stems)]
    legend_ax.axis("off")
    legend_ax.legend(handles, labels_, loc="center", fontsize=9, frameon=False,
                     title="Arm", title_fontsize=9, handlelength=2.2)
    for j in range(len(stems) + 1, len(axf)):
        axf[j].axis("off")

    fig.supxlabel(r"Retrieval depth / pool size $k$ (log scale); dotted line = Arm-2 pool ceiling",
                  fontsize=9, y=0.01)
    fig.supylabel(r"Recall@$k$", fontsize=10, x=0.005)

    fig.tight_layout(rect=(0.01, 0.02, 1, 1))
    paths = save_figure(fig, OUTPUT_NAME)
    plt.close(fig)
    for p in paths:
        print(f"[done] Wrote {p.relative_to(REPORT_DIR.parent)}")


if __name__ == "__main__":
    main()
