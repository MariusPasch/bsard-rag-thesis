"""
Build the RQ3 AQS weight-analysis artifacts (Chapter 14, synthesis).

Outputs:
  Report/figures/fig_rq3_aqs_weight_simplex.{pdf,png}
      Ternary plot of the AQS objective -- system-ranking Spearman rho vs gold
      Recall@10 -- over the weight simplex (w_umbrela + w_erag + w_ragas = 1,
      step 0.05). Coloured by rho; the data-driven optimum is marked. The
      maximum sits on the eRAG-dominant edge -- the composite collapses onto eRAG.
  Report/tables/tab_rq3_aqs_weights.tex
      eRAG-alone vs data-driven (in-sample and LOO) AQS weights, with rho vs gold R@10.

Source: output/results/RQ3/summary.csv + aqs_weights.json.

Usage:
  .venv/Scripts/python Report/scripts/build_rq3_aqs.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPORT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPORT_DIR))

from thesis_style import PALETTE, rcparams_report, save_figure, save_table  # noqa: E402

# REPORT_DIR is <project>/analysis/Report_Tables_Figures; the project root
# (holding output/) is two parents up.
REPO_ROOT = REPORT_DIR.parent.parent
RQ3_DIR = REPO_ROOT / "output" / "results" / "RQ3"


def _load():
    df = pd.read_csv(RQ3_DIR / "summary.csv")
    df = df[df["family"] != "oracle"].reset_index(drop=True)
    aqs = json.loads((RQ3_DIR / "aqs_weights.json").read_text(encoding="utf-8"))
    return df, aqs


def _aqs(df, w):
    return (w[0] * df["T3/umbrela/mean"] + w[1] * df["T3/erag/mean"]
            + w[2] * df["T3/ragas_wa/mean"])


def _tern_xy(a, b, c):
    """Barycentric (a=umbrela bottom-left, b=erag bottom-right, c=ragas top)."""
    x = 0.5 * (2 * b + c) / (a + b + c)
    y = (np.sqrt(3) / 2) * c / (a + b + c)
    return x, y


def build_simplex(df, aqs) -> None:
    step = 0.05
    grid = np.arange(0, 1 + 1e-9, step)
    xs, ys, rhos = [], [], []
    for wu in grid:
        for we in grid:
            wr = 1 - wu - we
            if wr < -1e-9:
                continue
            wr = max(wr, 0.0)
            rho = spearmanr(_aqs(df, (wu, we, wr)), df["Recall@10"]).statistic
            x, y = _tern_xy(wu, we, wr)
            xs.append(x); ys.append(y); rhos.append(rho)

    rcparams_report()
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    sc = ax.scatter(xs, ys, c=rhos, cmap="viridis", s=60, marker="h",
                    edgecolor="none")
    # triangle edges
    A, B, C = _tern_xy(1, 0, 0), _tern_xy(0, 1, 0), _tern_xy(0, 0, 1)
    tri = np.array([A, B, C, A])
    ax.plot(tri[:, 0], tri[:, 1], color="black", linewidth=0.8)
    # vertex labels
    ax.annotate("UMBRELA", A, textcoords="offset points", xytext=(-6, -10),
                ha="right", fontsize=8)
    ax.annotate("eRAG", B, textcoords="offset points", xytext=(6, -10),
                ha="left", fontsize=8)
    ax.annotate("RAGAS-WA", C, textcoords="offset points", xytext=(0, 6),
                ha="center", fontsize=8)
    # data-driven optimum marker
    wf = aqs["weights_full"]
    opt = _tern_xy(wf["umbrela"], wf["erag"], wf["ragas_wa"])
    ax.scatter(*opt, marker="*", s=180, color=PALETTE["vermillion"],
               edgecolor="black", linewidth=0.6, zorder=5,
               label=f"data-driven optimum ($\\rho={aqs['rho_in_sample']:.2f}$)")
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(r"ranking $\rho$ vs Recall@10", fontsize=8.5)
    cb.ax.tick_params(labelsize=7.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.legend(loc="upper right", fontsize=7.5, bbox_to_anchor=(1.02, 1.08))
    fig.tight_layout()
    for p in save_figure(fig, "fig_rq3_aqs_weight_simplex"):
        print(f"[done] {p.relative_to(REPO_ROOT)}")
    plt.close(fig)


def build_table(aqs) -> None:
    wf = aqs["weights_full"]
    lw = aqs["weights_loo"]["mean"]
    ca = aqs["components_alone"]

    def trip(w):
        return f"({w['umbrela']:.2f}, {w['erag']:.2f}, {w['ragas_wa']:.2f})"

    rows = [
        ["eRAG alone", "(0.00, 1.00, 0.00)", f"{ca['erag']:.3f}"],
        ["AQS -- data-driven", trip(wf), f"{aqs['rho_in_sample']:.3f}"],
        ["AQS -- LOO mean", trip(lw), f"{aqs['rho_loo_mean']:.3f}"],
    ]
    out = pd.DataFrame(
        rows, columns=["Variant", "Weights (UMB, eRAG, RAGAS)", r"$\rho$ vs R@10"])
    note = (f"% components alone: UMBRELA rho={ca['umbrela']:.3f}, "
            f"eRAG={ca['erag']:.3f}, RAGAS-WA={ca['ragas_wa']:.3f}\n")
    tex = note + out.to_latex(index=False, escape=False, column_format="lcc")
    p = save_table(tex, "tab_rq3_aqs_weights")
    print(f"[done] {p.relative_to(REPO_ROOT)}")
    print(out.to_string(index=False))


def main() -> None:
    df, aqs = _load()
    build_simplex(df, aqs)
    build_table(aqs)


if __name__ == "__main__":
    main()
