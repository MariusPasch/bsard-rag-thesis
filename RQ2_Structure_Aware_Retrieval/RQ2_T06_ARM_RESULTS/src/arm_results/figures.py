"""Thesis figures from the consolidated cross-arm metrics.

Pure presentation: reads the macro/headline frame produced by
:mod:`arm_results.tables` and renders grouped bar charts. Matplotlib only
(no seaborn dependency required at runtime).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import paths
from .tables import HEADLINE_METRICS, _macro_micro, _resolve_headline


def plot_headline_bars(
    long_df: pd.DataFrame, out_dir: Path | None = None, *, weighting: str = "micro"
) -> Path:
    """Grouped bar chart: each method's R@10 / R@100 / MRR@10 / nDCG@10.

    ``weighting`` selects the macro (per-PDF mean) or micro (question-weighted)
    aggregate. Returns the PNG path; an SVG twin is written alongside.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = out_dir or paths.FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    macro, micro = _macro_micro(_resolve_headline(long_df))
    agg = micro if weighting == "micro" else macro
    metric_cols = [d for d, _ in HEADLINE_METRICS]

    methods = agg["method"].tolist()
    x = range(len(metric_cols))
    n = len(methods)
    width = 0.8 / max(n, 1)

    fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(metric_cols) * n / 3), 5))
    for i, method in enumerate(methods):
        vals = [agg.loc[agg["method"] == method, m].iloc[0] for m in metric_cols]
        offsets = [xi + (i - (n - 1) / 2) * width for xi in x]
        ax.bar(offsets, vals, width=width, label=method)

    ax.set_xticks(list(x))
    ax.set_xticklabels(metric_cols)
    ax.set_ylabel("score")
    ax.set_ylim(0, 1)
    ax.set_title(f"Cross-arm retrieval metrics ({weighting} average over 5 PDFs)")
    ax.legend(fontsize="small", ncol=1, bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.tight_layout()

    png = out_dir / f"cross_arm_headline_{weighting}.png"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    fig.savefig(png.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    return png
