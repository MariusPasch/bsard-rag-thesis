"""
Build the Tier 4 matched-pool dumbbell -- the headline visualization of Act 3.

Output: Report/figures/fig_rq1_t4_matched_pool_dumbbell.{pdf,png}

What's shown
------------
Three dumbbells, each comparing two systems on Recall@10. Two dots per row
joined by a line; Delta and p-value annotated.

  Row 1 -- T3-A             vs  T4.0 (hybrid, top-50)
              delta = +0.043, p = 0.016 -- LLM-Judge significantly lifts the
              best non-LLM pipeline.
  Row 2 -- T4.0 top-50      vs  CRAG  (hybrid, v2)
              delta = -0.019, p = 0.163 (ns) -- adding a CRAG correction
              loop does *not* significantly improve over the non-agentic
              LLM-Judge canonical.
  Row 3 -- T4.0 top-50      vs  ReAct (hybrid, v2)
              delta = -0.020, p = 0.222 (ns) -- same negative result for the
              multi-step ReAct agent.

The p-values for rows 2-3 come from each canonical T4 JSON's
`secondary_significance.vs_llm_rerank_binary_top50_hybrid_rrf_k60_test` block.
Row 1's p comes from the T4.0 top-50 JSON's `significance_vs_anchor`
(anchor = T3-A `hybrid_rrf_k60`).

Read this figure as the Act 3 punchline: the LLM scorer adds significant
value; the agentic mechanism on top of it does not.

Usage:
  .venv/Scripts/python Report/scripts/build_t4_matched_pool_dumbbell.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

REPORT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPORT_DIR))

from thesis_style import (
    SIGNIFICANCE_MARKER,
    SYSTEMS,
    fmt_pvalue,
    rcparams_report,
    save_figure,
)

OUTPUT_NAME = "fig_rq1_t4_matched_pool_dumbbell"
# REPORT_DIR is <project>/analysis/Report_Tables_Figures; the project root
# (holding output/) is two parents up.
REPO_ROOT   = REPORT_DIR.parent.parent
RESULTS_DIR = REPO_ROOT / "output" / "results"


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as fp:
        return json.load(fp)


def _r10(payload: dict) -> float:
    return float(payload["metrics"]["Recall@10"])


def main() -> None:
    # Source JSONs.
    t3a   = _load_json(RESULTS_DIR / "hybrid" / "hybrid_rrf_k60_test.json")
    t40_50 = _load_json(
        RESULTS_DIR / "agentic" / "llm_judge" / "llm_rerank"
        / "llm_rerank_binary_top50_hybrid_rrf_k60_test.json"
    )
    crag = _load_json(
        RESULTS_DIR / "agentic" / "CRAG" / "crag_hybrid_rrf_k60_test_v2.json"
    )
    react = _load_json(
        RESULTS_DIR / "agentic" / "ReAct" / "round2"
        / "react_hybrid_rrf_k60_test_v2.json"
    )

    r_t3a    = _r10(t3a)
    r_t40_50 = _r10(t40_50)
    r_crag   = _r10(crag)
    r_react  = _r10(react)

    # p-values.
    p_t40_vs_t3a = t40_50.get("significance_vs_anchor", {}).get("p_value_recall10")
    p_crag_vs_t40 = (crag.get("secondary_significance", {})
                     .get("vs_llm_rerank_binary_top50_hybrid_rrf_k60_test", {})
                     .get("p_value_recall10"))
    p_react_vs_t40 = (react.get("secondary_significance", {})
                      .get("vs_llm_rerank_binary_top50_hybrid_rrf_k60_test", {})
                      .get("p_value_recall10"))

    rows = [
        {
            "label": r"T3-A $\to$ T4.0 top-50",
            "x_start": r_t3a,    "x_end": r_t40_50,
            "color_start": SYSTEMS["hybrid_rrf_k60"].color,
            "color_end":   SYSTEMS["llm_rerank_top50"].color,
            "p": p_t40_vs_t3a,
        },
        {
            "label": r"T4.0 top-50 $\to$ CRAG (hyb., v2)",
            "x_start": r_t40_50, "x_end": r_crag,
            "color_start": SYSTEMS["llm_rerank_top50"].color,
            "color_end":   SYSTEMS["crag_hybrid_v2"].color,
            "p": p_crag_vs_t40,
        },
        {
            "label": r"T4.0 top-50 $\to$ ReAct (hyb., v2)",
            "x_start": r_t40_50, "x_end": r_react,
            "color_start": SYSTEMS["llm_rerank_top50"].color,
            "color_end":   SYSTEMS["react_hybrid_v2"].color,
            "p": p_react_vs_t40,
        },
    ]

    # ----- plot ---------------------------------------------------------
    rcparams_report()
    fig, ax = plt.subplots(figsize=(7.0, 2.8))

    ys = list(range(len(rows)))

    for y, r in zip(ys, rows):
        x0, x1 = r["x_start"], r["x_end"]
        # Connecting segment.
        ax.plot([x0, x1], [y, y], color="#888888", linewidth=1.2, zorder=1)
        # End dots.
        ax.scatter([x0], [y], s=70, color=r["color_start"],
                   edgecolor="black", linewidth=0.5, zorder=2)
        ax.scatter([x1], [y], s=70, color=r["color_end"],
                   edgecolor="black", linewidth=0.5, zorder=2)
        # Value labels at the dots.
        ax.text(x0, y + 0.18, f"{x0:.3f}",
                ha="center", va="bottom", fontsize=8, color="#555555")
        ax.text(x1, y + 0.18, f"{x1:.3f}",
                ha="center", va="bottom", fontsize=8, color="#555555")
        # Delta + p annotation under the line.
        delta = x1 - x0
        delta_str = f"$\\Delta = {delta:+.3f}$"
        p_str = ""
        if r["p"] is not None:
            p_str = f", $p = ${fmt_pvalue(r['p']).replace('$', '')}"
            if r["p"] < 0.05:
                p_str = p_str + " " + SIGNIFICANCE_MARKER
            else:
                p_str = p_str + " (ns)"
        mid_x = (x0 + x1) / 2
        ax.text(mid_x, y - 0.22, delta_str + p_str,
                ha="center", va="top", fontsize=8.5,
                color="#000000")

    ax.set_yticks(ys, [r["label"] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel(r"Recall@10 (test, $n=222$)")
    ax.set_xlim(0.38, 0.47)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", which="both", length=0, labelsize=9)

    fig.tight_layout()
    paths = save_figure(fig, OUTPUT_NAME)
    plt.close(fig)
    for p in paths:
        print(f"[done] Wrote {p.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
