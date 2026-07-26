"""
Build the RQ3 stratified-agreement heatmap (Chapter 14, Act 1 quantification).

Output:
  Report/figures/fig_rq3_eval_strata_heatmap.{pdf,png}
      Pooled per-query Spearman rho between each evaluator score and gold
      Recall@10, computed across all (system, query) pairs within a stratum
      (n=19 systems). Rows = strata (gold-count / lexical-alignment /
      cross-reference levels); cols = UMBRELA / eRAG / RAGAS-WA. RdBu_r, centred
      0. This is the 19-system quantification behind fig_rq3_eval_trust_strata.

Source: per-query sidecars in output/results/RQ3/ + tier3_subset.json strata.

Usage:
  .venv/Scripts/python Report/scripts/build_rq3_strata_heatmap.py
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

from thesis_style import rcparams_report, save_figure  # noqa: E402

# REPORT_DIR is <project>/analysis/Report_Tables_Figures; the project root
# (holding output/ and evaluation/) is two parents up.
REPO_ROOT = REPORT_DIR.parent.parent
RQ3_DIR = REPO_ROOT / "output" / "results" / "RQ3"
SUBSET_FILE = REPO_ROOT / "evaluation" / "data" / "tier3_subset.json"

EVAL_ORDER = [("umb", "UMBRELA"), ("erg", "eRAG"), ("rag", "RAGAS-WA")]
# (axis_key, level_value, row_label) in display order
ROWS = [
    ("article_count", "single_article", "gold: single"),
    ("article_count", "multi_article", r"gold: multi ($\geq$2)"),
    ("lex_align", "semantically_paraphrased", "lex: paraphrased"),
    ("lex_align", "middle", "lex: middle"),
    ("lex_align", "lexically_aligned", "lex: aligned"),
    ("cross_ref", "without_cross_refs", "xref: without"),
    ("cross_ref", "with_cross_refs", "xref: with"),
]


def _per_query_frame() -> pd.DataFrame:
    records, sidecars = {}, {}
    for f in RQ3_DIR.glob("*_test.json"):
        d = json.loads(f.read_text(encoding="utf-8"))
        records[d["experiment_id"]] = d
        sidecars[d["experiment_id"]] = json.loads(
            (REPO_ROOT / d["per_query_sidecar"]).read_text(encoding="utf-8"))
    subset = json.loads(SUBSET_FILE.read_text(encoding="utf-8"))
    gold = {str(q["question_id"]): set(map(str, q["relevant_article_ids"]))
            for q in subset["questions"]}
    strata = {str(q["question_id"]): q["strata"] for q in subset["questions"]}

    rows = []
    for e, sc in sidecars.items():
        for qid, ranks in sc["ranks"].items():
            g = gold.get(qid, set())
            if not g:
                continue
            umb = sc["umbrela"].get(qid, {})
            erg = sc["erag"].get(qid, {})
            rows.append({
                "qid": qid,
                "R10": len(set(ranks[:10]) & g) / len(g),
                "umb": np.mean([v / 3 for v in umb.values()]) if umb else np.nan,
                "erg": np.mean(list(erg.values())) if erg else np.nan,
                "rag": sc["ragas_wa"].get(qid, np.nan),
                "article_count": strata[qid]["article_count"],
                "lex_align": strata[qid]["lex_align"],
                "cross_ref": strata[qid]["cross_ref"],
            })
    return pd.DataFrame(rows)


def main() -> None:
    pq = _per_query_frame()
    mat = np.full((len(ROWS), len(EVAL_ORDER)), np.nan)
    for i, (axis, level, _) in enumerate(ROWS):
        sub = pq[pq[axis] == level]
        for j, (col, _) in enumerate(EVAL_ORDER):
            x = sub[col].dropna()
            y = sub["R10"].loc[x.index]
            if len(x) >= 5 and x.std() > 0 and y.std() > 0:
                mat[i, j] = spearmanr(x, y).statistic

    rcparams_report()
    fig, ax = plt.subplots(figsize=(4.3, 3.6))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-0.7, vmax=0.7, aspect="auto")
    ax.set_xticks(range(len(EVAL_ORDER)), [n for _, n in EVAL_ORDER], fontsize=8.5)
    ax.set_yticks(range(len(ROWS)), [r for *_, r in ROWS], fontsize=8.5)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color=("white" if abs(v) > 0.5 else "black"))
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(r"pooled per-query $\rho$ vs R@10", fontsize=8)
    cb.ax.tick_params(labelsize=7.5)
    ax.tick_params(length=0)
    fig.tight_layout()
    for p in save_figure(fig, "fig_rq3_eval_strata_heatmap"):
        print(f"[done] {p.relative_to(REPO_ROOT)}")
    plt.close(fig)


if __name__ == "__main__":
    main()
