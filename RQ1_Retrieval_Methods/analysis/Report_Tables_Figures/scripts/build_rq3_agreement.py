"""
Build the RQ3 system-ranking agreement artifacts (Chapter 14, Act 1 opener).

Outputs:
  Report/figures/fig_rq3_eval_corr_heatmap.{pdf,png}
      Heatmap of system-level ranking agreement (n=19 retrievers) between each
      autonomous evaluator (+ data-driven AQS) and the gold IR metrics. Color =
      Spearman rho (vlag, centred 0); each cell annotated rho (top) / Kendall
      tau (bottom).
  Report/tables/tab_rq3_system_ranking_agreement.tex
      Evaluators x {rho vs R@10, rho vs NDCG@10, doc-level accuracy, Cohen kappa
      vs gold qrels}. RAGAS-WA has no per-document label, so its doc-level cells
      are "--".

Reads the consolidated RQ3 records + summary.csv (system-level evaluator means
and gold metrics) and the per-query sidecars (document-level labels for the
accuracy / kappa columns).

Usage:
  .venv/Scripts/python Report/scripts/build_rq3_agreement.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import cohen_kappa_score

REPORT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPORT_DIR))

from thesis_style import (  # noqa: E402
    fmt_metric,
    rcparams_report,
    save_figure,
    save_table,
)

# REPORT_DIR is <project>/analysis/Report_Tables_Figures; the project root
# (holding output/ and evaluation/) is two parents up.
REPO_ROOT = REPORT_DIR.parent.parent
RQ3_DIR = REPO_ROOT / "output" / "results" / "RQ3"
SUBSET_FILE = REPO_ROOT / "evaluation" / "data" / "tier3_subset.json"

# evaluator column key in summary.csv -> display
EVAL_COLS = {
    "T3/umbrela/mean": "UMBRELA",
    "T3/erag/mean": "eRAG",
    "T3/ragas_wa/mean": "RAGAS-WA",
    "T3/AQS_dd": "AQS (data-driven)",
}
GOLD_COLS = ["Recall@10", "NDCG@10", "MAP@10", "NDCG@100", "Recall@100"]


def _load_frame() -> pd.DataFrame:
    summary = pd.read_csv(RQ3_DIR / "summary.csv")
    summary = summary[summary["family"] != "oracle"].reset_index(drop=True)
    records = {}
    for f in RQ3_DIR.glob("*_test.json"):
        d = json.loads(f.read_text(encoding="utf-8"))
        records[d["experiment_id"]] = d
    for col in ["MAP@10", "NDCG@100", "Recall@100"]:
        summary[col] = summary["experiment_id"].map(
            lambda e: records[e]["subset_metrics"]["metrics"].get(col))
    return summary


def _doc_level(metric_eval: str):
    """Return (accuracy, kappa) of a binarised evaluator vs gold qrels, pooled
    over all (system, query, top-10 doc). metric_eval in {umbrela, erag}."""
    records, sidecars = {}, {}
    for f in RQ3_DIR.glob("*_test.json"):
        d = json.loads(f.read_text(encoding="utf-8"))
        records[d["experiment_id"]] = d
        sidecars[d["experiment_id"]] = json.loads(
            (REPO_ROOT / d["per_query_sidecar"]).read_text(encoding="utf-8"))
    subset = json.loads(SUBSET_FILE.read_text(encoding="utf-8"))
    gold = {str(q["question_id"]): set(map(str, q["relevant_article_ids"]))
            for q in subset["questions"]}
    y_gold, y_pred = [], []
    for e, sc in sidecars.items():
        for qid, ranks in sc["ranks"].items():
            gset = gold.get(qid, set())
            for d in ranks[:10]:
                v = sc[metric_eval].get(qid, {}).get(d)
                if v is None:
                    continue
                pred = int(v >= 2) if metric_eval == "umbrela" else int(v)
                y_gold.append(int(d in gset))
                y_pred.append(pred)
    acc = float(np.mean(np.array(y_gold) == np.array(y_pred)))
    kappa = float(cohen_kappa_score(y_gold, y_pred))
    return acc, kappa


def build_heatmap(df: pd.DataFrame) -> None:
    rho = pd.DataFrame(index=list(EVAL_COLS.values()), columns=GOLD_COLS, dtype=float)
    tau = rho.copy()
    for ecol, ename in EVAL_COLS.items():
        for g in GOLD_COLS:
            rho.loc[ename, g] = spearmanr(df[ecol], df[g]).statistic
            tau.loc[ename, g] = kendalltau(df[ecol], df[g]).statistic

    rcparams_report()
    fig, ax = plt.subplots(figsize=(6.7, 2.9))
    data = rho.values.astype(float)
    im = ax.imshow(data, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(GOLD_COLS)), GOLD_COLS, fontsize=8.5)
    ax.set_yticks(range(len(EVAL_COLS)), list(EVAL_COLS.values()), fontsize=8.5)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            r, t = rho.iloc[i, j], tau.iloc[i, j]
            txt = ("white" if abs(r) > 0.6 else "black")
            ax.text(j, i, f"{r:.2f}\n$\\tau$ {t:.2f}", ha="center", va="center",
                    fontsize=7, color=txt)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(r"Spearman $\rho$", fontsize=8.5)
    cb.ax.tick_params(labelsize=7.5)
    ax.set_xlabel("gold IR metric", fontsize=9)
    ax.tick_params(length=0)
    fig.tight_layout()
    for p in save_figure(fig, "fig_rq3_eval_corr_heatmap"):
        print(f"[done] {p.relative_to(REPO_ROOT)}")
    plt.close(fig)


def build_table(df: pd.DataFrame) -> None:
    doc = {"umbrela": _doc_level("umbrela"), "erag": _doc_level("erag")}
    rows = []
    keymap = {"UMBRELA": "umbrela", "eRAG": "erag", "RAGAS-WA": "ragas_wa"}
    for ecol, ename in EVAL_COLS.items():
        rho_r10 = spearmanr(df[ecol], df["Recall@10"]).statistic
        rho_ndcg = spearmanr(df[ecol], df["NDCG@10"]).statistic
        dk = keymap.get(ename)
        acc, kap = doc.get(dk, (None, None))
        rows.append({
            "Evaluator": ename,
            r"$\rho$ vs R@10": fmt_metric("spearman_rho", rho_r10),
            r"$\rho$ vs NDCG@10": fmt_metric("spearman_rho", rho_ndcg),
            "Doc accuracy": "--" if acc is None else f"{acc:.3f}",
            r"Cohen $\kappa$": "--" if kap is None else f"{kap:.3f}",
        })
    out = pd.DataFrame(rows)
    tex = out.to_latex(index=False, escape=False, column_format="lcccc")
    p = save_table(tex, "tab_rq3_system_ranking_agreement")
    print(f"[done] {p.relative_to(REPO_ROOT)}")
    print(out.to_string(index=False))


def main() -> None:
    df = _load_frame()
    build_heatmap(df)
    build_table(df)


if __name__ == "__main__":
    main()
