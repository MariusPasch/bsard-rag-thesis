"""
Build the two RQ3 stratified-evaluator figures (Chapter 14).

Outputs:
  Report/figures/fig_rq3_eval_trust_strata.{pdf,png}
      Trustworthiness map. 2 rows (difficulty axis: n_gold, lexical alignment)
      x 3 cols (UMBRELA / eRAG / RAGAS-WA). Each panel plots, across the
      stratum buckets, the autonomous evaluator score and the gold Recall@10
      for the best sparse (BM25 tuned) and best dense (BGE-M3) systems, plus
      the gold-ceiling (evaluator score on perfect retrieval). The evaluator
      is a trustworthy proxy where its score-line TRACKS the R@10 line.

  Report/figures/fig_rq3_eval_strata_dumbbell.{pdf,png}
      Gold-ceiling headroom. Same 2x3 layout. Each panel draws, per bucket, a
      dumbbell from the achieved evaluator score (BGE-M3) up to the gold
      ceiling -- the gap is the attainable headroom. Negative gaps (achieved
      above ceiling) mark where evaluators credit retrieved-but-not-gold
      passages over the gold articles themselves.

Both read from the canonical RQ3 consolidated records + per-query sidecars and
the 48-question subset. System colors come from the locked SYSTEMS map; the
evaluator is an axis (one per column), not a series -- per STYLEGUIDE sec 4.

Usage:
  .venv/Scripts/python Report/scripts/build_rq3_eval_strata.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPORT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPORT_DIR))

from thesis_style import (  # noqa: E402
    EVALUATORS,
    PALETTE,
    SYSTEMS,
    rcparams_report,
    save_figure,
)

# REPORT_DIR is <project>/analysis/Report_Tables_Figures; the project root
# (holding output/ and evaluation/) is two parents up.
REPO_ROOT = REPORT_DIR.parent.parent
RQ3_DIR = REPO_ROOT / "output" / "results" / "RQ3"
SUBSET_FILE = REPO_ROOT / "evaluation" / "data" / "tier3_subset.json"

# Two systems carried through these figures: best sparse and best dense on the
# 48-q subset. Keys resolve to colors/markers via the locked SYSTEMS map.
SPARSE_KEY, DENSE_KEY = "bm25_tuned", "dense_bge_m3"
EVAL_ORDER = ["umbrela", "erag", "ragas_wa"]

# Difficulty axes (row = axis). Each entry: (row_title, ordered levels, labels).
NGOLD_LEVELS = ["1", "2", "3-5", "6-10", ">10"]
LEX_LEVELS = ["semantically_paraphrased", "middle", "lexically_aligned"]
LEX_LABELS = {"semantically_paraphrased": "paraphrased", "middle": "middle",
              "lexically_aligned": "aligned"}


# ---------------------------------------------------------------------------
# Data loading + per-query helpers
# ---------------------------------------------------------------------------
def _load() -> dict:
    records, sidecars = {}, {}
    for f in sorted(RQ3_DIR.glob("*_test.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        records[d["experiment_id"]] = d
    for e, r in records.items():
        sidecars[e] = json.loads((REPO_ROOT / r["per_query_sidecar"])
                                 .read_text(encoding="utf-8"))
    subset = json.loads(SUBSET_FILE.read_text(encoding="utf-8"))
    gold = {str(q["question_id"]): set(map(str, q["relevant_article_ids"]))
            for q in subset["questions"]}
    ngold = {str(q["question_id"]): q["n_relevant_articles"]
             for q in subset["questions"]}
    lex = {str(q["question_id"]): q["strata"]["lex_align"]
           for q in subset["questions"]}
    ceil_sc = json.loads((RQ3_DIR / "gold_ceiling_per_query.json")
                         .read_text(encoding="utf-8"))
    return dict(records=records, sidecars=sidecars, gold=gold, ngold=ngold,
                lex=lex, ceil=ceil_sc)


def _ngold_bucket(n: int) -> str:
    return ("1" if n == 1 else "2" if n == 2 else "3-5" if n <= 5
            else "6-10" if n <= 10 else ">10")


def _r10(sc, rid, q, gold) -> float:
    g = gold.get(q, set())
    if not g:
        return np.nan
    return len(set(sc[rid]["ranks"][q][:10]) & g) / len(g)


def _eval_score(per_q_map, q, ev) -> float:
    """Mean evaluator score for one query from a sidecar-shaped map."""
    if ev == "ragas_wa":
        v = per_q_map["ragas_wa"].get(q)
        return float(v) if v is not None else np.nan
    d = per_q_map[ev].get(q, {})
    if not d:
        return np.nan
    vals = list(d.values())
    if ev == "umbrela":          # 0-3 grade -> [0,1]
        return float(np.mean(vals)) / 3.0
    return float(np.mean(vals))  # erag already 0/1


def _bucket_means(D) -> dict:
    """Return nested dict: axis -> level -> {metric: value}.

    metrics: r10_<sys>, ev_<sys>, ceil  (per evaluator, computed separately).
    """
    sc, gold, ceil = D["sidecars"], D["gold"], D["ceil"]
    rid_sparse = SYSTEMS[SPARSE_KEY].result_id
    rid_dense = SYSTEMS[DENSE_KEY].result_id

    # group qids per axis level
    groups = {"ngold": defaultdict(list), "lex": defaultdict(list)}
    for q in gold:
        groups["ngold"][_ngold_bucket(D["ngold"][q])].append(q)
        groups["lex"][D["lex"][q]].append(q)

    out = {}
    for axis, levels in (("ngold", NGOLD_LEVELS), ("lex", LEX_LEVELS)):
        out[axis] = {}
        for lv in levels:
            qs = groups[axis][lv]
            cell = {"n": len(qs)}
            cell["r10_sparse"] = np.nanmean([_r10(sc, rid_sparse, q, gold) for q in qs])
            cell["r10_dense"] = np.nanmean([_r10(sc, rid_dense, q, gold) for q in qs])
            for ev in EVAL_ORDER:
                cell[f"ev_sparse_{ev}"] = np.nanmean(
                    [_eval_score(sc[rid_sparse], q, ev) for q in qs])
                cell[f"ev_dense_{ev}"] = np.nanmean(
                    [_eval_score(sc[rid_dense], q, ev) for q in qs])
                cell[f"ceil_{ev}"] = np.nanmean(
                    [_eval_score(ceil, q, ev) for q in qs])
            out[axis][lv] = cell
    return out


# ---------------------------------------------------------------------------
# Figure 1 -- trustworthiness map
# ---------------------------------------------------------------------------
def fig_trust(B) -> None:
    rcparams_report()
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.7), sharey=True)
    axis_rows = [("ngold", NGOLD_LEVELS, None, r"gold articles per question"),
                 ("lex", LEX_LEVELS, LEX_LABELS, r"question$\to$gold lexical alignment")]
    c_sp, c_de = SYSTEMS[SPARSE_KEY].color, SYSTEMS[DENSE_KEY].color
    c_ceil = EVALUATORS["gold_ceiling"].color

    for ri, (axis, levels, lblmap, xlab) in enumerate(axis_rows):
        xs = list(range(len(levels)))
        xticklabels = [lblmap[l] if lblmap else l for l in levels]
        for ci, ev in enumerate(EVAL_ORDER):
            ax = axes[ri, ci]
            cells = [B[axis][l] for l in levels]
            ev_sp = [c[f"ev_sparse_{ev}"] for c in cells]
            ev_de = [c[f"ev_dense_{ev}"] for c in cells]
            r10_sp = [c["r10_sparse"] for c in cells]
            r10_de = [c["r10_dense"] for c in cells]
            ceil = [c[f"ceil_{ev}"] for c in cells]
            # gold ceiling (reference)
            ax.plot(xs, ceil, color=c_ceil, linestyle=(0, (4, 2)), linewidth=1.3,
                    marker="", zorder=2, label="gold ceiling")
            # evaluator score (solid) per system
            ax.plot(xs, ev_sp, color=c_sp, linestyle="-", marker="D",
                    markersize=4, linewidth=1.6, zorder=4, label="BM25 tuned — eval")
            ax.plot(xs, ev_de, color=c_de, linestyle="-", marker="o",
                    markersize=4, linewidth=1.6, zorder=4, label="BGE-M3 — eval")
            # gold R@10 (dotted) per system
            ax.plot(xs, r10_sp, color=c_sp, linestyle=":", marker="D",
                    markersize=3, linewidth=1.2, alpha=0.85, zorder=3,
                    label="BM25 tuned — R@10")
            ax.plot(xs, r10_de, color=c_de, linestyle=":", marker="o",
                    markersize=3, linewidth=1.2, alpha=0.85, zorder=3,
                    label="BGE-M3 — R@10")
            ax.set_xticks(xs)
            ax.set_xticklabels(xticklabels, fontsize=7.5, rotation=0)
            ax.set_ylim(-0.02, 1.0)
            ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
            ax.set_axisbelow(True)
            if ri == 0:
                ax.set_title(EVALUATORS[ev].display.split(" (")[0], fontsize=9)
            if ci == 0:
                ax.set_ylabel("score", fontsize=9)
            ax.set_xlabel(xlab, fontsize=7.5)

    # one shared legend below
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False,
               fontsize=7.2, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    paths = save_figure(fig, "fig_rq3_eval_trust_strata")
    plt.close(fig)
    for p in paths:
        print(f"[done] {p.relative_to(REPO_ROOT)}")


# ---------------------------------------------------------------------------
# Figure 2 -- gold-ceiling headroom (dumbbell)
# ---------------------------------------------------------------------------
def fig_headroom(B) -> None:
    rcparams_report()
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.7), sharey=True)
    axis_rows = [("ngold", NGOLD_LEVELS, None, r"gold articles per question"),
                 ("lex", LEX_LEVELS, LEX_LABELS, r"question$\to$gold lexical alignment")]
    c_de = SYSTEMS[DENSE_KEY].color
    c_ceil = EVALUATORS["gold_ceiling"].color

    for ri, (axis, levels, lblmap, xlab) in enumerate(axis_rows):
        xs = list(range(len(levels)))
        xticklabels = [lblmap[l] if lblmap else l for l in levels]
        for ci, ev in enumerate(EVAL_ORDER):
            ax = axes[ri, ci]
            cells = [B[axis][l] for l in levels]
            achieved = [c[f"ev_dense_{ev}"] for c in cells]
            ceil = [c[f"ceil_{ev}"] for c in cells]
            for x, a, cl in zip(xs, achieved, ceil):
                up = cl >= a
                ax.plot([x, x], [a, cl],
                        color=("#888888" if up else PALETTE["vermillion"]),
                        linewidth=1.6, zorder=1)
            ax.scatter(xs, ceil, s=42, color=c_ceil, marker="_",
                       linewidth=1.8, zorder=3, label="gold ceiling")
            ax.scatter(xs, achieved, s=34, color=c_de, marker="o",
                       edgecolor="black", linewidth=0.4, zorder=4,
                       label="BGE-M3 achieved")
            # annotate signed headroom (ceiling - achieved)
            for x, a, cl in zip(xs, achieved, ceil):
                gap = cl - a
                ax.annotate(f"{gap:+.2f}", (x, max(a, cl) + 0.03),
                            ha="center", va="bottom", fontsize=6.3,
                            color=("#555555" if gap >= 0 else PALETTE["vermillion"]))
            ax.set_xticks(xs)
            ax.set_xticklabels(xticklabels, fontsize=7.5)
            ax.set_xlim(-0.5, len(levels) - 0.5)
            ax.set_ylim(-0.02, 1.0)
            ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
            ax.set_axisbelow(True)
            if ri == 0:
                ax.set_title(EVALUATORS[ev].display.split(" (")[0], fontsize=9)
            if ci == 0:
                ax.set_ylabel("evaluator score", fontsize=9)
            ax.set_xlabel(xlab, fontsize=7.5)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
               fontsize=7.5, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    paths = save_figure(fig, "fig_rq3_eval_strata_dumbbell")
    plt.close(fig)
    for p in paths:
        print(f"[done] {p.relative_to(REPO_ROOT)}")


def _print_numbers(B) -> None:
    print("\n=== verified bucket numbers (achieved = mean over bucket queries) ===")
    for axis, levels, lblmap in (("ngold", NGOLD_LEVELS, None),
                                 ("lex", LEX_LEVELS, LEX_LABELS)):
        print(f"\n[{axis}]")
        for lv in levels:
            c = B[axis][lv]
            lab = lblmap[lv] if lblmap else lv
            row = (f"  {lab:12} n={c['n']:2d}  "
                   f"R@10 bm25t/bge={c['r10_sparse']:.2f}/{c['r10_dense']:.2f}")
            for ev in EVAL_ORDER:
                row += (f" | {ev[:3]} bge={c[f'ev_dense_{ev}']:.2f}"
                        f" ceil={c[f'ceil_{ev}']:.2f}")
            print(row)


def main() -> None:
    D = _load()
    B = _bucket_means(D)
    _print_numbers(B)
    fig_trust(B)
    fig_headroom(B)


if __name__ == "__main__":
    main()
