"""Build the Arm-1 (Naive chunking) retrieved-article deep-dive.

Reads only the persisted one-path artifacts (no retrieval rerun). Writes:
  - per-question + coverage CSVs and summary CSV/MD -> RQ2_T06_ARM_RESULTS/data/tables/
  - report-ready .tex (tab_rq2_arm1_*) and figures (fig_rq2_arm1_*) -> Report/

Run via the T03 venv (has data_loader + evaluation + pandas/mpl/seaborn):
  RQ2_T03_ARM1_NAIVE/.venv/Scripts/python.exe scripts/run_arm1_deep_dive.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# --- path injection (mirror run_consolidate.py) ----------------------------
_THIS = Path(__file__).resolve()
_T06 = _THIS.parents[1]
_RQ2 = _T06.parent
for _src in sorted(_RQ2.glob("RQ2_T0*/src")):
    sys.path.insert(0, str(_src))
sys.path.insert(0, str(_T06 / "src"))
sys.path.insert(0, str(_RQ2 / "Report"))           # thesis_style

from arm_results import arm1_deep_dive as dd  # noqa: E402
from arm_results import loaders, paths  # noqa: E402
import thesis_style as ts  # noqa: E402

TABLES = paths.TABLES_DIR
GREEN = ts.PALETTE["green"]
GREY = ts.PALETTE["grey_mid"]


# ---------------------------------------------------------------------------
# Markdown helper (no tabulate dependency)
# ---------------------------------------------------------------------------

def to_md(df: pd.DataFrame, *, floatfmt: str = "{:.3f}") -> str:
    def cell(v):
        if isinstance(v, float):
            return "--" if np.isnan(v) else floatfmt.format(v)
        return str(v)
    cols = list(df.columns)
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = "\n".join("| " + " | ".join(cell(v) for v in row) + " |"
                     for row in df.itertuples(index=False))
    return "\n".join([head, sep, body])


def dump(df: pd.DataFrame, name: str, *, floatfmt: str = "{:.3f}") -> None:
    df.to_csv(TABLES / f"{name}.csv", index=False)
    (TABLES / f"{name}.md").write_text(to_md(df, floatfmt=floatfmt), encoding="utf-8")
    print(f"  wrote {name}.csv / .md  ({len(df)} rows)")


# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------

def compute() -> dict:
    long = ts.load_long_dataframe()
    extraction = dd.load_extraction_status()
    gt_by_stem: dict[str, dict] = {}

    pq_parts, cov_parts = [], []
    for stem in paths.STEMS:
        art = dd.load_stem(stem)
        gt_by_stem[stem] = art.gt
        pq_parts.append(dd.per_question_table(art))
        cov_parts.append(dd.coverage_long_table(art))
        print(f"  {stem} ({art.label}): {len(art.gt)} GT q, "
              f"{len(art.results)} with results, weights={'yes' if art.weights else 'NO'}")

    pq = pd.concat(pq_parts, ignore_index=True)
    cov = pd.concat([c for c in cov_parts if not c.empty], ignore_index=True)
    pq = dd.annotate_strata(pq, gt_by_stem, extraction)

    per_pdf = dd.per_pdf_summary(pq, cov, long)
    agg = dd.aggregate_summary(pq, cov, per_pdf, long)
    strata = dd.strata_summary(pq, cov)
    return {"long": long, "pq": pq, "cov": cov, "per_pdf": per_pdf,
            "agg": agg, "strata": strata}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_coverage_dist(cov: pd.DataFrame) -> None:
    ts.rcparams_report()
    touched = cov[cov["touched@10"] > 0]["coverage@10"].to_numpy()
    fig, ax = plt.subplots(figsize=ts.FIGSIZE["report_onecol"])
    ax.hist(touched, bins=20, range=(0, 1), color=GREEN, edgecolor="white", alpha=0.9)
    ax.axvline(dd.FRAGMENT_TAU, color=ts.PALETTE["vermillion"], ls="--", lw=1.2,
               label=f"fragment < {dd.FRAGMENT_TAU}")
    ax.set_xlabel("Content coverage of gold article")
    ax.set_ylabel("Gold-article instances")
    ax.legend(loc="upper center")
    ts.save_figure(fig, "fig_rq2_arm1_coverage_dist")
    plt.close(fig)


def fig_rank_of_gold_cdf(pq: pd.DataFrame) -> None:
    ts.rcparams_report()
    r = pq["first_hit_rank"].dropna().to_numpy()
    r = np.sort(r)
    y = np.arange(1, len(r) + 1) / len(pq)          # denom = all q (misses flatten tail)
    fig, ax = plt.subplots(figsize=ts.FIGSIZE["report_onecol"])
    ax.step(r, y, where="post", color=GREEN, lw=1.6)
    med = np.median(r)
    ax.axvline(med, color=GREY, ls=":", lw=1.0)
    ax.text(med + 1.5, 0.1, f"median = {med:.0f}", color=GREY, fontsize=8)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Rank of first gold article")
    ax.set_ylabel("Cum. fraction of questions")
    ts.save_figure(fig, "fig_rq2_arm1_rank_of_gold_cdf")
    plt.close(fig)


# NOTE: the per-PDF precision figure was superseded by the cross-arm
# precision-vs-ceiling figure (Report/scripts/build_precision_ceiling_per_pdf.py).


def fig_recall_vs_coverage(per_pdf: pd.DataFrame) -> None:
    ts.rcparams_report()
    d = per_pdf.sort_values("n_q", ascending=False)
    y = np.arange(len(d))
    h = 0.38
    fig, ax = plt.subplots(figsize=ts.FIGSIZE["report_wide"])
    ax.barh(y + h / 2, d["recall@10"], height=h, color=GREEN, alpha=0.95,
            label="Binary article recall@10")
    ax.barh(y - h / 2, d["coverage@10_mean_touched"], height=h, color=GREEN,
            alpha=0.45, hatch="///", edgecolor="white",
            label="Mean content coverage of retrieved gold (top-10 chunks)")
    ax.set_yticks(y)
    ax.set_yticklabels(d["stem_label"])
    ax.invert_yaxis()
    ax.set_xlabel("Score")
    ax.set_xlim(0, 1)
    ax.legend(loc="lower right", fontsize=8, frameon=True, framealpha=0.92,
              facecolor="white", edgecolor="#DDDDDD")
    ts.save_figure(fig, "fig_rq2_arm1_recall_vs_coverage")
    plt.close(fig)


def fig_strata(strata: pd.DataFrame, cov: pd.DataFrame, pq: pd.DataFrame) -> None:
    ts.rcparams_report()
    fig, axes = plt.subplots(1, 2, figsize=ts.FIGSIZE["report_wide"], sharey=True)
    card = strata[strata["stratum"] == "cardinality"].set_index("level")
    card = card.reindex([x for x in ["single", "multi"] if x in card.index])
    axes[0].bar(card.index, card["recall@10"], color=GREEN, alpha=0.9,
                width=0.6)
    for i, (lev, row) in enumerate(card.iterrows()):
        axes[0].text(i, row["recall@10"] + 0.01, f"{row['recall@10']:.3f}\n(n={int(row['n_q'])})",
                     ha="center", fontsize=8)
    axes[0].set_title("by GT cardinality")
    axes[0].set_ylabel("Recall@10 (binary, full GT)")

    order = ["FOUND", "PARTIAL", "NOT_FOUND", "UNKNOWN"]
    ext = strata[strata["stratum"] == "extraction_class"].set_index("level")
    ext = ext.reindex([x for x in order if x in ext.index])
    axes[1].bar(ext.index, ext["recall@10"], color=GREEN, alpha=0.9, width=0.6)
    for i, (lev, row) in enumerate(ext.iterrows()):
        axes[1].text(i, row["recall@10"] + 0.01, f"{row['recall@10']:.3f}\n(n={int(row['n_q'])})",
                     ha="center", fontsize=8)
    axes[1].set_title("by extraction status of gold")
    axes[1].tick_params(axis="x", rotation=20)
    for ax in axes:
        ax.set_ylim(0, 1)
    ts.save_figure(fig, "fig_rq2_arm1_strata_smallmultiples")
    plt.close(fig)


# ---------------------------------------------------------------------------
# .tex tables (report-ready)
# ---------------------------------------------------------------------------

def tex_per_pdf(per_pdf: pd.DataFrame) -> None:
    cols = ["stem_label", "n_q", "recall@5", "recall@10", "E/recall@10", "hit@10",
            "recall@20", "recall@100", "precision@10", "distinct_articles@10",
            "median_first_hit_rank"]
    d = per_pdf[cols].copy()
    d = d.rename(columns={
        "stem_label": "PDF (law type)", "n_q": "n",
        "recall@5": "R@5", "recall@10": "R@10", "E/recall@10": "E/R@10",
        "hit@10": "Hit@10", "recall@20": "R@20", "recall@100": "R@100",
        "precision@10": "P@10 (chunks)", "distinct_articles@10": "distinct@10",
        "median_first_hit_rank": "med. rank",
    })
    for c in d.columns:
        if c not in ("PDF (law type)", "n"):
            d[c] = d[c].map(lambda v: f"{v:.3f}" if pd.notna(v) else "--")
    ts.save_table(d, "tab_rq2_arm1_per_pdf")


def tex_coverage(per_pdf: pd.DataFrame) -> None:
    cols = ["stem_label", "n_q", "recall@10", "coverage@10_mean_touched",
            "pct_fragment_of_touched@10", "pct_near_complete_of_touched@10",
            "chunks_per_article@10", "chunks_to_cover_gold_median"]
    d = per_pdf[cols].copy().rename(columns={
        "stem_label": "PDF (law type)", "n_q": "n", "recall@10": "R@10 (binary)",
        "coverage@10_mean_touched": "coverage (touched)",
        "pct_fragment_of_touched@10": "\\% fragment", "pct_near_complete_of_touched@10": "\\% near-complete",
        "chunks_per_article@10": "chunks/article", "chunks_to_cover_gold_median": "chunks to cover",
    })
    for c in d.columns:
        if c not in ("PDF (law type)", "n"):
            d[c] = d[c].map(lambda v: f"{v:.2f}" if pd.notna(v) else "--")
    ts.save_table(d, "tab_rq2_arm1_coverage")


def tex_strata(strata: pd.DataFrame) -> None:
    d = strata[["stratum", "level", "n_q", "recall@10", "hit@10",
                "median_first_hit_rank", "coverage@10_mean"]].copy()
    d = d.rename(columns={"stratum": "Stratum", "level": "Level", "n_q": "n",
                          "recall@10": "R@10", "hit@10": "Hit@10",
                          "median_first_hit_rank": "med. rank",
                          "coverage@10_mean": "coverage@10"})
    for c in ("R@10", "Hit@10", "med. rank", "coverage@10"):
        d[c] = d[c].map(lambda v: f"{v:.3f}" if pd.notna(v) else "--")
    ts.save_table(d, "tab_rq2_arm1_strata")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    paths.ensure_output_dirs()
    print("Computing Arm-1 deep-dive over the curated 5-PDF set ...")
    R = compute()

    print("Writing tables (CSV + MD) ...")
    dump(R["pq"], "arm1_per_question")
    dump(R["cov"], "arm1_coverage_per_gold_article")
    dump(R["per_pdf"], "arm1_per_pdf_summary")
    dump(R["agg"], "arm1_aggregate_summary")
    dump(R["strata"], "arm1_strata_summary")

    print("Writing report-ready .tex ...")
    tex_per_pdf(R["per_pdf"])
    tex_coverage(R["per_pdf"])
    tex_strata(R["strata"])

    print("Building figures ...")
    # All Arm-1-only figures were retired in the figure review and moved to
    # cross-arm builders in Report/scripts/: precision ->
    # build_precision_ceiling_per_pdf.py; cardinality strata ->
    # build_recall_by_cardinality_byarm.py. (coverage_dist, rank_of_gold_cdf,
    # recall_vs_coverage deleted outright.)

    # console headline
    agg = R["agg"]
    micro = agg[agg["aggregation"] == "micro"].iloc[0]
    print("\n=== HEADLINE (micro) ===")
    print(f"  n_q={int(micro['n_q'])}  recall@10={micro['recall@10']:.3f} "
          f"[{micro['recall@10_ci_lo']:.3f}, {micro['recall@10_ci_hi']:.3f}]  "
          f"E/recall@10={micro['E/recall@10']:.3f}")
    print(f"  coverage@10(touched)={micro['coverage@10_mean_touched']:.3f}  "
          f"%fragment={micro['pct_fragment_of_touched@10']:.1f}  "
          f"%near-complete={micro['pct_near_complete_of_touched@10']:.1f}")
    print("Done.")


if __name__ == "__main__":
    main()
