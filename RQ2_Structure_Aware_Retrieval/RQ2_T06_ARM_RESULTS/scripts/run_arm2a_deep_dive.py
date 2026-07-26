"""Build the Arm-2A (Metadata) retrieved-article deep-dive.

Reads only the persisted one-path artifacts (rankings, faiss_meta node index,
BSARD DB article text, GT, extraction status, cross_arm_long.csv). No retrieval,
indexing or linking is rerun. Writes:
  - per-question, coverage, node→bsard weight, missed-gold CSVs + summary CSV/MD
    -> RQ2_T06_ARM_RESULTS/data/tables/
  - report-ready .tex (tab_rq2_arm2a_*) and figures (fig_rq2_arm2a_*) -> Report/

Run via the T03 venv (data_loader + transformers + e5 tokenizer + mpl):
  RQ2_T03_ARM1_NAIVE/.venv/Scripts/python.exe scripts/run_arm2a_deep_dive.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# --- path injection (mirror run_arm1_deep_dive.py) -------------------------
_THIS = Path(__file__).resolve()
_T06 = _THIS.parents[1]
_RQ2 = _T06.parent
for _src in sorted(_RQ2.glob("RQ2_T0*/src")):
    sys.path.insert(0, str(_src))
sys.path.insert(0, str(_T06 / "src"))
sys.path.insert(0, str(_RQ2 / "Report"))           # thesis_style

from arm_results import arm2a_deep_dive as dd  # noqa: E402
from arm_results import deep_dive_common as common, paths  # noqa: E402
import thesis_style as ts  # noqa: E402

TABLES = paths.TABLES_DIR
ORANGE = ts.PALETTE["orange"]
GREY = ts.PALETTE["grey_mid"]
GREEN = ts.PALETTE["green"]          # Arm-1 reference overlays
ARM1_COVERAGE_TOUCHED = 0.806        # Arm-1 micro reference (reports/arm1_naive_deep_dive.md §3)
ARM1_FRAGMENT_PCT = 18.0
ARM1_NEAR_COMPLETE_PCT = 64.0


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
    extraction = common.load_extraction_status()
    gt_by_stem: dict[str, dict] = {}

    pq_parts, cov_parts, w_parts, miss_parts = [], [], [], []
    art_node_sums: list[dict] = []   # node-union sanity rows
    for stem in paths.STEMS:
        sd = dd.load_stem(stem)
        gt_by_stem[stem] = sd.gt
        W = dd.build_weights(sd)
        w_parts.append(W.table)
        pq_parts.append(dd.per_question_table(sd))
        cov_parts.append(dd.coverage_long_table(sd, W))
        miss_parts.append(dd.missed_gold_table(sd, extraction))
        # node-union vs DB coverage sanity per article
        for b, s in W.art_node_token_sum.items():
            art_node_sums.append({"stem": stem, "bsard_id": b,
                                  "db_coverage_sum": W.db_coverage_by_article.get(b, np.nan)})
        print(f"  {stem} ({sd.label}): {len(sd.gt)} GT q | "
              f"nodes={len(sd.node_table)} | present(node-idx)={len(sd.present[dd.NODE_TEXT_METHOD])} | "
              f"weight rows={len(W.table)}")

    pq = pd.concat(pq_parts, ignore_index=True)
    cov = pd.concat([c for c in cov_parts if not c.empty], ignore_index=True)
    weights = pd.concat(w_parts, ignore_index=True)
    missed = pd.concat([m for m in miss_parts if not m.empty], ignore_index=True)
    art_node = pd.DataFrame(art_node_sums)

    pq = common.annotate_strata(pq, gt_by_stem, extraction)

    chk = dd.assert_published_recall(pq, long)
    print(f"  [assert] recomputed article R@k reproduces published T1/R@k: "
          f"{chk['n_cells'] - chk['n_drift']}/{chk['n_cells']} cells exact (<=1e-6); "
          f"max dev={chk['max_dev']:.2e} (node-index resync drift), "
          f"anchors R@5/R@100 max dev={chk['max_anchor_dev']:.2e}")
    max_dev = chk["max_dev"]

    per_pdf = dd.per_pdf_summary(pq, cov, long)
    agg = dd.aggregate_summary(pq, cov, per_pdf, long)
    strata = dd.strata_summary(pq, cov)
    miss_sum = dd.entire_miss_summary(missed, pq)
    return {"long": long, "pq": pq, "cov": cov, "weights": weights, "missed": missed,
            "art_node": art_node, "per_pdf": per_pdf, "agg": agg, "strata": strata,
            "miss_sum": miss_sum, "max_dev": max_dev}


# ---------------------------------------------------------------------------
# Figures (Arm 2A = orange; Arm-1 reference in green where contrasted)
# ---------------------------------------------------------------------------

def fig_coverage_dist(cov: pd.DataFrame) -> None:
    """Node coverage-% distribution: summary_node vs raw_node overlay, with the
    Arm-1 fragment line for reference."""
    ts.rcparams_report()
    fig, ax = plt.subplots(figsize=ts.FIGSIZE["report_onecol"])
    for variant, color, alpha, hatch in [
        ("T04_summary_node_hybrid", ORANGE, 0.85, None),
        ("T04_raw_node_hybrid", GREY, 0.55, "///"),
    ]:
        t = cov[(cov["variant"] == variant) & (cov["touched@10"] > 0)]["coverage_db@10"].to_numpy()
        ax.hist(t, bins=20, range=(0, 1), color=color, alpha=alpha, hatch=hatch,
                edgecolor="white", label=dd.SHORT[variant])
    ax.axvline(0.5, color=ts.PALETTE["vermillion"], ls="--", lw=1.1, label="fragment < 0.5")
    ax.set_xlabel("Node token-coverage of gold article (DB-overlap, top-10 nodes)")
    ax.set_ylabel("Gold-article instances")
    ax.legend(loc="upper center", fontsize=7)
    ts.save_figure(fig, "fig_rq2_arm2a_coverage_dist")
    plt.close(fig)


def fig_recall_vs_coverage(per_pdf: pd.DataFrame) -> None:
    """Per-PDF binary recall@10 vs mean DB-coverage of retrieved gold (summary_node)."""
    ts.rcparams_report()
    d = per_pdf[per_pdf["variant"] == dd.PRIMARY].sort_values("n_q", ascending=False)
    y = np.arange(len(d))
    h = 0.38
    fig, ax = plt.subplots(figsize=ts.FIGSIZE["report_wide"])
    ax.barh(y + h / 2, d["recall@10"], height=h, color=ORANGE, alpha=0.95,
            label="Binary article recall@10")
    ax.barh(y - h / 2, d["coverage_db@10_touched"], height=h, color=ORANGE,
            alpha=0.45, hatch="///", edgecolor="white",
            label="Mean node coverage of retrieved gold (top-10 nodes)")
    ax.set_yticks(y)
    ax.set_yticklabels(d["stem_label"])
    ax.invert_yaxis()
    ax.set_xlabel("Score")
    ax.set_xlim(0, 1)
    ax.legend(loc="lower right", fontsize=8, frameon=True, framealpha=0.92,
              facecolor="white", edgecolor="#DDDDDD")
    ts.save_figure(fig, "fig_rq2_arm2a_recall_vs_coverage")
    plt.close(fig)


def fig_node_vs_article(agg: pd.DataFrame) -> None:
    """Granularity contrast: recall@10, coverage, fragment-rate, entire-miss for
    the node (summary) vs article (raw_article) unit (micro)."""
    ts.rcparams_report()
    m = agg[agg["aggregation"] == "micro"].set_index("variant")
    variants = ["T04_summary_node_hybrid", "T04_raw_node_hybrid", "T04_raw_article_hybrid"]
    labels = [dd.SHORT[v] for v in variants]
    metrics = [("recall@10", "Recall@10"), ("coverage_db@10_touched", "Coverage (touched)"),
               ("pct_fragment_db@10", "% fragment")]
    fig, axes = plt.subplots(1, 3, figsize=ts.FIGSIZE["report_wide"])
    for ax, (col, title) in zip(axes, metrics):
        vals = [m.loc[v, col] if v in m.index else np.nan for v in variants]
        if col == "pct_fragment_db@10":
            vals = [x / 100.0 for x in vals]
            title = "Fragment rate"
        ax.bar(labels, vals, color=ORANGE, alpha=0.9, width=0.6)
        for i, x in enumerate(vals):
            ax.text(i, (x or 0) + 0.01, f"{x:.3f}", ha="center", fontsize=7)
        ax.set_title(title, fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis="x", rotation=20, labelsize=7)
    ts.save_figure(fig, "fig_rq2_arm2a_node_vs_article")
    plt.close(fig)


def fig_entire_miss_extraction(missed: pd.DataFrame) -> None:
    """Missed-gold (top-100) by extraction status, ABSENT vs BEYOND_POOL, summary_node."""
    ts.rcparams_report()
    gm = missed[missed["variant"] == dd.PRIMARY]
    statuses = ["FOUND", "PARTIAL", "NOT_FOUND", "UNKNOWN"]
    statuses = [s for s in statuses if (gm["article_status"] == s).any()]
    absent = [int(((gm["article_status"] == s) & (gm["miss_class"] == "ABSENT")).sum()) for s in statuses]
    beyond = [int(((gm["article_status"] == s) & (gm["miss_class"] == "BEYOND_POOL")).sum()) for s in statuses]
    x = np.arange(len(statuses))
    fig, ax = plt.subplots(figsize=ts.FIGSIZE["report_wide"])
    ax.bar(x, beyond, color=ORANGE, alpha=0.9, label="BEYOND_POOL (present, ranked > 100)")
    ax.bar(x, absent, bottom=beyond, color=ts.PALETTE["vermillion"], alpha=0.9,
           label="ABSENT (no node maps to article)")
    ax.set_xticks(x)
    ax.set_xticklabels(statuses)
    ax.set_ylabel("Missed gold articles (top-100)")
    ax.set_xlabel("Extraction status of the missed gold article")
    ax.legend(loc="upper right", fontsize=8)
    ts.save_figure(fig, "fig_rq2_arm2a_entire_miss_extraction")
    plt.close(fig)


def fig_strata(strata: pd.DataFrame) -> None:
    """Small-multiples: recall@10 by cardinality and extraction_class for the 3 variants."""
    ts.rcparams_report()
    fig, axes = plt.subplots(1, 2, figsize=ts.FIGSIZE["report_wide"], sharey=True)
    variants = ["T04_summary_node_hybrid", "T04_raw_node_hybrid", "T04_raw_article_hybrid"]
    width = 0.26
    for ax, (stratum, order) in zip(
            axes, [("cardinality", ["single", "multi"]),
                   ("extraction_class", ["FOUND", "PARTIAL", "NOT_FOUND", "UNKNOWN"])]):
        sub = strata[strata["stratum"] == stratum]
        levels = [l for l in order if (sub["level"] == l).any()]
        x = np.arange(len(levels))
        for j, v in enumerate(variants):
            vals = [sub[(sub["variant"] == v) & (sub["level"] == l)]["recall@10"].mean() for l in levels]
            ax.bar(x + (j - 1) * width, vals, width=width, alpha=0.9,
                   color=ORANGE if "summary" in v else (GREY if "raw_node" in v else GREEN),
                   label=dd.SHORT[v] if stratum == "cardinality" else None)
        ax.set_xticks(x)
        ax.set_xticklabels(levels, rotation=15, fontsize=7)
        ax.set_title(f"by {stratum}", fontsize=9)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Recall@10 (binary, full GT)")
    axes[0].legend(fontsize=7, loc="upper right")
    ts.save_figure(fig, "fig_rq2_arm2a_strata_smallmultiples")
    plt.close(fig)


# ---------------------------------------------------------------------------
# .tex tables
# ---------------------------------------------------------------------------

def tex_per_pdf(per_pdf: pd.DataFrame) -> None:
    d = per_pdf[per_pdf["variant"] == dd.PRIMARY][
        ["stem_label", "n_q", "recall@5", "recall@10", "E/recall@10", "hit@10",
         "recall@20", "recall@100", "precision@10", "distinct_articles@10",
         "median_first_hit_rank", "coverage_db@10_touched"]].copy()
    d = d.rename(columns={
        "stem_label": "PDF (law type)", "n_q": "n", "recall@5": "R@5", "recall@10": "R@10",
        "E/recall@10": "E/R@10", "hit@10": "Hit@10", "recall@20": "R@20", "recall@100": "R@100",
        "precision@10": "P@10 (nodes)", "distinct_articles@10": "distinct@10",
        "median_first_hit_rank": "med. rank", "coverage_db@10_touched": "coverage"})
    for c in d.columns:
        if c not in ("PDF (law type)", "n"):
            d[c] = d[c].map(lambda v: f"{v:.3f}" if pd.notna(v) else "--")
    ts.save_table(d, "tab_rq2_arm2a_per_pdf")


def tex_within_arm(agg: pd.DataFrame) -> None:
    d = agg[agg["aggregation"] == "micro"][
        ["variant_short", "unit", "recall@10", "E/recall@10", "hit@10", "recall@100",
         "precision@10", "coverage_db@10_touched", "pct_fragment_db@10",
         "pct_near_complete_db@10"]].copy()
    d = d.rename(columns={
        "variant_short": "Variant", "unit": "unit", "recall@10": "R@10", "E/recall@10": "E/R@10",
        "hit@10": "Hit@10", "recall@100": "R@100", "precision@10": "P@10",
        "coverage_db@10_touched": "coverage", "pct_fragment_db@10": "\\% frag",
        "pct_near_complete_db@10": "\\% near-cmpl"})
    for c in d.columns:
        if c not in ("Variant", "unit"):
            d[c] = d[c].map(lambda v: f"{v:.3f}" if pd.notna(v) else "--")
    ts.save_table(d, "tab_rq2_arm2a_within_arm")


def tex_miss(miss_sum: pd.DataFrame) -> None:
    d = miss_sum[["variant_short", "unit", "pct_gold_missed", "pct_missed_absent",
                  "pct_entire_miss_q", "missed_FOUND", "missed_PARTIAL"]].copy()
    d = d.rename(columns={
        "variant_short": "Variant", "unit": "unit", "pct_gold_missed": "\\% gold missed",
        "pct_missed_absent": "\\% miss ABSENT", "pct_entire_miss_q": "\\% entire-miss q",
        "missed_FOUND": "missed FOUND", "missed_PARTIAL": "missed PARTIAL"})
    for c in ("\\% gold missed", "\\% miss ABSENT", "\\% entire-miss q"):
        d[c] = d[c].map(lambda v: f"{v:.1f}" if pd.notna(v) else "--")
    ts.save_table(d, "tab_rq2_arm2a_miss")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    paths.ensure_output_dirs()
    print("Computing Arm-2A deep-dive over the curated 5-PDF set ...")
    R = compute()

    print("Writing tables (CSV + MD) ...")
    dump(R["pq"], "arm2a_per_question")
    dump(R["cov"], "arm2a_coverage_per_gold_article")
    dump(R["weights"], "arm2a_node_bsard_weights", floatfmt="{:.6f}")
    dump(R["missed"], "arm2a_missed_gold")
    dump(R["per_pdf"], "arm2a_per_pdf_summary")
    dump(R["agg"], "arm2a_aggregate_summary")
    dump(R["strata"], "arm2a_strata_summary")
    dump(R["miss_sum"], "arm2a_miss_summary")

    print("Writing report-ready .tex ...")
    tex_per_pdf(R["per_pdf"])
    tex_within_arm(R["agg"])
    tex_miss(R["miss_sum"])

    print("Building figures ...")
    # All Arm-2A deep-dive figures were retired in the figure review.
    # The cardinality view moved to the cross-arm curve
    # Report/scripts/build_arm2a_recall_curve_cardinality.py.
    # (coverage_dist / recall_vs_coverage / node_vs_article /
    # entire_miss_extraction / strata deleted.)

    agg = R["agg"]
    prim = agg[(agg["variant"] == dd.PRIMARY) & (agg["aggregation"] == "micro")].iloc[0]
    print("\n=== HEADLINE — summary_node (canonical, micro) ===")
    print(f"  n_q={int(prim['n_q'])}  recall@10={prim['recall@10']:.3f} "
          f"[{prim['recall@10_ci_lo']:.3f}, {prim['recall@10_ci_hi']:.3f}]  "
          f"E/recall@10={prim['E/recall@10']:.3f}  hit@10={prim['hit@10']:.3f}")
    print(f"  coverage_db@10(touched)={prim['coverage_db@10_touched']:.3f}  "
          f"%fragment={prim['pct_fragment_db@10']:.1f}  %near-complete={prim['pct_near_complete_db@10']:.1f}")
    print(f"  (Arm-1 reference: coverage 0.806 / 18.0% frag / 64.0% near-complete)")
    print("Done.")


if __name__ == "__main__":
    main()
