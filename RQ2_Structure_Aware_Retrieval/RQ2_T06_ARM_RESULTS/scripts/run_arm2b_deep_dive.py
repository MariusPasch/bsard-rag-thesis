"""Build the Arm-2B (PageIndex) retrieved-article deep-dive.

Reads only the persisted one-path artifacts (raw T05 results + trace, tree.json,
GT, extraction status, cross_arm_long.csv). Navigation is NEVER rerun. Writes:
  - per-question, navigation-anatomy, gold-landing CSVs + summary CSV/MD
    -> RQ2_T06_ARM_RESULTS/data/tables/
  - report-ready .tex (tab_rq2_arm2b_*) and figures (fig_rq2_arm2b_*) -> Report/

Run via the T03 venv (data_loader + mpl; no tokenizer needed — Arm 2B has no
coverage):
  RQ2_T03_ARM1_NAIVE/.venv/Scripts/python.exe scripts/run_arm2b_deep_dive.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# --- path injection (mirror run_arm2a_deep_dive.py) ------------------------
_THIS = Path(__file__).resolve()
_T06 = _THIS.parents[1]
_RQ2 = _T06.parent
for _src in sorted(_RQ2.glob("RQ2_T0*/src")):
    sys.path.insert(0, str(_src))
sys.path.insert(0, str(_T06 / "src"))
sys.path.insert(0, str(_RQ2 / "Report"))           # thesis_style

from arm_results import arm2b_deep_dive as dd  # noqa: E402
from arm_results import deep_dive_common as common, paths  # noqa: E402
import thesis_style as ts  # noqa: E402

TABLES = paths.TABLES_DIR
VERM = ts.PALETTE["vermillion"]      # Arm 2B
GREY = ts.PALETTE["grey_mid"]
GREEN = ts.PALETTE["green"]          # Arm 1 reference
ORANGE = ts.PALETTE["orange"]        # Arm 2A reference

# Reference values from the prior dives (reports/arm1_*, arm2a_*); context only.
ARM1_R10, ARM1_HIT10 = 0.472, 0.630
ARM2A_R10, ARM2A_HIT10, ARM2A_MEDRANK = 0.482, 0.691, 2


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

    pq_parts, anat_parts, land_parts = [], [], []
    for stem in paths.STEMS:
        sd = dd.load_stem(stem)
        gt_by_stem[stem] = sd.gt
        pq_parts.append(dd.per_question_table(sd))
        anat_parts.append(dd.nav_anatomy_table(sd, extraction))
        land_parts.append(dd.gold_landing_table(sd))
        nq = len(sd.gt)
        print(f"  {stem} ({sd.label}): {nq} GT q | "
              f"tree articles={len(sd.tree.present)} | chapters={len(sd.tree.chapter_articles)}")

    pq = pd.concat(pq_parts, ignore_index=True)
    anat = pd.concat([a for a in anat_parts if not a.empty], ignore_index=True)
    landing = pd.concat(land_parts, ignore_index=True)

    pq = common.annotate_strata(pq, gt_by_stem, extraction)

    chk = dd.assert_published_recall(pq, long)
    print(f"  [assert] recomputed article R@k reproduces published T05 T1/R@k: "
          f"{chk['n_cells']}/{chk['n_cells']} cells; max dev={chk['max_dev']:.2e}")

    per_pdf = dd.per_pdf_summary(pq, landing, long)
    agg = dd.aggregate_summary(pq, per_pdf, long)
    strata = dd.strata_summary(pq)
    anat_sum = dd.nav_anatomy_summary(anat, landing, pq)
    miss_sum = dd.entire_miss_summary(landing, pq)

    # boundary-agreement diagnostics (the brief: how cleanly the methods agree)
    diag = {
        "exp_eq_nav_rate": float(pq["exp_eq_nav"].mean()),
        "exposed_subset_nav": bool((pq["n_exposed"] <= pq["n_navigated"]).all()),
        "n_exposed_le_navigated": int((pq["n_exposed"] <= pq["n_navigated"]).sum()),
        "n_q": int(len(pq)),
        "nav_le_rawsel_rate": float((pq["n_navigated"] <= pq["n_raw_selected"]).mean()),
    }
    return {"long": long, "pq": pq, "anat": anat, "landing": landing,
            "per_pdf": per_pdf, "agg": agg, "strata": strata,
            "anat_sum": anat_sum, "miss_sum": miss_sum, "diag": diag,
            "max_dev": chk["max_dev"]}


# ---------------------------------------------------------------------------
# Figures (Arm 2B = vermillion; Arm-1 green / Arm-2A orange references)
# ---------------------------------------------------------------------------

def fig_navigated_vs_padded(per_pdf: pd.DataFrame, agg: pd.DataFrame) -> None:
    """HEADLINE — per-PDF exposed-head vs padded recall@100, with the navigated
    recall marked and the random-padding counterfactual overlaid.

    Three levels read left-to-right on each row:
      * solid bar  — exposed-head recall (the LLM's own score>0 picks);
      * diamond    — recall if the SAME number of filler slots were drawn at
                     random from the document (expected, hypergeometric);
      * hatched bar end — shipped padded recall@100 (chapter-then-law fill).
    Gap solid->diamond = pure coverage artifact (a small doc + a 100-deep pool);
    gap diamond->bar end = the value the LLM's chapter localisation adds on top.
    """
    ts.rcparams_report()
    d = per_pdf.sort_values("n_q", ascending=False)
    y = np.arange(len(d))
    h = 0.4
    fig, ax = plt.subplots(figsize=ts.FIGSIZE["report_wide"])
    ax.barh(y + h / 2, d["padded_recall@100"], height=h, color=VERM, alpha=0.45,
            hatch="///", edgecolor="white", label="Padded recall (full top-100)")
    ax.barh(y - h / 2, d["exposed_recall"], height=h, color=VERM, alpha=0.95,
            label="Exposed-head recall (LLM picks, score>0)")
    # random-padding counterfactual on the padded bar: what coverage alone buys
    ax.scatter(d["random_recall@100"], y + h / 2, s=42, zorder=6, marker="D",
               facecolor="white", edgecolor=ts.PALETTE["black"], linewidth=1.1,
               label="Random padding (expected, same #fillers)")
    # navigated recall as a marker (what navigation reached, pre evaluate-gate)
    ax.scatter(d["nav_recall"], y - h / 2, color=ts.PALETTE["black"], s=18,
               zorder=5, marker="|", label="Navigated recall (pre evaluate-gate)")
    ax.set_yticks(y)
    ax.set_yticklabels(d["stem_label"])
    ax.invert_yaxis()
    ax.set_xlabel("Article recall (binary, full GT)")
    ax.set_xlim(0, 1)
    ax.legend(loc="lower right", fontsize=7.5, frameon=True, framealpha=0.92,
              facecolor="white", edgecolor="#DDDDDD")
    ts.save_figure(fig, "fig_rq2_arm2b_navigated_vs_padded")
    plt.close(fig)


def fig_nav_anatomy(anat_sum: pd.DataFrame) -> None:
    """Navigation-failure anatomy: per-PDF stacked bars of every gold article's
    fate — navigated, then the three failure classes (% of all gold)."""
    ts.rcparams_report()
    d = anat_sum[anat_sum["stem"] != "ALL"].sort_values("n_q", ascending=False)
    labels = list(d["stem_label"])
    x = np.arange(len(labels))
    nav = d["pct_navigated"].to_numpy()
    chom = d["pct_CHAPTER_OK_ARTICLE_MISSED"].to_numpy()
    wrong = d["pct_WRONG_CHAPTER"].to_numpy()
    absent = d["pct_ABSENT_FROM_TREE"].to_numpy()
    fig, ax = plt.subplots(figsize=ts.FIGSIZE["report_wide"])
    ax.bar(x, nav, color=GREEN, alpha=0.85, label="Navigated (reached)")
    ax.bar(x, chom, bottom=nav, color=ORANGE, alpha=0.85,
           label="Chapter OK, article missed")
    ax.bar(x, wrong, bottom=nav + chom, color=VERM, alpha=0.9,
           label="Wrong chapter")
    ax.bar(x, absent, bottom=nav + chom + wrong, color=ts.PALETTE["grey_dark"],
           alpha=0.9, label="Absent from tree")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, fontsize=7.5, ha="right")
    ax.set_ylabel("% of all gold articles")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=2, fontsize=7.5)
    ts.save_figure(fig, "fig_rq2_arm2b_nav_anatomy")
    plt.close(fig)


def fig_strata(strata: pd.DataFrame) -> None:
    """Exposed-head vs padded recall@100 split by query cardinality
    (single- vs multi-gold-article questions)."""
    ts.rcparams_report()
    fig, ax = plt.subplots(figsize=ts.FIGSIZE["report_square"])
    width = 0.38
    order = ["single", "multi"]
    sub = strata[strata["stratum"] == "cardinality"]
    levels = [l for l in order if (sub["level"] == l).any()]
    x = np.arange(len(levels))
    exp = [sub[sub["level"] == l]["exposed_recall"].mean() for l in levels]
    pad = [sub[sub["level"] == l]["recall@100"].mean() for l in levels]
    ax.bar(x - width / 2, exp, width=width, color=VERM, alpha=0.95,
           label="Exposed-head recall")
    ax.bar(x + width / 2, pad, width=width, color=VERM, alpha=0.45,
           hatch="///", edgecolor="white", label="Padded recall@100")
    ax.set_xticks(x)
    ax.set_xticklabels(levels)
    ax.set_xlabel("Query cardinality (gold articles per question)")
    ax.set_ylabel("Article recall (binary, full GT)")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, loc="upper left")
    ts.save_figure(fig, "fig_rq2_arm2b_strata_smallmultiples")
    plt.close(fig)


# ---------------------------------------------------------------------------
# .tex tables
# ---------------------------------------------------------------------------

def tex_per_pdf(per_pdf: pd.DataFrame) -> None:
    d = per_pdf[["stem_label", "n_q", "recall@10", "E/recall@10", "hit@10",
                 "recall@100", "exposed_recall", "nav_recall", "exposed_precision",
                 "n_exposed_median", "median_first_hit_rank"]].copy()
    d = d.rename(columns={
        "stem_label": "PDF (law type)", "n_q": "n", "recall@10": "R@10",
        "E/recall@10": "E/R@10", "hit@10": "Hit@10", "recall@100": "R@100 (padded)",
        "exposed_recall": "R (exposed)", "nav_recall": "R (navigated)",
        "exposed_precision": "P (exposed)", "n_exposed_median": "med. head",
        "median_first_hit_rank": "med. rank"})
    for c in d.columns:
        if c not in ("PDF (law type)", "n"):
            d[c] = d[c].map(lambda v: f"{v:.3f}" if pd.notna(v) else "--")
    ts.save_table(d, "tab_rq2_arm2b_per_pdf")


def tex_navigation(anat_sum: pd.DataFrame) -> None:
    d = anat_sum[["stem_label", "n_gold", "pct_navigated", "pct_WRONG_CHAPTER",
                  "pct_CHAPTER_OK_ARTICLE_MISSED", "pct_ABSENT_FROM_TREE",
                  "pct_exposed_head", "pct_padded_tail", "pct_outside_pool"]].copy()
    d = d.rename(columns={
        "stem_label": "PDF (law type)", "n_gold": "gold",
        "pct_navigated": "\\% nav.", "pct_WRONG_CHAPTER": "\\% wrong-ch",
        "pct_CHAPTER_OK_ARTICLE_MISSED": "\\% ch-ok-art-miss",
        "pct_ABSENT_FROM_TREE": "\\% absent",
        "pct_exposed_head": "\\% exposed", "pct_padded_tail": "\\% padded",
        "pct_outside_pool": "\\% outside"})
    for c in d.columns:
        if c not in ("PDF (law type)", "gold"):
            d[c] = d[c].map(lambda v: f"{v:.1f}" if pd.notna(v) else "--")
    ts.save_table(d, "tab_rq2_arm2b_navigation")


def tex_headline(agg: pd.DataFrame) -> None:
    d = agg[["aggregation", "recall@10", "E/recall@10", "hit@10", "recall@100",
             "nav_recall", "exposed_recall", "exposed_precision",
             "n_exposed_median", "median_first_hit_rank"]].copy()
    d = d.rename(columns={
        "aggregation": "agg", "recall@10": "R@10", "E/recall@10": "E/R@10",
        "hit@10": "Hit@10", "recall@100": "R@100", "nav_recall": "R(nav)",
        "exposed_recall": "R(exp)", "exposed_precision": "P(exp)",
        "n_exposed_median": "med.head", "median_first_hit_rank": "med.rank"})
    for c in d.columns:
        if c != "agg":
            d[c] = d[c].map(lambda v: f"{v:.3f}" if pd.notna(v) else "--")
    ts.save_table(d, "tab_rq2_arm2b_headline")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    paths.ensure_output_dirs()
    print("Computing Arm-2B (PageIndex) deep-dive over the curated 5-PDF set ...")
    R = compute()

    print("Writing tables (CSV + MD) ...")
    dump(R["pq"], "arm2b_per_question")
    dump(R["anat"], "arm2b_nav_anatomy")
    dump(R["landing"], "arm2b_gold_landing")
    dump(R["per_pdf"], "arm2b_per_pdf_summary")
    dump(R["agg"], "arm2b_aggregate_summary")
    dump(R["strata"], "arm2b_strata_summary")
    dump(R["anat_sum"], "arm2b_nav_anatomy_summary", floatfmt="{:.2f}")
    dump(R["miss_sum"], "arm2b_entire_miss_summary", floatfmt="{:.2f}")

    print("Writing report-ready .tex ...")
    tex_per_pdf(R["per_pdf"])
    tex_navigation(R["anat_sum"])
    tex_headline(R["agg"])

    print("Building figures ...")
    fig_navigated_vs_padded(R["per_pdf"], R["agg"])
    fig_nav_anatomy(R["anat_sum"])
    fig_strata(R["strata"])

    agg = R["agg"]
    mic = agg[agg["aggregation"] == "micro"].iloc[0]
    diag = R["diag"]
    print("\n=== HEADLINE — Arm 2B PageIndex (micro) ===")
    print(f"  n_q={int(mic['n_q'])}  recall@10={mic['recall@10']:.3f} "
          f"[{mic['recall@10_ci_lo']:.3f}, {mic['recall@10_ci_hi']:.3f}]  "
          f"E/recall@10={mic['E/recall@10']:.3f}  hit@10={mic['hit@10']:.3f}")
    print(f"  nav_recall={mic['nav_recall']:.3f}  exposed_recall={mic['exposed_recall']:.3f}  "
          f"padded(R@100)={mic['recall@100']:.3f}")
    print(f"  exposed_precision={mic['exposed_precision']:.3f}  "
          f"median exposed-head size={mic['n_exposed_median']:.0f}  "
          f"median first-gold rank={mic['median_first_hit_rank']:.0f}")
    print(f"  LLM calls/q={mic['llm_calls_mean']:.2f}  tokens/q={mic['tokens_per_q_mean']:.0f}")
    print(f"  [boundary] exposed==navigated in {100*diag['exp_eq_nav_rate']:.1f}% of q; "
          f"exposed is subset of navigated for {diag['n_exposed_le_navigated']}/{diag['n_q']} q")
    al = R["anat_sum"][R["anat_sum"]["stem"] == "ALL"].iloc[0]
    print(f"  [nav anatomy] navigated {al['pct_navigated']:.1f}% | "
          f"wrong-ch {al['pct_WRONG_CHAPTER']:.1f}% | "
          f"ch-ok-art-miss {al['pct_CHAPTER_OK_ARTICLE_MISSED']:.1f}% | "
          f"absent {al['pct_ABSENT_FROM_TREE']:.1f}%")
    em = R["miss_sum"][R["miss_sum"]["stratum"] == "overall"].iloc[0]
    print(f"  [entire-miss] {em['pct_entire_miss']:.1f}% of questions (gold never in top-100)")
    print(f"  (Arm-1 ref R@10 {ARM1_R10}, Hit@10 {ARM1_HIT10}; "
          f"Arm-2A ref R@10 {ARM2A_R10}, Hit@10 {ARM2A_HIT10}, med rank {ARM2A_MEDRANK})")
    print("Done.")


if __name__ == "__main__":
    main()
