"""Paired significance tests between arms, from per-query metric vectors.

Statistical comparison across arms is T06's lane (PROJECT_CONTEXT §4: scipy
paired tests / sign tests). Per-query metrics are computed with **T07's own**
``evaluation.metrics`` (``recall_at_k`` / ``mrr_at_k`` / ``ndcg_at_k``) over the
bsard-resolved rankings from ``evaluation.adapter.to_bsard_run`` — we call T07's
authoritative formulae, we do not reimplement them. Questions are the paired
unit; arms are aligned on the shared GT question set per PDF and pooled across
the 5 curated PDFs.

Metrics tested (B1 of ANALYSIS_PLAN.md): ``recall`` @{10,100}, ``mrr`` @10,
``ndcg`` @10. The MRR/nDCG rows close the rigor gap — Arm 2A's headline claim is
*ranking quality* (MRR@10, nDCG@10), which previously had no paired test.

This replaces T07's built-in ``comparator.significance``, which compares raw
per-query vectors keyed by chunk/node id (not bsard-resolved) and is therefore
unreliable for cross-arm use.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from . import loaders, paths

# metric name -> cutoffs tested
METRIC_KS: dict[str, tuple[int, ...]] = {
    "recall": (10, 100),
    "mrr": (10,),
    "ndcg": (10,),
}


def _per_query_metrics(stem: str) -> list[dict]:
    """Long rows {stem, qid, method, metric, k, value} for one PDF.

    Uses T07's recall_at_k / mrr_at_k / ndcg_at_k on the bsard-resolved ranking,
    on the shared GT question set.
    """
    from evaluation.adapter import to_bsard_run
    from evaluation.metrics import mrr_at_k, ndcg_at_k, recall_at_k

    funcs = {"recall": recall_at_k, "mrr": mrr_at_k, "ndcg": ndcg_at_k}

    gt = loaders.load_ground_truth(stem)
    gt_keys = set(gt)
    arms = loaders.load_all_arms(stem)

    rows: list[dict] = []
    for method, results in arms.items():
        results = [r for r in results if r.query_id in gt_keys]
        run = to_bsard_run(results)  # {qid: {bsard_str: score}}
        for qid, bids in gt.items():
            relevant = {str(b) for b in bids}
            scores = run.get(qid, {})
            ranked = sorted(scores, key=lambda b: -scores[b])
            for metric, ks in METRIC_KS.items():
                f = funcs[metric]
                for k in ks:
                    rows.append({
                        "stem": stem, "qid": qid, "method": method,
                        "metric": metric, "k": k, "value": f(ranked, relevant, k),
                    })
    return rows


def build_metric_matrix(stems: list[str] | None = None) -> pd.DataFrame:
    """Long per-query metric frame: stem, qid, method, metric, k, value."""
    stems = stems or paths.STEMS
    rows: list[dict] = []
    for stem in stems:
        per = _per_query_metrics(stem)
        rows.extend(per)
        n_methods = len({r["method"] for r in per})
        print(f"  [{stem}] per-query metrics built ({n_methods} methods)")
    return pd.DataFrame(rows)


def paired_tests(df: pd.DataFrame, pairs: list[tuple[str, str]]) -> pd.DataFrame:
    """Paired Wilcoxon + t-test per (pair, metric, k), aligned on (stem, qid)."""
    from scipy import stats

    out: list[dict] = []
    combos = df[["metric", "k"]].drop_duplicates().itertuples(index=False)
    combos = sorted(combos, key=lambda mk: (mk[0], mk[1]))
    for a, b in pairs:
        for metric, k in combos:
            sub = df[(df.metric == metric) & (df.k == k)]
            da = sub[sub.method == a].set_index(["stem", "qid"])["value"]
            db = sub[sub.method == b].set_index(["stem", "qid"])["value"]
            idx = da.index.intersection(db.index)
            if len(idx) == 0:
                continue
            xa = da.loc[idx].to_numpy(float)
            xb = db.loc[idx].to_numpy(float)
            diff = xa - xb

            try:
                w_p = float(stats.wilcoxon(xa, xb).pvalue) if np.any(diff) else 1.0
            except ValueError:
                w_p = float("nan")
            try:
                t_p = float(stats.ttest_rel(xa, xb).pvalue)
            except ValueError:
                t_p = float("nan")

            out.append({
                "A": a, "B": b, "metric": metric, "k": k,
                "mean_A": round(float(xa.mean()), 4),
                "mean_B": round(float(xb.mean()), 4),
                "mean_diff": round(float(diff.mean()), 4),
                "wins_A": int((diff > 0).sum()),
                "ties": int((diff == 0).sum()),
                "wins_B": int((diff < 0).sum()),
                "wilcoxon_p": w_p,
                "ttest_p": t_p,
                "n": int(len(idx)),
                "sig_0.05": bool(np.nanmin([w_p, t_p]) < 0.05),
            })
    return pd.DataFrame(out)


def default_pairs(methods: list[str]) -> list[tuple[str, str]]:
    """Champion (best pooled R@10) vs T03 / T05 / a raw T04, plus T03 vs T05."""
    present = set(methods)
    champ = "T04_summary_node_hybrid"
    if champ not in present:  # fall back to any summary/full node variant
        champ = next((m for m in methods if "summary" in m or "full_node" in m), methods[0])
    candidates = [
        (champ, "T03_naive"),
        (champ, "T05_pageindex"),
        ("T03_naive", "T05_pageindex"),
        (champ, "T04_raw_node_hybrid"),     # index-time enrichment effect
    ]
    return [(a, b) for a, b in candidates if a in present and b in present and a != b]


def _df_to_md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                cells.append(f"{v:.4g}" if c.endswith("_p") else f"{v:.4f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def run(stems: list[str] | None = None) -> pd.DataFrame:
    paths.ensure_output_dirs()
    print("Building per-query metric vectors (recall, mrr@10, ndcg@10)…")
    matrix = build_metric_matrix(stems)
    methods = sorted(matrix["method"].unique())
    res = paired_tests(matrix, default_pairs(methods))

    # Back-compat: per_query_recall.csv keeps its original schema
    # (stem, qid, method, k, recall) for the Report recall-curve / delta scripts.
    recall = (matrix[matrix.metric == "recall"]
              .rename(columns={"value": "recall"})[["stem", "qid", "method", "k", "recall"]])
    recall.to_csv(paths.TABLES_DIR / "per_query_recall.csv", index=False)
    # Full per-query metric matrix (recall + mrr + ndcg) for any later analysis.
    matrix.to_csv(paths.TABLES_DIR / "per_query_metrics.csv", index=False)

    res.to_csv(paths.TABLES_DIR / "cross_arm_significance.csv", index=False)
    md = [
        "# Cross-arm significance (paired, per question, pooled over 5 PDFs)",
        "",
        "Per-query metrics via T07 `recall_at_k` / `mrr_at_k` / `ndcg_at_k` on "
        "bsard-resolved rankings; paired Wilcoxon signed-rank + paired t-test on "
        "the shared GT question set. `wins_A`/`wins_B` count questions where A/B "
        "has strictly higher metric value. MRR/nDCG rows (k=10) test Arm 2A's "
        "ranking-quality claim.",
        "",
        _df_to_md(res),
        "",
        "> `sig_0.05` = min(wilcoxon_p, ttest_p) < 0.05.",
    ]
    (paths.TABLES_DIR / "cross_arm_significance.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nWrote {paths.TABLES_DIR / 'cross_arm_significance.csv'} ({len(res)} comparisons)")
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="Paired significance tests across arms.")
    ap.add_argument("--stems", nargs="*", default=None)
    args = ap.parse_args()
    run(args.stems)


if __name__ == "__main__":
    main()
