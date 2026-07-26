"""Generate ``local_t05_eval_and_compare.ipynb``.

Re-run whenever a cell changes:

    python notebooks/_build_local_eval_notebook.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


CELLS: list[tuple[str, str]] = []


def md(s: str) -> None:
    idx = len(CELLS)
    CELLS.append(("markdown", f"*— cell {idx} —*\n\n{s}"))


def code(s: str) -> None:
    idx = len(CELLS)
    CELLS.append(("code", f"# === cell {idx} ===\n{s}"))


# 0 — Title
md("""\
# T05 — local evaluation & cross-arm comparison

Reads per-query result bundles produced by each arm, aggregates the
binary IR metrics, and compares them on a shared ground-truth subset.

This runs **locally** — no GPU. T07 weighted metrics are not exercised
here (they need T03's chunk weights cache); we use the standalone
metric functions for binary R@k / MRR / NDCG.

Sections:
1. Config
2. Optional — pull T05 results from Azure Blob
3. Load all arms
4. Coverage check
5. Headline binary metrics
6. Cost summary
7. Stratified analysis
8. Significance tests
9. Cost-vs-recall plot
""")


# 1 — Config
md("## 1. Config")
code("""\
from pathlib import Path

# Edit these per run.
RUN_NAME = "headline_exact_test"
GT_PATH = Path("../RQ2_T07_EVALUATION/ground_truth/runs/headline_exact_test.json")

# Map arm name -> directory of q<qid>.json files (each = a serialized
# RetrievalResult). Add T03/T04 paths once their compatible bundles exist.
ARM_RESULTS = {
    "arm2b_pageindex": Path(f"data/results/{RUN_NAME}"),
    # "arm2a_metadata": Path("../RQ2_T04_ARM2_METADATA/data/<doc_id>/t08_bundles/<qset_hash>"),
    # "arm1_naive":     Path("../RQ2_T03_ARM1_NAIVE/data/<doc_id>/t08_bundles/<qset_hash>"),
}

QUESTION_STATUS_JSONL = Path(
    "../RQ2_T07_EVALUATION/ground_truth"
    "/question_extraction_status/questions_by_extraction_status.jsonl"
)

# Optional: pull T05 results from Azure Blob.
AZURE_CONTAINER_SAS_URL = ""
T05_RESULTS_BLOB_PREFIX = f"t05_results/{RUN_NAME}"
DOWNLOAD_T05 = False
""")


# 2 — Pull T05 from blob
md("""\
## 2. Optional — pull T05 results from Azure Blob

Toggle `DOWNLOAD_T05 = True` after the Azure run completes. Idempotent
(skips files already on disk).""")
code("""\
if DOWNLOAD_T05:
    from azure.storage.blob import ContainerClient

    container = ContainerClient.from_container_url(AZURE_CONTAINER_SAS_URL)
    target = ARM_RESULTS["arm2b_pageindex"]
    target.mkdir(parents=True, exist_ok=True)

    pulled = 0
    for blob in container.list_blobs(name_starts_with=f"{T05_RESULTS_BLOB_PREFIX}/"):
        name = blob.name.rsplit("/", 1)[-1]
        out = target / name
        if out.exists():
            continue
        with out.open("wb") as f:
            f.write(container.get_blob_client(blob).download_blob().readall())
        pulled += 1
    print(f"Downloaded {pulled} files to {target}")
else:
    print("Skipping blob download (DOWNLOAD_T05=False).")
""")


# 3 — Load arms
md("## 3. Load per-query results from each arm")
code("""\
import json
from dataclasses import dataclass


@dataclass
class LoadedResult:
    query_id: str
    query_text: str
    ranked_items: list   # raw dicts as written by run_subset
    cost: dict
    method: str


def load_arm(path: Path) -> dict[str, LoadedResult]:
    out: dict[str, LoadedResult] = {}
    if not path.exists():
        print(f"  WARNING: {path} does not exist - skipping arm")
        return out
    for fp in sorted(path.glob("q*.json")):
        d = json.loads(fp.read_text(encoding="utf-8"))
        out[str(d["query_id"])] = LoadedResult(
            query_id=str(d["query_id"]),
            query_text=d.get("query_text", ""),
            ranked_items=d.get("ranked_items", []),
            cost=d.get("cost", {}) or {},
            method=d.get("method", "?"),
        )
    return out


results_by_arm: dict[str, dict[str, LoadedResult]] = {
    arm: load_arm(p) for arm, p in ARM_RESULTS.items()
}
for arm, r in results_by_arm.items():
    print(f"  {arm}: {len(r):>4} queries  ({ARM_RESULTS[arm]})")
""")


# 4 — Coverage check
md("## 4. Coverage check\\n\\nDoes each arm cover the GT subset?")
code("""\
gt = json.loads(GT_PATH.read_text(encoding="utf-8"))
gt_qids = set(gt.keys())
print(f"GT subset size: {len(gt_qids)} questions")
print()
for arm, r in results_by_arm.items():
    have = set(r.keys())
    missing = sorted(gt_qids - have)
    extra = sorted(have - gt_qids)
    print(f"  {arm}: missing {len(missing):>3}, extra {len(extra):>3}")
    if missing:
        print(f"    missing sample: {missing[:10]}")
""")


# 5 — Headline metrics
md("## 5. Headline binary metrics\\n\\nR@k / MRR@10 / NDCG@10 across the GT-covered intersection.")
code("""\
import numpy as np
import pandas as pd


def ranked_bsard_ids(items: list) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for it in items:
        bid = (it.get("metadata") or {}).get("bsard_id")
        if bid is None:
            continue
        bid_int = int(bid)
        if bid_int in seen:
            continue
        seen.add(bid_int)
        out.append(bid_int)
    return out


def recall_at_k(ranked: list[int], gt_set: set[int], k: int) -> float:
    if not gt_set:
        return 0.0
    return len(set(ranked[:k]) & gt_set) / len(gt_set)


def mrr_at_k(ranked: list[int], gt_set: set[int], k: int) -> float:
    for i, b in enumerate(ranked[:k]):
        if b in gt_set:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(ranked: list[int], gt_set: set[int], k: int) -> float:
    gains = [1.0 if b in gt_set else 0.0 for b in ranked[:k]]
    dcg = sum(g / np.log2(i + 2) for i, g in enumerate(gains))
    ideal = sorted(gains, reverse=True)
    idcg = sum(g / np.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


KS = (1, 5, 10, 20, 100)


def per_query_metrics(arm_results: dict[str, LoadedResult]) -> pd.DataFrame:
    rows = []
    for qid, gt_bids in gt.items():
        if qid not in arm_results:
            continue
        ranked = ranked_bsard_ids(arm_results[qid].ranked_items)
        gt_set = {int(b) for b in gt_bids}
        rec = {"query_id": qid}
        for k in KS:
            rec[f"R@{k}"] = recall_at_k(ranked, gt_set, k)
        rec["MRR@10"] = mrr_at_k(ranked, gt_set, 10)
        rec["NDCG@10"] = ndcg_at_k(ranked, gt_set, 10)
        rows.append(rec)
    return pd.DataFrame(rows).set_index("query_id")


per_query_by_arm = {
    arm: per_query_metrics(results) for arm, results in results_by_arm.items()
}

agg_rows = []
for arm, pq in per_query_by_arm.items():
    if pq.empty:
        continue
    agg_rows.append({"arm": arm, "n_q": len(pq), **pq.mean().to_dict()})
table = pd.DataFrame(agg_rows).set_index("arm").round(4)
table
""")


# 6 — Cost summary
md("## 6. Cost summary\\n\\nLLM calls, tokens, latency. Zero LLM cost for T03/T04.")
code("""\
cost_rows = []
for arm, results in results_by_arm.items():
    calls = [r.cost.get("llm_calls", 0) for r in results.values()]
    latencies = [r.cost.get("latency_ms", 0.0) for r in results.values()]
    tok_in = [r.cost.get("tokens_in", 0) for r in results.values()]
    tok_out = [r.cost.get("tokens_out", 0) for r in results.values()]
    parse_fail = [r.cost.get("parse_failures", 0) for r in results.values()]
    cost_rows.append({
        "arm": arm,
        "n_q": len(results),
        "calls/q": np.mean(calls) if calls else 0.0,
        "tok_in/q": np.mean(tok_in) if tok_in else 0.0,
        "tok_out/q": np.mean(tok_out) if tok_out else 0.0,
        "lat_p50_ms": float(np.median(latencies)) if latencies else 0.0,
        "lat_p95_ms": float(np.percentile(latencies, 95)) if latencies else 0.0,
        "lat_total_s": sum(latencies) / 1000.0,
        "parse_fail_total": sum(parse_fail),
    })
pd.DataFrame(cost_rows).set_index("arm").round(2)
""")


# 7 — Stratification
md("""\
## 7. Stratified analysis

Two cuts:

1. **By BSARD `extraction_status`** — exact / partial / mixed / not_present.
2. **By `n_relevant_bsard_articles`** — single-article queries (1) vs
   multi-article (2-3, 4+). Multi-article is where T05's
   cross-reference-following is meant to pay off.""")
code("""\
status_df = pd.read_json(QUESTION_STATUS_JSONL, lines=True)
status_df["question_id"] = status_df["question_id"].astype(str)
status_df = status_df[status_df["question_id"].isin(gt_qids)].copy()
status_df["n_bucket"] = status_df["n_relevant_bsard_articles"].apply(
    lambda n: "1" if n == 1 else ("2-3" if n <= 3 else "4+")
)
print(f"Stratification base: {len(status_df)} questions matching GT")


def stratify(field: str) -> pd.DataFrame:
    rows = []
    for value in sorted(status_df[field].unique()):
        qids = set(status_df[status_df[field] == value]["question_id"])
        for arm, pq in per_query_by_arm.items():
            slice_ = pq[pq.index.isin(qids)]
            rows.append({
                field: value,
                "arm": arm,
                "n_q": len(slice_),
                "R@10": float(slice_["R@10"].mean()) if len(slice_) else float("nan"),
                "MRR@10": float(slice_["MRR@10"].mean()) if len(slice_) else float("nan"),
            })
    return pd.DataFrame(rows)


print("\\nBy extraction_status (R@10):")
print(stratify("extraction_status").pivot(index="extraction_status", columns="arm", values="R@10").round(4))

print("\\nBy n_relevant_bsard_articles bucket (R@10):")
print(stratify("n_bucket").pivot(index="n_bucket", columns="arm", values="R@10").round(4))
""")


# 8 — Significance
md("""\
## 8. Pairwise significance tests on R@10

Paired t-test + Wilcoxon signed-rank, restricted to queries shared by
both arms in each pair. Skipped when fewer than 5 paired observations.""")
code("""\
arms = list(per_query_by_arm.keys())
sig_rows = []
if len(arms) < 2:
    print("Need at least two arms with results — skipping significance.")
else:
    from scipy.stats import ttest_rel, wilcoxon

    for i, a in enumerate(arms):
        for b in arms[i + 1 :]:
            shared = per_query_by_arm[a].index.intersection(per_query_by_arm[b].index)
            if len(shared) < 5:
                sig_rows.append({"a": a, "b": b, "n": len(shared), "note": "<5 paired obs"})
                continue
            x = per_query_by_arm[a].loc[shared, "R@10"].to_numpy()
            y = per_query_by_arm[b].loc[shared, "R@10"].to_numpy()
            t_stat, p_t = ttest_rel(x, y)
            try:
                w_stat, p_w = wilcoxon(x, y)
            except ValueError:
                w_stat, p_w = float("nan"), 1.0
            sig_rows.append({
                "a": a, "b": b, "n": int(len(shared)),
                "mean_a": float(x.mean()), "mean_b": float(y.mean()),
                "ttest_p": float(p_t), "wilcoxon_p": float(p_w),
            })

pd.DataFrame(sig_rows).round(4) if sig_rows else "no pairs"
""")


# 9 — Plot
md("## 9. Cost-vs-recall plot\\n\\nVisualises the trade-off T05 makes against the zero-LLM arms.")
code("""\
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(7.5, 4.5))
cost_lookup = {row["arm"]: row for row in cost_rows}
for arm, pq in per_query_by_arm.items():
    if pq.empty or arm not in cost_lookup:
        continue
    r10 = float(pq["R@10"].mean())
    calls = float(cost_lookup[arm]["calls/q"])
    ax.scatter(calls, r10, s=120, edgecolors="black", linewidths=0.7)
    ax.annotate(
        arm, (calls, r10),
        fontsize=10, xytext=(7, 5), textcoords="offset points",
    )
ax.set_xlabel("Mean LLM calls per query")
ax.set_ylabel("R@10")
ax.set_title(f"Cost vs. recall — {RUN_NAME}")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
""")


# ── Builder (identical to azure builder) ─────────────────────────────────────


def _to_source(s: str) -> list[str]:
    return s.splitlines(keepends=True)


def _cell(kind: str, source: str) -> dict:
    base = {"cell_type": kind, "metadata": {}, "source": _to_source(source)}
    if kind == "code":
        base["execution_count"] = None
        base["outputs"] = []
    return base


def build() -> dict:
    return {
        "cells": [_cell(kind, src) for kind, src in CELLS],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> int:
    out = Path(__file__).resolve().parent / "local_t05_eval_and_compare.ipynb"
    out.write_text(
        json.dumps(build(), ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(f"Wrote {out} ({len(CELLS)} cells)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
