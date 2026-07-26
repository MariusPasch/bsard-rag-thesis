"""
Tidy long-format loader for every RQ1 experiment JSON in output/results/.

What this returns
-----------------
A single `pandas.DataFrame` -- one row per (experiment_id, metric, k) triple --
that every figure / table script can filter, group, and pivot from. Plus a
narrower "wide" view that pivots the standard panel (Recall@k, MRR@10,
NDCG@10, MAP@10, Hit@10, p-value) one row per experiment.

The loader is the *only* place that knows the on-disk schema, so a future
refactor of the harness output only touches this module.

Tiers walked
------------
- Tier 1 sparse  : output/results/sparse_retrieval/*.json
- Tier 2 dense   : output/results/dense_retrieval/*.json
- Tier 3 hybrid  : output/results/hybrid/*.json
- Tier 4 agentic : output/results/agentic/llm_judge/**/*.json,
                   output/results/agentic/CRAG/*.json,
                   output/results/agentic/ReAct/*.json

Anything that is not a result JSON (e.g. tier3_per_query/ sidecars, summary
CSVs) is filtered out by file-shape inspection -- a result must have a
top-level `metrics` dict with `Recall@10`.

n_zero and Hit@10
-----------------
Hit@10 is read directly from the harness as `HitRate@10`. `n_zero@10` is the
integer count of test queries with no relevant doc in top-10, computed as
round(n_queries * (1 - HitRate@10)).
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# This module lives at <project>/analysis/Report_Tables_Figures/load_results.py,
# so the project root is two parents up. The location of the (gitignored) result
# JSONs is resolved by the project's central data-root resolver, which honours
# BSARD_DATA_DIR and otherwise falls back to <project>/output. See
# evaluation/paths.py and scripts/download_data.py.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.paths import RESULTS_DIR  # noqa: E402

# Map (tier directory) -> tier label used in DataFrames.
_TIER_DIRS: dict[str, str] = {
    "sparse_retrieval":       "T1",
    "dense_retrieval":        "T2",
    "hybrid":                 "T3",
    "agentic/llm_judge":      "T4.0",
    "agentic/CRAG":           "T4.1",
    "agentic/ReAct":          "T4.2",
}

# Standard supervised panel pulled into the wide view. R@k entries are taken
# directly; Hit@10 / n_zero@10 are derived (see module docstring).
_WIDE_K_METRICS: list[str] = [
    "Recall@1", "Recall@5", "Recall@10", "Recall@20", "Recall@50", "Recall@100",
]
_WIDE_RANK_AWARE: list[str] = ["MRR@10", "NDCG@10", "MAP@10"]


@dataclass(frozen=True)
class ResultRecord:
    """One experiment's loaded JSON, with cheap accessors."""
    path: Path
    payload: dict

    @property
    def experiment_id(self) -> str:
        return str(self.payload.get("experiment_id") or self.path.stem)

    @property
    def metrics(self) -> dict:
        return self.payload.get("metrics") or {}

    @property
    def stratified(self) -> dict:
        return self.payload.get("stratified") or {}

    @property
    def significance(self) -> dict:
        return self.payload.get("significance_vs_anchor") or {}

    @property
    def hyperparameters(self) -> dict:
        return self.payload.get("hyperparameters") or {}

    @property
    def preprocessing(self) -> dict:
        return self.payload.get("preprocessing") or {}

    @property
    def n_queries(self) -> int | None:
        n = self.metrics.get("T0/n_queries")
        return int(n) if n is not None else None


def _iter_result_files() -> list[tuple[str, Path]]:
    """Yield (tier_label, path) for every result JSON in scope."""
    out: list[tuple[str, Path]] = []
    for rel_dir, tier_label in _TIER_DIRS.items():
        d = RESULTS_DIR / rel_dir
        if not d.exists():
            continue
        for p in sorted(d.rglob("*.json")):
            # Skip per-query sidecars and any non-result aux file.
            if "tier3_per_query" in p.parts or "per_query" in p.name:
                continue
            # Skip selection-step files (e.g. Tier 2's EXP-D7 train_sample_100
            # winner picks at _sel_dense_*.json) -- those are val/train-side
            # intermediate records, not test-set results in scope for the
            # report.
            if p.name.startswith("_"):
                continue
            # Skip val-split runs (e.g. T4.0 prompt-variant selection on
            # llm_rerank_binary_top10_val.json). The report only displays
            # test-split results.
            if p.stem.endswith("_val"):
                continue
            out.append((tier_label, p))
    return out


def _load_one(path: Path) -> dict | None:
    try:
        with path.open(encoding="utf-8") as fp:
            payload = json.load(fp)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict) or "Recall@10" not in metrics:
        return None
    # Cross-check: if the payload self-identifies as a non-test split (or has
    # a *_val_* segment in the experiment_id), skip. The earlier filename-
    # only check (`*_val.json`) misses files like
    # crag_hybrid_aligned_val_v2.json where `_val_` sits in the middle.
    split = (payload.get("split") or "").lower()
    if split and split != "test":
        return None
    eid = str(payload.get("experiment_id") or "")
    if "_val_" in eid or eid.endswith("_val"):
        return None
    return payload


def load_records() -> list[tuple[str, ResultRecord]]:
    """Return [(tier_label, ResultRecord), ...] for every result JSON found."""
    out: list[tuple[str, ResultRecord]] = []
    for tier, p in _iter_result_files():
        payload = _load_one(p)
        if payload is None:
            continue
        out.append((tier, ResultRecord(path=p, payload=payload)))
    return out


# ---------------------------------------------------------------------------
# Long-format view -----------------------------------------------------------
# ---------------------------------------------------------------------------

_METRIC_K_RE = re.compile(r"^(?P<metric>[A-Za-z@_]+?)@(?P<k>\d+)$")


def to_long(records: list[tuple[str, ResultRecord]] | None = None) -> pd.DataFrame:
    """
    One row per (tier, experiment_id, metric, k, stratum, value).

    - stratum is "overall" for the top-level metrics block, or the stratum
      key (e.g. "single_article", "lexically_aligned") for stratified entries
    - metric is "Recall", "Precision", "F1", "MRR", "MAP", "NDCG", "HitRate",
      "IDPrecision", "IDRecall", or any other harness key that follows the
      "<name>@<k>" convention
    """
    if records is None:
        records = load_records()
    rows: list[dict] = []
    for tier, rec in records:
        exp = rec.experiment_id
        n_q = rec.n_queries
        # ----- overall metrics block ----------------------------------------
        for key, val in rec.metrics.items():
            m = _METRIC_K_RE.match(key)
            if not m or not isinstance(val, (int, float)):
                continue
            rows.append({
                "tier":          tier,
                "experiment_id": exp,
                "metric":        m["metric"],
                "k":             int(m["k"]),
                "stratum":       "overall",
                "value":         float(val),
                "n_queries":     n_q,
            })
        # ----- stratified block ---------------------------------------------
        for stratum, block in rec.stratified.items():
            if not isinstance(block, dict):
                continue
            for key, val in block.items():
                m = _METRIC_K_RE.match(key)
                if not m or not isinstance(val, (int, float)):
                    continue
                rows.append({
                    "tier":          tier,
                    "experiment_id": exp,
                    "metric":        m["metric"],
                    "k":             int(m["k"]),
                    "stratum":       stratum,
                    "value":         float(val),
                    "n_queries":     n_q,
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Wide view (one row per experiment) -----------------------------------------
# ---------------------------------------------------------------------------

def to_wide(records: list[tuple[str, ResultRecord]] | None = None) -> pd.DataFrame:
    """
    One row per experiment with the standard panel as columns.

    Columns
    -------
      tier, experiment_id, n_queries,
      Recall@{1,5,10,20,50,100},
      Hit@10, n_zero@10,
      MRR@10, NDCG@10, MAP@10,
      p_value_recall10, significant_recall10
    """
    if records is None:
        records = load_records()
    rows: list[dict] = []
    for tier, rec in records:
        row: dict = {
            "tier":          tier,
            "experiment_id": rec.experiment_id,
            "n_queries":     rec.n_queries,
        }
        m = rec.metrics
        for key in _WIDE_K_METRICS + _WIDE_RANK_AWARE:
            row[key] = m.get(key)

        # Hit@10 lives in harness output as HitRate@10.
        hit10 = m.get("HitRate@10")
        row["Hit@10"] = hit10
        if hit10 is not None and rec.n_queries is not None:
            row["n_zero@10"] = int(round(rec.n_queries * (1.0 - hit10)))
        else:
            row["n_zero@10"] = None

        sig = rec.significance
        # Expose both p-values explicitly so tier builders pick the one tied
        # to their primary k (T1/T3/T4 use k=10; T2 uses k=100). The harness's
        # own `significant` flag is ignored -- we derive significance from the
        # p-value the tier actually cares about.
        p10  = sig.get("p_value_recall10")
        p100 = sig.get("p_value_recall100")
        row["p_value_recall10"]    = p10
        row["p_value_recall100"]   = p100
        row["significant_recall10"]  = (p10  is not None and p10  < 0.05)
        row["significant_recall100"] = (p100 is not None and p100 < 0.05)

        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tier-1 filter (convenience) ------------------------------------------------
# ---------------------------------------------------------------------------

def tier_records(tier: str,
                 records: list[tuple[str, ResultRecord]] | None = None,
                 ) -> list[ResultRecord]:
    """Return ResultRecords for the given tier label (e.g. 'T1', 'T2', 'T4.0')."""
    if records is None:
        records = load_records()
    return [rec for t, rec in records if t == tier]


__all__ = [
    "ResultRecord",
    "load_records",
    "to_long",
    "to_wide",
    "tier_records",
    "RESULTS_DIR",
]
