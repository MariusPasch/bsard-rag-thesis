"""Tidy loaders for the RQ2 cross-arm artifacts.

Mirrors the RQ1 ``Report/load_results.py`` separation: this module is the *only*
place that knows the on-disk schema of the T06 consolidation outputs, so build
scripts stay thin and a future schema change touches one file.

Everything sources from the single re-evaluation path (STYLEGUIDE §6) — the
persisted T06 frames, never a per-arm native CSV:

  - long frame            : ``cross_arm_long.csv`` (stem x method x metric)
  - significance          : ``cross_arm_significance.csv`` (paired, pooled)
  - per-question recall   : ``per_query_recall.csv``
  - failure profile       : ``error_analysis/failure_summary.csv``
  - cost                  : ``cross_arm_costs.json``

Aggregation note
----------------
``to_wide`` collapses the per-stem long frame to one row per method. Binary
metrics are weighted by ``n_gt_q`` (questions with GT) for the micro average;
``E/`` (drift-aware) metrics are weighted by ``n_evaluable`` (questions whose GT
reaches the PDF) because that is their actual denominator. ``macro`` is the
unweighted mean across the five PDFs. The micro binary R@10 values reproduce the
published T06 headline (summary 0.483 / naive 0.472 / pageindex 0.185).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPORT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import thesis_style as ts  # noqa: E402


# ---------------------------------------------------------------------------
# Method-name canonicalisation
# ---------------------------------------------------------------------------

def canonical_method(method: str) -> str:
    """Map an on-disk method name to the long-frame canonical (T03_naive alias)."""
    return ts.METHOD_ALIASES.get(method, method)


# ---------------------------------------------------------------------------
# Long frame + wide views
# ---------------------------------------------------------------------------

def load_long(path: Path | None = None, *, include_arm2c: bool = False) -> pd.DataFrame:
    """The consolidated long frame (delegates to thesis_style).

    ``include_arm2c`` appends the Arm-2C agentic-navigation rows computed from
    the experiment runs (load_arm2c), so cross-arm builders can carry the fourth
    arm without it being part of the T06 consolidation. Lazy import avoids a
    circular dependency (load_arm2c imports this module).
    """
    long = ts.load_long_dataframe(path)
    if include_arm2c:
        import load_arm2c as la
        long = pd.concat([long, la.arm2c_long_rows()], ignore_index=True)
    return long


def per_stem_wide(long: pd.DataFrame, metric_names: list[str]) -> pd.DataFrame:
    """One row per (stem, method) with the requested clean metrics as columns.

    Adds ``n_gt_q`` and (when present) ``n_evaluable`` for downstream weighting.
    """
    avail = long["metric"].unique()
    keymap = {name: ts.resolve_metric(avail, name) for name in metric_names}

    wide = long.pivot_table(
        index=["stem", "stem_label", "method", "n_gt_q"],
        columns="metric", values="value", aggfunc="first",
    ).reset_index()

    out = wide[["stem", "stem_label", "method", "n_gt_q"]].copy()
    if "E/n_evaluable" in wide.columns:
        out["n_evaluable"] = wide["E/n_evaluable"]
    for clean, key in keymap.items():
        out[clean] = wide[key] if key in wide.columns else pd.NA
    return out


def to_wide(long: pd.DataFrame, metric_names: list[str],
            *, weighting: str = "micro") -> pd.DataFrame:
    """Collapse to one row per method, aggregated across PDFs.

    ``weighting`` is ``"micro"`` (question-weighted; E/ by n_evaluable, others by
    n_gt_q) or ``"macro"`` (unweighted per-PDF mean). Result is sorted by the
    first metric descending and carries ``n_q_total`` / ``n_pdfs``.
    """
    per_stem = per_stem_wide(long, metric_names)
    has_eval = "n_evaluable" in per_stem.columns
    rows: list[dict] = []
    for method, g in per_stem.groupby("method"):
        row: dict = {"method": method}
        for clean in metric_names:
            vals = pd.to_numeric(g[clean], errors="coerce")
            if weighting == "macro":
                row[clean] = vals.mean()
            else:
                if clean.startswith("E/") and has_eval:
                    w = pd.to_numeric(g["n_evaluable"], errors="coerce")
                else:
                    w = pd.to_numeric(g["n_gt_q"], errors="coerce")
                mask = vals.notna() & w.notna() & (w > 0)
                denom = w[mask].sum()
                row[clean] = (vals[mask] * w[mask]).sum() / denom if denom else float("nan")
        row["n_q_total"] = int(pd.to_numeric(g["n_gt_q"], errors="coerce").sum())
        row["n_pdfs"] = int(g["stem"].nunique())
        rows.append(row)
    out = pd.DataFrame(rows)
    if metric_names and metric_names[0] in out.columns:
        out = out.sort_values(metric_names[0], ascending=False).reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Significance
# ---------------------------------------------------------------------------

def load_significance(path: Path | None = None) -> pd.DataFrame:
    """Paired pairwise significance frame (A, B, metric, k, means, wins, p-values).

    The ``metric`` column distinguishes recall / mrr / ndcg rows (B1). Older
    recall-only CSVs without that column are normalised to ``metric='recall'``.
    """
    path = path or ts.SIGNIFICANCE_CSV
    if not path.exists():
        raise FileNotFoundError(f"Significance CSV not found at {path}.")
    df = pd.read_csv(path)
    if "metric" not in df.columns:
        df["metric"] = "recall"
    df["A"] = df["A"].map(canonical_method)
    df["B"] = df["B"].map(canonical_method)
    return df


def pair_significance(sig: pd.DataFrame, m1: str, m2: str, k: int,
                      *, metric: str = "recall") -> dict | None:
    """Return the (m1, m2) row for ``metric`` at depth k, oriented A=m1, B=m2.

    Looks up the unordered pair; if stored as (m2, m1) the means / wins / diff
    are flipped so the returned dict always reads "A = m1, B = m2".
    """
    m1, m2 = canonical_method(m1), canonical_method(m2)
    base = sig[(sig["metric"] == metric) & (sig["k"] == k)]
    hit = base[(base["A"] == m1) & (base["B"] == m2)]
    if not hit.empty:
        return dict(hit.iloc[0])
    flip = base[(base["A"] == m2) & (base["B"] == m1)]
    if not flip.empty:
        r = flip.iloc[0]
        return {
            "A": m1, "B": m2, "metric": metric, "k": k,
            "mean_A": r["mean_B"], "mean_B": r["mean_A"],
            "mean_diff": -r["mean_diff"],
            "wins_A": r["wins_B"], "ties": r["ties"], "wins_B": r["wins_A"],
            "wilcoxon_p": r["wilcoxon_p"], "ttest_p": r["ttest_p"],
            "n": r["n"], "sig_0.05": r["sig_0.05"],
        }
    return None


def p_vs_anchor(sig: pd.DataFrame, method: str, anchor: str, k: int,
                *, metric: str = "recall") -> float | None:
    """Two-sided paired t-test p of ``method`` vs ``anchor`` for ``metric`` at k."""
    if canonical_method(method) == canonical_method(anchor):
        return None
    row = pair_significance(sig, method, anchor, k, metric=metric)
    return float(row["ttest_p"]) if row and pd.notna(row["ttest_p"]) else None


# ---------------------------------------------------------------------------
# Failure profile + per-query recall + cost
# ---------------------------------------------------------------------------

def load_failures(path: Path | None = None) -> pd.DataFrame:
    """Per (stem, method) failure profile, with derived n_zero@10 + missed %."""
    path = path or ts.FAILURE_SUMMARY_CSV
    if not path.exists():
        raise FileNotFoundError(f"Failure summary not found at {path}.")
    df = pd.read_csv(path)
    df["method"] = df["method"].map(canonical_method)
    # n_zero@10 = questions with no GT in top-10 (RQ1's n_zero concept).
    df["n_zero@10"] = df["n_questions"] - df["hit@10"]
    df["missed_pct"] = 100.0 * df["missed_entirely"] / df["n_questions"]
    return df


def failures_by_method(path: Path | None = None) -> pd.DataFrame:
    """Failure profile pooled over PDFs, one row per method."""
    df = load_failures(path)
    agg = df.groupby("method")[
        ["hit@10", "near_miss_below_10", "missed_entirely", "n_zero@10", "n_questions"]
    ].sum().reset_index()
    agg["missed_pct"] = 100.0 * agg["missed_entirely"] / agg["n_questions"]
    return agg


def load_per_query(path: Path | None = None, *,
                   include_arm2c: bool = False) -> pd.DataFrame:
    """Per (stem, qid, method, k) recall frame (method names canonicalised).

    ``include_arm2c`` appends the Arm-2C per-question recall rows (k in
    {5,10,20,100}) computed from the run q*.json files (load_arm2c).
    """
    path = path or ts.PER_QUERY_RECALL_CSV
    if not path.exists():
        raise FileNotFoundError(f"Per-query recall not found at {path}.")
    df = pd.read_csv(path)
    df["method"] = df["method"].map(canonical_method)
    if include_arm2c:
        import load_arm2c as la
        extra = la.arm2c_per_query_rows()
        df = pd.concat([df, extra[df.columns.intersection(extra.columns)]],
                       ignore_index=True)
    return df


def per_query_delta(m1: str, m2: str, k: int = 10,
                    path: Path | None = None) -> pd.DataFrame:
    """Per-question recall delta (m1 − m2) at depth k, joined on (stem, qid)."""
    pq = load_per_query(path)
    m1, m2 = canonical_method(m1), canonical_method(m2)
    sub = pq[(pq["k"] == k) & (pq["method"].isin([m1, m2]))]
    piv = sub.pivot_table(index=["stem", "qid"], columns="method",
                          values="recall", aggfunc="first").reset_index()
    piv = piv.dropna(subset=[m1, m2])
    piv["delta"] = piv[m1] - piv[m2]
    return piv


def load_costs(path: Path | None = None) -> dict:
    """Raw per-stem cost dict from cross_arm_costs.json (best-effort)."""
    path = path or ts.COSTS_JSON
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def costs_by_method(path: Path | None = None, *,
                    include_arm2c: bool = False) -> pd.DataFrame:
    """Pool per-stem cost blocks into one row per method.

    Columns: method, n_q, llm_calls_per_q, tokens_per_q, latency_ms_mean,
    latency_ms_p95. Latency is question-count-weighted across PDFs; p95 is the
    n-weighted mean of per-PDF p95 (an approximation — true pooled p95 needs raw
    per-query timings). ``latency_ms_mean`` is NaN for arms that recorded no
    latency (T03, served from a precomputed pool).
    """
    raw = load_costs(path)
    acc: dict[str, dict] = {}
    for _stem, arms in raw.items():
        for method, blk in arms.items():
            m = canonical_method(method)
            n = float(blk.get("n_queries", 0) or 0)
            if n <= 0:
                continue
            a = acc.setdefault(m, {"n_q": 0.0, "llm": 0.0, "tok": 0.0,
                                   "lat_sum": 0.0, "lat_n": 0.0,
                                   "p95_sum": 0.0})
            a["n_q"] += n
            a["llm"] += float(blk.get("total_llm_calls", blk.get("avg_llm_calls", 0) * n) or 0)
            a["tok"] += float(blk.get("total_tokens", blk.get("avg_tokens", 0) * n) or 0)
            lat = float(blk.get("avg_latency_ms", 0) or 0)
            if lat > 0:                       # 0 == not recorded (T03)
                a["lat_sum"] += lat * n
                a["lat_n"] += n
                a["p95_sum"] += float(blk.get("p95_latency_ms", 0) or 0) * n
    rows = []
    for m, a in acc.items():
        lat_mean = a["lat_sum"] / a["lat_n"] if a["lat_n"] else float("nan")
        p95 = a["p95_sum"] / a["lat_n"] if a["lat_n"] else float("nan")
        rows.append({
            "method": m, "n_q": int(a["n_q"]),
            "llm_calls_per_q": a["llm"] / a["n_q"],
            "tokens_per_q": a["tok"] / a["n_q"],
            "latency_ms_mean": lat_mean, "latency_ms_p95": p95,
        })
    if include_arm2c:
        import load_arm2c as la
        rows.append(la.arm2c_cost_row())
    return pd.DataFrame(rows)


__all__ = [
    "canonical_method",
    "load_long", "per_stem_wide", "to_wide",
    "load_significance", "pair_significance", "p_vs_anchor",
    "load_failures", "failures_by_method",
    "load_per_query", "per_query_delta", "load_costs",
]
