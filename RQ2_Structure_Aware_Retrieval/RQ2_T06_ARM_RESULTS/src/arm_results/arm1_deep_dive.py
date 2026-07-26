"""Arm 1 (Naive chunking) retrieved-article deep-dive — compute engine.

Profiles the *retrieved articles* of Arm 1 (T03 naive RRF over chunks) along the
properties that matter for downstream answer generation, per PDF and aggregated
over the curated 5-PDF set. Pure read from the persisted one-path artifacts — no
retrieval is ever re-run (STYLEGUIDE §6 / EVALUATION_METHODOLOGY §4).

Scope (confirmed with the user 2026-06-01): **Arm 1 only**. Groups A (article-
level retrieval), B (content coverage), C (redundancy) and D1 (overlap with gold)
are computed for Arm 1; the cross-arm overlap section (D2) is intentionally out of
scope here. Recall is reported under the **binary full per-PDF GT** as primary
(the brief's metric definitions + the per-query artifacts), with the **effective
drift-aware GT (E/)** shown alongside at per-PDF and aggregate granularity (read
straight from ``cross_arm_long.csv``, which carries both T1/ and E/ families).

Two senses of *k* are used and labelled explicitly everywhere:
  - **k = articles** (A1 recall, A2 hit, A5 rank-of-gold, D1 Jaccard): computed on
    the article-level ranking obtained by aggregating chunks to bsard articles by
    *max chunk score* — the documented Arm-1 adapter path. Verified to reproduce
    the published ``T1/R@k`` to 10 digits.
  - **k = chunks** (A3 precision, A4 distinct, A6 gold/distractor mix, all of
    Group B/C): computed on the top-k *chunks* the generator would actually
    receive. This is the Arm-1-specific lens — whole-unit arms (2A/2B) do not have
    a chunk budget, which is exactly why Group B/C are Arm-1-only.

Determinism: seed = 42 for the only stochastic step (bootstrap CIs). Idempotent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import deep_dive_common as common
from . import loaders, paths
from .weighted_metrics_shim import coverage_for_article  # local thin wrapper

# k-grids ---------------------------------------------------------------------
K_ARTICLE = [5, 10, 20, 100]   # article-ranking depths
K_CHUNK = [5, 10, 20]          # chunk-budget depths (generator context)
FRAGMENT_TAU = 0.5             # coverage < tau on a touched gold article = fragment
NEAR_COMPLETE_TAU = 0.9        # coverage >= this = near-complete
SEED = common.SEED
N_BOOT = common.N_BOOT

# Group-A / D1 primitives + strata are arm-agnostic and live in deep_dive_common
# so Arm 1 and Arm 2A compute them with provably identical code. These thin
# aliases preserve the historical Arm-1 call sites (chunks are the "units").
_article_ranking = common.article_ranking
_chunk_ranking = common.unit_ranking
_bootstrap_ci = common.bootstrap_ci
load_extraction_status = common.load_extraction_status
annotate_strata = common.annotate_strata


# ---------------------------------------------------------------------------
# Per-question metric record
# ---------------------------------------------------------------------------

@dataclass
class StemArtifacts:
    stem: str
    label: str
    gt: dict[str, list[int]]
    results: dict[str, object]                       # qid -> RetrievalResult
    weights: dict[tuple[str, int], float] | None     # (chunk_id, bsard_id) -> weight


def load_stem(stem: str) -> StemArtifacts:
    gt = loaders.load_ground_truth(stem)
    results = {r.query_id: r for r in loaders.load_t03(stem)}
    weights = loaders.load_t03_weights(stem)
    return StemArtifacts(stem, paths.stem_label(stem), gt, results, weights)


def per_question_table(art: StemArtifacts) -> pd.DataFrame:
    """One row per (stem, qid): Groups A, C, D1 scalars (binary full GT)."""
    rows: list[dict] = []
    for qid, gold_list in art.gt.items():
        gold = {int(b) for b in gold_list}
        if not gold or qid not in art.results:
            continue
        res = art.results[qid]
        art_rank = _article_ranking(res)
        chunk_rank = _chunk_ranking(res)

        row: dict = {
            "stem": art.stem, "stem_label": art.label, "qid": qid,
            "n_gold": len(gold),
            "is_multi": len(gold) > 1,
        }
        # Group A (k = articles), A5 rank-of-gold, A3/A4/A6 (k = chunks), C2 —
        # all via the shared deep_dive_common primitives. ``redundancy_label``
        # keeps Arm 1's native ``chunks_per_article@k`` column name.
        row.update(common.article_metrics(gold, art_rank, K_ARTICLE))
        row.update(common.rank_of_gold(gold, art_rank))
        row.update(common.unit_context_metrics(
            gold, chunk_rank, K_CHUNK, redundancy_label="chunks"))
        row["chunks_to_cover_gold"] = common.units_to_cover_gold(gold, chunk_rank)
        rows.append(row)
    return pd.DataFrame(rows)


def coverage_long_table(art: StemArtifacts) -> pd.DataFrame:
    """One row per (stem, qid, gold_bsard_id): Group B content coverage.

    coverage@k = min(1.0, sum of chunk->article token-overlap weights over the
    top-k chunks) = the fraction of that gold article's text actually retrieved.
    Whole-unit arms (2A/2B) would score ~1.0 on a hit by construction; this is the
    Arm-1-specific differentiator.
    """
    if art.weights is None:
        return pd.DataFrame()
    rows: list[dict] = []
    for qid, gold_list in art.gt.items():
        gold = {int(b) for b in gold_list}
        if not gold or qid not in art.results:
            continue
        chunk_rank = _chunk_ranking(art.results[qid])
        ranked_chunk_ids = [cid for cid, _ in chunk_rank]
        for b in gold:
            row: dict = {
                "stem": art.stem, "stem_label": art.label, "qid": qid,
                "bsard_id": b, "n_gold": len(gold), "is_multi": len(gold) > 1,
            }
            for k in K_CHUNK:
                cov = coverage_for_article(ranked_chunk_ids[:k], art.weights, b)
                row[f"coverage@{k}"] = cov
                row[f"touched@{k}"] = float(cov > 0.0)
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def per_pdf_summary(pq: pd.DataFrame, cov: pd.DataFrame,
                    long: pd.DataFrame) -> pd.DataFrame:
    """Per-PDF Group A/C/D1 means + Group B coverage + E/ alongside."""
    rows: list[dict] = []
    e_lookup = _effective_lookup(long)
    for stem, g in pq.groupby("stem"):
        row: dict = {
            "stem": stem, "stem_label": g["stem_label"].iloc[0], "n_q": len(g),
            "n_single": int((~g["is_multi"]).sum()), "n_multi": int(g["is_multi"].sum()),
        }
        for k in K_ARTICLE:
            row[f"recall@{k}"] = g[f"recall@{k}"].mean()
        row["hit@10"] = g["hit@10"].mean()
        row["E/recall@10"] = e_lookup.get((stem, "E/recall@10"), np.nan)
        row["E/recall@100"] = e_lookup.get((stem, "E/recall@100"), np.nan)
        for k in K_CHUNK:
            row[f"precision@{k}"] = g[f"precision@{k}"].mean()
            row[f"distinct_articles@{k}"] = g[f"distinct_articles@{k}"].mean()
            row[f"chunks_per_article@{k}"] = g[f"chunks_per_article@{k}"].mean()
        row["n_distractor_articles@10"] = g["n_distractor_articles@10"].mean()
        row["median_first_hit_rank"] = g["first_hit_rank"].median()
        row["p90_first_hit_rank"] = g["first_hit_rank"].quantile(0.90)
        row["jaccard@10"] = g["jaccard@10"].mean()
        row["chunks_to_cover_gold_median"] = g["chunks_to_cover_gold"].median()
        # Group B coverage (per gold-article rows for this stem)
        cg = cov[cov["stem"] == stem] if not cov.empty else cov
        if not cg.empty:
            touched10 = cg[cg["touched@10"] > 0]
            row["coverage@10_mean"] = cg["coverage@10"].mean()
            row["coverage@10_mean_touched"] = (
                touched10["coverage@10"].mean() if not touched10.empty else np.nan
            )
            row["pct_fragment_of_touched@10"] = (
                100.0 * (touched10["coverage@10"] < FRAGMENT_TAU).mean()
                if not touched10.empty else np.nan
            )
            row["pct_near_complete_of_touched@10"] = (
                100.0 * (touched10["coverage@10"] >= NEAR_COMPLETE_TAU).mean()
                if not touched10.empty else np.nan
            )
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("n_q", ascending=False).reset_index(drop=True)
    return out


def _effective_lookup(long: pd.DataFrame) -> dict[tuple[str, str], float]:
    sub = long[long["method"] == "T03_arm1_naive"]
    return {(r["stem"], r["metric"]): r["value"] for _, r in sub.iterrows()}


def aggregate_summary(pq: pd.DataFrame, cov: pd.DataFrame, per_pdf: pd.DataFrame,
                      long: pd.DataFrame) -> pd.DataFrame:
    """Macro (per-PDF mean) AND micro (question-weighted) aggregate, one row each."""
    rows: list[dict] = []
    e_lookup = _effective_lookup(long)
    # E/ micro needs n_evaluable weights; macro is unweighted per-PDF mean.
    e_per_stem = {
        s: (e_lookup.get((s, "E/recall@10"), np.nan),
            e_lookup.get((s, "E/recall@100"), np.nan),
            e_lookup.get((s, "E/n_evaluable"), np.nan))
        for s in pq["stem"].unique()
    }

    def make_row(kind: str) -> dict:
        row: dict = {"aggregation": kind, "n_q": len(pq), "n_pdfs": pq["stem"].nunique()}
        for k in K_ARTICLE:
            if kind == "micro":
                row[f"recall@{k}"] = pq[f"recall@{k}"].mean()
            else:
                row[f"recall@{k}"] = per_pdf[f"recall@{k}"].mean()
        # bootstrap CI on the headline recall@10
        if kind == "micro":
            lo, hi = _bootstrap_ci(pq["recall@10"].to_numpy())
        else:
            lo, hi = _bootstrap_ci(per_pdf["recall@10"].to_numpy())
        row["recall@10_ci_lo"], row["recall@10_ci_hi"] = lo, hi
        row["hit@10"] = pq["hit@10"].mean() if kind == "micro" else per_pdf["hit@10"].mean()
        # E/ alongside
        if kind == "macro":
            row["E/recall@10"] = np.nanmean([v[0] for v in e_per_stem.values()])
            row["E/recall@100"] = np.nanmean([v[1] for v in e_per_stem.values()])
        else:
            num10 = num100 = den = 0.0
            for v10, v100, n in e_per_stem.values():
                if not np.isnan(v10) and not np.isnan(n):
                    num10 += v10 * n
                    num100 += (v100 if not np.isnan(v100) else 0.0) * n
                    den += n
            row["E/recall@10"] = num10 / den if den else np.nan
            row["E/recall@100"] = num100 / den if den else np.nan
        for k in K_CHUNK:
            src = pq if kind == "micro" else per_pdf
            row[f"precision@{k}"] = src[f"precision@{k}"].mean()
            row[f"distinct_articles@{k}"] = src[f"distinct_articles@{k}"].mean()
            row[f"chunks_per_article@{k}"] = src[f"chunks_per_article@{k}"].mean()
        if kind == "micro":
            row["median_first_hit_rank"] = pq["first_hit_rank"].median()
            row["jaccard@10"] = pq["jaccard@10"].mean()
            if not cov.empty:
                touched = cov[cov["touched@10"] > 0]
                row["coverage@10_mean"] = cov["coverage@10"].mean()
                row["coverage@10_mean_touched"] = touched["coverage@10"].mean()
                row["pct_fragment_of_touched@10"] = 100.0 * (
                    touched["coverage@10"] < FRAGMENT_TAU).mean()
                row["pct_near_complete_of_touched@10"] = 100.0 * (
                    touched["coverage@10"] >= NEAR_COMPLETE_TAU).mean()
        else:
            row["median_first_hit_rank"] = per_pdf["median_first_hit_rank"].mean()
            row["jaccard@10"] = per_pdf["jaccard@10"].mean()
            row["coverage@10_mean"] = per_pdf["coverage@10_mean"].mean()
            row["coverage@10_mean_touched"] = per_pdf["coverage@10_mean_touched"].mean()
            row["pct_fragment_of_touched@10"] = per_pdf["pct_fragment_of_touched@10"].mean()
            row["pct_near_complete_of_touched@10"] = per_pdf[
                "pct_near_complete_of_touched@10"].mean()
        return row

    rows.append(make_row("micro"))
    rows.append(make_row("macro"))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Strata  (load_extraction_status / annotate_strata are aliased to
# deep_dive_common at the top of this module — shared verbatim with Arm 2A.)
# ---------------------------------------------------------------------------

def strata_summary(pq: pd.DataFrame, cov: pd.DataFrame) -> pd.DataFrame:
    """Recall@10 + coverage by cardinality and by extraction_class (micro)."""
    rows: list[dict] = []
    cov_q = None
    if not cov.empty:
        cov_q = cov.groupby(["stem", "qid"]).agg(
            coverage10=("coverage@10", "mean")).reset_index()
        cov_q["qid"] = cov_q["qid"].astype(str)

    def block(stratum: str, col: str):
        for val, g in pq.groupby(col):
            row = {
                "stratum": stratum, "level": val, "n_q": len(g),
                "recall@5": g["recall@5"].mean(),
                "recall@10": g["recall@10"].mean(),
                "recall@20": g["recall@20"].mean(),
                "hit@10": g["hit@10"].mean(),
                "median_first_hit_rank": g["first_hit_rank"].median(),
                "precision@10": g["precision@10"].mean(),
            }
            if cov_q is not None:
                merged = g[["stem", "qid"]].copy()
                merged["qid"] = merged["qid"].astype(str)
                merged = merged.merge(cov_q, on=["stem", "qid"], how="left")
                row["coverage@10_mean"] = merged["coverage10"].mean()
            rows.append(row)

    block("cardinality", "cardinality")
    block("extraction_class", "extraction_class")
    return pd.DataFrame(rows)
