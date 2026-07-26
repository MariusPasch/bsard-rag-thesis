"""Arm-agnostic deep-dive primitives shared by the Arm-1 and Arm-2A dives.

Every metric here is defined purely over ``(ranked bsard-id structures, gold
set)`` so that the Arm-1 (chunk) and Arm-2A (node / article) retrieved-article
dives compute **Group A** (article-level retrieval) and **Group D1** (overlap
with gold) by calling the *same code* — the "built for a later join" requirement
in both briefs. ``arm1_deep_dive`` and ``arm2a_deep_dive`` both import these, so
the metrics are provably identical (Arm-1's published tables are re-asserted
byte-identical after the refactor).

Content **coverage (Group B) is deliberately *not* here**: its weight
construction is arm-specific (Arm-1 uses T03's char spans; Arm-2A uses node text
vs the BSARD-DB article text), so each dive owns its own coverage module.

Two senses of *k*, labelled by the caller:
  - **k = articles** — recall / hit / jaccard / rank-of-gold, on the article
    ranking (units aggregated to bsard articles by *max unit score*).
  - **k = units** — chunks for Arm 1, nodes for Arm 2A: article-context
    precision, distinct units, gold/distractor mix (the generator's context
    budget). For a whole-unit ranking (raw_article) a "unit" already *is* an
    article, so precision/distinct collapse onto the article level by design.

Determinism: ``SEED = 42`` for the only stochastic primitive (bootstrap CIs).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import paths

SEED = 42
N_BOOT = 2000


# ---------------------------------------------------------------------------
# Ranking primitives (units -> article ranking by max score; unit ranking)
# ---------------------------------------------------------------------------

def article_ranking(result) -> list[int]:
    """Units -> distinct bsard articles, ranked by *max unit score* (desc).

    Reproduces the T07 adapter's article-aggregation path (max unit score per
    bsard_id) used by every arm; ties broken by first appearance. Works for any
    unit whose ``metadata`` carries a ``bsard_ids`` list (T03 chunks, T04 nodes,
    T04 article-units).
    """
    best: dict[int, float] = {}
    order: dict[int, int] = {}
    for idx, it in enumerate(result.ranked_items):
        s = float(it["score"])
        for b in it["metadata"].get("bsard_ids", []):
            b = int(b)
            if b not in best or s > best[b]:
                best[b] = s
                order[b] = idx
    return [b for b, _ in sorted(best.items(), key=lambda kv: (-kv[1], order[kv[0]]))]


def unit_ranking(result) -> list[tuple[str, list[int]]]:
    """Top units as ``(unit_id, [bsard_ids])`` ordered by fused score desc."""
    rows = [
        (it["id"], [int(b) for b in it["metadata"].get("bsard_ids", [])], float(it["score"]))
        for it in result.ranked_items
    ]
    rows.sort(key=lambda r: -r[2])
    return [(uid, bids) for uid, bids, _ in rows]


# ---------------------------------------------------------------------------
# Group A (article level) + Group D1 (overlap with gold)
# ---------------------------------------------------------------------------

def article_metrics(gold: set[int], art_rank: list[int], k_list) -> dict:
    """A1 recall@k (fractional), A2 hit@k, D1 jaccard@k (k = articles)."""
    out: dict = {}
    for k in k_list:
        topk = set(art_rank[:k])
        inter = gold & topk
        out[f"recall@{k}"] = len(inter) / len(gold)
        out[f"hit@{k}"] = float(len(inter) > 0)
        out[f"jaccard@{k}"] = len(inter) / len(gold | topk)
    return out


def rank_of_gold(gold: set[int], art_rank: list[int]) -> dict:
    """A5 rank of first gold + ranks of all gold (median / p90), on articles."""
    rank_pos = {b: i + 1 for i, b in enumerate(art_rank)}
    gr = [rank_pos[b] for b in gold if b in rank_pos]
    return {
        "first_hit_rank": min(gr) if gr else np.nan,
        "last_gold_rank": max(gr) if len(gr) == len(gold) else np.nan,
        "median_gold_rank": float(np.median(gr)) if gr else np.nan,
        "p90_gold_rank": float(np.percentile(gr, 90)) if gr else np.nan,
        "n_gold_found_in_pool": len(gr),
    }


def unit_context_metrics(
    gold: set[int],
    unit_rank: list[tuple[str, list[int]]],
    k_list,
    *,
    redundancy_label: str = "units",
) -> dict:
    """A3 article-context precision, A4 distinct, A6 gold/distractor mix (k = units).

    article-context precision@k = gold / distinct articles touched by the top-k
    units. ``redundancy_label`` names the C1 redundancy key
    (``{label}_per_article@k`` = k / distinct articles) so each arm keeps its
    native column name (``chunks`` for Arm 1, ``nodes`` for Arm 2A) while the
    arithmetic stays identical.
    """
    out: dict = {}
    for k in k_list:
        top = unit_rank[:k]
        distinct: set[int] = set()
        for _uid, bids in top:
            distinct.update(bids)
        gold_present = gold & distinct
        n_distinct = len(distinct)
        out[f"distinct_articles@{k}"] = n_distinct
        out[f"precision@{k}"] = len(gold_present) / n_distinct if n_distinct else 0.0
        out[f"n_gold_articles@{k}"] = len(gold_present)
        out[f"n_distractor_articles@{k}"] = n_distinct - len(gold_present)
        out[f"{redundancy_label}_per_article@{k}"] = k / n_distinct if n_distinct else np.nan
    return out


def units_to_cover_gold(
    gold: set[int],
    unit_rank: list[tuple[str, list[int]]],
) -> float:
    """C2 — number of top units consumed to first-touch the whole gold set."""
    touched: set[int] = set()
    for i, (_uid, bids) in enumerate(unit_rank):
        touched.update(b for b in bids if b in gold)
        if touched == gold:
            return i + 1
    return np.nan


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

def bootstrap_ci(values: np.ndarray, *, seed: int = SEED, n: int = N_BOOT,
                 alpha: float = 0.05) -> tuple[float, float]:
    if len(values) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n, len(values)))
    means = values[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))


# ---------------------------------------------------------------------------
# Strata: GT cardinality + extraction status (FOUND / PARTIAL / NOT_FOUND)
# ---------------------------------------------------------------------------

def load_extraction_status() -> dict[int, dict]:
    """question_id -> {bsard_id: (verification_status, extraction_cosine)}."""
    fp = (paths.REPO_ROOT / "RQ2_T07_EVALUATION" / "ground_truth"
          / "question_extraction_status" / "questions_by_extraction_status.jsonl")
    out: dict[int, dict] = {}
    with fp.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            qid = int(d["question_id"])
            arts = {}
            for a in d.get("bsard_articles", []):
                arts[int(a["bsard_id"])] = (
                    a.get("verification_status"), a.get("extraction_cosine"),
                )
            out[qid] = arts
    return out


def annotate_strata(pq: pd.DataFrame, gt_by_stem: dict[str, dict[str, list[int]]],
                    extraction: dict[int, dict]) -> pd.DataFrame:
    """Add ``cardinality``, ``extraction_class`` and ``mean_extraction_cosine``.

    ``extraction_class`` is over the GT bsard_ids of that question only:
      FOUND     — all GT articles FOUND
      PARTIAL   — >=1 PARTIAL, no NOT_FOUND
      NOT_FOUND — >=1 NOT_FOUND
      UNKNOWN   — no status available
    (Same bucketing rule for every arm so strata join across the dives.)
    """
    pq = pq.copy()
    pq["cardinality"] = np.where(pq["is_multi"], "multi", "single")
    classes, cosines = [], []
    for _, r in pq.iterrows():
        gold = gt_by_stem[r["stem"]].get(str(r["qid"]), [])
        statuses, cos_vals = [], []
        amap = extraction.get(int(r["qid"]), {})
        for b in gold:
            st, cos = amap.get(int(b), (None, None))
            if st is not None:
                statuses.append(st)
            if cos is not None:
                cos_vals.append(cos)
        if not statuses:
            classes.append("UNKNOWN")
        elif any(s == "NOT_FOUND" for s in statuses):
            classes.append("NOT_FOUND")
        elif any(s == "PARTIAL" for s in statuses):
            classes.append("PARTIAL")
        else:
            classes.append("FOUND")
        cosines.append(float(np.mean(cos_vals)) if cos_vals else np.nan)
    pq["extraction_class"] = classes
    pq["mean_extraction_cosine"] = cosines
    return pq
