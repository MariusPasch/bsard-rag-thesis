"""
Query stratification for failure-condition analysis.

Strata produced
---------------
1. single_article / multi_article — from n_relevant_articles
2. lexically_aligned / semantically_paraphrased / middle — (X-D1)
   Proxy: mean BM25 score of each query against its ground-truth articles
   using the anchor BM25 index (no normalization, text_only).
   Top quartile → lexically aligned; bottom quartile → semantically paraphrased.
3. with_cross_refs / without_cross_refs — from questions.has_cross_references
   NOTE: has_cross_references is stored on *articles*, not *questions*.
   We derive it from whether ANY ground-truth article has cross-references.

The strata are computed once and cached in evaluation/data/query_strata.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

STRATA_FILE = Path(__file__).parent / "data" / "query_strata.json"


def _bm25_scores_for_gt(
    questions: list[dict],
    retriever,          # BM25Retriever instance (anchor config)
) -> dict[int, float]:
    """
    For each question compute the mean BM25 score across its ground-truth articles.
    """
    id_to_idx = {aid: i for i, aid in enumerate(retriever._article_ids)}
    scores: dict[int, float] = {}

    for q in questions:
        tokens     = retriever.tokenize(q["question_text"])
        doc_scores = retriever.bm25.get_scores(tokens)
        gt_ids     = q["relevant_article_ids"]
        gt_sc      = [doc_scores[id_to_idx[aid]] for aid in gt_ids if aid in id_to_idx]
        scores[q["question_id"]] = float(np.mean(gt_sc)) if gt_sc else 0.0

    return scores


def compute_strata(
    questions: list[dict],
    anchor_retriever,           # BM25Retriever(normalization="none", field_weighting="text_only")
    cross_ref_map: dict[int, bool] | None = None,
) -> dict[int, dict]:
    """
    Compute all strata for a list of questions.

    Parameters
    ----------
    questions        : full list of questions (any subset)
    anchor_retriever : BM25Retriever with normalization="none", field_weighting="text_only"
    cross_ref_map    : optional {article_id: has_cross_references bool}

    Returns
    -------
    {question_id: {"lex_align": str, "article_count": str, "cross_ref": str}}
    """
    bm25_sc = _bm25_scores_for_gt(questions, anchor_retriever)
    vals    = list(bm25_sc.values())
    q25     = float(np.percentile(vals, 25))
    q75     = float(np.percentile(vals, 75))

    strata: dict[int, dict] = {}
    for q in questions:
        qid   = q["question_id"]
        score = bm25_sc[qid]

        lex = (
            "lexically_aligned"        if score >= q75
            else "semantically_paraphrased" if score <= q25
            else "middle"
        )

        art_count = "single_article" if q["n_relevant_articles"] == 1 else "multi_article"

        cross_ref = "unknown"
        if cross_ref_map is not None:
            has = any(cross_ref_map.get(aid, False) for aid in q["relevant_article_ids"])
            cross_ref = "with_cross_refs" if has else "without_cross_refs"

        strata[qid] = {
            "bm25_score":  score,
            "lex_align":   lex,
            "article_count": art_count,
            "cross_ref":   cross_ref,
        }

    return strata


def save_strata(strata: dict[int, dict]) -> None:
    STRATA_FILE.write_text(json.dumps(
        {str(k): v for k, v in strata.items()}, indent=2
    ))


def load_strata() -> dict[int, dict] | None:
    if not STRATA_FILE.exists():
        return None
    raw = json.loads(STRATA_FILE.read_text())
    return {int(k): v for k, v in raw.items()}


# ---------------------------------------------------------------------------
# Helpers used by the runner to slice results into stratum sub-dicts
# ---------------------------------------------------------------------------
# NOTE: filter_by() has moved to bsard_evaluation.stratify.
# This re-export is kept for backward compatibility with legacy callers.
# New code should import from bsard_evaluation directly.

def filter_by(
    ground_truth: dict[int, list[int]],
    strata: dict[int, dict],
    field: str,
    value: str,
) -> dict[int, list[int]]:
    """
    Return the ground-truth subset matching strata[qid][field] == value.

    DEPRECATED: Use bsard_evaluation.filter_by for TREC-format data.
    This legacy version accepts RQ1's int-keyed list-based format.
    """
    return {
        qid: gt for qid, gt in ground_truth.items()
        if strata.get(qid, {}).get(field) == value
    }
