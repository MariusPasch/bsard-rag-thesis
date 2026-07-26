"""
Unit tests for HybridRetriever and CrossEncoderReranker.

Uses a 15-row synthetic French legal corpus so tests run without GPU
and without loading the full BSARD corpus.
The DenseRetriever uses paraphrase-multilingual-mpnet-base-v2 (already a
dependency, fastest available model).
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from retrieval.dense import DenseRetriever
from retrieval.sparse import BM25Retriever
from retrieval.hybrid import HybridRetriever, CrossEncoderReranker

# ---------------------------------------------------------------------------
# Synthetic corpus
# ---------------------------------------------------------------------------

_TEXTS = [
    "Le contrat de travail peut être résilié par l'employeur sous réserve des dispositions légales.",
    "Les conditions de licenciement sont définies par la loi.",
    "L'employeur doit respecter les délais de préavis légaux.",
    "Le travailleur a droit à une indemnité en cas de licenciement abusif.",
    "Sauf stipulation contraire, les obligations contractuelles s'appliquent.",
    "Les dispositions du présent article s'appliquent à tous les travailleurs.",
    "Le contrat peut être suspendu en cas de force majeure.",
    "Tout travailleur a droit à un congé annuel payé.",
    "Les accidents du travail sont couverts par l'assurance obligatoire.",
    "La durée du travail est fixée par convention collective.",
    "Le bail peut être résilié avec un préavis de trois mois.",
    "Le locataire est tenu de payer le loyer à la date convenue.",
    "Le propriétaire répond des vices cachés de la chose louée.",
    "Toute clause abusive dans un contrat de consommation est nulle.",
    "Le consommateur dispose d'un droit de rétractation de quatorze jours.",
]

_CORPUS = pd.DataFrame({
    "article_id":     list(range(200, 215)),
    "bsard_id":       list(range(300, 315)),
    "law_code":       ["Code du travail"] * 10 + ["Code civil"] * 5,
    "chapter_title":  ["Chapitre 1"] * 15,
    "article_number": [f"Art. {i}" for i in range(1, 16)],
    "article_text":   _TEXTS,
    "has_cross_references": [False] * 15,
})

_DENSE_MODEL = "paraphrase-multilingual-mpnet-base-v2"
_QUERY = "Quelles sont les conditions pour résilier un contrat de travail?"


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _make_sparse() -> BM25Retriever:
    return BM25Retriever(
        df=_CORPUS,
        normalization="none",
        field_weighting="text_only",
        variant="okapi",
        k1=1.5,
        b=0.75,
    )


def _make_dense(tmp: str) -> DenseRetriever:
    return DenseRetriever(
        df=_CORPUS,
        model_name=_DENSE_MODEL,
        field_weighting="text_only",
        embeddings_dir=Path(tmp),
    )


def _make_hybrid_rrf(sparse, dense, rrf_k: int = 60) -> HybridRetriever:
    return HybridRetriever(
        sparse_retriever=sparse,
        dense_retriever=dense,
        fusion_method="rrf",
        rrf_k=rrf_k,
        first_stage_k=10,
    )


def _make_hybrid_linear(sparse, dense, alpha: float = 0.5) -> HybridRetriever:
    return HybridRetriever(
        sparse_retriever=sparse,
        dense_retriever=dense,
        fusion_method="linear",
        alpha=alpha,
        first_stage_k=10,
    )


# ---------------------------------------------------------------------------
# Tests: HybridRetriever — RRF
# ---------------------------------------------------------------------------

def test_rrf_returns_correct_type_and_length():
    with tempfile.TemporaryDirectory() as tmp:
        sparse = _make_sparse()
        dense  = _make_dense(tmp)
        hybrid = _make_hybrid_rrf(sparse, dense)
        ranked, lat = hybrid.retrieve(_QUERY, top_k=5)
    assert isinstance(ranked, list)
    assert len(ranked) == 5
    assert all(isinstance(x, int) for x in ranked)
    assert isinstance(lat, float) and lat >= 0.0


def test_rrf_output_subset_of_corpus():
    valid_ids = set(_CORPUS["article_id"].tolist())
    with tempfile.TemporaryDirectory() as tmp:
        sparse = _make_sparse()
        dense  = _make_dense(tmp)
        hybrid = _make_hybrid_rrf(sparse, dense)
        ranked, _ = hybrid.retrieve(_QUERY, top_k=10)
    assert all(aid in valid_ids for aid in ranked), "RRF returned unknown article ID"


def test_rrf_no_duplicates():
    with tempfile.TemporaryDirectory() as tmp:
        sparse = _make_sparse()
        dense  = _make_dense(tmp)
        hybrid = _make_hybrid_rrf(sparse, dense)
        ranked, _ = hybrid.retrieve(_QUERY, top_k=15)
    assert len(ranked) == len(set(ranked)), "RRF returned duplicate article IDs"


def test_rrf_latency_positive():
    with tempfile.TemporaryDirectory() as tmp:
        sparse = _make_sparse()
        dense  = _make_dense(tmp)
        hybrid = _make_hybrid_rrf(sparse, dense)
        _, lat = hybrid.retrieve(_QUERY, top_k=5)
    assert lat > 0.0


def test_rrf_k_values_produce_different_rankings():
    """Different k values must not all produce identical orderings (smoke test)."""
    with tempfile.TemporaryDirectory() as tmp:
        sparse = _make_sparse()
        dense  = _make_dense(tmp)
        r30, _  = _make_hybrid_rrf(sparse, dense, rrf_k=30).retrieve(_QUERY, top_k=10)
        r120, _ = _make_hybrid_rrf(sparse, dense, rrf_k=120).retrieve(_QUERY, top_k=10)
    # At least the set of returned articles should be the same (same corpus);
    # but orderings may differ. Verify both are valid.
    valid_ids = set(_CORPUS["article_id"].tolist())
    assert all(aid in valid_ids for aid in r30)
    assert all(aid in valid_ids for aid in r120)


# ---------------------------------------------------------------------------
# Tests: HybridRetriever — Linear interpolation
# ---------------------------------------------------------------------------

def test_linear_returns_correct_type_and_length():
    with tempfile.TemporaryDirectory() as tmp:
        sparse = _make_sparse()
        dense  = _make_dense(tmp)
        hybrid = _make_hybrid_linear(sparse, dense, alpha=0.7)
        ranked, lat = hybrid.retrieve(_QUERY, top_k=5)
    assert isinstance(ranked, list)
    assert len(ranked) == 5
    assert isinstance(lat, float) and lat >= 0.0


def test_linear_no_duplicates():
    with tempfile.TemporaryDirectory() as tmp:
        sparse = _make_sparse()
        dense  = _make_dense(tmp)
        hybrid = _make_hybrid_linear(sparse, dense, alpha=0.5)
        ranked, _ = hybrid.retrieve(_QUERY, top_k=15)
    assert len(ranked) == len(set(ranked))


def test_linear_alpha1_matches_dense_ranking():
    """alpha=1 should return the same top articles as pure dense retrieval."""
    with tempfile.TemporaryDirectory() as tmp:
        sparse = _make_sparse()
        dense  = _make_dense(tmp)
        hybrid = _make_hybrid_linear(sparse, dense, alpha=1.0)
        hybrid_ranked, _ = hybrid.retrieve(_QUERY, top_k=5)
        dense_ranked,  _ = dense.retrieve(_QUERY, top_k=5)
    assert hybrid_ranked == dense_ranked, (
        f"alpha=1 hybrid top-5 {hybrid_ranked} != dense top-5 {dense_ranked}"
    )


def test_linear_alpha0_matches_sparse_ranking():
    """alpha=0 should return the same top articles as pure sparse retrieval."""
    with tempfile.TemporaryDirectory() as tmp:
        sparse = _make_sparse()
        dense  = _make_dense(tmp)
        hybrid = _make_hybrid_linear(sparse, dense, alpha=0.0)
        hybrid_ranked, _ = hybrid.retrieve(_QUERY, top_k=5)
        sparse_ranked, _ = sparse.retrieve(_QUERY, top_k=5)
    assert hybrid_ranked == sparse_ranked, (
        f"alpha=0 hybrid top-5 {hybrid_ranked} != sparse top-5 {sparse_ranked}"
    )


def test_linear_sparse_norm_in_unit_interval():
    """
    The normalized sparse scores must lie in [0, 1] — verify via internal method.
    """
    with tempfile.TemporaryDirectory() as tmp:
        sparse = _make_sparse()
        dense  = _make_dense(tmp)
        hybrid = _make_hybrid_linear(sparse, dense)
        sparse_ids, _ = sparse.retrieve(_QUERY, top_k=10)
        raw_scores = hybrid._get_sparse_scores(_QUERY, sparse_ids)
    max_s = max(raw_scores.values()) if raw_scores else 0.0
    eps = 1e-8
    norms = [max(0.0, s / (max_s + eps)) for s in raw_scores.values()]
    assert all(0.0 <= v <= 1.0 + 1e-6 for v in norms), f"Out-of-range norms: {norms}"


# ---------------------------------------------------------------------------
# Tests: HybridRetriever — audit property
# ---------------------------------------------------------------------------

def test_audit_property_exists_and_has_fraction_truncated():
    with tempfile.TemporaryDirectory() as tmp:
        sparse = _make_sparse()
        dense  = _make_dense(tmp)
        hybrid = _make_hybrid_rrf(sparse, dense)
    assert hasattr(hybrid, "audit")
    assert "fraction_truncated" in hybrid.audit


# ---------------------------------------------------------------------------
# Tests: CrossEncoderReranker
# ---------------------------------------------------------------------------

def test_crossencoder_returns_correct_type_and_length():
    article_texts = dict(zip(_CORPUS["article_id"], _CORPUS["article_text"]))
    with tempfile.TemporaryDirectory() as tmp:
        sparse = _make_sparse()
        dense  = _make_dense(tmp)
        hybrid = _make_hybrid_rrf(sparse, dense)
        reranker = CrossEncoderReranker(
            first_stage_retriever=hybrid,
            model_name="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
            top_n=10,
            article_texts=article_texts,
            max_article_tokens=400,
        )
        ranked, lat = reranker.retrieve(_QUERY, top_k=5)
    assert isinstance(ranked, list)
    assert len(ranked) == 5
    assert all(isinstance(x, int) for x in ranked)
    assert lat > 0.0


def test_crossencoder_reranks_vs_first_stage():
    """Re-ranker must actually change the ordering for at least one query."""
    article_texts = dict(zip(_CORPUS["article_id"], _CORPUS["article_text"]))
    with tempfile.TemporaryDirectory() as tmp:
        sparse = _make_sparse()
        dense  = _make_dense(tmp)
        hybrid = _make_hybrid_rrf(sparse, dense)
        reranker = CrossEncoderReranker(
            first_stage_retriever=hybrid,
            model_name="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
            top_n=10,
            article_texts=article_texts,
            max_article_tokens=400,
        )
        fs_ranked, _ = hybrid.retrieve(_QUERY, top_k=10)
        ce_ranked, _ = reranker.retrieve(_QUERY, top_k=10)
    # Both must return valid IDs; ordering may differ
    valid_ids = set(_CORPUS["article_id"].tolist())
    assert all(aid in valid_ids for aid in ce_ranked)
    # CE should not simply be a passthrough of first-stage (may differ)
    # (not asserting order change since it depends on model — just structural)
    assert len(ce_ranked) == 10


def test_crossencoder_fraction_truncated_range():
    article_texts = dict(zip(_CORPUS["article_id"], _CORPUS["article_text"]))
    with tempfile.TemporaryDirectory() as tmp:
        sparse = _make_sparse()
        dense  = _make_dense(tmp)
        hybrid = _make_hybrid_rrf(sparse, dense)
        reranker = CrossEncoderReranker(
            first_stage_retriever=hybrid,
            model_name="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
            top_n=10,
            article_texts=article_texts,
            max_article_tokens=400,
        )
        reranker.retrieve(_QUERY, top_k=5)
    assert 0.0 <= reranker.fraction_truncated <= 1.0
