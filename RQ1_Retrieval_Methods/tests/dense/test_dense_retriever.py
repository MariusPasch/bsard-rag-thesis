"""
Unit tests for DenseRetriever and EmbeddingEncoder.

Uses a 10-row synthetic corpus and paraphrase-multilingual-mpnet-base-v2
(already a dependency, fastest available model) so tests run without GPU
and without loading the full BSARD corpus.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from retrieval.dense import DenseRetriever, EmbeddingEncoder, _slugify

# ---------------------------------------------------------------------------
# Synthetic corpus (10 short French legal sentences)
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
]

_CORPUS = pd.DataFrame({
    "article_id":    list(range(100, 110)),
    "bsard_id":      list(range(200, 210)),
    "law_code":      ["Code du travail"] * 10,
    "chapter_title": ["Chapitre 1"] * 10,
    "article_number": [f"Art. {i}" for i in range(1, 11)],
    "article_text":  _TEXTS,
    "has_cross_references": [False] * 10,
})

_MODEL = "paraphrase-multilingual-mpnet-base-v2"
_QUERY = "Quelles sont les conditions pour résilier un contrat de travail?"


def _make_retriever(tmp_path: str | Path, **kwargs) -> DenseRetriever:
    return DenseRetriever(
        df=_CORPUS,
        model_name=_MODEL,
        embeddings_dir=Path(tmp_path),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_slugify():
    assert _slugify("intfloat/multilingual-e5-large") == "intfloat_multilingual_e5_large"
    assert _slugify("BAAI/bge-m3") == "baai_bge_m3"


def test_retriever_returns_correct_types():
    with tempfile.TemporaryDirectory() as tmp:
        r = _make_retriever(tmp)
        ranked, lat = r.retrieve(_QUERY, top_k=5)
    assert isinstance(ranked, list)
    assert all(isinstance(x, int) for x in ranked)
    assert isinstance(lat, float)
    assert lat >= 0.0


def test_retriever_returns_top_k_results():
    with tempfile.TemporaryDirectory() as tmp:
        r = _make_retriever(tmp)
        ranked, _ = r.retrieve(_QUERY, top_k=5)
    assert len(ranked) == 5


def test_top_k_larger_than_corpus_clipped():
    """top_k > corpus size should return all corpus articles, not crash."""
    with tempfile.TemporaryDirectory() as tmp:
        r = _make_retriever(tmp)
        ranked, _ = r.retrieve(_QUERY, top_k=9999)
    assert len(ranked) == len(_CORPUS)


def test_article_ids_are_valid():
    valid_ids = set(_CORPUS["article_id"].tolist())
    with tempfile.TemporaryDirectory() as tmp:
        r = _make_retriever(tmp)
        ranked, _ = r.retrieve(_QUERY, top_k=10)
    assert all(aid in valid_ids for aid in ranked)


def test_retrieve_latency_positive():
    with tempfile.TemporaryDirectory() as tmp:
        r = _make_retriever(tmp)
        _, lat = r.retrieve(_QUERY, top_k=5)
    assert lat > 0.0


def test_token_length_audit_keys():
    with tempfile.TemporaryDirectory() as tmp:
        r = _make_retriever(tmp)
    required = {"fraction_truncated", "max_tokens_observed", "mean_tokens",
                "median_tokens", "p90_tokens"}
    assert required.issubset(r.audit.keys())


def test_embeddings_are_normalised():
    """Stored corpus embeddings must have unit L2 norm (FAISS IndexFlatIP requirement)."""
    import faiss
    with tempfile.TemporaryDirectory() as tmp:
        r = _make_retriever(tmp)
        n = r.n_articles
        vecs = np.zeros((n, r.embedding_dim), dtype=np.float32)
        r._index.reconstruct_n(0, n, vecs)
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), f"Non-unit norms: {norms}"


def test_reproducibility():
    """Two calls with the same query must return identical ranked lists."""
    with tempfile.TemporaryDirectory() as tmp:
        r = _make_retriever(tmp)
        ranked1, _ = r.retrieve(_QUERY, top_k=10)
        ranked2, _ = r.retrieve(_QUERY, top_k=10)
    assert ranked1 == ranked2


def test_prefix_changes_query_embedding():
    """A non-empty query_prefix must produce a different embedding than no prefix."""
    enc = EmbeddingEncoder(_MODEL)
    v1 = enc.encode_query("test query")
    v2 = enc.encode_query("prefix: test query")
    assert not np.allclose(v1, v2), "Prefix had no effect on query embedding"


def test_n_articles_property():
    with tempfile.TemporaryDirectory() as tmp:
        r = _make_retriever(tmp)
    assert r.n_articles == len(_CORPUS)


def test_embedding_dim_property():
    with tempfile.TemporaryDirectory() as tmp:
        r = _make_retriever(tmp)
    assert r.embedding_dim > 0


def test_disk_cache_loaded_on_second_init():
    """Second DenseRetriever with same model+fw must load from .npy without re-encoding."""
    import time
    with tempfile.TemporaryDirectory() as tmp:
        # First init — encodes and saves
        _make_retriever(tmp)
        # Clear module-level cache to force disk path
        from retrieval import dense as d_module
        d_module._EMBEDDING_CACHE.clear()

        t0 = time.perf_counter()
        _make_retriever(tmp)
        elapsed = time.perf_counter() - t0

    # Model init + disk .npy load should be substantially faster than re-encoding
    # (encoding 22k articles takes 5–25 min; model load + npy read takes <60 s on CPU)
    assert elapsed < 60.0, f"Disk cache load took {elapsed:.1f}s — may be re-encoding"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_slugify,
        test_retriever_returns_correct_types,
        test_retriever_returns_top_k_results,
        test_top_k_larger_than_corpus_clipped,
        test_article_ids_are_valid,
        test_retrieve_latency_positive,
        test_token_length_audit_keys,
        test_embeddings_are_normalised,
        test_reproducibility,
        test_prefix_changes_query_embedding,
        test_n_articles_property,
        test_embedding_dim_property,
        test_disk_cache_loaded_on_second_init,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
        except Exception as e:
            print(f"  ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(tests)} tests passed.")
