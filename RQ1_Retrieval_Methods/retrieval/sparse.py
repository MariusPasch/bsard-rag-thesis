"""
Sparse retrievers for Tier 1.

Classes
-------
BM25Retriever   — rank_bm25 (Okapi / BM25+ / BM25L)
TFIDFRetriever  — scikit-learn TF-IDF with cosine similarity
FTS5Retriever   — SQLite FTS5 built-in BM25 (text_only, no external normalisation)

All retrievers expose a uniform interface:
    retriever.retrieve(query: str, top_k: int = 100) -> tuple[list[int], float]
    Returns (ranked article_ids, latency_ms).
"""

from __future__ import annotations

import pickle
import re
import sqlite3
import time
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from retrieval.preprocessing import (
    DOC_BUILDERS,
    TOKENIZERS,
    batch_tokenize,
)

# ---------------------------------------------------------------------------
# BM25 retriever
# ---------------------------------------------------------------------------

BM25Variant = Literal["okapi", "plus", "l"]

# Module-level cache: (normalization, field_weighting) -> tokenized corpus list
# Avoids re-running spaCy pipeline when only k1/b or variant changes.
_TOKENIZED_CACHE: dict[tuple[str, str], list[list[str]]] = {}

_DISK_CACHE_DIR = Path(__file__).parent.parent / "output" / "cache"


def _disk_cache_path(normalization: str, field_weighting: str) -> Path:
    return _DISK_CACHE_DIR / f"tokenized_{normalization}_{field_weighting}.pkl"


def _load_disk_cache(normalization: str, field_weighting: str) -> list[list[str]] | None:
    p = _disk_cache_path(normalization, field_weighting)
    if p.exists():
        print(f"      Loading tokenized corpus from disk cache ({p.name}) ...",
              end=" ", flush=True)
        with open(p, "rb") as f:
            data = pickle.load(f)
        print("done")
        return data
    return None


def _save_disk_cache(normalization: str, field_weighting: str,
                     tokenized: list[list[str]]) -> None:
    _DISK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = _disk_cache_path(normalization, field_weighting)
    with open(p, "wb") as f:
        pickle.dump(tokenized, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"      Tokenized corpus cached to disk ({p.name})")


class BM25Retriever:
    """
    BM25 retriever backed by rank_bm25.

    Parameters
    ----------
    df            : DataFrame from bsard_articles_only.parquet
    normalization : "none" | "lemmatize" | "stem"
    field_weighting: "text_only" | "concat_2x"
    variant       : "okapi" | "plus" | "l"
    k1, b         : BM25Okapi parameters (ignored for plus/l)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        normalization: str = "lemmatize",
        field_weighting: str = "text_only",
        variant: BM25Variant = "okapi",
        k1: float = 1.5,
        b: float = 0.75,
    ):
        from rank_bm25 import BM25L, BM25Okapi, BM25Plus

        _classes = {"okapi": BM25Okapi, "plus": BM25Plus, "l": BM25L}

        self.normalization   = normalization
        self.field_weighting = field_weighting
        self.variant         = variant
        self.k1              = k1
        self.b               = b

        self._tokenize    = TOKENIZERS[normalization]
        self._article_ids = df["article_id"].tolist()

        cache_key = (normalization, field_weighting)
        if cache_key in _TOKENIZED_CACHE:
            tokenized = _TOKENIZED_CACHE[cache_key]
        else:
            tokenized = _load_disk_cache(normalization, field_weighting)
            if tokenized is None:
                build_doc = DOC_BUILDERS[field_weighting]
                raw_docs  = [build_doc(row) for _, row in df.iterrows()]
                print(f"      Tokenizing {len(raw_docs):,} docs "
                      f"(norm={normalization}, fw={field_weighting}) ...",
                      end=" ", flush=True)
                tokenized = batch_tokenize(raw_docs, normalization)
                print("done")
                _save_disk_cache(normalization, field_weighting, tokenized)
            _TOKENIZED_CACHE[cache_key] = tokenized

        cls = _classes[variant]
        if variant == "okapi":
            self.bm25 = cls(tokenized, k1=k1, b=b)
        else:
            self.bm25 = cls(tokenized)

    # T1-D3 — preprocessing symmetry: same tokenizer used here and at query time
    def tokenize(self, text: str) -> list[str]:
        return self._tokenize(text)

    def retrieve(self, query: str, top_k: int = 100) -> tuple[list[int], float]:
        t0     = time.perf_counter()
        tokens = self._tokenize(query)
        scores = self.bm25.get_scores(tokens)
        top_idx = np.argsort(scores)[::-1][:top_k]
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return [self._article_ids[i] for i in top_idx], latency_ms

    @property
    def vocab(self) -> set[str]:
        """All terms present in the BM25 index (for OOV check)."""
        if hasattr(self.bm25, "idf"):
            return set(self.bm25.idf.keys())
        # BM25Plus / BM25L store scores differently
        if hasattr(self.bm25, "df"):
            return set(self.bm25.df.keys())
        return set()


# ---------------------------------------------------------------------------
# TF-IDF retriever
# ---------------------------------------------------------------------------

class TFIDFRetriever:
    """
    TF-IDF retriever using scikit-learn, with cosine similarity retrieval.

    Parameters
    ----------
    df            : DataFrame from bsard_articles_only.parquet
    normalization : "none" | "lemmatize" | "stem"
    field_weighting: "text_only" | "concat_2x"
    """

    def __init__(
        self,
        df: pd.DataFrame,
        normalization: str = "lemmatize",
        field_weighting: str = "text_only",
    ):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.normalization   = normalization
        self.field_weighting = field_weighting
        self._tokenize       = TOKENIZERS[normalization]
        self._article_ids    = df["article_id"].tolist()

        cache_key = (normalization, field_weighting)
        if cache_key in _TOKENIZED_CACHE:
            tokenized_lists = _TOKENIZED_CACHE[cache_key]
        else:
            tokenized_lists = _load_disk_cache(normalization, field_weighting)
            if tokenized_lists is None:
                build_doc = DOC_BUILDERS[field_weighting]
                raw_docs  = [build_doc(row) for _, row in df.iterrows()]
                print(f"      Tokenizing {len(raw_docs):,} docs "
                      f"(norm={normalization}, fw={field_weighting}) ...",
                      end=" ", flush=True)
                tokenized_lists = batch_tokenize(raw_docs, normalization)
                print("done")
                _save_disk_cache(normalization, field_weighting, tokenized_lists)
            _TOKENIZED_CACHE[cache_key] = tokenized_lists
        tokenized_str = [" ".join(toks) for toks in tokenized_lists]

        self._vectorizer = TfidfVectorizer(
            analyzer="word",
            token_pattern=r"\S+",   # space-split tokens already processed
            sublinear_tf=True,
        )
        self._matrix = self._vectorizer.fit_transform(tokenized_str)

    def tokenize(self, text: str) -> list[str]:
        return self._tokenize(text)

    def retrieve(self, query: str, top_k: int = 100) -> tuple[list[int], float]:
        from sklearn.metrics.pairwise import cosine_similarity

        t0    = time.perf_counter()
        q_str = " ".join(self._tokenize(query))
        q_vec = self._vectorizer.transform([q_str])
        scores = cosine_similarity(q_vec, self._matrix).flatten()
        top_idx    = np.argsort(scores)[::-1][:top_k]
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return [self._article_ids[i] for i in top_idx], latency_ms


# ---------------------------------------------------------------------------
# FTS5 retriever
# ---------------------------------------------------------------------------

# Matches runs of French/Latin word characters (no apostrophes, dots, operators)
_FTS5_TERM_RE = re.compile(
    r"[a-zA-Z0-9àâäãæçéèêëîïíìôöóòœùûüúñýÿÀÂÄÃÆÇÉÈÊËÎÏÍÌÔÖÓÒŒÙÛÜÚÑÝŸ]+"
)


def _escape_fts5(query: str) -> str:
    """
    Convert a raw French query into a safe FTS5 OR expression.

    Space-separated terms in FTS5 use implicit AND — a single missing term
    (e.g. an abbreviation not in any article) zeroes the whole result set.
    We use explicit OR so FTS5 BM25 ranks documents by how many query terms
    they contain, which is the correct bag-of-words retrieval behaviour.

    Terms are lowercased to match the unicode61 case-folding used at index time.
    """
    terms = _FTS5_TERM_RE.findall(query.lower())
    # Deduplicate while preserving order
    seen: set[str] = set()
    unique = [t for t in terms if not (t in seen or seen.add(t))]
    return " OR ".join(unique) if unique else ""


class FTS5Retriever:
    """
    SQLite FTS5 built-in BM25 retriever.

    Uses the pre-built `articles_fts` virtual table (unicode61 tokenizer).
    No external normalisation — this is a separate baseline, not part of
    the normalization ablation grid.

    Parameters
    ----------
    db_path : path to bsard_corpus.db
    """

    def __init__(self, db_path: str | Path = "output/bsard_corpus.db"):
        self._db_path = str(db_path)
        # Test connection immediately to fail fast
        conn = sqlite3.connect(self._db_path)
        conn.close()

    def retrieve(self, query: str, top_k: int = 100) -> tuple[list[int], float]:
        from retrieval.preprocessing import get_stopwords
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        t0 = time.perf_counter()

        # Strip stopwords (except legal keeps) from query terms before FTS5
        stops = get_stopwords()
        safe_query = _escape_fts5(query)
        if not safe_query:
            conn.close()
            return [], (time.perf_counter() - t0) * 1000.0

        # Remove high-frequency stopword terms that swamp BM25 ranking
        filtered_terms = [
            t for t in safe_query.split(" OR ") if t not in stops
        ]
        if not filtered_terms:
            filtered_terms = safe_query.split(" OR ")  # fallback: keep all
        safe_query = " OR ".join(filtered_terms)

        try:
            rows = conn.execute(
                """
                SELECT a.article_id
                FROM   articles_fts
                JOIN   articles a ON a.article_id = articles_fts.rowid
                WHERE  articles_fts MATCH ?
                  AND  a.is_bsard_article = 1
                ORDER  BY rank
                LIMIT  ?
                """,
                (safe_query, top_k),
            ).fetchall()
            ranked = [r["article_id"] for r in rows]
        except sqlite3.OperationalError:
            ranked = []
        finally:
            conn.close()
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return ranked, latency_ms
