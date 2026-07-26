"""BM25 lexical index with French legal stopword list."""
from __future__ import annotations

import pickle
import re
from typing import Callable

from rank_bm25 import BM25Okapi

from .faiss_store import SearchResult
from .utils import setup_logger

logger = setup_logger(__name__)

# Minimal French legal stopword list (function words, not domain-specific terms).
_STOPWORDS = {
    "le", "la", "les", "de", "du", "des", "un", "une", "et", "en",
    "au", "aux", "à", "a", "ce", "ces", "cet", "cette", "qui", "que",
    "qu", "se", "sa", "son", "ses", "il", "elle", "ils", "elles",
    "y", "ou", "où", "est", "sont", "par", "sur", "sous", "dans",
    "avec", "pour", "plus", "mais", "ou", "donc", "ni", "car",
    "ne", "pas", "non", "tout", "tous", "même", "aussi",
}

_TOKEN_RE = re.compile(r"[^\w]+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    tokens = _TOKEN_RE.split(text.lower())
    return [t for t in tokens if t and t not in _STOPWORDS]


# BM25 hyperparameters — see CN-T01-001 / PROJECT_CONTEXT §3.4.
# `b=0.25` matches RQ1's canonical hybrid (much milder length penalty than
# rank_bm25's `b=0.75` default). Empirically required: with `b=0.75`, long
# legal articles (22.7% of BSARD bodies > 512 tokens) are over-penalised.
_BM25_K1_DEFAULT = 1.5
_BM25_B_DEFAULT = 0.25


class BM25Store:
    """BM25 index over a fixed corpus with metadata round-trip safety.

    Metadata is stored as a plain list[dict]; pickle is used for persistence
    so that list[int] values are never coerced.
    """

    def __init__(
        self,
        k1: float = _BM25_K1_DEFAULT,
        b: float = _BM25_B_DEFAULT,
    ) -> None:
        self._bm25: BM25Okapi | None = None
        self._ids: list[str] = []
        self._metadata: list[dict] = []
        self._k1 = k1
        self._b = b

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def build(
        self,
        ids: list[str],
        texts: list[str],
        metadata: list[dict],
    ) -> None:
        """Tokenize and index all documents."""
        tokenized = [_tokenize(t) for t in texts]
        self._bm25 = BM25Okapi(tokenized, k1=self._k1, b=self._b)
        self._ids = list(ids)
        self._metadata = list(metadata)
        logger.info(
            "BM25Store built with %d documents (k1=%.3f, b=%.3f)",
            len(ids), self._k1, self._b,
        )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        k: int = 100,
        filter_fn: Callable[[dict], bool] | None = None,
    ) -> list[SearchResult]:
        """BM25 search with optional metadata post-filtering."""
        if self._bm25 is None:
            raise RuntimeError("BM25Store has not been built yet")

        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)

        # Sort by score descending; apply filter; collect top-k.
        order = scores.argsort()[::-1]
        results: list[SearchResult] = []
        for idx in order:
            meta = self._metadata[idx]
            if filter_fn is not None and not filter_fn(meta):
                continue
            results.append(
                SearchResult(id=self._ids[idx], score=float(scores[idx]), metadata=meta)
            )
            if len(results) >= k:
                break

        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Pickle the entire store to *path* (a file path, not a directory)."""
        payload = {
            "bm25": self._bm25,
            "ids": self._ids,
            "metadata": self._metadata,
            "bm25_params": {"k1": self._k1, "b": self._b},
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(
            "BM25Store saved to %s (%d documents, k1=%.3f, b=%.3f)",
            path, len(self._ids), self._k1, self._b,
        )

    @classmethod
    def load(cls, path: str) -> BM25Store:
        """Load a pickled BM25Store from *path*.

        Pickles saved before CN-T01-001 lack a ``bm25_params`` field; they were
        built with rank_bm25's ``b=0.75`` default and should be rebuilt to
        match the current ``b=0.25`` default. Such caches load with a loud
        warning rather than failing, so callers can decide when to invalidate.
        """
        with open(path, "rb") as f:
            payload = pickle.load(f)
        params = payload.get("bm25_params")
        if params is None:
            logger.warning(
                "BM25Store at %s has no bm25_params (pre-CN-T01-001 cache, "
                "built with b=0.75). Rebuild to use the new b=0.25 default.",
                path,
            )
            store = cls()
        else:
            store = cls(k1=params["k1"], b=params["b"])
        store._bm25 = payload["bm25"]
        store._ids = payload["ids"]
        store._metadata = payload["metadata"]
        logger.info(
            "BM25Store loaded from %s (%d documents, k1=%.3f, b=%.3f)",
            path, len(store._ids), store._k1, store._b,
        )
        return store

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._ids)
