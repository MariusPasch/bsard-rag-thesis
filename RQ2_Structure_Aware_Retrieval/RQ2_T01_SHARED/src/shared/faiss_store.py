"""FAISS IndexFlatIP vector store with metadata support."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Callable

import faiss
import numpy as np

from .utils import setup_logger

logger = setup_logger(__name__)

_INDEX_FILE = "faiss.index"
_META_FILE = "faiss_meta.json"


@dataclass
class SearchResult:
    id: str
    score: float
    metadata: dict = field(default_factory=dict)


class FAISSStore:
    """FAISS IndexFlatIP store with parallel metadata list.

    Metadata is stored as a plain list[dict] and pickled to disk so that
    list[int] values round-trip without coercion.
    """

    def __init__(self, dimension: int) -> None:
        self._dimension = dimension
        self._index: faiss.IndexFlatIP = faiss.IndexFlatIP(dimension)
        self._ids: list[str] = []
        self._metadata: list[dict] = []

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def add(
        self,
        ids: list[str],
        vectors: np.ndarray,
        metadata: list[dict],
    ) -> None:
        """Add vectors with string IDs and metadata dicts."""
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self._dimension:
            raise ValueError(
                f"Expected shape (N, {self._dimension}), got {vectors.shape}"
            )
        self._index.add(vectors)
        self._ids.extend(ids)
        self._metadata.extend(metadata)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 100,
        filter_fn: Callable[[dict], bool] | None = None,
        fetch_k: int = 500,
    ) -> list[SearchResult]:
        """Search for nearest neighbours with optional metadata post-filtering.

        fetch_k candidates are retrieved from FAISS, then filter_fn is applied
        and the first k passing results are returned.
        """
        query = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        n_stored = self._index.ntotal
        if n_stored == 0:
            return []

        retrieve = min(fetch_k if filter_fn else k, n_stored)
        scores, indices = self._index.search(query, retrieve)

        results: list[SearchResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            meta = self._metadata[idx]
            if filter_fn is not None and not filter_fn(meta):
                continue
            results.append(SearchResult(id=self._ids[idx], score=float(score), metadata=meta))
            if len(results) >= k:
                break

        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save index and metadata to *path* (a directory)."""
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self._index, os.path.join(path, _INDEX_FILE))
        payload = {"ids": self._ids, "metadata": self._metadata, "dimension": self._dimension}
        with open(os.path.join(path, _META_FILE), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        logger.info("FAISSStore saved to %s (%d vectors)", path, self._index.ntotal)

    @classmethod
    def load(cls, path: str) -> FAISSStore:
        """Load index and metadata from *path*."""
        with open(os.path.join(path, _META_FILE), "r", encoding="utf-8") as f:
            payload = json.load(f)
        store = cls(dimension=payload["dimension"])
        store._index = faiss.read_index(os.path.join(path, _INDEX_FILE))
        store._ids = payload["ids"]
        store._metadata = payload["metadata"]
        logger.info("FAISSStore loaded from %s (%d vectors)", path, store._index.ntotal)
        return store

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._index.ntotal
