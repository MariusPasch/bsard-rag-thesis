"""Embedding model loading and encoding (mE5-instruct / BGE-M3)."""
from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from .utils import setup_logger

logger = setup_logger(__name__)

_ME5_INSTRUCT_NAMES = ("multilingual-e5-large-instruct",)


def _is_me5_instruct(model_name: str) -> bool:
    lower = model_name.lower()
    return any(tag in lower for tag in _ME5_INSTRUCT_NAMES)


class EmbeddingModel:
    """Sentence-transformer wrapper with lazy initialization and automatic prefix handling."""

    def __init__(self, model_name: str, max_tokens: int = 512, fallback_model: str | None = None) -> None:
        self._model_name = model_name
        self._max_tokens = max_tokens
        self._fallback_model = fallback_model
        self._model: SentenceTransformer | None = None
        self._use_me5_prefix = _is_me5_instruct(model_name)

    # ------------------------------------------------------------------
    # Lazy initialisation
    # ------------------------------------------------------------------

    def _load(self) -> SentenceTransformer:
        if self._model is None:
            logger.info("Loading embedding model: %s", self._model_name)
            self._model = SentenceTransformer(self._model_name)
            logger.info("Embedding model loaded (dim=%d)", self.dimension)
        return self._model

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def dimension(self) -> int:
        """Embedding dimensionality."""
        return self._load().get_sentence_embedding_dimension()

    @property
    def tokenizer(self):
        """The underlying HuggingFace fast tokenizer used by this model.

        Supports return_offsets_mapping=True for chunk-boundary alignment.
        """
        return self._load().tokenizer

    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Encode texts to L2-normalised vectors, shape (N, dim).

        For mE5-instruct models each text is prefixed with 'passage: '.
        """
        model = self._load()
        if self._use_me5_prefix:
            texts = [f"passage: {t}" for t in texts]
        vectors = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vectors.astype(np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query to a normalised vector, shape (dim,).

        For mE5-instruct models the 'query: ' prefix is applied automatically.
        """
        model = self._load()
        text = f"query: {query}" if self._use_me5_prefix else query
        vector = model.encode(
            [text],
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vector[0].astype(np.float32)
