"""
Dense bi-encoder retriever for the BSARD corpus.

Architecture
------------
  EmbeddingEncoder  — thin wrapper around sentence-transformers SentenceTransformer.
                      Handles prefix injection, L2 normalisation, token audit.
                      Supports Matryoshka dimension truncation via truncate_dim.
  DenseRetriever    — public retriever with the same interface as BM25Retriever.
                      Embeds corpus once, persists to .npy, builds FAISS IndexFlatIP.

Module-level cache
------------------
  _EMBEDDING_CACHE: dict[(model_slug, field_weighting)] -> (embeddings, article_ids)
  Mirrors _TOKENIZED_CACHE in sparse.py — prevents re-loading from disk when
  running multiple experiments with the same model in a single process.

Embedding persistence convention
---------------------------------
  output/embeddings/{model_slug}_{field_weighting}.npy         — shape (N, dim), float32
  output/embeddings/{model_slug}_{field_weighting}_ids.npy     — shape (N,),     int64

  model_slug = model_name with / and - replaced by _, lower-cased.
  If truncate_dim is set, slug is suffixed with _dim{truncate_dim} to avoid
  cache collisions between full-dim and truncated-dim embeddings.
  FAISS indices are rebuilt at load time (<1 s for 22k articles); not persisted.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from retrieval.preprocessing import DOC_BUILDERS

# ---------------------------------------------------------------------------
# Module-level embedding cache
# ---------------------------------------------------------------------------

_EMBEDDING_CACHE: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}


def _slugify(model_name: str) -> str:
    """Convert a HuggingFace model ID to a safe filename stem."""
    return model_name.replace("/", "_").replace("-", "_").lower()


# ---------------------------------------------------------------------------
# EmbeddingEncoder
# ---------------------------------------------------------------------------

class EmbeddingEncoder:
    """
    Wraps sentence-transformers SentenceTransformer.

    Parameters
    ----------
    model_name           : HuggingFace model ID or local path
    device               : "cpu" | "cuda" | "mps"
    max_seq_length_override : if set, overrides model.max_seq_length before
                              encoding (use 1024 for BAAI/bge-m3)
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        max_seq_length_override: int | None = None,
        truncate_dim: int | None = None,
    ):
        from sentence_transformers import SentenceTransformer

        self.model_name   = model_name
        self.device       = device
        self.truncate_dim = truncate_dim

        kwargs: dict = {"device": device}
        if truncate_dim is not None:
            kwargs["truncate_dim"] = truncate_dim

        self._model = SentenceTransformer(model_name, **kwargs)

        if max_seq_length_override is not None:
            self._model.max_seq_length = max_seq_length_override

        self.max_seq_length = self._model.max_seq_length

    # ── Token audit ─────────────────────────────────────────────────────────

    def token_length_audit(self, texts: list[str]) -> dict:
        """
        Return token-length statistics for *texts* under this model's tokenizer.
        Used to measure truncation risk before encoding the full corpus.
        Falls back to a character-based approximation if tokenizer access fails.
        """
        lengths = None
        try:
            try:
                tokenizer = self._model.tokenizer
            except AttributeError:
                tokenizer = self._model[0].tokenizer
            lengths = [len(tokenizer.encode(t, add_special_tokens=True)) for t in texts]
        except Exception:
            # Fallback: rough token estimate from character count (~4 chars/token)
            lengths = [min(max(1, len(t) // 4), 99999) for t in texts]

        arr = np.array(lengths, dtype=np.float32)
        fraction_truncated = float(np.mean(arr > self.max_seq_length))

        return {
            "mean_tokens":          float(np.mean(arr)),
            "median_tokens":        float(np.median(arr)),
            "p90_tokens":           float(np.percentile(arr, 90)),
            "max_tokens_observed":  int(np.max(arr)),
            "fraction_truncated":   fraction_truncated,
        }

    # ── Encoding ─────────────────────────────────────────────────────────────

    def encode_corpus(
        self,
        texts: list[str],
        batch_size: int = 64,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Encode *texts* in batches. Returns L2-normalised float32 array (N, dim).
        """
        import faiss

        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        faiss.normalize_L2(embeddings)  # ensure unit norm (encode may not guarantee it)
        return embeddings

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a single query string. Returns L2-normalised float32 vector (dim,).
        """
        import faiss

        vec = self._model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)
        faiss.normalize_L2(vec)
        return vec[0]


# ---------------------------------------------------------------------------
# DenseRetriever
# ---------------------------------------------------------------------------

class DenseRetriever:
    """
    Dense bi-encoder retriever backed by FAISS IndexFlatIP.

    Parameters
    ----------
    df                   : DataFrame from the BSARD corpus (HuggingFace or local).
                           Must have columns: article_id, article_text
    model_name           : HuggingFace model ID or local path
    field_weighting      : "text_only" | "concat_2x" (mirrors sparse.py convention)
    passage_prefix       : prepended to each article at encoding time
    query_prefix         : prepended to each query at retrieval time
    batch_size           : encoding batch size
    device               : "cpu" | "cuda" | "mps"
    embeddings_dir       : directory for persisted .npy files (output/embeddings/)
    max_seq_length_override : if set, overrides model.max_seq_length before encoding
                              (use 1024 for BAAI/bge-m3 to avoid 8192-token padding)
    truncate_dim         : Matryoshka dimension truncation. If set, embeddings are
                           truncated to this many dimensions after encoding. Required
                           for Qwen3-Embedding-0.6B (use truncate_dim=1024).
                           Suffix _dim{truncate_dim} is appended to the embedding
                           filename slug to avoid cache collisions.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        model_name: str,
        field_weighting: str = "text_only",
        passage_prefix: str = "",
        query_prefix: str = "",
        batch_size: int = 64,
        device: str = "cpu",
        embeddings_dir: Path = Path("output/embeddings"),
        max_seq_length_override: int | None = None,
        truncate_dim: int | None = None,
    ):
        import faiss

        self._model_name      = model_name
        self._field_weighting = field_weighting
        self._query_prefix    = query_prefix
        self._batch_size      = batch_size

        slug = _slugify(model_name)
        if truncate_dim is not None:
            slug = f"{slug}_dim{truncate_dim}"
        cache_key    = (slug, field_weighting)
        emb_path     = Path(embeddings_dir) / f"{slug}_{field_weighting}.npy"
        ids_path     = Path(embeddings_dir) / f"{slug}_{field_weighting}_ids.npy"

        # ── Step 1: encoder ─────────────────────────────────────────────────
        self._encoder = EmbeddingEncoder(
            model_name, device=device,
            max_seq_length_override=max_seq_length_override,
            truncate_dim=truncate_dim,
        )

        # ── Step 2: token audit ─────────────────────────────────────────────
        texts_for_audit = df["article_text"].fillna("").tolist()
        self._audit = self._encoder.token_length_audit(texts_for_audit)

        # ── Step 3: embeddings (cache → disk → encode) ──────────────────────
        # Time steps 3+4 together for Tier 0 index_build_time_s.
        # "from_cache" path: in-process cache hit, no disk I/O — build time ≈ 0.
        # "from_disk" path: np.load + FAISS index rebuild.
        # "encode" path: full corpus encoding + save + FAISS index rebuild.
        _t_build = time.perf_counter()

        if cache_key in _EMBEDDING_CACHE:
            embeddings, article_ids = _EMBEDDING_CACHE[cache_key]
            _build_source = "cache"
        elif emb_path.exists() and ids_path.exists():
            embeddings   = np.load(emb_path)
            article_ids  = np.load(ids_path)
            _EMBEDDING_CACHE[cache_key] = (embeddings, article_ids)
            _build_source = "disk"
        else:
            doc_builder = DOC_BUILDERS[field_weighting]
            texts = [
                passage_prefix + doc_builder(row)
                for row in df.to_dict("records")
            ]
            print(f"  Encoding {len(texts):,} docs ({model_name}, fw={field_weighting}) ...")
            embeddings  = self._encoder.encode_corpus(texts, batch_size=batch_size)
            article_ids = df["article_id"].to_numpy(dtype=np.int64)

            Path(embeddings_dir).mkdir(parents=True, exist_ok=True)
            np.save(emb_path, embeddings)
            np.save(ids_path, article_ids)
            _EMBEDDING_CACHE[cache_key] = (embeddings, article_ids)
            _build_source = "encode"

        self._article_ids = article_ids

        # ── Step 4: FAISS index ─────────────────────────────────────────────
        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(embeddings)

        self._index_build_time_s  = time.perf_counter() - _t_build
        self._index_build_source  = _build_source  # "cache" | "disk" | "encode"

    # ── Public interface ─────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 500) -> tuple[list[int], float]:
        """
        Encode *query* and return the top-k article_ids by cosine similarity.

        Returns
        -------
        (ranked_article_ids, latency_ms)
        Latency covers query encoding + FAISS search only (not index load).
        """
        t0 = time.perf_counter()

        q_text = self._query_prefix + query
        q_vec  = self._encoder.encode_query(q_text).reshape(1, -1)

        k = min(top_k, self._index.ntotal)
        _, faiss_indices = self._index.search(q_vec, k)

        ranked = self._article_ids[faiss_indices[0]].tolist()
        latency_ms = (time.perf_counter() - t0) * 1000.0

        return ranked, latency_ms

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def audit(self) -> dict:
        """Token length audit dict — read by evaluation/runner.py."""
        return self._audit

    @property
    def embedding_dim(self) -> int:
        return self._index.d

    @property
    def n_articles(self) -> int:
        return self._index.ntotal
