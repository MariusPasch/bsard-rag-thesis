"""Hybrid retrieval pipeline for Arm 1 (dense + sparse + RRF).

Pipeline:
  1. Dense retrieval   — FAISS top-k via EmbeddingModel
  2. Sparse retrieval  — BM25 top-k
  3. RRF fusion        — Reciprocal Rank Fusion (k=60, parameter-free)

The ``run_arm1()`` entry point ties together chunking, indexing, and retrieval
for a single document bundle and returns a list of ``RetrievalResult`` objects
consumed by the orchestrator and evaluation pipeline.

On cache hits (CACHE_DESIGN.md §4), extract/locate/chunk/embed are skipped
entirely; only FAISS + BM25 are loaded from disk before retrieval starts. Per-
run results are persisted to ``<cache>/<doc_id>/results/<config_hash>/<run_label>.jsonl``.

Requires the shared package:
  ``pip install -e ../RQ2_T01_SHARED``
  ``pip install -e ../RQ2_T02_DATA_LOADER``
"""

from __future__ import annotations

import dataclasses
import logging
import sqlite3
import time
from collections import defaultdict
from pathlib import Path

from arm1_naive import cache as _cache
from arm1_naive import chunker as _chunker
from arm1_naive import indexer as _indexer
from arm1_naive.cache import (
    CacheRoot,
    ConfigCache,
    PdfCache,
    compute_bsard_db_fingerprint,
    compute_config_hash,
    compute_pdf_sha256,
    default_run_label,
)
from arm1_naive.chunker import ArticleSpan, Chunk
from data_loader.models import DocumentBundle, RetrievalResult
from shared.bm25_store import BM25Store
from shared.embeddings import EmbeddingModel
from shared.faiss_store import FAISSStore, SearchResult

logger = logging.getLogger(__name__)


# ── Reciprocal Rank Fusion ────────────────────────────────────────────────────────


def reciprocal_rank_fusion(
    results_a: list["SearchResult"],
    results_b: list["SearchResult"],
    k: int = 60,
) -> list["SearchResult"]:
    """Combine two ranked result lists with RRF: score = Σ 1 / (k + rank).

    Args:
        results_a: First ranked list (e.g. dense results).
        results_b: Second ranked list (e.g. sparse results).
        k:         RRF constant (default 60 — standard parameter, not tuned).

    Returns:
        Merged and re-ranked list of SearchResult objects.
    """
    scores: dict[str, float] = defaultdict(float)
    all_results: dict[str, "SearchResult"] = {}

    for rank, result in enumerate(results_a):
        scores[result.id] += 1.0 / (k + rank + 1)
        all_results[result.id] = result

    for rank, result in enumerate(results_b):
        scores[result.id] += 1.0 / (k + rank + 1)
        all_results.setdefault(result.id, result)

    fused = sorted(scores.items(), key=lambda x: -x[1])
    return [
        SearchResult(id=rid, score=score, metadata=all_results[rid].metadata)
        for rid, score in fused
    ]


# ── Core retrieval function ───────────────────────────────────────────────────────


def retrieve_arm1(
    query: str,
    faiss_store: "FAISSStore",
    bm25_store: "BM25Store",
    embedding_model: "EmbeddingModel",
    retrieval_top_k: int = 100,
    top_k: int = 100,
) -> list["SearchResult"]:
    """Run the Arm 1 retrieval pipeline for a single query.

    Steps:
      1. Dense retrieval via FAISS (top retrieval_top_k).
      2. Sparse retrieval via BM25 (top retrieval_top_k).
      3. Reciprocal Rank Fusion of the two lists.

    Args:
        query:            Natural-language query string.
        faiss_store:      Populated FAISSStore for this document.
        bm25_store:       Populated BM25Store for this document.
        embedding_model:  EmbeddingModel (same as used to build the index).
        retrieval_top_k:  Number of candidates to retrieve from each retriever.
        top_k:            Final number of results returned (default 100).

    Returns:
        List of up to top_k SearchResult objects, RRF-fused.
    """
    query_vec = embedding_model.encode_query(query)

    dense_results = faiss_store.search(query_vec, k=retrieval_top_k)
    sparse_results = bm25_store.search(query, k=retrieval_top_k)

    fused = reciprocal_rank_fusion(dense_results, sparse_results)

    return fused[:top_k]


# ── Helpers ───────────────────────────────────────────────────────────────────────


def _chunking_params(arm1_cfg: dict) -> dict:
    """Extract the chunking parameter dict that contributes to the config hash.

    Only the parameters relevant to a given strategy are included so that
    flipping ``window_size`` doesn't artificially invalidate a recursive-mode
    cache and vice versa.
    """
    strategy = arm1_cfg.get("strategy", "sliding_window")
    if strategy == "sliding_window":
        return {
            "strategy": "sliding_window",
            "window_size": int(arm1_cfg.get("window_size", 512)),
            "stride": int(arm1_cfg.get("stride", 256)),
        }
    if strategy == "recursive":
        return {
            "strategy": "recursive",
            "max_tokens": int(arm1_cfg.get("max_tokens", 512)),
        }
    raise ValueError(f"Unknown chunking strategy: {strategy!r}")


def _resolve_model_names(
    config: dict,
    embedding_model: EmbeddingModel,
    tokenizer,
) -> tuple[str, str]:
    """Return ``(tokenizer_name, embedding_model_name)`` for the cache hash.

    Prefers explicit names from ``config['models']`` so the hash stays stable
    even if the model object's introspected attributes shift between releases.
    """
    embedding_model_name = (
        config.get("models", {}).get("embedding_model")
        or getattr(embedding_model, "_model_name", None)
    )
    if not embedding_model_name:
        raise ValueError(
            "Could not resolve embedding model name. "
            "Set config['models']['embedding_model']."
        )

    tokenizer_name = (
        config.get("models", {}).get("tokenizer")
        or getattr(tokenizer, "name_or_path", None)
        or embedding_model_name
    )
    return tokenizer_name, embedding_model_name


# ── Full pipeline entry point ─────────────────────────────────────────────────────


def run_arm1(
    bundle: "DocumentBundle",
    embedding_model: "EmbeddingModel",
    tokenizer,
    db_path: Path,
    index_dir: Path,
    config: dict,
    force_reindex: bool = False,
    *,
    run_label: str | None = None,
) -> list["RetrievalResult"]:
    """Run the full Arm 1 pipeline for one document bundle.

    Cache-aware order:
      1. Compute ``pdf_sha256`` + ``config_hash`` up front.
      2. If ``configs/<config_hash>/`` exists and validates → load FAISS + BM25
         and skip extract/locate/chunk/embed entirely (this is the hot path).
      3. Otherwise run the full pipeline: extract → locate → chunk → save chunks
         → build indices → write manifest. Each stage caches its output so that
         later runs short-circuit at the earliest valid stage.
      4. Run retrieval per test question and persist the resulting
         ``RetrievalResult`` list to
         ``data/<doc_id>/results/<config_hash>/<run_label>.jsonl``.

    Args:
        bundle:          DocumentBundle from T02_DATA_LOADER (provides
                         ``pdf_path``, ``pdf_filename``, ``raw_text``).
        embedding_model: Loaded EmbeddingModel (shared.embeddings).
        tokenizer:       HuggingFace fast tokenizer matching the embedding model.
        db_path:         Path to ``bsard_corpus.db``.
        index_dir:       Cache root (the ``data/`` link pointing at the shared
                         data root's ``pdf_cache/``). Reused as the legacy
                         ``index_dir`` arg name to preserve the public signature.
        config:          Configuration dict with keys:
                           config['arm1']['strategy']        — "sliding_window" | "recursive"
                           config['arm1']['window_size']     — tokens (sliding window only)
                           config['arm1']['stride']          — tokens (sliding window only)
                           config['arm1']['max_tokens']      — tokens (recursive only)
                           config['arm1']['retrieval_top_k'] — default 100
                           config['arm1']['top_k']           — default 100 (final result cut)
                           config['models']['embedding_model'] — name string for hashing
                           config['models']['tokenizer']     — optional; defaults to embedding model
        force_reindex:   When True, invalidate the current ``config_hash`` folder
                         and rebuild even if the manifest validates.
        run_label:       Filename stem for the per-run results JSONL. Defaults
                         to a UTC timestamp.

    Returns:
        List of RetrievalResult, one per test question.
    """
    arm1_cfg = config.get("arm1", {})
    retrieval_top_k = int(arm1_cfg.get("retrieval_top_k", 100))
    top_k = int(arm1_cfg.get("top_k", 100))
    chunking_params = _chunking_params(arm1_cfg)

    doc_id = Path(bundle.pdf_filename).stem
    pdf_path = Path(bundle.pdf_path)

    cache_root = CacheRoot(index_dir)
    pdf_cache = PdfCache(cache_root, doc_id)

    pdf_sha256 = compute_pdf_sha256(pdf_path)
    tokenizer_name, embedding_model_name = _resolve_model_names(
        config, embedding_model, tokenizer
    )
    config_hash = compute_config_hash(
        pdf_sha256=pdf_sha256,
        tokenizer_name=tokenizer_name,
        embedding_model=embedding_model_name,
        chunking_params=chunking_params,
    )
    cfg_cache = ConfigCache(pdf_cache, config_hash)

    cache_hit = (
        not force_reindex
        and cfg_cache.exists()
        and cfg_cache.validate(
            pdf_sha256=pdf_sha256,
            tokenizer_name=tokenizer_name,
            embedding_model=embedding_model_name,
            chunking_params=chunking_params,
        )
    )

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    try:
        if cache_hit:
            logger.info(
                "[cache hit] doc=%s config=%s — skipping extract/locate/chunk/embed",
                doc_id, config_hash,
            )
            faiss_store, bm25_store = cfg_cache.load_indices()
        else:
            logger.info(
                "[cache miss] doc=%s config=%s — running full pipeline",
                doc_id, config_hash,
            )
            db_fingerprint = compute_bsard_db_fingerprint(conn)

            raw_text = pdf_cache.load_or_extract_text(pdf_path)
            article_spans = pdf_cache.load_or_locate_articles(
                raw_text, bundle.pdf_filename, conn, db_fingerprint
            )

            chunks = _build_chunks(
                raw_text, tokenizer, doc_id, article_spans, arm1_cfg
            )
            cfg_cache.save_chunks(chunks)

            faiss_store, bm25_store = _indexer.build_arm1_index(
                chunks, embedding_model, doc_id, index_dir,
                output_dir=cfg_cache.path,
            )
            cfg_cache.write_manifest(
                pdf_sha256=pdf_sha256,
                tokenizer_name=tokenizer_name,
                embedding_model=embedding_model_name,
                embedding_dim=embedding_model.dimension,
                chunking_params=chunking_params,
                n_chunks=len(chunks),
            )

        # ── Load test questions ─────────────────────────────────────────────
        question_rows = conn.execute(
            "SELECT question_id, question_text FROM questions WHERE split = 'test'"
        ).fetchall()

        if not question_rows:
            return []

        # ── Retrieve per question ───────────────────────────────────────────
        results: list[RetrievalResult] = []

        for row in question_rows:
            q_id = row["question_id"]
            q_text = row["question_text"]

            t0 = time.perf_counter()
            ranked = retrieve_arm1(
                q_text, faiss_store, bm25_store, embedding_model,
                retrieval_top_k=retrieval_top_k,
                top_k=top_k,
            )
            latency_ms = (time.perf_counter() - t0) * 1000

            ranked_items = [
                {
                    "id": r.id,
                    "score": r.score,
                    "text": r.metadata.get("text", ""),
                    "metadata": {
                        "bsard_ids": r.metadata.get("bsard_ids", []),
                        "doc_id": r.metadata.get("doc_id"),
                        "start_char": r.metadata.get("start_char"),
                        "end_char": r.metadata.get("end_char"),
                        "start_token": r.metadata.get("start_token"),
                        "end_token": r.metadata.get("end_token"),
                    },
                }
                for r in ranked
            ]

            results.append(
                RetrievalResult(
                    query_id=str(q_id),
                    query_text=q_text,
                    ranked_items=ranked_items,
                    method="arm1",
                    cost={
                        "llm_calls": 0,
                        "tokens_in": 0,
                        "tokens_out": 0,
                        "latency_ms": round(latency_ms, 1),
                    },
                )
            )

    finally:
        conn.close()

    # ── Persist per-run results ─────────────────────────────────────────────
    if results:
        out_path = pdf_cache.write_results_jsonl(
            config_hash, run_label or default_run_label(), results
        )
        logger.info("Wrote %d results → %s", len(results), out_path)

    return results


def _build_chunks(
    raw_text: str,
    tokenizer,
    doc_id: str,
    article_spans: list[ArticleSpan],
    arm1_cfg: dict,
) -> list[Chunk]:
    strategy = arm1_cfg.get("strategy", "sliding_window")
    if strategy == "sliding_window":
        return _chunker.chunk_sliding_window(
            raw_text, tokenizer, doc_id, article_spans,
            window_size=int(arm1_cfg.get("window_size", 512)),
            stride=int(arm1_cfg.get("stride", 256)),
        )
    if strategy == "recursive":
        return _chunker.chunk_recursive(
            raw_text, tokenizer, doc_id, article_spans,
            max_tokens=int(arm1_cfg.get("max_tokens", 512)),
        )
    raise ValueError(f"Unknown chunking strategy: {strategy!r}")
