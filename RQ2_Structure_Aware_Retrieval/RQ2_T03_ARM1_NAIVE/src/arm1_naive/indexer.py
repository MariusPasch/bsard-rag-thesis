"""FAISS + BM25 index construction and persistence for Arm 1 chunks.

Default layout when ``output_dir`` is omitted (legacy callers, e.g. T08):
  <index_dir>/<doc_id>/{faiss.index, faiss_meta.json, bm25.pkl}

Cache-aware callers pass ``output_dir`` explicitly so the indices land in the
PDF-centric experiment cache (CACHE_DESIGN.md §4):
  <pdf_cache>/configs/<config_hash>/{faiss.index, faiss_meta.json, bm25.pkl}

FAISSStore.save() / .load() take the directory path; BM25Store.save() / .load()
take the file path.

Requires: ``pip install -e ../RQ2_T01_SHARED``.
"""

from __future__ import annotations

from pathlib import Path

from arm1_naive.chunker import Chunk
from shared.bm25_store import BM25Store
from shared.embeddings import EmbeddingModel
from shared.faiss_store import FAISSStore


def build_arm1_index(
    chunks: list[Chunk],
    embedding_model: EmbeddingModel,
    doc_id: str,
    index_dir: Path,
    batch_size: int = 64,
    *,
    output_dir: Path | None = None,
) -> tuple[FAISSStore, BM25Store]:
    """Embed chunks and build FAISS + BM25 indices.

    Args:
        chunks:          Chunks produced by ``chunker.chunk_sliding_window`` /
                         ``chunker.chunk_recursive``.
        embedding_model: Loaded ``EmbeddingModel`` from ``shared.embeddings``.
        doc_id:          Document identifier (used as a sub-directory name in
                         the legacy layout when ``output_dir`` is omitted).
        index_dir:       Root directory for the legacy ``<index_dir>/<doc_id>/``
                         layout. Ignored when ``output_dir`` is provided.
        batch_size:      Embedding batch size.
        output_dir:      Explicit save target (cache-aware callers). When set,
                         indices are written directly here and ``index_dir`` /
                         ``doc_id`` are not used to derive the path.

    Returns:
        ``(FAISSStore, BM25Store)`` — both populated and ready for search.

    Side-effects:
        Saves indices to ``output_dir`` if provided, otherwise to
        ``<index_dir>/<doc_id>/``.
    """
    if not chunks:
        raise ValueError(f"Cannot build index for '{doc_id}': chunk list is empty.")

    texts = [c.text for c in chunks]

    def _meta(c: Chunk) -> dict:
        return {
            "text": c.text,
            "doc_id": c.doc_id,
            "start_char": c.start_char,
            "end_char": c.end_char,
            "start_token": c.start_token,
            "end_token": c.end_token,
            "bsard_ids": c.bsard_ids,
        }

    # ── Dense index ───────────────────────────────────────────────────────────
    vectors = embedding_model.encode(texts, batch_size=batch_size, show_progress=True)

    faiss_store = FAISSStore(dimension=embedding_model.dimension)
    faiss_store.add(
        ids=[c.chunk_id for c in chunks],
        vectors=vectors,
        metadata=[_meta(c) for c in chunks],
    )

    # ── Sparse index ──────────────────────────────────────────────────────────
    bm25_store = BM25Store()
    bm25_store.build(
        ids=[c.chunk_id for c in chunks],
        texts=texts,
        metadata=[_meta(c) for c in chunks],
    )

    # ── Persist ───────────────────────────────────────────────────────────────
    out_dir = Path(output_dir) if output_dir is not None else (Path(index_dir) / doc_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    faiss_store.save(str(out_dir))            # writes faiss.index + faiss_meta.json
    bm25_store.save(str(out_dir / "bm25.pkl"))

    return faiss_store, bm25_store


def load_arm1_index(
    doc_id: str,
    index_dir: Path,
    *,
    source_dir: Path | None = None,
) -> tuple[FAISSStore, BM25Store]:
    """Load previously saved FAISS + BM25 indices from disk.

    Args:
        doc_id:     Document identifier (used in the legacy layout).
        index_dir:  Root directory for the legacy ``<index_dir>/<doc_id>/``
                    layout. Ignored when ``source_dir`` is provided.
        source_dir: Explicit load directory (cache-aware callers).

    Raises:
        FileNotFoundError: if the index files do not exist at the resolved path.
    """
    doc_dir = Path(source_dir) if source_dir is not None else (Path(index_dir) / doc_id)
    faiss_index_path = doc_dir / "faiss.index"
    bm25_path = doc_dir / "bm25.pkl"

    if not faiss_index_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {faiss_index_path}")
    if not bm25_path.exists():
        raise FileNotFoundError(f"BM25 index not found: {bm25_path}")

    faiss_store = FAISSStore.load(str(doc_dir))       # directory path
    bm25_store = BM25Store.load(str(bm25_path))

    return faiss_store, bm25_store
