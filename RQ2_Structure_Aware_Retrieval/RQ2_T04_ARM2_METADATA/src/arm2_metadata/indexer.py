"""Build FAISS + BM25 indices over the chosen retrieval unit.

Two unit modes:
  * ``unit='node'``    — one entry per (BSARD-mapped, content) AzureDI node.
                         Headers and orphan content are dropped from the index.
  * ``unit='article'`` — one entry per (doc_id, bsard_id), built from
                         ``enricher.build_articles_from_nodes``.

``metadata['text']`` always holds the *clean original French body text*
(``page_content`` for nodes, the aggregated body for articles), even when
the embedding/BM25 input was the enriched variant. The enriched text only
feeds the dense+sparse retrievers; ``text`` is what downstream consumers
(viewer, T07 adapter) read for presentation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

from data_loader.models import Article
from shared.bm25_store import BM25Store
from shared.embeddings import EmbeddingModel
from shared.faiss_store import FAISSStore

from .azuredi_loader import AzureNode
from .enricher import EnrichmentStats, enrich_unit

logger = logging.getLogger(__name__)


# ── Indexable-unit filter ────────────────────────────────────────────────────


def select_indexable(units: Iterable[AzureNode | Article]) -> list[AzureNode | Article]:
    """Drop header nodes and units with empty body text.

    Header nodes (``is_header=True``) carry only structural labels; including
    them dilutes BM25 scores. They remain accessible as enrichment context for
    leaf nodes via parent_id, just not as retrieval targets.
    """
    out: list[AzureNode | Article] = []
    for u in units:
        if isinstance(u, AzureNode):
            if u.is_header:
                continue
            if not u.page_content.strip():
                continue
        else:
            if not u.text.strip():
                continue
        out.append(u)
    return out


# ── Per-entry metadata ───────────────────────────────────────────────────────


def _meta_for_unit(unit: AzureNode | Article, *, clean_text: str) -> dict:
    """Persisted metadata dict — read by boost and T07 adapter."""
    if isinstance(unit, AzureNode):
        return {
            "text": clean_text,
            "doc_id": unit.doc_id,
            "node_id": unit.node_id,
            "page_number": unit.page_number,
            "is_header": unit.is_header,
            "bsard_ids": list(unit.bsard_ids),
            "matched_article_numbers": list(unit.matched_article_numbers),
            "defined_terms": list(unit.defined_terms),
            "jurisdiction": unit.jurisdiction,
            "regulatory_status": unit.regulatory_status,
            "document_status": unit.document_status,
            "document_number": unit.document_number,
            "entry_into_force": unit.entry_into_force,
            "issuing_authority": unit.issuing_authority,
            "document_title": unit.document_title,
            "node_source": unit.node_source,
            "path_to_item": unit.path_to_item,
            "bounding_boxes": unit.bounding_boxes,
            "pdf_filename": unit.pdf_filename,
        }
    return {
        "text": clean_text,
        "doc_id": unit.document_id,
        "bsard_id": unit.bsard_id,
        "bsard_ids": [unit.bsard_id] if unit.bsard_id is not None else [],
        "article_number": unit.article_number,
        "defined_terms": list(unit.defined_terms),
        "jurisdiction": unit.jurisdiction,
        "regulatory_status": unit.regulatory_status,
        "document_status": "Effective",
        "document_number": unit.document_number,
        "entry_into_force": unit.entry_into_force,
        "issuing_authority": unit.issuing_authority,
        "document_title": unit.document_title,
    }


def _entry_id(unit: AzureNode | Article) -> str:
    """ID used by FAISS + BM25 + the term-match boost.

    For nodes: ``"node_<doc>_<id>"`` (uniqueness across the corpus).
    For articles: the Article.article_id which is already namespaced
    ``"doc_<doc>_bsard_<bid>"``.
    """
    if isinstance(unit, AzureNode):
        return f"node_{unit.doc_id}_{unit.node_id}"
    return unit.article_id


# ── Build entry point ────────────────────────────────────────────────────────


def build_arm2_index(
    units: list[AzureNode | Article],
    embedding_model: EmbeddingModel | None,
    doc_id: str | int,
    output_dir: Path,
    *,
    variant: str,
    tokenizer=None,
    max_tokens: int = 512,
    definitions_map: dict[str, str] | None = None,
    batch_size: int = 64,
    sparse_only: bool = False,
    apply_truncation: bool | None = None,
) -> tuple[FAISSStore | None, BM25Store, EnrichmentStats]:
    """Embed and index ``units``, persist to ``output_dir``.

    Args:
        units:            AzureNode list or Article list — already filtered.
        embedding_model:  EmbeddingModel from shared.embeddings. May be None
                          when ``sparse_only=True``.
        doc_id:           Used only for log lines (output_dir is explicit).
        output_dir:       Cache config dir; receives ``faiss.index``,
                          ``faiss_meta.json``, and ``bm25.pkl``.
        variant:          enricher variant name (raw/enriched/summary/filtered/full/terms).
        tokenizer:        Tokenizer used for the truncation cascade. Optional.
        max_tokens:       Target token budget for the variant text.
        definitions_map:  Term→Definition map for the ``terms`` variant.
        batch_size:       Embedding batch size.
        sparse_only:      Skip the FAISS dense index entirely. Returns
                          (None, BM25Store, stats). Use this when the
                          downstream retriever runs BM25-only (CPU-friendly).
        apply_truncation: When True, run the enricher's token-budget cascade.
                          When False, the assembled text is passed through
                          un-truncated (BM25 has no sequence-length cap).
                          Default ``None`` means "auto" — apply truncation
                          iff the dense index will be built (i.e. NOT
                          sparse_only). Review #8-A.

    Behaviour notes:
      * The doc-context header is dropped from header-using variants when
        ``len(set(doc_ids)) == 1`` for the input units (``single_doc_corpus``
        in the enricher; review #8-C). On a single-doc corpus the header is
        constant noise on every entry.

    Returns:
        (FAISSStore | None, BM25Store, EnrichmentStats) — both stores are
        persisted to ``output_dir``; FAISS is None and unwritten when
        ``sparse_only=True``.
    """
    if not units:
        raise ValueError(f"Cannot build index for doc={doc_id}: no units passed.")
    if not sparse_only and embedding_model is None:
        raise ValueError(
            "embedding_model is required unless sparse_only=True."
        )

    if apply_truncation is None:
        apply_truncation = not sparse_only

    # Detect single-doc corpus from the input units (review #8-C). On a
    # single-doc corpus the doc-context header is constant noise on every
    # entry; the enricher will omit it from header-using variants.
    doc_ids = {
        (u.doc_id if isinstance(u, AzureNode) else u.document_id) for u in units
    }
    single_doc_corpus = len(doc_ids) == 1

    stats = EnrichmentStats()
    enriched_texts: list[str] = []
    clean_texts: list[str] = []
    ids: list[str] = []
    metadata: list[dict] = []

    for unit in units:
        text_for_embed = enrich_unit(
            unit, variant,
            tokenizer=tokenizer, max_tokens=max_tokens,
            definitions_map=definitions_map, stats=stats,
            apply_truncation=apply_truncation,
            single_doc_corpus=single_doc_corpus,
        )
        clean = unit.page_content if isinstance(unit, AzureNode) else unit.text
        ids.append(_entry_id(unit))
        enriched_texts.append(text_for_embed)
        clean_texts.append(clean)
        metadata.append(_meta_for_unit(unit, clean_text=clean))

    stats.finalise()
    if stats.n_truncated:
        logger.warning(
            "doc=%s variant=%s: %d/%d units truncated (%.1f%%) — "
            "dropped: summary=%d terms=%d path=%d tail=%d",
            doc_id, variant, stats.n_truncated, stats.n_units,
            100 * stats.truncation_rate,
            stats.n_dropped_summary, stats.n_dropped_terms,
            stats.n_dropped_path, stats.n_tail_trimmed,
        )

    # ── Dense ────────────────────────────────────────────────────────────
    faiss_store: FAISSStore | None = None
    if not sparse_only:
        vectors = embedding_model.encode(
            enriched_texts, batch_size=batch_size, show_progress=True,
        )
        faiss_store = FAISSStore(dimension=embedding_model.dimension)
        faiss_store.add(ids=ids, vectors=vectors, metadata=metadata)

    # ── Sparse ───────────────────────────────────────────────────────────
    bm25_store = BM25Store()
    bm25_store.build(ids=ids, texts=enriched_texts, metadata=metadata)

    # ── Persist ──────────────────────────────────────────────────────────
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if faiss_store is not None:
        faiss_store.save(str(output_dir))
    bm25_store.save(str(output_dir / "bm25.pkl"))

    return faiss_store, bm25_store, stats
