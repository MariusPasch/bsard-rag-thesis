"""Hybrid retrieval pipeline for T04 (Arm 2A — AzureDI metadata).

Mirrors T03's structure: dense (FAISS) + sparse (BM25) → RRF fusion → optional
boost/filter. The novelty is at index-time enrichment and at the optional
query-time boost stage; the hybrid recipe itself is identical to T03 by
design (per PROJECT_CONTEXT.md / IMPLEMENTATION_PROMPT.md).

Entry point: ``run_arm2_metadata`` runs end-to-end for one (doc_id, variant,
unit) combination and returns ``list[RetrievalResult]``. The orchestrator
calls this once per variant.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from collections import defaultdict
from pathlib import Path

from arm1_naive.retriever import reciprocal_rank_fusion
from data_loader.models import RetrievalResult
from shared.bm25_store import BM25Store
from shared.embeddings import EmbeddingModel
from shared.faiss_store import FAISSStore, SearchResult

from .azuredi_loader import AzureNode, load_azuredi_corpus
from .boost import BoostConfig, apply_filters_and_boosts, normalise_boost
from .bsard_link import link_corpus_to_bsard
from .cache import (
    T04CacheRoot,
    T04ConfigCache,
    T04DocCache,
    azuredi_dump_fingerprint,
    compute_config_hash,
    compute_pdf_sha256,
    default_run_label,
)
from .enricher import (
    BOOST_VARIANTS,
    VARIANTS_ARTICLE,
    VARIANTS_NODE,
    build_articles_from_nodes,
    build_definitions_map,
)
from .indexer import build_arm2_index, select_indexable
from .query_extractor import QuerySignals, extract_query_signals

logger = logging.getLogger(__name__)

# Bump on linker / enricher / indexer behaviour changes that affect cached
# metadata or BM25 token streams. Folded into chunking_params so caches
# auto-invalidate without manual --force-reindex. See cache.py / CHANGE_NOTES.
#   v2 — preceding-content-node inheritance pass in bsard_link.
#   v3 — review #8 enricher changes: cascade reorder (drop path before
#        summary), apply_truncation flag (skip cascade under sparse_only),
#        single_doc_corpus header elision for header-using variants.
#   v4 — bsard_link `_ART_RE_AZURE` tolerates Markdown-prefixed headers
#        ("##### Art. NNN"). Fixes catastrophic doc-8 under-linking where
#        110 article anchors were emitted only in Markdown form, causing
#        ~25 % of body text to inherit bsard_id 814 instead of the correct
#        anchor. Other curated PDFs gain 17-33 newly-anchored headers each.
LINKER_VERSION = 4


# ── Helpers ──────────────────────────────────────────────────────────────────


def _resolve_model_names(config: dict, embedding_model: EmbeddingModel, tokenizer):
    embedding_model_name = (
        config.get("models", {}).get("embedding_model")
        or getattr(embedding_model, "_model_name", None)
    )
    if not embedding_model_name:
        raise ValueError(
            "Could not resolve embedding model name. Set config['models']['embedding_model']."
        )
    tokenizer_name = (
        config.get("models", {}).get("tokenizer")
        or getattr(tokenizer, "name_or_path", None)
        or embedding_model_name
    )
    return tokenizer_name, embedding_model_name


def _chunking_params(arm2_cfg: dict) -> dict:
    """Hash-key payload identifying this variant + unit settings.

    Includes the boost_config so a tweak to multipliers cleanly invalidates
    the per-config cache (FAISS itself doesn't change, but downstream metadata
    snapshot does).
    """
    return {
        "arm": "2a",
        "variant": str(arm2_cfg["variant"]),
        "unit": str(arm2_cfg["unit"]),
        # Frozen at the original precompute values so the on-disk
        # caches built under the previous (reranked) pipeline stay valid after
        # the cross-encoder was removed from the RQ2 architecture. These keys
        # never affected the index — they were retrieval-time knobs whose
        # presence in the hash is now purely a compatibility shim. Do NOT
        # change either value; runtime cut is controlled by `top_k` directly
        # on retrieve_arm2.
        "retrieval_top_k": 100,
        "rerank_top_k": 10,
        "max_tokens": int(arm2_cfg.get("max_tokens", 512)),
        "boost": normalise_boost(arm2_cfg.get("boost")),
        "sparse_only": bool(arm2_cfg.get("sparse_only", False)),
        # Bump when bsard_link / enricher / indexer change in a way that would
        # alter the persisted metadata or the BM25 token stream. This makes
        # caches automatically invalidate on linker fixes (e.g. v2 added
        # preceding-content-node inheritance for orphan paragraphs).
        "linker_version": LINKER_VERSION,
    }


def _load_questions(
    conn: sqlite3.Connection,
    *,
    question_ids: set[str] | None,
    split: str | None,
) -> list[tuple[str, str]]:
    """Return (question_id, question_text) for the requested questions."""
    if question_ids:
        rows = conn.execute(
            "SELECT question_id, question_text FROM questions"
        ).fetchall()
        wanted = set(question_ids)
        return [
            (str(r["question_id"]), r["question_text"])
            for r in rows
            if str(r["question_id"]) in wanted
        ]
    if split:
        rows = conn.execute(
            "SELECT question_id, question_text FROM questions WHERE split = ?",
            (split,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT question_id, question_text FROM questions"
        ).fetchall()
    return [(str(r["question_id"]), r["question_text"]) for r in rows]


# ── Per-query retrieval ──────────────────────────────────────────────────────


def retrieve_arm2(
    query: str,
    faiss_store: FAISSStore | None,
    bm25_store: BM25Store,
    embedding_model: EmbeddingModel | None,
    *,
    signals: QuerySignals,
    boost_cfg: BoostConfig,
    apply_boost: bool,
    retrieval_top_k: int = 100,
    top_k: int = 100,
    sparse_only: bool = False,
) -> list[SearchResult]:
    """Dense + sparse → RRF → (optional boost) → top_k.

    When ``sparse_only=True`` the dense step and RRF are skipped — BM25 ranks
    feed the boost stage directly. ``faiss_store`` and ``embedding_model`` are
    then unused (may be None).
    """
    if sparse_only:
        candidates = bm25_store.search(query, k=retrieval_top_k)
    else:
        if faiss_store is None or embedding_model is None:
            raise ValueError(
                "Hybrid mode requires faiss_store and embedding_model "
                "(pass sparse_only=True to skip dense)."
            )
        query_vec = embedding_model.encode_query(query)
        dense = faiss_store.search(query_vec, k=retrieval_top_k)
        sparse = bm25_store.search(query, k=retrieval_top_k)
        candidates = reciprocal_rank_fusion(dense, sparse)

    if apply_boost:
        candidates = apply_filters_and_boosts(candidates, signals, boost_cfg)

    return candidates[:top_k]


# ── Full pipeline ────────────────────────────────────────────────────────────


def run_arm2_metadata(
    *,
    azuredi_dir: Path,
    pdf_document_map_csv: Path,
    bsard_db_path: Path,
    pdf_path: Path | None,
    embedding_model: EmbeddingModel,
    tokenizer,
    llm,
    index_root: Path,
    config: dict,
    doc_id: str,
    azure_doc_id: int,
    variant: str,
    unit: str,                       # "node" | "article"
    run_label: str | None = None,
    question_ids: set[str] | None = None,
    split: str | None = "test",
    force_reindex: bool = False,
    apply_boost_override: bool | None = None,
) -> list[RetrievalResult]:
    """Run T04 end-to-end for one (doc_id, variant, unit) and persist results.

    Args:
        azuredi_dir:          Path to the AzureDI dump (a junction in this repo).
        pdf_document_map_csv: BSARD doc_id ↔ pdf_filename map.
        bsard_db_path:        Path to bsard_corpus.db (for the question set
                              and the article-number → bsard_id lookup).
        pdf_path:             Optional path to the source PDF — when given, the
                              cache manifest pins its sha256 so swapping the
                              PDF invalidates the index.
        embedding_model:      EmbeddingModel from shared.embeddings.
        tokenizer:            HF tokenizer for the truncation cascade.
        llm:                  Optional LLMClient for query_extractor method 3.
        index_root:           Cache root (e.g. ``RQ2_T04_ARM2_METADATA/data``).
        config:               Dict with keys:
                                config['arm2']['variant']         (overridden by ``variant``)
                                config['arm2']['unit']            (overridden by ``unit``)
                                config['arm2']['retrieval_top_k'] (default 100)
                                config['arm2']['top_k']           (default 100, result cut)
                                config['arm2']['max_tokens']      (default 512)
                                config['arm2']['boost']           (BoostConfig dict)
                                config['arm2']['use_llm_classifier']  (bool, default False)
                                config['models']['embedding_model']
                                config['models']['tokenizer']
        doc_id:               BSARD PDF stem (e.g. ``2004_05_27_2004A27101``)
                              — used as the cache folder name.
        azure_doc_id:         AzureDI integer ``DocumentId`` — used to filter
                              nodes from the dump. Recorded in the manifest as
                              ``azure_document_id``.
        variant:              raw / enriched / summary / filtered / full / terms.
        unit:                 "node" or "article".
        run_label:            Per-run filename stem (defaults to UTC timestamp).
        question_ids:         Optional set of question ids — when None, falls
                              back to ``split``.
        split:                "test" / "train" / None — used when question_ids
                              is None.
        force_reindex:        Invalidate the per-config cache and rebuild.
    """
    if unit == "node":
        if variant not in VARIANTS_NODE:
            raise ValueError(f"variant {variant!r} not allowed for unit=node")
    elif unit == "article":
        if variant not in VARIANTS_ARTICLE:
            raise ValueError(f"variant {variant!r} not allowed for unit=article")
    else:
        raise ValueError(f"unit must be 'node' or 'article', got {unit!r}")

    arm2_cfg = dict(config.get("arm2", {}))
    arm2_cfg["variant"] = variant
    arm2_cfg["unit"] = unit
    chunking_params = _chunking_params(arm2_cfg)
    sparse_only = chunking_params["sparse_only"]

    tokenizer_name, embedding_model_name = _resolve_model_names(
        config, embedding_model, tokenizer,
    )

    azuredi_dir = Path(azuredi_dir)
    azuredi_fp = azuredi_dump_fingerprint(azuredi_dir, pdf_document_map_csv)
    # Single canonical PDF identifier used by BOTH compute_config_hash AND
    # cfg_cache.validate / write_manifest. Review #15-A6: previously the hash
    # used `pdf_sha or azuredi_fp[:64]` while the manifest stored `pdf_sha`
    # (potentially None) — two hosts producing the same logical config got
    # different config_hashes when one had the PDF on disk and the other
    # didn't, breaking cross-machine cache reuse.
    pdf_sha = compute_pdf_sha256(pdf_path) if pdf_path else None
    pdf_sha_for_cache = pdf_sha or f"azuredi_fp:{azuredi_fp[:64]}"

    config_hash = compute_config_hash(
        pdf_sha256=pdf_sha_for_cache,
        tokenizer_name=tokenizer_name,
        embedding_model=embedding_model_name,
        chunking_params=chunking_params,
    )

    cache_root = T04CacheRoot(index_root)
    doc_cache = T04DocCache(cache_root, doc_id)
    cfg_cache = T04ConfigCache(doc_cache, config_hash)

    boost_cfg = BoostConfig(**chunking_params["boost"])
    if apply_boost_override is not None:
        apply_boost = bool(apply_boost_override)
    else:
        apply_boost = variant in BOOST_VARIANTS

    cache_hit = (
        not force_reindex
        and cfg_cache.exists()
        and cfg_cache.validate(
            azuredi_fingerprint=azuredi_fp,
            pdf_sha256=pdf_sha_for_cache,
            tokenizer_name=tokenizer_name,
            embedding_model=embedding_model_name,
            chunking_params=chunking_params,
        )
    )

    conn = sqlite3.connect(str(bsard_db_path))
    conn.row_factory = sqlite3.Row

    try:
        nodes, my_documents, definitions_df = load_azuredi_corpus(
            azuredi_dir, pdf_document_map_csv,
        )
        nodes = link_corpus_to_bsard(nodes, bsard_db_path)
        # Restrict to the requested AzureDI doc_id (the public metric is per-doc).
        nodes_for_doc = [n for n in nodes if n.doc_id == azure_doc_id]
        if not nodes_for_doc:
            raise RuntimeError(
                f"No AzureDI nodes for azure_doc_id={azure_doc_id} "
                f"(kept docs: {sorted(my_documents)})"
            )

        definitions_map = build_definitions_map(definitions_df)

        if unit == "node":
            indexable = select_indexable(nodes_for_doc)
        else:
            articles = build_articles_from_nodes(
                nodes_for_doc, my_documents, definitions_map,
            )
            indexable = select_indexable(articles)

        if cache_hit:
            logger.info(
                "[cache hit] doc=%s config=%s variant=%s unit=%s sparse_only=%s",
                doc_id, config_hash, variant, unit, sparse_only,
            )
            if sparse_only:
                faiss_store = None
                bm25_store = BM25Store.load(str(cfg_cache.bm25_path()))
            else:
                faiss_store, bm25_store = cfg_cache.load_indices()
        else:
            logger.info(
                "[cache miss] doc=%s config=%s variant=%s unit=%s sparse_only=%s — rebuilding",
                doc_id, config_hash, variant, unit, sparse_only,
            )
            faiss_store, bm25_store, stats = build_arm2_index(
                indexable, embedding_model, doc_id, cfg_cache.path,
                variant=variant,
                tokenizer=tokenizer,
                max_tokens=arm2_cfg.get("max_tokens", 512),
                definitions_map=definitions_map,
                sparse_only=sparse_only,
            )
            cfg_cache.write_manifest(
                azuredi_fingerprint=azuredi_fp,
                azure_document_id=azure_doc_id,
                pdf_sha256=pdf_sha_for_cache,
                tokenizer_name=tokenizer_name,
                embedding_model=embedding_model_name,
                embedding_dim=(0 if sparse_only else embedding_model.dimension),
                chunking_params=chunking_params,
                n_units=len(indexable),
            )
            cfg_cache.write_enrichment_stats(stats)
            doc_cache.write_doc_manifest(
                azuredi_fingerprint=azuredi_fp,
                azure_document_id=azure_doc_id,
                pdf_sha256=pdf_sha_for_cache,
                pdf_filename=next((n.pdf_filename for n in nodes_for_doc if n.pdf_filename), None),
                n_nodes=len(nodes_for_doc),
                n_articles=sum(1 for u in indexable if not isinstance(u, AzureNode)),
            )

        # ── Per-query retrieval ─────────────────────────────────────────
        questions = _load_questions(conn, question_ids=question_ids, split=split)
        if not questions:
            logger.warning(
                "No questions selected (question_ids=%s split=%s)", question_ids, split,
            )
            return []

        llm_cache: dict[str, dict] = {}
        method_label = f"arm2_metadata_{unit}_{variant}"
        results: list[RetrievalResult] = []

        # Linker output as a (doc_id, node_id) -> [bsard_id...] map. The query
        # extractor uses this to emit BOTH node-level and article-level entry
        # ids for the `used_in` boost — without it, the boost is silently dead
        # for unit="article" (review #15-A1).
        node_to_bsard = {
            (n.doc_id, n.node_id): list(n.bsard_ids)
            for n in nodes_for_doc if n.bsard_ids
        }

        for q_id, q_text in questions:
            if apply_boost:
                signals = extract_query_signals(
                    q_text, definitions_df, llm,
                    llm_cache=llm_cache,
                    use_llm=arm2_cfg.get("use_llm_classifier", False),
                    node_to_bsard=node_to_bsard,
                )
            else:
                signals = QuerySignals()

            t0 = time.perf_counter()
            ranked = retrieve_arm2(
                q_text, faiss_store, bm25_store, embedding_model,
                signals=signals,
                boost_cfg=boost_cfg,
                apply_boost=apply_boost,
                retrieval_top_k=int(arm2_cfg.get("retrieval_top_k", 100)),
                top_k=int(arm2_cfg.get("top_k", 100)),
                sparse_only=sparse_only,
            )
            latency_ms = (time.perf_counter() - t0) * 1000

            ranked_items = []
            for r in ranked:
                meta = dict(r.metadata or {})
                # Preserve only the fields T07 + downstream consumers need; the
                # full per-entry metadata stays in the index store.
                if "bsard_id" in meta and meta["bsard_id"] is not None:
                    meta_out = {
                        "bsard_id": int(meta["bsard_id"]),
                        "bsard_ids": list(meta.get("bsard_ids", [])),
                        "doc_id": meta.get("doc_id"),
                        "article_number": meta.get("article_number"),
                        "jurisdiction": meta.get("jurisdiction"),
                        "regulatory_status": meta.get("regulatory_status"),
                        "document_number": meta.get("document_number"),
                    }
                else:
                    meta_out = {
                        "bsard_ids": list(meta.get("bsard_ids", [])),
                        "doc_id": meta.get("doc_id"),
                        "node_id": meta.get("node_id"),
                        "page_number": meta.get("page_number"),
                        "is_header": meta.get("is_header"),
                        "matched_article_numbers": list(
                            meta.get("matched_article_numbers", [])
                        ),
                        "jurisdiction": meta.get("jurisdiction"),
                        "regulatory_status": meta.get("regulatory_status"),
                    }
                ranked_items.append({
                    "id": r.id,
                    "score": float(r.score),
                    "text": meta.get("text", ""),
                    "metadata": meta_out,
                })

            results.append(RetrievalResult(
                query_id=str(q_id),
                query_text=q_text,
                ranked_items=ranked_items,
                method=method_label,
                cost={
                    "llm_calls": len(llm_cache),
                    "tokens_in": 0,
                    "tokens_out": 0,
                    "latency_ms": round(latency_ms, 1),
                },
                trace={
                    "variant": variant,
                    "unit": unit,
                    "sparse_only": sparse_only,
                    "boost_applied": apply_boost,
                    "n_terms_matched": len(signals.matched_terms),
                    "jurisdiction_signal": signals.jurisdiction,
                    "article_refs": list(signals.article_refs),
                    "doc_id": doc_id,
                    "config_hash": config_hash,
                },
            ))

    finally:
        conn.close()

    if results:
        out = doc_cache.write_results_jsonl(
            config_hash, run_label or default_run_label(), results,
        )
        logger.info("Wrote %d results -> %s", len(results), out)

    return results
