"""Build and persist T04 FAISS + BM25 indices for one doc, no queries.

Mirrors T03's ``scripts/precompute_pdf.py`` but for T04: it loads the AzureDI
corpus, links to BSARD, optionally aggregates to articles, then builds and
saves indices for every (unit, variant) in the plan. No per-query retrieval
is run — this script's job is just to do the expensive embedding pass once.

Designed for an offline GPU host so the slow dense-embedding step can happen
elsewhere. The cached ``data/<doc_id>/configs/<config_hash>/`` folder is
written under the data root (``$RQ2_DATA_DIR``, default ``<repo>/data``) and
uploaded to the companion Hugging Face dataset
``Marios-Paschalidis-Thesis/bsard-rag-thesis-data``; another machine obtains it by
downloading that dataset.

Usage (from the repo root, with the venv active):

    # Default plan = full SMOKE_PLAN of compare_t03_vs_t04.py
    python RQ2_T04_ARM2_METADATA/scripts/precompute_t04_indices.py -v

    # Build only specific variants
    python RQ2_T04_ARM2_METADATA/scripts/precompute_t04_indices.py \\
        --variants node/raw article/full -v

    # Full plan (all 11 variants)
    python RQ2_T04_ARM2_METADATA/scripts/precompute_t04_indices.py --full -v

    # Override paths for a second machine
    python RQ2_T04_ARM2_METADATA/scripts/precompute_t04_indices.py \\
        --azuredi-dir D:/path/to/AzureDI \\
        --pdf-path D:/path/to/2004_05_27_2004A27101.pdf \\
        --bsard-db D:/path/to/bsard_corpus.db \\
        --cache-root D:/path/to/RQ2_T04_ARM2_METADATA/data \\
        --pdf-doc-map D:/path/to/pdf_document_map.csv

Once finished, the populated ``data/<doc_id>/configs/<hash>/`` lives under
the data root and can be uploaded to the companion Hugging Face dataset
``Marios-Paschalidis-Thesis/bsard-rag-thesis-data``. Re-running ``compare_t03_vs_t04.py``
against that cache root will hit the cache and skip the dense-embedding pass
entirely.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent

# Make sibling src/ importable when running outside an installed-editable env.
for _sibling in _REPO.glob("RQ2_T0*/src"):
    s = str(_sibling)
    if s not in sys.path:
        sys.path.insert(0, s)

from arm2_metadata.azuredi_loader import load_azuredi_corpus
from arm2_metadata.boost import normalise_boost
from arm2_metadata.bsard_link import link_corpus_to_bsard
from arm2_metadata.cache import (
    T04CacheRoot,
    T04ConfigCache,
    T04DocCache,
    azuredi_dump_fingerprint,
    compute_config_hash,
    compute_pdf_sha256,
)
from arm2_metadata.enricher import (
    VARIANTS_ARTICLE,
    VARIANTS_NODE,
    build_articles_from_nodes,
    build_definitions_map,
)
from arm2_metadata.indexer import build_arm2_index, select_indexable
from arm2_metadata.retriever import LINKER_VERSION
from shared.embeddings import EmbeddingModel
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


# Same as compare_t03_vs_t04.py — keep the two scripts in sync.
DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-large-instruct"
DEFAULT_DOC_ID = "2004_05_27_2004A27101"   # BSARD PDF stem; AzureDI int resolved via pdf_document_map.csv

DEFAULT_AZUREDI_DIR = _REPO / "RQ2_T04_ARM2_METADATA" / "azuredi"
DEFAULT_PDF_DOC_MAP_CSV = _REPO / "RQ2_T02_DATA_LOADER" / "data" / "csv" / "pdf_document_map.csv"
DEFAULT_BSARD_DB_PATH = _REPO / "dataset_creation_output" / "bsard_corpus.db"
DEFAULT_T04_CACHE_ROOT = _REPO / "RQ2_T04_ARM2_METADATA" / "data"
DEFAULT_PDF_DIR = _REPO / "RQ2_T02_DATA_LOADER" / "data" / "pdfs"


def _resolve_azure_doc_id(doc_id: str, pdf_doc_map_csv: Path) -> int:
    """Map a BSARD PDF stem to its AzureDI integer DocumentId via pdf_document_map.csv."""
    import pandas as pd
    df = pd.read_csv(pdf_doc_map_csv, encoding="utf-8")
    target = f"{doc_id}.pdf"
    row = df[df["pdf_filename"] == target]
    if row.empty:
        raise ValueError(
            f"doc_id {doc_id!r} not in {pdf_doc_map_csv} — has_azure_extraction is false for this PDF"
        )
    return int(row.iloc[0]["document_id"])


# ── Plans (mirror compare_t03_vs_t04.py) ─────────────────────────────────────


SMOKE_PLAN: list[tuple[str, str]] = [
    ("node", "raw"),
    ("node", "enriched"),
    ("node", "summary"),
    ("node", "full"),
    ("article", "raw"),
    ("article", "full"),
]

FULL_PLAN: list[tuple[str, str]] = [
    ("node", "raw"),
    ("node", "enriched"),
    ("node", "summary"),
    ("node", "filtered"),
    ("node", "full"),
    ("node", "terms"),
    ("article", "raw"),
    ("article", "enriched"),
    ("article", "filtered"),
    ("article", "full"),
    ("article", "terms"),
]


# ── Build one (unit, variant) ────────────────────────────────────────────────


def precompute_one(
    *,
    unit: str,
    variant: str,
    indexable: list,
    embedding_model: EmbeddingModel,
    tokenizer,
    definitions_map: dict[str, str],
    cache_root: T04CacheRoot,
    doc_id: str,
    azure_doc_id: int,
    azuredi_fp: str,
    pdf_sha: str | None,
    embedding_model_name: str,
    tokenizer_name: str,
    max_tokens: int = 512,
    boost: dict | None = None,
    sparse_only: bool = False,
    force: bool = False,
) -> tuple[str, str]:
    """Build (or skip if cache hits) the indices for one (unit, variant) pair.

    Returns (config_hash, status) where status is one of "built" / "cached" /
    "skipped".
    """
    if unit == "node":
        if variant not in VARIANTS_NODE:
            raise ValueError(f"variant {variant!r} not allowed for unit=node")
    elif unit == "article":
        if variant not in VARIANTS_ARTICLE:
            raise ValueError(f"variant {variant!r} not allowed for unit=article")
    else:
        raise ValueError(f"unit must be 'node' or 'article', got {unit!r}")

    chunking_params = {
        "arm": "2a",
        "variant": variant,
        "unit": unit,
        # Frozen at the original precompute values — see
        # arm2_metadata.retriever._chunking_params for the rationale.
        "retrieval_top_k": 100,
        "rerank_top_k": 10,
        "max_tokens": int(max_tokens),
        "boost": boost or normalise_boost(None),
        "sparse_only": bool(sparse_only),
        "linker_version": LINKER_VERSION,
    }
    # Single canonical PDF identifier — must match retriever.py's choice so
    # cache_hash and validate() agree on hosts that don't have the PDF on
    # disk. See review #15-A6.
    pdf_sha_for_cache = pdf_sha or f"azuredi_fp:{azuredi_fp[:64]}"

    config_hash = compute_config_hash(
        pdf_sha256=pdf_sha_for_cache,
        tokenizer_name=tokenizer_name,
        embedding_model=embedding_model_name,
        chunking_params=chunking_params,
    )

    doc_cache = T04DocCache(cache_root, doc_id)
    cfg_cache = T04ConfigCache(doc_cache, config_hash)

    if not force and cfg_cache.exists() and cfg_cache.validate(
        azuredi_fingerprint=azuredi_fp,
        pdf_sha256=pdf_sha_for_cache,
        tokenizer_name=tokenizer_name,
        embedding_model=embedding_model_name,
        chunking_params=chunking_params,
    ):
        logger.info(
            "[cache hit] %s/%s -> %s — nothing to do", unit, variant, config_hash,
        )
        return config_hash, "cached"

    logger.info(
        "[building] %s/%s -> %s (sparse_only=%s)",
        unit, variant, config_hash, sparse_only,
    )
    t0 = time.perf_counter()
    faiss_store, bm25_store, stats = build_arm2_index(
        indexable, embedding_model, doc_id, cfg_cache.path,
        variant=variant,
        tokenizer=tokenizer,
        max_tokens=max_tokens,
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
    elapsed = time.perf_counter() - t0
    logger.info(
        "[built]    %s/%s -> %s   n_units=%d  truncated=%d (%.1f%%)  in %.1fs",
        unit, variant, config_hash, len(indexable),
        stats.n_truncated, 100 * stats.truncation_rate, elapsed,
    )
    return config_hash, "built"


# ── Driver ───────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python RQ2_T04_ARM2_METADATA/scripts/precompute_t04_indices.py",
        description=(
            "Build T04 FAISS + BM25 indices for one AzureDI doc — no queries. "
            "Designed for an offline GPU host; the resulting cache is written "
            "under the data root and shared via the companion Hugging Face "
            "dataset Marios-Paschalidis-Thesis/bsard-rag-thesis-data."
        ),
    )
    parser.add_argument(
        "--doc-id", type=str, default=DEFAULT_DOC_ID,
        help="BSARD PDF stem (e.g. 2004_05_27_2004A27101). The AzureDI integer "
             "DocumentId is resolved via pdf_document_map.csv. Default: %(default)s.",
    )
    parser.add_argument(
        "--pdf-dir", type=Path, default=DEFAULT_PDF_DIR,
        help="PDF directory; pdf_path is computed as <pdf_dir>/<doc_id>.pdf. "
             "Default: %(default)s.",
    )
    parser.add_argument(
        "--variants", nargs="*", default=None,
        help="Restrict to (unit/variant) pairs (e.g. node/raw article/full). "
             "Defaults to SMOKE_PLAN unless --full is set.",
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Build the full 11-variant plan instead of SMOKE_PLAN.",
    )
    parser.add_argument(
        "--sparse-only", action="store_true",
        help="Skip the FAISS dense build; only persist BM25 + manifest. "
             "Useful for sanity-checking the pipeline without the embedding cost.",
    )
    parser.add_argument(
        "--embedding-model", default=DEFAULT_EMBEDDING_MODEL,
        help="Sentence-transformer name (default: %(default)s).",
    )
    parser.add_argument(
        "--tokenizer", default=None,
        help="HF tokenizer name (defaults to --embedding-model).",
    )
    parser.add_argument(
        "--azuredi-dir", type=Path, default=DEFAULT_AZUREDI_DIR,
        help="Path to azuredi/ (default: %(default)s).",
    )
    parser.add_argument(
        "--pdf-doc-map", type=Path, default=DEFAULT_PDF_DOC_MAP_CSV,
        help="Path to pdf_document_map.csv (default: %(default)s).",
    )
    parser.add_argument(
        "--bsard-db", type=Path, default=DEFAULT_BSARD_DB_PATH,
        help="Path to bsard_corpus.db (default: %(default)s).",
    )
    parser.add_argument(
        "--pdf-path", type=Path, default=None,
        help="Path to the source PDF (used only for pdf_sha256 fingerprint; "
             "fall back to AzureDI dump fingerprint if missing). "
             "Default: <pdf_dir>/<doc_id>.pdf.",
    )
    parser.add_argument(
        "--cache-root", type=Path, default=DEFAULT_T04_CACHE_ROOT,
        help="T04 cache root (default: %(default)s).",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=512,
        help="Truncation budget for the enricher cascade (default: %(default)s). "
             "Ignored under --sparse-only (BM25 is sequence-length agnostic).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Rebuild even if the per-config manifest validates.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("arm2_metadata").setLevel(logging.INFO)
    logging.getLogger("shared").setLevel(logging.INFO)

    # ── Pick the plan ───────────────────────────────────────────────────────
    base_plan = FULL_PLAN if args.full else SMOKE_PLAN
    if args.variants:
        wanted = set()
        for v in args.variants:
            unit, variant = v.split("/")
            wanted.add((unit, variant))
        # Filter from base_plan (smoke or full per --full), not always FULL_PLAN.
        # Same shape as #15-A5 in compare_t03_vs_t04.py.
        plan = [p for p in base_plan if p in wanted]
    else:
        plan = base_plan

    if not plan:
        print("Empty plan — nothing to do.", file=sys.stderr)
        return 1

    # ── Validate inputs ─────────────────────────────────────────────────────
    if not args.azuredi_dir.exists():
        print(f"AzureDI dir not found: {args.azuredi_dir}", file=sys.stderr)
        return 2
    if not args.bsard_db.exists():
        print(f"BSARD DB not found: {args.bsard_db}", file=sys.stderr)
        return 2
    if not args.pdf_doc_map.exists():
        print(f"pdf_document_map.csv not found: {args.pdf_doc_map}", file=sys.stderr)
        return 2

    embedding_model_name = args.embedding_model
    tokenizer_name = args.tokenizer or embedding_model_name

    azure_doc_id = _resolve_azure_doc_id(args.doc_id, args.pdf_doc_map)
    pdf_path = args.pdf_path or (args.pdf_dir / f"{args.doc_id}.pdf")

    print(f"plan ({len(plan)}):                 {plan}")
    print(f"doc_id (BSARD stem):          {args.doc_id}")
    print(f"azure_document_id:            {azure_doc_id}")
    print(f"embedding model:              {embedding_model_name}")
    print(f"tokenizer:                    {tokenizer_name}")
    print(f"sparse_only:                  {args.sparse_only}")
    print(f"force:                        {args.force}")
    print(f"azuredi:                      {args.azuredi_dir}")
    print(f"bsard_db:                     {args.bsard_db}")
    print(f"pdf_path:                     {pdf_path} "
          f"({'OK' if pdf_path.exists() else 'NOT FOUND — using AzureDI fp instead'})")
    print(f"cache_root:                   {args.cache_root}")

    # ── Load corpus once ────────────────────────────────────────────────────
    nodes, my_documents, definitions_df = load_azuredi_corpus(
        args.azuredi_dir, args.pdf_doc_map,
    )
    nodes = link_corpus_to_bsard(nodes, args.bsard_db)
    nodes_for_doc = [n for n in nodes if n.doc_id == azure_doc_id]
    if not nodes_for_doc:
        print(
            f"\nNo AzureDI nodes for azure_doc_id={azure_doc_id} (kept docs: "
            f"{sorted(my_documents)}). Bailing.",
            file=sys.stderr,
        )
        return 3

    definitions_map = build_definitions_map(definitions_df)

    indexable_node = select_indexable(nodes_for_doc)
    articles = build_articles_from_nodes(nodes_for_doc, my_documents, definitions_map)
    indexable_article = select_indexable(articles)
    print(f"\nindexable nodes:              {len(indexable_node)}")
    print(f"indexable articles:           {len(indexable_article)}")

    # ── Models ──────────────────────────────────────────────────────────────
    if args.sparse_only:
        print("\nSkipping embedding-model load (--sparse-only).")
        embedding_model = None
    else:
        print(f"\nLoading embedding model …  (CUDA: {_cuda_status()})")
        embedding_model = EmbeddingModel(model_name=embedding_model_name)
        _ = embedding_model.dimension  # force load
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # ── Fingerprints ────────────────────────────────────────────────────────
    azuredi_fp = azuredi_dump_fingerprint(args.azuredi_dir, args.pdf_doc_map)
    pdf_sha = compute_pdf_sha256(pdf_path) if pdf_path.exists() else None
    print(f"\nazuredi_fingerprint:          {azuredi_fp[:24]}…")
    print(f"pdf_sha256:                   {(pdf_sha or '(absent)')[:24]}…")

    # ── Cache root ──────────────────────────────────────────────────────────
    cache_root = T04CacheRoot(args.cache_root)

    # ── Loop the plan ───────────────────────────────────────────────────────
    print(f"\nBuilding {len(plan)} (unit, variant) configs …")
    summary: list[tuple[str, str, str, str]] = []
    for unit, variant in plan:
        indexable = indexable_node if unit == "node" else indexable_article
        if not indexable:
            print(f"  ! {unit}/{variant} skipped — no indexable units")
            summary.append((unit, variant, "-", "skipped"))
            continue
        config_hash, status = precompute_one(
            unit=unit,
            variant=variant,
            indexable=indexable,
            embedding_model=embedding_model,
            tokenizer=tokenizer,
            definitions_map=definitions_map,
            cache_root=cache_root,
            doc_id=args.doc_id,
            azure_doc_id=azure_doc_id,
            azuredi_fp=azuredi_fp,
            pdf_sha=pdf_sha,
            embedding_model_name=embedding_model_name,
            tokenizer_name=tokenizer_name,
            max_tokens=args.max_tokens,
            sparse_only=args.sparse_only,
            force=args.force,
        )
        summary.append((unit, variant, config_hash, status))

    # ── Final summary ───────────────────────────────────────────────────────
    print("\nDone.")
    print(f"  {'unit':<8s} {'variant':<10s} {'config_hash':<14s} {'status':<8s}")
    for unit, variant, h, status in summary:
        print(f"  {unit:<8s} {variant:<10s} {h:<14s} {status:<8s}")
    return 0


def _cuda_status() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return f"available — device 0 = {torch.cuda.get_device_name(0)}"
        return "not available — running on CPU"
    except ImportError:
        return "torch not installed"


if __name__ == "__main__":
    raise SystemExit(main())
