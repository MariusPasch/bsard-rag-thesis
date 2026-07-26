"""Populate the experiment cache for one PDF — no queries are run.

After this script finishes, ``run_arm1`` (or ``query_cached.py``) on the same
PDF + same params hits a fully populated cache: extract / locate / chunk /
embed / build-index are all skipped (CACHE_DESIGN.md §11.6–11.7).

Usage (from RQ2_T03_ARM1_NAIVE/, with the venv active):
    python scripts/precompute_pdf.py --pdf 1804_03_21_1804032150.pdf
    python scripts/precompute_pdf.py --pdf 1804_03_21_1804032150.pdf --strategy recursive --max-tokens 512
    python scripts/precompute_pdf.py --pdf 1804_03_21_1804032150.pdf --force

Prints the resolved ``config_hash`` and cache path so downstream tools (T07,
T08) can address the same folder.
"""

from __future__ import annotations

import argparse
import logging
import os
import sqlite3
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent

for _sibling in _REPO.glob("RQ2_T0*/src"):
    _s = str(_sibling)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from arm1_naive import chunker as _chunker
from arm1_naive import indexer as _indexer
from arm1_naive.cache import (
    CacheRoot,
    ConfigCache,
    PdfCache,
    compute_bsard_db_fingerprint,
    compute_config_hash,
    compute_pdf_sha256,
)
from shared.embeddings import EmbeddingModel
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path(
    os.environ.get(
        "RQ2_BSARD_DB",
        str(_REPO / "data" / "bsard_corpus.db"),
    )
)
DEFAULT_PDF_DIR = _REPO / "RQ2_T02_DATA_LOADER/data/pdfs"
DEFAULT_CACHE_ROOT = _REPO / "RQ2_T03_ARM1_NAIVE" / "data"
DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-large-instruct"


def _build_chunking_params(args: argparse.Namespace) -> dict:
    if args.strategy == "sliding_window":
        return {
            "strategy": "sliding_window",
            "window_size": int(args.window_size),
            "stride": int(args.stride),
        }
    return {"strategy": "recursive", "max_tokens": int(args.max_tokens)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--pdf", required=True, help="PDF filename (basename)")
    parser.add_argument("--strategy", choices=("sliding_window", "recursive"),
                        default="sliding_window")
    parser.add_argument("--window-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--force", action="store_true",
                        help="Invalidate and rebuild even if the manifest validates")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL,
                        help="Sentence-transformer model name (default: %(default)s)")
    parser.add_argument("--tokenizer", default=None,
                        help="Tokenizer name (defaults to --embedding-model)")
    parser.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIR)
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    pdf_path = (args.pdf_dir / args.pdf).resolve()
    if not pdf_path.exists():
        print(f"PDF not found: {pdf_path}", file=sys.stderr)
        return 1
    if not args.db_path.exists():
        print(f"DB not found: {args.db_path}", file=sys.stderr)
        return 1
    if not args.cache_root.exists():
        print(
            f"Cache root not found: {args.cache_root}\n"
            "Hint: create the data/ junction first (see README §Setup).",
            file=sys.stderr,
        )
        return 1

    embedding_model_name = args.embedding_model
    tokenizer_name = args.tokenizer or embedding_model_name
    chunking_params = _build_chunking_params(args)

    doc_id = pdf_path.stem
    cache_root = CacheRoot(args.cache_root)
    pdf_cache = PdfCache(cache_root, doc_id)

    pdf_sha256 = compute_pdf_sha256(pdf_path)
    config_hash = compute_config_hash(
        pdf_sha256=pdf_sha256,
        tokenizer_name=tokenizer_name,
        embedding_model=embedding_model_name,
        chunking_params=chunking_params,
    )
    cfg_cache = ConfigCache(pdf_cache, config_hash)

    print(f"doc_id      = {doc_id}")
    print(f"pdf_sha256  = {pdf_sha256}")
    print(f"config_hash = {config_hash}")
    print(f"cache_path  = {cfg_cache.path}")

    if not args.force and cfg_cache.exists() and cfg_cache.validate(
        pdf_sha256=pdf_sha256,
        tokenizer_name=tokenizer_name,
        embedding_model=embedding_model_name,
        chunking_params=chunking_params,
    ):
        print("[cache hit] manifest valid — nothing to do")
        return 0

    if args.force:
        print("[force] rebuilding regardless of manifest state")
    else:
        print("[cache miss] running full pipeline …")

    conn = sqlite3.connect(str(args.db_path))
    conn.row_factory = sqlite3.Row

    try:
        db_fingerprint = compute_bsard_db_fingerprint(conn)

        raw_text = pdf_cache.load_or_extract_text(pdf_path)
        print(f"  raw_text     {len(raw_text):,} chars")

        article_spans = pdf_cache.load_or_locate_articles(
            raw_text, args.pdf, conn, db_fingerprint
        )
        print(f"  article_spans {len(article_spans)} spans")

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if args.strategy == "sliding_window":
            chunks = _chunker.chunk_sliding_window(
                raw_text, tokenizer, doc_id, article_spans,
                window_size=int(args.window_size), stride=int(args.stride),
            )
        else:
            chunks = _chunker.chunk_recursive(
                raw_text, tokenizer, doc_id, article_spans,
                max_tokens=int(args.max_tokens),
            )
        print(f"  chunks       {len(chunks)}")
        cfg_cache.save_chunks(chunks)

        embedding_model = EmbeddingModel(model_name=embedding_model_name)
        _indexer.build_arm1_index(
            chunks, embedding_model, doc_id, args.cache_root,
            batch_size=int(args.batch_size),
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
        print(f"  manifest     written to {cfg_cache.manifest_path()}")

    finally:
        conn.close()

    print("[done] cache populated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
