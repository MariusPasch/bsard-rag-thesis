"""Pure retrieval against a populated cache (errors out on cache miss).

Loads the FAISS + BM25 stores from
``<cache_root>/<doc_id>/configs/<config_hash>/`` and runs the standard Arm 1
hybrid pipeline (dense + BM25 + RRF). No PDF extraction, article location,
chunking, or embedding happens — those must have already been produced by
``precompute_pdf.py`` or a prior ``run_arm1`` call.

Usage (from RQ2_T03_ARM1_NAIVE/, with the venv active):
    python scripts/query_cached.py --pdf 1804_03_21_1804032150.pdf --query "Je me marie ..."
    python scripts/query_cached.py --pdf <name>.pdf --query "..." --top-k 5

The (strategy, window_size/stride/max_tokens, embedding_model, tokenizer)
flags must match what was used to build the cache; otherwise the resolved
``config_hash`` won't point to a real folder.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent

for _sibling in _REPO.glob("RQ2_T0*/src"):
    _s = str(_sibling)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from arm1_naive.cache import (
    CacheRoot,
    ConfigCache,
    PdfCache,
    compute_config_hash,
    compute_pdf_sha256,
)
from arm1_naive.retriever import retrieve_arm1
from shared.embeddings import EmbeddingModel

logger = logging.getLogger(__name__)

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
    parser.add_argument("--query", required=True, help="Natural-language query string")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--retrieval-top-k", type=int, default=100)
    parser.add_argument("--strategy", choices=("sliding_window", "recursive"),
                        default="sliding_window")
    parser.add_argument("--window-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--tokenizer", default=None,
                        help="Tokenizer name (defaults to --embedding-model)")
    parser.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIR)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--json", action="store_true",
                        help="Emit results as JSON (one object per result on stdout)")
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
    if not args.cache_root.exists():
        print(f"Cache root not found: {args.cache_root}", file=sys.stderr)
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

    if not cfg_cache.exists():
        print(
            f"[cache miss] no manifest at {cfg_cache.manifest_path()}\n"
            "Run scripts/precompute_pdf.py first with the same flags.",
            file=sys.stderr,
        )
        return 2
    if not cfg_cache.validate(
        pdf_sha256=pdf_sha256,
        tokenizer_name=tokenizer_name,
        embedding_model=embedding_model_name,
        chunking_params=chunking_params,
    ):
        print(
            f"[cache stale] manifest at {cfg_cache.manifest_path()} does not match "
            "current params; rerun precompute_pdf.py --force.",
            file=sys.stderr,
        )
        return 2

    faiss_store, bm25_store = cfg_cache.load_indices()
    embedding_model = EmbeddingModel(model_name=embedding_model_name)

    ranked = retrieve_arm1(
        args.query, faiss_store, bm25_store, embedding_model,
        retrieval_top_k=int(args.retrieval_top_k),
        top_k=int(args.top_k),
    )

    if args.json:
        for r in ranked:
            print(json.dumps({
                "id": r.id,
                "score": r.score,
                "bsard_ids": r.metadata.get("bsard_ids", []),
                "text": r.metadata.get("text", ""),
            }, ensure_ascii=False))
    else:
        for i, r in enumerate(ranked, 1):
            preview = r.metadata.get("text", "").replace("\n", " ")[:200]
            bsard = r.metadata.get("bsard_ids", [])
            print(f"#{i:2d}  score={r.score:.4f}  bsard_ids={bsard}")
            print(f"     {r.id}")
            print(f"     {preview}")
            print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
