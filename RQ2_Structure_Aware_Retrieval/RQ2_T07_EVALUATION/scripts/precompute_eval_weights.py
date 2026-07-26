"""Populate the chunk-bsard weight cache for one PDF + arm1_config_hash.

Reads chunks + raw_text from T03's cache and writes
``data/<doc_id>/configs/<arm1_config_hash>/eval/{manifest.json,
chunk_bsard_weights.csv, chunk_bsard_weights.pkl, article_token_ranges.json}``
(CACHE_DESIGN.md §4 + §6.2).

Usage (from RQ2_T07_EVALUATION/, with the venv active):
    python scripts/precompute_eval_weights.py --pdf 1804_03_21_1804032150.pdf --arm1-config-hash a3f7c2cd1e90
    python scripts/precompute_eval_weights.py --pdf <name>.pdf --arm1-config-hash <hash> --force

The ``--arm1-config-hash`` value is printed by ``precompute_pdf.py`` in
RQ2_T03_ARM1_NAIVE; pasting it here ties the eval weights to those exact chunks.
"""

from __future__ import annotations

import os

import argparse
import logging
import sqlite3
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent

for _sibling in _REPO.glob("RQ2_T0*/src"):
    _s = str(_sibling)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from arm1_naive.cache import CacheRoot
from evaluation.cache import T07CacheRoot
from evaluation.weight_precomputer import precompute_weights_cached

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path(os.environ.get("RQ2_BSARD_DB", str(_REPO / "data" / "bsard_corpus.db")))
DEFAULT_CACHE_ROOT = _REPO / "RQ2_T07_EVALUATION" / "data"
DEFAULT_T03_CACHE_ROOT = _REPO / "RQ2_T03_ARM1_NAIVE" / "data"
DEFAULT_TOKENIZER = "intfloat/multilingual-e5-large-instruct"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--pdf", required=True, help="PDF filename (basename), e.g. 1804_03_21_1804032150.pdf")
    parser.add_argument("--arm1-config-hash", required=True,
                        help="config_hash from T03 (printed by precompute_pdf.py)")
    parser.add_argument("--force", action="store_true",
                        help="Recompute even if the eval manifest validates")
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER,
                        help="Tokenizer name (must match T03's; default: %(default)s)")
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT,
                        help="T07 cache junction (default: %(default)s)")
    parser.add_argument("--t03-cache-root", type=Path, default=DEFAULT_T03_CACHE_ROOT,
                        help="T03 cache junction (default: %(default)s)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if not args.db_path.exists():
        print(f"DB not found: {args.db_path}", file=sys.stderr)
        return 1
    if not args.cache_root.exists():
        print(
            f"T07 cache root not found: {args.cache_root}\n"
            "Hint: create the data/ junction first (see README §Setup).",
            file=sys.stderr,
        )
        return 1
    if not args.t03_cache_root.exists():
        print(f"T03 cache root not found: {args.t03_cache_root}", file=sys.stderr)
        return 1

    doc_id = Path(args.pdf).stem

    conn = sqlite3.connect(str(args.db_path))
    conn.row_factory = sqlite3.Row
    try:
        weights = precompute_weights_cached(
            doc_id=doc_id,
            arm1_config_hash=args.arm1_config_hash,
            t07_cache_root=T07CacheRoot(args.cache_root),
            t03_cache_root=CacheRoot(args.t03_cache_root),
            db_conn=conn,
            tokenizer_name=args.tokenizer,
            pdf_filename=args.pdf,
            force=args.force,
        )
    finally:
        conn.close()

    print(f"doc_id           = {doc_id}")
    print(f"arm1_config_hash = {args.arm1_config_hash}")
    print(f"n_weight_rows    = {len(weights)}")
    out_dir = args.cache_root / doc_id / "configs" / args.arm1_config_hash / "eval"
    print(f"cache_path       = {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
