"""Precompute question→PDF GT projections for one qset across one or more PDFs.

For each (qset, PDF) pair, writes
``data/<doc_id>/question_projections/<qset_hash>.json`` (CACHE_DESIGN.md §6.3).
The qset hash is the first 12 hex of sha256 of the comma-joined sorted unique
question ids; identical qsets always map to the same file.

Usage (from RQ2_T07_EVALUATION/, with the venv active):
    python scripts/precompute_question_projection.py \\
        --qset ground_truth/bsard_test.json \\
        --pdfs 1804_03_21_1804032150.pdf 1804_03_21_1804032151.pdf

Accepted ``--qset`` formats (auto-detected):
  • flat list of ints                ``[192, 193, …]``
  • legacy GT dict (default)         ``{"192": [...], "193": [...], …}``
  • dict with ``question_ids`` key   ``{"question_ids": [192, 193, …]}``
"""

from __future__ import annotations

import os

import argparse
import json
import logging
import sqlite3
import sys
from pathlib import Path
from typing import Iterable

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent

for _sibling in _REPO.glob("RQ2_T0*/src"):
    _s = str(_sibling)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from arm1_naive.cache import CacheRoot, PdfCache
from evaluation.cache import (
    PdfQuestionProjectionCache,
    T07CacheRoot,
    compute_bsard_db_fingerprint,
    compute_qset_hash,
)
from evaluation.projection import build_projection

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path(os.environ.get("RQ2_BSARD_DB", str(_REPO / "data" / "bsard_corpus.db")))
DEFAULT_CACHE_ROOT = _REPO / "RQ2_T07_EVALUATION" / "data"
DEFAULT_T03_CACHE_ROOT = _REPO / "RQ2_T03_ARM1_NAIVE" / "data"


def _load_qset_question_ids(path: Path) -> list[int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [int(q) for q in payload]
    if isinstance(payload, dict):
        if "question_ids" in payload and isinstance(payload["question_ids"], list):
            return [int(q) for q in payload["question_ids"]]
        return [int(k) for k in payload.keys()]
    raise ValueError(f"Unsupported qset format in {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--qset", required=True, type=Path,
                        help="Path to a JSON file with question ids (see formats above)")
    parser.add_argument("--pdfs", required=True, nargs="+",
                        help="One or more PDF basenames to project onto")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing projection files")
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

    if not args.qset.exists():
        print(f"qset file not found: {args.qset}", file=sys.stderr)
        return 1
    if not args.db_path.exists():
        print(f"DB not found: {args.db_path}", file=sys.stderr)
        return 1
    if not args.cache_root.exists():
        print(f"T07 cache root not found: {args.cache_root}", file=sys.stderr)
        return 1
    if not args.t03_cache_root.exists():
        print(f"T03 cache root not found: {args.t03_cache_root}", file=sys.stderr)
        return 1

    question_ids = _load_qset_question_ids(args.qset)
    qset_hash = compute_qset_hash(question_ids)
    print(f"qset_hash    = {qset_hash}")
    print(f"n_questions  = {len(question_ids)}")
    print(f"n_pdfs       = {len(args.pdfs)}")

    t07_root = T07CacheRoot(args.cache_root)
    t03_root = CacheRoot(args.t03_cache_root)

    conn = sqlite3.connect(str(args.db_path))
    conn.row_factory = sqlite3.Row
    try:
        db_fingerprint = compute_bsard_db_fingerprint(conn)

        for pdf in args.pdfs:
            doc_id = Path(pdf).stem
            t03_pdf_cache = PdfCache(t03_root, doc_id)
            try:
                rows = build_projection(
                    question_ids, doc_id, t03_pdf_cache, conn,
                )
            except FileNotFoundError as exc:
                print(f"  {doc_id}: skipped — {exc}", file=sys.stderr)
                continue

            cache = PdfQuestionProjectionCache(t07_root, doc_id)
            target = cache.path_for(qset_hash)
            if target.exists() and not args.force:
                print(f"  {doc_id}: [skip] {target} already exists (use --force)")
                continue

            cache.save(qset_hash, rows, bsard_db_fingerprint=db_fingerprint)
            covered = sum(1 for r in rows if r.gt_in_pdf)
            print(
                f"  {doc_id}: {len(rows)} rows  "
                f"(covered={covered}, mean_recall_ceiling="
                f"{(sum(r.recall_ceiling for r in rows) / len(rows) if rows else 0.0):.3f})  "
                f"→ {target}"
            )
    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
