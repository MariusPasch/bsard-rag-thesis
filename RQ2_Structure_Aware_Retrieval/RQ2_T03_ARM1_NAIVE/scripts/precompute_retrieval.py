"""Populate the per-(config_hash, qset_hash) retrieval candidate pool.

For each question in --qset, runs hybrid retrieval (FAISS dense + BM25 sparse
+ RRF) over the cached arm-1 index for --pdf and saves the top-K candidate
pool — raw input for downstream evaluation methodologies that may apply
their own scoring.

Cache location:
    data/<doc_id>/results/<config_hash>/retrieval_pool/<qset_hash>.jsonl
    data/<doc_id>/results/<config_hash>/retrieval_pool/<qset_hash>.manifest.json

Each JSONL line:
    {
        "question_id": int,
        "question_text": str,
        "candidates": [
            {
                "rank": int,                  // 1-indexed in fused list
                "chunk_id": str,
                "fused_score": float,         // RRF score
                "dense_rank": int | null,
                "dense_score": float | null,
                "sparse_rank": int | null,
                "sparse_score": float | null,
                "bsard_ids": list[int],
                "start_char": int,
                "end_char": int
            },
            ...
        ]
    }

Usage (from RQ2_T03_ARM1_NAIVE/, with the venv active):
    python scripts/precompute_retrieval.py --pdf 1804_03_21_1804032150.pdf \
        --qset ../RQ2_T07_EVALUATION/ground_truth/bsard_test.json
    python scripts/precompute_retrieval.py --pdf 1804_03_21_1804032150.pdf \
        --qset ../RQ2_T07_EVALUATION/ground_truth/bsard_test.json \
                ../RQ2_T07_EVALUATION/ground_truth/bsard_train.json --top-k 200
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent

for _sibling in _REPO.glob("RQ2_T0*/src"):
    _s = str(_sibling)
    if _s not in sys.path:
        sys.path.insert(0, _s)

import sqlite3

from arm1_naive.cache import (
    CacheRoot,
    ConfigCache,
    PdfCache,
    compute_config_hash,
    compute_pdf_sha256,
)
from evaluation.cache import compute_qset_hash
from shared.embeddings import EmbeddingModel
from shared.faiss_store import SearchResult

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
SCHEMA_VERSION = 1
RRF_K = 60


# ── Qset loader (mirrors precompute_question_projection._load_qset_question_ids) ─
def _load_qset_question_ids(path: Path) -> list[int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [int(q) for q in payload]
    if isinstance(payload, dict):
        if "question_ids" in payload and isinstance(payload["question_ids"], list):
            return [int(q) for q in payload["question_ids"]]
        return [int(k) for k in payload.keys()]
    raise ValueError(f"Unsupported qset format in {path}")


# ── RRF that preserves dense/sparse rank+score per chunk ────────────────────
def _fuse_with_provenance(
    dense: list[SearchResult],
    sparse: list[SearchResult],
    rrf_k: int = RRF_K,
) -> list[dict]:
    """Return list of {chunk_id, fused_score, dense_rank, dense_score, sparse_rank, sparse_score, metadata}."""
    fused: dict[str, dict] = {}

    for rank0, r in enumerate(dense):
        fused[r.id] = {
            "chunk_id": r.id,
            "fused_score": 1.0 / (rrf_k + rank0 + 1),
            "dense_rank": rank0 + 1,
            "dense_score": r.score,
            "sparse_rank": None,
            "sparse_score": None,
            "metadata": r.metadata,
        }
    for rank0, r in enumerate(sparse):
        if r.id in fused:
            fused[r.id]["fused_score"] += 1.0 / (rrf_k + rank0 + 1)
            fused[r.id]["sparse_rank"] = rank0 + 1
            fused[r.id]["sparse_score"] = r.score
        else:
            fused[r.id] = {
                "chunk_id": r.id,
                "fused_score": 1.0 / (rrf_k + rank0 + 1),
                "dense_rank": None,
                "dense_score": None,
                "sparse_rank": rank0 + 1,
                "sparse_score": r.score,
                "metadata": r.metadata,
            }
    return sorted(fused.values(), key=lambda x: -x["fused_score"])


def _candidate_payload(rank1: int, fused_row: dict) -> dict:
    meta = fused_row["metadata"]
    return {
        "rank": rank1,
        "chunk_id": fused_row["chunk_id"],
        "fused_score": fused_row["fused_score"],
        "dense_rank": fused_row["dense_rank"],
        "dense_score": fused_row["dense_score"],
        "sparse_rank": fused_row["sparse_rank"],
        "sparse_score": fused_row["sparse_score"],
        "bsard_ids": list(meta.get("bsard_ids", [])),
        "start_char": meta.get("start_char"),
        "end_char": meta.get("end_char"),
    }


# ── Manifest ────────────────────────────────────────────────────────────────
def _manifest_payload(*, doc_id, config_hash, qset_hash, qset_size, embedding_model,
                      tokenizer_name, top_k) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "doc_id": doc_id,
        "config_hash": config_hash,
        "qset_hash": qset_hash,
        "qset_size": qset_size,
        "embedding_model": embedding_model,
        "tokenizer_name": tokenizer_name,
        "top_k": top_k,
        "rrf_k": RRF_K,
        "built_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def _is_manifest_valid(path: Path, *, config_hash, qset_hash, qset_size,
                       embedding_model, tokenizer_name, top_k) -> bool:
    if not path.exists():
        return False
    try:
        m = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return (
        m.get("schema_version") == SCHEMA_VERSION
        and m.get("config_hash") == config_hash
        and m.get("qset_hash") == qset_hash
        and m.get("qset_size") == qset_size
        and m.get("embedding_model") == embedding_model
        and m.get("tokenizer_name") == tokenizer_name
        and m.get("top_k") == top_k
        and m.get("rrf_k") == RRF_K
    )


# ── Main ────────────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--pdf", required=True, help="PDF filename (basename)")
    parser.add_argument("--qset", required=True, nargs="+", type=Path,
                        help="One or more qset JSON files")
    parser.add_argument("--top-k", type=int, default=200,
                        help="Top-K candidates to keep per question (default: %(default)s)")
    parser.add_argument("--strategy", choices=("sliding_window", "recursive"),
                        default="sliding_window")
    parser.add_argument("--window-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--tokenizer", default=None,
                        help="Tokenizer name (defaults to --embedding-model)")
    parser.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIR)
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--force", action="store_true")
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

    embedding_model_name = args.embedding_model
    tokenizer_name = args.tokenizer or embedding_model_name
    top_k = int(args.top_k)

    if args.strategy == "sliding_window":
        chunking_params = {"strategy": "sliding_window",
                           "window_size": int(args.window_size),
                           "stride": int(args.stride)}
    else:
        chunking_params = {"strategy": "recursive",
                           "max_tokens": int(args.max_tokens)}

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

    if not cfg_cache.exists() or not cfg_cache.validate(
        pdf_sha256=pdf_sha256,
        tokenizer_name=tokenizer_name,
        embedding_model=embedding_model_name,
        chunking_params=chunking_params,
    ):
        print(
            f"No valid index at {cfg_cache.path}.\n"
            "Hint: run scripts/precompute_pdf.py first to build the canonical config.",
            file=sys.stderr,
        )
        return 1

    print(f"doc_id      = {doc_id}")
    print(f"config_hash = {config_hash}")
    print(f"top_k       = {top_k}")
    print(f"cache_path  = {cfg_cache.path}")

    out_dir = pdf_cache.path / "results" / config_hash / "retrieval_pool"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load FAISS + BM25 once.
    print("[load] FAISS + BM25 …", flush=True)
    faiss_store, bm25_store = cfg_cache.load_indices()

    # Load embedding model once.
    print(f"[load] embedding model {embedding_model_name} …", flush=True)
    embedding_model = EmbeddingModel(model_name=embedding_model_name)

    conn = sqlite3.connect(str(args.db_path))
    conn.row_factory = sqlite3.Row

    try:
        for qset_path in args.qset:
            qset_path = qset_path.resolve()
            print(f"\n──── qset: {qset_path.name} ────")
            question_ids = _load_qset_question_ids(qset_path)
            qset_hash = compute_qset_hash(question_ids)
            print(f"qset_hash = {qset_hash}  ({len(question_ids)} question ids)")

            jsonl_path = out_dir / f"{qset_hash}.jsonl"
            manifest_path = out_dir / f"{qset_hash}.manifest.json"

            if not args.force and _is_manifest_valid(
                manifest_path,
                config_hash=config_hash, qset_hash=qset_hash,
                qset_size=len(question_ids),
                embedding_model=embedding_model_name,
                tokenizer_name=tokenizer_name, top_k=top_k,
            ) and jsonl_path.exists():
                print(f"[cache hit] manifest valid — skipping")
                continue

            # Pull question texts in one round-trip.
            placeholders = ",".join("?" for _ in question_ids)
            rows = conn.execute(
                f"SELECT question_id, question_text FROM questions "
                f"WHERE question_id IN ({placeholders})",
                question_ids,
            ).fetchall()
            qtext = {int(r["question_id"]): r["question_text"] for r in rows}
            missing = [q for q in question_ids if q not in qtext]
            if missing:
                print(f"  warning: {len(missing)} qids in qset not found in DB "
                      f"(first 5: {missing[:5]})")

            t0 = time.perf_counter()
            n_written = 0
            with jsonl_path.open("w", encoding="utf-8") as fh:
                for qid in question_ids:
                    q_text = qtext.get(qid)
                    if q_text is None:
                        continue
                    qv = embedding_model.encode_query(q_text)
                    dense = faiss_store.search(qv, k=top_k)
                    sparse = bm25_store.search(q_text, k=top_k)
                    fused = _fuse_with_provenance(dense, sparse)[:top_k]
                    payload = {
                        "question_id": qid,
                        "question_text": q_text,
                        "candidates": [_candidate_payload(i + 1, c) for i, c in enumerate(fused)],
                    }
                    fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
                    n_written += 1

            dt = time.perf_counter() - t0
            print(f"  wrote {n_written} rows to {jsonl_path.name} in {dt:.1f}s "
                  f"({dt / max(n_written, 1) * 1000:.0f} ms/query)")

            manifest = _manifest_payload(
                doc_id=doc_id, config_hash=config_hash, qset_hash=qset_hash,
                qset_size=len(question_ids),
                embedding_model=embedding_model_name,
                tokenizer_name=tokenizer_name, top_k=top_k,
            )
            manifest_path.write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"  manifest → {manifest_path.name}")
    finally:
        conn.close()

    print("\n[done]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
