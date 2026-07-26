"""Build per-question T08 viewer bundles from the cached retrieval pool.

Reads the precomputed pool at
    data/<doc_id>/results/<config_hash>/retrieval_pool/<qset_hash>.jsonl
and the question projection at
    data/<doc_id>/question_projections/<qset_hash>.json
to emit one self-contained bundle JSON per selected question, ready to feed
into ``streamlit run app.py -- --bundle-json q<qid>.json``.

The bundle holds the full pool (top-200 by default of the cached run); top-K
is filtered at render time by the T08 slider, not at bake time.

Output layout (default):
    data/<doc_id>/t08_bundles/<qset_hash>/q<qid>.json
    data/<doc_id>/t08_bundles/<qset_hash>/manifest.json

t08_bundles/ sits as a sibling of results/ rather than under
configs/<config_hash>/ because the bundle is a viewer artifact, not a
config-level cache (although it is derived from one specific config's pool —
the source config_hash is recorded in each bundle's _provenance and in the
sibling manifest.json).

Usage (from RQ2_T03_ARM1_NAIVE/, with the venv active):
    python scripts/build_t08_bundle.py --pdf 1804_03_21_1804032150.pdf \\
        --qset ../RQ2_T07_EVALUATION/ground_truth/bsard_train.json \\
        --question-ids 192 -v

    python scripts/build_t08_bundle.py --pdf 1804_03_21_1804032150.pdf \\
        --qset ../RQ2_T07_EVALUATION/ground_truth/bsard_train.json \\
        --filter-recall-ceiling-min 0.5 --filter-gt-in-pdf-min 2
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
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
from evaluation.cache import (
    PdfQuestionProjectionCache,
    T07CacheRoot,
    compute_qset_hash,
)

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


def _load_qset_question_ids(path: Path) -> list[int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [int(q) for q in payload]
    if isinstance(payload, dict):
        if "question_ids" in payload and isinstance(payload["question_ids"], list):
            return [int(q) for q in payload["question_ids"]]
        return [int(k) for k in payload.keys()]
    raise ValueError(f"Unsupported qset format in {path}")


def _build_chunking_params(args: argparse.Namespace) -> dict:
    if args.strategy == "sliding_window":
        return {
            "strategy": "sliding_window",
            "window_size": int(args.window_size),
            "stride": int(args.stride),
        }
    return {"strategy": "recursive", "max_tokens": int(args.max_tokens)}


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _select_question_ids(
    *,
    pool_qids: set[int],
    projection_rows: list,
    explicit_qids: list[int] | None,
    filter_recall_ceiling_min: float | None,
    filter_gt_in_pdf_min: int | None,
) -> list[int]:
    proj_by_qid = {r.question_id: r for r in projection_rows}

    if explicit_qids:
        candidates = [int(q) for q in explicit_qids]
    elif (
        filter_recall_ceiling_min is not None
        or filter_gt_in_pdf_min is not None
    ):
        candidates = [r.question_id for r in projection_rows]
    else:
        candidates = [r.question_id for r in projection_rows if r.recall_ceiling > 0.0]

    selected: list[int] = []
    for qid in candidates:
        row = proj_by_qid.get(qid)
        if row is None:
            logger.warning("qid %d has no projection row — skipped", qid)
            continue
        if filter_recall_ceiling_min is not None and row.recall_ceiling < filter_recall_ceiling_min:
            continue
        if filter_gt_in_pdf_min is not None and len(row.gt_in_pdf) < filter_gt_in_pdf_min:
            continue
        selected.append(qid)

    return [q for q in selected if q in pool_qids or _warn_pool_miss(q, pool_qids)]


def _warn_pool_miss(qid: int, pool_qids: set[int]) -> bool:
    if qid not in pool_qids:
        logger.warning("qid %d is not in the retrieval pool — skipped", qid)
        return False
    return True


def _build_ranked_item(candidate: dict, chunk: dict, doc_id: str) -> dict:
    bsard_ids = [int(b) for b in (candidate.get("bsard_ids") or [])]
    return {
        "id": candidate["chunk_id"],
        "score": candidate["fused_score"],
        "text": chunk["text"],
        "bsard_ids": bsard_ids,
        "start_char": chunk["start_char"],
        "end_char": chunk["end_char"],
        "start_token": chunk["start_token"],
        "end_token": chunk["end_token"],
        "metadata": {
            "bsard_ids": bsard_ids,
            "doc_id": doc_id,
            "start_char": chunk["start_char"],
            "end_char": chunk["end_char"],
            "start_token": chunk["start_token"],
            "end_token": chunk["end_token"],
            "rank": candidate["rank"],
            "dense_rank": candidate.get("dense_rank"),
            "dense_score": candidate.get("dense_score"),
            "sparse_rank": candidate.get("sparse_rank"),
            "sparse_score": candidate.get("sparse_score"),
        },
    }


def _build_bundle(
    *,
    qid: int,
    pool_row: dict,
    chunks_by_id: dict[str, dict],
    projection_row,
    doc_id: str,
    pdf_path: Path,
    pdf_filename: str,
    raw_text: str,
    article_spans: list[dict],
    config_hash: str,
    qset_hash: str,
) -> dict:
    candidates = pool_row.get("candidates", [])
    ranked_items: list[dict] = []
    for cand in candidates:
        chunk = chunks_by_id.get(cand["chunk_id"])
        if chunk is None:
            logger.warning(
                "chunk_id %s referenced by qid %d not found in chunks.json — skipped",
                cand["chunk_id"], qid,
            )
            continue
        ranked_items.append(_build_ranked_item(cand, chunk, doc_id))

    return {
        "document_id": 0,
        "pdf_path": str(pdf_path),
        "pdf_filename": pdf_filename,
        "metadata": {},
        "definitions": [],
        "articles": [],
        "raw_text": raw_text,
        "has_azure_extraction": False,
        "active_query_id": str(qid),
        "active_query_text": pool_row.get("question_text", ""),
        "gt_article_ids": list(projection_row.gt_in_pdf),
        "article_spans": article_spans,
        "initial_mode": "retrieval",
        "initial_arm": "arm1",
        "retrieval_result": {
            "query_id": str(qid),
            "query_text": pool_row.get("question_text", ""),
            "ranked_items": ranked_items,
            "method": "arm1_pool",
            "cost": {},
        },
        "_provenance": {
            "schema_version": SCHEMA_VERSION,
            "doc_id": doc_id,
            "config_hash": config_hash,
            "qset_hash": qset_hash,
            "n_pool_candidates": len(ranked_items),
            "gt_bsard_ids": list(projection_row.gt_bsard_ids),
            "gt_in_pdf": list(projection_row.gt_in_pdf),
            "recall_ceiling": projection_row.recall_ceiling,
            "built_at": _now_utc(),
        },
    }


def _is_existing_bundle_fresh(
    path: Path, *, config_hash: str, qset_hash: str, n_pool_candidates: int,
) -> bool:
    if not path.exists():
        return False
    try:
        prov = json.loads(path.read_text(encoding="utf-8")).get("_provenance", {})
    except (json.JSONDecodeError, OSError):
        return False
    return (
        prov.get("schema_version") == SCHEMA_VERSION
        and prov.get("config_hash") == config_hash
        and prov.get("qset_hash") == qset_hash
        and prov.get("n_pool_candidates") == n_pool_candidates
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--pdf", required=True, help="PDF filename (basename)")
    parser.add_argument("--qset", required=True, type=Path,
                        help="Path to a JSON qset file (auto-detects list / dict / question_ids forms)")
    parser.add_argument("--config-hash", default=None,
                        help="Override config_hash (default: derive from --strategy/window/stride/model)")
    parser.add_argument("--strategy", choices=("sliding_window", "recursive"),
                        default="sliding_window")
    parser.add_argument("--window-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--tokenizer", default=None,
                        help="Tokenizer name (defaults to --embedding-model)")
    parser.add_argument("--question-ids", type=int, nargs="+", default=None,
                        help="Explicit qid list (intersected with --filter-* if both given)")
    parser.add_argument("--filter-recall-ceiling-min", type=float, default=None,
                        help="Drop questions with recall_ceiling below this value")
    parser.add_argument("--filter-gt-in-pdf-min", type=int, default=None,
                        help="Drop questions with fewer than N gt articles found in this PDF")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Override output dir (default: data/<doc_id>/t08_bundles/<qset_hash>/)")
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
    if not args.qset.exists():
        print(f"qset file not found: {args.qset}", file=sys.stderr)
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

    if args.config_hash:
        config_hash = args.config_hash
    else:
        pdf_sha256 = compute_pdf_sha256(pdf_path)
        config_hash = compute_config_hash(
            pdf_sha256=pdf_sha256,
            tokenizer_name=tokenizer_name,
            embedding_model=embedding_model_name,
            chunking_params=chunking_params,
        )

    cfg_cache = ConfigCache(pdf_cache, config_hash)
    if not cfg_cache.chunks_path().exists():
        print(
            f"chunks.json missing at {cfg_cache.chunks_path()}.\n"
            "Hint: run scripts/precompute_pdf.py first to populate the config cache.",
            file=sys.stderr,
        )
        return 1

    question_ids = _load_qset_question_ids(args.qset)
    qset_hash = compute_qset_hash(question_ids)

    pool_jsonl = pdf_cache.path / "results" / config_hash / "retrieval_pool" / f"{qset_hash}.jsonl"
    if not pool_jsonl.exists():
        print(
            f"Retrieval pool missing at {pool_jsonl}.\n"
            "Hint: run scripts/precompute_retrieval.py first to build the pool.",
            file=sys.stderr,
        )
        return 1

    t07_root = T07CacheRoot(args.cache_root)
    proj_cache = PdfQuestionProjectionCache(t07_root, doc_id)
    if not proj_cache.exists(qset_hash):
        print(
            f"Question projection missing at {proj_cache.path_for(qset_hash)}.\n"
            "Hint: run RQ2_T07_EVALUATION/scripts/precompute_question_projection.py first.",
            file=sys.stderr,
        )
        return 1

    print(f"doc_id      = {doc_id}")
    print(f"config_hash = {config_hash}")
    print(f"qset_hash   = {qset_hash}")
    print(f"pool        = {pool_jsonl}")

    chunks_payload = json.loads(cfg_cache.chunks_path().read_text(encoding="utf-8"))
    chunks_by_id = {c["chunk_id"]: c for c in chunks_payload.get("chunks", [])}
    print(f"chunks      = {len(chunks_by_id)}")

    raw_text = pdf_cache.raw_text_path().read_text(encoding="utf-8")

    # Bake article spans (bsard_id → char offsets) into each bundle so the
    # viewer can render GT overlays without needing the BSARD DB at runtime.
    spans_payload = json.loads(pdf_cache.article_spans_path().read_text(encoding="utf-8"))
    article_spans = [
        {
            "bsard_id": int(s["bsard_id"]),
            "start_char": int(s["start_char"]),
            "end_char": int(s["end_char"]),
        }
        for s in spans_payload.get("spans", [])
    ]
    print(f"article_spans = {len(article_spans)}")

    pool_by_qid: dict[int, dict] = {}
    with pool_jsonl.open("r", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            pool_by_qid[int(row["question_id"])] = row
    print(f"pool size   = {len(pool_by_qid)} questions")

    projection_rows = proj_cache.load(qset_hash)
    print(f"projection  = {len(projection_rows)} rows")

    selected_qids = _select_question_ids(
        pool_qids=set(pool_by_qid.keys()),
        projection_rows=projection_rows,
        explicit_qids=args.question_ids,
        filter_recall_ceiling_min=args.filter_recall_ceiling_min,
        filter_gt_in_pdf_min=args.filter_gt_in_pdf_min,
    )

    if not selected_qids:
        print("No questions selected — nothing to do.")
        return 0

    out_dir = args.out_dir or (pdf_cache.path / "t08_bundles" / qset_hash)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"out_dir     = {out_dir}")
    print(f"selected    = {len(selected_qids)} qids")

    proj_by_qid = {r.question_id: r for r in projection_rows}
    pdf_filename = pdf_path.name

    n_written = 0
    n_skipped = 0
    for qid in selected_qids:
        pool_row = pool_by_qid[qid]
        projection_row = proj_by_qid[qid]
        n_pool = len(pool_row.get("candidates", []))
        out_path = out_dir / f"q{qid}.json"

        if not args.force and _is_existing_bundle_fresh(
            out_path,
            config_hash=config_hash,
            qset_hash=qset_hash,
            n_pool_candidates=n_pool,
        ):
            logger.info("[cache hit] q%d.json — skipping", qid)
            n_skipped += 1
            continue

        bundle = _build_bundle(
            qid=qid,
            pool_row=pool_row,
            chunks_by_id=chunks_by_id,
            projection_row=projection_row,
            doc_id=doc_id,
            pdf_path=pdf_path,
            pdf_filename=pdf_filename,
            raw_text=raw_text,
            article_spans=article_spans,
            config_hash=config_hash,
            qset_hash=qset_hash,
        )

        with out_path.open("w", encoding="utf-8", newline="\n") as fh:
            json.dump(bundle, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
        logger.info("wrote %s (%d candidates)", out_path.name, len(bundle["retrieval_result"]["ranked_items"]))
        n_written += 1

    filters_applied: dict = {}
    if args.question_ids:
        filters_applied["explicit_question_ids"] = list(args.question_ids)
    if args.filter_recall_ceiling_min is not None:
        filters_applied["recall_ceiling_min"] = args.filter_recall_ceiling_min
    if args.filter_gt_in_pdf_min is not None:
        filters_applied["gt_in_pdf_min"] = args.filter_gt_in_pdf_min

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "doc_id": doc_id,
        "pdf_filename": pdf_filename,
        "config_hash": config_hash,
        "qset_hash": qset_hash,
        "n_bundles": len(selected_qids),
        "n_written": n_written,
        "n_skipped": n_skipped,
        "filters_applied": filters_applied,
        "top_k_in_pool": max((len(pool_by_qid[q].get("candidates", [])) for q in selected_qids), default=0),
        "built_at": _now_utc(),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"\n[done] {n_written} bundles written, {n_skipped} cache hits")
    print(f"manifest    = {out_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
