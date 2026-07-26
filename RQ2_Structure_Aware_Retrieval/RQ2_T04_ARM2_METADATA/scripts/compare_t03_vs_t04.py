"""Compare T03 (naive hybrid) against every T04 variant on one PDF.

What it does:
  1. Build the per-doc question subset for ``--doc-id <stem>.pdf`` — restrict
     to the BSARD test+train questions whose ground truth intersects the
     bsard_ids reachable in T04's index for that PDF. Each question's GT is
     clipped to that intersection so the "recoverable recall" is measurable.
  2. Run T03 on that subset (full Arm 1 hybrid pipeline; cache hits free).
  3. Run T04 for every (unit, variant) pair listed in ``RUN_PLAN`` (or
     ``SMOKE_PLAN`` under ``--smoke``).
  4. Compute Recall@10/100, MRR@10, nDCG@10 per query per method, then
     aggregate to a stem-namespaced CSV.

Usage:
    python scripts/compare_t03_vs_t04.py --doc-id 1967_10_10_1967101056 --smoke -v
    python scripts/compare_t03_vs_t04.py --doc-id <stem> --skip-t03     # T04 only
    python scripts/compare_t03_vs_t04.py --doc-id <stem> --variants raw enriched

The five curated stems live in RQ2_T00_ORCHESTRATOR/data/selected_pdfs.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sqlite3
import sys
import time
from pathlib import Path
from typing import Iterable

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent

# Make sibling src/ importable when running outside an installed-editable env.
for _sibling in _REPO.glob("RQ2_T0*/src"):
    s = str(_sibling)
    if s not in sys.path:
        sys.path.insert(0, s)

from arm1_naive.cache import (
    CacheRoot,
    ConfigCache,
    PdfCache,
    compute_bsard_db_fingerprint,
    compute_config_hash,
    compute_pdf_sha256,
)
from arm1_naive import chunker as _chunker
from arm1_naive import indexer as _arm1_indexer
from arm1_naive.retriever import retrieve_arm1
from shared.bm25_store import BM25Store

# Optional cosine-weighted recall (T07 CN-T07-001, shipped 2026-05-01).
# Falls back gracefully if T07 hasn't shipped these symbols (older venv).
try:
    from evaluation.metrics import cosine_weighted_recall_at_k
    from evaluation.question_subsets import (
        build_ground_truth_with_cosines,
        load_extraction_status,
    )
    _COSINE_AVAILABLE = True
except ImportError:
    _COSINE_AVAILABLE = False
from arm2_metadata.azuredi_loader import load_azuredi_corpus
from arm2_metadata.bsard_link import link_corpus_to_bsard
from arm2_metadata.cache import default_run_label
from arm2_metadata.retriever import run_arm2_metadata
from data_loader.models import RetrievalResult
from evaluation.metrics import (
    mrr_at_k, ndcg_at_k, recall_at_k,
)
from shared.embeddings import EmbeddingModel
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-large-instruct"
DEFAULT_PDF_DIR = _REPO / "RQ2_T02_DATA_LOADER" / "data" / "pdfs"

# Module-level holders. All four DOC_*/PDF_* values are resolved from --doc-id
# at runtime by _resolve_doc() (or overridden by --pdf-path / --pdf-dir).
EMBEDDING_MODEL = DEFAULT_EMBEDDING_MODEL
PDF_FILENAME: str | None = None
AZURE_DOC_ID: int | None = None
DOC_ID: str | None = None
PDF_DOC_ID: str | None = None
PDF_PATH: Path | None = None

BSARD_DB_PATH = _REPO / "dataset_creation_output" / "bsard_corpus.db"
PDF_DOC_MAP_CSV = _REPO / "RQ2_T02_DATA_LOADER" / "data" / "csv" / "pdf_document_map.csv"
AZUREDI_DIR = _REPO / "RQ2_T04_ARM2_METADATA" / "azuredi"
T03_CACHE_ROOT = _REPO / "RQ2_T03_ARM1_NAIVE" / "data"
T04_INDEX_ROOT = _REPO / "RQ2_T04_ARM2_METADATA" / "data"
RUNS_DIR = _REPO / "RQ2_T07_EVALUATION" / "ground_truth" / "runs"
GT_TEST = _REPO / "RQ2_T07_EVALUATION" / "ground_truth" / "bsard_test.json"
GT_TRAIN = _REPO / "RQ2_T07_EVALUATION" / "ground_truth" / "bsard_train.json"

# Output file paths are stem-namespaced (filled at runtime by _resolve_doc).
OUT_CSV: Path | None = None
OUT_PER_QUERY_JSON: Path | None = None


def _resolve_doc(
    doc_id: str,
    pdf_dir: Path,
    pdf_doc_map_csv: Path,
) -> tuple[str, int, Path]:
    """Map a BSARD PDF stem to (pdf_filename, azure_doc_id, pdf_path).

    Reads ``pdf_document_map.csv`` to look up the AzureDI integer DocumentId
    for the given stem. ``pdf_path`` is computed as ``<pdf_dir>/<doc_id>.pdf``
    (existence is not enforced here so callers can override via ``--pdf-path``).
    """
    import pandas as pd
    df = pd.read_csv(pdf_doc_map_csv, encoding="utf-8")
    target = f"{doc_id}.pdf"
    row = df[df["pdf_filename"] == target]
    if row.empty:
        raise ValueError(
            f"doc_id {doc_id!r} not in {pdf_doc_map_csv} — has_azure_extraction "
            f"is false for this PDF, or the stem is wrong."
        )
    return target, int(row.iloc[0]["document_id"]), pdf_dir / target


# ── Run plan ──────────────────────────────────────────────────────────────────


RUN_PLAN: list[tuple[str, str]] = [
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

# Smaller plan used by --smoke (one canonical pick per (unit, axis)).
SMOKE_PLAN: list[tuple[str, str]] = [
    ("node", "raw"),
    ("node", "enriched"),
    ("node", "summary"),
    ("node", "full"),
    ("article", "raw"),
    ("article", "full"),
]


# ── Subset builder ────────────────────────────────────────────────────────────


def build_subset(out_path: Path) -> tuple[dict[str, list[int]], set[int]]:
    """Build the per-doc subset and return (gt_clipped, reachable_bsard_ids).

    Strategy: restrict to questions whose GT intersects the bsard_ids reachable
    in T04's index. Each question's GT is clipped to that intersection so that
    metrics measure "recoverable recall" rather than penalising both methods
    for an out-of-corpus floor.
    """
    nodes, _, _ = load_azuredi_corpus(AZUREDI_DIR, PDF_DOC_MAP_CSV)
    nodes = link_corpus_to_bsard(nodes, BSARD_DB_PATH)
    reachable = {b for n in nodes for b in n.bsard_ids if n.doc_id == AZURE_DOC_ID}

    gt: dict[str, list[int]] = {}
    for path in (GT_TEST, GT_TRAIN):
        with open(path, encoding="utf-8") as f:
            for qid, bids in json.load(f).items():
                inter = sorted(set(bids) & reachable)
                if inter:
                    gt[str(qid)] = inter

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(gt, f, ensure_ascii=False, indent=2)

    return gt, reachable


def _build_cosine_subset(
    gt: dict[str, list[int]],
) -> dict[str, dict[int, float | None]] | None:
    """Build the cosine-weighted GT for the same subset, via T07's accessor.

    Returns ``{qid: {bsard_id: cosine | None}}`` keyed identically to the
    plain GT. Returns None if T07's cosine API isn't available, or if the
    upstream JSONL has zero non-null cosines for our subset (in which case
    the cosine columns would be uninformative).

    Per T07 CN-T07-001 implementation notes, JSONL cosine coverage is
    ~14.2% today (974 / 6,844 article entries), entirely from the two T03
    docs regenerated under CN-T03-002. Doc 2 has 9 of those entries — all
    in the 0.09–0.40 drift band per T03 review.
    """
    if not _COSINE_AVAILABLE:
        return None
    df = load_extraction_status()
    qids = {int(q) for q in gt}
    cosines = build_ground_truth_with_cosines(df, qids)
    # Filter to bsard_ids that are in our (clipped) GT — the JSONL accessor
    # returns the full GT for each question; we only want what survived
    # the "intersect with reachable bsard_ids" clipping in _build_subset.
    keep = {qid: set(bids) for qid, bids in gt.items()}
    filtered = {
        qid: {bid: cos for bid, cos in row.items() if bid in keep.get(qid, set())}
        for qid, row in cosines.items()
        if qid in gt
    }
    n_nonnull = sum(1 for row in filtered.values() for c in row.values() if c is not None)
    if n_nonnull == 0:
        logger.info("No non-null extraction_cosine values in subset — Cw columns suppressed.")
        return None
    return filtered


# ── Metric helpers ────────────────────────────────────────────────────────────


def _bsard_ranking_from_result(result: RetrievalResult) -> list[str]:
    """Linearise ``result.ranked_items`` into one bsard_id per row.

    For chunk-level results (multiple bsard_ids per item): expand to one row
    per bsard_id, preserving order, deduping (keep first occurrence). For
    article-level results: take metadata.bsard_id directly.
    """
    out: list[str] = []
    seen: set[str] = set()
    for item in result.ranked_items:
        meta = item.get("metadata") or {}
        ids = []
        if meta.get("bsard_id") is not None:
            ids = [str(int(meta["bsard_id"]))]
        elif meta.get("bsard_ids"):
            ids = [str(int(b)) for b in meta["bsard_ids"]]
        else:
            ids = [str(item["id"])]
        for bid in ids:
            if bid not in seen:
                out.append(bid)
                seen.add(bid)
    return out


def compute_metrics(
    results: list[RetrievalResult],
    gt: dict[str, list[int]],
    cosine_gt: dict[str, dict[int, float | None]] | None = None,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Return (aggregate, per_query) — both keyed by metric name.

    When ``cosine_gt`` is provided (and T07's `cosine_weighted_recall_at_k`
    is importable), additional `Cw/R@10` and `Cw/R@100` columns appear in
    both the per-query and aggregate dicts. Cosine-weighted recall
    discounts GT items by their `extraction_cosine` (per CN-T03-002),
    automatically deweighting GT-drift cases like doc 2.
    """
    per_query: dict[str, dict[str, float]] = {}
    base_keys = ["R@10", "R@100", "MRR@10", "nDCG@10", "latency_ms"]
    if cosine_gt is not None and _COSINE_AVAILABLE:
        base_keys += ["Cw/R@10", "Cw/R@100"]
    sums: dict[str, float] = {k: 0.0 for k in base_keys}

    for r in results:
        if r.query_id not in gt:
            continue
        rel = {str(b) for b in gt[r.query_id]}
        ranked = _bsard_ranking_from_result(r)
        m: dict[str, float] = {
            "R@10": recall_at_k(ranked, rel, 10),
            "R@100": recall_at_k(ranked, rel, 100),
            "MRR@10": mrr_at_k(ranked, rel, 10),
            "nDCG@10": ndcg_at_k(ranked, rel, 10),
            "latency_ms": float((r.cost or {}).get("latency_ms", 0.0)),
        }
        if cosine_gt is not None and _COSINE_AVAILABLE:
            cos_row = cosine_gt.get(r.query_id, {})
            m["Cw/R@10"] = cosine_weighted_recall_at_k(ranked, cos_row, 10)
            m["Cw/R@100"] = cosine_weighted_recall_at_k(ranked, cos_row, 100)
        per_query[r.query_id] = m
        for k, v in m.items():
            sums[k] += v

    n = len(per_query) or 1
    aggregate = {k: v / n for k, v in sums.items()}
    aggregate["n_queries"] = n
    return aggregate, per_query


# ── T03 runner ────────────────────────────────────────────────────────────────


def run_t03_on_subset(
    embedding_model: EmbeddingModel | None,
    tokenizer,
    question_ids: set[str],
    *,
    sparse_only: bool = False,
) -> list[RetrievalResult]:
    """Run T03 hybrid retrieval on the per-doc subset using its on-disk cache.

    Falls back to a fresh chunk → embed pass if the FAISS+BM25 indices haven't
    been persisted yet for this PDF (i.e. a prior precompute_pdf.py was not
    run).
    """
    chunking_params = {
        "strategy": "sliding_window",
        "window_size": 512,
        "stride": 256,
    }
    pdf_sha = compute_pdf_sha256(PDF_PATH)

    cache_root = CacheRoot(T03_CACHE_ROOT)
    pdf_cache = PdfCache(cache_root, PDF_DOC_ID)
    faiss_store = None
    bm25_store: BM25Store | None = None

    if sparse_only:
        # Sparse-only path: build BM25 in-memory from cached chunks (or chunk
        # fresh if no cache hit). No FAISS, no embedding model required.
        # We don't persist via T03's ConfigCache because the BM25 build is
        # cheap and T03's manifest pins the embedding model name.
        logger.info("[T03 sparse-only] building BM25 in-memory")
        conn = sqlite3.connect(str(BSARD_DB_PATH))
        conn.row_factory = sqlite3.Row
        try:
            db_fp = compute_bsard_db_fingerprint(conn)
            raw_text = pdf_cache.load_or_extract_text(PDF_PATH)
            spans = pdf_cache.load_or_locate_articles(
                raw_text, PDF_FILENAME, conn, db_fp,
            )
            chunks = _chunker.chunk_sliding_window(
                raw_text, tokenizer, PDF_DOC_ID, spans,
                window_size=chunking_params["window_size"],
                stride=chunking_params["stride"],
            )
        finally:
            conn.close()

        bm25_store = BM25Store()
        bm25_store.build(
            ids=[c.chunk_id for c in chunks],
            texts=[c.text for c in chunks],
            metadata=[
                {
                    "text": c.text,
                    "doc_id": c.doc_id,
                    "start_char": c.start_char,
                    "end_char": c.end_char,
                    "start_token": c.start_token,
                    "end_token": c.end_token,
                    "bsard_ids": c.bsard_ids,
                }
                for c in chunks
            ],
        )
    else:
        config_hash = compute_config_hash(
            pdf_sha256=pdf_sha,
            tokenizer_name=EMBEDDING_MODEL,
            embedding_model=EMBEDDING_MODEL,
            chunking_params=chunking_params,
        )
        cfg_cache = ConfigCache(pdf_cache, config_hash)

        if cfg_cache.validate(
            pdf_sha256=pdf_sha,
            tokenizer_name=EMBEDDING_MODEL,
            embedding_model=EMBEDDING_MODEL,
            chunking_params=chunking_params,
        ):
            logger.info("[T03 cache hit] %s", cfg_cache.path)
            faiss_store, bm25_store = cfg_cache.load_indices()
        else:
            logger.info("[T03 cache miss] %s — rebuilding", cfg_cache.path)
            conn = sqlite3.connect(str(BSARD_DB_PATH))
            conn.row_factory = sqlite3.Row
            try:
                db_fp = compute_bsard_db_fingerprint(conn)
                raw_text = pdf_cache.load_or_extract_text(PDF_PATH)
                spans = pdf_cache.load_or_locate_articles(
                    raw_text, PDF_FILENAME, conn, db_fp,
                )
                chunks = _chunker.chunk_sliding_window(
                    raw_text, tokenizer, PDF_DOC_ID, spans,
                    window_size=chunking_params["window_size"],
                    stride=chunking_params["stride"],
                )
                cfg_cache.save_chunks(chunks)
                faiss_store, bm25_store, *_ = (*_arm1_indexer.build_arm1_index(
                    chunks, embedding_model, PDF_DOC_ID, T03_CACHE_ROOT,
                    output_dir=cfg_cache.path,
                ), )
                cfg_cache.write_manifest(
                    pdf_sha256=pdf_sha,
                    tokenizer_name=EMBEDDING_MODEL,
                    embedding_model=EMBEDDING_MODEL,
                    embedding_dim=embedding_model.dimension,
                    chunking_params=chunking_params,
                    n_chunks=len(chunks),
                )
            finally:
                conn.close()

    conn = sqlite3.connect(str(BSARD_DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("SELECT question_id, question_text FROM questions").fetchall()
    finally:
        conn.close()

    out: list[RetrievalResult] = []
    for r in rows:
        qid = str(r["question_id"])
        if qid not in question_ids:
            continue
        q_text = r["question_text"]
        t0 = time.perf_counter()
        if sparse_only:
            ranked = bm25_store.search(q_text, k=100)
        else:
            ranked = retrieve_arm1(
                q_text, faiss_store, bm25_store, embedding_model,
                retrieval_top_k=100, top_k=100,
            )
        latency_ms = (time.perf_counter() - t0) * 1000
        out.append(RetrievalResult(
            query_id=qid,
            query_text=q_text,
            ranked_items=[
                {
                    "id": x.id,
                    "score": float(x.score),
                    "text": x.metadata.get("text", ""),
                    "metadata": {
                        "bsard_ids": x.metadata.get("bsard_ids", []),
                        "doc_id": x.metadata.get("doc_id"),
                        "start_char": x.metadata.get("start_char"),
                        "end_char": x.metadata.get("end_char"),
                    },
                }
                for x in ranked
            ],
            method=("arm1_naive_sparse_only" if sparse_only else "arm1_naive"),
            cost={"llm_calls": 0, "tokens_in": 0, "tokens_out": 0,
                  "latency_ms": round(latency_ms, 1)},
        ))
    return out


# ── T04 runner ────────────────────────────────────────────────────────────────


def run_t04_variants(
    embedding_model: EmbeddingModel | None,
    tokenizer,
    question_ids: set[str],
    plan: Iterable[tuple[str, str]],
    *,
    sparse_only: bool = False,
    force_reindex: bool = False,
) -> dict[str, list[RetrievalResult]]:
    """Run every (unit, variant) in ``plan`` and return {method_label: results}."""
    out: dict[str, list[RetrievalResult]] = {}
    config = {
        "models": {
            "embedding_model": EMBEDDING_MODEL,
            "tokenizer": EMBEDDING_MODEL,
        },
        "arm2": {
            "retrieval_top_k": 100,
            "top_k": 100,
            "max_tokens": 512,
            "use_llm_classifier": False,
            "sparse_only": sparse_only,
            "boost": {
                "term_match": 1.3,
                "used_in": 1.5,
                "jurisdiction": 1.1,
                "drop_non_effective": True,
                "hard_filter_jurisdiction": False,
            },
        },
    }
    suffix = "_sparse_only" if sparse_only else ""
    for unit, variant in plan:
        method = f"arm2_metadata_{unit}_{variant}{suffix}"
        logger.info("=== %s ===", method)
        results = run_arm2_metadata(
            azuredi_dir=AZUREDI_DIR,
            pdf_document_map_csv=PDF_DOC_MAP_CSV,
            bsard_db_path=BSARD_DB_PATH,
            pdf_path=PDF_PATH if PDF_PATH.exists() else None,
            embedding_model=embedding_model,
            tokenizer=tokenizer,
            llm=None,
            index_root=T04_INDEX_ROOT,
            config=config,
            doc_id=DOC_ID,
            azure_doc_id=AZURE_DOC_ID,
            variant=variant,
            unit=unit,
            run_label=f"compare_{default_run_label()}",
            question_ids=question_ids,
            split=None,
            force_reindex=force_reindex,
        )
        out[method] = results
    return out


# ── CSV writer ────────────────────────────────────────────────────────────────


METRIC_KEYS_BASE = ("R@10", "R@100", "MRR@10", "nDCG@10", "latency_ms", "n_queries")
METRIC_KEYS_WITH_COSINE = ("R@10", "R@100", "Cw/R@10", "Cw/R@100",
                           "MRR@10", "nDCG@10", "latency_ms", "n_queries")


def write_csv(
    rows: list[tuple[str, dict[str, float]]],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Choose column set based on whether ANY row has cosine columns populated.
    has_cosine = any("Cw/R@10" in r for _, r in rows)
    metric_keys = METRIC_KEYS_WITH_COSINE if has_cosine else METRIC_KEYS_BASE
    bests: dict[str, str] = {}
    # "best" = highest for metrics; lowest for latency_ms
    for k in metric_keys:
        if k == "n_queries":
            continue
        higher_is_better = k != "latency_ms"
        relevant = [(m, r[k]) for m, r in rows if k in r]
        if not relevant:
            continue
        winner = max(relevant, key=lambda x: x[1]) if higher_is_better else min(relevant, key=lambda x: x[1])
        bests[k] = winner[0]

    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method"] + list(metric_keys) + [f"best:{k}" for k in metric_keys if k != "n_queries"])
        for m, r in rows:
            row = [m]
            for k in metric_keys:
                v = r.get(k, "")
                if isinstance(v, float):
                    v = round(v, 4)
                row.append(v)
            for k in metric_keys:
                if k == "n_queries":
                    continue
                row.append("*" if bests.get(k) == m else "")
            w.writerow(row)


# ── Main ──────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    # Declare globals at the very top so the `default=PDF_PATH` lookups in
    # argparse below don't trip Python's "use before global declaration"
    # SyntaxError. We mutate these after parse so downstream module-level
    # constants reflect the CLI overrides.
    global EMBEDDING_MODEL, PDF_PATH, AZUREDI_DIR, BSARD_DB_PATH, T04_INDEX_ROOT
    global PDF_FILENAME, AZURE_DOC_ID, DOC_ID, PDF_DOC_ID, OUT_CSV, OUT_PER_QUERY_JSON

    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--doc-id", type=str, required=True,
        help="BSARD PDF stem (e.g. 1967_10_10_1967101056). The AzureDI integer "
             "DocumentId, pdf_filename, and pdf_path are resolved from "
             "pdf_document_map.csv. Required — pick from the curated 5 in "
             "RQ2_T00_ORCHESTRATOR/data/selected_pdfs.json.",
    )
    parser.add_argument(
        "--pdf-dir", type=Path, default=DEFAULT_PDF_DIR,
        help="Source PDF directory. pdf_path defaults to <pdf_dir>/<doc_id>.pdf. "
             "Default: %(default)s.",
    )
    parser.add_argument("--skip-t03", action="store_true",
                        help="Don't (re)run T03; results-only mode for T04 ablations.")
    parser.add_argument("--variants", nargs="*", default=None,
                        help="Restrict to (unit/variant) pairs (e.g. node/raw, article/full).")
    parser.add_argument("--smoke", action="store_true",
                        help="Use SMOKE_PLAN (smaller variant set) instead of RUN_PLAN.")
    parser.add_argument("--sparse-only", action="store_true",
                        help="Skip dense (FAISS) retrieval entirely — use BM25 only "
                             "for both T03 and T04. No embedding model is loaded.")
    parser.add_argument("--force-reindex", action="store_true",
                        help="Invalidate every existing T04 cache config and rebuild. "
                             "Use after pipeline-level changes that don't bump the "
                             "linker_version automatically.")
    parser.add_argument(
        "--pdf-path", type=Path, default=None,
        help="Override the path to the source PDF (used for the pdf_sha256 "
             "cache fingerprint). Default: <pdf_dir>/<doc_id>.pdf — typically "
             "only needed when running on a different machine where the data "
             "root resolves to a different path.",
    )
    parser.add_argument(
        "--azuredi-dir", type=Path, default=AZUREDI_DIR,
        help="Override the AzureDI dump directory. Default: %(default)s.",
    )
    parser.add_argument(
        "--bsard-db", type=Path, default=BSARD_DB_PATH,
        help="Override the BSARD corpus DB path. Default: %(default)s.",
    )
    parser.add_argument(
        "--cache-root", type=Path, default=T04_INDEX_ROOT,
        help="Override the T04 cache root. Default: %(default)s.",
    )
    parser.add_argument(
        "--embedding-model", default=DEFAULT_EMBEDDING_MODEL,
        help=(
            "Sentence-transformer name. Default is mE5-large-instruct (the "
            "production model). Pass intfloat/multilingual-e5-small-instruct "
            "for a CPU-friendly smoke run."
        ),
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    # Resolve doc-specific constants from --doc-id (via pdf_document_map.csv).
    pdf_filename, azure_doc_id, default_pdf_path = _resolve_doc(
        args.doc_id, args.pdf_dir, PDF_DOC_MAP_CSV,
    )
    DOC_ID = args.doc_id
    PDF_DOC_ID = args.doc_id
    PDF_FILENAME = pdf_filename
    AZURE_DOC_ID = azure_doc_id
    PDF_PATH = args.pdf_path if args.pdf_path is not None else default_pdf_path

    EMBEDDING_MODEL = args.embedding_model
    AZUREDI_DIR = args.azuredi_dir
    BSARD_DB_PATH = args.bsard_db
    T04_INDEX_ROOT = args.cache_root

    # Stem-namespaced output paths so per-doc runs don't clobber each other.
    OUT_CSV = T04_INDEX_ROOT / f"comparison_t03_vs_t04_{DOC_ID}.csv"
    OUT_PER_QUERY_JSON = T04_INDEX_ROOT / f"comparison_per_query_{DOC_ID}.json"

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("arm2_metadata").setLevel(logging.INFO)

    plan = SMOKE_PLAN if args.smoke else RUN_PLAN
    if args.variants:
        wanted = set()
        for v in args.variants:
            unit, variant = v.split("/")
            wanted.add((unit, variant))
        # Filter from the resolved `plan` (NOT from RUN_PLAN), so --smoke
        # composes correctly with --variants. Review #15-A5: previously the
        # filter source was hardcoded to RUN_PLAN, silently overriding --smoke
        # for variants only present in the full plan.
        plan = [p for p in plan if p in wanted]

    print(f"PDF                : {PDF_FILENAME}")
    print(f"doc_id (BSARD stem): {DOC_ID}")
    print(f"azure_document_id  : {AZURE_DOC_ID}")
    print(f"BSARD DB           : {BSARD_DB_PATH}")
    print(f"AzureDI dir        : {AZUREDI_DIR}")
    print(f"T03 cache root     : {T03_CACHE_ROOT}")
    print(f"T04 index root     : {T04_INDEX_ROOT}")

    # 1. Subset
    subset_path = RUNS_DIR / f"t04_{DOC_ID}.json"
    gt, reachable = build_subset(subset_path)
    print(f"\nSubset: {len(gt)} questions, {len(reachable)} reachable bsard_ids")
    print(f"  questions: {sorted(gt, key=int)}")
    print(f"  saved -> {subset_path}")
    question_ids = set(gt)
    if not question_ids:
        print("Empty subset — bailing.", file=sys.stderr)
        return 1

    # 2. Models (load once)
    if args.sparse_only:
        print("\nLoading models (BM25 only) — embedding model skipped.")
        embedding_model = None
    else:
        print(f"\nLoading models ({EMBEDDING_MODEL}) …")
        embedding_model = EmbeddingModel(model_name=EMBEDDING_MODEL)
        _ = embedding_model.dimension  # force load
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)

    all_results: dict[str, list[RetrievalResult]] = {}

    # 3. T03
    if not args.skip_t03:
        suffix = " (sparse-only)" if args.sparse_only else ""
        print(f"\n=== T03 (arm1_naive, sliding_window 512/256){suffix} ===")
        method_key = "arm1_naive" + ("_sparse_only" if args.sparse_only else "")
        all_results[method_key] = run_t03_on_subset(
            embedding_model, tokenizer, question_ids,
            sparse_only=args.sparse_only,
        )

    # 4. T04 variants
    suffix = " (sparse-only)" if args.sparse_only else ""
    print(f"\n=== T04 variants{suffix} ===")
    all_results.update(run_t04_variants(
        embedding_model, tokenizer, question_ids, plan,
        sparse_only=args.sparse_only,
        force_reindex=args.force_reindex,
    ))

    # 5. Metrics
    print("\n=== Aggregate metrics ===")
    cosine_gt = _build_cosine_subset(gt)
    if cosine_gt is not None:
        n_with_cos = sum(1 for row in cosine_gt.values()
                         for v in row.values() if v is not None)
        n_total = sum(len(row) for row in cosine_gt.values())
        print(f"  (cosine GT: {n_with_cos}/{n_total} GT items have non-null "
              f"extraction_cosine — Cw/R@k columns enabled)")
    metric_keys = METRIC_KEYS_WITH_COSINE if cosine_gt is not None else METRIC_KEYS_BASE
    aggregate_rows: list[tuple[str, dict[str, float]]] = []
    per_query_table: dict[str, dict] = {}
    for method, results in all_results.items():
        agg, per_q = compute_metrics(results, gt, cosine_gt=cosine_gt)
        aggregate_rows.append((method, agg))
        per_query_table[method] = per_q
        line = " | ".join(
            f"{k}={agg[k]:.4f}" if isinstance(agg.get(k), float) else f"{k}={agg.get(k)}"
            for k in metric_keys
        )
        print(f"  {method:42s}  {line}")

    write_csv(aggregate_rows, OUT_CSV)
    OUT_PER_QUERY_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_PER_QUERY_JSON.write_text(
        json.dumps({
            "subset_path": str(subset_path),
            "ground_truth": gt,
            "cosine_ground_truth": cosine_gt,
            "per_query_metrics": per_query_table,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nCSV       -> {OUT_CSV}")
    print(f"per-query -> {OUT_PER_QUERY_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
