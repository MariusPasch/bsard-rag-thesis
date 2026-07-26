"""PDF-centric experiment cache for Arm 1.

Layout under the cache root (a link into the shared data root's ``pdf_cache/``):

    <root>/<doc_id>/
    ├── pdf_meta.json
    ├── raw_text.txt
    ├── article_spans.json
    ├── configs/<config_hash>/
    │   ├── manifest.json
    │   ├── chunks.json
    │   ├── faiss.index, faiss_meta.json   (written by FAISSStore)
    │   ├── bm25.pkl                        (written by BM25Store)
    │   └── eval/                           (T07-owned; see RQ2_T07_EVALUATION)
    └── results/<config_hash>/<run_label>.jsonl

T03 owns everything outside ``eval/`` and ``question_projections/``; T07 owns
those two trees (CACHE_DESIGN.md §4). Cross-process write collisions are
avoided by ownership of paths, not by locking.

Invalidation
------------
Manifest-driven. ``pdf_meta.json`` is the integrity anchor for the doc tree;
``configs/<hash>/manifest.json`` is the anchor for that config. Mismatched
``pdf_sha256``, tokenizer/embedding-model name, or chunking params return False
from ``ConfigCache.validate(...)`` and the caller rebuilds. A
``bsard_db_fingerprint`` change additionally invalidates ``article_spans.json``
(article→bsard_id assignments may have shifted).

Schema versioning
-----------------
Every JSON manifest / sidecar carries ``schema_version: 1``. Readers reject
unknown versions (loud failure, not silent fallback). Bump the constant and
add a migration when introducing a breaking change.

Future hook
-----------
T08 RAG-VISUALIZATION currently re-chunks in memory; once published it can
read chunks/indices via ``PdfCache`` + ``ConfigCache`` (CACHE_DESIGN.md §10.8).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import fitz  # PyMuPDF

from arm1_naive.chunker import ArticleSpan, Chunk, locate_articles
from shared.bm25_store import BM25Store
from shared.faiss_store import FAISSStore

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
_HASH_LEN = 12

_PDF_META_FILE = "pdf_meta.json"
_RAW_TEXT_FILE = "raw_text.txt"
_ARTICLE_SPANS_FILE = "article_spans.json"
_DRIFT_STATUS_FILE = "drift_status.json"
_MANIFEST_FILE = "manifest.json"
_CHUNKS_FILE = "chunks.json"
_BM25_FILE = "bm25.pkl"
_FAISS_INDEX_FILE = "faiss.index"


# ── Internals ────────────────────────────────────────────────────────────────


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _check_schema(payload: dict, source: Path) -> None:
    v = payload.get("schema_version")
    if v != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported schema_version={v!r} in {source} "
            f"(expected {SCHEMA_VERSION}). Refusing to read."
        )


# ── Hashing ──────────────────────────────────────────────────────────────────


def compute_pdf_sha256(pdf_path: Path | str) -> str:
    """SHA-256 of the PDF binary content as 64-char hex.

    Streamed in 1 MiB blocks so very large PDFs don't blow up memory.
    """
    h = hashlib.sha256()
    with open(pdf_path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def compute_config_hash(
    pdf_sha256: str,
    tokenizer_name: str,
    embedding_model: str,
    chunking_params: dict,
) -> str:
    """First 12 hex chars of sha256 of canonical-JSON of the four hash inputs.

    Canonicalisation uses ``sort_keys=True`` and tight separators so equivalent
    dicts always serialise to the same bytes regardless of insertion order.
    """
    payload = {
        "pdf_sha256": pdf_sha256,
        "tokenizer_name": tokenizer_name,
        "embedding_model": embedding_model,
        "chunking": chunking_params,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:_HASH_LEN]


def compute_bsard_db_fingerprint(db_conn: sqlite3.Connection) -> str:
    """SHA-256 of (count(articles), max(bsard_id), count(questions), max(question_id)).

    Captures bulk content shifts (regenerated DB, new articles/questions, new
    bsard_id assignments) without scanning every row.
    """
    n_art, max_bsard, n_q, max_q = db_conn.execute(
        "SELECT (SELECT COUNT(*) FROM articles), "
        "(SELECT COALESCE(MAX(bsard_id), 0) FROM articles WHERE bsard_id IS NOT NULL), "
        "(SELECT COUNT(*) FROM questions), "
        "(SELECT COALESCE(MAX(question_id), 0) FROM questions)"
    ).fetchone()
    blob = json.dumps(
        [int(n_art), int(max_bsard), int(n_q), int(max_q)],
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


# ── Cache root + per-PDF cache ───────────────────────────────────────────────


class CacheRoot:
    """Wraps the shared cache root (a link into the data root's ``pdf_cache/``)."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def pdf(self, doc_id: str) -> "PdfCache":
        return PdfCache(self, doc_id)


class PdfCache:
    """One folder per PDF (``<root>/<doc_id>/``).

    Holds the PDF-level invariants (raw_text, article_spans, pdf_meta) shared
    across all configs of that PDF. Per-config artifacts live under
    ``configs/<config_hash>/`` via :class:`ConfigCache`.
    """

    def __init__(self, root: CacheRoot, doc_id: str) -> None:
        self.root = root
        self.doc_id = doc_id
        self.path = root.path / doc_id

    # ── Path helpers ────────────────────────────────────────────────
    def pdf_meta_path(self) -> Path:
        return self.path / _PDF_META_FILE

    def raw_text_path(self) -> Path:
        return self.path / _RAW_TEXT_FILE

    def article_spans_path(self) -> Path:
        return self.path / _ARTICLE_SPANS_FILE

    def drift_status_path(self) -> Path:
        return self.path / _DRIFT_STATUS_FILE

    def _load_drift_filter(self) -> tuple[set[int] | None, str | None]:
        """Read ``drift_status.json`` and return (allowed_bsard_ids, signature).

        Opt-in: returns ``(None, None)`` if the file is absent. When present,
        only verdicts == ``IN_PDF`` are kept — a bsard_id whose canonical text
        is genuinely in the PDF. ABROGATED / MISSING / PARTIAL are dropped.

        The signature is a short hash over the sorted allowed-id list, used to
        invalidate cached ``article_spans.json`` when the filter changes.
        """
        path = self.drift_status_path()
        if not path.exists():
            return None, None
        payload = _read_json(path)
        _check_schema(payload, path)
        allowed = {
            int(v["bsard_id"])
            for v in payload.get("verdicts", [])
            if v.get("verdict") == "IN_PDF"
        }
        blob = json.dumps(sorted(allowed), separators=(",", ":")).encode("utf-8")
        signature = hashlib.sha256(blob).hexdigest()[:_HASH_LEN]
        return allowed, signature

    def configs_dir(self) -> Path:
        return self.path / "configs"

    def results_dir(self, config_hash: str) -> Path:
        return self.path / "results" / config_hash

    # ── PDF text + meta ─────────────────────────────────────────────
    def validate_pdf(self, pdf_path: Path | str) -> bool:
        """Return True iff cached ``pdf_meta.pdf_sha256`` matches the PDF on disk."""
        meta_path = self.pdf_meta_path()
        if not meta_path.exists():
            return False
        try:
            payload = _read_json(meta_path)
            _check_schema(payload, meta_path)
        except (json.JSONDecodeError, ValueError):
            return False
        return payload.get("pdf_sha256") == compute_pdf_sha256(pdf_path)

    def load_or_extract_text(self, pdf_path: Path | str) -> str:
        """Return raw_text from cache; extract via PyMuPDF on miss/staleness.

        Side-effect on miss: writes ``pdf_meta.json`` + ``raw_text.txt``.
        Concatenation strategy mirrors :func:`inspect_chunking.main` so that
        offsets line up with whatever any earlier ad-hoc analysis produced.
        """
        pdf_path = Path(pdf_path)
        if self.validate_pdf(pdf_path) and self.raw_text_path().exists():
            return self.raw_text_path().read_text(encoding="utf-8")

        with fitz.open(str(pdf_path)) as fz:
            n_pages = len(fz)
            raw_text = "\n\n".join(page.get_text() for page in fz)

        self.path.mkdir(parents=True, exist_ok=True)
        # Force LF newlines so character offsets are stable across OSes.
        with open(self.raw_text_path(), "w", encoding="utf-8", newline="\n") as f:
            f.write(raw_text)
        _write_json(
            self.pdf_meta_path(),
            {
                "schema_version": SCHEMA_VERSION,
                "doc_id": self.doc_id,
                "pdf_sha256": compute_pdf_sha256(pdf_path),
                "n_pages": n_pages,
                "n_chars": len(raw_text),
                "extracted_at": _now_utc(),
            },
        )
        return raw_text

    def pdf_meta(self) -> dict:
        payload = _read_json(self.pdf_meta_path())
        _check_schema(payload, self.pdf_meta_path())
        return payload

    # ── BSARD article spans ─────────────────────────────────────────
    def load_or_locate_articles(
        self,
        raw_text: str,
        pdf_filename: str,
        db_conn: sqlite3.Connection,
        bsard_db_fingerprint: str,
    ) -> list[ArticleSpan]:
        """Return article spans from cache; recompute on miss / stale fingerprint.

        ``bsard_db_fingerprint`` mismatches force re-derivation because article
        bsard_id assignments could have shifted in a regenerated DB.

        If a sibling ``drift_status.json`` exists, only spans whose bsard_id is
        verdict==IN_PDF are kept (opt-in PDF-level drift filter). The applied
        filter signature is stamped into ``article_spans.json`` so cache
        validation invalidates when the drift status changes or is removed.
        """
        allowed_bsard_ids, drift_signature = self._load_drift_filter()

        spans_path = self.article_spans_path()
        if spans_path.exists():
            try:
                payload = _read_json(spans_path)
                _check_schema(payload, spans_path)
                if (
                    payload.get("bsard_db_fingerprint") == bsard_db_fingerprint
                    and payload.get("drift_filter_signature") == drift_signature
                ):
                    return _decode_spans(payload.get("spans", []))
            except (json.JSONDecodeError, ValueError, KeyError):
                pass  # fall through and recompute

        spans = locate_articles(
            raw_text, pdf_filename, db_conn,
            allowed_bsard_ids=allowed_bsard_ids,
        )
        _write_json(
            spans_path,
            {
                "schema_version": SCHEMA_VERSION,
                "doc_id": self.doc_id,
                "pdf_filename": pdf_filename,
                "bsard_db_fingerprint": bsard_db_fingerprint,
                "drift_filter_signature": drift_signature,
                "computed_at": _now_utc(),
                "spans": [
                    {
                        "bsard_id": s.bsard_id,
                        "start_char": s.start_char,
                        "end_char": s.end_char,
                    }
                    for s in spans
                ],
            },
        )
        return spans

    def load_article_spans(self) -> list[ArticleSpan]:
        """Read-only consumer (used by T07). Raises if no spans cached."""
        path = self.article_spans_path()
        if not path.exists():
            raise FileNotFoundError(
                f"article_spans.json missing for doc {self.doc_id} at {path}"
            )
        payload = _read_json(path)
        _check_schema(payload, path)
        return _decode_spans(payload.get("spans", []))

    def pdf_bsard_ids(self) -> set[int]:
        """Set of bsard_ids identified within this PDF."""
        return {s.bsard_id for s in self.load_article_spans()}

    # ── Per-run results ─────────────────────────────────────────────
    def write_results_jsonl(
        self,
        config_hash: str,
        run_label: str,
        results: Iterable[Any],
    ) -> Path:
        """Persist one ``RetrievalResult`` per JSONL line under ``results/<hash>/``."""
        out_path = self.results_dir(config_hash) / f"{run_label}.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8", newline="\n") as f:
            for r in results:
                f.write(json.dumps(_to_serialisable(r), ensure_ascii=False))
                f.write("\n")
        return out_path


def _decode_spans(rows: list[dict]) -> list[ArticleSpan]:
    return [
        ArticleSpan(
            bsard_id=int(s["bsard_id"]),
            start_char=int(s["start_char"]),
            end_char=int(s["end_char"]),
        )
        for s in rows
    ]


def _to_serialisable(obj: Any) -> Any:
    """Coerce a dataclass (or already-dict) to a plain dict for JSON output."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    return obj


# ── Per-config sub-cache ─────────────────────────────────────────────────────


class ConfigCache:
    """One folder per (chunking params + tokenizer + embedding model)."""

    def __init__(self, pdf_cache: PdfCache, config_hash: str) -> None:
        self.pdf_cache = pdf_cache
        self.config_hash = config_hash
        self.path = pdf_cache.configs_dir() / config_hash

    # ── Path helpers ────────────────────────────────────────────────
    def manifest_path(self) -> Path:
        return self.path / _MANIFEST_FILE

    def chunks_path(self) -> Path:
        return self.path / _CHUNKS_FILE

    def faiss_dir(self) -> Path:
        """FAISSStore writes ``faiss.index`` + ``faiss_meta.json`` into this directory."""
        return self.path

    def bm25_path(self) -> Path:
        return self.path / _BM25_FILE

    # ── State checks ────────────────────────────────────────────────
    def exists(self) -> bool:
        return self.manifest_path().exists()

    def validate(
        self,
        *,
        pdf_sha256: str,
        tokenizer_name: str,
        embedding_model: str,
        chunking_params: dict,
    ) -> bool:
        """Return True iff manifest matches the inputs and all artifacts exist."""
        if not self.exists():
            return False
        try:
            m = self.read_manifest()
        except (json.JSONDecodeError, ValueError):
            return False

        if m.get("config_hash") != self.config_hash:
            return False
        if m.get("pdf_sha256") != pdf_sha256:
            return False
        if m.get("tokenizer_name") != tokenizer_name:
            return False
        if m.get("embedding_model") != embedding_model:
            return False
        if m.get("chunking") != chunking_params:
            return False

        if not self.chunks_path().exists():
            return False
        if not (self.faiss_dir() / _FAISS_INDEX_FILE).exists():
            return False
        if not self.bm25_path().exists():
            return False
        return True

    # ── Chunks ──────────────────────────────────────────────────────
    def save_chunks(self, chunks: list[Chunk]) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        _write_json(
            self.chunks_path(),
            {
                "schema_version": SCHEMA_VERSION,
                "doc_id": self.pdf_cache.doc_id,
                "config_hash": self.config_hash,
                "n_chunks": len(chunks),
                "chunks": [dataclasses.asdict(c) for c in chunks],
            },
        )

    def load_chunks(self) -> list[Chunk]:
        payload = _read_json(self.chunks_path())
        _check_schema(payload, self.chunks_path())
        return [
            Chunk(
                chunk_id=c["chunk_id"],
                text=c["text"],
                doc_id=c["doc_id"],
                start_char=int(c["start_char"]),
                end_char=int(c["end_char"]),
                start_token=int(c["start_token"]),
                end_token=int(c["end_token"]),
                bsard_ids=[int(b) for b in c.get("bsard_ids", [])],
            )
            for c in payload.get("chunks", [])
        ]

    # ── Indices ─────────────────────────────────────────────────────
    def save_indices(self, faiss_store: FAISSStore, bm25_store: BM25Store) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        faiss_store.save(str(self.faiss_dir()))
        bm25_store.save(str(self.bm25_path()))

    def load_indices(self) -> tuple[FAISSStore, BM25Store]:
        faiss_store = FAISSStore.load(str(self.faiss_dir()))
        bm25_store = BM25Store.load(str(self.bm25_path()))
        return faiss_store, bm25_store

    # ── Manifest ────────────────────────────────────────────────────
    def write_manifest(
        self,
        *,
        pdf_sha256: str,
        tokenizer_name: str,
        embedding_model: str,
        embedding_dim: int,
        chunking_params: dict,
        n_chunks: int,
    ) -> None:
        _write_json(
            self.manifest_path(),
            {
                "schema_version": SCHEMA_VERSION,
                "doc_id": self.pdf_cache.doc_id,
                "config_hash": self.config_hash,
                "pdf_sha256": pdf_sha256,
                "tokenizer_name": tokenizer_name,
                "embedding_model": embedding_model,
                "embedding_dim": embedding_dim,
                "chunking": chunking_params,
                "n_chunks": n_chunks,
                "built_at": _now_utc(),
            },
        )

    def read_manifest(self) -> dict:
        payload = _read_json(self.manifest_path())
        _check_schema(payload, self.manifest_path())
        return payload


# ── Convenience: default run-label generator ─────────────────────────────────


def default_run_label() -> str:
    """ISO-8601-ish UTC timestamp with ``-`` instead of ``:`` (Windows-safe)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
