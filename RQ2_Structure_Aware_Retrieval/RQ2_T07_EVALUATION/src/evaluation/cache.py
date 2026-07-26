"""T07's view of the shared PDF-centric experiment cache.

T07 owns three trees inside the shared layout (PLAN_per_pdf_pipeline §B.8):

    <root>/<doc_id>/eval/<arm1_config_hash>/
        ├── manifest.json                 — T07 eval manifest
        ├── chunk_bsard_weights.csv       — canonical (CSV-with-header)
        ├── chunk_bsard_weights.pkl       — fast-load mirror; regenerable from CSV
        └── article_token_ranges.json     — debug: bsard_id → (start_token, end_token)

    <root>/<doc_id>/question_projections/<qset_hash>.json
        — per-(qset, doc) ground-truth projection

    <root>/<doc_id>/eval/last_run.json
        — T07-side stamp of the most recent successful evaluation pass
          (written by ``evaluation.eval_stamp``, not by this module)

T07 reads T03's outputs (raw_text, article_spans, chunks, FAISS+BM25, T03
manifest) via ``arm1_naive.cache``. T07 never writes outside its three trees.
In particular, T07 never writes inside ``<doc_id>/configs/<arm1_config_hash>/``
— that subtree is T03-only per §B.4 of the per-PDF plan.

Invalidation
------------
Eval manifest mismatches on (``tokenizer_name``, ``bsard_db_fingerprint``,
``t03_manifest_resolved_path``) force a rebuild. ``arm1_config_hash`` is
already encoded in the directory path, so a chunking/model change moves to a
new sibling folder rather than invalidating in-place.

Schema versioning
-----------------
All JSON manifests carry ``schema_version: 1``. Readers reject unknown
versions (loud failure, not silent fallback).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

# T07 reads T03's cache via this public API only.
from arm1_naive.cache import (
    PdfCache as _T03PdfCache,  # noqa: F401 — re-exported for callers
    compute_bsard_db_fingerprint as compute_bsard_db_fingerprint,  # noqa: PLC0414
)

from .weight_precomputer import (
    ChunkArticleWeight,
    load_weights_csv,
    load_weights_pickle,
    save_weights_csv,
    save_weights_pickle,
)

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
_HASH_LEN = 12

_MANIFEST_FILE = "manifest.json"
_WEIGHTS_CSV_FILE = "chunk_bsard_weights.csv"
_WEIGHTS_PKL_FILE = "chunk_bsard_weights.pkl"
_RANGES_FILE = "article_token_ranges.json"


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


def compute_qset_hash(question_ids: Iterable[int]) -> str:
    """First 12 hex of sha256 of comma-joined sorted unique question ids.

    Stable under reordering or duplication so the same question subset always
    maps to the same projection file.
    """
    sorted_ids = sorted({int(q) for q in question_ids})
    blob = ",".join(str(q) for q in sorted_ids).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:_HASH_LEN]


# ── Cache root ───────────────────────────────────────────────────────────────


class T07CacheRoot:
    """Wraps the shared cache root (junction → ``pdf_cache\\`` — same target as T03's)."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def chunk_weights(
        self, doc_id: str, arm1_config_hash: str
    ) -> "ChunkWeightsCache":
        return ChunkWeightsCache(self, doc_id, arm1_config_hash)

    def question_projections(self, doc_id: str) -> "PdfQuestionProjectionCache":
        return PdfQuestionProjectionCache(self, doc_id)


# ── Chunk × bsard-article weight cache ───────────────────────────────────────


class ChunkWeightsCache:
    """Per (PDF, arm1_config) chunk-bsard fractional-overlap weight cache.

    Lives at ``<root>/<doc_id>/eval/<arm1_config_hash>/``. T07-owned subtree
    sibling to T03's ``configs/`` so the two projects never write into each
    other's territory.
    """

    def __init__(
        self,
        root: T07CacheRoot,
        doc_id: str,
        arm1_config_hash: str,
    ) -> None:
        self.root = root
        self.doc_id = doc_id
        self.arm1_config_hash = arm1_config_hash
        self.path = root.path / doc_id / "eval" / arm1_config_hash

    # ── Path helpers ────────────────────────────────────────────────
    def manifest_path(self) -> Path:
        return self.path / _MANIFEST_FILE

    def weights_csv_path(self) -> Path:
        return self.path / _WEIGHTS_CSV_FILE

    def weights_pkl_path(self) -> Path:
        return self.path / _WEIGHTS_PKL_FILE

    def ranges_path(self) -> Path:
        return self.path / _RANGES_FILE

    # ── State checks ────────────────────────────────────────────────
    def exists(self) -> bool:
        return self.manifest_path().exists()

    def read_manifest(self) -> dict:
        payload = _read_json(self.manifest_path())
        _check_schema(payload, self.manifest_path())
        return payload

    def validate(
        self,
        *,
        tokenizer_name: str,
        bsard_db_fingerprint: str,
        t03_manifest_resolved_path: str,
    ) -> bool:
        """Return True iff manifest matches the inputs and required artifacts exist."""
        if not self.exists():
            return False
        try:
            m = self.read_manifest()
        except (json.JSONDecodeError, ValueError):
            return False

        if m.get("doc_id") != self.doc_id:
            return False
        if m.get("arm1_config_hash") != self.arm1_config_hash:
            return False
        if m.get("tokenizer_name") != tokenizer_name:
            return False
        if m.get("bsard_db_fingerprint") != bsard_db_fingerprint:
            return False
        if m.get("t03_manifest_resolved_path") != t03_manifest_resolved_path:
            return False

        if not self.weights_csv_path().exists():
            return False
        # T03's manifest must still be where we stored its resolved path.
        if not Path(t03_manifest_resolved_path).exists():
            return False
        return True

    # ── Load / save ─────────────────────────────────────────────────
    def load_weights(self) -> list[ChunkArticleWeight]:
        """Prefer the pickle (fast load); fall back to CSV when absent or stale."""
        pkl = self.weights_pkl_path()
        csv_path = self.weights_csv_path()
        if pkl.exists():
            try:
                return load_weights_pickle(pkl)
            except Exception as exc:  # noqa: BLE001 — pickle errors vary by Python version
                logger.warning(
                    "Failed to load weights pickle %s (%s); falling back to CSV",
                    pkl, exc,
                )
        if not csv_path.exists():
            raise FileNotFoundError(f"Neither {pkl} nor {csv_path} exists")
        return load_weights_csv(csv_path)

    def save(
        self,
        *,
        weights: list[ChunkArticleWeight],
        ranges: dict[int, tuple[int, int]],
        tokenizer_name: str,
        bsard_db_fingerprint: str,
        t03_manifest_resolved_path: str,
        n_chunks: int,
        n_bsard_in_pdf: int,
        coverage_stats: dict[str, int],
    ) -> None:
        """Persist weights (CSV + PKL), ranges, and the manifest atomically.

        ``coverage_stats`` should be ``{ok, warning, missing}`` from
        ``compute_weight_coverage``; surfaced verbatim into the manifest so
        readers can flag suspicious caches without rerunning the validator.
        """
        self.path.mkdir(parents=True, exist_ok=True)
        save_weights_csv(weights, self.weights_csv_path())
        save_weights_pickle(weights, self.weights_pkl_path())

        _write_json(
            self.ranges_path(),
            {
                "schema_version": SCHEMA_VERSION,
                "doc_id": self.doc_id,
                "arm1_config_hash": self.arm1_config_hash,
                "ranges": {
                    str(int(b)): [int(s), int(e)] for b, (s, e) in ranges.items()
                },
            },
        )

        _write_json(
            self.manifest_path(),
            {
                "schema_version": SCHEMA_VERSION,
                "doc_id": self.doc_id,
                "arm1_config_hash": self.arm1_config_hash,
                "tokenizer_name": tokenizer_name,
                "bsard_db_fingerprint": bsard_db_fingerprint,
                "t03_manifest_resolved_path": t03_manifest_resolved_path,
                "n_chunks": int(n_chunks),
                "n_bsard_in_pdf": int(n_bsard_in_pdf),
                "n_weight_rows": int(len(weights)),
                "weight_coverage": dict(coverage_stats),
                "computed_at": _now_utc(),
            },
        )

    def load_ranges(self) -> dict[int, tuple[int, int]]:
        """Read ``article_token_ranges.json`` (debug-only consumer)."""
        payload = _read_json(self.ranges_path())
        _check_schema(payload, self.ranges_path())
        return {
            int(k): (int(v[0]), int(v[1]))
            for k, v in payload.get("ranges", {}).items()
        }


# ── Question-projection cache ────────────────────────────────────────────────


class PdfQuestionProjectionCache:
    """Per-PDF ground-truth projections, keyed by ``qset_hash``.

    Lives at ``<root>/<doc_id>/question_projections/<qset_hash>.json``. Only
    depends on (article_spans, ground truth, BSARD DB) — not on chunking
    parameters, so it sits at the PDF folder level rather than inside any
    config sub-folder.
    """

    def __init__(self, root: T07CacheRoot, doc_id: str) -> None:
        self.root = root
        self.doc_id = doc_id
        self.path = root.path / doc_id / "question_projections"

    def path_for(self, qset_hash: str) -> Path:
        return self.path / f"{qset_hash}.json"

    def exists(self, qset_hash: str) -> bool:
        return self.path_for(qset_hash).exists()

    def load(self, qset_hash: str) -> "list":
        """Return the list of ProjectionRow rows."""
        # Local import to avoid cycle with projection.py at module load time.
        from .projection import ProjectionRow

        payload = _read_json(self.path_for(qset_hash))
        _check_schema(payload, self.path_for(qset_hash))
        return [
            ProjectionRow(
                question_id=int(r["question_id"]),
                gt_bsard_ids=[int(b) for b in r.get("gt_bsard_ids", [])],
                gt_in_pdf=[int(b) for b in r.get("gt_in_pdf", [])],
                gt_missing_from_pdf=[int(b) for b in r.get("gt_missing_from_pdf", [])],
                recall_ceiling=float(r.get("recall_ceiling", 0.0)),
            )
            for r in payload.get("rows", [])
        ]

    def save(
        self,
        qset_hash: str,
        rows: "list",
        *,
        bsard_db_fingerprint: str,
    ) -> None:
        _write_json(
            self.path_for(qset_hash),
            {
                "schema_version": SCHEMA_VERSION,
                "qset_hash": qset_hash,
                "doc_id": self.doc_id,
                "bsard_db_fingerprint": bsard_db_fingerprint,
                "n_questions": len(rows),
                "computed_at": _now_utc(),
                "rows": [dataclasses.asdict(r) for r in rows],
            },
        )


__all__ = [
    "compute_bsard_db_fingerprint",
    "compute_qset_hash",
    "ChunkWeightsCache",
    "PdfQuestionProjectionCache",
    "T07CacheRoot",
    "SCHEMA_VERSION",
]
