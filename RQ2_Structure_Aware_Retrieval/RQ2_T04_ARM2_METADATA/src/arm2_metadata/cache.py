"""Simple T04-side cache layout mirroring T03's CACHE_DESIGN.md.

Layout (one tree per BSARD PDF stem; AzureDI's integer ``DocumentId`` is
recorded as a manifest field but no longer keys the path — see
PLAN_per_pdf_pipeline §B.5):

    data/<doc_id>/
    ├── manifest.json                 # doc-level fingerprints (azuredi files, pdf)
    ├── configs/<config_hash>/
    │   ├── manifest.json             # per-config invariants
    │   ├── faiss.index, faiss_meta.json
    │   ├── bm25.pkl
    │   └── enrichment_stats.json
    └── results/<config_hash>/<run_label>.jsonl

The config_hash is computed via T03's ``compute_config_hash`` so the function
is shared with the rest of the pipeline. T04 puts its variant + unit choice
into the chunking_params slot.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from shared.bm25_store import BM25Store
from shared.faiss_store import FAISSStore

# Re-use T03 hashing so configs from different arms can be told apart only by
# the chunking_params blob (which includes our "variant" / "unit" keys).
from arm1_naive.cache import (
    SCHEMA_VERSION,
    compute_config_hash,
    compute_pdf_sha256,
)

logger = logging.getLogger(__name__)

_FAISS_INDEX_FILE = "faiss.index"
_BM25_FILE = "bm25.pkl"
_MANIFEST_FILE = "manifest.json"


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
            f"Unsupported schema_version={v!r} in {source} (expected {SCHEMA_VERSION})."
        )


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def azuredi_dump_fingerprint(
    azuredi_dir: str | Path,
    pdf_document_map_csv: str | Path | None = None,
) -> str:
    """SHA-256 of (sorted list of (filename, sha256-or-MISSING)) for the four
    top-level AzureDI files plus optional ``pdf_document_map.csv``.

    A change in any of the four files (or a new dump replacing them) shifts
    this fingerprint and invalidates every config_hash that pinned the old
    one. ``backups/`` is ignored.

    Review #15-A4 / #15-B6:
      * Missing files are recorded as ``"MISSING"`` (not silently skipped) so
        a partial-dump fingerprint can never collide with a full-dump
        fingerprint that happens to be missing a different file.
      * ``pdf_document_map.csv`` is optional but, when supplied, is folded
        into the fingerprint — editing the map shifts which doc-ids the
        linker can resolve and therefore which nodes carry ``bsard_ids``.
        Editing it without re-fingerprinting was producing stale indices.
    """
    azuredi_dir = Path(azuredi_dir)
    targets = [
        "MyDocuments.csv",
        "DocumentDefinitions.csv",
        "VectorDB_Documents.json",
        "VectorDBdev_Documents.json",
    ]
    parts: list[tuple[str, str]] = []
    for name in targets:
        p = azuredi_dir / name
        parts.append((name, file_sha256(p) if p.exists() else "MISSING"))
    if pdf_document_map_csv is not None:
        p = Path(pdf_document_map_csv)
        parts.append(("pdf_document_map.csv", file_sha256(p) if p.exists() else "MISSING"))
    blob = json.dumps(parts, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


# ── Doc / config caches ──────────────────────────────────────────────────────


class T04CacheRoot:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def doc(self, doc_id: str) -> "T04DocCache":
        return T04DocCache(self, doc_id)


class T04DocCache:
    def __init__(self, root: T04CacheRoot, doc_id: str) -> None:
        if not isinstance(doc_id, str) or not doc_id:
            raise TypeError(
                f"T04DocCache.doc_id must be a non-empty BSARD PDF stem (str), got {doc_id!r}. "
                f"Pass the AzureDI integer separately as azure_document_id."
            )
        self.root = root
        self.doc_id = doc_id
        self.path = root.path / self.doc_id

    def manifest_path(self) -> Path:
        return self.path / _MANIFEST_FILE

    def configs_dir(self) -> Path:
        return self.path / "configs"

    def results_dir(self, config_hash: str) -> Path:
        return self.path / "results" / config_hash

    def write_doc_manifest(
        self,
        *,
        azuredi_fingerprint: str,
        azure_document_id: int | None,
        pdf_sha256: str | None,
        pdf_filename: str | None,
        n_nodes: int,
        n_articles: int,
    ) -> None:
        _write_json(self.manifest_path(), {
            "schema_version": SCHEMA_VERSION,
            "doc_id": self.doc_id,
            "azure_document_id": azure_document_id,
            "azuredi_fingerprint": azuredi_fingerprint,
            "pdf_sha256": pdf_sha256,
            "pdf_filename": pdf_filename,
            "n_nodes": n_nodes,
            "n_articles": n_articles,
            "computed_at": _now_utc(),
        })

    def write_results_jsonl(
        self,
        config_hash: str,
        run_label: str,
        results: Iterable[Any],
    ) -> Path:
        out = self.results_dir(config_hash) / f"{run_label}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8", newline="\n") as f:
            for r in results:
                row = _to_serialisable(r)
                # Slim form: drop chunk text from each ranked item. The id +
                # cached index let downstream analysis recover text on demand,
                # so persisting it per-row inflates the JSONL ~10× for no gain.
                if isinstance(row, dict):
                    for item in row.get("ranked_items") or []:
                        if isinstance(item, dict):
                            item.pop("text", None)
                f.write(json.dumps(row, ensure_ascii=False))
                f.write("\n")
        return out


class T04ConfigCache:
    def __init__(self, doc_cache: T04DocCache, config_hash: str) -> None:
        self.doc_cache = doc_cache
        self.config_hash = config_hash
        self.path = doc_cache.configs_dir() / config_hash

    def manifest_path(self) -> Path:
        return self.path / _MANIFEST_FILE

    def faiss_dir(self) -> Path:
        return self.path

    def bm25_path(self) -> Path:
        return self.path / _BM25_FILE

    def stats_path(self) -> Path:
        return self.path / "enrichment_stats.json"

    def exists(self) -> bool:
        return self.manifest_path().exists()

    def validate(
        self,
        *,
        azuredi_fingerprint: str,
        pdf_sha256: str | None,
        tokenizer_name: str,
        embedding_model: str,
        chunking_params: dict,
    ) -> bool:
        if not self.exists():
            return False
        try:
            m = _read_json(self.manifest_path())
            _check_schema(m, self.manifest_path())
        except (json.JSONDecodeError, ValueError):
            return False
        if m.get("config_hash") != self.config_hash:
            return False
        if m.get("azuredi_fingerprint") != azuredi_fingerprint:
            return False
        if m.get("pdf_sha256") != pdf_sha256:
            return False
        if m.get("tokenizer_name") != tokenizer_name:
            return False
        if m.get("embedding_model") != embedding_model:
            return False
        if m.get("chunking") != chunking_params:
            return False
        if not self.bm25_path().exists():
            return False
        # FAISS only required when this config wasn't built sparse-only.
        if not chunking_params.get("sparse_only"):
            if not (self.faiss_dir() / _FAISS_INDEX_FILE).exists():
                return False
        return True

    def load_indices(self) -> tuple[FAISSStore, BM25Store]:
        return FAISSStore.load(str(self.faiss_dir())), BM25Store.load(str(self.bm25_path()))

    def write_manifest(
        self,
        *,
        azuredi_fingerprint: str,
        azure_document_id: int | None,
        pdf_sha256: str | None,
        tokenizer_name: str,
        embedding_model: str,
        embedding_dim: int,
        chunking_params: dict,
        n_units: int,
    ) -> None:
        _write_json(self.manifest_path(), {
            "schema_version": SCHEMA_VERSION,
            "doc_id": self.doc_cache.doc_id,
            "azure_document_id": azure_document_id,
            "config_hash": self.config_hash,
            "azuredi_fingerprint": azuredi_fingerprint,
            "pdf_sha256": pdf_sha256,
            "tokenizer_name": tokenizer_name,
            "embedding_model": embedding_model,
            "embedding_dim": embedding_dim,
            "chunking": chunking_params,
            "n_units": n_units,
            "built_at": _now_utc(),
        })

    def write_enrichment_stats(self, stats: Any) -> None:
        _write_json(self.stats_path(), _to_serialisable(stats))


def _to_serialisable(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    return obj


def default_run_label() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


__all__ = [
    "T04CacheRoot",
    "T04DocCache",
    "T04ConfigCache",
    "azuredi_dump_fingerprint",
    "compute_config_hash",
    "compute_pdf_sha256",
    "default_run_label",
    "file_sha256",
]
