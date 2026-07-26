"""Discover cached T03 (Arm 1 naive) indices for the live arm1 retrieval path.

T03 stores its index artefacts under:

    RQ2_T03_ARM1_NAIVE/data/<doc_stem>/
        configs/<config_hash>/
            faiss.index
            bm25.pkl
            chunks.json
            faiss_meta.json
            manifest.json

Multiple config_hashes can coexist (different embedding models / chunking
params). This module surfaces them so the viewer can pick one and pass
its path directly to ``arm1_naive.indexer.load_arm1_index(source_dir=...)``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class T03Config:
    """A discovered T03 index cache entry — one (embedding_model, chunking) combo."""

    config_hash: str
    embedding_model: str
    chunking: dict
    n_chunks: int
    built_at: str
    source_dir: Path     # contains faiss.index + bm25.pkl + chunks.json
    manifest: dict


def doc_dir_for_pdf(t03_data_root: str | Path, pdf_filename: str) -> Path | None:
    """Map ``"2004_05_27_2004A27101.pdf"`` → ``<t03_data_root>/2004_05_27_2004A27101``."""
    if not pdf_filename:
        return None
    candidate = Path(t03_data_root) / Path(pdf_filename).stem
    return candidate if candidate.is_dir() else None


def discover_t03_configs(doc_dir: str | Path) -> list[T03Config]:
    """Return every config_hash under ``doc_dir/configs/`` that has both a faiss + bm25 index."""
    doc_dir = Path(doc_dir)
    configs_dir = doc_dir / "configs"
    if not configs_dir.is_dir():
        return []

    configs: list[T03Config] = []
    for cfg_subdir in sorted(configs_dir.iterdir()):
        if not cfg_subdir.is_dir():
            continue
        if not (cfg_subdir / "faiss.index").exists():
            continue
        if not (cfg_subdir / "bm25.pkl").exists():
            continue
        manifest_path = cfg_subdir / "manifest.json"
        if not manifest_path.exists():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        configs.append(
            T03Config(
                config_hash=cfg_subdir.name,
                embedding_model=str(manifest.get("embedding_model", "?")),
                chunking=dict(manifest.get("chunking") or {}),
                n_chunks=int(manifest.get("n_chunks", 0)),
                built_at=str(manifest.get("built_at", "")),
                source_dir=cfg_subdir,
                manifest=manifest,
            )
        )
    return configs


def pick_default_config(configs: list[T03Config]) -> T03Config | None:
    """Pick the most recent T03 config by ``built_at``; fall back to first."""
    if not configs:
        return None
    return max(configs, key=lambda c: c.built_at or "")
