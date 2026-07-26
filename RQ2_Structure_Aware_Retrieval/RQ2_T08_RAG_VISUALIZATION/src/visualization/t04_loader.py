"""Loader for cached T04 (Arm 2A metadata) retrieval results.

T04 stores artefacts under a per-document directory:

    RQ2_T04_ARM2_METADATA/data/<doc_stem>/
        configs/<config_hash>/manifest.json   # variant, unit, sparse_only, ...
        results/<config_hash>/compare_*.jsonl # one RetrievalResult dict per query

This module discovers the variants present for a given PDF and reads the
matching ``compare_*.jsonl`` so the viewer can render them through the
existing arm-agnostic ``retrieval_result_to_regions`` pipeline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass
class T04Variant:
    """A discovered T04 cache entry — one (variant, unit, sparse_only) combo."""

    config_hash: str
    variant: str            # raw | enriched | summary | full | filtered | terms
    unit: str               # article | node
    sparse_only: bool
    n_units: int
    manifest: dict
    results_path: Path

    @property
    def arm_label(self) -> str:
        """Stable label used as the value in T08's arm dropdown."""
        suffix = "-sparse" if self.sparse_only else ""
        return f"2A-{self.variant}-{self.unit}{suffix}"


def discover_t04_variants(doc_dir: str | Path) -> list[T04Variant]:
    """Return every variant under ``doc_dir`` that has both a manifest and a result file."""
    doc_dir = Path(doc_dir)
    configs_dir = doc_dir / "configs"
    results_dir = doc_dir / "results"
    if not configs_dir.is_dir() or not results_dir.is_dir():
        return []

    variants: list[T04Variant] = []
    for cfg_subdir in sorted(configs_dir.iterdir()):
        if not cfg_subdir.is_dir():
            continue
        manifest_path = cfg_subdir / "manifest.json"
        if not manifest_path.exists():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        config_hash = cfg_subdir.name
        res_subdir = results_dir / config_hash
        if not res_subdir.is_dir():
            continue
        compare_files = sorted(res_subdir.glob("compare_*.jsonl"))
        if not compare_files:
            continue

        chunking = manifest.get("chunking") or {}
        variants.append(
            T04Variant(
                config_hash=config_hash,
                variant=str(chunking.get("variant", "?")),
                unit=str(chunking.get("unit", "?")),
                sparse_only=bool(chunking.get("sparse_only", False)),
                n_units=int(manifest.get("n_units", 0)),
                manifest=manifest,
                results_path=compare_files[-1],
            )
        )
    return variants


@lru_cache(maxsize=32)
def _load_bm25_text_map(bm25_path: str) -> dict[str, str]:
    """Build ``{item_id: chunk_text}`` from the cached BM25 store at ``bm25_path``.

    Used to hydrate slim T04 JSONLs (written without per-item ``text`` to keep
    the on-disk pool affordable at top-100) so the viewer's tooltip / region
    text still has chunk content. Cached because the BM25 pickle is a few MB.
    """
    from shared.bm25_store import BM25Store

    store = BM25Store.load(bm25_path)
    out: dict[str, str] = {}
    for item_id, meta in zip(store._ids, store._metadata):
        text = (meta or {}).get("text")
        if text:
            out[str(item_id)] = text
    return out


def _sibling_bm25_path(jsonl_path: Path) -> Path | None:
    """Map ``.../results/<hash>/compare_*.jsonl`` → ``.../configs/<hash>/bm25.pkl``."""
    hash_dir = jsonl_path.parent
    if hash_dir.parent.name != "results":
        return None
    candidate = hash_dir.parent.parent / "configs" / hash_dir.name / "bm25.pkl"
    return candidate if candidate.exists() else None


def load_t04_results(
    jsonl_path: str | Path,
    *,
    hydrate_text: bool = True,
) -> dict[str, dict]:
    """Read a T04 ``compare_*.jsonl``; return ``{query_id: result_dict}``.

    Each result_dict mirrors the fields of ``data_loader.models.RetrievalResult``:
    ``query_id``, ``query_text``, ``ranked_items``, ``method``, ``cost``, ``trace``.

    When ``hydrate_text`` is True (default), each ``ranked_items[i]`` that
    lacks a ``text`` field is populated by looking the id up in the sibling
    ``configs/<hash>/bm25.pkl`` store — needed because the JSONL writer no
    longer persists chunk text per row.
    """
    path = Path(jsonl_path)
    out: dict[str, dict] = {}
    if not path.exists():
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = str(d.get("query_id", ""))
            if not qid:
                continue
            out[qid] = d

    if hydrate_text and out:
        needs_hydration = any(
            "text" not in item
            for d in out.values()
            for item in (d.get("ranked_items") or [])
        )
        if needs_hydration:
            bm25_path = _sibling_bm25_path(path)
            if bm25_path is not None:
                text_map = _load_bm25_text_map(str(bm25_path))
                for d in out.values():
                    for item in (d.get("ranked_items") or []):
                        if "text" not in item:
                            item["text"] = text_map.get(str(item.get("id", "")), "")
    return out


def t04_dict_to_retrieval_result(d: dict):
    """Materialise a T04 result dict as a ``RetrievalResult`` instance."""
    from data_loader.models import RetrievalResult
    return RetrievalResult(
        query_id=str(d.get("query_id", "")),
        query_text=str(d.get("query_text", "")),
        ranked_items=list(d.get("ranked_items") or []),
        method=str(d.get("method", "arm2_metadata")),
        cost=dict(d.get("cost") or {}),
        trace=dict(d.get("trace") or {}),
    )


def doc_dir_for_pdf(t04_data_root: str | Path, pdf_filename: str) -> Path | None:
    """Map ``"2004_05_27_2004A27101.pdf"`` → ``<t04_data_root>/2004_05_27_2004A27101``."""
    if not pdf_filename:
        return None
    candidate = Path(t04_data_root) / Path(pdf_filename).stem
    return candidate if candidate.is_dir() else None
