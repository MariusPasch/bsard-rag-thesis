"""Load cached FAISS + BM25 index bundles for each arm.

Viewer-side index loading is intentionally thin: the arm modules already
implement load_arm1_index / load_arm2_*_index.  This module wraps those
calls with a unified interface and path-discovery logic so that app.py
doesn't need to know the on-disk layout of each arm.

Index directory layout (from arm indexers):
    data/indices/arm1/<doc_stem>/faiss.index + faiss_meta.json + bm25.pkl
    data/indices/arm2_metadata/<doc_stem>/...
    data/indices/arm2_pageindex/<doc_stem>/...
"""

from __future__ import annotations

from pathlib import Path


ARM_INDEX_DIRS: dict[str, str] = {
    "arm1":         "arm1",
    "2A-full":      "arm2_metadata",
    "2A-filtered":  "arm2_metadata",
    "2B":           "arm2_pageindex",
}


def resolve_index_dir(arm: str, base_index_dir: Path) -> Path:
    """Return the arm-specific sub-directory under base_index_dir."""
    sub = ARM_INDEX_DIRS.get(arm, arm)
    return base_index_dir / sub


def load_arm1_index_cached(doc_stem: str, index_dir: Path):
    """Load FAISS + BM25 for Arm 1 from <index_dir>/<doc_stem>/."""
    try:
        from arm1_naive.indexer import load_arm1_index
        return load_arm1_index(doc_stem, index_dir)
    except ImportError as exc:
        raise RuntimeError("arm1_naive package not installed") from exc
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Arm 1 index not found at {index_dir / doc_stem}. "
            "Run the arm1 pipeline first to build the index."
        ) from exc


def index_exists(arm: str, doc_stem: str, base_index_dir: Path) -> bool:
    """Return True if a persisted index exists for this arm + document."""
    arm_dir = resolve_index_dir(arm, base_index_dir)
    doc_dir = arm_dir / doc_stem

    if arm == "arm1":
        return (doc_dir / "faiss.index").exists() and (doc_dir / "bm25.pkl").exists()

    # For arm2 variants, just check the directory exists and is non-empty
    return doc_dir.exists() and any(doc_dir.iterdir())


def list_indexed_documents(arm: str, base_index_dir: Path) -> list[str]:
    """Return document stems that have a cached index for this arm."""
    arm_dir = resolve_index_dir(arm, base_index_dir)
    if not arm_dir.exists():
        return []
    return [d.name for d in arm_dir.iterdir() if d.is_dir()]
