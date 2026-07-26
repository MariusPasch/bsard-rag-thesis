"""Loader for cached T05 (Arm 2B PageIndex) retrieval results.

T05 stores artefacts under:

    RQ2_T05_ARM2_PAGEINDEX/data/<doc_stem>/results/q<qid>.json
    RQ2_T05_ARM2_PAGEINDEX/data/trees/doc_<id>.json

Per-query JSONs are produced by ``arm2_pageindex.retriever.state_to_retrieval_result``
and serialise the full ``RetrievalResult`` (ranked_items, method, cost, trace).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class T05Coverage:
    """What's available for a given PDF in the T05 cache."""

    doc_dir: Path                  # <t05_data_root>/<doc_stem>
    results_dir: Path              # <doc_dir>/results
    tree_path: Path | None         # matching tree under <t05_data_root>/trees/
    query_files: dict[str, Path]   # {query_id: q<qid>.json}

    @property
    def has_results(self) -> bool:
        return bool(self.query_files)


def doc_dir_for_pdf(t05_data_root: str | Path, pdf_filename: str) -> Path | None:
    """Map ``"2004_05_27_2004A27101.pdf"`` → ``<t05_data_root>/2004_05_27_2004A27101``."""
    if not pdf_filename:
        return None
    candidate = Path(t05_data_root) / Path(pdf_filename).stem
    return candidate if candidate.is_dir() else None


def find_tree_for_pdf(
    t05_data_root: str | Path,
    pdf_filename: str,
) -> Path | None:
    """Scan ``<t05_data_root>/trees/doc_*.json`` for the tree built from this PDF.

    Match is by ``manifest.pdf_filename`` so we don't have to assume the doc_id.
    """
    trees_dir = Path(t05_data_root) / "trees"
    if not trees_dir.is_dir() or not pdf_filename:
        return None
    target = Path(pdf_filename).name
    for tree_path in sorted(trees_dir.glob("doc_*.json")):
        try:
            data = json.loads(tree_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        manifest = data.get("manifest") or {}
        if manifest.get("pdf_filename") == target:
            return tree_path
    return None


def discover_t05_coverage(
    t05_data_root: str | Path,
    pdf_filename: str,
) -> T05Coverage | None:
    """Return everything T05 has cached for this PDF, or None if no doc dir exists."""
    doc_dir = doc_dir_for_pdf(t05_data_root, pdf_filename)
    if doc_dir is None:
        return None
    results_dir = doc_dir / "results"
    query_files: dict[str, Path] = {}
    if results_dir.is_dir():
        for path in sorted(results_dir.glob("q*.json")):
            qid = path.stem.lstrip("q")
            if qid:
                query_files[qid] = path
    tree_path = find_tree_for_pdf(t05_data_root, pdf_filename)
    return T05Coverage(
        doc_dir=doc_dir,
        results_dir=results_dir,
        tree_path=tree_path,
        query_files=query_files,
    )


def load_t05_result(path: str | Path) -> dict | None:
    """Read one ``q<qid>.json`` file. Returns the dict (RetrievalResult shape) or None."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def t05_dict_to_retrieval_result(d: dict):
    """Materialise a T05 result dict as a ``RetrievalResult`` instance."""
    from data_loader.models import RetrievalResult
    return RetrievalResult(
        query_id=str(d.get("query_id", "")),
        query_text=str(d.get("query_text", "")),
        ranked_items=list(d.get("ranked_items") or []),
        method=str(d.get("method", "arm2b_pageindex")),
        cost=dict(d.get("cost") or {}),
        trace=dict(d.get("trace") or {}),
    )


def load_tree(tree_path: str | Path) -> dict | None:
    """Read a T05 tree file. Returns ``{manifest, tree}`` dict or None."""
    try:
        return json.loads(Path(tree_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
