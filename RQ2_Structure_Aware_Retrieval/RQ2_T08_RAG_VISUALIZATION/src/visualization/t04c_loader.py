"""Loader for exported Arm2C (agentic ReAct tree-navigation) retrieval results.

Arm2C results are normalised out of ``experiments/arm2c/`` by
``export_for_t08.py`` into a per-doc layout under T04's ``data/`` junction:

    RQ2_T04_ARM2_METADATA/data/_arm2c/<doc_stem>/results/q<qid>.json
    RQ2_T04_ARM2_METADATA/data/_arm2c/<doc_stem>/deep_tree.json

Each ``q<qid>.json`` is already RetrievalResult-shaped (ranked_items, method,
cost, trace) so this loader mirrors ``t05_loader`` almost exactly. The
``deep_tree.json`` is the depth-7 navigation tree rendered by the recursive
explorer in ``ui/tree_explorer.render_deep_tree_explorer``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Arm2cCoverage:
    """What's available for a given PDF in the exported Arm2C cache."""

    doc_dir: Path                  # <arm2c_data_root>/<doc_stem>
    results_dir: Path              # <doc_dir>/results
    tree_path: Path | None         # <doc_dir>/deep_tree.json
    query_files: dict[str, Path]   # {query_id: q<qid>.json}

    @property
    def has_results(self) -> bool:
        return bool(self.query_files)


def doc_dir_for_pdf(arm2c_data_root: str | Path, pdf_filename: str) -> Path | None:
    """Map ``"1804_03_21_1804032150.pdf"`` → ``<arm2c_data_root>/1804_03_21_1804032150``."""
    if not pdf_filename:
        return None
    candidate = Path(arm2c_data_root) / Path(pdf_filename).stem
    return candidate if candidate.is_dir() else None


def discover_arm2c_coverage(
    arm2c_data_root: str | Path,
    pdf_filename: str,
) -> Arm2cCoverage | None:
    """Return everything Arm2C has exported for this PDF, or None if no doc dir."""
    doc_dir = doc_dir_for_pdf(arm2c_data_root, pdf_filename)
    if doc_dir is None:
        return None
    results_dir = doc_dir / "results"
    query_files: dict[str, Path] = {}
    if results_dir.is_dir():
        for path in sorted(results_dir.glob("q*.json")):
            qid = path.stem.lstrip("q")
            if qid:
                query_files[qid] = path
    tree_path = doc_dir / "deep_tree.json"
    return Arm2cCoverage(
        doc_dir=doc_dir,
        results_dir=results_dir,
        tree_path=tree_path if tree_path.exists() else None,
        query_files=query_files,
    )


def load_arm2c_result(path: str | Path) -> dict | None:
    """Read one exported ``q<qid>.json``. Returns the dict (RetrievalResult shape)."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def arm2c_dict_to_retrieval_result(d: dict):
    """Materialise an Arm2C result dict as a ``RetrievalResult`` instance."""
    from data_loader.models import RetrievalResult
    return RetrievalResult(
        query_id=str(d.get("query_id", "")),
        query_text=str(d.get("query_text", "")),
        ranked_items=list(d.get("ranked_items") or []),
        method=str(d.get("method", "2C-enriched-rerank")),
        cost=dict(d.get("cost") or {}),
        trace=dict(d.get("trace") or {}),
    )


def load_deep_tree(tree_path: str | Path) -> dict | None:
    """Read an Arm2C deep tree file. Returns ``{manifest, tree}`` dict or None."""
    try:
        return json.loads(Path(tree_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
