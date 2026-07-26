"""Path resolution helpers for the RQ2 pipeline.

Resolves the repo root (`RQ2_ROOT`), each sub-project's directory, and
each sub-project's per-PDF data directory. Reads
`RQ2_T00_ORCHESTRATOR/data/selected_pdfs.json` to expose the curated
6-PDF set. Performs no other I/O.

Project codes used by callers: "T00", "T01", ..., "T08". The full
folder name (e.g. "RQ2_T03_ARM1_NAIVE") is resolved at module load via
the glob `RQ2_{code}_*` against `RQ2_ROOT`. Ambiguous matches raise
`AmbiguousProjectError`.

Per-project `data/` directories are links (directory junctions on
Windows, symlinks on POSIX) into the shared RQ2 data root — see
`scripts/setup/link_data.py`. This module returns the in-repo path; the
OS resolves the link transparently when callers read or write through it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

__all__ = [
    "RQ2_ROOT",
    "PROJECT_CODES",
    "AmbiguousProjectError",
    "UnknownProjectError",
    "project_dir",
    "project_data_dir",
    "selected_pdfs",
    "selected_pdf_doc_ids",
]


def _resolve_rq2_root() -> Path:
    # paths.py → orchestrator/ → src/ → RQ2_T00_ORCHESTRATOR/ → RQ2_ROOT/
    return Path(__file__).resolve().parents[3]


RQ2_ROOT: Final[Path] = _resolve_rq2_root()

PROJECT_CODES: Final[tuple[str, ...]] = (
    "T00", "T01", "T02", "T03", "T04", "T05", "T06", "T07", "T08",
)


class UnknownProjectError(LookupError):
    """Raised when a project code matches no folder under `RQ2_ROOT`."""


class AmbiguousProjectError(LookupError):
    """Raised when a project code matches more than one folder under `RQ2_ROOT`."""


def _scan_project_dirs() -> dict[str, Path]:
    resolved: dict[str, Path] = {}
    for code in PROJECT_CODES:
        matches = sorted(p for p in RQ2_ROOT.glob(f"RQ2_{code}_*") if p.is_dir())
        if len(matches) > 1:
            raise AmbiguousProjectError(
                f"project code {code!r} matches multiple folders under {RQ2_ROOT}: "
                + ", ".join(p.name for p in matches)
            )
        if matches:
            resolved[code] = matches[0]
    return resolved


_PROJECT_DIRS: Final[dict[str, Path]] = _scan_project_dirs()


def project_dir(code: str) -> Path:
    """Return the absolute path of the sub-project folder for `code` (e.g. "T03").

    Raises `UnknownProjectError` if the code does not match any folder.
    """
    if code not in PROJECT_CODES:
        raise UnknownProjectError(
            f"unknown project code {code!r}; expected one of {PROJECT_CODES}"
        )
    try:
        return _PROJECT_DIRS[code]
    except KeyError as exc:
        raise UnknownProjectError(
            f"no folder matching RQ2_{code}_* under {RQ2_ROOT}"
        ) from exc


def project_data_dir(code: str, doc_id: str | None = None) -> Path:
    """Return the path to a sub-project's `data/` directory, or to `data/<doc_id>/`
    when `doc_id` is given.

    Performs no existence check — the status command treats missing paths as
    legitimate "missing" findings rather than errors.
    """
    base = project_dir(code) / "data"
    return base if doc_id is None else base / doc_id


_SELECTED_PDFS_PATH: Final[Path] = (
    project_dir("T00") / "data" / "selected_pdfs.json"
)


def selected_pdfs() -> list[dict]:
    """Return the curated PDF set from `RQ2_T00_ORCHESTRATOR/data/selected_pdfs.json`.

    Each entry includes `doc_id`, `n_questions`, `pct_found`, `n_pages`, etc.
    See `selected_pdfs.json` for the full schema.
    """
    with _SELECTED_PDFS_PATH.open(encoding="utf-8") as f:
        payload = json.load(f)
    pdfs = payload.get("pdfs")
    if not isinstance(pdfs, list):
        raise ValueError(
            f"{_SELECTED_PDFS_PATH} is missing a top-level 'pdfs' list"
        )
    return pdfs


def selected_pdf_doc_ids() -> list[str]:
    """Return just the `doc_id`s from `selected_pdfs()`, in registry order."""
    return [entry["doc_id"] for entry in selected_pdfs()]
