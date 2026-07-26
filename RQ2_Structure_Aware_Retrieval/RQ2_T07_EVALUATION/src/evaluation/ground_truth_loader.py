"""Load ground truth from disk into the evaluator's expected shape.

Honoured config keys (under ``config["data"]``):
    ground_truth_file: path to a single curated GT JSON (output of
                       ``evaluation.question_subsets build``). When set, this
                       wins over ``ground_truth_dir``.
    ground_truth_dir:  path to a directory of GT JSONs (e.g. one per BSARD
                       split). All top-level ``*.json`` files in the directory
                       are union-merged by query_id. ``schema.json`` and any
                       ``_*.json`` are skipped as metadata.

All GT files share the shape ``{query_id_str: [bsard_id_int, ...]}``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_SKIP_FILENAMES = {"schema.json"}


def _data_section(config: dict) -> dict:
    return config.get("data", {}) if isinstance(config, dict) else {}


def _resolve_file(config: dict) -> Path | None:
    p = _data_section(config).get("ground_truth_file")
    return Path(p) if p else None


def _resolve_dir(config: dict) -> Path | None:
    p = _data_section(config).get("ground_truth_dir")
    return Path(p) if p else None


def _is_gt_file(p: Path) -> bool:
    return p.is_file() and p.name not in _SKIP_FILENAMES and not p.name.startswith("_")


def ground_truth_exists(config: dict) -> bool:
    """Return True iff `load_ground_truth(config)` would succeed."""
    f = _resolve_file(config)
    if f is not None:
        return f.is_file()
    d = _resolve_dir(config)
    if d is not None and d.is_dir():
        return any(_is_gt_file(p) for p in d.glob("*.json"))
    return False


def load_ground_truth(config: dict) -> dict[str, list[int]]:
    """Load ground truth honouring `ground_truth_file` (preferred) then `ground_truth_dir`.

    Returns ``{query_id_str: [bsard_id_int, ...]}`` with bsard_ids deduped and
    sorted per query.

    Raises:
        FileNotFoundError: when the chosen source is missing or empty.
    """
    f = _resolve_file(config)
    if f is not None:
        if not f.is_file():
            raise FileNotFoundError(f"data.ground_truth_file not found: {f}")
        logger.info("Loading ground truth from file: %s", f)
        return _normalise(json.loads(f.read_text(encoding="utf-8")))

    d = _resolve_dir(config)
    if d is None:
        raise FileNotFoundError(
            "Neither data.ground_truth_file nor data.ground_truth_dir is set."
        )
    if not d.is_dir():
        raise FileNotFoundError(f"data.ground_truth_dir not found: {d}")

    files = sorted(p for p in d.glob("*.json") if _is_gt_file(p))
    if not files:
        raise FileNotFoundError(f"No GT *.json files found in {d}")

    merged: dict[str, set[int]] = {}
    for path in files:
        for qid, bids in _normalise(json.loads(path.read_text(encoding="utf-8"))).items():
            merged.setdefault(qid, set()).update(bids)
    logger.info("Loaded ground truth from %d file(s) in %s", len(files), d)
    return {qid: sorted(bids) for qid, bids in merged.items()}


def _normalise(raw: dict) -> dict[str, list[int]]:
    return {str(qid): sorted({int(b) for b in bids}) for qid, bids in raw.items()}
