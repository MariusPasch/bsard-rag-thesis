"""Question→PDF ground-truth projections.

For one PDF and one question subset (``qset``), produce the GT slice that is
actually retrievable from this PDF: the intersection of each question's
``relevant_bsard_ids`` with the bsard ids identified inside the PDF
(``PdfCache.pdf_bsard_ids()``). The recall ceiling is ``|gt ∩ pdf| / |gt|``
and represents the upper bound of any retriever that only sees this PDF.

Results are arm-independent — they depend on (article_spans, GT, BSARD DB)
only — so they live at PDF-folder level rather than inside any config
sub-folder (CACHE_DESIGN.md §4).
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from typing import Iterable

from arm1_naive.cache import PdfCache as _T03PdfCache

logger = logging.getLogger(__name__)


@dataclass
class ProjectionRow:
    """One question's projection onto a PDF.

    ``recall_ceiling`` is ``|gt_in_pdf| / |gt_bsard_ids|`` — the maximum
    fractional recall any retriever restricted to this PDF can achieve, ``0.0``
    when the question has no GT.
    """

    question_id: int
    gt_bsard_ids: list[int] = field(default_factory=list)
    gt_in_pdf: list[int] = field(default_factory=list)
    gt_missing_from_pdf: list[int] = field(default_factory=list)
    recall_ceiling: float = 0.0


def build_projection(
    qset_question_ids: Iterable[int],
    doc_id: str,
    t03_pdf_cache: _T03PdfCache,
    db_conn: sqlite3.Connection,
) -> list[ProjectionRow]:
    """Compute one ``ProjectionRow`` per qset question.

    Reads ``pdf_bsard_ids`` from T03's per-PDF cache (raises if article spans
    aren't yet cached for this doc — fix by running T03's cache first). Pulls
    ``relevant_bsard_ids`` directly from ``questions.relevant_bsard_ids``,
    which is bsard-id-keyed per the migration.

    Questions absent from the DB are silently skipped (with an INFO log).
    """
    if doc_id != t03_pdf_cache.doc_id:
        raise ValueError(
            f"doc_id mismatch: argument {doc_id!r} vs t03_pdf_cache.doc_id "
            f"{t03_pdf_cache.doc_id!r}"
        )

    pdf_bsard_ids = t03_pdf_cache.pdf_bsard_ids()

    rows: list[ProjectionRow] = []
    for qid in qset_question_ids:
        result = db_conn.execute(
            "SELECT relevant_bsard_ids FROM questions WHERE question_id = ?",
            (int(qid),),
        ).fetchone()
        if result is None:
            logger.info("question_id=%s not in DB — skipping", qid)
            continue

        raw = result[0] if not isinstance(result, sqlite3.Row) else result["relevant_bsard_ids"]
        gt_bsard_ids = sorted({int(b) for b in json.loads(raw or "[]")})

        gt_set = set(gt_bsard_ids)
        gt_in = sorted(gt_set & pdf_bsard_ids)
        gt_missing = sorted(gt_set - pdf_bsard_ids)
        recall_ceiling = (len(gt_in) / len(gt_bsard_ids)) if gt_bsard_ids else 0.0

        rows.append(
            ProjectionRow(
                question_id=int(qid),
                gt_bsard_ids=gt_bsard_ids,
                gt_in_pdf=gt_in,
                gt_missing_from_pdf=gt_missing,
                recall_ceiling=float(recall_ceiling),
            )
        )

    return rows


__all__ = ["ProjectionRow", "build_projection"]
