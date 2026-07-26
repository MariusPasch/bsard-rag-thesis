"""Load queries and ground-truth bsard_ids from the BSARD corpus SQLite database."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Query:
    query_id: str
    query_text: str
    # Canonical BSARD article ids — populated from `questions.relevant_bsard_ids`
    # when present; field name is kept for backwards compatibility with existing
    # callsites in the viewer.
    relevant_article_ids: list[int] = field(default_factory=list)
    split: str = "test"


def load_queries(
    db_path: str | Path,
    split: str | None = "test",
) -> list[Query]:
    """Load questions (and their GT article_ids) from the corpus DB.

    Tries two schemas:
      Schema A — questions table + question_articles junction table
      Schema B — questions table with inline article_ids JSON column
      Schema C — questions table only (no GT article info in DB)

    Args:
        db_path: Path to bsard_corpus.db.
        split:   "test", "dev", "train", or None for all splits.

    Returns:
        List of Query objects, empty if the DB does not exist.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        return []

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    try:
        return _load_queries(conn, split)
    finally:
        conn.close()


def _load_queries(conn: sqlite3.Connection, split: str | None) -> list[Query]:
    # Detect available tables
    tables = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }

    if "questions" not in tables:
        return []

    # Detect columns in questions table
    cols = {row[1] for row in conn.execute("PRAGMA table_info(questions)").fetchall()}

    where = ""
    params: list = []
    if split:
        if "split" in cols:
            where = " WHERE split = ?"
            params = [split]

    rows = conn.execute(
        f"SELECT * FROM questions{where}", params
    ).fetchall()

    queries: list[Query] = []
    for row in rows:
        row_dict = dict(row)
        q_id = str(row_dict.get("question_id", row_dict.get("id", "")))
        q_text = str(row_dict.get("question_text", row_dict.get("text", "")))
        q_split = str(row_dict.get("split", split or ""))

        bsard_ids: list[int] = []

        # Preferred: bsard_id-native column added by the 2026-04-26 migration.
        if "relevant_bsard_ids" in cols:
            import json
            raw = row_dict.get("relevant_bsard_ids", "")
            if raw:
                try:
                    bsard_ids = [int(x) for x in json.loads(raw)]
                except Exception:
                    try:
                        bsard_ids = [int(x.strip()) for x in str(raw).split(",") if x.strip()]
                    except Exception:
                        pass

        # Legacy fallbacks (pre-migration DBs).
        elif "question_articles" in tables:
            art_rows = conn.execute(
                "SELECT article_id FROM question_articles WHERE question_id = ?",
                (q_id,),
            ).fetchall()
            bsard_ids = [int(r[0]) for r in art_rows]
        elif "article_ids" in cols:
            import json
            raw = row_dict.get("article_ids", "")
            if raw:
                try:
                    bsard_ids = [int(x) for x in json.loads(raw)]
                except Exception:
                    try:
                        bsard_ids = [int(x.strip()) for x in str(raw).split(",") if x.strip()]
                    except Exception:
                        pass

        queries.append(
            Query(
                query_id=q_id,
                query_text=q_text,
                relevant_article_ids=bsard_ids,
                split=q_split,
            )
        )

    return queries


def queries_for_document(
    queries: list[Query],
    bundle_bsard_ids: set[int],
) -> list[Query]:
    """Filter queries whose GT bsard_ids overlap this document's bsard_id set."""
    return [
        q for q in queries
        if any(bs in bundle_bsard_ids for bs in q.relevant_article_ids)
    ]


def load_bsard_id_to_pdf(db_path: str | Path) -> dict[int, str]:
    """Return ``{bsard_id: pdf_filename}`` from the corpus DB.

    Used by the retrieval sidebar to label "all questions" entries with the
    PDF stem of their first GT article, when the user opts out of the
    "questions relevant to this PDF" filter.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        return {}
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            "SELECT bsard_id, pdf_filename FROM articles "
            "WHERE bsard_id IS NOT NULL AND pdf_filename IS NOT NULL"
        ).fetchall()
    finally:
        conn.close()
    out: dict[int, str] = {}
    for bs, pdf in rows:
        try:
            bs_int = int(bs)
        except (TypeError, ValueError):
            continue
        if bs_int not in out and pdf:
            out[bs_int] = str(pdf)
    return out
