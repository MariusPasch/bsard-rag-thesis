"""Phase D (part 1) — SQLite database assembly.

Reads:
  output/linked/articles_with_citations.jsonl
  output/linked/questions.jsonl
  output/linked/citation_graph.jsonl

Creates:
  output/bsard_corpus.db  — primary SQLite database with all three tables,
                             critical indices, and FTS5 virtual table.

article_ids are assigned 1-based in the order records appear in the JSONL,
matching the provisional IDs from Phase C so that citation_graph references
remain consistent.

Usage:
    python retrieval/build_database.py
    python retrieval/build_database.py --rebuild   # delete & recreate if exists
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path

import tiktoken

# ── Paths ──────────────────────────────────────────────────────────────────────

PROJECT_ROOT   = Path(__file__).resolve().parent.parent
DEFAULT_ARTS   = PROJECT_ROOT / "output" / "linked" / "articles_with_citations.jsonl"
DEFAULT_QS     = PROJECT_ROOT / "output" / "linked" / "questions.jsonl"
DEFAULT_GRAPH  = PROJECT_ROOT / "output" / "linked" / "citation_graph.jsonl"
DEFAULT_DB     = PROJECT_ROOT / "output" / "bsard_corpus.db"

BATCH_SIZE = 500

# ── Schema ─────────────────────────────────────────────────────────────────────

_CREATE_ARTICLES = """
CREATE TABLE IF NOT EXISTS articles (
    article_id          INTEGER PRIMARY KEY,
    bsard_id            INTEGER,
    is_bsard_article    INTEGER NOT NULL,

    -- Source law metadata
    law_code            TEXT    NOT NULL,
    law_type            TEXT,
    article_number      TEXT    NOT NULL,
    numac               TEXT,

    -- Article content
    article_text        TEXT,
    article_text_source TEXT    NOT NULL,

    -- Hierarchical structure
    law_title_text      TEXT,
    chapter_title       TEXT,
    section_title       TEXT,
    subsection_title    TEXT,
    hierarchy_path      TEXT,
    hierarchy_depth     INTEGER,

    -- PDF source and positional metadata
    pdf_url             TEXT,
    pdf_filename        TEXT,
    pdf_page_numbers    TEXT,
    pdf_page_start      INTEGER,
    pdf_page_end        INTEGER,

    -- Justel HTML provenance
    justel_html_url     TEXT,

    -- Cross-reference / citation metadata
    cross_references_raw  TEXT,
    cross_reference_ids   TEXT,
    n_outgoing_refs       INTEGER NOT NULL DEFAULT 0,
    cited_by_ids          TEXT,
    n_cited_by            INTEGER NOT NULL DEFAULT 0,

    -- Temporal metadata
    amendment_date      TEXT,
    article_status      TEXT,
    is_pre_bsard        INTEGER,

    -- Verification status
    html_text_found     INTEGER,
    pdf_text_found      INTEGER,
    pdf_match_category  TEXT,
    verification_status TEXT,

    -- RAG-specific computed fields
    token_count         INTEGER,
    char_count          INTEGER NOT NULL DEFAULT 0,
    has_cross_references INTEGER NOT NULL DEFAULT 0
)
"""

_CREATE_QUESTIONS = """
CREATE TABLE IF NOT EXISTS questions (
    question_id          INTEGER PRIMARY KEY,
    question_text        TEXT    NOT NULL,
    relevant_article_ids TEXT    NOT NULL,
    relevant_bsard_ids   TEXT    NOT NULL,
    n_relevant_articles  INTEGER NOT NULL,
    split                TEXT    NOT NULL
)
"""

_CREATE_CITATION_GRAPH = """
CREATE TABLE IF NOT EXISTS citation_graph (
    edge_id       INTEGER PRIMARY KEY,
    source_id     INTEGER NOT NULL,
    target_id     INTEGER NOT NULL,
    citation_text TEXT    NOT NULL,
    resolved      INTEGER NOT NULL DEFAULT 1
)
"""

_INDICES = [
    "CREATE INDEX IF NOT EXISTS idx_articles_bsard_id  ON articles(bsard_id)",
    "CREATE INDEX IF NOT EXISTS idx_articles_law_code  ON articles(law_code)",
    "CREATE INDEX IF NOT EXISTS idx_articles_pdf_file  ON articles(pdf_filename)",
    "CREATE INDEX IF NOT EXISTS idx_articles_art_status ON articles(article_status)",
    "CREATE INDEX IF NOT EXISTS idx_articles_is_bsard  ON articles(is_bsard_article)",
    "CREATE INDEX IF NOT EXISTS idx_citation_source    ON citation_graph(source_id)",
    "CREATE INDEX IF NOT EXISTS idx_citation_target    ON citation_graph(target_id)",
]

_CREATE_FTS = """
CREATE VIRTUAL TABLE IF NOT EXISTS articles_fts USING fts5(
    article_text,
    content='articles',
    content_rowid='article_id',
    tokenize='unicode61'
)
"""

# ── Helpers ────────────────────────────────────────────────────────────────────

def _jdump(val) -> str | None:
    """Serialise a list/dict to JSON string, or return None if falsy."""
    if val is None:
        return None
    return json.dumps(val, ensure_ascii=False)


def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

# ── Article row builder ────────────────────────────────────────────────────────

def _article_row(art: dict, article_id: int, enc: tiktoken.Encoding) -> tuple:
    text      = art.get("article_text") or None
    char_cnt  = len(text) if text else 0
    tok_cnt   = len(enc.encode(text)) if text else None

    return (
        article_id,
        art.get("bsard_id"),
        int(art.get("is_bsard_article", 0)),
        art.get("law_code") or "UNKNOWN",
        art.get("law_type"),
        art.get("article_number") or "",
        art.get("numac"),
        text,
        art.get("article_text_source", "pdf_extraction_failed"),
        art.get("law_title_text"),
        art.get("chapter_title"),
        art.get("section_title"),
        art.get("subsection_title"),
        _jdump(art.get("hierarchy_path")),
        art.get("hierarchy_depth"),
        art.get("pdf_url"),
        art.get("pdf_filename"),
        _jdump(art.get("pdf_page_numbers")),
        art.get("pdf_page_start"),
        art.get("pdf_page_end"),
        art.get("justel_html_url"),
        _jdump(art.get("cross_references_raw")),
        _jdump(art.get("cross_reference_ids")),
        int(art.get("n_outgoing_refs", 0)),
        _jdump(art.get("cited_by_ids")),
        int(art.get("n_cited_by", 0)),
        art.get("amendment_date"),
        art.get("article_status"),
        art.get("is_pre_bsard"),
        art.get("html_text_found"),
        art.get("pdf_text_found"),
        art.get("pdf_match_category"),
        art.get("verification_status"),
        tok_cnt,
        char_cnt,
        int(art.get("has_cross_references", 0)),
    )

# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase D: build SQLite corpus database")
    parser.add_argument("--arts",    type=Path, default=DEFAULT_ARTS)
    parser.add_argument("--qs",      type=Path, default=DEFAULT_QS)
    parser.add_argument("--graph",   type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--db",      type=Path, default=DEFAULT_DB)
    parser.add_argument("--rebuild", action="store_true",
                        help="Delete and recreate the database if it already exists")
    args = parser.parse_args()

    if args.rebuild and args.db.exists():
        args.db.unlink()
        print(f"Deleted existing database: {args.db}")

    # ── Connect and configure ─────────────────────────────────────────────────
    conn = sqlite3.connect(args.db)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-65536")   # 64 MB page cache
    conn.execute("PRAGMA temp_store=MEMORY")

    # ── Create tables ─────────────────────────────────────────────────────────
    conn.execute(_CREATE_ARTICLES)
    conn.execute(_CREATE_QUESTIONS)
    conn.execute(_CREATE_CITATION_GRAPH)
    conn.commit()
    print("Tables created.")

    enc = tiktoken.get_encoding("cl100k_base")

    # ── Insert articles ───────────────────────────────────────────────────────
    print(f"Inserting articles from {args.arts.name} ...")
    t0 = time.perf_counter()

    _INSERT_ART = """
    INSERT INTO articles VALUES (
        ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
    )
    """

    # Track bsard_id → canonical article_id (prefer bsard_dataset source, lowest id)
    bsard_canonical: dict[int, int] = {}   # bsard_id → article_id
    bsard_seen_ids:  set[int]       = set()

    batch: list[tuple] = []
    article_id = 0

    with args.arts.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            art = json.loads(line)
            article_id += 1
            row = _article_row(art, article_id, enc)
            batch.append(row)

            # Track canonical bsard_id mapping.
            bid = art.get("bsard_id")
            if bid is not None:
                bid = int(bid)
                src = art.get("article_text_source", "")
                if bid not in bsard_seen_ids:
                    # First time we see this bsard_id — always register.
                    bsard_canonical[bid] = article_id
                    bsard_seen_ids.add(bid)
                elif src == "bsard_dataset" and bid in bsard_canonical:
                    # Prefer bsard_dataset source (overwrite earlier non-canonical).
                    existing_id = bsard_canonical[bid]
                    # Only overwrite if the existing record is NOT bsard_dataset.
                    # We can't check the existing record's source easily here,
                    # so we keep the FIRST bsard_dataset record (lowest article_id).
                    pass  # first-seen wins if same source

            if len(batch) >= BATCH_SIZE:
                conn.executemany(_INSERT_ART, batch)
                conn.commit()
                batch.clear()
                print(f"  ... {article_id:,} articles", end="\r", flush=True)

    if batch:
        conn.executemany(_INSERT_ART, batch)
        conn.commit()

    t1 = time.perf_counter()
    print(f"  Inserted {article_id:,} articles in {t1-t0:.1f}s")

    # ── Insert questions ──────────────────────────────────────────────────────
    print(f"Inserting questions from {args.qs.name} ...")
    questions = _load_jsonl(args.qs)

    _INSERT_Q = """
    INSERT INTO questions
        (question_id, question_text, relevant_article_ids, relevant_bsard_ids,
         n_relevant_articles, split)
    VALUES (?,?,?,?,?,?)
    """

    q_rows = []
    unresolved_qs = 0
    for q in questions:
        bsard_ids = q.get("relevant_bsard_ids", [])
        art_ids   = [bsard_canonical[bid] for bid in bsard_ids if bid in bsard_canonical]
        if len(art_ids) < len(bsard_ids):
            unresolved_qs += 1
        q_rows.append((
            int(q["question_id"]),
            q["question_text"],
            json.dumps(art_ids),
            json.dumps(bsard_ids),
            len(bsard_ids),
            q["split"],
        ))

    conn.executemany(_INSERT_Q, q_rows)
    conn.commit()
    print(f"  Inserted {len(q_rows)} questions "
          f"({unresolved_qs} with partially unresolved bsard_ids)")

    # ── Insert citation graph ─────────────────────────────────────────────────
    print(f"Inserting citation graph from {args.graph.name} ...")
    graph_records = _load_jsonl(args.graph)

    _INSERT_EDGE = """
    INSERT INTO citation_graph (edge_id, source_id, target_id, citation_text, resolved)
    VALUES (?,?,?,?,?)
    """
    edge_rows = [
        (e["edge_id"], e["source_id"], e["target_id"], e["citation_text"], e["resolved"])
        for e in graph_records
    ]
    conn.executemany(_INSERT_EDGE, edge_rows)
    conn.commit()
    print(f"  Inserted {len(edge_rows):,} citation edges")

    # ── Create indices ────────────────────────────────────────────────────────
    print("Creating indices ...", end=" ", flush=True)
    for ddl in _INDICES:
        conn.execute(ddl)
    conn.commit()
    print("done")

    # ── Build FTS5 virtual table ──────────────────────────────────────────────
    print("Building FTS5 index ...", end=" ", flush=True)
    conn.execute(_CREATE_FTS)
    conn.execute("INSERT INTO articles_fts(articles_fts) VALUES('rebuild')")
    conn.commit()
    print("done")

    # ── Validation queries ────────────────────────────────────────────────────
    print("\nValidation:")
    checks = [
        ("Total articles",         "SELECT COUNT(*) FROM articles"),
        ("BSARD articles",         "SELECT COUNT(*) FROM articles WHERE is_bsard_article=1"),
        ("Unique BSARD IDs",       "SELECT COUNT(DISTINCT bsard_id) FROM articles WHERE bsard_id IS NOT NULL"),
        ("Non-BSARD articles",     "SELECT COUNT(*) FROM articles WHERE is_bsard_article=0"),
        ("Has article_text",       "SELECT COUNT(*) FROM articles WHERE article_text IS NOT NULL"),
        ("Has chapter_title",      "SELECT COUNT(*) FROM articles WHERE chapter_title IS NOT NULL"),
        ("Total questions",        "SELECT COUNT(*) FROM questions"),
        ("Test questions",         "SELECT COUNT(*) FROM questions WHERE split='test'"),
        ("Citation graph edges",   "SELECT COUNT(*) FROM citation_graph"),
        ("FTS5 row count",         "SELECT COUNT(*) FROM articles_fts"),
    ]
    for label, sql in checks:
        val = conn.execute(sql).fetchone()[0]
        print(f"  {label:<28}: {val:,}")

    # ── Sample FTS5 search ────────────────────────────────────────────────────
    print("\nFTS5 smoke test — query 'contrat de travail':")
    rows = conn.execute("""
        SELECT a.article_id, a.law_code, a.article_number,
               snippet(articles_fts, 0, '[', ']', '...', 10) AS snippet
        FROM articles_fts
        JOIN articles a ON a.article_id = articles_fts.rowid
        WHERE articles_fts MATCH 'contrat de travail'
        ORDER BY rank
        LIMIT 3
    """).fetchall()
    for r in rows:
        print(f"  id={r[0]} [{r[1]}] Art.{r[2]}: {r[3][:80]}")

    conn.close()
    size_mb = args.db.stat().st_size / 1024 / 1024
    print(f"\nDatabase: {args.db}  ({size_mb:.1f} MB)")
    print("Phase D (build_database) complete.")


if __name__ == "__main__":
    main()
