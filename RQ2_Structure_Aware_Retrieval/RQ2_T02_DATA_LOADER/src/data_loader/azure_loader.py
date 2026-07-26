"""Parse Azure DI CSVs into Article objects."""

import json
import logging
import re
import sqlite3
from collections import defaultdict
from pathlib import Path

import pandas as pd

from .models import Article, DocumentBundle

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def populate_articles(
    bundle: DocumentBundle,
    articles_csv_path: str,
    bsard_id_lookup: dict[str, int] | None = None,
) -> DocumentBundle:
    """
    Load article text from articles_csv_path and attach Article objects to bundle.

    articles_csv_path must contain at minimum: chunk_id (int), text (str), document_id (int).
    Additional columns (chapter, section, article_number) are used if present.

    bsard_id_lookup maps article_id (str) → bsard_id (int) for BSARD articles.
    Articles whose article_id is absent from the lookup get bsard_id=None
    (non-BSARD articles or HF-only stubs).

    TODO: finalise once the articles CSV schema is confirmed and the file is available.
    """
    if not bundle.has_azure_extraction:
        return bundle

    articles_df = pd.read_csv(articles_csv_path, encoding="utf-8", errors="replace")
    doc_articles = articles_df[articles_df["document_id"] == bundle.document_id].copy()

    if len(doc_articles) == 0:
        logger.warning(f"No article rows found for DocumentId={bundle.document_id} in {articles_csv_path}")
        return bundle

    doc_meta = bundle.metadata
    term_usage = _build_term_usage_map(bundle.definitions)
    lookup = bsard_id_lookup or {}

    articles: list[Article] = []
    for _, row in doc_articles.iterrows():
        article_id = str(int(row["chunk_id"]))
        text = str(row["text"])
        article_terms = term_usage.get(article_id, [])

        article = Article(
            article_id=article_id,
            text=text,
            document_id=bundle.document_id,
            document_title=doc_meta.get("DocumentTitle", ""),
            jurisdiction=doc_meta.get("Jurisdiction", ""),
            issuing_authority=doc_meta.get("IssuingAuthority", ""),
            document_number=doc_meta.get("DocumentNumber", ""),
            issue_date=doc_meta.get("IssueDate", ""),
            entry_into_force=doc_meta.get("EntryIntoForceDate", ""),
            regulatory_status=doc_meta.get("RegulatoryStatus", ""),
            language=_parse_language(doc_meta.get("Language", "[]")),
            summary_sentence=_first_sentence(doc_meta.get("Summary", "")),
            defined_terms=[t["term"] for t in article_terms],
            term_definitions={t["term"]: t["definition"] for t in article_terms},
            cross_references=_extract_cross_references(text),
            token_count=len(text.split()),
            chapter=str(row.get("chapter", "")),
            section=str(row.get("section", "")),
            article_number=str(row.get("article_number", "")),
            bsard_id=lookup.get(article_id),
        )
        articles.append(article)

    bundle.articles = articles
    n_with_bsard = sum(1 for a in articles if a.bsard_id is not None)
    logger.info(
        "Populated %d articles for DocumentId=%d (%d with bsard_id)",
        len(articles), bundle.document_id, n_with_bsard,
    )
    return bundle


def build_bsard_id_lookup(db_path: str | Path) -> dict[str, int]:
    """Read article_id → bsard_id from the BSARD corpus DB.

    Returns a dict keyed by stringified article_id (matching the form used in
    Article.article_id). Articles with bsard_id IS NULL are omitted —
    callers should treat absence as "non-BSARD article".
    """
    db = Path(db_path)
    if not db.exists():
        raise FileNotFoundError(f"BSARD corpus DB not found: {db}")

    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:
        rows = conn.execute(
            "SELECT article_id, bsard_id FROM articles WHERE bsard_id IS NOT NULL"
        ).fetchall()

    return {str(int(article_id)): int(bsard_id) for article_id, bsard_id in rows}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_term_usage_map(defs_df: pd.DataFrame) -> dict[str, list[dict]]:
    """Map article_id → list of {term, definition, ...} for terms used in that article."""
    usage_map: dict[str, list[dict]] = defaultdict(list)
    for _, row in defs_df.iterrows():
        try:
            used_in_ids = json.loads(row["UsedIn"])
        except (json.JSONDecodeError, TypeError):
            continue
        for article_id in used_in_ids:
            usage_map[str(article_id)].append({
                "term": row.get("Term", ""),
                "definition": row.get("Definition", ""),
                "translated_term": row.get("TranslatedTerm", ""),
                "translated_definition": row.get("TranslatedDefinition", ""),
            })
    return dict(usage_map)


def _parse_language(raw: str) -> list[str]:
    try:
        result = json.loads(raw)
        return result if isinstance(result, list) else [str(result)]
    except (json.JSONDecodeError, TypeError):
        return []


def _first_sentence(text: str) -> str:
    if not text:
        return ""
    match = re.search(r"[^.!?]*[.!?]", text)
    return match.group(0).strip() if match else text.strip()


def _extract_cross_references(text: str) -> list[str]:
    """Extract article IDs cited in Belgian statutory text."""
    patterns = [
        r"(?:article|art\.?)\s+([D\.]?\s*\d+(?:\.\d+)?)",
        r"(?:visé|prévu|conformément)\s+(?:à|par)\s+(?:l['\u2019])?article\s+([D\.]?\s*\d+)",
    ]
    refs: set[str] = set()
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            refs.add(match.group(1).strip())
    return sorted(refs)
