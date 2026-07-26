"""Auto-discovery of PDF↔CSV pairs, raw text extraction, Article dataclass construction."""

import logging

from .models import Article, Chunk, DocumentBundle, RetrievalResult
from . import auto_discover, pdf_loader, azure_loader

logger = logging.getLogger(__name__)


def load_documents(config: dict, articles_csv_path: str | None = None) -> list[DocumentBundle]:
    """
    Main entry point called by the orchestrator.

    articles_csv_path: path to the articles CSV (chunk_id, text, document_id, ...).
    If None, azure_loader is skipped and bundle.articles will be empty for all bundles.

    BSARD-id resolution: when config["data"]["bsard_db"] is set, a one-shot
    article_id → bsard_id lookup is built from the DB and used to populate
    Article.bsard_id. Articles absent from the lookup keep bsard_id=None.
    """
    bundles = auto_discover.discover_documents(config)

    bsard_id_lookup: dict[str, int] | None = None
    bsard_db_path = config.get("data", {}).get("bsard_db")
    if articles_csv_path and bsard_db_path:
        try:
            bsard_id_lookup = azure_loader.build_bsard_id_lookup(bsard_db_path)
            logger.info("Loaded bsard_id lookup: %d article_ids", len(bsard_id_lookup))
        except FileNotFoundError as exc:
            logger.warning("bsard_db not found, skipping bsard_id population: %s", exc)

    for bundle in bundles:
        pdf_loader.populate_raw_text(bundle)
        if bundle.has_azure_extraction and articles_csv_path:
            azure_loader.populate_articles(bundle, articles_csv_path, bsard_id_lookup)
    return bundles


__all__ = [
    "Article",
    "Chunk",
    "DocumentBundle",
    "RetrievalResult",
    "load_documents",
]
