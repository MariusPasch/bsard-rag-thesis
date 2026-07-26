from __future__ import annotations

from dataclasses import dataclass, field
import pandas as pd


@dataclass
class Article:
    article_id: str
    text: str
    document_id: int
    document_title: str
    jurisdiction: str
    issuing_authority: str
    document_number: str
    issue_date: str
    entry_into_force: str
    regulatory_status: str
    language: list[str]
    summary_sentence: str
    defined_terms: list[str]
    term_definitions: dict[str, str]
    cross_references: list[str]
    token_count: int
    chapter: str = ""
    section: str = ""
    article_number: str = ""
    bsard_id: int | None = None


@dataclass
class Chunk:
    chunk_id: str
    text: str
    source_pdf: str
    start_char: int
    end_char: int
    start_token: int
    end_token: int


@dataclass
class DocumentBundle:
    document_id: int
    pdf_path: str
    pdf_filename: str        # bare basename — always Path(pdf_path).name
    metadata: dict
    definitions: pd.DataFrame
    articles: list[Article]
    raw_text: str
    has_azure_extraction: bool

    @property
    def defined_terms_list(self) -> list[str]:
        if len(self.definitions) == 0:
            return []
        return self.definitions["Term"].unique().tolist()


@dataclass
class RetrievalResult:
    """
    All values inside ranked_items and cost must be plain Python types
    (str, int, float, list, dict) so that json.dumps(dataclasses.asdict(result))
    succeeds without error. Never store numpy scalars or sets here.
    """
    query_id: str
    query_text: str
    ranked_items: list[dict]   # [{id, score, text, metadata, article_ids}, ...]
    method: str
    cost: dict
    trace: dict | None = None
