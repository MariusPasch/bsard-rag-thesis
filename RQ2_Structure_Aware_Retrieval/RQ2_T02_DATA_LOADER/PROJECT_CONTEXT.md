# 02 — DATA LOADER CONTEXT
## Auto-Discovery, PDF↔CSV Matching, Article Data Model

---

## 1. PURPOSE

Handles all data loading: auto-discovers PDF files, matches them to their Azure Document Intelligence extractions via MyDocuments.csv, joins with DocumentDefinitions.csv, and produces `DocumentBundle` objects consumed by all downstream sub-projects.

## 2. DIRECTORY STRUCTURE

```
data_loader/
├── __init__.py          # load_documents() orchestrator entry point + re-exports
├── auto_discover.py     # PDF → DocumentId matching via pdf_document_map.csv
├── pdf_loader.py        # PyMuPDF raw text extraction
├── azure_loader.py      # Article objects from articles CSV + bsard_id lookup (BSARD DB)
├── bootstrap.py         # CLI: per-PDF discovery.json stamp (python -m data_loader.bootstrap)
└── models.py            # Data classes (DocumentBundle, Article, Chunk, RetrievalResult)
```

## 3. DATA MODEL (models.py)

```python
from dataclasses import dataclass, field
import pandas as pd

@dataclass
class Article:
    article_id: str              # Unique identifier (chunk/article ID within document)
    text: str                    # Full article text
    document_id: int             # FK to MyDocuments
    document_title: str          # From MyDocuments.DocumentTitle
    jurisdiction: str            # From MyDocuments.Jurisdiction
    issuing_authority: str       # From MyDocuments.IssuingAuthority
    document_number: str         # From MyDocuments.DocumentNumber
    issue_date: str              # From MyDocuments.IssueDate
    entry_into_force: str        # From MyDocuments.EntryIntoForceDate
    regulatory_status: str       # From MyDocuments.RegulatoryStatus
    language: list[str]          # From MyDocuments.Language (parsed from JSON)
    summary_sentence: str        # First sentence of MyDocuments.Summary
    defined_terms: list[str]     # Terms from DocumentDefinitions where article_id ∈ UsedIn
    term_definitions: dict       # {term: definition_text} for terms used in this article
    cross_references: list[str]  # Article IDs cited within this article (regex-extracted)
    token_count: int             # Token count of article text
    # Hierarchy fields (from Azure extraction if available):
    chapter: str = ""            # Chapter title/number
    section: str = ""            # Section title/number
    article_number: str = ""     # Article number within the law (e.g., "D.35")

@dataclass
class Chunk:
    chunk_id: str
    text: str
    source_pdf: str
    start_char: int              # Character offset in raw text
    end_char: int
    start_token: int             # Token offset (for weighted metrics)
    end_token: int
    
@dataclass
class DocumentBundle:
    document_id: int
    pdf_path: str
    pdf_filename: str            # e.g., "2.pdf"
    metadata: dict               # Full row from MyDocuments.csv as dict
    definitions: pd.DataFrame    # Filtered DocumentDefinitions rows for this document
    articles: list[Article]      # Parsed articles (populated by azure_loader)
    raw_text: str                # PyMuPDF extracted text (populated by pdf_loader)
    has_azure_extraction: bool   # True if matched in MyDocuments.csv
    
    @property
    def defined_terms_list(self) -> list[str]:
        """All unique defined terms for this document."""
        return self.definitions['Term'].unique().tolist() if len(self.definitions) > 0 else []

@dataclass
class RetrievalResult:
    query_id: str
    query_text: str
    ranked_items: list[dict]     # [{id, score, text, metadata, article_ids}, ...]
    method: str                  # e.g. "arm1", "2A-raw", "2A-full", "2B"
    cost: dict                   # {llm_calls, tokens_in, tokens_out, latency_ms}
    trace: dict | None = None    # Reasoning trace (PageIndex only)
    # All nested values must be plain Python types so json.dumps(asdict(result)) succeeds.
```

`Article` also carries `bsard_id: int | None` (populated from the BSARD corpus DB when
available; `None` for non-BSARD articles).

## 4. AUTO-DISCOVERY (auto_discover.py)

### 4.1 Matching Logic

```python
def discover_documents(config: dict) -> list[DocumentBundle]:
    """
    Scan pdf_dir, match each PDF to MyDocuments.csv via pdf_document_map.csv,
    join definitions from DocumentDefinitions.csv via DocumentId.
    """
    pdf_dir  = config["data"]["pdf_dir"]
    docs_csv = pd.read_csv(config["data"]["documents_csv"], encoding_errors="replace")
    defs_csv = pd.read_csv(config["data"]["definitions_csv"], encoding_errors="replace")
    pdf_map  = pd.read_csv(config["data"]["pdf_document_map"])   # pdf_filename → document_id

    filename_to_doc_id = dict(zip(pdf_map["pdf_filename"], pdf_map["document_id"].astype(int)))
    doc_id_to_row = {int(r["DocumentId"]): r.to_dict() for _, r in docs_csv.iterrows()}

    bundles = []
    for pdf_path in sorted(glob(f"{pdf_dir}/*.pdf")):
        filename = Path(pdf_path).name                          # e.g. "2004_05_27_2004A27101.pdf"
        if filename in filename_to_doc_id:                      # mapped → Azure extraction available
            doc_id   = filename_to_doc_id[filename]
            doc_row  = doc_id_to_row[doc_id]
            doc_defs = defs_csv[defs_csv["DocumentId"] == doc_id].copy()
            bundle = DocumentBundle(document_id=doc_id, pdf_path=pdf_path, pdf_filename=filename,
                                    metadata=doc_row, definitions=doc_defs, articles=[],
                                    raw_text="", has_azure_extraction=True)
        else:                                                   # unmapped → Arm 1 only
            bundle = DocumentBundle(document_id=-1, pdf_path=pdf_path, pdf_filename=filename,
                                    metadata={}, definitions=pd.DataFrame(), articles=[],
                                    raw_text="", has_azure_extraction=False)
        bundles.append(bundle)
    return bundles
```

> Matching is by **`pdf_document_map.csv`** (a manual `pdf_filename → document_id` map),
> **not** by `DocumentPdfUrl` — the URL field uses the original upload filename, which no
> longer matches the local `{YYYY_MM_DD}_{code}.pdf` names.

### 4.2 Edge Cases

- PDF with no matching MyDocuments row → `has_azure_extraction=False`, Arm 1 only
- PDF with matching MyDocuments row but 0 definitions → `has_azure_extraction=True`, definitions DataFrame is empty, term-based features disabled
- Multiple PDFs matching the same DocumentId → should not happen, log error if it does
- MyDocuments rows with no corresponding PDF → ignored (only process PDFs that exist)

## 5. PDF LOADER (pdf_loader.py)

```python
def extract_raw_text(pdf_path: str) -> str:
    """Extract raw text from PDF using PyMuPDF. Page-by-page concatenation."""
    import fitz
    doc = fitz.open(pdf_path)
    pages = []
    for page_num, page in enumerate(doc):
        text = page.get_text()
        pages.append(text)
    return "\n\n".join(pages)

def populate_raw_text(bundle: DocumentBundle) -> DocumentBundle:
    """Fill the raw_text field of a DocumentBundle."""
    bundle.raw_text = extract_raw_text(bundle.pdf_path)
    return bundle
```

## 6. AZURE LOADER (azure_loader.py)

```python
# Article TEXT comes from a separate articles CSV (chunk_id, text, document_id,
# + optional chapter/section/article_number) — NOT parsed live from Azure output.
def populate_articles(
    bundle: DocumentBundle,
    articles_csv_path: str,
    bsard_id_lookup: dict[str, int] | None = None,
) -> DocumentBundle:
    """Attach Article objects from articles_csv_path (filtered by document_id).
    bsard_id_lookup maps article_id -> bsard_id; absent ids get bsard_id=None.
    No-op when has_azure_extraction is False."""
    if not bundle.has_azure_extraction:
        return bundle

    articles_df  = pd.read_csv(articles_csv_path, encoding_errors="replace")
    doc_articles = articles_df[articles_df["document_id"] == bundle.document_id]
    term_usage   = _build_term_usage_map(bundle.definitions)
    lookup       = bsard_id_lookup or {}

    articles = []
    for _, row in doc_articles.iterrows():
        article_id = str(int(row["chunk_id"]))
        text       = str(row["text"])
        terms      = term_usage.get(article_id, [])
        articles.append(Article(
            article_id=article_id, text=text, document_id=bundle.document_id,
            document_title=bundle.metadata.get("DocumentTitle", ""),
            # ... remaining metadata fields pulled from bundle.metadata ...
            defined_terms=[t["term"] for t in terms],
            term_definitions={t["term"]: t["definition"] for t in terms},
            cross_references=_extract_cross_references(text),
            token_count=len(text.split()),                    # whitespace word count, not a tokenizer
            chapter=str(row.get("chapter", "")), section=str(row.get("section", "")),
            article_number=str(row.get("article_number", "")),
            bsard_id=lookup.get(article_id),
        ))
    bundle.articles = articles
    return bundle


def build_bsard_id_lookup(db_path: str | Path) -> dict[str, int]:
    """article_id -> bsard_id from the BSARD corpus SQLite DB (read-only),
    omitting rows where bsard_id IS NULL (treated as non-BSARD)."""
    ...


def build_term_usage_map(defs_df: pd.DataFrame) -> dict:
    """
    Build mapping: article_id → list of {term, definition, ...} used in that article.
    Uses the UsedIn field (JSON list of article/chunk IDs).
    """
    usage_map = defaultdict(list)
    for _, row in defs_df.iterrows():
        used_in_ids = json.loads(row['UsedIn'])
        for article_id in used_in_ids:
            usage_map[str(article_id)].append({
                'term': row['Term'],
                'definition': row['Definition'],
                'translated_term': row['TranslatedTerm'],
                'translated_definition': row['TranslatedDefinition'],
            })
    return dict(usage_map)

def extract_cross_references(article_text: str) -> list[str]:
    """
    Extract cross-referenced article IDs from article text using regex.
    Belgian statutory citation patterns.
    """
    patterns = [
        r"(?:article|art\.?)\s+([D\.]?\s*\d+(?:\.\d+)?)",
        r"(?:visé|prévu|conformément)\s+(?:à|par)\s+(?:l['\u2019])?article\s+([D\.]?\s*\d+)",
    ]
    refs = set()
    for pattern in patterns:
        for match in re.finditer(pattern, article_text, re.IGNORECASE):
            refs.add(match.group(1).strip())
    return sorted(refs)
```

## 7. INTERFACE FOR ORCHESTRATOR

```python
# Main entry point called by orchestrator (data_loader/__init__.py)
def load_documents(config: dict, articles_csv_path: str | None = None) -> list[DocumentBundle]:
    bundles = auto_discover.discover_documents(config)
    # Build a one-shot article_id -> bsard_id lookup if a BSARD DB is configured
    bsard_lookup = (azure_loader.build_bsard_id_lookup(config["data"]["bsard_db"])
                    if articles_csv_path and config.get("data", {}).get("bsard_db") else None)
    for bundle in bundles:
        pdf_loader.populate_raw_text(bundle)
        if bundle.has_azure_extraction and articles_csv_path:
            azure_loader.populate_articles(bundle, articles_csv_path, bsard_lookup)
    return bundles
```

> If `articles_csv_path` is `None`, `azure_loader` is skipped and every bundle's
> `articles` stays empty (raw text is still populated, so Arm 1 still runs).
