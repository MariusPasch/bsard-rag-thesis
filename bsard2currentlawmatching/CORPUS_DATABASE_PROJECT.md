# BSARD Corpus Database — New Project Specification
*Parent project: BSARD | Author: Marios Paschalidis | Last updated: 2026-03-20*

---

## 1. Project Goal

Build a **SQLite corpus database** (with Parquet and JSONL exports) containing **every article from all 49 Justel consolidated PDFs**, enriched with retrieval-ready structural and relational metadata. The database will serve as the single source of truth for all downstream RAG retrieval experiments described in the [master thesis proposal](thesis_proposal_BSARD_RAG_WITH_timeline_v0.1.pdf) (RQ1–RQ3).

The corpus has two tiers:
- **BSARD articles** — the ~22,633 articles from the benchmark dataset, linked by `bsard_id`. These have ground truth question–article annotations and are used to compute Recall@k, NDCG, MRR across all experiments.
- **Non-BSARD articles** — all remaining articles found in the same 49 PDFs. These are critical as "distractors" for meaningful retrieval evaluation and as complete context for Graph RAG citation-neighbourhood expansion (Stage 5.5 of the thesis).

**No Azure Document Intelligence is used.** All article extraction and structural hierarchy detection is performed using PyMuPDF (`fitz`) with font-size heuristics and regex-based pattern matching, directly extending the approach established in the parent project.

---

## 2. Parent Project — What Already Exists

This new project is built on top of the BSARD pipeline project. Do not re-implement anything listed here — import or directly reference these artefacts.

### 2.1 Existing Pipeline Scripts

| Script | What it provides | Link |
|--------|-----------------|------|
| `bsard_full_verify.py` | CODE_MAP (all 32 Belgian law codes → Justel URLs), PDF URL extraction, HTML + PDF verification logic, temporal classification regex | [retrieval/bsard_full_verify.py](retrieval/bsard_full_verify.py) |
| `augment_markers.py` | Per-article temporal status: `amendment_date`, `article_status` (ORIGINAL / PRE_BSARD / POST_BSARD), `is_pre_bsard` flag | [analysis/augment_markers.py](analysis/augment_markers.py) |
| `build_pdf_article_cache.py` | Article-number regex (`\bArt\.?\s+([A-Z0-9][A-Z0-9_./-]*)`) that detects all Belgian article number formats; PyPDF2 full-text extraction pattern | [retrieval/build_pdf_article_cache.py](retrieval/build_pdf_article_cache.py) |
| `pdf_page_coverage.py` | PyMuPDF page-level text extraction pattern (`fitz.open`, `page.get_text`), snippet normalisation (`normalise()`), page-to-article mapping | [retrieval/pdf_page_coverage.py](retrieval/pdf_page_coverage.py) |
| `download_pdfs.py` | PDF download logic, URL-to-filename conversion (`url_to_filename()`) | [retrieval/download_pdfs.py](retrieval/download_pdfs.py) |

### 2.2 Existing Data Files

| File | Content | Link |
|------|---------|------|
| `bsard_full_verify.csv` | 22,633 rows × 14 columns: `id, code, article_no, law_type, url_type, url, pdf_url, http_status, anchor_names, anchor_found, text_found, pdf_text_found, pdf_match_category, status` | [bsard_full_verify.csv](bsard_full_verify.csv) |
| `output/pdf_article_counts.json` | 49 PDF URLs → count of distinct article numbers found in each PDF | [output/pdf_article_counts.json](output/pdf_article_counts.json) |
| `output/pdf_page_coverage.json` | 49 PDF URLs → per-article page index `{bsard_id: [page_numbers]}`, total pages, pages needed/skippable | [output/pdf_page_coverage.json](output/pdf_page_coverage.json) |
| `output/pdfs/` | 49 downloaded Justel consolidated PDFs (59 MB total; not republished on HF — regenerate via `pipeline/download_pdfs.py`) | [output/pdfs/](output/pdfs/) |

### 2.3 Existing Documentation

| Document | Content | Link |
|----------|---------|------|
| `BSARD_STATUS_REPORT.md` | Full pipeline design: CODE_MAP structure, URL construction logic, HTML/PDF verification rules, temporal classification, results | [BSARD_STATUS_REPORT.md](BSARD_STATUS_REPORT.md) |
| `PDF_EXTRACTION_FINDINGS.md` | Categorised root causes of PDF text extraction failures; OCR limitations; amendment-driven mismatches | [PDF_EXTRACTION_FINDINGS.md](PDF_EXTRACTION_FINDINGS.md) |
| `justel_marker_analysis.txt` | Deep analysis of Justel `En vigueur` inline markers; edge cases; marker format variations | [justel_marker_analysis.txt](justel_marker_analysis.txt) |

### 2.4 External Data Sources

| Source | What it provides | Access |
|--------|-----------------|--------|
| HuggingFace BSARD corpus | Canonical article text for all 22,633 BSARD articles; article `id`, `article` text, `code` (law name), `article_no` | `load_dataset("maastrichtlawtech/bsard", "corpus", split="corpus")` |
| HuggingFace BSARD questions | 1,100 natural language legal questions; `question` text, `relevant_articles` (list of BSARD article IDs), `split` (train/test) | `load_dataset("maastrichtlawtech/bsard", "questions", split="train")` and `split="test"` |

---

## 3. Target Database Schema

The database consists of three tables. All output files live under `output/` (gitignored locally; published to the Hugging Face dataset for distribution).

### 3.1 Table: `articles`

One row per extracted article — BSARD and non-BSARD combined. Total expected row count: larger than 22,633 (the 49 PDFs contain substantially more articles than the BSARD subset).

#### Identity and linkage

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `article_id` | INTEGER | NOT NULL | Auto-assigned primary key |
| `bsard_id` | INTEGER | YES | BSARD dataset `id` field. NULL for non-BSARD articles. Join key for ground truth evaluation |
| `is_bsard_article` | INTEGER | NOT NULL | 1 if this article is in the BSARD benchmark; 0 otherwise |

#### Source law metadata

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `law_code` | TEXT | NOT NULL | Law name, e.g. `"Code Civil"`, `"Code Pénal"`. From BSARD HF dataset for BSARD articles; inferred from PDF for others |
| `law_type` | TEXT | YES | `national` or `regional`. Sourced from `bsard_full_verify.csv` for BSARD articles |
| `article_number` | TEXT | NOT NULL | Article number as it appears in the PDF, e.g. `"42"`, `"1.1.1"`, `"D.II.46"` |
| `numac` | TEXT | YES | Moniteur belge publication ID for the source PDF, extracted from the PDF filename/URL. Used for precise provenance and law identity |

#### Article content

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `article_text` | TEXT | YES | Full article body text. For BSARD articles: canonical text from HuggingFace dataset. For non-BSARD: text extracted from PDF |
| `article_text_source` | TEXT | NOT NULL | `bsard_dataset` — HF canonical text used; `pdf_extracted` — text extracted from PDF; `pdf_extraction_failed` — text could not be extracted |

#### Hierarchical structure (from PyMuPDF heading detection)

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `law_title_text` | TEXT | YES | Top-level document title or book heading (e.g. `"LIVRE Ier — Des personnes"`). Level 0 |
| `chapter_title` | TEXT | YES | Chapter heading containing this article (e.g. `"CHAPITRE II — Des obligations"`). Level 1 |
| `section_title` | TEXT | YES | Section heading (e.g. `"Section 1re — Des contrats"`). Level 2 |
| `subsection_title` | TEXT | YES | Subsection heading if present. Level 3 |
| `hierarchy_path` | TEXT | YES | JSON array representing the full path from law title to article: `["TITRE Ier", "CHAPITRE II", "Section 1", "Art. 42"]`. Used directly by PageIndex (Stage 5.3 of thesis) |
| `hierarchy_depth` | INTEGER | YES | Number of levels in `hierarchy_path`. 0 = law title, 1 = chapter, 2 = section, 3 = subsection |

#### PDF source and positional metadata

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `pdf_url` | TEXT | YES | Justel consolidated PDF URL, e.g. `https://www.ejustice.just.fgov.be/img_l/pdf/1804/03/21/1804032150_F.pdf` |
| `pdf_filename` | TEXT | YES | Local filename under `output/pdfs/`, e.g. `img_l_pdf_1804_03_21_1804032150_F.pdf` |
| `pdf_page_numbers` | TEXT | YES | JSON array of 0-indexed page numbers where this article appears, e.g. `[12, 13]`. From PyMuPDF parsing |
| `pdf_page_start` | INTEGER | YES | First page (0-indexed). Enables targeted single-page re-reads |
| `pdf_page_end` | INTEGER | YES | Last page (0-indexed) |

#### Justel HTML provenance (BSARD articles only)

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `justel_html_url` | TEXT | YES | Full Justel `article.pl` URL for this article. NULL for non-BSARD articles. Sourced from `bsard_full_verify.csv` |

#### Cross-reference / citation metadata

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `cross_references_raw` | TEXT | YES | JSON array of raw citation strings found in `article_text` by regex, e.g. `["art. 1382", "l'article 3 de la loi du 10 juin 1998", "art. D.I.1"]` |
| `cross_reference_ids` | TEXT | YES | JSON array of resolved `article_id` values for citations that could be matched within this corpus. Unresolvable references are omitted |
| `n_outgoing_refs` | INTEGER | NOT NULL | Count of entries in `cross_references_raw`. 0 if no citations found. Used for cross-reference density statistics (Chapter 3 of thesis) |
| `cited_by_ids` | TEXT | YES | JSON array of `article_id` values of articles that cite this article. Inverse of `cross_reference_ids`. Computed as a post-processing step after all articles are inserted |
| `n_cited_by` | INTEGER | NOT NULL | In-degree in the citation graph. 0 if no other article cites this one |

#### Temporal metadata (BSARD articles only)

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `amendment_date` | TEXT | YES | Date string of last official amendment (`YYYY-MM-DD`). From `augment_markers.py` output |
| `article_status` | TEXT | YES | `ORIGINAL_NEVER_AMENDED`, `PRE_BSARD`, `POST_BSARD`, or `PARSE_ERROR`. From `augment_markers.py` |
| `is_pre_bsard` | INTEGER | YES | 1 if the article text was unchanged since BSARD collection (May 2021); 0 if modified since; NULL for non-BSARD articles |

#### Verification status (BSARD articles only)

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `html_text_found` | INTEGER | YES | 1 if BSARD article text was found in the Justel HTML page. From `bsard_full_verify.csv` |
| `pdf_text_found` | INTEGER | YES | 1 if BSARD article text was found in the PDF via stripped-text matching. From `bsard_full_verify.csv` |
| `pdf_match_category` | TEXT | YES | PDF failure reason if `pdf_text_found = 0`. From `bsard_full_verify.csv` |
| `verification_status` | TEXT | YES | `FOUND`, `PARTIAL`, or `NOT FOUND`. From `bsard_full_verify.csv` |

#### RAG-specific computed fields

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `token_count` | INTEGER | YES | Article token count using `tiktoken` with the `cl100k_base` encoding. Used for chunking ablation (Stage 5.1) and corpus length distribution statistics |
| `char_count` | INTEGER | NOT NULL | `len(article_text)`. Cheap size proxy |
| `has_cross_references` | INTEGER | NOT NULL | 1 if `n_outgoing_refs > 0`. Stratification flag for RQ1 failure-condition analysis (§4.5 of thesis) — isolates queries over articles with cross-reference dependencies |

---

### 3.2 Table: `questions`

One row per BSARD benchmark question. These are the queries used in all retrieval experiments.

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `question_id` | INTEGER | NOT NULL | BSARD question `id` field. Primary key |
| `question_text` | TEXT | NOT NULL | Natural language legal question in French |
| `relevant_article_ids` | TEXT | NOT NULL | JSON array of `article_id` values (resolved from BSARD `relevant_articles` via `bsard_id` linkage). These are the ground truth answers used to compute Recall@k, MRR@10, NDCG@10 |
| `relevant_bsard_ids` | TEXT | NOT NULL | JSON array of raw BSARD `id` values (keep alongside resolved IDs for traceability) |
| `n_relevant_articles` | INTEGER | NOT NULL | Length of `relevant_article_ids`. Used for failure-condition stratification: single-article queries vs. multi-article queries requiring synthesis (§4.5 of thesis) |
| `split` | TEXT | NOT NULL | `train` or `test`. The test split is used for all reported retrieval experiments |

---

### 3.3 Table: `citation_graph`

Materialized edge list for the citation graph. Loaded directly into NetworkX for Graph RAG (Stage 5.5 of thesis). Each row is one directed citation edge.

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `edge_id` | INTEGER | NOT NULL | Auto-assigned primary key |
| `source_id` | INTEGER | NOT NULL | `article_id` of the article containing the citation |
| `target_id` | INTEGER | NOT NULL | `article_id` of the cited article |
| `citation_text` | TEXT | NOT NULL | Raw citation string that produced this edge, e.g. `"art. 1382"` |
| `resolved` | INTEGER | NOT NULL | 1 if `target_id` was successfully resolved to an article in this corpus; 0 if the citation points outside the corpus (different law not in the 49 PDFs) |

---

## 4. Implementation Phases

### Phase A — Full PDF Article Extraction

**New script:** `retrieval/extract_articles_from_pdf.py`

This is the core new piece. It extends the PyMuPDF approach already established in [`retrieval/pdf_page_coverage.py`](retrieval/pdf_page_coverage.py) from page-level coverage to full structural article extraction.

#### A.1 Font-size analysis per PDF

Before extracting anything, run a frequency analysis of all font sizes in the PDF to establish the font-size hierarchy:

```python
import fitz
from collections import Counter

def analyse_font_sizes(doc: fitz.Document) -> dict:
    """
    Returns a dict mapping font_size -> total character count across the document.
    The most frequent non-trivial size is body text.
    Sizes significantly larger than body are headings.
    """
    freq = Counter()
    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    size = round(span["size"], 1)
                    freq[size] += len(span["text"].strip())
    return freq
```

**Heading level assignment** — run this once per PDF before parsing:
1. Find the body font size: the `size` value with the highest total character count
2. Assign heading levels by size relative to body: sizes > body+4 are likely chapter/section headings; sizes > body+2 but with `flags & 16` (bold) are likely subsection/article-number headings
3. Also check `span["flags"]`: flag bit `2^4 = 16` indicates bold; `2^1 = 2` indicates italic

#### A.2 Structural marker detection

Belgian legislative PDFs use consistent French-language heading patterns. Detect these **in addition to** font-size signals — the two signals combined are more reliable than either alone.

**French structural markers (regex patterns):**

```python
import re

# Ordered from highest to lowest hierarchy level
STRUCTURE_PATTERNS = [
    # Book level (highest — only in large codes)
    ("LIVRE",       re.compile(r'^\s*LIVRE\s+(?:[IVX]+|\d+)\b', re.MULTILINE)),
    # Title level
    ("TITRE",       re.compile(r'^\s*TITRE\s+(?:[IVX]+|\d+)\b', re.MULTILINE | re.IGNORECASE)),
    # Chapter level
    ("CHAPITRE",    re.compile(r'^\s*CHAPITRE\s+(?:[IVX]+|\d+|[Ii]er)\b', re.MULTILINE | re.IGNORECASE)),
    # Section level
    ("SECTION",     re.compile(r'^\s*(?:Section|SECTION)\s+(?:\d+(?:re|ère|e)?|[IVX]+)\b', re.MULTILINE)),
    # Subsection level
    ("SOUS-SECTION",re.compile(r'^\s*(?:Sous-section|SOUS-SECTION)\s+\d+\b', re.MULTILINE | re.IGNORECASE)),
    # Article marker (boundary signal — not a heading but must be detected)
    ("ARTICLE",     re.compile(r'^\s*Art\.?\s+([A-Z0-9][A-Z0-9_./-]*)\b', re.MULTILINE)),
]
```

**Dutch structural markers** (some PDFs are bilingual — detect both, prefer French):

```python
STRUCTURE_PATTERNS_NL = [
    ("BOEK",        re.compile(r'^\s*BOEK\s+(?:[IVX]+|\d+)\b', re.MULTILINE)),
    ("TITEL",       re.compile(r'^\s*TITEL\s+(?:[IVX]+|\d+)\b', re.MULTILINE | re.IGNORECASE)),
    ("HOOFDSTUK",   re.compile(r'^\s*HOOFDSTUK\s+(?:[IVX]+|\d+|[Ii])\b', re.MULTILINE | re.IGNORECASE)),
    ("AFDELING",    re.compile(r'^\s*(?:Afdeling|AFDELING)\s+\d+\b', re.MULTILINE)),
    ("ONDERAFDELING", re.compile(r'^\s*(?:Onderafdeling|ONDERAFDELING)\s+\d+\b', re.MULTILINE | re.IGNORECASE)),
    ("ARTIKEL",     re.compile(r'^\s*Art\.?\s+([A-Z0-9][A-Z0-9_./-]*)\b', re.MULTILINE)),
]
```

**Article number formats** in Belgian law — the regex from [`retrieval/build_pdf_article_cache.py:45`](retrieval/build_pdf_article_cache.py#L45) covers all known variants:

```python
# Handles: Art. 1 / Art. 1bis / Art. 1.1 / Art. I.1 / Art. D1 / Art. D.II.46 / Art. VI.4-5
ART_RE = re.compile(r'\bArt\.?\s+([A-Z0-9][A-Z0-9_./-]*)', re.MULTILINE)
```

**Important — do NOT split on these** (they appear inside article bodies, not as article boundaries):
- `§ 1.`, `§ 2.` — paragraph markers within a single article
- `1°`, `2°`, `a)`, `b)` — enumeration inside article text
- `al.` — alinéa markers

#### A.3 Extraction algorithm

```
For each PDF in output/pdfs/:
  1. Open with fitz.open()
  2. Analyse font sizes → establish body_size, heading_sizes
  3. Iterate page by page, extracting text blocks with get_text("dict")
  4. Maintain a running state: {current_livre, current_titre, current_chapitre,
                                current_section, current_sous_section,
                                current_article_no, current_article_lines,
                                current_article_pages}
  5. For each text block on each page:
     a. Determine if it is a structural heading (font size + pattern match)
     b. If structural heading → update running state; if a new article was being
        accumulated, flush it to the output list first
     c. If article marker (Art. X) → flush previous article (if any); start
        new article accumulation
     d. Otherwise → append text to current_article_lines
  6. After last page → flush the final article
  7. Output: list of article dicts with all state captured at flush time
```

**Page tracking:** Record the page number every time you start or continue appending to `current_article_lines`. An article that spans pages will have `pdf_page_numbers = [p1, p2]`.

#### A.4 Text cleaning

After extraction, apply the `normalise()` function pattern from [`retrieval/pdf_page_coverage.py:40`](retrieval/pdf_page_coverage.py#L40) to clean the extracted text, then store the cleaned version. Keep a flag indicating whether cleaning changed the text significantly (may indicate OCR noise in older PDFs).

Specific cleaning steps for Belgian legal PDFs:
- Collapse multiple whitespace / soft hyphens (common in narrow-column PDFs)
- Remove page headers/footers (typically short repeated lines at very top/bottom of each page — detect by appearance on ≥3 pages unchanged)
- Strip `[...]` placeholders that Justel inserts for repealed/amended sub-sections (flag their presence with `has_truncation_markers = 1`)

#### A.5 Output per PDF

Write intermediate JSONL to `output/extracted/{pdf_filename}.jsonl` — one JSON object per article. This allows the extraction to be resumed if interrupted (one file per PDF).

---

### Phase B — BSARD Linkage and Metadata Merge

**New script:** `retrieval/link_bsard.py`

#### B.1 Load BSARD HuggingFace corpus

```python
from datasets import load_dataset

corpus = load_dataset("maastrichtlawtech/bsard", "corpus", split="corpus")
# Fields: id (int), article (str), article_no (str), code (str)
# Build lookup: (normalised_code, normalised_article_no) -> {id, article, code}
```

Use the `normalise()` helper from [`retrieval/pdf_page_coverage.py:40`](retrieval/pdf_page_coverage.py#L40) for accent-stripping when building the lookup key — the same BSARD corpus has accent variants (e.g. `"Code Pénal"` vs `"Code Penal"`), exactly as handled in the CODE_MAP in [`retrieval/bsard_full_verify.py:75`](retrieval/bsard_full_verify.py#L75).

#### B.2 Linkage logic

For each article extracted in Phase A:
1. Build lookup key: `(normalise(law_code), normalise(article_number))`
2. If key found in BSARD lookup → set `bsard_id`, `is_bsard_article = 1`, override `article_text` with BSARD canonical text, set `article_text_source = 'bsard_dataset'`
3. If not found → `is_bsard_article = 0`, keep PDF-extracted text, set `article_text_source = 'pdf_extracted'`

**Known edge cases** (documented in [`BSARD_STATUS_REPORT.md`](BSARD_STATUS_REPORT.md)):
- Multi-part codes (Code Civil, Code Judiciaire, Code d'Instruction Criminelle) use overlapping article number ranges across PDFs — article number alone is not unique; must combine with `law_code` AND the article number range encoded in CODE_MAP
- Some article numbers in BSARD use leading integers only (e.g. `"42"`) while the PDF may show `"42bis"` or `"42/1"` — normalise both to leading integer for fuzzy matching, but preserve original `article_number` in the database

#### B.3 Merge `bsard_full_verify.csv`

Join on `bsard_id` to pull in: `pdf_url`, `justel_html_url` (the `url` column), `law_type`, `html_text_found` (from `text_found`), `pdf_text_found`, `pdf_match_category`, `verification_status` (the `status` column).

CSV path: [`bsard_full_verify.csv`](bsard_full_verify.csv). Column mapping:

| CSV column | DB column |
|-----------|-----------|
| `id` | → join key for `bsard_id` |
| `url` | → `justel_html_url` |
| `pdf_url` | → `pdf_url` |
| `law_type` | → `law_type` |
| `text_found` | → `html_text_found` |
| `pdf_text_found` | → `pdf_text_found` |
| `pdf_match_category` | → `pdf_match_category` |
| `status` | → `verification_status` |

#### B.4 Merge temporal metadata

Run (or load output of) [`analysis/augment_markers.py`](analysis/augment_markers.py) on the BSARD subset. This script fetches only ~35–40 unique Justel HTML pages and requires no PDF access. Its output CSV adds `last_ev_date`, `article_status`, `is_pre_bsard` columns.

Map onto BSARD articles: `last_ev_date` → `amendment_date`, `article_status` → `article_status`, `is_pre_bsard` → `is_pre_bsard`.

#### B.5 Load BSARD questions

```python
train_qs = load_dataset("maastrichtlawtech/bsard", "questions", split="train")
test_qs  = load_dataset("maastrichtlawtech/bsard", "questions", split="test")
# Fields: id (int), question (str), relevant_articles (str — comma-separated BSARD ids)
```

For each question, resolve `relevant_articles` (comma-separated BSARD IDs) to `article_id` values using the `bsard_id → article_id` mapping built during linkage. Store both `relevant_article_ids` (resolved) and `relevant_bsard_ids` (original).

---

### Phase C — Cross-Reference Extraction

**New script:** `retrieval/extract_citations.py`

#### C.1 Regex patterns for Belgian legal citations

```python
# Patterns are applied to article_text for each article.
# All patterns are case-insensitive.

CITATION_PATTERNS = [
    # Simple article reference: "art. 42", "Art. 1.1.1", "article 3bis"
    re.compile(
        r'\bart(?:icle)?s?\.?\s+([A-Z0-9][A-Z0-9_./-]*(?:bis|ter|quater|quinquies)?)',
        re.IGNORECASE
    ),
    # Range: "articles 12 à 15", "art. 3 et 4"
    re.compile(
        r'\bart(?:icle)?s?\s+(\d[\w./-]*)\s+(?:et|à|jusqu(?:\'|')au?)\s+(\d[\w./-]*)',
        re.IGNORECASE
    ),
    # Named law + article: "l'article 3 de la loi du 10 juin 1998"
    re.compile(
        r'\bl\'?art(?:icle)?\.?\s+(\d[\w./-]*)\s+de\s+la\s+(?:loi|arrêté|décret)',
        re.IGNORECASE
    ),
    # Present code self-reference: "du présent code", "du présent titre"
    # These are relative references — flag separately, do not resolve to ID
    re.compile(
        r'\bdu\s+présent\s+(?:code|titre|chapitre|article)',
        re.IGNORECASE
    ),
    # "l'alinéa précédent", "l'article précédent" — relative, flag only
    re.compile(
        r'\b(?:l\'|l')alinéa\s+précédent\b|\bde\s+l\'article\s+précédent\b',
        re.IGNORECASE
    ),
]
```

#### C.2 Resolution strategy

For each extracted raw citation string:
1. Extract the article number portion
2. Determine the likely law code from context:
   - If the citation includes `"du présent code"` or similar → same `law_code` as source article
   - If the citation includes a law name or date → attempt to match against the CODE_MAP codes in [`retrieval/bsard_full_verify.py:75`](retrieval/bsard_full_verify.py#L75)
   - Otherwise → search across all articles with matching `article_number` (may be ambiguous)
3. If a unique match is found → record as resolved edge in `citation_graph`
4. If ambiguous (multiple matching articles in different codes) → keep as `cross_references_raw` only, `resolved = 0`
5. If no match (citation to law outside the 49 PDFs) → keep as `cross_references_raw` only, `resolved = 0`

#### C.3 Build inverse index

After all citations are resolved, compute `cited_by_ids` and `n_cited_by` for every article by inverting the `cross_reference_ids` mapping.

---

### Phase D — Database Assembly and Exports

**New script:** `retrieval/build_database.py`

#### D.1 SQLite construction

```python
import sqlite3

DB_PATH = Path("output/bsard_corpus.db")
conn = sqlite3.connect(DB_PATH)

# Enable WAL mode for faster writes
conn.execute("PRAGMA journal_mode=WAL")
conn.execute("PRAGMA synchronous=NORMAL")

# Create tables with schema above
# ...

# Critical indices for RAG query patterns
conn.execute("CREATE INDEX idx_articles_bsard_id   ON articles(bsard_id)")
conn.execute("CREATE INDEX idx_articles_law_code    ON articles(law_code)")
conn.execute("CREATE INDEX idx_articles_pdf_file    ON articles(pdf_filename)")
conn.execute("CREATE INDEX idx_articles_art_status  ON articles(article_status)")
conn.execute("CREATE INDEX idx_articles_is_bsard    ON articles(is_bsard_article)")
conn.execute("CREATE INDEX idx_citation_source      ON citation_graph(source_id)")
conn.execute("CREATE INDEX idx_citation_target      ON citation_graph(target_id)")
```

#### D.2 FTS5 virtual table

Add a full-text search virtual table over `article_text`. This enables BM25-style full-text search directly in SQLite — useful for quick validation of sparse retrieval results before standing up Elasticsearch.

```python
conn.execute("""
    CREATE VIRTUAL TABLE articles_fts USING fts5(
        article_text,
        content='articles',
        content_rowid='article_id',
        tokenize='unicode61'
    )
""")
conn.execute("INSERT INTO articles_fts(articles_fts) VALUES('rebuild')")
```

#### D.3 Exports

```python
import pandas as pd

df = pd.read_sql("SELECT * FROM articles", conn)

# Parquet — for pandas/polars/retrieval workflows
df.to_parquet("output/bsard_articles.parquet", index=False, engine="pyarrow")

# JSONL — for vector store ingestion (one JSON object per line)
df.to_json("output/bsard_articles.jsonl", orient="records", lines=True, force_ascii=False)

# BSARD-only subset for benchmarking
bsard_df = df[df["is_bsard_article"] == 1]
bsard_df.to_parquet("output/bsard_articles_only.parquet", index=False)
bsard_df.to_json("output/bsard_articles_only.jsonl", orient="records", lines=True, force_ascii=False)
```

---

### Phase E — Corpus Statistics

**New script:** `retrieval/corpus_stats.py`

Compute and print all statistics required for Chapter 3 of the thesis. Save results to `output/corpus_stats.json`.

| Statistic | Method |
|-----------|--------|
| Total articles extracted | `SELECT COUNT(*) FROM articles` |
| BSARD articles vs. non-BSARD | `SELECT is_bsard_article, COUNT(*) FROM articles GROUP BY is_bsard_article` |
| Article length distribution (tokens) | percentiles over `token_count` where `article_text IS NOT NULL` |
| Cross-reference density | `SELECT AVG(n_outgoing_refs), MAX(n_outgoing_refs)` per law_code |
| Hierarchy coverage | `SELECT COUNT(*) WHERE chapter_title IS NOT NULL` / total |
| Articles with complete hierarchy (all levels) | `chapter_title AND section_title NOT NULL` |
| Query–article lexical overlap | For each BSARD question, compute Jaccard overlap between question tokens and ground-truth article tokens — motivates semantic retrieval |
| PDF extraction failure rate | `SELECT article_text_source, COUNT(*)` |
| Image-based / OCR-failed PDFs | articles where `pdf_text_found = 0` and `article_text_source = 'pdf_extracted'` |

---

## 5. File Structure for the New Project

```
bsard-corpus-db/                       ← New project root
│
├── retrieval/
│   ├── extract_articles_from_pdf.py   ← Phase A: PyMuPDF structural extraction
│   ├── link_bsard.py                  ← Phase B: BSARD linkage + metadata merge
│   ├── extract_citations.py           ← Phase C: cross-reference extraction
│   ├── build_database.py              ← Phase D: SQLite assembly
│   ├── export_corpus.py               ← Phase D: Parquet + JSONL exports
│   └── corpus_stats.py                ← Phase E: Chapter 3 statistics
│
├── output/                            ← Generated artifacts (gitignored; mirrored to Hugging Face)
│   ├── bsard_corpus.db                ← Primary SQLite database
│   ├── bsard_articles.parquet         ← Full corpus flat export
│   ├── bsard_articles.jsonl           ← Full corpus for vector store ingestion
│   ├── bsard_articles_only.parquet    ← BSARD subset only
│   ├── bsard_articles_only.jsonl      ← BSARD subset for benchmarking
│   ├── corpus_stats.json              ← Chapter 3 statistics
│   └── extracted/                     ← Per-PDF intermediate JSONL (Phase A)
│       ├── img_l_pdf_1804_03_21_1804032150_F.jsonl
│       └── ...
│
├── requirements.txt
└── README.md
```

The new project reads from — but never writes to — the parent BSARD project's `output/pdfs/` directory and `bsard_full_verify.csv`. Configure the path to the parent project via an environment variable or config file.

---

## 6. Technical Stack

| Category | Library | Notes |
|----------|---------|-------|
| PDF parsing | `PyMuPDF` (`fitz`) | Already used in parent project. `get_text("dict")` for font-size-aware extraction |
| Dataset loading | `datasets` (HuggingFace) | Already used in parent project. Stream BSARD corpus and questions |
| Database | `sqlite3` (stdlib) | No extra dependency. WAL mode for performance |
| Token counting | `tiktoken` | Use `cl100k_base` encoding (OpenAI standard, also used by most embedding models) |
| Data processing | `pandas` | Already in parent project requirements |
| Parquet export | `pyarrow` | Fast, preserves types |
| Regex | `re` (stdlib) | All pattern matching |
| Normalisation | `unicodedata` (stdlib) | Reuse `normalise()` from [`retrieval/pdf_page_coverage.py:40`](retrieval/pdf_page_coverage.py#L40) |

---

## 7. Key Design Decisions

### Canonical text priority
BSARD HuggingFace text is used as canonical for BSARD articles — it was the exact text used to create the ground truth annotations. PDF-extracted text is used for non-BSARD articles only. This ensures that Recall@k computed during experiments is evaluated against the correct text.

### Extraction failures are not errors
Articles where PDF extraction fails (image-based PDFs, OCR noise) are still inserted into the database with `article_text = NULL` and `article_text_source = 'pdf_extraction_failed'`. This preserves their structural metadata (hierarchy path, page numbers, citation edges pointing to them). A BSARD article with `is_bsard_article = 1` always has text from the HF dataset regardless of PDF extraction success.

### Per-PDF intermediate JSONL
Phase A writes one JSONL per PDF before any database work. This makes extraction resumable (skip already-extracted PDFs) and separates the slow PDF parsing step from the database assembly step.

### FTS5 virtual table
The SQLite FTS5 table over `article_text` enables BM25-style retrieval directly in the database. This is useful as a quick baseline validation before standing up Elasticsearch or implementing BM25 through `rank_bm25`. It uses `tokenize='unicode61'` which handles French accents correctly.

### All 49 PDFs, not just BSARD-linked
Some non-BSARD articles are the targets of cross-references from BSARD articles. Having them in the database means Graph RAG can follow citation edges to their full depth (1-hop and 2-hop) rather than hitting a dead end when the cited article is outside the BSARD benchmark.

### `numac` as law identity key
The `numac` field (Moniteur belge publication ID extracted from PDF URL, e.g. `1804032150` from `img_l_pdf_1804_03_21_1804032150_F.pdf`) is the most reliable identity key for a specific law document. It should be stored alongside `law_code` to enable precise cross-reference resolution when article numbers alone are ambiguous across codes.

---

## 8. Integration with RAG Experiments (Thesis Map)

| DB Feature | Experiment Stage |
|-----------|-----------------|
| `article_text` + `is_bsard_article` | Corpus for all Tier 1–4 retrieval (RQ1 Ch. 4) |
| `bsard_id` + questions table | Ground truth for Recall@k, MRR@10, NDCG@10 |
| `token_count` | Chunking ablation (Stage 5.1, RQ2) |
| `chapter_title`, `section_title`, `hierarchy_path` | Hierarchical chunking, metadata filtering, PageIndex tree (Stages 5.1–5.3, RQ2) |
| `law_code`, `law_type`, `amendment_date` | Metadata-filtered retrieval pre-filter (Stage 5.2, RQ2) |
| `cross_reference_ids`, `cited_by_ids` + `citation_graph` table | Graph RAG knowledge graph — nodes + directed edges (Stage 5.5, RQ2) |
| `has_cross_references`, `n_relevant_articles` (questions) | Failure-condition stratification (§4.5, RQ1) |
| `article_status`, `is_pre_bsard` | Temporal filtering; exclude POST_BSARD articles from strict evaluation |
| FTS5 virtual table | BM25 baseline validation; sparse retrieval Tier 1 quick prototyping |
