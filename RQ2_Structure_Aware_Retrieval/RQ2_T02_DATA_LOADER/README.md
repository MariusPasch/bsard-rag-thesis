# RQ2_T02_DATA_LOADER

Auto-discovery of PDF↔CSV pairs, raw text extraction, Article dataclass construction.

---

## Setup

> **Prerequisite:** the source PDFs and CSVs are not in git. Pull them from the companion Hugging Face dataset `Marios-Paschalidis-Thesis/bsard-rag-thesis-data` (subset `rq2`) into the shared RQ2 data root before step 1:
>
> ```bash
> python scripts/download_data.py        # downloads into $RQ2_DATA_DIR (default <repo>/data)
> ```

### Step 1 — Link `data/` into the shared source data

`data/` is a link (directory junction on Windows, symlink on POSIX) into the **shared source data** slice of the RQ2 data root (`<data root>/shared_source`) — the single authoritative location for all raw input files used by the entire RQ2 pipeline. Wire it up once per clone from the project root:

```bash
python scripts/setup/link_data.py
```

After this step, `data/pdfs/` and `data/csv/` should be accessible.

### Step 2 — Create and activate the virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Step 3 — Install dependencies

```powershell
pip install -r requirements.txt
```

### Step 4 — Install sibling package

```powershell
pip install -e ..\RQ2_T01_SHARED
```

---

## Storage

### What lives in the Git repository

- All Python source files (`src/data_loader/`)
- `requirements.txt`, `.gitignore`, `README.md`, `PROJECT_CONTEXT.md`

### What lives in the shared data root only (via `data/`)

`data/` links into the **shared source folder** (`<data root>/shared_source`) — unlike other sub-projects, this link is shared across the pipeline:

| Path | Contents | Reason |
|---|---|---|
| `data/pdfs/` | Source PDF files (Belgian statutory documents) | Large binary files |
| `data/csv/MyDocuments.csv` | Document metadata + Azure DI extraction results | Source dataset |
| `data/csv/DocumentDefinitions.csv` | Defined terms per document from Azure Document Intelligence | Source dataset |
| `data/csv/pdf_document_map.csv` | Manual mapping of PDF filename → `DocumentId` | Needed because `DocumentPdfUrl` in MyDocuments uses the original upload filename, not the local filename |

> **PDF naming convention:** All PDFs in `data/pdfs/` follow the pattern `{YYYY_MM_DD}_{code}.pdf` (e.g., `2004_05_27_2004A27101.pdf`), derived from the original source filenames by stripping the `img_l_pdf_` prefix and `_F` suffix. Do **not** rely on file size matching to link PDFs to `DocumentId` — use `pdf_document_map.csv` instead.

> Other sub-projects access source data by configuring their `config.yaml` to point at this project's `data/` path, e.g.: `pdf_dir: ../RQ2_T02_DATA_LOADER/data/pdfs`.

---

## Project context

See [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) for full design specification.

## Public API

Install as a sibling dependency (required by T03–T08):

```powershell
pip install -e ..\RQ2_T02_DATA_LOADER
```

### Central data contracts

`DocumentBundle` is the primary input to every arm. It is defined here and passed through the entire pipeline:

```python
from data_loader.models import DocumentBundle, Article, Chunk, RetrievalResult
```

| Type | Description |
|---|---|
| `DocumentBundle` | One statutory document — holds the PDF path, list of `Article` objects, and document metadata |
| `Article` | One article within a document — holds `article_id`, raw text, hierarchy + Azure DI fields, optional `bsard_id` |
| `Chunk` | One sliding-window chunk (T03 only) — holds text, char offsets, and token offsets |
| `RetrievalResult` | The cross-arm result record returned by every arm and consumed by T07 — `query_id`, `query_text`, `ranked_items`, `method`, `cost`, optional `trace` |

### Loading documents

The orchestrator calls the package-level entry point:

```python
from data_loader import load_documents

bundles: list[DocumentBundle] = load_documents(config, articles_csv_path="…/articles.csv")
# discover_documents(config) matches PDFs → DocumentId; pdf_loader fills raw_text;
# azure_loader fills articles (only when articles_csv_path is given and has_azure_extraction).
```

Discovery alone (no article population) is available via `auto_discover.discover_documents(config)`.

Other projects configure their `config.yaml` to point at this project's `data/` path:

```yaml
pdf_dir: ../RQ2_T02_DATA_LOADER/data/pdfs
csv_dir: ../RQ2_T02_DATA_LOADER/data/csv
```

## Module

```
src/data_loader/
├── __init__.py        # load_documents() entry point + re-exports
├── auto_discover.py   # PDF → DocumentId matching via pdf_document_map.csv
├── pdf_loader.py      # PyMuPDF raw text extraction
├── azure_loader.py    # Article objects from the articles CSV + bsard_id lookup (BSARD DB)
├── bootstrap.py       # CLI: write per-PDF discovery.json stamp (python -m data_loader.bootstrap)
└── models.py          # DocumentBundle, Article, Chunk, RetrievalResult dataclasses
```

## Part of

RQ2 pipeline — Belgian Statutory Article Retrieval (BSARD) thesis project.
