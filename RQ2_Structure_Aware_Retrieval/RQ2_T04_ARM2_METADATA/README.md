# RQ2_T04_ARM2_METADATA

Arm 2A: enriched article embeddings with LLM-based metadata extraction and filtering/boosting.

---

## Setup

> **Prerequisite:** the AzureDI dump, the BSARD corpus DB, and the prebuilt indices are not in git. Pull them from the companion Hugging Face dataset `mpaschalidis/bsard-rag-thesis-data` (subset `rq2`) into the shared RQ2 data root before step 1:
>
> ```bash
> python scripts/download_data.py        # downloads into $RQ2_DATA_DIR (default <repo>/data)
> ```

### Step 1 — Link `data/` into the shared data root

`data/` is a link (directory junction on Windows, symlink on POSIX) into this project's slice of the shared RQ2 data root (`<data root>/RQ2_T04_ARM2_METADATA`). Wire it up once per clone from the project root:

```bash
python scripts/setup/link_data.py
```

After this step, `data/indices/` and `data/results/` should be accessible.

### Step 2 — Create and activate the virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Step 3 — Install dependencies

```powershell
pip install -r requirements.txt
```

### Step 4 — Install sibling packages

```powershell
pip install -e ..\RQ2_T01_SHARED
pip install -e ..\RQ2_T02_DATA_LOADER
```

---

## Storage

### What lives in the Git repository

- All Python source files (`src/arm2_metadata/`)
- `requirements.txt`, `.gitignore`, `README.md`, `PROJECT_CONTEXT.md`

### What lives in the shared data root only (via `data/`)

The `data/` directory links into `<data root>/RQ2_T04_ARM2_METADATA/`:

| Path | Contents | Reason |
|---|---|---|
| `data/indices/<doc_id>/<variant>/faiss.index` | FAISS index per enrichment variant (raw, enriched, full, terms, summary) | Large, rebuilt from source |
| `data/indices/<doc_id>/<variant>/faiss_meta.json` | Article metadata for FAISS results | Auto-generated |
| `data/indices/<doc_id>/<variant>/bm25.pkl` | BM25 index pickle per variant | Auto-generated |
| `data/results/` | `RetrievalResult` JSONs per document per variant | Auto-generated |

---

## Project context

See [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) for full design specification.

## Public API

Install as a sibling dependency (used by T00 and T08):

```powershell
pip install -e ..\RQ2_T04_ARM2_METADATA
```

### Entry point

The whole pipeline (load AzureDI corpus → enrich → index → retrieve → persist) runs through
a single keyword-only function in `retriever.py`. It loads the AzureDI dump directly (not a
T02 `DocumentBundle`) and is cache-aware per `(doc_id, variant, unit)`:

```python
from arm2_metadata.retriever import run_arm2_metadata

results = run_arm2_metadata(                        # -> list[RetrievalResult]
    azuredi_dir=Path("azuredi"),                   # AzureDI dump (link into the data root)
    pdf_document_map_csv=Path(".../pdf_document_map.csv"),
    bsard_db_path=Path(".../bsard_corpus.db"),
    pdf_path=Path(".../<doc>.pdf"),                # pins pdf_sha256 in the cache manifest
    embedding_model=embedding_model, tokenizer=embedding_model.tokenizer, llm=llm_client,
    index_root=Path("data"), config=config,
    doc_id="1804_03_21_1804032150", azure_doc_id=5,
    variant="full", unit="node",                   # unit: "node" | "article"
    force_reindex=False,
)
```

In practice the arm is driven by its own scripts/notebook rather than the T00 orchestrator
(see [scripts/precompute_t04_indices.py](scripts/precompute_t04_indices.py),
[scripts/compare_t03_vs_t04.py](scripts/compare_t03_vs_t04.py), and
[notebooks/azure_t04_precompute_run.ipynb](notebooks/azure_t04_precompute_run.ipynb)).
Indices are persisted under `data/<doc_id>/configs/<config_hash>/` and reused on cache hits.

## Module

```
src/arm2_metadata/
├── __init__.py
├── azuredi_loader.py    # Read MyDocuments / Definitions / VectorDB_*.json → AzureNode list
├── bsard_link.py        # AzureNode → bsard_id resolver (uses T03's _ART_RE + DB lookup)
├── enricher.py          # Six variants: raw/enriched/summary/filtered/full/terms
├── indexer.py           # FAISS + BM25 over node or article units
├── query_extractor.py   # Term-dictionary + regex + LLM signal extraction
├── boost.py             # apply_filters_and_boosts: hard filter + soft multipliers
├── cache.py             # Doc / config caches (mirrors T03 layout under data/<doc_id>/)
└── retriever.py         # run_arm2_metadata: full pipeline + persistence
```

## Quick start

> The four AzureDI files live behind `azuredi/`, a link into the shared data
> root's `AzureDI/` slice (created by `python scripts/setup/link_data.py`).

```powershell
# 1. Inspect what survived the AzureDI loader filter (drops doc_id=1 always).
python -m arm2_metadata.azuredi_loader --inspect

# 2. Inspect BSARD coverage (compares against T03 article_spans.json).
python -m arm2_metadata.bsard_link --inspect

# 3. Run a single (unit, variant) end-to-end and persist the indices + results.
python -c "from arm2_metadata.retriever import run_arm2_metadata; ..."

# 4. Compare T03 baseline against every T04 variant on the doc-2 question subset.
python scripts/compare_t03_vs_t04.py --smoke -v \
    --embedding-model intfloat/multilingual-e5-small
```

## Variants

| variant   | unit  | embedding text                                     | query-time filter |
|-----------|-------|----------------------------------------------------|-------------------|
| raw       | both  | body only                                          | no                |
| enriched  | both  | doc-context header + body                          | no                |
| summary   | node  | header + AzureDI English summary + keywords + body | no                |
| filtered  | both  | body only                                          | yes               |
| full      | both  | header + body                                      | yes               |
| terms     | both  | header + body + defined-terms block                | yes               |

The `summary` variant is T04-only (it consumes the AzureDI-produced
`content_summary` + `keywords` fields — free metadata that T03 doesn't have).

## Outputs (under `data/<doc_id>/`)

| Path                                                  | Contents                                       |
|-------------------------------------------------------|------------------------------------------------|
| `manifest.json`                                       | Doc-level fingerprint (azuredi files, pdf)     |
| `configs/<config_hash>/manifest.json`                 | Per-config invariants (variant, unit, models)  |
| `configs/<config_hash>/{faiss.index,faiss_meta.json}` | FAISS dense index for this variant             |
| `configs/<config_hash>/bm25.pkl`                      | BM25 sparse index for this variant             |
| `configs/<config_hash>/enrichment_stats.json`         | Truncation telemetry per build                 |
| `results/<config_hash>/<run_label>.jsonl`             | One `RetrievalResult` per query, per run       |
| `comparison_t03_vs_t04.csv`                           | Aggregate comparison table (one row / method). Columns: `R@10`, `R@100`, `MRR@10`, `nDCG@10`, `latency_ms`, `n_queries`, plus `best:<metric>` markers. **`Cw/R@10` and `Cw/R@100`** appear when T07's `cosine_weighted_recall_at_k` is importable AND the upstream JSONL has a non-null `extraction_cosine` for at least one GT item in the subset (per CN-T03-002 / CN-T07-001, both closed 2026-05-01). |
| `comparison_per_query.json`                           | GT + per-query metrics for inspection. When cosine columns are populated, also contains a `cosine_ground_truth` block (`{qid: {bsard_id: cosine \| null}}`). |

`config_hash` is computed via T03's `compute_config_hash` over
`(pdf_sha256_or_dump_fingerprint, tokenizer_name, embedding_model_name,
{variant, unit, retrieval_top_k, rerank_top_k (frozen=10 for hash compat after
cross-encoder removal), max_tokens, boost})`.

## Part of

RQ2 pipeline — Belgian Statutory Article Retrieval (BSARD) thesis project.
