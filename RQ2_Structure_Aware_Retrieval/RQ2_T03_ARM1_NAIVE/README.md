# RQ2_T03_ARM1_NAIVE

Arm 1: naive sliding-window chunking with hybrid retrieval (dense + sparse + RRF).

---

## Setup

> **Prerequisite:** the PDF cache (raw text, spans, chunks, FAISS/BM25 indices) and the BSARD corpus DB are not in git. Pull them from the companion Hugging Face dataset `Marios-Paschalidis-Thesis/bsard-rag-thesis-data` (subset `rq2`) into the shared RQ2 data root before step 1:
>
> ```bash
> python scripts/download_data.py        # downloads into $RQ2_DATA_DIR (default <repo>/data)
> ```

### Step 1 — Link `data/` into the shared `pdf_cache/`

`data/` is a link (directory junction on Windows, symlink on POSIX) into the **shared** PDF cache root (`<data root>/pdf_cache`). The same target is reused by `RQ2_T07_EVALUATION/data/` so both sub-projects address the same per-PDF folders. Wire it up once per clone from the project root:

```bash
python scripts/setup/link_data.py
```

After this step, `data/<doc_id>/` resolves to a folder under the shared `pdf_cache/` tree.

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

- All Python source files (`src/arm1_naive/`)
- `scripts/precompute_pdf.py`, `scripts/precompute_retrieval.py`, `scripts/build_t08_bundle.py`, `scripts/query_cached.py`, `scripts/verify_chunking.py`
- `inspect_chunking.py`, `requirements.txt`, `.gitignore`, `README.md`, `PROJECT_CONTEXT.md`

### What lives in the shared data root only (via `data/`)

The `data/` link targets the shared `pdf_cache/` root; `RQ2_T07_EVALUATION/data/` points at the same target. Per-PDF layout:

```
data/<doc_id>/
├── pdf_meta.json                       T03 — pdf_sha256, n_pages, n_chars, extracted_at
├── raw_text.txt                        T03 — PyMuPDF extraction (UTF-8 LF)
├── article_spans.json                  T03 — locate_articles() output
├── configs/<config_hash>/
│   ├── manifest.json                   T03 — config integrity anchor
│   ├── chunks.json                     T03 — canonical chunks
│   ├── faiss.index, faiss_meta.json    T03 — FAISSStore.save()
│   ├── bm25.pkl                        T03 — BM25Store.save()
│   └── eval/                           T07 — chunk-bsard weights (Phase 2)
├── question_projections/               T07 — qset → PDF projections (Phase 2)
├── results/<config_hash>/
│   ├── <run_label>.jsonl               T03 — one RetrievalResult per line (run_arm1)
│   └── retrieval_pool/                 T03 — top-K hybrid candidate pool
│       ├── <qset_hash>.jsonl                — one row per question, K candidates
│       └── <qset_hash>.manifest.json        — config + qset + top_k integrity anchor
└── t08_bundles/<qset_hash>/            T03 — per-question viewer bundles (build_t08_bundle.py)
    ├── q<qid>.json                          — DocumentBundle + RetrievalResult for T08
    └── manifest.json                        — selection filters + counts + provenance
```

`config_hash` is the first 12 hex of `sha256(canonical_json({pdf_sha256, tokenizer_name, embedding_model, chunking}))`. Changing chunk size / model / tokenizer creates a new sibling folder; old configs are preserved.

### Cache invalidation

Manifest-driven; no filename heuristics:

| Trigger | Effect |
|---|---|
| `pdf_sha256` mismatch | `validate()` fails → whole doc tree (raw text, spans, all configs) is rebuilt on next call |
| `tokenizer_name` / `embedding_model` change | New `config_hash` → new sibling folder; old preserved |
| `chunking.*` params change | New `config_hash` → new sibling folder; old preserved |
| `bsard_db_fingerprint` change | `article_spans.json` invalidated (article→bsard_id assignments may have shifted) |
| Unknown `schema_version` in any manifest | Reader rejects loudly; no silent fallback |

Editing a manifest (or deleting `pdf_meta.json`) is the canonical way to force a rebuild. `--force` on the scripts and `force_reindex=True` in `run_arm1` short-circuit validation.

### Cross-clone caveat

Links are per-clone. On a new machine, run `python scripts/download_data.py` to populate the shared data root, then `python scripts/setup/link_data.py` to recreate the `data/` link with the same target.

---

## Project context

See [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) for the full design specification. The cache
architecture is implemented in [src/arm1_naive/cache.py](src/arm1_naive/cache.py) (the
per-PDF layout and invalidation rules are summarised under **Storage** above).

## Public API

Install as a sibling dependency (used by T00 orchestrator and T08 viewer):

```powershell
pip install -e ..\RQ2_T03_ARM1_NAIVE
```

### Calling from the orchestrator

```python
from arm1_naive.retriever import run_arm1

# data_dir is the cache root (the data/ junction target).
results = run_arm1(
    bundle=bundle,                              # DocumentBundle from T02
    embedding_model=embedding_model,
    tokenizer=embedding_model.tokenizer,
    db_path=Path(".../bsard_corpus.db"),
    index_dir=Path("data"),                     # cache root
    config=config,                              # see config keys below
    force_reindex=False,
    run_label=None,                             # optional; default is UTC timestamp
)
```

`run_arm1` reads from cache when `configs/<config_hash>/manifest.json` validates and skips extract / locate / chunk / embed entirely. Per-run `RetrievalResult` lists are written to `data/<doc_id>/results/<config_hash>/<run_label>.jsonl`.

Required config keys:

```yaml
arm1:
  strategy: sliding_window     # or recursive
  window_size: 512             # sliding_window only
  stride: 256                  # sliding_window only
  max_tokens: 512              # recursive only
  retrieval_top_k: 100
  top_k: 100
models:
  embedding_model: intfloat/multilingual-e5-large-instruct
  tokenizer: intfloat/multilingual-e5-large-instruct  # optional; defaults to embedding_model
```

### CLI scripts

The full preprocessing chain for one PDF (run from `RQ2_T03_ARM1_NAIVE/` with the venv active):

```powershell
# 1. Populate the corpus-side cache (raw text, spans, chunks, FAISS, BM25):
python scripts/precompute_pdf.py --pdf 1804_03_21_1804032150.pdf

# 2. Precompute the per-(config, qset) top-K hybrid retrieval pool — raw
#    input for downstream evaluation methodologies. One pool file per qset
#    (e.g. bsard_test, bsard_train, custom subsets):
python scripts/precompute_retrieval.py --pdf 1804_03_21_1804032150.pdf `
    --qset ..\RQ2_T07_EVALUATION\ground_truth\bsard_test.json `
           ..\RQ2_T07_EVALUATION\ground_truth\bsard_train.json `
    --top-k 200

# 3. Pure retrieval against the populated cache (full hybrid pipeline):
python scripts/query_cached.py --pdf 1804_03_21_1804032150.pdf --query "Je me marie ..."

# 4. Bake per-question bundles for the T08 viewer from the cached pool +
#    question projection (no models loaded — JSON in / JSON out). Filter on
#    recall_ceiling / gt_in_pdf or pass --question-ids to pick specific qids:
python scripts/build_t08_bundle.py --pdf 1804_03_21_1804032150.pdf `
    --qset ..\RQ2_T07_EVALUATION\ground_truth\bsard_train.json `
    --question-ids 192
```

`precompute_retrieval.py` skips on cache hit (`<qset_hash>.manifest.json` matches `config_hash`, `qset_hash`, `top_k`, and `embedding_model`); pass `--force` to override. Each row in the resulting JSONL preserves per-retriever rank + score (dense, sparse, fused) so downstream tools can re-rank, ablate, or evaluate without re-running retrieval.

`build_t08_bundle.py` writes `data/<doc_id>/t08_bundles/<qset_hash>/q<qid>.json`, one self-contained bundle per selected question (full pool baked, top-K filtered at render time by the T08 slider). Idempotent: existing bundles whose `_provenance` matches the current `(config_hash, qset_hash, n_pool_candidates)` are skipped; pass `--force` to rebuild. Requires `precompute_retrieval.py` and `RQ2_T07_EVALUATION/scripts/precompute_question_projection.py` to have run first for the same `(pdf, qset)` pair.

Run any of these with `--help` for the full flag list (strategy, window/stride/max-tokens, embedding model, top-k, etc.).

### Inspection

`inspect_chunking.py` is cache-aware — it loads `raw_text.txt` and `article_spans.json` from the cache when valid, prints `[cache hit]` / `[cache miss]` per stage, and supports `--use-cached-chunks` to additionally skip tokenizer load + chunking when chunks for the same `config_hash` are already on disk.

```powershell
python inspect_chunking.py                            # default PDF + Q
python inspect_chunking.py --use-cached-chunks        # fast path
```

## Module

```
src/arm1_naive/
├── __init__.py
├── cache.py        # PDF-centric experiment cache (CacheRoot / PdfCache / ConfigCache)
├── chunker.py      # Sliding-window + recursive chunking + BSARD article location
├── indexer.py      # FAISS + BM25 index construction
└── retriever.py    # Cache-first retrieval pipeline (dense + sparse + RRF)
```

## Part of

RQ2 pipeline — Belgian Statutory Article Retrieval (BSARD) thesis project.
