# RQ2_T05_ARM2_PAGEINDEX

Arm 2B: ToC-tree construction and LLM-guided hierarchical navigation (vectorless).

---

## Setup

### Step 1 — Download the data bundle and wire `data/`

Large artefacts are not committed to git. They live in the companion Hugging Face dataset `mpaschalidis/bsard-rag-thesis-data` (subset `rq2`) and download into the data root (env `RQ2_DATA_DIR`, default `<repo>/data`). Run once from the component root:

```powershell
python scripts/download_data.py
python scripts/setup/link_data.py
```

After this step, `data\trees\` and `data\results\` should be accessible (linked to `<data root>/RQ2_T05_ARM2_PAGEINDEX`).

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

T05 reuses T04's `bsard_link` for node→bsard_id mapping and T04 transitively imports T03's `arm1_naive.chunker`, so all five must be on the path.

```powershell
pip install -e ..\RQ2_T01_SHARED
pip install -e ..\RQ2_T02_DATA_LOADER
pip install -e ..\RQ2_T03_ARM1_NAIVE
pip install -e ..\RQ2_T04_ARM2_METADATA
pip install -e .
```

---

## Storage

### What lives in the Git repository

- All Python source files (`src/arm2_pageindex/`)
- `requirements.txt`, `.gitignore`, `README.md`, `PROJECT_CONTEXT.md`

### What lives in the data bundle only (via `data/`)

This project's `data/` is wired (via `scripts/setup/link_data.py`) to the data root:
```
<data root>/RQ2_T05_ARM2_PAGEINDEX/
```

| Path | Contents | Reason |
|---|---|---|
| `data/trees/<doc_id>.json` | Serialized ToC tree (ToCNode hierarchy with LLM-generated summaries) | Expensive to rebuild (one LLM call per node) |
| `data/results/` | `RetrievalResult` JSONs per document — includes full reasoning traces | Large due to traces |

> Trees are expensive to build. Cache them in `data/trees/` and reuse across runs with `--skip-indexing`.

---

## Project context

See [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) for full design specification.

## Public API

```python
from arm2_pageindex import (
    build_law_tree, save_law_tree, load_law_tree, compose_corpus_tree,
    run_arm2b, run_subset, NavigatorConfig,
)
```

### Per-query call

```python
result = run_arm2b(
    query="...",                                    # French legal question
    query_id="42",
    tree=tree,                                      # ToCNode (single-law or composed corpus)
    llm=llm,                                        # shared.llm.LLMClient
    cfg=NavigatorConfig(max_iterations=2),
)
# result is data_loader.models.RetrievalResult — ranked_items, cost, trace.
# Each ranked_items[*]['metadata']['bsard_id'] is a non-null int (CN-T05-001).
```

### Batch run

```python
results = run_subset(
    tree=tree,
    queries=[{"query_id": "...", "query_text": "..."}, ...],
    llm=llm,
    cfg=cfg,
    out_dir=Path("data/results/<run_name>/"),       # idempotent on per-query JSON
)
```

The tree is **deterministic** (no LLM calls at build time) and cached on disk with `pdf_sha256` + `tree_builder_version` pinned. Rebuild only fires when the source PDF changes or the builder version bumps.

`RetrievalResult.trace` carries the full navigation path (per-step prompt sizes, parsed responses, latencies) for T08 visualisation; cost dict carries `llm_calls`, `tokens_in`, `tokens_out`, `latency_ms`, `parse_failures`, `exit_reason`, `iterations` for T07's cost tracker.

## Module

```
src/arm2_pageindex/
├── __init__.py
├── tree_builder.py   # Deterministic ToC tree from AzureNode + Article
├── navigator.py      # Per-query nav loop (law -> chapter -> article -> evaluate)
├── prompts.py        # French JSON-strict prompts for each step
├── retriever.py      # run_arm2b() — wraps navigator + emits RetrievalResult
└── pipeline.py       # run_subset() — batch runner with per-query JSON cache
```

## Run sequence

```text
local: scripts/build_tree.py           (data/trees/doc_<doc_id>.json, cached)
local: evaluation.question_subsets     (curated GT subset JSON)
local: scripts/prepare_azure_bundle.py (zips trees + queries + GT + manifest)
   |
   v upload to Azure Blob
   |
azure: notebooks/azure_t05_pageindex_run.ipynb
       (Ollama + LLaMA 3.1 8B on GPU; 4 LLM calls/query baseline,
        12-15 with iterations; idempotent re-run; uploads results to blob)
   |
   v download from Azure Blob
   |
local: notebooks/local_t05_eval_and_compare.ipynb
       (T07-style metrics, cross-arm table, stratification, sig tests, plot)
```

## Part of

RQ2 pipeline — Belgian Statutory Article Retrieval (BSARD) thesis project.
