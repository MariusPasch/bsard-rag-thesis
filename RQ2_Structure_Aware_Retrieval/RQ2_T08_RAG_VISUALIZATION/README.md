# RQ2_T08_RAG_VISUALIZATION

Local Streamlit viewer for inspecting chunks, articles, and retrieval results on the original PDF page layout. A debugging and thesis-figure tool — read-only with respect to the pipeline.

---

## Setup

### Step 1 — Download the data bundle and wire `data/`

`data/` is a link into the shared data root (`$RQ2_DATA_DIR`, default
`<repo>/data`), which is populated from the companion Hugging Face dataset
`Marios-Paschalidis-Thesis/bsard-rag-thesis-data` (subset `rq2`). Run once from the
component root:

```powershell
python scripts/download_data.py
python scripts/setup/link_data.py
```

After this step, `data\exports\` should be accessible.

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
pip install -e ..\RQ2_T03_ARM1_NAIVE
pip install -e ..\RQ2_T04_ARM2_METADATA
pip install -e ..\RQ2_T05_ARM2_PAGEINDEX
```

---

## Storage

### What lives in the Git repository

- All Python source files (`src/visualization/`) and `demo_bm25.py`
- `requirements.txt`, `.gitignore`, `README.md`, `PROJECT_CONTEXT.md`
- Tests (`tests/`)

### What lives in the data bundle only (via `data/`)

The `data/` directory is a link into the shared data root, resolving to:
```
<data root>/RQ2_T08_RAG_VISUALIZATION/
```

| Path | Contents | Reason |
|---|---|---|
| `data/exports/` | PNG / SVG thesis figures exported from the viewer | Auto-generated, referenced in thesis |

> This viewer is **read-only with respect to the pipeline** — it never writes to indices, ground-truth stores, or result directories. Indices and source data are read directly from sibling projects' `data/` paths, passed in as `launch()` arguments (or baked into a T03 bundle JSON).

---

## Usage

### Standalone app

```bash
streamlit run src/visualization/app.py
```

### Inspecting one question against the cached retrieval pool

T03's `scripts/build_t08_bundle.py` bakes per-question viewer bundles from the
cached `retrieval_pool/<qset_hash>.jsonl` and the T07 question projection.
Each bundle holds the full pool (top-200 by default); the **Top-K** slider in
the retrieval-mode sidebar filters the rendered chunk rectangles in real time
without requiring a Retrieve click.

```bash
streamlit run src/visualization/app.py -- --bundle-json \
    ../RQ2_T03_ARM1_NAIVE/data/1804_03_21_1804032150/t08_bundles/483beb56a47b/q192.json
```

Move the slider between 1 and 200 to see how candidate coverage grows as the
pool deepens. GT articles for the question (the projection's `gt_in_pdf`)
overlay in the GT colour and persist independently of the slider.

### Python API — calling `launch()` from another project

Install this package as a sibling dependency first:

```powershell
pip install -e ..\RQ2_T08_RAG_VISUALIZATION
```

Then call `launch()` from any arm or orchestrator script:

```python
from visualization.launcher import launch

# Review mode — inspect chunks of a document
proc = launch(bundle, mode="review", arm="arm1")

# Retrieval mode — overlay GT + retrieved + overlap on the PDF
proc = launch(
    bundle,
    mode="retrieval",
    arm="arm1",
    retrieval_results=results,       # list[RetrievalResult] from run_arm1()
    db_path=Path("data/bsard_corpus.db"),
    arm_index_dir=Path("data/indices"),
    port=8501,
    open_browser=True,
)

# Block until the user closes the tab, then clean up
try:
    proc.wait()
finally:
    proc.terminate()
```

`launch()` serialises the bundle to a temporary JSON side-file, spawns
`streamlit run app.py` in a subprocess, and returns the `Popen` handle.
The viewer is **read-only** — it never writes to indices, the ground-truth
DB, or any result file.

#### `launch()` parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `bundle` | `DocumentBundle` | required | Document to visualise |
| `mode` | `"review"` \| `"retrieval"` | `"review"` | Viewer mode |
| `arm` | `str` | `"arm1"` | `"arm1"` \| `"2A-full"` \| `"2A-filtered"` \| `"2B"` |
| `retrieval_results` | `list[RetrievalResult]` \| `None` | `None` | Pre-computed results for retrieval mode |
| `queries` | `list[Query]` \| `None` | `None` | Pre-populate query dropdown |
| `arm_index_dir` | `Path` \| `None` | `None` | Root of `data/indices/` for fresh retrieval |
| `db_path` | `Path` \| `None` | `None` | Path to `bsard_corpus.db` |
| `config` | `dict` \| `None` | `None` | Optional config dict (reserved; unused by the viewer) |
| `port` | `int` | `8501` | Streamlit server port |
| `open_browser` | `bool` | `True` | Open browser automatically |

---

## Project context

See [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) for full design specification, including the tool survey, coordinate-mapping approach, color scheme, and fallback plan.

## Module

```
src/visualization/
├── __init__.py
├── app.py                   # Streamlit entry point (streamlit run app.py)
├── launcher.py              # Public launch(): spawns `streamlit run app.py` in a subprocess
├── coord_mapper.py          # char-offset → (page, bbox) via PyMuPDF word index
├── region_adapter.py        # Chunk / Article / RetrievalResult → HighlightRegion
├── annotation_builder.py    # streamlit-pdf-viewer annotation dicts + overlap layering
├── query_loader.py          # Loads queries + GT article_ids from ground-truth store
├── trace_renderer.py        # Formats PageIndex trace for side panel
├── index_cache.py           # Loads cached FAISS+BM25 bundles from sibling data/indices/
├── t03_loader.py            # Adapt T03 Arm 1 chunks/results for the viewer
├── t04_loader.py            # Adapt T04 Arm 2A node/article results for the viewer
├── t05_loader.py            # Adapt T05 Arm 2B tree + navigation trace for the viewer
└── ui/
    ├── __init__.py
    ├── layout.py            # Three-column layout (sidebar | PDF | text panels)
    ├── sidebar_review.py    # Chunk / article list panel
    ├── sidebar_retrieval.py # Query box + ranked results
    ├── text_panels.py       # Question, selected item, GT, trace panels
    ├── tree_explorer.py     # T05 ToC-tree explorer panel
    └── legend.py            # Color legend
```

## Part of

RQ2 pipeline — Belgian Statutory Article Retrieval (BSARD) thesis project.
