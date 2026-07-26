# RQ2_T00_ORCHESTRATOR

CLI entry point, config loading, and pipeline coordination across all sub-projects.

---

## Setup

> **Prerequisite:** the large data artefacts (caches, indices, the BSARD corpus DB, ground truth) are not in git. Pull them from the companion Hugging Face dataset `mpaschalidis/bsard-rag-thesis-data` (subset `rq2`) into the shared RQ2 data root before step 1:
>
> ```bash
> python scripts/download_data.py        # downloads into $RQ2_DATA_DIR (default <repo>/data)
> ```

### Step 1 — Link `data/` into the shared data root

`data/` is a link (directory junction on Windows, symlink on POSIX) into this project's slice of the shared RQ2 data root (`RQ2_DATA_DIR`, default `<repo>/data`, where `<repo>` is `RQ2_Structure_Aware_Retrieval`). Wire it up once per clone from the project root:

```bash
python scripts/setup/link_data.py
```

After this step, `data/results/` and `data/logs/` should be accessible.

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
pip install -e ..\RQ2_T07_EVALUATION
```

---

## Storage

### What lives in the Git repository

- All Python source files (`src/orchestrator/`)
- `config.yaml` — pipeline configuration (paths use `data/` junction, no hardcoded local paths)
- `requirements.txt`, `.gitignore`, `README.md`, `PROJECT_CONTEXT.md`

### What lives in the shared data root only (via `data/`)

The `data/` directory links into `<data root>/RQ2_T00_ORCHESTRATOR/`:

| Path | Contents | Reason |
|---|---|---|
| `data/results/` | Final experiment result JSONs (all arms, all documents) | Large, auto-generated |
| `data/logs/` | Execution logs per run | Large, auto-generated |

> Source data (PDFs, CSVs) is owned by **RQ2_T02_DATA_LOADER**. Point `config.yaml` at that project's `data/` path (e.g. `../RQ2_T02_DATA_LOADER/data/`).

---

## Usage

```bash
python src/orchestrator/run_experiment.py                         # all methods, all PDFs
python src/orchestrator/run_experiment.py --methods arm1 2A 2B    # choices: arm1, 2A, 2B
python src/orchestrator/run_experiment.py --pdf 2.pdf             # one or more PDF filenames
python src/orchestrator/run_experiment.py --variants raw full     # restrict Arm 2A variants
python src/orchestrator/run_experiment.py --skip-indexing         # reuse existing indices
python src/orchestrator/run_experiment.py --force-reindex         # rebuild all indices
python src/orchestrator/run_experiment.py --eval-only             # skip retrieval, eval saved results
python src/orchestrator/run_experiment.py --config custom.yaml    # custom config (default: config.yaml)
```

### Per-PDF status surface (read-only)

`orchestrator.status` discovers each sub-project's artefacts for a given PDF, validates
them, and writes a registry under `data/status/<doc_id>.json`. It never invokes or mutates
a sibling project:

```bash
python -m orchestrator.status --doc-id 1804_03_21_1804032150   # one PDF
python -m orchestrator.status --all                            # every PDF in selected_pdfs.json
python -m orchestrator.status --doc-id <id> --project T04 --json
```

---

## Project context

See [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) for full design specification.

## Inter-project interface

**This project is a CLI runner, not a library.** Other projects do not import from it.

It imports from every arm (T01–T05 + T07) and coordinates them. The T08 visualization
viewer is launched separately from its own project — see `RQ2_T08_RAG_VISUALIZATION`'s
README; the orchestrator does not spawn it.

### What the orchestrator expects from each arm

Each retrieval arm exposes a top-level entry function that accepts a `DocumentBundle`
(from T02) and returns a list of retrieval-result records (each with a `to_dict()`):

- Arm 1 — `arm1_naive.retriever.run_arm1(bundle, embedding_model, tokenizer, db_path, index_dir, config, force_reindex)`
- Arm 2A — `arm2_metadata.run_metadata_filtering(bundle, config, variant=...)`
- Arm 2B — `arm2_pageindex.run_pageindex(bundle, config)`

The orchestrator calls these per document, then passes all results to T07
(`evaluation.evaluate` / `generate_comparison`). T06_ARM_RESULTS is a post-pipeline
consolidation project and is **not** invoked by the orchestrator.

> **Note:** `run_experiment.py` still imports legacy arm-2 entry names
> (`arm2_metadata.run_metadata_filtering`, `arm2_pageindex.run_pageindex`) that predate the
> arms' current cache-aware APIs (`run_arm2_metadata`, `run_arm2b`). In practice Arms 2A/2B
> are run standalone via each arm's own scripts/notebooks, not end-to-end through this CLI
> (each arm was evaluated through a different path — see T06's `EVALUATION_METHODOLOGY.md`).

## Modules

```
src/orchestrator/
├── __init__.py
├── run_experiment.py    # main pipeline CLI (entry point)
├── status.py            # read-only per-PDF status registry CLI (python -m orchestrator.status)
└── paths.py             # RQ2 root + per-project data-dir resolution helpers
```

## Part of

RQ2 pipeline — Belgian Statutory Article Retrieval (BSARD) thesis project.
