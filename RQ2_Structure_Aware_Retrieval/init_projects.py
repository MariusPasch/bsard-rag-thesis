"""
Bootstrap script for RQ2 pipeline sub-projects.

Creates each sub-project folder with:
  - src/<module>/__init__.py
  - PROJECT_CONTEXT.md  (copied from Project_Contexts_v0.0/)
  - .gitignore
  - README.md
  - requirements.txt
  - .venv/              (per-project virtual environment)
  - git repository      (init + initial commit)

Run from the RQ2_Structure_Aware_Retrieval/ root directory.
Re-running is safe: existing folders are skipped unless --force is passed.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Project definitions
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent

DATA_BASE = os.environ.get("RQ2_DATA_DIR", str(ROOT / "data"))

PROJECTS = [
    {
        "folder": "RQ2_T00_ORCHESTRATOR",
        "module": "orchestrator",
        "context_file": "Project_Contexts_v0.0/RQ2_T00_ORCHESTRATOR_CONTEXT.md",
        "description": "CLI entry point, config loading, and pipeline coordination across all sub-projects.",
        "data_target": r"RQ2_T00_ORCHESTRATOR",
        "data_contents": [
            ("data/results/", "Final experiment result JSONs (all arms, all documents)", "Large, auto-generated"),
            ("data/logs/", "Execution logs per run", "Large, auto-generated"),
        ],
        "data_note": "> Source data (PDFs, CSVs) is owned by **RQ2_T02_DATA_LOADER**. Point `config.yaml` at that project's `data/` path (e.g. `../RQ2_T02_DATA_LOADER/data/`).",
        "sibling_installs": [
            "RQ2_T01_SHARED", "RQ2_T02_DATA_LOADER", "RQ2_T03_ARM1_NAIVE",
            "RQ2_T04_ARM2_METADATA", "RQ2_T05_ARM2_PAGEINDEX",
            "RQ2_T07_EVALUATION",
        ],
        "requirements": [
            "# Core",
            "pandas>=2.0",
            "numpy>=1.26",
            "scipy>=1.11",
            "pyyaml>=6.0",
            "tqdm>=4.66",
            "",
            "# PDF processing",
            "PyMuPDF>=1.23",
            "",
            "# Embeddings & retrieval",
            "sentence-transformers>=2.7",
            "faiss-cpu>=1.8",
            "rank-bm25>=0.2",
            "transformers>=4.40",
            "",
            "# Agentic framework",
            "langgraph>=0.1",
            "langchain>=0.2",
            "langchain-community>=0.2",
            "",
            "# LLM backend (Ollama must be running separately)",
            "ollama>=0.2",
        ],
    },
    {
        "folder": "RQ2_T01_SHARED",
        "module": "shared",
        "context_file": "Project_Contexts_v0.0/RQ2_T01_SHARED_CONTEXT.md",
        "description": "Shared components: embedding model, LLM wrapper, FAISS/BM25 stores.",
        "data_target": None,  # pure library — no large files
        "data_contents": [],
        "data_note": None,
        "sibling_installs": [],
        "requirements": [
            "# Core",
            "numpy>=1.26",
            "tqdm>=4.66",
            "",
            "# Embeddings",
            "sentence-transformers>=2.7",
            "transformers>=4.40",
            "",
            "# Vector & lexical stores",
            "faiss-cpu>=1.8",
            "rank-bm25>=0.2",
            "",
            "# LLM backend (Ollama must be running separately)",
            "ollama>=0.2",
        ],
    },
    {
        "folder": "RQ2_T02_DATA_LOADER",
        "module": "data_loader",
        "context_file": "Project_Contexts_v0.0/RQ2_T02_DATA_LOADER_CONTEXT.md",
        "description": "Auto-discovery of PDF↔CSV pairs, raw text extraction, Article dataclass construction.",
        "data_target": r"shared_source",
        "data_contents": [
            ("data/pdfs/", "Source PDF files (Belgian statutory documents)", "Large binary files"),
            ("data/csv/MyDocuments.csv", "PDF metadata + Azure DI URL mapping", "Source dataset"),
            ("data/csv/DocumentDefinitions.csv", "Article-level extractions from Azure Document Intelligence", "Source dataset"),
        ],
        "data_note": "> Other sub-projects access source data by configuring their `config.yaml`, e.g.: `pdf_dir: ../RQ2_T02_DATA_LOADER/data/pdfs`.",
        "sibling_installs": ["RQ2_T01_SHARED"],
        "requirements": [
            "# Core",
            "pandas>=2.0",
            "numpy>=1.26",
            "tqdm>=4.66",
            "",
            "# PDF processing",
            "PyMuPDF>=1.23",
        ],
    },
    {
        "folder": "RQ2_T03_ARM1_NAIVE",
        "module": "arm1_naive",
        "context_file": "Project_Contexts_v0.0/RQ2_T03_ARM1_NAIVE_CONTEXT.md",
        "description": "Arm 1: naive sliding-window chunking with hybrid retrieval (dense + sparse + RRF).",
        "data_target": r"RQ2_T03_ARM1_NAIVE",
        "data_contents": [
            ("data/indices/<doc_id>/faiss.index", "FAISS IndexFlatIP over sliding-window chunks", "Large, rebuilt from source"),
            ("data/indices/<doc_id>/faiss_meta.json", "Chunk metadata for FAISS results", "Auto-generated"),
            ("data/indices/<doc_id>/bm25.pkl", "BM25 index pickle", "Auto-generated"),
            ("data/results/", "RetrievalResult JSONs per document", "Auto-generated"),
        ],
        "data_note": None,
        "sibling_installs": ["RQ2_T01_SHARED", "RQ2_T02_DATA_LOADER"],
        "requirements": [
            "# Core",
            "numpy>=1.26",
            "tqdm>=4.66",
            "",
            "# Tokenizer (for chunking)",
            "transformers>=4.40",
            "",
            "# Sibling packages — install with: pip install -e ../RQ2_T01_SHARED",
            "# pip install -e ../RQ2_T02_DATA_LOADER",
        ],
    },
    {
        "folder": "RQ2_T04_ARM2_METADATA",
        "module": "arm2_metadata",
        "context_file": "Project_Contexts_v0.0/RQ2_T04_ARM2_METADATA_CONTEXT.md",
        "description": "Arm 2A: enriched article embeddings with LLM-based metadata extraction and filtering/boosting.",
        "data_target": r"RQ2_T04_ARM2_METADATA",
        "data_contents": [
            ("data/indices/<doc_id>/<variant>/faiss.index", "FAISS index per enrichment variant (raw, enriched, full, terms, summary)", "Large, rebuilt from source"),
            ("data/indices/<doc_id>/<variant>/faiss_meta.json", "Article metadata for FAISS results", "Auto-generated"),
            ("data/indices/<doc_id>/<variant>/bm25.pkl", "BM25 index pickle per variant", "Auto-generated"),
            ("data/results/", "RetrievalResult JSONs per document per variant", "Auto-generated"),
        ],
        "data_note": None,
        "sibling_installs": ["RQ2_T01_SHARED", "RQ2_T02_DATA_LOADER"],
        "requirements": [
            "# Core",
            "numpy>=1.26",
            "pandas>=2.0",
            "tqdm>=4.66",
            "",
            "# LLM backend (Ollama must be running separately)",
            "ollama>=0.2",
            "",
            "# Sibling packages — install with: pip install -e ../RQ2_T01_SHARED",
            "# pip install -e ../RQ2_T02_DATA_LOADER",
        ],
    },
    {
        "folder": "RQ2_T05_ARM2_PAGEINDEX",
        "module": "arm2_pageindex",
        "context_file": "Project_Contexts_v0.0/RQ2_T05_ARM2_PAGEINDEX_CONTEXT.md",
        "description": "Arm 2B: ToC-tree construction and LLM-guided hierarchical navigation (vectorless).",
        "data_target": r"RQ2_T05_ARM2_PAGEINDEX",
        "data_contents": [
            ("data/trees/<doc_id>.json", "Serialized ToC tree (ToCNode hierarchy with LLM-generated summaries)", "Expensive to rebuild (one LLM call per node)"),
            ("data/results/", "RetrievalResult JSONs per document — includes full reasoning traces", "Large due to traces"),
        ],
        "data_note": "> Trees are expensive to build. Cache them in `data/trees/` and reuse across runs with `--skip-indexing`.",
        "sibling_installs": ["RQ2_T01_SHARED", "RQ2_T02_DATA_LOADER"],
        "requirements": [
            "# LLM backend (Ollama must be running separately)",
            "ollama>=0.2",
            "",
            "# Sibling packages — install with: pip install -e ../RQ2_T01_SHARED",
            "# pip install -e ../RQ2_T02_DATA_LOADER",
        ],
    },
    {
        "folder": "RQ2_T06_ARM_RESULTS",
        "module": "arm_results",
        "context_file": "Project_Contexts_v0.0/RQ2_T06_ARM_RESULTS_CONTEXT.md",
        "description": "Cross-arm results consolidation: tables, figures, error analyses, and slide assets aggregating T03/T04/T05 retrieval outputs and T07 metrics. Presentation layer only — no metric computation.",
        "data_target": r"RQ2_T06_ARM_RESULTS",
        "data_contents": [
            ("data/tables/", "Consolidated cross-arm metric tables (CSV + MD) summarising T07 outputs", "Auto-generated"),
            ("data/figures/", "Plots and thesis figures (PNG / SVG) across arms", "Auto-generated, referenced in thesis"),
            ("data/per_query/", "Per-question side-by-side comparisons across T03 / T04 / T05", "Auto-generated"),
            ("data/error_analysis/", "Categorised failure cases per arm with example chunks/articles", "Auto-generated"),
        ],
        "data_note": "> Consumes outputs only — never writes back into T03/T04/T05/T07 data directories.",
        "sibling_installs": ["RQ2_T01_SHARED", "RQ2_T02_DATA_LOADER", "RQ2_T07_EVALUATION"],
        "requirements": [
            "# Core",
            "pandas>=2.0",
            "numpy>=1.26",
            "",
            "# Statistics & visualisation",
            "scipy>=1.11",
            "matplotlib>=3.8",
            "seaborn>=0.13",
            "",
            "# Sibling packages — install with: pip install -e ../RQ2_T01_SHARED",
            "# pip install -e ../RQ2_T02_DATA_LOADER",
            "# pip install -e ../RQ2_T07_EVALUATION",
        ],
    },
    {
        "folder": "RQ2_T07_EVALUATION",
        "module": "evaluation",
        "context_file": "Project_Contexts_v0.0/RQ2_T07_EVALUATION_CONTEXT.md",
        "description": "Evaluation harness: binary + weighted IR metrics, autonomous LLM judge, cost tracking, reporting.",
        "data_target": r"RQ2_T07_EVALUATION",
        "data_contents": [
            ("data/results/arm1/", "Collected RetrievalResult JSONs from T03", "Large, auto-generated"),
            ("data/results/arm2_metadata/", "Collected RetrievalResult JSONs from T04", "Large, auto-generated"),
            ("data/results/arm2_pageindex/", "Collected RetrievalResult JSONs from T05", "Large, includes traces"),
            ("data/reports/", "EvalReport JSONs, metric tables (CSV), plots (PNG)", "Auto-generated"),
            ("data/llm_judge/", "LLM-as-judge responses and cost logs", "Large, expensive to regenerate"),
        ],
        "data_note": None,
        "sibling_installs": ["RQ2_T01_SHARED", "RQ2_T02_DATA_LOADER"],
        "requirements": [
            "# Eval frameworks",
            "ragas>=0.1",
            "deepeval>=0.21",
            "",
            "# Statistics & visualisation",
            "scipy>=1.11",
            "matplotlib>=3.8",
            "seaborn>=0.13",
            "",
            "# Sibling packages — install with: pip install -e ../RQ2_T01_SHARED",
            "# pip install -e ../RQ2_T02_DATA_LOADER",
        ],
    },
    {
        "folder": "RQ2_T08_RAG_VISUALIZATION",
        "module": "visualization",
        "context_file": "Project_Contexts_v0.0/RQ2_T08_RAG_VISUALIZATION_CONTEXT.md",
        "description": "Local Streamlit viewer for inspecting chunks, articles, and retrieval results on the original PDF page layout. A debugging and thesis-figure tool — read-only with respect to the pipeline.",
        "data_target": r"RQ2_T08_RAG_VISUALIZATION",
        "data_contents": [
            ("data/exports/", "PNG / SVG thesis figures exported from the viewer", "Auto-generated, referenced in thesis"),
        ],
        "data_note": "> This viewer is **read-only with respect to the pipeline** — it never writes to indices, ground-truth stores, or result directories. Indices and source data are read from sibling projects' `data/` paths via `config.yaml`.",
        "sibling_installs": [
            "RQ2_T01_SHARED", "RQ2_T02_DATA_LOADER", "RQ2_T03_ARM1_NAIVE",
            "RQ2_T04_ARM2_METADATA", "RQ2_T05_ARM2_PAGEINDEX",
        ],
        "extra_src_dirs": ["ui"],
        "requirements": [
            "# Streamlit app",
            "streamlit>=1.35",
            "streamlit-pdf-viewer==0.0.19   # pinned — do not float (see PROJECT_CONTEXT.md §10)",
            "",
            "# PDF coordinate mapping (word-index build)",
            "PyMuPDF>=1.23",
            "",
            "# Data handling",
            "pandas>=2.0",
            "pyyaml>=6.0",
            "",
            "# Sibling packages — install with: pip install -e ../RQ2_T01_SHARED",
            "# pip install -e ../RQ2_T02_DATA_LOADER",
            "# pip install -e ../RQ2_T03_ARM1_NAIVE",
            "# pip install -e ../RQ2_T04_ARM2_METADATA",
            "# pip install -e ../RQ2_T05_ARM2_PAGEINDEX",
        ],
    },
]

GITIGNORE = """\
# Virtual environment
.venv/
venv/
env/

# Python
__pycache__/
*.py[cod]
*.pyo
*.pyd
*.egg-info/
dist/
build/
.eggs/

# Data & outputs (large files — store separately)
data/
outputs/
results/
*.pkl
*.npy
*.faiss
*.index

# Jupyter
.ipynb_checkpoints/
*.ipynb

# IDE
.vscode/
.idea/
*.code-workspace

# OS
.DS_Store
Thumbs.db

# Secrets / config overrides
.env
config.local.yaml
"""

CHANGE_NOTES_TEMPLATE = """\
# CHANGE NOTES — {name}

## Workflow

This file is the cross-project change log for **{name}**. All sibling projects write to it when they need something from this project, and this project tracks what it has requested from others.

### Rules

**When another project needs a change here:**
1. The requesting project adds an entry under [Incoming Requests] with status `[ ]` and mirrors it under [Outgoing Requests] in their own `CHANGE_NOTES.md`.
2. This project implements the change and updates the entry status to `[x] Done — YYYY-MM-DD`.
3. The requesting project reviews and confirms, then **both sides delete their entry** (incoming here + outgoing in their file).

**When this project needs a change elsewhere:**
1. Add an entry under [Outgoing Requests] below, and add the mirror entry under [Incoming Requests] in the target project's `CHANGE_NOTES.md`.
2. When the target project marks their entry as `[x] Done`, review and confirm.
3. Once confirmed, **delete both entries** (outgoing here + incoming in their file).

### Entry format

```
### [ ] CN-<TARGET_CODE>-<NNN> — Short title
- **Requested by / to:** RQ2_T0X_PROJECTNAME
- **Priority:** High | Medium | Low
- **Opened:** YYYY-MM-DD
- **Description:** What is needed and why.
- **Acceptance criteria:** How to verify it is done.
```

Status markers: `[ ]` Pending · `[x] Done — YYYY-MM-DD` (set by implementing project) · deleted when confirmed by requester.

---

## Incoming Requests

> Changes requested by other projects that must be implemented **in this project**.

*No pending requests.*

---

## Outgoing Requests

> Changes this project has requested from other projects. Delete each entry once you have reviewed and confirmed the implementation.

*No outgoing requests.*
"""


def readme(project: dict) -> str:
    folder = project["folder"]
    module = project["module"]
    description = project["description"]
    target = project.get("data_target")
    data_contents = project.get("data_contents", [])
    data_note = project.get("data_note")
    sibling_installs = project.get("sibling_installs", [])

    # --- Storage section ---
    if target is None:
        storage_section = """\
## Storage

This project is a **pure library** — it contains no large files and does not write to disk directly. No `data/` directory is required.

FAISS indices and BM25 pickles are written by the projects that call this library and stored in their respective `data/` directories.

### What lives in the Git repository

- All Python source files (`src/{module}/`)
- `requirements.txt`, `.gitignore`, `README.md`, `PROJECT_CONTEXT.md`
""".format(module=module)
    else:
        data_target = rf"<data root>/{target}"
        table_rows = "\n".join(
            f"| `{path}` | {contents} | {reason} |"
            for path, contents, reason in data_contents
        )
        note_block = f"\n{data_note}\n" if data_note else ""
        storage_section = f"""\
## Storage

### What lives in the Git repository

- All Python source files (`src/{module}/`)
- `requirements.txt`, `.gitignore`, `README.md`, `PROJECT_CONTEXT.md`

### What lives in the data bundle only (via `data/`)

Large artefacts are not committed to git. They live in the companion Hugging Face
dataset `mpaschalidis/bsard-rag-thesis-data` (subset `rq2`) and download into a
local gitignored data root (env `RQ2_DATA_DIR`, default `<repo>/data`). This
project's `data/` is wired to that root via `python scripts/setup/link_data.py`,
resolving to:
```
{data_target}/
```

| Path | Contents | Reason |
|---|---|---|
{table_rows}
{note_block}"""

    # --- Setup section ---
    data_step = ""
    if target is not None:
        data_step = """\
### Step 1 — Download the data bundle and wire `data/`

Large artefacts live in the companion Hugging Face dataset
`mpaschalidis/bsard-rag-thesis-data` (subset `rq2`). Pull them into the data root
(env `RQ2_DATA_DIR`, default `<repo>/data`) and link this project's `data/` to it,
run once from the component root:

```powershell
python scripts/download_data.py
python scripts/setup/link_data.py
```

"""
        venv_step_num = 2
        pip_step_num = 3
        sibling_step_num = 4
    else:
        venv_step_num = 1
        pip_step_num = 2
        sibling_step_num = 3

    prereq = ""

    sibling_block = ""
    if sibling_installs:
        installs = "\n".join(f"pip install -e ..\\{s}" for s in sibling_installs)
        sibling_block = f"""\

### Step {sibling_step_num} — Install sibling packages

```powershell
{installs}
```
"""

    setup_section = f"""\
## Setup

{prereq}{data_step}### Step {venv_step_num} — Create and activate the virtual environment

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
```

### Step {pip_step_num} — Install dependencies

```powershell
pip install -r requirements.txt
```
{sibling_block}"""

    return f"""\
# {folder}

{description}

---

{setup_section}
---

{storage_section}
---

## Project context

See [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) for full design specification.

## Module

```
src/{module}/
└── __init__.py
```

## Part of

RQ2 pipeline — Belgian Statutory Article Retrieval (BSARD) thesis project.
"""


def run(cmd: list[str], cwd: Path, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=cwd,
        check=check,
        capture_output=True,
        text=True,
    )


def create_project(project: dict, force: bool = False) -> None:
    folder = ROOT / project["folder"]
    name = project["folder"]

    if folder.exists() and not force:
        print(f"  [skip] {name} already exists (use --force to recreate)")
        return

    print(f"\n{'='*60}")
    print(f"  Creating: {name}")
    print(f"{'='*60}")

    # 1. Directory structure (including optional extra_src_dirs)
    src_module = folder / "src" / project["module"]
    src_module.mkdir(parents=True, exist_ok=True)
    for extra in project.get("extra_src_dirs", []):
        extra_dir = src_module / extra
        extra_dir.mkdir(parents=True, exist_ok=True)
        (extra_dir / "__init__.py").write_text(
            f'"""Sub-package: {extra}"""\n', encoding="utf-8"
        )
    print(f"  [ok] created src/{project['module']}/")

    # 2. src/__init__.py
    (src_module / "__init__.py").write_text(
        f'"""{ project["description"] }"""\n', encoding="utf-8"
    )

    # 3. PROJECT_CONTEXT.md
    context_src = ROOT / project["context_file"]
    context_dst = folder / "PROJECT_CONTEXT.md"
    if context_src.exists():
        shutil.copy2(context_src, context_dst)
        print(f"  [ok] copied PROJECT_CONTEXT.md")
    else:
        context_dst.write_text(f"# {name}\n\nContext file not found: {project['context_file']}\n")
        print(f"  [warn] context file not found: {project['context_file']}")

    # 4. .gitignore
    (folder / ".gitignore").write_text(GITIGNORE, encoding="utf-8")
    print(f"  [ok] wrote .gitignore")

    # 5. README.md
    (folder / "README.md").write_text(readme(project), encoding="utf-8")
    print(f"  [ok] wrote README.md")

    # 6. requirements.txt
    req_lines = project["requirements"]
    (folder / "requirements.txt").write_text("\n".join(req_lines) + "\n", encoding="utf-8")
    print(f"  [ok] wrote requirements.txt")

    # 6b. CHANGE_NOTES.md
    change_notes = CHANGE_NOTES_TEMPLATE.format(name=name)
    (folder / "CHANGE_NOTES.md").write_text(change_notes, encoding="utf-8")
    print(f"  [ok] wrote CHANGE_NOTES.md")

    # 7. Virtual environment
    print(f"  [..] creating .venv (this may take a moment)...")
    result = run([sys.executable, "-m", "venv", ".venv"], cwd=folder, check=False)
    if result.returncode == 0:
        print(f"  [ok] .venv created")
    else:
        print(f"  [warn] venv creation failed: {result.stderr.strip()}")

    # 8. Git init + initial commit
    print(f"  [..] initialising git repository...")
    run(["git", "init"], cwd=folder, check=False)
    tracked = ["README.md", "requirements.txt", ".gitignore",
               "PROJECT_CONTEXT.md", "CHANGE_NOTES.md",
               f"src/{project['module']}/__init__.py"]
    for extra in project.get("extra_src_dirs", []):
        tracked.append(f"src/{project['module']}/{extra}/__init__.py")
    run(["git", "add"] + tracked, cwd=folder, check=False)
    result = run(
        ["git", "commit", "-m", f"chore: initial scaffold for {name}"],
        cwd=folder,
        check=False,
    )
    if result.returncode == 0:
        print(f"  [ok] git init + initial commit")
    else:
        print(f"  [warn] git commit failed: {result.stderr.strip()}")

    print(f"  [done] {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true",
        help="Recreate projects that already exist"
    )
    parser.add_argument(
        "--only", nargs="+", metavar="FOLDER",
        help="Only initialise specific project folders (e.g. RQ2_T01_SHARED)"
    )
    parser.add_argument(
        "--skip-venv", action="store_true",
        help="Skip virtual environment creation (faster for re-runs)"
    )
    args = parser.parse_args()

    targets = PROJECTS
    if args.only:
        targets = [p for p in PROJECTS if p["folder"] in args.only]
        if not targets:
            print(f"No matching projects for: {args.only}")
            sys.exit(1)

    if args.skip_venv:
        # Monkey-patch venv step out
        for p in targets:
            p["_skip_venv"] = True

    print(f"\nRQ2 Project Bootstrap")
    print(f"Root: {ROOT}")
    print(f"Projects to initialise: {len(targets)}")

    for project in targets:
        create_project(project, force=args.force)

    print(f"\n{'='*60}")
    print(f"  Bootstrap complete.")
    print(f"{'='*60}")
    print(f"""
Next steps per project:
  1. cd <project-folder>
  2. Activate venv:
       Windows:  .venv\\Scripts\\activate
       macOS/Linux: source .venv/bin/activate
  3. pip install -r requirements.txt
  4. gh repo create <repo-name> --private --source=. --push

For sibling packages (shared, data_loader):
  pip install -e ../RQ2_T01_SHARED
  pip install -e ../RQ2_T02_DATA_LOADER
""")


if __name__ == "__main__":
    main()
