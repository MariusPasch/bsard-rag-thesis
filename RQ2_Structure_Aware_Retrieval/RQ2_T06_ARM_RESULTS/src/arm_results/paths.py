"""Path resolution and the curated-PDF registry for cross-arm consolidation.

T06 reads each arm's *persisted* per-query rankings (no retrieval rerun) and the
per-PDF ground-truth files owned by T07. The large data artefacts are not in git:
they live in the companion Hugging Face dataset
``Marios-Paschalidis-Thesis/bsard-rag-thesis-data`` (subset ``rq2``) and download into a local
gitignored data root via ``scripts/download_data.py``. Each sub-project's
``data/`` is wired to that root via ``scripts/setup/link_data.py``.

Everything here is overridable by environment variable so the module is portable:

  RQ2_DATA_DIR  → the data root (default ``<repo>/data``)
  RQ2_BSARD_DB  → the BSARD corpus SQLite DB (default ``<data root>/bsard_corpus.db``)
"""

from __future__ import annotations

import os
from pathlib import Path

# --- repo + data roots -----------------------------------------------------

# .../RQ2_Structure_Aware_Retrieval/RQ2_T06_ARM_RESULTS/src/arm_results/paths.py
#   parents[0]=arm_results [1]=src [2]=RQ2_T06_ARM_RESULTS [3]=RQ2_Structure_Aware_Retrieval
REPO_ROOT = Path(__file__).resolve().parents[3]

def _data_dir() -> Path:
    """The shared data root: $RQ2_DATA_DIR, else <repo>/data.

    Populated by ``scripts/download_data.py``; mirrors the project's shared data
    tree (pdf_cache/, AzureDI/, RQ2_T04_*/, RQ2_T05_*/, shared_source/, plus
    bsard_corpus.db).
    """
    env = os.environ.get("RQ2_DATA_DIR")
    return Path(env).expanduser() if env else REPO_ROOT / "data"


DATA_BASE = _data_dir()

BSARD_DB = (
    Path(os.environ["RQ2_BSARD_DB"]).expanduser()
    if os.environ.get("RQ2_BSARD_DB")
    else _data_dir() / "bsard_corpus.db"
)

# --- arm input roots (under the shared data root) --------------------------

T03_PDF_CACHE = DATA_BASE / "pdf_cache"
T04_BASE = DATA_BASE / "RQ2_T04_ARM2_METADATA"
T05_BASE = DATA_BASE / "RQ2_T05_ARM2_PAGEINDEX"

# Ground truth: T07 owns it; per-PDF curated subsets used by every arm.
GT_RUNS_DIR = REPO_ROOT / "RQ2_T07_EVALUATION" / "ground_truth" / "runs"

# --- output root (this project's data/ directory) --------------------------

OUTPUT_BASE = REPO_ROOT / "RQ2_T06_ARM_RESULTS" / "data"
TABLES_DIR = OUTPUT_BASE / "tables"
FIGURES_DIR = OUTPUT_BASE / "figures"
PER_QUERY_DIR = OUTPUT_BASE / "per_query"
ERROR_DIR = OUTPUT_BASE / "error_analysis"


# --- the curated 5-PDF set -------------------------------------------------
# stem → (azure_doc_id, human label). doc 2 (2004A27101) is excluded for GT
# drift (see project_doc2_gt_drift / project_curated_pdf_set).

CURATED_STEMS: dict[str, tuple[int, str]] = {
    "1804_03_21_1804032150": (9, "Code Civil"),
    "1867_06_08_1867060850": (6, "Code Pénal"),
    "1967_10_10_1967101055": (7, "Code Judiciaire (larger)"),
    "1967_10_10_1967101056": (5, "Code Judiciaire (smaller)"),
    "2003_07_17_2013A31614": (8, "Code du Logement"),
}

STEMS = list(CURATED_STEMS)


def stem_label(stem: str) -> str:
    """Human-readable label for a PDF stem, falling back to the stem itself."""
    return CURATED_STEMS.get(stem, (None, stem))[1]


def gt_path(stem: str) -> Path:
    """Path to the per-PDF curated ground-truth file for ``stem``."""
    return GT_RUNS_DIR / f"t04_{stem}.json"


def ensure_output_dirs() -> None:
    """Create the output directory tree under this project's data/ directory."""
    for d in (TABLES_DIR, FIGURES_DIR, PER_QUERY_DIR, ERROR_DIR):
        d.mkdir(parents=True, exist_ok=True)


# --- T03 cache hashes / weights (for opt-in W/ + E/ paths) -----------------

def t03_config_hash(stem: str) -> str | None:
    """T03's config hash for ``stem`` (the results sub-dir name), or None."""
    rdir = T03_PDF_CACHE / stem / "results"
    if not rdir.exists():
        return None
    for d in sorted(rdir.iterdir()):
        if d.is_dir():
            return d.name
    return None


def t03_weights_csv(stem: str) -> Path | None:
    """Path to T03's cached chunk→bsard overlap weights CSV for ``stem``."""
    h = t03_config_hash(stem)
    if not h:
        return None
    candidates = [
        T03_PDF_CACHE / stem / "eval" / h / "chunk_bsard_weights.csv",
        T03_PDF_CACHE / stem / "configs" / h / "eval" / "chunk_bsard_weights.csv",
    ]
    return next((c for c in candidates if c.exists()), None)
