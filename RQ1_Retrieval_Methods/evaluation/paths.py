"""Central resolver for the location of large data artefacts.

All bulky inputs (the SQLite corpus DB, parquet exports, embeddings, FAISS
indices) and experiment result JSONs live outside the git repository. They are
downloaded from the companion Hugging Face dataset (see scripts/download_data.py)
into a single data root.

Resolution order for the data root:
  1. The ``BSARD_DATA_DIR`` environment variable, if set.
  2. ``<repo>/output`` otherwise (created on demand).

Any machine works as long as the data bundle has been downloaded (or
BSARD_DATA_DIR points at it).
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def data_root() -> Path:
    """Return the directory holding large data artefacts and results."""
    env = os.environ.get("BSARD_DATA_DIR")
    root = Path(env).expanduser() if env else PROJECT_ROOT / "output"
    return root


DATA_ROOT = data_root()
RESULTS_DIR = DATA_ROOT / "results"
CORPUS_DB = DATA_ROOT / "bsard_corpus.db"
CORPUS_PARQUET = DATA_ROOT / "bsard_articles_dedup.parquet"
EMBEDDINGS_DIR = DATA_ROOT / "embeddings"
