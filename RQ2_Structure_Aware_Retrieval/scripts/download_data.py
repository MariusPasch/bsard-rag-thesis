"""Download the RQ2 data from the combined BSARD-RAG Hugging Face dataset.

RQ2 spans nine sub-projects whose large inputs (the BSARD corpus DB, the Azure
Document Intelligence exports, the per-arm FAISS / BM25 / PageIndex indices and
their cached embeddings, and the experiment result JSONs) are not stored in git.
They live in the ``rq2/`` subset of the combined dataset
``Marios-Paschalidis-Thesis/bsard-rag-thesis-data`` and are downloaded into the local data
root, with any byte-shards transparently reassembled. The subset's internal
layout mirrors the project's shared data tree:

    <RQ2_DATA_DIR>/
        bsard_corpus.db
        shared_source/   pdf_cache/   AzureDI/
        RQ2_T04_ARM2_METADATA/   RQ2_T05_ARM2_PAGEINDEX/

Data root resolution:
  1. ``RQ2_DATA_DIR`` environment variable, if set.
  2. ``<repo>/data`` otherwise.

After downloading, run ``python scripts/setup/link_data.py`` to (re)create the
per-sub-project ``data/`` junctions that point into this tree.

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --allow "bsard_corpus.db" "AzureDI/**"
    RQ2_HF_REPO=you/your-dataset python scripts/download_data.py

The dataset repo id can be overridden with ``RQ2_HF_REPO`` (or
``BSARD_HF_COMBINED_REPO``); the default is the combined thesis dataset.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

DEFAULT_HF_REPO = "Marios-Paschalidis-Thesis/bsard-rag-thesis-data"
SUBSET = "rq2"

REPO_ROOT = Path(__file__).resolve().parent.parent


def data_dir() -> Path:
    """Resolve the data root: $RQ2_DATA_DIR, else <repo>/data."""
    env = os.environ.get("RQ2_DATA_DIR")
    return Path(env).expanduser() if env else REPO_ROOT / "data"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        default=(os.environ.get("RQ2_HF_REPO")
                 or os.environ.get("BSARD_HF_COMBINED_REPO") or DEFAULT_HF_REPO),
        help="Combined HF dataset repo id (default: $RQ2_HF_REPO / "
             "$BSARD_HF_COMBINED_REPO or %(default)s)",
    )
    parser.add_argument("--revision", default=None, help="Dataset revision / tag.")
    parser.add_argument("--allow", nargs="*", default=None,
                        help="Subset-relative globs to fetch only part of the bundle "
                             "(e.g. 'bsard_corpus.db' 'AzureDI/**'). Default: whole subset.")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "data_tooling"))  # mono-repo/data_tooling
    try:
        from download_combined_hf import download_subset  # noqa: E402
    except ImportError as e:
        print(f"Import failed ({e}). Ensure huggingface_hub is installed and you run "
              "this from the RQ2_Structure_Aware_Retrieval component of the mono-repo.", file=sys.stderr)
        return 1

    root = data_dir()
    print(f"Downloading '{args.repo}' subset '{SUBSET}' -> {root}")
    download_subset(args.repo, SUBSET, args.revision, root, allow_patterns=args.allow)
    print("\nDone. Next: python scripts/setup/link_data.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
