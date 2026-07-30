"""Download the corpus data from the combined BSARD-RAG Hugging Face dataset.

Pulls the ``corpus/`` subset (the two authoritative SQLite databases, the small
Parquet snapshots, the verification CSV, id-mappings, corpus statistics, and the
question-extraction analysis) into the local ``output/`` directory so the
pipeline and analysis notebooks can run without re-building. Any byte-shards are
reassembled transparently.

Usage:
    python scripts/download_from_hf.py
    python scripts/download_from_hf.py --include "*.parquet" "*.db"
    python scripts/download_from_hf.py --repo OTHER/REPO --revision v1

The dataset repo id can be overridden with ``BSARD_HF_REPO`` (or
``BSARD_HF_COMBINED_REPO``); the default is the combined thesis dataset.

Requires `huggingface_hub` (already pinned in requirements.txt).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPO_ID = "Marios-Paschalidis-Thesis/bsard-rag-thesis-data"
SUBSET = "corpus"
DEFAULT_LOCAL_DIR = PROJECT_ROOT / "output"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--repo",
        default=(os.environ.get("BSARD_HF_REPO")
                 or os.environ.get("BSARD_HF_COMBINED_REPO") or DEFAULT_REPO_ID),
        help=f"Combined HF dataset repo id (default: $BSARD_HF_REPO / "
             f"$BSARD_HF_COMBINED_REPO or {DEFAULT_REPO_ID})")
    parser.add_argument("--revision", default=None, help="Optional branch / tag / commit")
    parser.add_argument("--local-dir", type=Path, default=DEFAULT_LOCAL_DIR,
                        help=f"Where to mirror the corpus subset (default: {DEFAULT_LOCAL_DIR})")
    parser.add_argument("--include", nargs="*", default=None,
                        help="Subset-relative globs to selectively download (default: everything)")
    args = parser.parse_args()

    sys.path.insert(0, str(PROJECT_ROOT.parent / "data_tooling"))  # <mono-repo>/data_tooling
    from download_combined_hf import download_subset  # noqa: E402

    print(f"Downloading '{args.repo}' subset '{SUBSET}' -> {args.local_dir}")
    download_subset(args.repo, SUBSET, args.revision, args.local_dir,
                    allow_patterns=args.include)


if __name__ == "__main__":
    main()
