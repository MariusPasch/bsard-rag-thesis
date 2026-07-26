"""Download the RQ1 data from the combined BSARD-RAG Hugging Face dataset.

RQ1's large artefacts (the SQLite corpus + FTS5 index, the deduplicated parquet
export, cached dense embeddings, and the experiment result JSONs) are not stored
in git. They live in the ``rq1/`` subset of the combined dataset
``mpaschalidis/bsard-rag-thesis-data`` and are downloaded into the local data
root, with any byte-shards transparently reassembled.

Data root resolution (see evaluation/paths.py):
  1. ``BSARD_DATA_DIR`` environment variable, if set.
  2. ``<repo>/output`` otherwise.

Usage:
    python scripts/download_data.py                 # full rq1 subset
    python scripts/download_data.py --no-embeddings # skip cached embeddings
    BSARD_HF_REPO=you/your-dataset python scripts/download_data.py

The dataset repo id can be overridden with ``BSARD_HF_REPO`` (or
``BSARD_HF_COMBINED_REPO``); the default is the combined thesis dataset.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

DEFAULT_HF_REPO = "mpaschalidis/bsard-rag-thesis-data"
SUBSET = "rq1"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        default=(os.environ.get("BSARD_HF_REPO")
                 or os.environ.get("BSARD_HF_COMBINED_REPO") or DEFAULT_HF_REPO),
        help="Combined HF dataset repo id (default: $BSARD_HF_REPO / "
             "$BSARD_HF_COMBINED_REPO or %(default)s)",
    )
    parser.add_argument("--no-embeddings", action="store_true",
                        help="Skip the cached dense embeddings (they can be regenerated).")
    parser.add_argument("--revision", default=None, help="Dataset revision / tag.")
    args = parser.parse_args()

    # Resolve the data root via the same logic the rest of the code uses, and the
    # shared subset-downloader from the mono-repo root scripts/.
    here = Path(__file__).resolve()
    sys.path.insert(0, str(here.parents[1]))                 # RQ1_Retrieval_Methods/
    sys.path.insert(0, str(here.parents[2] / "data_tooling"))     # <mono-repo>/data_tooling
    try:
        from evaluation.paths import DATA_ROOT  # noqa: E402
        from download_combined_hf import download_subset  # noqa: E402
    except ImportError as e:
        print(f"Import failed ({e}). Ensure huggingface_hub is installed and you run "
              "this from the RQ1_Retrieval_Methods component of the mono-repo.", file=sys.stderr)
        return 1

    ignore = ["embeddings/**"] if args.no_embeddings else None
    print(f"Downloading '{args.repo}' subset '{SUBSET}' -> {DATA_ROOT}"
          + ("  (no embeddings)" if args.no_embeddings else ""))
    download_subset(args.repo, SUBSET, args.revision, DATA_ROOT, ignore_patterns=ignore)
    print("\nDone. Verify the corpus DB:", DATA_ROOT / "bsard_corpus.db")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
