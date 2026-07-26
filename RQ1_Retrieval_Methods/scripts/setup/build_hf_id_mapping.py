"""
Build a bidirectional mapping between HuggingFace BSARD corpus IDs and
local article IDs (from the SQLite corpus DB).

The HuggingFace dataset `maastrichtlawtech/bsard` uses sequential IDs
(1–22,633) stored in the `id` column.  The local SQLite database uses a
different `article_id` column (range ~1–40,231) with a `bsard_id` column
that links back to the HF ID.

This script creates two mapping files:

    output/hf_to_local_id_mapping.json   — { "hf_id": local_article_id }
    output/local_to_hf_id_mapping.json   — { "local_article_id": hf_id }

Usage (from project root):
    .venv/Scripts/python scripts/setup/build_hf_id_mapping.py

The mapping is deterministic (based on the SQLite DB content).  Re-run
whenever the SQLite DB is rebuilt.

Schema relationships:
    HuggingFace `id`  ==  SQLite `bsard_id`  (1–22,633)
    SQLite `article_id`  ==  Dedup parquet `article_id`  ==  Ground-truth IDs
"""

import json
import sqlite3
import sys
from pathlib import Path

DB_PATH = Path("output/bsard_corpus.db")
OUT_DIR = Path("output")

def main():
    if not DB_PATH.exists():
        print(f"ERROR: {DB_PATH} not found. Ensure `output/` junction is set up.")
        sys.exit(1)

    conn = sqlite3.connect(str(DB_PATH))

    # Build the mapping using the dedup articles only (is_bsard_article flag or bsard_id)
    rows = conn.execute("""
        SELECT article_id, bsard_id
        FROM articles
        WHERE bsard_id IS NOT NULL
    """).fetchall()

    # bsard_id can map to multiple article_ids (pre-dedup).
    # We keep the mapping from the dedup parquet — which is 1:1.
    # Load the dedup parquet to get the canonical set.
    import pandas as pd
    dedup = pd.read_parquet("output/bsard_articles_dedup.parquet")
    dedup_ids = set(dedup["article_id"].tolist())

    hf_to_local = {}
    local_to_hf = {}
    skipped_dups = 0

    for article_id, bsard_id in rows:
        if article_id not in dedup_ids:
            skipped_dups += 1
            continue
        hf_id_str = str(bsard_id)
        local_id_str = str(article_id)

        if hf_id_str in hf_to_local:
            print(f"  WARNING: HF ID {bsard_id} maps to multiple dedup articles: "
                  f"{hf_to_local[hf_id_str]} and {article_id}")
        hf_to_local[hf_id_str] = article_id
        local_to_hf[local_id_str] = bsard_id

    conn.close()

    # Save
    hf_path = OUT_DIR / "hf_to_local_id_mapping.json"
    local_path = OUT_DIR / "local_to_hf_id_mapping.json"

    hf_path.write_text(json.dumps(hf_to_local, indent=2))
    local_path.write_text(json.dumps(local_to_hf, indent=2))

    print(f"Mapping built successfully:")
    print(f"  HF -> Local  : {len(hf_to_local):,} entries -> {hf_path}")
    print(f"  Local -> HF  : {len(local_to_hf):,} entries -> {local_path}")
    print(f"  Skipped (not in dedup): {skipped_dups:,} rows")
    print(f"\nSample mappings (HF ID -> Local article_id):")
    for i, (hf, local) in enumerate(sorted(hf_to_local.items(), key=lambda x: int(x[0]))[:5]):
        print(f"  HF {hf:>5} -> article_id {local}")


if __name__ == "__main__":
    main()
