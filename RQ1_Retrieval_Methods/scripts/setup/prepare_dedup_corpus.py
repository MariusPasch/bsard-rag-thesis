"""
Deduplicate bsard_articles_only.parquet → bsard_articles_dedup.parquet

Strategy:
  - Keep one row per bsard_id.
  - Prefer the sibling whose article_id appears in any ground-truth set
    (train + val + test), so evaluation article_ids are never missing.
  - Tie-break on min(article_id) for determinism.
  - The original file is left untouched as a backup.

Usage (from project root):
    .venv/Scripts/python scripts/prepare_dedup_corpus.py
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

SRC_PARQUET  = ROOT / "output" / "bsard_articles_only.parquet"
DEST_PARQUET = ROOT / "output" / "bsard_articles_dedup.parquet"
DB_PATH      = ROOT / "output" / "bsard_corpus.db"


def collect_gt_article_ids(db_path: Path) -> set[int]:
    """Return every article_id referenced in any question's ground truth."""
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT relevant_article_ids FROM questions").fetchall()
    conn.close()
    ids: set[int] = set()
    for (raw,) in rows:
        ids.update(json.loads(raw))
    return ids


def main() -> None:
    print("Loading source parquet …")
    df = pd.read_parquet(SRC_PARQUET)
    print(f"  Source rows  : {len(df):>7,}")
    print(f"  Unique bsard_ids: {df['bsard_id'].nunique():>7,}")

    print("Collecting ground-truth article_ids …")
    gt_ids = collect_gt_article_ids(DB_PATH)
    print(f"  GT article_ids : {len(gt_ids):>7,}")

    # Sort so GT-anchored rows sort first within each bsard_id group,
    # then by article_id ascending as stable tiebreaker.
    print("Deduplicating (GT-anchored, then min article_id) …")
    df = df.copy()
    df["_gt_priority"] = (~df["article_id"].isin(gt_ids)).astype(int)  # 0 = GT, 1 = non-GT
    df_sorted = df.sort_values(["bsard_id", "_gt_priority", "article_id"])
    dedup = df_sorted.drop_duplicates(subset="bsard_id", keep="first")
    dedup = dedup.drop(columns=["_gt_priority"]).reset_index(drop=True)

    print(f"  Dedup rows   : {len(dedup):>7,}")

    # Sanity: every GT article_id must survive deduplication
    surviving_ids = set(dedup["article_id"])
    missing = gt_ids - surviving_ids
    if missing:
        print(f"ERROR: {len(missing)} GT article_ids lost after dedup — aborting.")
        print(f"  First 10 missing: {sorted(missing)[:10]}")
        sys.exit(1)
    print(f"  GT coverage check: PASS (0 missing)")

    print(f"Writing {DEST_PARQUET.name} …")
    dedup.to_parquet(DEST_PARQUET, index=False)
    print(f"  Done. Original file kept at: {SRC_PARQUET.name}")
    print(f"\nSummary: {len(df) - len(dedup):,} duplicate rows removed "
          f"({len(df):,} -> {len(dedup):,})")


if __name__ == "__main__":
    main()
