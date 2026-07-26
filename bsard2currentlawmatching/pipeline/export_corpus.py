"""Phase D (part 2) — Parquet and JSONL exports from the SQLite database.

Reads:
  output/bsard_corpus.db

Writes:
  output/bsard_articles.parquet          — full corpus (all 40 K+ articles)
  output/bsard_articles.jsonl            — full corpus for vector store ingestion
  output/bsard_articles_only.parquet     — BSARD benchmark subset only
  output/bsard_articles_only.jsonl       — BSARD subset for benchmarking

Usage:
    python retrieval/export_corpus.py
    python retrieval/export_corpus.py --db output/bsard_corpus.db
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
DEFAULT_DB    = PROJECT_ROOT / "output" / "bsard_corpus.db"
DEFAULT_OUTDIR = PROJECT_ROOT / "output"

# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase D: export corpus to Parquet/JSONL")
    parser.add_argument("--db",      type=Path, default=DEFAULT_DB)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()

    if not args.db.exists():
        print(f"Database not found: {args.db}")
        print("Run build_database.py first.")
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row

    # ── Load full articles table ───────────────────────────────────────────────
    print("Loading articles from database ...", end=" ", flush=True)
    df = pd.read_sql("SELECT * FROM articles ORDER BY article_id", conn)
    print(f"{len(df):,} rows, {len(df.columns)} columns")

    # ── Full corpus exports ────────────────────────────────────────────────────
    parquet_path = args.out_dir / "bsard_articles.parquet"
    jsonl_path   = args.out_dir / "bsard_articles.jsonl"

    print(f"Writing {parquet_path.name} ...", end=" ", flush=True)
    df.to_parquet(parquet_path, index=False, engine="pyarrow")
    print(f"{parquet_path.stat().st_size // 1024:,} KB")

    print(f"Writing {jsonl_path.name} ...", end=" ", flush=True)
    df.to_json(jsonl_path, orient="records", lines=True, force_ascii=False)
    print(f"{jsonl_path.stat().st_size // 1024:,} KB")

    # ── BSARD-only subset ──────────────────────────────────────────────────────
    bsard_df = df[df["is_bsard_article"] == 1].copy()
    print(f"\nBSARD subset: {len(bsard_df):,} records "
          f"({bsard_df['bsard_id'].nunique():,} unique bsard_ids)")

    parquet_bsard = args.out_dir / "bsard_articles_only.parquet"
    jsonl_bsard   = args.out_dir / "bsard_articles_only.jsonl"

    print(f"Writing {parquet_bsard.name} ...", end=" ", flush=True)
    bsard_df.to_parquet(parquet_bsard, index=False, engine="pyarrow")
    print(f"{parquet_bsard.stat().st_size // 1024:,} KB")

    print(f"Writing {jsonl_bsard.name} ...", end=" ", flush=True)
    bsard_df.to_json(jsonl_bsard, orient="records", lines=True, force_ascii=False)
    print(f"{jsonl_bsard.stat().st_size // 1024:,} KB")

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\nExport summary:")
    for p in [parquet_path, jsonl_path, parquet_bsard, jsonl_bsard]:
        print(f"  {p.name:<40} {p.stat().st_size // 1024:>7,} KB")

    # ── Quick Parquet round-trip check ─────────────────────────────────────────
    test = pd.read_parquet(parquet_path, columns=["article_id", "law_code", "article_text"])
    assert len(test) == len(df), "Parquet row count mismatch"
    print(f"\nParquet round-trip check passed ({len(test):,} rows).")

    conn.close()
    print("Phase D (export_corpus) complete.")


if __name__ == "__main__":
    main()
