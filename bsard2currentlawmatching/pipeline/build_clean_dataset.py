"""Build a cleaned, deduplicated companion dataset from the existing BSARD corpus.

Reads output/bsard_corpus.db and produces a companion dataset containing
only articles Phase A actually extracted from a PDF, with duplicates
collapsed. The original bsard_corpus.db is NOT modified.

Filter:
  pdf_page_start IS NOT NULL  — excludes Phase B append stubs (BSARD articles
                                 whose text came from HuggingFace canonical
                                 only, without a PDF-extracted counterpart).

Deduplication:
  Key: (pdf_filename, article_number).
  Within a duplicate group, the row with the best score wins, where score is:
    (has_text, is_bsard, text_len, pdf_page_start)
  That is: prefer rows with a body, prefer BSARD-linked rows to preserve the
  join key, prefer longer text, prefer rows appearing later in the PDF (body
  pages beat table-of-contents phantoms from the front matter).

Article IDs are preserved from the source DB, so the old citation_graph and
questions tables can be joined directly on article_id.

Outputs:
  output/bsard_corpus_clean.db        — SQLite (same schema as source)
  output/bsard_articles_clean.parquet
  output/bsard_articles_clean.jsonl

Usage:
  python pipeline/build_clean_dataset.py
  python pipeline/build_clean_dataset.py --src output/bsard_corpus.db
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DB      = PROJECT_ROOT / "output" / "bsard_corpus.db"
DST_DB      = PROJECT_ROOT / "output" / "bsard_corpus_clean.db"
DST_PARQUET = PROJECT_ROOT / "output" / "bsard_articles_clean.parquet"
DST_JSONL   = PROJECT_ROOT / "output" / "bsard_articles_clean.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build cleaned companion dataset")
    parser.add_argument("--src", type=Path, default=SRC_DB)
    parser.add_argument("--dst-db", type=Path, default=DST_DB)
    parser.add_argument("--dst-parquet", type=Path, default=DST_PARQUET)
    parser.add_argument("--dst-jsonl", type=Path, default=DST_JSONL)
    args = parser.parse_args()

    # ── 1. Load source ─────────────────────────────────────────────────────
    print(f"Reading {args.src} ...")
    src = sqlite3.connect(str(args.src))
    df = pd.read_sql("SELECT * FROM articles", src)
    n_initial = len(df)
    print(f"  {n_initial:,} rows in source")

    # Capture the table DDL so the clean DB can mirror the schema exactly.
    cur = src.cursor()
    cur.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='articles'")
    table_ddl = cur.fetchone()[0]
    src.close()

    # ── 2. Filter: Phase A PDF extractions only ────────────────────────────
    # pdf_page_start is NULL only for Phase B stubs; Phase A always sets it.
    df_ext = df[df["pdf_page_start"].notna()].copy()
    n_stubs = n_initial - len(df_ext)
    print(f"  {n_stubs:,} Phase B append stubs excluded "
          f"({n_stubs / n_initial * 100:.1f}%)")
    print(f"  {len(df_ext):,} rows remain (Phase A PDF extractions)")

    # ── 3. Deduplicate by (pdf_filename, article_number) ───────────────────
    df_ext["_has_text"]   = df_ext["article_text"].fillna("").str.len().gt(0).astype(int)
    df_ext["_is_bsard"]   = df_ext["bsard_id"].notna().astype(int)
    df_ext["_text_len"]   = df_ext["article_text"].fillna("").str.len()
    df_ext["_page_start"] = df_ext["pdf_page_start"].fillna(-1).astype(int)

    # Sort ascending on score dimensions so that drop_duplicates(keep='last')
    # retains the highest-score row in each duplicate group.
    df_ext = df_ext.sort_values(
        by=["pdf_filename", "article_number",
            "_has_text", "_is_bsard", "_text_len", "_page_start"],
        ascending=[True, True, True, True, True, True],
    )

    df_clean = (
        df_ext.drop_duplicates(subset=["pdf_filename", "article_number"], keep="last")
              .drop(columns=["_has_text", "_is_bsard", "_text_len", "_page_start"])
              .sort_values("article_id")
              .reset_index(drop=True)
    )

    n_dupes = len(df_ext) - len(df_clean)
    print(f"  {n_dupes:,} duplicate rows collapsed "
          f"({n_dupes / len(df_ext) * 100:.1f}% of Phase A rows)")
    print(f"  {len(df_clean):,} unique articles in clean dataset")

    # ── 4. Analysis ────────────────────────────────────────────────────────
    n_total       = len(df_clean)
    n_bsard       = int(df_clean["is_bsard_article"].sum())
    n_nonbsard    = n_total - n_bsard
    n_unique_bid  = int(df_clean["bsard_id"].nunique())
    n_with_text   = int(df_clean["article_text"].notna().sum())
    n_null_text   = n_total - n_with_text

    print("\n" + "=" * 60)
    print("CLEAN DATASET — SUMMARY")
    print("=" * 60)
    print(f"Total unique articles       : {n_total:,}")
    print(f"  BSARD-linked              : {n_bsard:,}  "
          f"({n_bsard / n_total * 100:.1f}%)")
    print(f"  Non-BSARD                 : {n_nonbsard:,}  "
          f"({n_nonbsard / n_total * 100:.1f}%)")
    print(f"  Unique bsard_id values    : {n_unique_bid:,}")
    print(f"  Rows with article_text    : {n_with_text:,}  "
          f"({n_with_text / n_total * 100:.1f}%)")
    print(f"  Rows with NULL text       : {n_null_text:,}  "
          f"(range headers, abrogation-only, etc.)")

    print(f"\nvs source corpus:")
    print(f"  Source rows               : {n_initial:,}")
    print(f"  Clean rows                : {n_total:,}  "
          f"(-{n_initial - n_total:,}, {(n_initial - n_total) / n_initial * 100:.1f}%)")

    # Per-law-code breakdown
    by_code = (
        df_clean.groupby("law_code")
                .agg(total=("article_id", "count"),
                     bsard=("is_bsard_article", "sum"))
                .assign(non_bsard=lambda d: d["total"] - d["bsard"])
                .sort_values("total", ascending=False)
    )
    print(f"\nPer-law-code breakdown ({len(by_code)} codes):")
    print(f"  {'law_code':<60s}  {'total':>6s}  {'bsard':>6s}  {'non-b':>6s}")
    for code, row in by_code.iterrows():
        print(f"  {code[:60]:<60s}  {row['total']:>6d}  "
              f"{row['bsard']:>6d}  {row['non_bsard']:>6d}")

    # Source-of-text breakdown
    by_src = df_clean["article_text_source"].value_counts()
    print(f"\narticle_text_source distribution:")
    for src_name, n in by_src.items():
        print(f"  {src_name:<25s} {n:>6,d}")

    # ── 5. Write clean SQLite DB ───────────────────────────────────────────
    print(f"\nWriting {args.dst_db} ...")
    if args.dst_db.exists():
        args.dst_db.unlink()

    dst = sqlite3.connect(str(args.dst_db))
    dst.execute("PRAGMA journal_mode=WAL")
    dst.execute("PRAGMA synchronous=NORMAL")
    dst.execute(table_ddl)

    # Insert rows preserving original article_id values.
    cols = list(df_clean.columns)
    placeholders = ",".join(["?"] * len(cols))
    insert_sql = f"INSERT INTO articles ({','.join(cols)}) VALUES ({placeholders})"
    rows = [tuple(None if pd.isna(v) else v for v in r)
            for r in df_clean.itertuples(index=False, name=None)]
    dst.executemany(insert_sql, rows)

    # Indices mirroring the main corpus.
    dst.execute("CREATE INDEX idx_clean_bsard_id  ON articles(bsard_id)")
    dst.execute("CREATE INDEX idx_clean_law_code  ON articles(law_code)")
    dst.execute("CREATE INDEX idx_clean_pdf_file  ON articles(pdf_filename)")
    dst.execute("CREATE INDEX idx_clean_is_bsard  ON articles(is_bsard_article)")
    dst.execute("CREATE INDEX idx_clean_art_status ON articles(article_status)")

    # FTS5 virtual table over article_text.
    dst.execute("""
        CREATE VIRTUAL TABLE articles_fts USING fts5(
            article_text,
            content='articles',
            content_rowid='article_id',
            tokenize='unicode61'
        )
    """)
    dst.execute("INSERT INTO articles_fts(articles_fts) VALUES('rebuild')")

    dst.commit()
    dst.close()
    print(f"  SQLite written ({args.dst_db.stat().st_size // 1024:,} KB)")

    # ── 6. Exports ─────────────────────────────────────────────────────────
    print(f"Writing {args.dst_parquet} ...")
    df_clean.to_parquet(args.dst_parquet, index=False, engine="pyarrow")
    print(f"  Parquet written ({args.dst_parquet.stat().st_size // 1024:,} KB)")

    print(f"Writing {args.dst_jsonl} ...")
    df_clean.to_json(args.dst_jsonl, orient="records", lines=True,
                     force_ascii=False)
    print(f"  JSONL written ({args.dst_jsonl.stat().st_size // 1024:,} KB)")

    print("\nDone.")


if __name__ == "__main__":
    main()
