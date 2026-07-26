"""
Verify chunking and chunk-to-article allocation for a single PDF.

Runs:
  1. PDF text extraction (PyMuPDF)
  2. Article location (locate_articles)
  3. Chunking (sliding_window or recursive)
  4. Prints a summary + sample chunks with their article_ids

Does NOT embed, index, or retrieve anything.

Usage:
    python scripts/verify_chunking.py --pdf 2004_05_27_2004A27101.pdf
    python scripts/verify_chunking.py --pdf 2004_05_27_2004A27101.pdf --strategy recursive
    python scripts/verify_chunking.py --pdf 2004_05_27_2004A27101.pdf --samples 5
"""

import argparse
import os
import sqlite3
from pathlib import Path

import fitz  # PyMuPDF

from arm1_naive.chunker import chunk_recursive, chunk_sliding_window, locate_articles

DB_PATH = Path(
    os.environ.get(
        "RQ2_BSARD_DB",
        str(Path(__file__).resolve().parent.parent.parent / "data" / "bsard_corpus.db"),
    )
)
PDF_DIR = Path(__file__).parent.parent.parent / "RQ2_T02_DATA_LOADER/data/pdfs"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True, help="PDF filename (basename only)")
    parser.add_argument("--strategy", default="sliding_window", choices=["sliding_window", "recursive"])
    parser.add_argument("--window-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--samples", type=int, default=3, help="Number of sample chunks to print")
    args = parser.parse_args()

    pdf_path = PDF_DIR / args.pdf
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # ── 1. Extract raw text ───────────────────────────────────────────────────
    print(f"\n[1] Extracting text from {args.pdf} ...")
    doc = fitz.open(str(pdf_path))
    raw_text = "\n\n".join(page.get_text() for page in doc)
    print(f"    {len(doc)} pages, {len(raw_text):,} chars")

    # ── 2. Locate articles ────────────────────────────────────────────────────
    print(f"\n[2] Locating BSARD articles in text ...")
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    article_spans = locate_articles(raw_text, args.pdf, conn)
    print(f"    {len(article_spans)} article spans found")

    if article_spans:
        sample_ids = [s.article_id for s in article_spans[:5]]
        rows = conn.execute(
            f"SELECT article_id, article_number FROM articles WHERE article_id IN ({','.join('?'*len(sample_ids))})",
            sample_ids,
        ).fetchall()
        id_to_num = {r["article_id"]: r["article_number"] for r in rows}
        print("    First 5 spans:")
        for s in article_spans[:5]:
            print(f"      article_id={s.article_id}  article_number={id_to_num.get(s.article_id,'?')!r:20s}  chars [{s.start_char:6d}, {s.end_char:6d})")

    conn.close()

    # ── 3. Chunk ──────────────────────────────────────────────────────────────
    print(f"\n[3] Chunking with strategy={args.strategy!r} ...")

    # Use a lightweight tokenizer just for token counting (no model weights needed)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large-instruct")

    if args.strategy == "sliding_window":
        chunks = chunk_sliding_window(
            raw_text, tokenizer, Path(args.pdf).stem, article_spans,
            window_size=args.window_size, stride=args.stride,
        )
    else:
        chunks = chunk_recursive(
            raw_text, tokenizer, Path(args.pdf).stem, article_spans,
            max_tokens=args.max_tokens,
        )

    n_annotated = sum(1 for c in chunks if c.article_ids)
    print(f"    {len(chunks)} chunks produced")
    print(f"    {n_annotated} chunks have at least one article_id ({n_annotated/len(chunks)*100:.1f}%)")

    article_id_coverage = len({aid for c in chunks for aid in c.article_ids})
    print(f"    {article_id_coverage} distinct article_ids covered out of {len(article_spans)} located spans")

    # ── 4. Sample output ──────────────────────────────────────────────────────
    annotated = [c for c in chunks if c.article_ids]
    unannotated = [c for c in chunks if not c.article_ids]

    print(f"\n[4] Sample annotated chunks ({min(args.samples, len(annotated))}):")
    for c in annotated[:args.samples]:
        preview = c.text[:120].replace("\n", " ")
        print(f"  [{c.chunk_id}]  article_ids={c.article_ids}  tokens=[{c.start_token},{c.end_token}]")
        print(f"    {preview!r}")

    if unannotated:
        print(f"\n    Sample unannotated chunk:")
        c = unannotated[0]
        preview = c.text[:120].replace("\n", " ")
        print(f"  [{c.chunk_id}]  article_ids=[]  tokens=[{c.start_token},{c.end_token}]")
        print(f"    {preview!r}")

    print("\nDone.")


if __name__ == "__main__":
    main()
