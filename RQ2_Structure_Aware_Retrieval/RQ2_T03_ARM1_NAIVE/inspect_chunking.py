"""Chunking inspection script for ARM 1 NAIVE.

Cache-aware: raw text and article spans are loaded from
``data/<doc_id>/{raw_text.txt, article_spans.json}`` when a valid cache exists
(CACHE_DESIGN.md §4). Pass ``--use-cached-chunks`` to additionally load the
sliding-window and recursive chunk lists from
``data/<doc_id>/configs/<config_hash>/chunks.json``, which lets the whole
inspection run in <2s after the first build.

Hardcoded defaults: 1804_03_21_1804032150.pdf, question_id=192.

Writes results to ``inspect_output/`` as CSV + HTML.

Usage (from RQ2_T03_ARM1_NAIVE/, with the venv active):
    python inspect_chunking.py
    python inspect_chunking.py --use-cached-chunks
    python inspect_chunking.py --pdf <name>.pdf --question-id <qid>
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sqlite3
import sys
from pathlib import Path

import fitz  # PyMuPDF

# ── Resolve sibling packages ──────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent

for _sibling in _REPO.glob("RQ2_T0*/src"):
    _s = str(_sibling)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from arm1_naive.cache import (
    CacheRoot,
    ConfigCache,
    PdfCache,
    compute_bsard_db_fingerprint,
    compute_config_hash,
    compute_pdf_sha256,
)
from arm1_naive.chunker import (
    ArticleSpan,
    Chunk,
    chunk_recursive,
    chunk_sliding_window,
    locate_articles,
)

# ── Config ────────────────────────────────────────────────────────────────────
DB_PATH        = Path(
    os.environ.get(
        "RQ2_BSARD_DB",
        str(_REPO / "data" / "bsard_corpus.db"),
    )
)
PDF_DIR        = _REPO / "RQ2_T02_DATA_LOADER/data/pdfs"
TOKENIZER_NAME = "intfloat/multilingual-e5-large-instruct"
CACHE_ROOT     = _HERE / "data"
OUT_DIR        = _HERE / "inspect_output"

DEFAULT_PDF         = "1804_03_21_1804032150.pdf"
DEFAULT_QUESTION_ID = 192
WINDOW_SIZE         = 512
STRIDE              = 256
MAX_TOKENS          = 512


# ── CSV writers ───────────────────────────────────────────────────────────────

def write_article_spans_csv(spans: list[ArticleSpan], raw_text: str, path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["bsard_id", "start_char", "end_char", "char_length", "text_preview"])
        for s in spans:
            preview = raw_text[s.start_char : s.start_char + 120].replace("\n", " ").strip()
            w.writerow([s.bsard_id, s.start_char, s.end_char,
                        s.end_char - s.start_char, preview])
    print(f"    → {path.name}")


def write_chunks_csv(
    chunks: list[Chunk],
    gt_bsard_ids: list[int],
    path: Path,
) -> None:
    gt_set = set(gt_bsard_ids)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "index", "chunk_id", "start_token", "end_token", "token_count",
            "start_char", "end_char", "char_count",
            "bsard_ids", "n_bsard_ids", "gt_hit", "text_preview", "text_full",
        ])
        for idx, c in enumerate(chunks):
            tok_count  = c.end_token - c.start_token
            char_count = c.end_char - c.start_char
            gt_hit     = 1 if set(c.bsard_ids) & gt_set else 0
            preview    = c.text[:200].replace("\n", " ").strip()
            w.writerow([
                idx, c.chunk_id,
                c.start_token, c.end_token, tok_count,
                c.start_char, c.end_char, char_count,
                json.dumps(c.bsard_ids), len(c.bsard_ids),
                gt_hit, preview, c.text,
            ])
    print(f"    → {path.name}")


# ── HTML report ───────────────────────────────────────────────────────────────

_HTML_STYLE = """
<style>
  body { font-family: sans-serif; font-size: 13px; margin: 20px; }
  h2   { margin-top: 2em; color: #333; }
  table { border-collapse: collapse; width: 100%; margin-bottom: 1em; }
  th   { background: #4472C4; color: white; padding: 6px 10px; text-align: left; }
  td   { padding: 5px 10px; border-bottom: 1px solid #ddd; vertical-align: top; }
  tr.gt-hit  td { background: #e2efda; }
  tr.unassigned td { background: #fce4d6; }
  tr:hover td  { background: #fffbcc; }
  .mono { font-family: monospace; font-size: 12px; }
  .pill { display: inline-block; background: #4472C4; color: white;
          border-radius: 4px; padding: 1px 6px; margin: 1px; font-size: 11px; }
  .pill.gt { background: #375623; }
  .summary { background: #f2f2f2; padding: 12px; border-radius: 6px;
             margin-bottom: 1em; line-height: 1.8; }
</style>
"""


def _art_pills(bsard_ids: list[int], gt_set: set[int]) -> str:
    if not bsard_ids:
        return "<em style='color:#999'>none</em>"
    pills = []
    for a in bsard_ids:
        cls = "pill gt" if a in gt_set else "pill"
        pills.append(f'<span class="{cls}">{a}</span>')
    return " ".join(pills)


def write_html_report(
    chunks_sw: list[Chunk],
    chunks_rec: list[Chunk],
    article_spans: list[ArticleSpan],
    gt_bsard_ids: list[int],
    q_text: str,
    raw_text: str,
    pdf_name: str,
    question_id: int,
    path: Path,
) -> None:
    gt_set = set(gt_bsard_ids)

    def _chunk_rows(chunks: list[Chunk]) -> str:
        rows = []
        for idx, c in enumerate(chunks):
            tok_count  = c.end_token - c.start_token
            gt_hit     = bool(set(c.bsard_ids) & gt_set)
            unassigned = not c.bsard_ids
            row_cls    = "gt-hit" if gt_hit else ("unassigned" if unassigned else "")
            preview    = c.text[:300].replace("<", "&lt;").replace(">", "&gt;").replace("\n", " ").strip()
            pills      = _art_pills(c.bsard_ids, gt_set)
            rows.append(
                f'<tr class="{row_cls}">'
                f'<td>{idx}</td>'
                f'<td class="mono">{c.chunk_id}</td>'
                f'<td>{tok_count}</td>'
                f'<td>{c.start_char:,}–{c.end_char:,}</td>'
                f'<td>{pills}</td>'
                f'<td>{"✔" if gt_hit else ""}</td>'
                f'<td>{preview}</td>'
                f'</tr>'
            )
        return "\n".join(rows)

    def _span_rows() -> str:
        rows = []
        for s in article_spans:
            preview = raw_text[s.start_char : s.start_char + 120].replace("<", "&lt;").replace(">", "&gt;").replace("\n", " ").strip()
            gt_cls  = ' class="gt-hit"' if s.bsard_id in gt_set else ""
            rows.append(
                f'<tr{gt_cls}>'
                f'<td>{s.bsard_id}</td>'
                f'<td>{s.start_char:,}</td>'
                f'<td>{s.end_char:,}</td>'
                f'<td>{s.end_char - s.start_char:,}</td>'
                f'<td>{preview}</td>'
                f'</tr>'
            )
        return "\n".join(rows)

    def _summary(chunks: list[Chunk], label: str) -> str:
        hit  = sum(1 for c in chunks if set(c.bsard_ids) & gt_set)
        unassigned = sum(1 for c in chunks if not c.bsard_ids)
        tok_counts = [c.end_token - c.start_token for c in chunks]
        return (
            f'<div class="summary">'
            f'<strong>{label}</strong> &nbsp;|&nbsp; '
            f'{len(chunks)} chunks &nbsp;|&nbsp; '
            f'tokens: {min(tok_counts)}–{max(tok_counts)} (mean {sum(tok_counts)/len(tok_counts):.0f}) &nbsp;|&nbsp; '
            f'<span style="color:#375623">GT hits: {hit}</span> &nbsp;|&nbsp; '
            f'<span style="color:#c00">unassigned: {unassigned}</span>'
            f'</div>'
        )

    col_headers = "<tr><th>#</th><th>chunk_id</th><th>tokens</th><th>char range</th><th>bsard_ids</th><th>GT?</th><th>text preview</th></tr>"

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Chunking Inspection — {pdf_name} Q{question_id}</title>
{_HTML_STYLE}
</head><body>
<h1>Chunking Inspection</h1>
<p><strong>PDF:</strong> {pdf_name} &nbsp; <strong>Question:</strong> {question_id}</p>
<p><strong>Q:</strong> {q_text}</p>
<p><strong>GT bsard_ids:</strong> {" ".join(f'<span class="pill gt">{a}</span>' for a in gt_bsard_ids)}</p>
<p style="font-size:11px">
  <span style="background:#e2efda;padding:2px 8px">green = GT hit</span> &nbsp;
  <span style="background:#fce4d6;padding:2px 8px">orange = no article assigned</span>
</p>

<h2>Article Spans ({len(article_spans)} found)</h2>
<table>
  <tr><th>bsard_id</th><th>start_char</th><th>end_char</th><th>length</th><th>text preview</th></tr>
  {_span_rows()}
</table>

<h2>Sliding-Window Chunks (window={WINDOW_SIZE}, stride={STRIDE})</h2>
{_summary(chunks_sw, "sliding-window")}
<table>
  {col_headers}
  {_chunk_rows(chunks_sw)}
</table>

<h2>Recursive Chunks (max_tokens={MAX_TOKENS})</h2>
{_summary(chunks_rec, "recursive")}
<table>
  {col_headers}
  {_chunk_rows(chunks_rec)}
</table>

</body></html>"""

    path.write_text(html, encoding="utf-8")
    print(f"    → {path.name}")


# ── Cache-aware loaders ───────────────────────────────────────────────────────

def _load_raw_text(pdf_cache: PdfCache, pdf_path: Path) -> str:
    if pdf_cache.validate_pdf(pdf_path) and pdf_cache.raw_text_path().exists():
        print("    [cache hit] raw_text.txt")
        return pdf_cache.raw_text_path().read_text(encoding="utf-8")
    print("    [cache miss] extracting via PyMuPDF …")
    return pdf_cache.load_or_extract_text(pdf_path)


def _load_article_spans(
    pdf_cache: PdfCache,
    raw_text: str,
    pdf_filename: str,
    conn: sqlite3.Connection,
) -> list[ArticleSpan]:
    db_fp = compute_bsard_db_fingerprint(conn)
    spans_path = pdf_cache.article_spans_path()
    if spans_path.exists():
        try:
            payload = json.loads(spans_path.read_text(encoding="utf-8"))
            if (
                payload.get("schema_version") == 1
                and payload.get("bsard_db_fingerprint") == db_fp
            ):
                print("    [cache hit] article_spans.json")
                return pdf_cache.load_article_spans()
        except (json.JSONDecodeError, ValueError):
            pass
    print("    [cache miss] running locate_articles …")
    return pdf_cache.load_or_locate_articles(raw_text, pdf_filename, conn, db_fp)


def _try_load_cached_chunks(
    pdf_cache: PdfCache,
    pdf_sha256: str,
    chunking_params: dict,
) -> list[Chunk] | None:
    config_hash = compute_config_hash(
        pdf_sha256=pdf_sha256,
        tokenizer_name=TOKENIZER_NAME,
        embedding_model=TOKENIZER_NAME,
        chunking_params=chunking_params,
    )
    cfg = ConfigCache(pdf_cache, config_hash)
    if cfg.exists() and cfg.chunks_path().exists():
        try:
            return cfg.load_chunks()
        except (ValueError, json.JSONDecodeError):
            return None
    return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--pdf", default=DEFAULT_PDF, help="PDF filename (default: %(default)s)")
    parser.add_argument("--question-id", type=int, default=DEFAULT_QUESTION_ID,
                        help="Question id (default: %(default)s)")
    parser.add_argument("--use-cached-chunks", action="store_true",
                        help="Load chunks from configs/<hash>/chunks.json when available")
    args = parser.parse_args()

    pdf_name = args.pdf
    question_id = args.question_id

    OUT_DIR.mkdir(exist_ok=True)

    pdf_path = PDF_DIR / pdf_name
    if not pdf_path.exists():
        sys.exit(f"PDF not found: {pdf_path}")
    if not DB_PATH.exists():
        sys.exit(f"DB not found: {DB_PATH}")

    cache_root = CacheRoot(CACHE_ROOT)
    doc_id = pdf_path.stem
    pdf_cache = PdfCache(cache_root, doc_id)

    print(f"\n[1] Loading raw text for {pdf_name} …")
    raw_text = _load_raw_text(pdf_cache, pdf_path)
    print(f"    {len(raw_text):,} chars")

    print("[2] Locating BSARD articles …")
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    article_spans = _load_article_spans(pdf_cache, raw_text, pdf_name, conn)
    print(f"    {len(article_spans)} article span(s)")

    print(f"[3] Loading question {question_id} …")
    q_row = conn.execute(
        "SELECT question_text, relevant_bsard_ids FROM questions WHERE question_id = ?",
        (question_id,),
    ).fetchone()
    conn.close()

    if q_row is None:
        sys.exit(f"Question {question_id} not found in DB.")
    q_text = q_row["question_text"]
    gt_bsard_ids: list[int] = json.loads(q_row["relevant_bsard_ids"] or "[]")
    print(f"    Q: {q_text}")
    print(f"    GT bsard_ids: {gt_bsard_ids}")

    sw_chunks: list[Chunk] | None = None
    rec_chunks: list[Chunk] | None = None

    if args.use_cached_chunks:
        pdf_sha256 = compute_pdf_sha256(pdf_path)
        sw_chunks = _try_load_cached_chunks(
            pdf_cache, pdf_sha256,
            {"strategy": "sliding_window", "window_size": WINDOW_SIZE, "stride": STRIDE},
        )
        rec_chunks = _try_load_cached_chunks(
            pdf_cache, pdf_sha256,
            {"strategy": "recursive", "max_tokens": MAX_TOKENS},
        )
        if sw_chunks is not None:
            print(f"    [cache hit] sliding-window chunks ({len(sw_chunks)})")
        else:
            print("    [cache miss] sliding-window chunks — will recompute")
        if rec_chunks is not None:
            print(f"    [cache hit] recursive chunks ({len(rec_chunks)})")
        else:
            print("    [cache miss] recursive chunks — will recompute")

    if sw_chunks is None or rec_chunks is None:
        print(f"\n[4] Loading tokenizer ({TOKENIZER_NAME}) …")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

        if sw_chunks is None:
            print(f"[5a] Sliding-window chunking (window={WINDOW_SIZE}, stride={STRIDE}) …")
            sw_chunks = chunk_sliding_window(
                raw_text, tokenizer, doc_id, article_spans,
                window_size=WINDOW_SIZE, stride=STRIDE,
            )
            print(f"    {len(sw_chunks)} chunks")
        if rec_chunks is None:
            print(f"[5b] Recursive chunking (max_tokens={MAX_TOKENS}) …")
            rec_chunks = chunk_recursive(
                raw_text, tokenizer, doc_id, article_spans,
                max_tokens=MAX_TOKENS,
            )
            print(f"    {len(rec_chunks)} chunks")

    print(f"\n[6] Writing output to {OUT_DIR} …")
    write_article_spans_csv(article_spans, raw_text, OUT_DIR / "article_spans.csv")
    write_chunks_csv(sw_chunks,  gt_bsard_ids, OUT_DIR / "chunks_sliding_window.csv")
    write_chunks_csv(rec_chunks, gt_bsard_ids, OUT_DIR / "chunks_recursive.csv")
    write_html_report(sw_chunks, rec_chunks, article_spans, gt_bsard_ids,
                      q_text, raw_text, pdf_name, question_id, OUT_DIR / "report.html")

    print(f"\n{'='*60}")
    print(f"Open: {OUT_DIR / 'report.html'}")


if __name__ == "__main__":
    main()
