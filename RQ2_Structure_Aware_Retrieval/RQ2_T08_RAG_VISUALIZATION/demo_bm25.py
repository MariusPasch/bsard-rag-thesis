"""
BM25-only retrieval demo for the RAG Visualization viewer.

Chunks a PDF, builds an in-memory BM25 index (no GPU / no embeddings needed),
runs retrieval for a chosen question, then launches the Streamlit viewer.

Usage:
    python demo_bm25.py
    python demo_bm25.py --pdf 2004_05_27_2004A27101.pdf --question-id 38
    python demo_bm25.py --question-id 37 --top-k 10
"""

from __future__ import annotations

import os

import argparse
import json
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

import fitz  # PyMuPDF

# ── Resolve sibling packages ─────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent  # ...RQ2_Structure_Aware_Retrieval

for _sibling in _REPO.glob("RQ2_T0*/src"):
    _s = str(_sibling)
    if _s not in sys.path:
        sys.path.insert(0, _s)

import re
from dataclasses import dataclass

from arm1_naive.chunker import chunk_sliding_window, locate_articles
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer

# Minimal French stopwords (mirrors shared/bm25_store.py)
_STOPWORDS = {
    "le","la","les","de","du","des","un","une","et","en","au","aux","à","a",
    "ce","ces","cet","cette","qui","que","qu","se","sa","son","ses","il","elle",
    "ils","elles","y","ou","où","est","sont","par","sur","sous","dans","avec",
    "pour","plus","mais","ni","car","ne","pas","non","tout","tous","même","aussi",
}
_TOK_RE = re.compile(r"[^\w]+", re.UNICODE)

def _tok(text: str) -> list[str]:
    return [t for t in _TOK_RE.split(text.lower()) if t and t not in _STOPWORDS]

@dataclass
class _Hit:
    id: str
    score: float
    metadata: dict

def bm25_search(ids, texts, metadata, query: str, k: int) -> list[_Hit]:
    corpus = [_tok(t) for t in texts]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(_tok(query))
    order = scores.argsort()[::-1]
    return [_Hit(id=ids[i], score=float(scores[i]), metadata=metadata[i]) for i in order[:k]]

# ── Hardcoded paths (same pattern as verify_chunking.py) ─────────────────────
DB_PATH = Path(os.environ.get("RQ2_BSARD_DB", str(_REPO / "data" / "bsard_corpus.db")))
PDF_DIR = _REPO / "RQ2_T02_DATA_LOADER/data/pdfs"
APP_PY = _HERE / "src/visualization/app.py"
VENV_STREAMLIT = _HERE / ".venv/Scripts/streamlit.exe"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", default="2004_05_27_2004A27101.pdf")
    parser.add_argument("--question-id", type=int, default=38)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--window-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    pdf_path = PDF_DIR / args.pdf
    if not pdf_path.exists():
        sys.exit(f"PDF not found: {pdf_path}")

    # ── 1. Extract text ───────────────────────────────────────────────────────
    print(f"[1] Extracting text from {args.pdf} …")
    doc = fitz.open(str(pdf_path))
    raw_text = "\n\n".join(page.get_text() for page in doc)
    print(f"    {len(doc)} pages, {len(raw_text):,} chars")

    # ── 2. Locate articles ────────────────────────────────────────────────────
    print("[2] Locating BSARD articles …")
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    pdf_filename = args.pdf
    article_spans = locate_articles(raw_text, pdf_filename, conn)
    print(f"    {len(article_spans)} article spans found")

    # ── 3. Load question ──────────────────────────────────────────────────────
    print(f"[3] Loading question {args.question_id} …")
    q_row = conn.execute(
        "SELECT question_id, question_text, relevant_article_ids FROM questions WHERE question_id=?",
        (args.question_id,),
    ).fetchone()
    if q_row is None:
        conn.close()
        sys.exit(f"Question {args.question_id} not found in DB.")

    q_text = q_row["question_text"]
    try:
        gt_article_ids = json.loads(q_row["relevant_article_ids"])
    except Exception:
        gt_article_ids = []
    print(f"    Q: {q_text}")
    print(f"    GT article_ids: {gt_article_ids}")

    conn.close()

    # ── 4. Chunk ──────────────────────────────────────────────────────────────
    print(f"[4] Chunking (sliding_window {args.window_size}/{args.stride}) …")
    tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large-instruct")
    doc_stem = Path(args.pdf).stem
    chunks = chunk_sliding_window(
        raw_text, tokenizer, doc_stem, article_spans,
        window_size=args.window_size, stride=args.stride,
    )
    print(f"    {len(chunks)} chunks produced")

    # ── 5 + 6. BM25 retrieval (no index object needed) ───────────────────────
    print(f"[5] Retrieving top-{args.top_k} with BM25 (no embeddings) …")
    chunk_ids = [c.chunk_id for c in chunks]
    chunk_texts = [c.text for c in chunks]
    chunk_meta = [
        {
            "text": c.text,
            "doc_id": c.doc_id,
            "start_char": c.start_char,
            "end_char": c.end_char,
            "start_token": c.start_token,
            "end_token": c.end_token,
            "article_ids": c.article_ids,
        }
        for c in chunks
    ]
    results = bm25_search(chunk_ids, chunk_texts, chunk_meta, q_text, k=args.top_k)
    print(f"    {len(results)} results returned")
    for i, r in enumerate(results, 1):
        art_ids = r.metadata.get("article_ids", [])
        hit = "[HIT]" if set(art_ids) & set(gt_article_ids) else "[miss]"
        print(f"    {i:2d}. {r.id} score={r.score:.4f} article_ids={art_ids} {hit}")

    # ── 7. Build bundle JSON side-file ────────────────────────────────────────
    print("[7] Preparing viewer bundle …")
    ranked_items = [
        {
            "id": r.id,
            "score": r.score,
            "text": r.metadata.get("text", ""),
            "article_ids": r.metadata.get("article_ids", []),
            "start_char": r.metadata.get("start_char"),
            "end_char": r.metadata.get("end_char"),
            "start_token": r.metadata.get("start_token"),
            "end_token": r.metadata.get("end_token"),
            "metadata": r.metadata,
        }
        for r in results
    ]

    retrieval_result = {
        "query_id": str(args.question_id),
        "query_text": q_text,
        "ranked_items": ranked_items,
        "method": "arm1_bm25",
        "cost": {},
    }

    bundle_json = {
        "document_id": 0,
        "pdf_path": str(pdf_path),
        "pdf_filename": pdf_filename,
        "metadata": {},
        "definitions": [],
        "articles": [],
        "raw_text": raw_text,
        "has_azure_extraction": False,
        # Extra fields read by app.py startup
        "db_path": str(DB_PATH),
        "initial_mode": "retrieval",
        "initial_arm": "arm1_bm25",
        "retrieval_result": retrieval_result,
        "gt_article_ids": gt_article_ids,
        "active_query_text": q_text,
        "active_query_id": str(args.question_id),
        "top_k": args.top_k,
    }

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    )
    json.dump(bundle_json, tmp, ensure_ascii=False, indent=2)
    tmp.close()
    print(f"    Side-file: {tmp.name}")

    # ── 8. Launch Streamlit ───────────────────────────────────────────────────
    streamlit_exe = str(VENV_STREAMLIT) if VENV_STREAMLIT.exists() else "streamlit"
    cmd = [
        streamlit_exe, "run", str(APP_PY),
        "--",
        f"--bundle-json={tmp.name}",
        f"--db={DB_PATH}",
        "--mode=retrieval",
        "--arm=arm1_bm25",
    ]
    if args.no_browser:
        cmd = [streamlit_exe, "run", str(APP_PY),
               "--server.headless=true",
               "--",
               f"--bundle-json={tmp.name}",
               f"--db={DB_PATH}",
               "--mode=retrieval",
               "--arm=arm1_bm25",
               ]

    print(f"\n[8] Launching viewer …\n    {' '.join(cmd[:4])} …\n")
    subprocess.run(cmd)


if __name__ == "__main__":
    main()
