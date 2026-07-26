"""Chunking strategies for Arm 1 (naive, structure-agnostic).

Two strategies are provided:
  - chunk_sliding_window: fixed-size overlapping token windows
  - chunk_recursive:      paragraph-first splitting, sentence fallback for long paragraphs

Both strategies annotate each Chunk with the BSARD bsard_ids that overlap its
character range.  These bsard_ids are required by the weighted evaluation metrics
in T07_EVALUATION.

BSARD article identification
-----------------------------
locate_articles() re-runs article-boundary detection (Art. X regex) on the raw PDF
text and joins results against the corpus DB to map article numbers to bsard_ids.
Non-BSARD articles (bsard_id IS NULL) are skipped: chunks overlapping only
non-BSARD content carry an empty bsard_ids list and contribute no recall.

Limitation: the ART_RE regex will match any "Art. X" occurrence in the text
(including table-of-contents entries).  For structure-agnostic Arm 1 this is an
acceptable approximation; Arm 2 uses structural-aware extraction.
"""

from __future__ import annotations

import re
import sqlite3
import unicodedata
from pathlib import Path
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerFast

# ── Article-marker regex (mirrors Dataset_Creation/extract_articles_from_pdf.py) ─

_ART_RE = re.compile(
    r"^[ \t]*Art\.?\s+([A-Z0-9][A-Z0-9_./-]*)",
    re.MULTILINE | re.IGNORECASE,
)

_SENT_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


# ── Data models ───────────────────────────────────────────────────────────────────


@dataclass
class ArticleSpan:
    """Character-level position of a BSARD article within a document's raw text.

    Identified by bsard_id (the canonical BSARD article id). Non-BSARD articles
    are not represented as ArticleSpans — locate_articles() skips them.
    """

    bsard_id: int
    start_char: int
    end_char: int


@dataclass
class Chunk:
    """A contiguous text segment produced by a chunking strategy."""

    chunk_id: str
    text: str
    doc_id: str
    start_char: int
    end_char: int
    start_token: int
    end_token: int
    bsard_ids: list[int] = field(default_factory=list)


# ── Helpers ───────────────────────────────────────────────────────────────────────


def _normalise_key(text: str) -> str:
    """Accent-stripped, lowercase key — mirrors link_bsard.py normalise_key."""
    nfkd = unicodedata.normalize("NFKD", str(text))
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower().strip()


def _identify_bsard_ids(
    start_char: int, end_char: int, spans: list[ArticleSpan]
) -> list[int]:
    """Return bsard_ids whose span overlaps [start_char, end_char)."""
    return [
        s.bsard_id
        for s in spans
        if s.start_char < end_char and s.end_char > start_char
    ]


def _count_tokens(text: str, tokenizer: "PreTrainedTokenizerFast") -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def _split_sentences(
    text: str,
    text_offset: int,
    max_tokens: int,
    tokenizer: "PreTrainedTokenizerFast",
) -> list[tuple[str, int, int]]:
    """Group sentences to stay under max_tokens. Returns (text, start_char, end_char)."""
    parts: list[tuple[str, int, int]] = []
    prev = 0
    for m in _SENT_BOUNDARY.finditer(text):
        part = text[prev : m.end()]
        parts.append((part, text_offset + prev, text_offset + m.end()))
        prev = m.end()
    if prev < len(text):
        parts.append((text[prev:], text_offset + prev, text_offset + len(text)))

    groups: list[tuple[str, int, int]] = []
    buf = ""
    buf_start = buf_end = 0

    for part_text, ps, pe in parts:
        if not buf:
            buf, buf_start, buf_end = part_text, ps, pe
        elif _count_tokens(buf + part_text, tokenizer) <= max_tokens:
            buf += part_text
            buf_end = pe
        else:
            groups.append((buf, buf_start, buf_end))
            buf, buf_start, buf_end = part_text, ps, pe

    if buf:
        groups.append((buf, buf_start, buf_end))

    return groups


# ── BSARD article locator ─────────────────────────────────────────────────────────


def locate_articles(
    raw_text: str,
    pdf_filename: str,
    db_conn: sqlite3.Connection,
    *,
    allowed_bsard_ids: "set[int] | None" = None,
) -> list[ArticleSpan]:
    """Find BSARD article character spans within raw PDF text.

    Algorithm:
      1. Scan raw_text for all "Art. X" markers using _ART_RE.
      2. Each marker's span = [marker_start, next_marker_start).
      3. Look up (pdf_filename, article_number) in the corpus DB to get article_id.
      4. Return only markers that resolve to a known article_id.

    The corpus DB lookup uses normalised article numbers with the same strategy as
    link_bsard.py: exact match → leading-integer match (unique only).

    If ``allowed_bsard_ids`` is provided, spans whose bsard_id is not in that set
    are dropped. Used by :class:`PdfCache.load_or_locate_articles` to apply a
    PDF-level drift filter (skip ToC phantoms and abrogated-body markers whose
    bsard_id has been judged not present in the canonical PDF — see
    ``drift_status.json``).
    """
    markers: list[tuple[str, int]] = []
    for m in _ART_RE.finditer(raw_text):
        art_no = m.group(1).rstrip(".")
        markers.append((art_no, m.start()))

    if not markers:
        return []

    rows = db_conn.execute(
        "SELECT bsard_id, article_number FROM articles "
        "WHERE pdf_filename = ? AND bsard_id IS NOT NULL",
        (pdf_filename,),
    ).fetchall()

    # Fallback: DB may store filenames as "img_l_pdf_<stem>_F.pdf"
    if not rows:
        stem = Path(pdf_filename).stem
        db_filename = f"img_l_pdf_{stem}_F.pdf"
        rows = db_conn.execute(
            "SELECT bsard_id, article_number FROM articles "
            "WHERE pdf_filename = ? AND bsard_id IS NOT NULL",
            (db_filename,),
        ).fetchall()

    exact: dict[str, int] = {}
    lead: dict[str, list[int]] = {}

    for bsard_id, art_no in rows:
        if not art_no:
            continue
        nk = _normalise_key(art_no)
        exact[nk] = bsard_id
        li = re.match(r"^(\d+)", art_no)
        if li:
            lead.setdefault(li.group(1), []).append(bsard_id)

    spans: list[ArticleSpan] = []
    for i, (art_no, start) in enumerate(markers):
        end = markers[i + 1][1] if i + 1 < len(markers) else len(raw_text)
        nk = _normalise_key(art_no)

        bsard_id = exact.get(nk)
        if bsard_id is None:
            li = re.match(r"^(\d+)", art_no)
            if li:
                candidates = lead.get(li.group(1), [])
                if len(candidates) == 1:
                    bsard_id = candidates[0]

        if bsard_id is not None:
            if allowed_bsard_ids is not None and bsard_id not in allowed_bsard_ids:
                continue
            spans.append(ArticleSpan(bsard_id=bsard_id, start_char=start, end_char=end))

    return spans


# ── Chunking strategies ───────────────────────────────────────────────────────────


def chunk_sliding_window(
    text: str,
    tokenizer: "PreTrainedTokenizerFast",
    doc_id: str,
    article_spans: list[ArticleSpan] | None = None,
    window_size: int = 512,
    stride: int = 256,
) -> list[Chunk]:
    """Fixed-size overlapping sliding-window chunker.

    Uses the tokenizer's offset mapping to track exact character positions so that
    article_ids can be identified per chunk and so that chunk boundaries can be
    reconstructed from the original text without token-decoding artefacts.

    Args:
        text:          Raw document text.
        tokenizer:     A HuggingFace fast tokenizer (PreTrainedTokenizerFast).
        doc_id:        Document identifier used to prefix chunk IDs.
        article_spans: BSARD article character spans (from locate_articles()).
        window_size:   Chunk size in tokens.
        stride:        Step size in tokens between windows.

    Returns:
        List of Chunk objects with start/end token and character positions.
    """
    if article_spans is None:
        article_spans = []

    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=False,
        truncation=False,
    )
    offsets: list[tuple[int, int]] = encoding["offset_mapping"]
    n = len(offsets)

    if n == 0:
        return []

    chunks: list[Chunk] = []
    i = 0

    while i < n:
        end_i = min(i + window_size, n)
        start_char = offsets[i][0]
        end_char = offsets[end_i - 1][1]

        chunks.append(
            Chunk(
                chunk_id=f"{doc_id}_sw_{len(chunks)}",
                text=text[start_char:end_char],
                doc_id=doc_id,
                start_char=start_char,
                end_char=end_char,
                start_token=i,
                end_token=end_i,
                bsard_ids=_identify_bsard_ids(start_char, end_char, article_spans),
            )
        )

        if end_i == n:
            break
        i += stride

    return chunks


def chunk_recursive(
    text: str,
    tokenizer: "PreTrainedTokenizerFast",
    doc_id: str,
    article_spans: list[ArticleSpan] | None = None,
    max_tokens: int = 512,
) -> list[Chunk]:
    """Paragraph-first recursive chunker.

    Splits text at double-newline paragraph boundaries, accumulating paragraphs
    into chunks up to max_tokens.  Paragraphs longer than max_tokens are split
    further at sentence boundaries.

    Args:
        text:          Raw document text.
        tokenizer:     A HuggingFace fast tokenizer (PreTrainedTokenizerFast).
        doc_id:        Document identifier used to prefix chunk IDs.
        article_spans: BSARD article character spans (from locate_articles()).
        max_tokens:    Maximum tokens per chunk.

    Returns:
        List of Chunk objects with start/end token and character positions.
    """
    if article_spans is None:
        article_spans = []

    # Split into paragraphs while tracking character offsets.
    para_pattern = re.compile(r"\n\n+")
    paragraphs: list[tuple[str, int, int]] = []
    prev_end = 0
    for m in para_pattern.finditer(text):
        para = text[prev_end : m.start()]
        if para.strip():
            paragraphs.append((para, prev_end, m.start()))
        prev_end = m.end()
    if prev_end < len(text):
        remaining = text[prev_end:]
        if remaining.strip():
            paragraphs.append((remaining, prev_end, len(text)))

    chunks: list[Chunk] = []
    running_tokens = 0

    def _flush(t: str, sc: int, ec: int) -> None:
        nonlocal running_tokens
        t = t.strip()
        if not t:
            return
        n_tok = _count_tokens(t, tokenizer)
        chunks.append(
            Chunk(
                chunk_id=f"{doc_id}_rec_{len(chunks)}",
                text=t,
                doc_id=doc_id,
                start_char=sc,
                end_char=ec,
                start_token=running_tokens,
                end_token=running_tokens + n_tok,
                bsard_ids=_identify_bsard_ids(sc, ec, article_spans),
            )
        )
        running_tokens += n_tok

    buf_text = ""
    buf_start = buf_end = 0
    buf_tokens = 0

    for para_text, para_start, para_end in paragraphs:
        para_tokens = _count_tokens(para_text, tokenizer)

        if buf_tokens + para_tokens <= max_tokens:
            if not buf_text:
                buf_start = para_start
            buf_text += para_text + "\n\n"
            buf_end = para_end
            buf_tokens += para_tokens
        else:
            if buf_text:
                _flush(buf_text, buf_start, buf_end)
                buf_text, buf_tokens = "", 0

            if para_tokens > max_tokens:
                for sent_text, ss, se in _split_sentences(
                    para_text, para_start, max_tokens, tokenizer
                ):
                    _flush(sent_text, ss, se)
            else:
                buf_text = para_text + "\n\n"
                buf_start = para_start
                buf_end = para_end
                buf_tokens = para_tokens

    if buf_text:
        _flush(buf_text, buf_start, buf_end)

    return chunks
