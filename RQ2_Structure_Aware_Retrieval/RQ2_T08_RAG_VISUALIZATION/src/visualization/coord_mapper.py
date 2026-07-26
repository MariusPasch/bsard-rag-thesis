"""char-offset → (page, bbox) mapping via PyMuPDF word index.

The pipeline builds bundle.raw_text as:
    '\\n\\n'.join(page.get_text() for page in doc)

This module maps HighlightRegion.start_char / end_char (in that flat string)
back to (page_number, bbox) tuples suitable for streamlit-pdf-viewer annotations.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import TYPE_CHECKING

import fitz  # PyMuPDF

if TYPE_CHECKING:
    from visualization.region_adapter import HighlightRegion


@dataclass
class WordSpan:
    """One word's position in both the flat text and the PDF page."""

    page_num: int                              # 0-based page index
    global_char_start: int                    # offset into bundle.raw_text
    global_char_end: int
    bbox: tuple[float, float, float, float]   # (x0, y0, x1, y1) in PyMuPDF page coords
    block_no: int
    line_no: int


# ── Index building ────────────────────────────────────────────────────────────


def build_word_index(pdf_path: str) -> list[WordSpan]:
    """Build a word-level index mapping flat-text char positions to page bboxes.

    Algorithm:
      1. Compute per-page text and cumulative offsets mirroring the pipeline's
         '\\n\\n'.join(...) construction.
      2. For each page, call get_text("words") to obtain word bboxes.
      3. Walk the page text with a forward pointer to find each word's char
         position, then offset by the page's cumulative start to get global
         char positions.

    Returns a list of WordSpan sorted by global_char_start.
    """
    doc = fitz.open(pdf_path)

    # Per-page texts exactly as in pdf_loader.py
    page_texts: list[str] = [page.get_text() for page in doc]

    # Cumulative char offsets — page i starts at page_offsets[i] in raw_text
    page_offsets: list[int] = []
    offset = 0
    for t in page_texts:
        page_offsets.append(offset)
        offset += len(t) + 2  # +2 for the "\n\n" separator

    word_spans: list[WordSpan] = []

    for page_num, page in enumerate(doc):
        page_text = page_texts[page_num]
        page_offset = page_offsets[page_num]

        # (x0, y0, x1, y1, word_text, block_no, line_no, word_no)
        raw_words = page.get_text("words")

        ptr = 0  # forward-only scanning pointer within page_text
        for entry in raw_words:
            x0, y0, x1, y1, word, block_no, line_no = (
                entry[0], entry[1], entry[2], entry[3], entry[4], entry[5], entry[6]
            )
            if not word:
                continue

            idx = page_text.find(word, ptr)
            if idx == -1:
                # Fallback: word may appear earlier (e.g., repeated short tokens)
                idx = page_text.find(word)
            if idx == -1:
                continue

            global_start = page_offset + idx
            global_end = global_start + len(word)

            word_spans.append(
                WordSpan(
                    page_num=page_num,
                    global_char_start=global_start,
                    global_char_end=global_end,
                    bbox=(x0, y0, x1, y1),
                    block_no=int(block_no),
                    line_no=int(line_no),
                )
            )
            ptr = idx + len(word)

    word_spans.sort(key=lambda ws: ws.global_char_start)
    return word_spans


# ── Region → per-page bboxes ──────────────────────────────────────────────────


def regions_to_page_bboxes(
    regions: list["HighlightRegion"],
    word_index: list[WordSpan],
) -> dict[int, list[tuple["HighlightRegion", tuple[float, float, float, float]]]]:
    """Map HighlightRegion list to per-page merged line bboxes.

    For each region:
      1. Binary-search word_index to find words overlapping [start_char, end_char).
      2. Group matching words by (page_num, block_no, line_no).
      3. Merge each line's word bboxes into one bbox covering the full line slice.

    Returns:
        dict mapping 0-based page_num → list of (region, merged_bbox) pairs.
        Multi-page regions appear under multiple page keys.
    """
    if not word_index:
        return {}

    starts = [ws.global_char_start for ws in word_index]

    result: dict[int, list[tuple["HighlightRegion", tuple[float, float, float, float]]]] = {}

    for region in regions:
        if region.start_char >= region.end_char:
            continue

        # Find first word that might overlap: search from just before region.start_char
        lo = max(0, bisect.bisect_left(starts, region.start_char) - 1)

        # Collect all words overlapping [start_char, end_char)
        matching: list[WordSpan] = []
        for i in range(lo, len(word_index)):
            ws = word_index[i]
            if ws.global_char_start >= region.end_char:
                break
            if ws.global_char_end > region.start_char:
                matching.append(ws)

        if not matching:
            continue

        # Group by (page_num, block_no, line_no) and union bboxes
        line_bboxes: dict[tuple[int, int, int], list[float]] = {}
        for ws in matching:
            key = (ws.page_num, ws.block_no, ws.line_no)
            x0, y0, x1, y1 = ws.bbox
            if key not in line_bboxes:
                line_bboxes[key] = [x0, y0, x1, y1]
            else:
                b = line_bboxes[key]
                b[0] = min(b[0], x0)
                b[1] = min(b[1], y0)
                b[2] = max(b[2], x1)
                b[3] = max(b[3], y1)

        # Emit one (region, merged_bbox) per visual line per page
        for (page_num, _block, _line), bbox_list in line_bboxes.items():
            bbox = (bbox_list[0], bbox_list[1], bbox_list[2], bbox_list[3])
            result.setdefault(page_num, []).append((region, bbox))

    return result


def get_first_page(
    region: "HighlightRegion",
    word_index: list[WordSpan],
) -> int | None:
    """Return the 0-based page number of the first word in region, or None."""
    if not word_index or region.start_char >= region.end_char:
        return None
    starts = [ws.global_char_start for ws in word_index]
    lo = max(0, bisect.bisect_left(starts, region.start_char) - 1)
    for i in range(lo, len(word_index)):
        ws = word_index[i]
        if ws.global_char_start >= region.end_char:
            break
        if ws.global_char_end > region.start_char:
            return ws.page_num
    return None
