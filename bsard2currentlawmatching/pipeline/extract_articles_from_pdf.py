"""Phase A — Full PDF article extraction using PyMuPDF.

Reads all Justel PDFs from output/pdfs/, extracts every article with its
structural hierarchy context (LIVRE / TITRE / CHAPITRE / SECTION / Art.),
and writes one JSONL file per PDF to output/extracted/.

Resumable: PDFs whose output JSONL already exists are skipped unless --force
is passed.

Usage:
    python retrieval/extract_articles_from_pdf.py
    python retrieval/extract_articles_from_pdf.py --pdf-dir output/pdfs --out-dir output/extracted
    python retrieval/extract_articles_from_pdf.py --force
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import fitz  # PyMuPDF

# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PDF_DIR = PROJECT_ROOT / "output" / "pdfs"
DEFAULT_OUT_DIR = PROJECT_ROOT / "output" / "extracted"

# ── Article number regex ───────────────────────────────────────────────────────
# Covers: Art. 1 / Art. 1bis / Art. 1.1 / Art. I.1 / Art. D.II.46 / Art. VI.4-5
# Anchored to start of string (used with .match() on stripped block text).

ART_RE = re.compile(
    r"^\s*Art\.?\s+([A-Z0-9][A-Z0-9_./-]*)",
    re.MULTILINE | re.IGNORECASE,
)

# ── Structural marker patterns ─────────────────────────────────────────────────
# French — ordered highest to lowest level.

_FR_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("LIVRE",        re.compile(r"^\s*LIVRE\s+(?:[IVX]+|\d+)\b",                                re.MULTILINE)),
    ("TITRE",        re.compile(r"^\s*TITRE\s+(?:[IVX]+|\d+)\b",                                re.MULTILINE | re.IGNORECASE)),
    ("CHAPITRE",     re.compile(r"^\s*CHAPITRE\s+(?:[IVX]+|\d+|[Ii]er)\b",                      re.MULTILINE | re.IGNORECASE)),
    ("SECTION",      re.compile(r"^\s*(?:Section|SECTION)\s+(?:\d+(?:re|ère|e)?|[IVX]+)\b",     re.MULTILINE)),
    ("SOUS-SECTION", re.compile(r"^\s*(?:Sous-section|SOUS-SECTION)\s+\d+\b",                   re.MULTILINE | re.IGNORECASE)),
]

# Dutch — same level mapping as French equivalents.

_NL_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("BOEK",          re.compile(r"^\s*BOEK\s+(?:[IVX]+|\d+)\b",                                    re.MULTILINE)),
    ("TITEL",         re.compile(r"^\s*TITEL\s+(?:[IVX]+|\d+)\b",                                   re.MULTILINE | re.IGNORECASE)),
    ("HOOFDSTUK",     re.compile(r"^\s*HOOFDSTUK\s+(?:[IVX]+|\d+|[Ii])\b",                          re.MULTILINE | re.IGNORECASE)),
    ("AFDELING",      re.compile(r"^\s*(?:Afdeling|AFDELING)\s+\d+\b",                              re.MULTILINE)),
    ("ONDERAFDELING", re.compile(r"^\s*(?:Onderafdeling|ONDERAFDELING)\s+\d+\b",                    re.MULTILINE | re.IGNORECASE)),
]

ALL_PATTERNS = _FR_PATTERNS + _NL_PATTERNS

# Hierarchy level (0 = highest) for each marker type.
_LEVEL: dict[str, int] = {
    "LIVRE": 0, "BOEK": 0,
    "TITRE": 0, "TITEL": 0,
    "CHAPITRE": 1, "HOOFDSTUK": 1,
    "SECTION": 2, "AFDELING": 2,
    "SOUS-SECTION": 3, "ONDERAFDELING": 3,
}

# Maps marker type to the DB column it populates.
_COLUMN: dict[str, str] = {
    "LIVRE": "law_title_text", "BOEK": "law_title_text",
    "TITRE": "law_title_text", "TITEL": "law_title_text",
    "CHAPITRE": "chapter_title", "HOOFDSTUK": "chapter_title",
    "SECTION": "section_title", "AFDELING": "section_title",
    "SOUS-SECTION": "subsection_title", "ONDERAFDELING": "subsection_title",
}

# ── Text normalisation ─────────────────────────────────────────────────────────

def normalise(text: str) -> str:
    """Collapse whitespace, strip soft hyphens and zero-width characters."""
    text = text.replace("\u00ad", "").replace("\u200b", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalise_key(text: str) -> str:
    """Accent-stripped lowercase key for fuzzy matching (used in Phase B)."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower().strip()

# ── Font-size analysis ─────────────────────────────────────────────────────────

def analyse_font_sizes(doc: fitz.Document) -> Counter:
    """Map font_size → total character count across all pages."""
    freq: Counter = Counter()
    for page in doc:
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    size = round(span["size"], 1)
                    freq[size] += len(span["text"].strip())
    return freq


def get_body_size(freq: Counter) -> float:
    """Body text size = the font size with the highest total character count."""
    return freq.most_common(1)[0][0] if freq else 10.0

# ── Header / footer detection ──────────────────────────────────────────────────

def detect_header_footer_lines(doc: fitz.Document, min_pages: int = 3) -> set[str]:
    """
    Return normalised text strings that appear unchanged on ≥ min_pages pages.
    These are page headers or footers that should be excluded from article text.
    """
    line_page_count: Counter = Counter()
    for page in doc:
        seen_on_page: set[str] = set()
        for block in page.get_text("blocks"):
            text = block[4].strip()
            if text and len(text) < 150:
                norm = normalise(text)
                if norm:
                    seen_on_page.add(norm)
        for line in seen_on_page:
            line_page_count[line] += 1
    return {line for line, count in line_page_count.items() if count >= min_pages}

# ── numac extraction ───────────────────────────────────────────────────────────

_NUMAC_RE = re.compile(r"_(\d{8,12})_[FfNn](?:_\w+)?\.pdf$", re.IGNORECASE)


def extract_numac(pdf_filename: str) -> str | None:
    """Extract the Moniteur belge publication ID from a Justel PDF filename."""
    m = _NUMAC_RE.search(pdf_filename)
    return m.group(1) if m else None

# ── Running state helpers ──────────────────────────────────────────────────────

def _empty_state(pdf_filename: str, numac: str | None) -> dict:
    return {
        "law_title_text": None,
        "chapter_title": None,
        "section_title": None,
        "subsection_title": None,
        "article_number": None,
        "article_lines": [],
        "article_pages": [],
        "pdf_filename": pdf_filename,
        "numac": numac,
    }


def _build_hierarchy_path(state: dict) -> list[str]:
    path = []
    for col in ("law_title_text", "chapter_title", "section_title", "subsection_title"):
        if state[col]:
            path.append(state[col])
    if state["article_number"]:
        path.append(f"Art. {state['article_number']}")
    return path


def _flush_article(state: dict) -> dict | None:
    """Serialise the current running state into an article record."""
    if state["article_number"] is None:
        return None

    raw_text = "\n".join(state["article_lines"]).strip()
    cleaned = normalise(raw_text)
    has_truncation = 1 if re.search(r"\[\s*\.\.\.\s*\]|\[…\]", cleaned) else 0

    pages = sorted(set(state["article_pages"]))
    heading_levels = [
        state["law_title_text"],
        state["chapter_title"],
        state["section_title"],
        state["subsection_title"],
    ]

    return {
        "article_number": state["article_number"],
        "law_title_text": state["law_title_text"],
        "chapter_title": state["chapter_title"],
        "section_title": state["section_title"],
        "subsection_title": state["subsection_title"],
        "hierarchy_path": _build_hierarchy_path(state),
        "hierarchy_depth": sum(1 for h in heading_levels if h is not None),
        "article_text": cleaned or None,
        "article_text_source": "pdf_extracted" if cleaned else "pdf_extraction_failed",
        "has_truncation_markers": has_truncation,
        "pdf_filename": state["pdf_filename"],
        "pdf_url": None,  # filled in Phase B from bsard_full_verify.csv
        "numac": state["numac"],
        "pdf_page_numbers": pages,
        "pdf_page_start": pages[0] if pages else None,
        "pdf_page_end": pages[-1] if pages else None,
    }

# ── Block classification helpers ───────────────────────────────────────────────

def _match_structural_marker(text: str) -> tuple[str, str] | None:
    """Return (marker_type, heading_text) if text is a structural heading, else None."""
    stripped = text.strip()
    for name, pattern in ALL_PATTERNS:
        if pattern.match(stripped):
            return name, stripped
    return None


def _is_article_marker(text: str) -> str | None:
    """Return the article number if text starts with 'Art. X', else None."""
    m = ART_RE.match(text.strip())
    if m:
        art_no = m.group(1).rstrip(".")  # strip trailing period common in Belgian law
        return art_no if art_no else None
    return None

# ── Core PDF extraction ────────────────────────────────────────────────────────

def extract_pdf(pdf_path: Path) -> list[dict]:
    """
    Extract all articles from a single Justel consolidated PDF.

    Algorithm (per spec §A.3):
      For each page → for each text block:
        1. Skip if structural heading → update hierarchy state, flush any open article.
        2. Skip if article marker (Art. X) → flush previous article, open new one.
        3. Otherwise → append block text to the current article accumulator.

    Returns a list of article dicts ready for JSONL serialisation.
    """
    pdf_filename = pdf_path.name
    numac = extract_numac(pdf_filename)

    doc = fitz.open(str(pdf_path))

    freq = analyse_font_sizes(doc)
    body_size = get_body_size(freq)
    skip_lines = detect_header_footer_lines(doc)

    state = _empty_state(pdf_filename, numac)
    articles: list[dict] = []

    for page_num, page in enumerate(doc):
        blocks = page.get_text("dict")["blocks"]

        for block in blocks:
            if block.get("type") != 0:  # only text blocks
                continue

            # Reconstruct block text and collect font sizes span by span.
            block_lines: list[str] = []
            span_sizes: list[float] = []

            for line in block.get("lines", []):
                line_text = ""
                for span in line.get("spans", []):
                    line_text += span["text"]
                    size = round(span["size"], 1)
                    if span["text"].strip():
                        span_sizes.append(size)
                block_lines.append(line_text)

            block_text = "\n".join(block_lines).strip()
            if not block_text:
                continue

            # Skip repeated header/footer lines.
            if normalise(block_text) in skip_lines:
                continue

            dominant_size = Counter(span_sizes).most_common(1)[0][0] if span_sizes else body_size

            # ── 1. Structural heading? ─────────────────────────────────────
            # Pattern match is primary; font size must be ≥ body (not a footnote).
            structural = _match_structural_marker(block_text)
            if structural and dominant_size >= body_size - 1.0:
                marker_type, heading_text = structural
                col = _COLUMN[marker_type]
                level = _LEVEL[marker_type]

                # Flush any open article before changing hierarchy.
                article = _flush_article(state)
                if article:
                    articles.append(article)
                state["article_number"] = None
                state["article_lines"] = []
                state["article_pages"] = []

                # Update this heading level; clear everything below it.
                state[col] = heading_text
                if level <= 0:
                    state["chapter_title"] = None
                    state["section_title"] = None
                    state["subsection_title"] = None
                elif level <= 1:
                    state["section_title"] = None
                    state["subsection_title"] = None
                elif level <= 2:
                    state["subsection_title"] = None
                continue

            # ── 2. Article marker? ─────────────────────────────────────────
            art_no = _is_article_marker(block_text)
            if art_no:
                # Flush previous article.
                article = _flush_article(state)
                if article:
                    articles.append(article)

                # Start new article accumulator.
                state["article_number"] = art_no
                state["article_lines"] = []
                state["article_pages"] = [page_num]

                # Collect any body text that follows "Art. X" on the same block.
                m = ART_RE.match(block_text.strip())
                if m:
                    remainder = block_text[m.end():].lstrip(" .\u2014\u2013-").strip()
                    if remainder:
                        state["article_lines"].append(remainder)
                continue

            # ── 3. Regular body text ───────────────────────────────────────
            if state["article_number"] is not None:
                state["article_lines"].append(block_text)
                state["article_pages"].append(page_num)

    # Flush the final article.
    article = _flush_article(state)
    if article:
        articles.append(article)

    doc.close()
    return articles

# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase A: extract articles from all Justel consolidated PDFs"
    )
    parser.add_argument(
        "--pdf-dir", type=Path, default=DEFAULT_PDF_DIR,
        help=f"Directory containing Justel PDFs (default: {DEFAULT_PDF_DIR})",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=DEFAULT_OUT_DIR,
        help=f"Output directory for per-PDF JSONL files (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-extract even if output JSONL already exists",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(args.pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {args.pdf_dir}")
        return

    print(f"Found {len(pdf_files)} PDF(s) in {args.pdf_dir}\n")

    total_articles = 0
    failed = []

    for i, pdf_path in enumerate(pdf_files, 1):
        out_path = args.out_dir / f"{pdf_path.stem}.jsonl"

        if out_path.exists() and not args.force:
            n = sum(1 for _ in out_path.open(encoding="utf-8"))
            print(f"[{i:2d}/{len(pdf_files)}] SKIP  {pdf_path.name}  ({n} articles already extracted)")
            total_articles += n
            continue

        print(f"[{i:2d}/{len(pdf_files)}] {pdf_path.name} ...", end=" ", flush=True)

        try:
            articles = extract_pdf(pdf_path)
        except Exception as exc:
            print(f"ERROR — {exc}")
            failed.append(pdf_path.name)
            continue

        # Write atomically: temp file → rename.
        tmp_path = out_path.with_suffix(".jsonl.tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            for art in articles:
                f.write(json.dumps(art, ensure_ascii=False) + "\n")
        tmp_path.replace(out_path)

        print(f"{len(articles)} articles -> {out_path.name}")
        total_articles += len(articles)

    print(f"\nTotal articles extracted: {total_articles}")
    if failed:
        print(f"Failed PDFs ({len(failed)}): {', '.join(failed)}")


if __name__ == "__main__":
    main()
