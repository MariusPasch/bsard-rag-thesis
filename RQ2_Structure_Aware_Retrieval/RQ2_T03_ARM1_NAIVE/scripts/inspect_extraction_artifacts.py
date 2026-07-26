"""Diagnose U+FFFD replacement-character contamination in cached PDF extractions.

PyMuPDF emits U+FFFD ("\\uFFFD") whenever a glyph cannot be mapped to Unicode
(missing toUnicode CMap, embedded subset without a glyph-name table, etc.).
Once these characters land in ``data/<doc_id>/raw_text.txt`` they propagate
into ``chunks.json``, the FAISS index, and any baked T08 bundle. This script
quantifies the contamination per cached PDF and (optionally) compares against
a fresh PyMuPDF extraction with alternative flag combinations, so you can
decide which docs warrant cache invalidation + re-extraction.

Read-only with respect to the T03 cache: never writes, never deletes.

Usage (from RQ2_T03_ARM1_NAIVE/, with the venv active):
    # Survey every cached doc:
    python scripts/inspect_extraction_artifacts.py

    # Drill into one doc, show samples + flag comparison:
    python scripts/inspect_extraction_artifacts.py --pdf 1804_03_21_1804032150 \\
        --samples 5 --compare-flags
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent

logger = logging.getLogger(__name__)

DEFAULT_PDF_DIR = _REPO / "RQ2_T02_DATA_LOADER/data/pdfs"
DEFAULT_CACHE_ROOT = _REPO / "RQ2_T03_ARM1_NAIVE" / "data"

REPLACEMENT_CHAR = "�"


@dataclass
class DocReport:
    doc_id: str
    n_chars: int
    n_replacements: int
    pct: float
    chunks_n_replacements: int | None
    cache_path: Path

    @property
    def is_clean(self) -> bool:
        return self.n_replacements == 0


def _scan_doc(doc_root: Path) -> DocReport | None:
    raw_path = doc_root / "raw_text.txt"
    if not raw_path.exists():
        return None

    text = raw_path.read_text(encoding="utf-8")
    n_replacements = text.count(REPLACEMENT_CHAR)
    pct = (n_replacements / len(text) * 100.0) if text else 0.0

    chunks_n_replacements: int | None = None
    configs_dir = doc_root / "configs"
    if configs_dir.exists():
        chunks_n_replacements = 0
        for chunks_path in configs_dir.glob("*/chunks.json"):
            try:
                payload = json.loads(chunks_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            for chunk in payload.get("chunks", []):
                chunks_n_replacements += chunk.get("text", "").count(REPLACEMENT_CHAR)

    return DocReport(
        doc_id=doc_root.name,
        n_chars=len(text),
        n_replacements=n_replacements,
        pct=pct,
        chunks_n_replacements=chunks_n_replacements,
        cache_path=doc_root,
    )


def _print_samples(text: str, doc_id: str, n_samples: int, window: int = 60) -> None:
    indices: list[int] = []
    start = 0
    while True:
        idx = text.find(REPLACEMENT_CHAR, start)
        if idx == -1:
            break
        indices.append(idx)
        start = idx + 1
        if len(indices) >= n_samples * 4:
            break

    if not indices:
        print(f"    (no replacement chars in {doc_id})")
        return

    print(f"    showing {min(n_samples, len(indices))} of {len(indices)} "
          f"earliest occurrences in {doc_id}:")
    for idx in indices[:n_samples]:
        lo = max(0, idx - window)
        hi = min(len(text), idx + window + 1)
        snippet = text[lo:hi].replace("\n", "\\n")
        marker = " " * (idx - lo) + "^"
        print(f"      @char {idx:>8}: …{snippet}…")
        print(f"                     {marker}")


def _compare_flag_variants(pdf_path: Path) -> dict[str, int]:
    """Re-extract the PDF under a few flag combinations and report U+FFFD counts."""
    try:
        import fitz
    except ImportError:
        return {"_error": "PyMuPDF (fitz) not installed in this venv"}

    variants: dict[str, int] = {}

    def _extract(option: str, flags: int | None = None) -> str:
        with fitz.open(str(pdf_path)) as doc:
            if flags is None:
                pages = [page.get_text(option) for page in doc]
            else:
                pages = [page.get_text(option, flags=flags) for page in doc]
        return "\n\n".join(pages)

    variants["default (page.get_text())"] = _extract("text").count(REPLACEMENT_CHAR)
    try:
        variants["TEXT_PRESERVE_LIGATURES"] = _extract(
            "text", flags=fitz.TEXT_PRESERVE_LIGATURES
        ).count(REPLACEMENT_CHAR)
    except AttributeError:
        pass
    try:
        variants["TEXT_PRESERVE_WHITESPACE"] = _extract(
            "text", flags=fitz.TEXT_PRESERVE_WHITESPACE
        ).count(REPLACEMENT_CHAR)
    except AttributeError:
        pass
    try:
        variants["TEXT_DEHYPHENATE"] = _extract(
            "text", flags=fitz.TEXT_DEHYPHENATE
        ).count(REPLACEMENT_CHAR)
    except AttributeError:
        pass
    variants["blocks (page.get_text('blocks'))"] = sum(
        b[4].count(REPLACEMENT_CHAR)
        for page in fitz.open(str(pdf_path))
        for b in page.get_text("blocks")
    )

    return variants


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--pdf", default=None,
                        help="doc_id (PDF stem) to drill into; survey-mode if omitted")
    parser.add_argument("--samples", type=int, default=3,
                        help="When --pdf is set, number of replacement-context snippets to print")
    parser.add_argument("--compare-flags", action="store_true",
                        help="When --pdf is set, re-extract with alternative PyMuPDF flags and compare counts")
    parser.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIR)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if not args.cache_root.exists():
        print(f"Cache root not found: {args.cache_root}", file=sys.stderr)
        return 1

    if args.pdf:
        doc_root = args.cache_root / args.pdf
        if not doc_root.exists():
            print(f"Doc cache not found: {doc_root}", file=sys.stderr)
            return 1
        report = _scan_doc(doc_root)
        if report is None:
            print(f"No raw_text.txt at {doc_root}", file=sys.stderr)
            return 1

        print(f"doc_id      = {report.doc_id}")
        print(f"raw_text    = {report.n_chars:,} chars, "
              f"{report.n_replacements} U+FFFD ({report.pct:.4f}%)")
        if report.chunks_n_replacements is not None:
            print(f"chunks.json = {report.chunks_n_replacements} U+FFFD across all configs")

        if report.n_replacements > 0:
            print("\n[samples]")
            _print_samples(
                doc_root.joinpath("raw_text.txt").read_text(encoding="utf-8"),
                report.doc_id, args.samples,
            )

        if args.compare_flags:
            pdf_path = args.pdf_dir / f"{args.pdf}.pdf"
            if not pdf_path.exists():
                print(f"\n[compare-flags] PDF not found: {pdf_path} — skipping",
                      file=sys.stderr)
            else:
                print(f"\n[compare-flags] re-extracting {pdf_path.name} …")
                variants = _compare_flag_variants(pdf_path)
                if "_error" in variants:
                    print(f"  {variants['_error']}")
                else:
                    cached = report.n_replacements
                    width = max(len(k) for k in variants)
                    print(f"  {'cached raw_text.txt':<{width}}  {cached:>6} U+FFFD")
                    for label, count in variants.items():
                        delta = count - cached
                        flag = "" if delta == 0 else (f"  ({delta:+d} vs cache)")
                        print(f"  {label:<{width}}  {count:>6} U+FFFD{flag}")
        return 0

    # Survey mode: walk every doc folder.
    rows: list[DocReport] = []
    for doc_root in sorted(p for p in args.cache_root.iterdir() if p.is_dir()):
        report = _scan_doc(doc_root)
        if report is not None:
            rows.append(report)

    if not rows:
        print(f"No raw_text.txt files found under {args.cache_root}")
        return 0

    affected = [r for r in rows if r.n_replacements > 0]
    print(f"scanned     = {len(rows)} cached docs ({len(affected)} affected)")
    print(f"total chars = {sum(r.n_chars for r in rows):,}")
    print(f"total U+FFFD= {sum(r.n_replacements for r in rows)}")

    if not affected:
        print("\nAll cached extractions are clean.")
        return 0

    affected.sort(key=lambda r: r.n_replacements, reverse=True)
    name_w = max(len(r.doc_id) for r in affected)
    print(f"\n{'doc_id':<{name_w}}  {'raw_text U+FFFD':>16}  {'chars':>10}  "
          f"{'pct':>7}  {'chunks U+FFFD':>14}")
    for r in affected:
        chunks_str = "—" if r.chunks_n_replacements is None else f"{r.chunks_n_replacements}"
        print(f"{r.doc_id:<{name_w}}  {r.n_replacements:>16}  {r.n_chars:>10,}  "
              f"{r.pct:>6.3f}%  {chunks_str:>14}")

    print(
        "\nNext step: pick the worst offender and rerun with --pdf <doc_id> "
        "--compare-flags to see if alternative PyMuPDF flags reduce the count. "
        "If yes, you can invalidate that doc's cache (delete pdf_meta.json + raw_text.txt + "
        "configs/) and rerun precompute_pdf.py to repopulate."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
