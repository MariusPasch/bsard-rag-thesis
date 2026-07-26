"""Compute per-PDF BSARD drift status.

For each BSARD article tied to a given PDF (in the BSARD corpus DB), check
whether the article's HuggingFace canonical ``article_text`` is actually present
in the canonical source PDF's extracted ``raw_text.txt``. Writes
``drift_status.json`` next to ``article_spans.json``.

Outputs one verdict per bsard_id:
  - IN_PDF     canonical text substring found in raw_text
  - ABROGATED  not in raw_text; the largest body span at an Art. marker for
               this bsard_id contains an ``<Abrogé par ...>`` stub
  - MISSING    not in raw_text; no body span resolves for this bsard_id
  - PARTIAL    not in raw_text; has a body span without an abrogation stub
               (residual / format drift / partial extraction)

Usage (from RQ2_T03_ARM1_NAIVE/, with the venv active):
    python scripts/compute_drift_status.py --doc-id 1804_03_21_1804032153
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent

DEFAULT_DB_PATH = Path(
    os.environ.get(
        "RQ2_BSARD_DB",
        str(_REPO / "data" / "bsard_corpus.db"),
    )
)
DEFAULT_CACHE_ROOT = _REPO / "RQ2_T03_ARM1_NAIVE" / "data"

NEEDLE_LEN = 80          # chars of normalised HF text used as substring needle
MIN_NEEDLE_LEN = 30      # below this, skip substring check (unreliable)
NEEDLE_STRIDE = 60       # step between needle starts when sweeping HF text
ABROG_RE = re.compile(r"<\s*Abrog[^>]{0,120}>", re.IGNORECASE)


def _normalise(s: str) -> str:
    """NFKD-strip-diacritics, lowercase, collapse whitespace."""
    if not s:
        return ""
    nfkd = unicodedata.normalize("NFKD", s)
    no_diac = "".join(c for c in nfkd if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", no_diac.lower()).strip()


def _hf_text_in_raw(hf_norm: str, raw_norm: str) -> bool:
    """True if any NEEDLE_LEN-char window of hf_norm appears in raw_norm.

    Sweeps the HF text at NEEDLE_STRIDE intervals so legal-annotation insertions
    in the PDF body (e.g. ``<L 1998-11-23/35 …>`` mid-article) don't cause a
    false negative just because the first 80 chars of HF text differ.
    """
    if len(hf_norm) < MIN_NEEDLE_LEN:
        return False
    if len(hf_norm) <= NEEDLE_LEN:
        return hf_norm in raw_norm
    last_start = len(hf_norm) - NEEDLE_LEN
    for off in range(0, last_start + 1, NEEDLE_STRIDE):
        if hf_norm[off : off + NEEDLE_LEN] in raw_norm:
            return True
    # Always test the tail window so we don't miss it when the stride doesn't
    # land on it.
    return hf_norm[last_start : last_start + NEEDLE_LEN] in raw_norm


def _classify(
    hf_text: str,
    raw_norm: str,
    spans_for_bid: list[tuple[int, int]],
    raw_text: str,
) -> tuple[str, int, str]:
    """Return (verdict, body_span_chars, evidence_snippet)."""
    in_pdf = _hf_text_in_raw(_normalise(hf_text), raw_norm)

    # Inspect every span this bsard_id resolves to, not just the widest.
    abrog_span_idx = -1
    widest_idx = -1
    widest_chars = -1
    for i, (s, e) in enumerate(spans_for_bid):
        chunk = raw_text[s:e]
        if abrog_span_idx < 0 and ABROG_RE.search(chunk):
            abrog_span_idx = i
        if (e - s) > widest_chars:
            widest_chars = e - s
            widest_idx = i

    if spans_for_bid:
        evidence_idx = abrog_span_idx if abrog_span_idx >= 0 else widest_idx
        s, e = spans_for_bid[evidence_idx]
        evidence = raw_text[s:e].replace("\n", " ").replace("  ", " ").strip()[:120]
        body_chars = widest_chars
    else:
        evidence = ""
        body_chars = 0

    if in_pdf:
        verdict = "IN_PDF"
    elif abrog_span_idx >= 0:
        verdict = "ABROGATED"
    elif body_chars == 0:
        verdict = "MISSING"
    else:
        verdict = "PARTIAL"
    return verdict, body_chars, evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--doc-id", required=True,
                        help="Doc id (e.g. 1804_03_21_1804032153)")
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--bsard-pdf-filename", default=None,
                        help="Override the BSARD-side pdf_filename "
                             "(default: img_l_pdf_<doc_id>_F.pdf)")
    args = parser.parse_args()

    cache_dir = args.cache_root / args.doc_id
    raw_text_path = cache_dir / "raw_text.txt"
    spans_path = cache_dir / "article_spans.json"
    pdf_meta_path = cache_dir / "pdf_meta.json"
    out_path = cache_dir / "drift_status.json"

    for p in (raw_text_path, spans_path, pdf_meta_path):
        if not p.exists():
            print(f"missing input: {p}", file=sys.stderr)
            return 1
    if not args.db_path.exists():
        print(f"DB not found: {args.db_path}", file=sys.stderr)
        return 1

    raw_text = raw_text_path.read_text(encoding="utf-8")
    raw_norm = _normalise(raw_text)
    spans_doc = json.loads(spans_path.read_text(encoding="utf-8"))
    pdf_meta = json.loads(pdf_meta_path.read_text(encoding="utf-8"))

    by_bid: dict[int, list[tuple[int, int]]] = {}
    for sp in spans_doc["spans"]:
        by_bid.setdefault(sp["bsard_id"], []).append(
            (sp["start_char"], sp["end_char"])
        )
    for lst in by_bid.values():
        lst.sort(key=lambda t: t[1] - t[0], reverse=True)

    bsard_pdf = args.bsard_pdf_filename or f"img_l_pdf_{args.doc_id}_F.pdf"

    conn = sqlite3.connect(str(args.db_path))
    rows = conn.execute(
        """
        SELECT bsard_id, article_number, article_text, char_count,
               article_status, verification_status, pdf_match_category
        FROM articles
        WHERE pdf_filename = ?
          AND bsard_id IS NOT NULL
          AND article_text IS NOT NULL
        """,
        (bsard_pdf,),
    ).fetchall()
    conn.close()

    if not rows:
        print(
            f"no BSARD articles found for pdf_filename={bsard_pdf!r}\n"
            "Hint: pass --bsard-pdf-filename if your DB uses a different naming.",
            file=sys.stderr,
        )
        return 1

    # Multiple DB rows can share a bsard_id (different article_number spellings).
    # Keep the row with the longest text per bsard_id; capture all article_numbers.
    by_bid_row: dict[int, dict] = {}
    for bid, art_no, txt, cc, art_st, ver_st, mc in rows:
        slot = by_bid_row.get(bid)
        text_len = cc if cc is not None else len(txt or "")
        if slot is None or text_len > slot["_text_len"]:
            by_bid_row[bid] = {
                "bsard_id": bid,
                "article_number": art_no,
                "_text": txt,
                "_text_len": text_len,
                "article_status": art_st,
                "verification_status": ver_st,
                "pdf_match_category": mc,
                "all_article_numbers": set(),
            }
        by_bid_row[bid]["all_article_numbers"].add(art_no)

    counts = {"IN_PDF": 0, "ABROGATED": 0, "MISSING": 0, "PARTIAL": 0}
    verdicts = []
    for bid in sorted(by_bid_row):
        rec = by_bid_row[bid]
        verdict, body_chars, evidence = _classify(
            rec["_text"], raw_norm, by_bid.get(bid, []), raw_text
        )
        counts[verdict] += 1
        verdicts.append({
            "bsard_id": bid,
            "article_number": rec["article_number"],
            "all_article_numbers": sorted(rec["all_article_numbers"]),
            "verdict": verdict,
            "hf_text_len": rec["_text_len"],
            "n_spans": len(by_bid.get(bid, [])),
            "body_span_chars": body_chars,
            "evidence": evidence,
            "bsard_article_status": rec["article_status"],
            "bsard_verification_status": rec["verification_status"],
            "bsard_pdf_match_category": rec["pdf_match_category"],
        })

    out = {
        "schema_version": 1,
        "doc_id": args.doc_id,
        "pdf_filename": pdf_meta.get("doc_id", args.doc_id) + ".pdf",
        "bsard_pdf_filename": bsard_pdf,
        "pdf_sha256": pdf_meta.get("pdf_sha256"),
        "bsard_db_fingerprint": spans_doc.get("bsard_db_fingerprint"),
        "computed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "method": {
            "needle_len": NEEDLE_LEN,
            "needle_stride": NEEDLE_STRIDE,
            "min_needle_len": MIN_NEEDLE_LEN,
            "normalisation": "NFKD-strip-diacritics, lowercase, collapse whitespace",
            "abrogation_regex": ABROG_RE.pattern,
        },
        "summary": {
            "n_bsard_articles": len(by_bid_row),
            "n_in_pdf": counts["IN_PDF"],
            "n_abrogated": counts["ABROGATED"],
            "n_missing": counts["MISSING"],
            "n_partial": counts["PARTIAL"],
        },
        "verdicts": verdicts,
    }

    out_path.write_text(
        json.dumps(out, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {out_path}")
    print(f"summary: {out['summary']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
