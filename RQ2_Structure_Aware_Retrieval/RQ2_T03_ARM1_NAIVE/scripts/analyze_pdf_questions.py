"""Find BSARD questions relevant to a single PDF and emit a per-PDF analysis file.

A question is "relevant" to a target PDF when its ground-truth bsard articles
satisfy the chosen ``--relevance-criterion``:
  * ``strict``   — at least one GT article has verification_status == 'FOUND' in this PDF
  * ``lenient``  — at least one GT article has FOUND or PARTIAL in this PDF (default,
                   backwards-compatible with schema_version=1)
  * ``linked``   — at least one GT article's ``pdf_filenames`` includes this PDF, regardless
                   of verification_status. Robust to upstream article-number-vs-canonical-text
                   drift, where the linkage step assigned a PDF by article-number coincidence
                   but the canonical BSARD text disagrees with the PDF text.

Per-article rows (``bsard_id_matching``) are augmented from the corpus DB and the
per-PDF extraction JSONL when available, so the consumer can always tell *why* a
linkage may have failed verification:

  - article_number_in_pdf       # article_number recorded for this bsard_id in this PDF
  - pdf_extraction_status       # 'pdf_extracted' / 'pdf_extraction_failed' / 'not_in_extraction'
  - extraction_cosine           # float | None — char-4-gram cosine between canonical
                                # BSARD article_text and the extracted span from this PDF

When ``pdf_extraction_status == 'pdf_extracted'`` but ``verification_status == 'NOT FOUND'``,
that's the numbering-drift signal: the article number's text *was* successfully extracted
from this PDF, but it doesn't match the canonical BSARD text for this bsard_id.
``extraction_cosine`` makes that signal continuous: 0.12–0.30 means the article number
matches but the canonical text disagrees materially; ~1.0 means they line up.

Output schema (per row):
  - question_id, split
  - extraction_status                # global bucket from the JSONL
  - per_pdf_extraction_status        # exact / partial / mixed for *this* PDF only
  - n_relevant_bsard_articles        # total GT articles for the question
  - n_in_target_pdf                  # GT articles linked to the target PDF (any status)
  - bsard_ids_context                # full list of GT bsard_ids
  - bsard_id_matching                # per-bsard_id breakdown (see above)
  - strict_gt_in_pdf                 # bsard_ids with FOUND in target PDF
  - lenient_gt_in_pdf                # bsard_ids with FOUND or PARTIAL in target PDF
  - linked_gt_in_pdf                 # bsard_ids whose pdf_filenames includes target PDF
  - strict_recall_ceiling            # |strict_gt_in_pdf| / |bsard_ids_context|
  - lenient_recall_ceiling           # |lenient_gt_in_pdf| / |bsard_ids_context|
  - linked_recall_ceiling            # |linked_gt_in_pdf|  / |bsard_ids_context|

Usage (from RQ2_T03_ARM1_NAIVE/, with the venv active):
    python scripts/analyze_pdf_questions.py --doc-id 1804_03_21_1804032150
    python scripts/analyze_pdf_questions.py --doc-id 2004_05_27_2004A27101 --relevance-criterion linked
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
import sys
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_T03_REPO = _HERE.parent
_REPO = _T03_REPO.parent

DEFAULT_STATUS_JSONL = (
    _REPO
    / "RQ2_T07_EVALUATION"
    / "ground_truth"
    / "question_extraction_status"
    / "questions_by_extraction_status.jsonl"
)
DEFAULT_DATA_ROOT = _T03_REPO / "data"
DEFAULT_CORPUS_DB = _REPO / "dataset_creation_output" / "bsard_corpus.db"
DEFAULT_EXTRACTED_DIR = _REPO / "dataset_creation_output" / "extracted"


def _doc_id_to_db_filename(doc_id: str) -> str:
    """The corpus DB stores PDFs as ``img_l_pdf_<doc_id>_F.pdf``."""
    return f"img_l_pdf_{doc_id}_F.pdf"


def _per_pdf_bucket(verifications_in_pdf: list[str]) -> str:
    """Bucket the question for *this* PDF only.

    Mirrors the global JSONL semantics ('exact' / 'partial' / 'not_present' /
    'mixed') but restricted to the GT articles whose pdf_filenames include the
    target PDF.
    """
    if not verifications_in_pdf:
        return "no_articles_in_pdf"
    distinct = set(verifications_in_pdf)
    if distinct == {"FOUND"}:
        return "exact"
    if distinct == {"PARTIAL"}:
        return "partial"
    if distinct == {"NOT FOUND"}:
        return "not_present"
    return "mixed"


def _load_bsard_pdf_link(
    db_path: Path, target_pdf: str
) -> dict[int, dict[str, str | None]]:
    """Map ``bsard_id -> {article_number, article_text}`` for rows linked to ``target_pdf``.

    A bsard_id can in principle have multiple rows for the same PDF; we keep the
    first one. ``article_text`` is the canonical BSARD article body — used as the
    reference side of the per-row ``extraction_cosine``.
    """
    if not db_path.exists():
        return {}
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.execute(
            "SELECT bsard_id, article_number, article_text FROM articles "
            "WHERE pdf_filename = ? AND bsard_id IS NOT NULL",
            (target_pdf,),
        )
        out: dict[int, dict[str, str | None]] = {}
        for bsard_id, article_number, article_text in cur.fetchall():
            out.setdefault(
                int(bsard_id),
                {"article_number": article_number, "article_text": article_text},
            )
        return out
    finally:
        conn.close()


def _load_extraction_index(
    extracted_jsonl: Path,
) -> dict[str, dict[str, str | None]]:
    """Map ``article_number -> {source, text}`` for one PDF's extraction JSONL.

    ``source`` is ``article_text_source`` (``pdf_extracted`` / ``pdf_extraction_failed``);
    ``text`` is the extracted article body when extraction succeeded, else None —
    used as the candidate side of the per-row ``extraction_cosine``.
    """
    if not extracted_jsonl.exists():
        return {}
    out: dict[str, dict[str, str | None]] = {}
    with open(extracted_jsonl, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            num = rec.get("article_number")
            src = rec.get("article_text_source")
            if num and src:
                out.setdefault(num, {"source": src, "text": rec.get("article_text")})
    return out


# ── Char-4-gram cosine (informational signal for downstream consumers) ────────


def _normalise_for_cosine(text: str) -> str:
    """NFKD strip-accents, lowercase, collapse whitespace."""
    if not text:
        return ""
    nfkd = unicodedata.normalize("NFKD", str(text))
    no_acc = "".join(c for c in nfkd if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", no_acc).lower().strip()


def _char_4grams(text: str) -> Counter[str]:
    if len(text) < 4:
        return Counter()
    return Counter(text[i : i + 4] for i in range(len(text) - 3))


def _cosine(a: Counter[str], b: Counter[str]) -> float:
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    dot = sum(v * b.get(k, 0) for k, v in a.items())
    if dot == 0:
        return 0.0
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    return dot / (norm_a * norm_b)


def _extraction_cosine(canonical: str | None, extracted: str | None) -> float | None:
    """Char-4-gram cosine between canonical BSARD text and an extracted PDF span.

    Returns None when either side is missing, empty after normalisation, or
    shorter than the n-gram window (4 chars). The downstream consumer treats
    this as 'not computable' — distinct from a low-but-real similarity.
    """
    if not canonical or not extracted:
        return None
    a = _char_4grams(_normalise_for_cosine(canonical))
    b = _char_4grams(_normalise_for_cosine(extracted))
    if not a or not b:
        return None
    return _cosine(a, b)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--doc-id",
        required=True,
        help="PDF doc_id without extension or 'img_l_pdf_' prefix, e.g. 1804_03_21_1804032150",
    )
    parser.add_argument("--status-jsonl", type=Path, default=DEFAULT_STATUS_JSONL)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--corpus-db", type=Path, default=DEFAULT_CORPUS_DB)
    parser.add_argument("--extracted-dir", type=Path, default=DEFAULT_EXTRACTED_DIR)
    parser.add_argument(
        "--output-name",
        default="question_relevance.json",
        help="Filename inside data/<doc_id>/ (default: %(default)s)",
    )
    parser.add_argument(
        "--include-irrelevant",
        action="store_true",
        help="Also write rows for questions with zero matched GT in this PDF",
    )
    parser.add_argument(
        "--relevance-criterion",
        choices=["strict", "lenient", "linked"],
        default="lenient",
        help="What counts as 'relevant'. strict=FOUND only; lenient=FOUND or PARTIAL "
             "(default, backwards-compatible); linked=any GT article whose pdf_filenames "
             "includes this PDF, regardless of verification (robust to upstream "
             "article-number-vs-canonical-text drift).",
    )
    args = parser.parse_args()

    if not args.status_jsonl.exists():
        print(f"status JSONL not found: {args.status_jsonl}", file=sys.stderr)
        return 1

    target_db_filename = _doc_id_to_db_filename(args.doc_id)
    out_dir = args.data_root / args.doc_id
    if not out_dir.exists():
        print(f"target data folder does not exist: {out_dir}", file=sys.stderr)
        return 1
    out_path = out_dir / args.output_name

    bsard_pdf_link = _load_bsard_pdf_link(args.corpus_db, target_db_filename)
    extracted_jsonl = args.extracted_dir / f"img_l_pdf_{args.doc_id}_F.jsonl"
    extraction_index = _load_extraction_index(extracted_jsonl)
    db_available = bool(bsard_pdf_link)
    jsonl_available = bool(extraction_index)

    relevant_rows: list[dict] = []
    irrelevant_rows: list[dict] = []
    split_counts: Counter[str] = Counter()
    per_pdf_bucket_counts: Counter[str] = Counter()
    criterion_counts = {"strict": 0, "lenient": 0, "linked": 0}

    with open(args.status_jsonl, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            q = json.loads(line)

            qid = int(q["question_id"])
            split = q["split"]
            articles = q.get("bsard_articles", [])

            bsard_ids_context = [int(a["bsard_id"]) for a in articles]

            matching = []
            verifications_in_pdf: list[str] = []
            strict: list[int] = []
            lenient: list[int] = []
            linked: list[int] = []
            n_in_target = 0
            for a in articles:
                bsard_id = int(a["bsard_id"])
                pdf_filenames = list(a.get("pdf_filenames") or [])
                in_target = target_db_filename in pdf_filenames
                vstatus = a["verification_status"]

                row = {
                    "bsard_id": bsard_id,
                    "verification_status": vstatus,
                    "pdf_filenames": pdf_filenames,
                    "in_target_pdf": in_target,
                }
                canonical_text: str | None = None
                extracted_text: str | None = None
                if in_target and db_available:
                    link = bsard_pdf_link.get(bsard_id)
                    article_number = link["article_number"] if link else None
                    canonical_text = link["article_text"] if link else None
                    row["article_number_in_pdf"] = article_number
                    if jsonl_available:
                        if article_number is None:
                            row["pdf_extraction_status"] = "not_in_extraction"
                        else:
                            entry = extraction_index.get(article_number)
                            row["pdf_extraction_status"] = (
                                entry["source"] if entry else "not_in_extraction"
                            )
                            if entry:
                                extracted_text = entry["text"]
                row["extraction_cosine"] = _extraction_cosine(canonical_text, extracted_text)
                matching.append(row)

                if in_target:
                    n_in_target += 1
                    verifications_in_pdf.append(vstatus)
                    linked.append(bsard_id)
                    if vstatus == "FOUND":
                        strict.append(bsard_id)
                        lenient.append(bsard_id)
                    elif vstatus == "PARTIAL":
                        lenient.append(bsard_id)

            n_total = len(bsard_ids_context)
            row = {
                "question_id": qid,
                "split": split,
                "extraction_status": q.get("extraction_status"),
                "per_pdf_extraction_status": _per_pdf_bucket(verifications_in_pdf),
                "n_relevant_bsard_articles": n_total,
                "n_in_target_pdf": n_in_target,
                "bsard_ids_context": bsard_ids_context,
                "bsard_id_matching": matching,
                "strict_gt_in_pdf": sorted(strict),
                "lenient_gt_in_pdf": sorted(lenient),
                "linked_gt_in_pdf": sorted(linked),
                "strict_recall_ceiling": (len(strict) / n_total) if n_total else 0.0,
                "lenient_recall_ceiling": (len(lenient) / n_total) if n_total else 0.0,
                "linked_recall_ceiling": (len(linked) / n_total) if n_total else 0.0,
            }

            if strict:
                criterion_counts["strict"] += 1
            if lenient:
                criterion_counts["lenient"] += 1
            if linked:
                criterion_counts["linked"] += 1

            criterion_set = {"strict": strict, "lenient": lenient, "linked": linked}[
                args.relevance_criterion
            ]
            is_relevant = len(criterion_set) > 0
            if is_relevant:
                relevant_rows.append(row)
                split_counts[split] += 1
                per_pdf_bucket_counts[row["per_pdf_extraction_status"]] += 1
            else:
                irrelevant_rows.append(row)

    relevant_rows.sort(key=lambda r: (r["split"], r["question_id"]))
    irrelevant_rows.sort(key=lambda r: (r["split"], r["question_id"]))

    payload: dict = {
        "schema_version": 3,
        "doc_id": args.doc_id,
        "target_pdf_filename_db": target_db_filename,
        "source_jsonl": str(args.status_jsonl).replace("\\", "/"),
        "corpus_db": str(args.corpus_db).replace("\\", "/") if db_available else None,
        "extracted_jsonl": str(extracted_jsonl).replace("\\", "/") if jsonl_available else None,
        "computed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "relevance_criterion": args.relevance_criterion,
        "definitions": {
            "relevant": (
                "strict: ≥1 GT article FOUND in this PDF; "
                "lenient: ≥1 GT article FOUND or PARTIAL in this PDF (default, backwards-compatible); "
                "linked: ≥1 GT article whose pdf_filenames includes this PDF, regardless of verification "
                "(robust to upstream article-number-vs-canonical-text drift)."
            ),
            "strict_gt_in_pdf": "GT bsard_ids with verification_status == 'FOUND' inside this PDF",
            "lenient_gt_in_pdf": "GT bsard_ids with verification_status in {'FOUND','PARTIAL'} inside this PDF",
            "linked_gt_in_pdf": "GT bsard_ids whose pdf_filenames includes this PDF, regardless of verification",
            "per_pdf_extraction_status": "exact / partial / mixed / not_present / no_articles_in_pdf — bucket restricted to this PDF only",
            "pdf_extraction_status": (
                "Per-article evidence from the per-PDF extraction JSONL: "
                "'pdf_extracted' (article_number's text was extracted from this PDF), "
                "'pdf_extraction_failed' (article_number is in the JSONL but extraction failed), "
                "'not_in_extraction' (article_number not present in this PDF's extraction). "
                "When this disagrees with verification_status (e.g. 'pdf_extracted' but 'NOT FOUND'), "
                "upstream linkage matched by article_number alone while the canonical BSARD text differs "
                "from this PDF's article text — i.e., numbering drift across legal-code revisions."
            ),
            "extraction_cosine": (
                "Char-4-gram cosine similarity between the canonical BSARD article_text "
                "(corpus DB) and the extracted span from this PDF (extraction JSONL), with "
                "NFKD-strip-accents + lowercase + whitespace-collapse normalisation. "
                "Continuous companion to verification_status: low values (~0.1–0.3) on a "
                "verification_status='FOUND' row mean the bsard_id is present-by-number "
                "but substantively different in content; ~1.0 means they agree. "
                "null when canonical or extracted text is missing/empty (e.g. "
                "pdf_extraction_status in {'pdf_extraction_failed','not_in_extraction'}, "
                "or in_target_pdf=false). Informational only — verification_status semantics "
                "are unchanged."
            ),
        },
        "summary": {
            "n_relevant": len(relevant_rows),
            "n_relevant_train": split_counts.get("train", 0),
            "n_relevant_test": split_counts.get("test", 0),
            "per_pdf_bucket_counts": dict(per_pdf_bucket_counts),
            "relevance_by_criterion": criterion_counts,
        },
        "relevant_questions": relevant_rows,
    }
    if args.include_irrelevant:
        payload["irrelevant_questions"] = irrelevant_rows

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.write("\n")

    print(f"wrote {out_path}")
    print(f"  criterion={args.relevance_criterion}: {len(relevant_rows)} relevant "
          f"(train={split_counts.get('train', 0)}, test={split_counts.get('test', 0)})")
    print(f"  counts by criterion: strict={criterion_counts['strict']}, "
          f"lenient={criterion_counts['lenient']}, linked={criterion_counts['linked']}")
    print(f"  per-PDF buckets:   {dict(per_pdf_bucket_counts)}")
    if db_available and jsonl_available:
        print("  augmented with corpus DB + extraction JSONL")
    else:
        print(f"  WARNING: augmentation partial (db={db_available}, jsonl={jsonl_available}) "
              "— pdf_extraction_status may be omitted for some/all rows")
    if args.include_irrelevant:
        print(f"  irrelevant rows also included: {len(irrelevant_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
