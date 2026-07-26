"""Phase B — BSARD linkage and metadata merge.

For each article extracted in Phase A:
  1. Tries to link it to a BSARD benchmark article by (pdf_filename, article_no).
  2. Overrides article_text with canonical HuggingFace text for matched articles.
  3. Merges pdf_url, law_type, temporal metadata, and verification status from
     bsard_full_verify.csv.
  4. Infers law_code for all articles (HF code for BSARD; majority code per PDF
     for non-BSARD).

Also loads and resolves the 1,100 BSARD benchmark questions.

Outputs:
  output/linked/articles_linked.jsonl  — all articles with full metadata
  output/linked/questions.jsonl        — questions with relevant_bsard_ids

Usage:
    python retrieval/link_bsard.py
    python retrieval/link_bsard.py --csv PATH --extracted-dir PATH
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from datasets import load_dataset

# ── Paths ──────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# bsard_full_verify.csv is published in the Hugging Face dataset
# (mpaschalidis/bsard2currentlawmatching) and lands at this path when pulled
# via `python scripts/download_from_hf.py`. Override with --csv if needed.
DEFAULT_CSV = PROJECT_ROOT / "output" / "bsard_full_verify.csv"
DEFAULT_EXTRACTED_DIR = PROJECT_ROOT / "output" / "extracted"
DEFAULT_OUT_DIR = PROJECT_ROOT / "output" / "linked"

# ── Normalisation helpers ──────────────────────────────────────────────────────

def normalise_key(text: str) -> str:
    """Accent-stripped, lowercase key for fuzzy matching."""
    nfkd = unicodedata.normalize("NFKD", str(text))
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower().strip()


_LEADING_INT_RE = re.compile(r"^(\d+)")
# Matches sub-paragraph suffix: "1.1.1-1", "I.1-2", "L1111-1_COMMUNAUTE_GERMANOPHONE"
# Group 1 = the base article number before the first "-digit" separator.
_BASE_ART_RE = re.compile(r"^([A-Z0-9][A-Z0-9_.]*)-\d", re.IGNORECASE)


def leading_int(art_no: str) -> str | None:
    """Return the leading integer of an article number, e.g. '42bis' → '42'."""
    m = _LEADING_INT_RE.match(str(art_no).strip())
    return m.group(1) if m else None


def base_article_no(art_no: str) -> str | None:
    """Strip sub-paragraph suffix from Belgian law paragraph IDs.

    Examples:
      '1.1.1-1'                       → '1.1.1'
      'I.1-2'                          → 'I.1'
      'L1111-1'                        → 'L1111'
      'L1111-1_COMMUNAUTE_GERMANOPHONE' → 'L1111'  (underscore variant stripped first)
      '42bis'                          → None       (no suffix to strip)
    """
    # Strip trailing _UPPERCASE_WORD variants (regional qualifiers).
    cleaned = re.sub(r"_[A-Z][A-Z_]+$", "", str(art_no).strip())
    m = _BASE_ART_RE.match(cleaned)
    return m.group(1) if m else None

# ── URL / filename helpers ─────────────────────────────────────────────────────

def url_to_filename(url: str) -> str:
    """Convert a Justel PDF URL to its local filename (same as download_pdfs.py)."""
    path = url.split("ejustice.just.fgov.be/")[-1]
    return path.replace("/", "_")


_NUMAC_RE = re.compile(r"_(\d{8,12})_[FfNn](?:_\w+)?\.pdf$", re.IGNORECASE)


def extract_numac(pdf_filename: str) -> str | None:
    """Extract the Moniteur belge publication ID from a Justel PDF filename."""
    m = _NUMAC_RE.search(pdf_filename)
    return m.group(1) if m else None

# ── Article number extraction from HF reference field ─────────────────────────
# reference format: "Art. 1.1.1, Code Name (Book, Title)"

_HF_ART_RE = re.compile(r"^Art\.\s+(.+?)\s*,", re.IGNORECASE)


def extract_article_no_from_reference(reference: str) -> str | None:
    """Extract the article number from an HF BSARD corpus reference string."""
    m = _HF_ART_RE.match(reference.strip())
    return m.group(1).strip() if m else None

# ── Load bsard_full_verify.csv ─────────────────────────────────────────────────

def load_verify_csv(csv_path: Path) -> pd.DataFrame:
    """Read the CSV (UTF-8 encoded); standardise column names."""
    df = pd.read_csv(csv_path, encoding="utf-8")
    # Rename columns to match DB schema
    df = df.rename(columns={
        "id":                "bsard_id",
        "url":               "justel_html_url",
        "text_found":        "html_text_found",
        "status":            "verification_status",
        "last_ev_date":      "amendment_date",
    })
    return df

# ── Build lookup tables from CSV and HF corpus ────────────────────────────────

def build_pdf_bsard_lookup(
    df: pd.DataFrame,
    hf_text: dict[int, str],
) -> tuple[
    dict[str, dict[str, dict]],   # exact: pdf_filename → {norm_art_no → row}
    dict[str, dict[str, list]],   # leading_int: pdf_filename → {lead_int → [rows]}
    dict[str, str],               # pdf_filename → dominant law_code
]:
    """
    Returns:
      exact_lookup[pdf_filename][norm_article_no]   → merged row dict
      lead_lookup[pdf_filename][leading_int]         → list of merged row dicts
      pdf_code_map[pdf_filename]                     → most common law code in that PDF
    """
    exact_lookup: dict[str, dict[str, dict]] = defaultdict(dict)
    lead_lookup: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    pdf_code_counts: dict[str, Counter] = defaultdict(Counter)

    for _, row in df.iterrows():
        if pd.isna(row.get("pdf_url")):
            continue

        pdf_fn = url_to_filename(str(row["pdf_url"]))
        art_no = str(row["article_no"]) if not pd.isna(row.get("article_no")) else ""
        bsard_id = int(row["bsard_id"])
        code = str(row["code"]) if not pd.isna(row.get("code")) else ""

        merged = {
            "bsard_id":           bsard_id,
            "is_bsard_article":   1,
            "law_code":           code,
            "law_type":           str(row["law_type"]) if not pd.isna(row.get("law_type")) else None,
            "justel_html_url":    str(row["justel_html_url"]) if not pd.isna(row.get("justel_html_url")) else None,
            "pdf_url":            str(row["pdf_url"]) if not pd.isna(row.get("pdf_url")) else None,
            "html_text_found":    int(row["html_text_found"]) if not pd.isna(row.get("html_text_found")) else None,
            "pdf_text_found":     int(row["pdf_text_found"]) if not pd.isna(row.get("pdf_text_found")) else None,
            "pdf_match_category": str(row["pdf_match_category"]) if not pd.isna(row.get("pdf_match_category")) else None,
            "verification_status":str(row["verification_status"]) if not pd.isna(row.get("verification_status")) else None,
            "amendment_date":     str(row["amendment_date"]) if not pd.isna(row.get("amendment_date")) else None,
            "article_status":     str(row["article_status"]) if not pd.isna(row.get("article_status")) else None,
            "is_pre_bsard":       int(row["is_pre_bsard"]) if not pd.isna(row.get("is_pre_bsard")) else None,
            "canonical_text":     hf_text.get(bsard_id),
        }

        norm_no = normalise_key(art_no)
        exact_lookup[pdf_fn][norm_no] = merged

        li = leading_int(art_no)
        if li:
            lead_lookup[pdf_fn][li].append(merged)

        pdf_code_counts[pdf_fn][code] += 1

    # Dominant code per PDF
    pdf_code_map: dict[str, str] = {
        fn: counts.most_common(1)[0][0]
        for fn, counts in pdf_code_counts.items()
    }

    return dict(exact_lookup), dict(lead_lookup), pdf_code_map

# ── Link a single Phase A article ─────────────────────────────────────────────

def link_article(
    art: dict,
    exact_lookup: dict[str, dict[str, dict]],
    lead_lookup: dict[str, dict[str, list]],
    pdf_code_map: dict[str, str],
) -> dict:
    """
    Merge BSARD metadata into a Phase A article record.
    Returns a new dict with all Phase B fields populated.
    """
    pdf_fn = art["pdf_filename"]
    art_no = str(art["article_number"])
    norm_no = normalise_key(art_no)

    # ── Try exact match ──────────────────────────────────────────────────────
    pdf_exact = exact_lookup.get(pdf_fn, {})
    bsard_row = pdf_exact.get(norm_no)

    # ── Fallback 2: leading-integer match (only if unique in this PDF) ───────
    if bsard_row is None:
        li = leading_int(art_no)
        if li:
            pdf_lead = lead_lookup.get(pdf_fn, {})
            candidates = pdf_lead.get(li, [])
            if len(candidates) == 1:
                bsard_row = candidates[0]

    # ── Fallback 3: base-article match (strips sub-paragraph suffix) ─────────
    # Handles PDFs where each BSARD article is split into numbered paragraphs,
    # e.g. PDF has "Art. 1.1.1-1", "Art. 1.1.1-2" → BSARD has "Art. 1.1.1".
    if bsard_row is None:
        base_no = base_article_no(art_no)
        if base_no:
            base_norm = normalise_key(base_no)
            bsard_row = pdf_exact.get(base_norm)
            if bsard_row is None:
                li_base = leading_int(base_no)
                if li_base:
                    candidates = lead_lookup.get(pdf_fn, {}).get(li_base, [])
                    if len(candidates) == 1:
                        bsard_row = candidates[0]

    # ── Build merged record ──────────────────────────────────────────────────
    out = dict(art)  # copy Phase A fields

    if bsard_row is not None:
        out["bsard_id"]            = bsard_row["bsard_id"]
        out["is_bsard_article"]    = 1
        out["law_code"]            = bsard_row["law_code"]
        out["law_type"]            = bsard_row["law_type"]
        out["justel_html_url"]     = bsard_row["justel_html_url"]
        out["pdf_url"]             = bsard_row["pdf_url"]
        out["html_text_found"]     = bsard_row["html_text_found"]
        out["pdf_text_found"]      = bsard_row["pdf_text_found"]
        out["pdf_match_category"]  = bsard_row["pdf_match_category"]
        out["verification_status"] = bsard_row["verification_status"]
        out["amendment_date"]      = bsard_row["amendment_date"]
        out["article_status"]      = bsard_row["article_status"]
        out["is_pre_bsard"]        = bsard_row["is_pre_bsard"]
        # Override article_text with canonical HF text when available.
        if bsard_row["canonical_text"]:
            out["article_text"]        = bsard_row["canonical_text"]
            out["article_text_source"] = "bsard_dataset"
    else:
        out["bsard_id"]            = None
        out["is_bsard_article"]    = 0
        out["law_code"]            = pdf_code_map.get(pdf_fn)
        out["law_type"]            = None
        out["justel_html_url"]     = None
        # pdf_url kept from Phase A (None); fill from pdf_code_map key
        out["html_text_found"]     = None
        out["pdf_text_found"]      = None
        out["pdf_match_category"]  = None
        out["verification_status"] = None
        out["amendment_date"]      = None
        out["article_status"]      = None
        out["is_pre_bsard"]        = None

    return out

# ── Load and resolve questions ─────────────────────────────────────────────────

def load_questions() -> list[dict]:
    """Load BSARD train + test questions from HuggingFace."""
    questions = []
    for split in ("train", "test"):
        ds = load_dataset("maastrichtlawtech/bsard", "questions", split=split)
        for q in ds:
            bsard_ids = [
                int(x.strip())
                for x in str(q["article_ids"]).split(",")
                if x.strip().isdigit()
            ]
            questions.append({
                "question_id":       int(q["id"]),
                "question_text":     q["question"],
                "category":          q.get("category"),
                "subcategory":       q.get("subcategory"),
                "extra_description": q.get("extra_description"),
                "relevant_bsard_ids": bsard_ids,
                "n_relevant_articles": len(bsard_ids),
                "split":             split,
            })
    return questions

# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase B: link Phase A articles to the BSARD benchmark"
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--extracted-dir", type=Path, default=DEFAULT_EXTRACTED_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load bsard_full_verify.csv ─────────────────────────────────────────
    print(f"Loading {args.csv.name} ...", end=" ", flush=True)
    df = load_verify_csv(args.csv)
    print(f"{len(df)} rows, {df['bsard_id'].nunique()} unique BSARD articles, "
          f"{df['pdf_url'].nunique()} PDFs")

    # ── 2. Load HF BSARD corpus ───────────────────────────────────────────────
    print("Loading HuggingFace BSARD corpus ...", end=" ", flush=True)
    corpus = load_dataset("maastrichtlawtech/bsard", "corpus", split="corpus")
    hf_text: dict[int, str] = {r["id"]: r["article"] for r in corpus}
    # Also extract article_no from the reference field for cross-checking.
    hf_art_no: dict[int, str] = {}
    for r in corpus:
        no = extract_article_no_from_reference(r["reference"])
        if no:
            hf_art_no[r["id"]] = no
    print(f"{len(hf_text)} articles loaded")

    # ── 3. Build lookup tables ────────────────────────────────────────────────
    print("Building linkage lookup tables ...", end=" ", flush=True)
    exact_lookup, lead_lookup, pdf_code_map = build_pdf_bsard_lookup(df, hf_text)
    total_exact_keys = sum(len(v) for v in exact_lookup.values())
    print(f"{len(exact_lookup)} PDFs, {total_exact_keys} exact keys")

    # ── 4. Load and link Phase A articles ─────────────────────────────────────
    jsonl_files = sorted(args.extracted_dir.glob("*.jsonl"))
    if not jsonl_files:
        print(f"No JSONL files found in {args.extracted_dir}")
        return

    print(f"\nLinking {len(jsonl_files)} JSONL files from {args.extracted_dir} ...")

    out_path = args.out_dir / "articles_linked.jsonl"
    matched = unmatched = 0

    with out_path.open("w", encoding="utf-8") as out_f:
        for jf in jsonl_files:
            file_matched = file_total = 0
            with jf.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    art = json.loads(line)
                    linked = link_article(art, exact_lookup, lead_lookup, pdf_code_map)
                    out_f.write(json.dumps(linked, ensure_ascii=False) + "\n")
                    if linked["is_bsard_article"]:
                        file_matched += 1
                    file_total += 1

            matched += file_matched
            unmatched += file_total - file_matched
            pct = file_matched / file_total * 100 if file_total else 0
            print(f"  {jf.name}: {file_matched}/{file_total} BSARD ({pct:.1f}%)")

    total = matched + unmatched
    print(f"\nTotal articles : {total}")
    print(f"BSARD matched  : {matched} / {len(hf_text)} ({matched/len(hf_text)*100:.1f}% of benchmark)")
    print(f"Non-BSARD      : {unmatched}")

    # Check for BSARD articles that were NOT matched (missed by Phase A extraction)
    matched_bsard_ids = set()
    with out_path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["is_bsard_article"] and r["bsard_id"] is not None:
                matched_bsard_ids.add(r["bsard_id"])
    missed = set(hf_text.keys()) - matched_bsard_ids
    print(f"BSARD missed   : {len(missed)} (not found in Phase A extraction)")
    if missed and len(missed) <= 20:
        print(f"  IDs: {sorted(missed)}")

    # ── 5. Append missed BSARD articles directly from HF corpus ──────────────
    # BSARD articles not found in Phase A still need to be in the database.
    # They get canonical HF text but no PDF positional or hierarchy metadata.
    if missed:
        print(f"\nAppending {len(missed)} missed BSARD articles from HF corpus ...")
        # Build a CSV lookup by bsard_id for metadata
        csv_by_id = df.set_index("bsard_id").to_dict("index")
        # Build pdf_url → pdf_filename map for these rows
        url_fn_map = {url: url_to_filename(url) for url in df["pdf_url"].dropna().unique()}

        with out_path.open("a", encoding="utf-8") as out_f:
            for bsard_id in sorted(missed):
                row = csv_by_id.get(bsard_id, {})
                pdf_url_val = str(row.get("pdf_url", "")) if row.get("pdf_url") else None
                art_no_val  = str(row.get("article_no", "")) if row.get("article_no") else None
                code_val    = str(row.get("code", "")) if row.get("code") else None
                pdf_fn_val  = url_fn_map.get(pdf_url_val) if pdf_url_val else None

                hf_corpus_row = corpus[bsard_id - 1] if bsard_id <= len(corpus) else None
                article_text  = hf_text.get(bsard_id)

                stub = {
                    # Phase A fields — empty since no PDF extraction
                    "article_number":      art_no_val,
                    "law_title_text":      None,
                    "chapter_title":       None,
                    "section_title":       None,
                    "subsection_title":    None,
                    "hierarchy_path":      [f"Art. {art_no_val}"] if art_no_val else [],
                    "hierarchy_depth":     0,
                    "article_text":        article_text,
                    "article_text_source": "bsard_dataset",
                    "has_truncation_markers": 0,
                    "pdf_filename":        pdf_fn_val,
                    "numac":               extract_numac(pdf_fn_val) if pdf_fn_val else None,
                    # Phase B fields
                    "bsard_id":            bsard_id,
                    "is_bsard_article":    1,
                    "law_code":            code_val,
                    "law_type":            str(row.get("law_type", "")) or None,
                    "pdf_url":             pdf_url_val,
                    "justel_html_url":     str(row.get("justel_html_url", "")) or None,
                    "html_text_found":     int(row["html_text_found"]) if not pd.isna(row.get("html_text_found")) else None,
                    "pdf_text_found":      int(row["pdf_text_found"]) if not pd.isna(row.get("pdf_text_found")) else None,
                    "pdf_match_category":  str(row.get("pdf_match_category", "")) or None,
                    "verification_status": str(row.get("verification_status", "")) or None,
                    "amendment_date":      str(row.get("amendment_date", "")) or None,
                    "article_status":      str(row.get("article_status", "")) or None,
                    "is_pre_bsard":        int(row["is_pre_bsard"]) if not pd.isna(row.get("is_pre_bsard")) else None,
                    "pdf_page_numbers":    [],
                    "pdf_page_start":      None,
                    "pdf_page_end":        None,
                }
                out_f.write(json.dumps(stub, ensure_ascii=False) + "\n")

        print(f"  Appended {len(missed)} articles -> {out_path}")

    # ── 6. Load and save questions ─────────────────────────────────────────────
    print("\nLoading BSARD questions ...", end=" ", flush=True)
    questions = load_questions()
    print(f"{len(questions)} questions (train + test)")

    q_path = args.out_dir / "questions.jsonl"
    with q_path.open("w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    train_n = sum(1 for q in questions if q["split"] == "train")
    test_n  = sum(1 for q in questions if q["split"] == "test")
    print(f"  train={train_n}, test={test_n}")
    print(f"  Saved to {q_path}")

    print(f"\nPhase B complete.")
    print(f"  Articles : {out_path}  ({out_path.stat().st_size // 1024} KB)")
    print(f"  Questions: {q_path}   ({q_path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
