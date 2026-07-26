"""Phase E — Corpus statistics for Chapter 3 of the thesis.

Reads:
  output/bsard_corpus.db
  output/linked/questions.jsonl

Writes:
  output/corpus_stats.json   — machine-readable statistics
  stdout                     — human-readable report

Usage:
    python retrieval/corpus_stats.py
    python retrieval/corpus_stats.py --db output/bsard_corpus.db
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from pathlib import Path

import numpy as np

# ── Paths ───────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB   = PROJECT_ROOT / "output" / "bsard_corpus.db"
DEFAULT_QS   = PROJECT_ROOT / "output" / "linked" / "questions.jsonl"
DEFAULT_OUT  = PROJECT_ROOT / "output" / "corpus_stats.json"

# ── Helpers ──────────────────────────────────────────────────────────────────

_TOK_RE = re.compile(r"[^\w]", re.UNICODE)


def tokenise(text: str) -> set[str]:
    return {t for t in _TOK_RE.split(text.lower()) if t}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def percentiles(values: list[float], pcts: list[int]) -> dict[str, float]:
    arr = np.array(values, dtype=float)
    return {f"p{p}": float(np.percentile(arr, p)) for p in pcts}


def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase E: corpus statistics")
    parser.add_argument("--db",  type=Path, default=DEFAULT_DB)
    parser.add_argument("--qs",  type=Path, default=DEFAULT_QS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if not args.db.exists():
        print(f"Database not found: {args.db}")
        print("Run build_database.py first.")
        return

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row

    stats: dict = {}

    # ── 1. Corpus overview ───────────────────────────────────────────────────
    print("1. Corpus overview ...")
    total       = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
    n_bsard     = conn.execute("SELECT COUNT(*) FROM articles WHERE is_bsard_article=1").fetchone()[0]
    n_non_bsard = conn.execute("SELECT COUNT(*) FROM articles WHERE is_bsard_article=0").fetchone()[0]
    n_law_codes = conn.execute("SELECT COUNT(DISTINCT law_code) FROM articles").fetchone()[0]
    n_pdfs      = conn.execute("SELECT COUNT(DISTINCT pdf_filename) FROM articles WHERE pdf_filename IS NOT NULL").fetchone()[0]
    unique_bsard_ids = conn.execute(
        "SELECT COUNT(DISTINCT bsard_id) FROM articles WHERE bsard_id IS NOT NULL"
    ).fetchone()[0]

    stats["corpus_overview"] = {
        "total_articles":       total,
        "bsard_articles":       n_bsard,
        "non_bsard_articles":   n_non_bsard,
        "unique_bsard_ids":     unique_bsard_ids,
        "unique_law_codes":     n_law_codes,
        "unique_pdfs":          n_pdfs,
    }

    # ── 2. Article length distribution ──────────────────────────────────────
    print("2. Article length distribution ...")
    rows = conn.execute(
        "SELECT token_count, char_count FROM articles WHERE article_text IS NOT NULL"
    ).fetchall()
    tok_vals  = [r["token_count"]  for r in rows if r["token_count"]  is not None]
    char_vals = [r["char_count"]   for r in rows if r["char_count"]   is not None]

    pcts = [10, 25, 50, 75, 90, 95, 99]
    stats["article_length"] = {
        "n_with_text":        len(rows),
        "token_count":        {
            "mean":   float(np.mean(tok_vals)),
            "std":    float(np.std(tok_vals)),
            **percentiles(tok_vals, pcts),
        },
        "char_count":         {
            "mean":   float(np.mean(char_vals)),
            "std":    float(np.std(char_vals)),
            **percentiles(char_vals, pcts),
        },
    }

    # ── 3. Text source breakdown ─────────────────────────────────────────────
    print("3. Text source breakdown ...")
    src_rows = conn.execute(
        "SELECT article_text_source, COUNT(*) AS n FROM articles GROUP BY article_text_source ORDER BY n DESC"
    ).fetchall()
    stats["text_source"] = {r["article_text_source"]: r["n"] for r in src_rows}

    # ── 4. Hierarchy coverage ────────────────────────────────────────────────
    print("4. Hierarchy coverage ...")
    def _pct(col: str) -> float:
        n = conn.execute(f"SELECT COUNT(*) FROM articles WHERE {col} IS NOT NULL").fetchone()[0]
        return round(100.0 * n / total, 2)

    stats["hierarchy_coverage"] = {
        "has_law_title":    _pct("law_title_text"),
        "has_chapter":      _pct("chapter_title"),
        "has_section":      _pct("section_title"),
        "has_subsection":   _pct("subsection_title"),
        "has_hierarchy_path": _pct("hierarchy_path"),
    }

    # ── 5. Cross-reference density ───────────────────────────────────────────
    print("5. Cross-reference density ...")
    total_edges   = conn.execute("SELECT COUNT(*) FROM citation_graph").fetchone()[0]
    total_outgoing = conn.execute("SELECT SUM(n_outgoing_refs) FROM articles").fetchone()[0] or 0
    total_cited_by = conn.execute("SELECT SUM(n_cited_by) FROM articles").fetchone()[0] or 0
    has_refs      = conn.execute("SELECT COUNT(*) FROM articles WHERE has_cross_references=1").fetchone()[0]
    max_cited     = conn.execute("SELECT MAX(n_cited_by), article_id, law_code, article_number FROM articles").fetchone()

    # Per-law average outgoing refs
    law_ref_rows = conn.execute(
        "SELECT law_code, AVG(n_outgoing_refs) AS avg_out FROM articles GROUP BY law_code ORDER BY avg_out DESC LIMIT 10"
    ).fetchall()

    stats["cross_references"] = {
        "total_citation_edges":        total_edges,
        "total_raw_outgoing_refs":     int(total_outgoing),
        "articles_with_references":    has_refs,
        "pct_with_references":         round(100.0 * has_refs / total, 2),
        "max_in_degree":               max_cited[0],
        "max_in_degree_article_id":    max_cited[1],
        "max_in_degree_law_code":      max_cited[2],
        "max_in_degree_article_no":    max_cited[3],
        "top10_law_avg_outgoing":      [
            {"law_code": r["law_code"], "avg_outgoing": round(r["avg_out"], 2)}
            for r in law_ref_rows
        ],
    }

    # ── 6. Extraction failure rate ───────────────────────────────────────────
    print("6. Extraction failure rate ...")
    n_no_text   = conn.execute("SELECT COUNT(*) FROM articles WHERE article_text IS NULL").fetchone()[0]
    n_pdf_fail  = conn.execute(
        "SELECT COUNT(*) FROM articles WHERE article_text_source='pdf_extraction_failed'"
    ).fetchone()[0]
    n_hf_only   = conn.execute(
        "SELECT COUNT(*) FROM articles WHERE article_text_source='bsard_dataset'"
    ).fetchone()[0]
    n_pdf_html  = conn.execute(
        "SELECT COUNT(*) FROM articles WHERE article_text_source='pdf_extracted'"
    ).fetchone()[0]
    n_html      = conn.execute(
        "SELECT COUNT(*) FROM articles WHERE article_text_source='html_extracted'"
    ).fetchone()[0]

    match_cats = conn.execute(
        "SELECT pdf_match_category, COUNT(*) AS n FROM articles "
        "WHERE pdf_match_category IS NOT NULL GROUP BY pdf_match_category ORDER BY n DESC"
    ).fetchall()

    stats["extraction"] = {
        "no_text":            n_no_text,
        "pct_no_text":        round(100.0 * n_no_text / total, 2),
        "source_pdf_only":    n_pdf_html,
        "source_html_only":   n_html,
        "source_hf_dataset":  n_hf_only,
        "source_pdf_failed":  n_pdf_fail,
        "pdf_match_categories": {r["pdf_match_category"]: r["n"] for r in match_cats},
    }

    # ── 7. Question statistics ───────────────────────────────────────────────
    print("7. Question statistics ...")
    n_questions = conn.execute("SELECT COUNT(*) FROM questions").fetchone()[0]
    n_train     = conn.execute("SELECT COUNT(*) FROM questions WHERE split='train'").fetchone()[0]
    n_test      = conn.execute("SELECT COUNT(*) FROM questions WHERE split='test'").fetchone()[0]
    n_multi     = conn.execute(
        "SELECT COUNT(*) FROM questions WHERE n_relevant_articles > 1"
    ).fetchone()[0]
    avg_rel     = conn.execute("SELECT AVG(n_relevant_articles) FROM questions").fetchone()[0]

    rel_dist = conn.execute(
        "SELECT n_relevant_articles, COUNT(*) AS n FROM questions GROUP BY n_relevant_articles ORDER BY n_relevant_articles"
    ).fetchall()

    stats["questions"] = {
        "total":              n_questions,
        "train":              n_train,
        "test":               n_test,
        "multi_article":      n_multi,
        "pct_multi_article":  round(100.0 * n_multi / n_questions, 2),
        "avg_relevant_articles": round(float(avg_rel), 3),
        "relevant_articles_distribution": {str(r["n_relevant_articles"]): r["n"] for r in rel_dist},
    }

    # ── 8. Jaccard lexical overlap ───────────────────────────────────────────
    print("8. Jaccard lexical overlap (this may take a moment) ...")
    if args.qs.exists():
        questions = _load_jsonl(args.qs)

        # Build bsard_id -> article_text lookup (first/canonical record per bsard_id)
        bsard_text: dict[int, str] = {}
        for row in conn.execute(
            "SELECT bsard_id, article_text FROM articles "
            "WHERE bsard_id IS NOT NULL AND article_text IS NOT NULL "
            "ORDER BY article_id"
        ):
            bid = row[0]
            if bid not in bsard_text:
                bsard_text[bid] = row[1]

        jaccard_scores: list[float] = []
        q_tok_counts: list[int] = []
        for q in questions:
            q_toks = tokenise(q["question_text"])
            q_tok_counts.append(len(q_toks))
            bsard_ids = q.get("relevant_bsard_ids", [])

            for bid in bsard_ids:
                if bid in bsard_text:
                    a_toks = tokenise(bsard_text[bid])
                    jaccard_scores.append(jaccard(q_toks, a_toks))

        stats["jaccard_overlap"] = {
            "n_pairs":      len(jaccard_scores),
            "mean":         round(float(np.mean(jaccard_scores)), 4) if jaccard_scores else None,
            "std":          round(float(np.std(jaccard_scores)), 4) if jaccard_scores else None,
            **({k: round(v, 4) for k, v in percentiles(jaccard_scores, pcts).items()} if jaccard_scores else {}),
            "query_token_count": {
                "mean": round(float(np.mean(q_tok_counts)), 2),
                **{k: round(v, 1) for k, v in percentiles(q_tok_counts, pcts).items()},
            },
        }
    else:
        print(f"  questions.jsonl not found at {args.qs}, skipping Jaccard.")
        stats["jaccard_overlap"] = None

    # ── 9. Temporal metadata ─────────────────────────────────────────────────
    print("9. Temporal metadata ...")
    pre_bsard  = conn.execute("SELECT COUNT(*) FROM articles WHERE is_pre_bsard=1").fetchone()[0]
    post_bsard = conn.execute("SELECT COUNT(*) FROM articles WHERE is_pre_bsard=0").fetchone()[0]
    pre_null   = conn.execute("SELECT COUNT(*) FROM articles WHERE is_pre_bsard IS NULL").fetchone()[0]
    has_amend  = conn.execute("SELECT COUNT(*) FROM articles WHERE amendment_date IS NOT NULL").fetchone()[0]

    status_rows = conn.execute(
        "SELECT article_status, COUNT(*) AS n FROM articles GROUP BY article_status ORDER BY n DESC"
    ).fetchall()

    stats["temporal"] = {
        "pre_bsard":           pre_bsard,
        "post_bsard_or_same":  post_bsard,
        "pre_bsard_unknown":   pre_null,
        "has_amendment_date":  has_amend,
        "article_status":      {(r["article_status"] or "NULL"): r["n"] for r in status_rows},
    }

    # ── 10. Verification status ──────────────────────────────────────────────
    print("10. Verification status ...")
    verif_rows = conn.execute(
        "SELECT verification_status, COUNT(*) AS n FROM articles GROUP BY verification_status ORDER BY n DESC"
    ).fetchall()
    stats["verification_status"] = {(r["verification_status"] or "NULL"): r["n"] for r in verif_rows}

    conn.close()

    # ── Save JSON ────────────────────────────────────────────────────────────
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {args.out}")

    # ── Human-readable report ────────────────────────────────────────────────
    ov  = stats["corpus_overview"]
    al  = stats["article_length"]
    src = stats["text_source"]
    hc  = stats["hierarchy_coverage"]
    cr  = stats["cross_references"]
    ex  = stats["extraction"]
    qs  = stats["questions"]
    jq  = stats["jaccard_overlap"]
    tm  = stats["temporal"]
    vs  = stats["verification_status"]

    print("""
================================================================================
 BSARD CORPUS — STATISTICS REPORT (Phase E)
================================================================================

1. CORPUS OVERVIEW
   Total articles         : {total:,}
   BSARD articles         : {bsard:,}  ({pct_bsard:.1f}%)
   Non-BSARD articles     : {non_bsard:,}  ({pct_non:.1f}%)
   Unique BSARD IDs       : {ubsard:,}
   Unique law codes       : {law_codes:,}
   Unique PDFs            : {pdfs:,}

2. ARTICLE LENGTH (articles with text, n={n_text:,})
   Token count  mean={tok_mean:.0f}  std={tok_std:.0f}
     p10={tok_p10:.0f}  p25={tok_p25:.0f}  p50={tok_p50:.0f}  p75={tok_p75:.0f}  p90={tok_p90:.0f}  p95={tok_p95:.0f}  p99={tok_p99:.0f}
   Char count   mean={chr_mean:.0f}  std={chr_std:.0f}
     p10={chr_p10:.0f}  p25={chr_p25:.0f}  p50={chr_p50:.0f}  p75={chr_p75:.0f}  p90={chr_p90:.0f}  p95={chr_p95:.0f}  p99={chr_p99:.0f}

3. TEXT SOURCE BREAKDOWN
{src_lines}

4. HIERARCHY COVERAGE
   Law title    : {ht:.1f}%
   Chapter      : {ch:.1f}%
   Section      : {sc:.1f}%
   Subsection   : {ss:.1f}%
   Hierarchy path: {hp:.1f}%

5. CROSS-REFERENCES
   Total citation edges   : {edges:,}
   Articles with refs     : {has_refs:,}  ({pct_refs:.1f}%)
   Max in-degree          : {max_in} (Art.{max_art_no}, {max_law})

6. EXTRACTION
   No text                : {no_text:,}  ({pct_no_text:.1f}%)
   Source = pdf           : {src_pdf:,}
   Source = html          : {src_html:,}
   Source = bsard_dataset : {src_hf:,}
   Source = failed        : {src_fail:,}

7. QUESTIONS
   Total                  : {n_q:,}  (train={n_train:,}, test={n_test:,})
   Multi-article          : {n_multi:,}  ({pct_multi:.1f}%)
   Avg relevant articles  : {avg_rel}

8. JACCARD LEXICAL OVERLAP (query vs. relevant article)
{jac_lines}

9. TEMPORAL METADATA
   Pre-BSARD              : {pre:,}
   Post/Same-era          : {post:,}
   Unknown                : {unk:,}
   Has amendment date     : {amend:,}

10. VERIFICATION STATUS
{verif_lines}
================================================================================
""".format(
        total=ov["total_articles"], bsard=ov["bsard_articles"],
        pct_bsard=100*ov["bsard_articles"]/ov["total_articles"],
        non_bsard=ov["non_bsard_articles"],
        pct_non=100*ov["non_bsard_articles"]/ov["total_articles"],
        ubsard=ov["unique_bsard_ids"], law_codes=ov["unique_law_codes"], pdfs=ov["unique_pdfs"],

        n_text=al["n_with_text"],
        tok_mean=al["token_count"]["mean"], tok_std=al["token_count"]["std"],
        tok_p10=al["token_count"]["p10"], tok_p25=al["token_count"]["p25"],
        tok_p50=al["token_count"]["p50"], tok_p75=al["token_count"]["p75"],
        tok_p90=al["token_count"]["p90"], tok_p95=al["token_count"]["p95"],
        tok_p99=al["token_count"]["p99"],
        chr_mean=al["char_count"]["mean"], chr_std=al["char_count"]["std"],
        chr_p10=al["char_count"]["p10"], chr_p25=al["char_count"]["p25"],
        chr_p50=al["char_count"]["p50"], chr_p75=al["char_count"]["p75"],
        chr_p90=al["char_count"]["p90"], chr_p95=al["char_count"]["p95"],
        chr_p99=al["char_count"]["p99"],

        src_lines="\n".join(f"   {k:<35}: {v:,}" for k, v in src.items()),

        ht=hc["has_law_title"], ch=hc["has_chapter"],
        sc=hc["has_section"], ss=hc["has_subsection"],
        hp=hc["has_hierarchy_path"],

        edges=cr["total_citation_edges"],
        has_refs=cr["articles_with_references"],
        pct_refs=cr["pct_with_references"],
        max_in=cr["max_in_degree"],
        max_art_no=cr["max_in_degree_article_no"],
        max_law=cr["max_in_degree_law_code"],

        no_text=ex["no_text"], pct_no_text=ex["pct_no_text"],
        src_pdf=ex["source_pdf_only"], src_html=ex["source_html_only"],
        src_hf=ex["source_hf_dataset"], src_fail=ex["source_pdf_failed"],

        n_q=qs["total"], n_train=qs["train"], n_test=qs["test"],
        n_multi=qs["multi_article"], pct_multi=qs["pct_multi_article"],
        avg_rel=qs["avg_relevant_articles"],

        jac_lines=(
            "   n_pairs={n}  mean={mean}  std={std}\n"
            "   p10={p10}  p25={p25}  p50={p50}  p75={p75}  p90={p90}  p95={p95}  p99={p99}".format(
                n=jq["n_pairs"], mean=jq["mean"], std=jq["std"],
                p10=jq.get("p10"), p25=jq.get("p25"), p50=jq.get("p50"),
                p75=jq.get("p75"), p90=jq.get("p90"), p95=jq.get("p95"), p99=jq.get("p99"),
            ) if jq and jq.get("n_pairs", 0) > 0 else "   (skipped)"
        ),

        pre=tm["pre_bsard"], post=tm["post_bsard_or_same"],
        unk=tm["pre_bsard_unknown"], amend=tm["has_amendment_date"],

        verif_lines="\n".join(f"   {k:<35}: {v:,}" for k, v in vs.items()),
    ))

    print("Phase E (corpus_stats) complete.")


if __name__ == "__main__":
    main()
