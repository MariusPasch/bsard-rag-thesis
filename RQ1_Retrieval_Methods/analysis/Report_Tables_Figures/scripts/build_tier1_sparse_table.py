"""
Build the RQ1 Tier 1 (sparse) results table -- Chapter 5, Table 5.1.

Reads the twelve canonical Tier 1 sparse result records (222-question BSARD
test split) and emits one GitHub-flavoured **markdown** results table for the
thesis report.

Output:
  - printed to stdout
  - written to analysis/tables/tier1_sparse_results.md

Columns (in order):
  System | Normalisation | Fields | R@1 | R@5 | R@10 | R@50 | R@100
         | MRR@100 | NDCG@10 | MAP@10 | Hit@10

Conventions for this tier:
  - Sparse systems are first-stage retrievers, so every cut-off is genuine --
    no cells are masked. (Sparse recall caps at R@100; R@100 == R@200 == R@500,
    so R@100 is kept with no footnote.)
  - One record per configuration => no file-selection ambiguity.
  - Metrics rounded to 3 dp; the best value in each metric column is **bolded**.
  - HitRate@10 maps to the Hit@10 column.

Row order: BM25 Okapi variants -> BM25 Plus -> BM25 L -> TF-IDF -> FTS5, with
the paper-aligned anchor last. The display label + order are curated below; the
Normalisation / Fields columns and all metrics are read straight from the
record so the table cannot drift from the data.

Usage:
  python analysis/Report_Tables_Figures/scripts/build_tier1_sparse_table.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPORT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPORT_DIR))

from load_results import load_records  # noqa: E402

OUT_DIR = REPORT_DIR.parent / "tables"          # <project>/analysis/tables
OUT_FILE = OUT_DIR / "tier1_sparse_results.md"

# Curated row order + display label, keyed by experiment_id. Order encodes the
# method grouping (BM25 Okapi -> Plus -> L -> TF-IDF -> FTS5, anchor last).
ROWS: list[tuple[str, str]] = [
    # ---- BM25, Okapi variant ------------------------------------------------
    ("bm25_tuned_k11.5_b0.25_test", "Tuned BM25 (Okapi, k1=1.5, b=0.25)"),
    ("bm25_lemmatize_concat_2x_test", "BM25 (Okapi)"),
    ("bm25_none_concat_2x_test", "BM25 (Okapi)"),
    ("bm25_stem_text_only_test", "BM25 (Okapi)"),
    ("bm25_lemmatize_text_only_test", "BM25 (Okapi)"),
    # ---- BM25, other variants ----------------------------------------------
    ("bm25_plus_lemmatize_test", "BM25 Plus"),
    ("bm25_l_lemmatize_test", "BM25 L"),
    # ---- TF-IDF -------------------------------------------------------------
    ("tfidf_lemmatize_concat_2x_test", "TF-IDF"),
    ("tfidf_none_text_only_test", "TF-IDF"),
    ("tfidf_lemmatize_text_only_test", "TF-IDF"),
    # ---- FTS5 ---------------------------------------------------------------
    ("fts5_default_test", "FTS5 (default)"),
    # ---- paper-aligned anchor (last) ---------------------------------------
    ("bm25_anchor_test", "Paper-aligned BM25 (anchor)"),
]

# (column header, record metrics key). All are higher-is-better.
METRIC_COLS: list[tuple[str, str]] = [
    ("R@1", "Recall@1"),
    ("R@5", "Recall@5"),
    ("R@10", "Recall@10"),
    ("R@50", "Recall@50"),
    ("R@100", "Recall@100"),
    ("MRR@100", "MRR@100"),
    ("NDCG@10", "NDCG@10"),
    ("MAP@10", "MAP@10"),
    ("Hit@10", "HitRate@10"),
]

_NORM = {"none": "none", "stem": "stem", "lemmatize": "lemma"}
_FIELDS = {"text_only": "text", "concat_2x": "text+meta"}


def _records_by_id() -> dict:
    """experiment_id -> ResultRecord for every loaded result JSON."""
    return {rec.experiment_id: rec for _tier, rec in load_records()}


def build() -> str:
    recs = _records_by_id()

    missing = [eid for eid, _ in ROWS if eid not in recs]
    if missing:
        raise SystemExit(
            "Missing Tier 1 sparse records (is the data bundle present under "
            f"output/results/sparse_retrieval/?):\n  " + "\n  ".join(missing)
        )

    # Gather raw values per row so we can find the column maxima for bolding.
    raw: list[list[float]] = []
    meta: list[tuple[str, str, str]] = []   # (system, normalisation, fields)
    for eid, label in ROWS:
        rec = recs[eid]
        pp = rec.preprocessing
        meta.append((
            label,
            _NORM.get(pp.get("normalization", ""), pp.get("normalization", "--")),
            _FIELDS.get(pp.get("field_weighting", ""), pp.get("field_weighting", "--")),
        ))
        raw.append([float(rec.metrics[key]) for _hdr, key in METRIC_COLS])

    # Column maxima, compared on the displayed (3 dp) value so the bolded cell
    # always matches what the reader sees.
    col_max = [max(round(r[c], 3) for r in raw) for c in range(len(METRIC_COLS))]

    def cell(value: float, col: int) -> str:
        s = f"{value:.3f}"
        return f"**{s}**" if round(value, 3) == col_max[col] else s

    headers = ["System", "Normalisation", "Fields"] + [h for h, _ in METRIC_COLS]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for (system, norm, fields), values in zip(meta, raw):
        cells = [system, norm, fields] + [cell(v, c) for c, v in enumerate(values)]
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines) + "\n"


def main() -> None:
    table = build()
    print(table)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(table, encoding="utf-8")
    print(f"[done] Wrote {OUT_FILE.relative_to(REPORT_DIR.parent.parent)}")


if __name__ == "__main__":
    main()
