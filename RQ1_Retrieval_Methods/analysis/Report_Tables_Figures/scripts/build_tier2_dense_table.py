"""
Build the RQ1 Tier 2 (dense) results table -- Chapter 6.

Reads the nine canonical Tier 2 dense result records (222-question BSARD test
split) and emits one GitHub-flavoured **markdown** results table for the thesis
report.

Output:
  - printed to stdout
  - written to analysis/tables/tier2_dense_results.md

Columns (in order):
  System | Size | Weighting | R@1 | R@5 | R@10 | R@50 | R@100
         | MRR@100 | NDCG@10 | MAP@10 | Hit@10

Config columns (the Tier 2 analogue of Tier 1's Normalisation / Fields -- dense
encoders have no normalisation step):
  - Size      : approximate parameter count of the encoder. These are REFERENCE
                FACTS taken from the public model cards, not values computed from
                the result records (there is no param count in the JSON). They
                make the "scale matters" finding legible (base -> large climbing)
                rather than asserted in prose. See SIZES below.
  - Weighting : text (text_only) vs text+meta (concat_2x), read from the record's
                preprocessing.field_weighting.

Conventions:
  - Dense systems are first-stage retrievers, so every cut-off is genuine -- no
    cells are masked.
  - Metrics rounded to 3 dp; the best value in each metric column is **bolded**.
  - HitRate@10 maps to the Hit@10 column.

Row order: by Recall@100 descending. This puts the strongest encoders on top and
sinks the collapsed CamemBERT MLM baseline to the bottom.

Usage:
  python analysis/Report_Tables_Figures/scripts/build_tier2_dense_table.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPORT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPORT_DIR))

from load_results import load_records  # noqa: E402

OUT_DIR = REPORT_DIR.parent / "tables"          # <project>/analysis/tables
OUT_FILE = OUT_DIR / "tier2_dense_results.md"

# experiment_id -> (display name, approximate parameter count).
# Sizes are public-model-card reference facts (NOT record-derived); see docstring.
SPEC: dict[str, tuple[str, str]] = {
    "dense_me5_large_concat2x_zeroshot_test":   ("Multilingual E5-large", "560M"),
    "dense_me5_large_zeroshot_test":            ("Multilingual E5-large", "560M"),
    "dense_me5_base_zeroshot_test":             ("Multilingual E5-base", "278M"),
    "dense_bge_m3_zeroshot_test":               ("BGE-M3", "568M"),
    "dense_qwen3_0_6b_instruct_zeroshot_test":  ("Qwen3-Embedding-0.6B (instruction-prefixed)", "600M"),
    "dense_qwen3_0_6b_zeroshot_test":           ("Qwen3-Embedding-0.6B", "600M"),
    "dense_mpnet_multi_zeroshot_test":          ("Multilingual MPNet", "278M"),
    "dense_camembert_lg_zeroshot_test":         ("CamemBERT-large", "335M"),
    "dense_camembert_base_zeroshot_test":       ("CamemBERT-base", "110M"),
}

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

_FIELDS = {"text_only": "text", "concat_2x": "text+meta"}


def _records_by_id() -> dict:
    """experiment_id -> ResultRecord for every loaded result JSON."""
    return {rec.experiment_id: rec for _tier, rec in load_records()}


def build() -> str:
    recs = _records_by_id()

    missing = [eid for eid in SPEC if eid not in recs]
    if missing:
        raise SystemExit(
            "Missing Tier 2 dense records (is the data bundle present under "
            f"output/results/dense_retrieval/?):\n  " + "\n  ".join(missing)
        )

    # Build one row per spec'd record, then order by Recall@100 descending.
    rows = []   # (system, size, weighting, [metric values])
    for eid, (label, size) in SPEC.items():
        rec = recs[eid]
        fw = rec.preprocessing.get("field_weighting", "")
        weighting = _FIELDS.get(fw, fw or "--")
        values = [float(rec.metrics[key]) for _hdr, key in METRIC_COLS]
        rows.append((label, size, weighting, values))

    r100_idx = [h for h, _ in METRIC_COLS].index("R@100")
    rows.sort(key=lambda r: r[3][r100_idx], reverse=True)

    # Column maxima, compared on the displayed (3 dp) value so the bolded cell
    # always matches what the reader sees.
    col_max = [max(round(r[3][c], 3) for r in rows) for c in range(len(METRIC_COLS))]

    def cell(value: float, col: int) -> str:
        s = f"{value:.3f}"
        return f"**{s}**" if round(value, 3) == col_max[col] else s

    headers = ["System", "Size", "Weighting"] + [h for h, _ in METRIC_COLS]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for system, size, weighting, values in rows:
        cells = [system, size, weighting] + [cell(v, c) for c, v in enumerate(values)]
        lines.append("| " + " | ".join(cells) + " |")

    table = "\n".join(lines) + "\n"
    note = ("\n_Size = approximate parameter count from the public model card "
            "(reference fact, not measured from the records)._\n")
    return table + note


def main() -> None:
    table = build()
    print(table)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(table, encoding="utf-8")
    print(f"[done] Wrote {OUT_FILE.relative_to(REPORT_DIR.parent.parent)}")


if __name__ == "__main__":
    main()
