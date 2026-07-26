"""Concatenate the 5 per-PDF cross_arm_comparison.csv files into one
long-format table for plotting / inclusion in the synthesis report.

Adds a ``pdf_stem`` column and a row count.

Usage:
    python scripts/combine_cross_arm_csvs.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent.parent
PDF_STEMS = [
    "1804_03_21_1804032150",
    "1867_06_08_1867060850",
    "1967_10_10_1967101055",
    "1967_10_10_1967101056",
    "2003_07_17_2013A31614",
]


def main() -> None:
    pieces: list[pd.DataFrame] = []
    for stem in PDF_STEMS:
        fp = ROOT / f"RQ2_T05_ARM2_PAGEINDEX/data/{stem}/experiments/cross_arm_comparison.csv"
        if not fp.exists():
            print(f"  MISSING: {fp}")
            continue
        df = pd.read_csv(fp)
        df.insert(0, "pdf_stem", stem)
        pieces.append(df)
        print(f"  loaded {stem}: {len(df)} rows")

    if not pieces:
        raise RuntimeError("no per-PDF CSVs found")
    combined = pd.concat(pieces, ignore_index=True)
    out_path = ROOT / "RQ2_T05_ARM2_PAGEINDEX/data/cross_pdf_cross_arm_long.csv"
    combined.to_csv(out_path, index=False)
    print(f"\nWrote {out_path} ({len(combined)} rows)")


if __name__ == "__main__":
    main()
