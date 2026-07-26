"""Build the Arm-2C component-ablation table (Code Civil).

Output: Report/tables/tab_rq2_arm2c_ablation.tex  (tabular only).

Columns: Configuration | R@10 | R@100 | Hit@10 | empty-selection q
Three cumulative steps on the Code Civil ($n=252$): deep tree alone (labels
only) -> + native metadata -> + re-rank. Shows that metadata fixes selection
(empty-selection queries fall, R@100 rises) and the re-rank converts the reached
pool into top-10 (R@10 jumps while R@100 is ~flat). Bold = best per column.

Usage:
  .venv/Scripts/python Report/scripts/build_arm2c_ablation_table.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPORT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import load_arm2c as la  # noqa: E402
from thesis_style import fmt_metric, save_table  # noqa: E402

OUTPUT_NAME = "tab_rq2_arm2c_ablation"
METRICS = [("R@10", "R@10", False), ("R@100", "R@100", False),
           ("Hit@10", "Hit@10", False), ("empty_sel_q", "empty-sel. $q$", True)]


def main() -> None:
    abl = la.arm2c_ablation()

    best: dict[str, float] = {}
    for col, _hdr, lower_better in METRICS:
        vals = abl[col].astype(float)
        best[col] = vals.min() if lower_better else vals.max()

    rows: list[str] = []
    for _, r in abl.iterrows():
        cells = [r["label"]]
        for col, _hdr, lower_better in METRICS:
            v = float(r[col])
            if col == "empty_sel_q":
                txt = f"{int(v)} / {int(r['n'])}"
            else:
                txt = fmt_metric(col, v)
            if abs(v - best[col]) < 1e-9:
                txt = rf"\textbf{{{txt}}}"
            cells.append(txt)
        rows.append(" & ".join(cells) + r" \\")

    header = " & ".join(hdr for _c, hdr, _l in METRICS)
    out = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        rf"Configuration & {header} \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
    ]
    tex = "\n".join(out) + "\n"
    out_path = save_table(tex, OUTPUT_NAME)
    print(f"[done] Wrote {out_path.relative_to(REPORT_DIR.parent)}")


if __name__ == "__main__":
    main()
