"""Build the Arm-2C Simple-navigation vs CNR cost+outcome table.

Output: Report/tables/tab_rq2_arm2c_navigation.tex  (tabular only).

Per document (large-first) plus corpus-weighted, the two §4e configs side by side:
  - Simple navigation = shipped Arm-2C (single greedy descent, 8B re-rank).
  - CNR               = Corrective Navigation and Re-Ranking (corrective loop +
                        e5 re-rank over the corrective reach).
Three metric blocks, each split Simple | CNR:
  LLM calls/q  -- navigation cost (the corrective loop roughly doubles it);
  Reach        -- padding-free navigation coverage (the corrective lever);
  R@10         -- the headline (the e5-ranking lever converting reach to rank).
Reach / R@10 are question-weighted (matching the §4e headline numbers); both
come from the controlled E2 per-query output, so the contrast is confound-free.

Usage:
  .venv/Scripts/python Report/scripts/build_arm2c_navigation_table.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPORT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import load_arm2c as la  # noqa: E402
from thesis_style import fmt_metric, save_table  # noqa: E402

OUTPUT_NAME = "tab_rq2_arm2c_navigation"


def main() -> None:
    df = la.arm2c_e2_cost_perf()

    rows: list[str] = []
    for _, r in df.iterrows():
        is_corpus = r["stem"] == "corpus"
        name = rf"\textbf{{{r['stem_label']}}}" if is_corpus else r["stem_label"]
        cells = [
            f"{name} ({int(r['n'])})",
            f"{r['llm_single']:.1f}", f"{r['llm_cnr']:.1f}",
            fmt_metric("R@10", r["reach_single"]), fmt_metric("R@10", r["reach_cnr"]),
            fmt_metric("R@10", r["r10_single"]),
            rf"\textbf{{{fmt_metric('R@10', r['r10_cnr'])}}}",
        ]
        if is_corpus:
            rows.append(r"\midrule")
        rows.append(" & ".join(cells) + r" \\")

    out = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r" & \multicolumn{2}{c}{LLM calls/q} & \multicolumn{2}{c}{Reach} "
        r"& \multicolumn{2}{c}{Recall@10} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
        r"PDF ($n$) & Simple & CNR & Simple & CNR & Simple & CNR \\",
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
