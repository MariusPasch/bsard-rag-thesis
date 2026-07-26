"""Build the RQ2 cost table.

Output: Report/tables/tab_rq2_cost.tex (tabular only).

Columns: Arm | LLM calls/q | tokens/q | latency mean (s) | latency p95 (s)
         | E/R@10. Three canonical arms. Pooled over the 5 PDFs (725 q).

T03 served from a precomputed top-200 pool, so its per-query retrieval latency
was not recorded (shown as "--"); T04 and T05 logged real query-time latency and
are mutually comparable. The point: T04 matches T03's recall with zero query-time
LLM calls at sub-second latency, while T05 spends ~6 LLM calls and ~38 s/query
for less than half the recall.

Usage:
  .venv/Scripts/python Report/scripts/build_cost_table.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPORT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import load_arm2c as la
import load_results as lr
from thesis_style import fmt_metric, get_variant, save_table

OUTPUT_NAME = "tab_rq2_cost"
ROWS = ["T03_arm1_naive", "T04_summary_node_hybrid", "T05_pageindex", "ARM2C_agentic"]
# Display override now that the agentic arm has two configs in the table.
DISPLAY = {"ARM2C_agentic": "Arm 2C (simple nav)"}


def _fmt_lat_s(ms: float) -> str:
    if ms != ms:  # NaN
        return "--"
    return f"{ms / 1000:.1f}"


def _fmt_int(x: float) -> str:
    if x != x:
        return "--"
    n = int(round(x))
    s = f"{n:,}".replace(",", r"\,")    # LaTeX thin-space thousands separator
    return s


def main() -> None:
    long = lr.load_long(include_arm2c=True)
    er = lr.to_wide(long, ["E/R@10"], weighting="micro").set_index("method")["E/R@10"]
    cost = lr.costs_by_method(include_arm2c=True).set_index("method")

    col_spec = "l S[table-format=1.1] r S[table-format=2.1] S[table-format=2.1] S[table-format=1.3]"
    out = [r"\begin{tabular}{" + col_spec + "}", r"\toprule",
           r"Arm & {LLM calls/q} & {tokens/q} & {lat.\ mean (s)} "
           r"& {lat.\ p95 (s)} & {E/R@10} \\", r"\midrule"]
    def _row(display: str, c: dict, er_val: float) -> str:
        cells = [
            display,
            f"{c['llm_calls_per_q']:.1f}",
            _fmt_int(c["tokens_per_q"]),
            f"{{{_fmt_lat_s(c['latency_ms_mean'])}}}",
            f"{{{_fmt_lat_s(c['latency_ms_p95'])}}}",
            fmt_metric("E/R@10", er_val),
        ]
        return " & ".join(cells) + r" \\"

    for m in ROWS:
        display = DISPLAY.get(m) or get_variant(m).display
        out.append(_row(display, cost.loc[m], float(er.loc[m])))
        # CNR row directly under the simple-nav agentic row.
        if m == "ARM2C_agentic":
            cnr_cost = la.arm2c_cnr_cost_row()
            cnr_er = la.arm2c_cnr_metrics()["E/R@10"]
            out.append(_row("Arm 2C (CNR)", cnr_cost, cnr_er))
    out += [r"\bottomrule", r"\end{tabular}"]
    out_path = save_table("\n".join(out) + "\n", OUTPUT_NAME)
    print(f"[done] Wrote {out_path.relative_to(REPORT_DIR.parent)}")


if __name__ == "__main__":
    main()
