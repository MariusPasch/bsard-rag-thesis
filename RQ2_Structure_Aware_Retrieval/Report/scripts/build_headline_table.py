"""Build the RQ2 cross-arm headline table.

Output:  Report/tables/tab_rq2_headline.tex  (tabular environment only -- the
         caption is written by the report's LaTeX source around it.)

Columns: Arm | E/R@10 | R@10 | Hit@10 | E/R@100 | R@100 | MRR@10 | nDCG@10
         | p vs Arm 1

- Three canonical arm rows (arm1_naive, arm2a_summary_node, arm2b_pageindex),
  ordered by the headline metric E/R@10 descending.
- Micro average over the curated 5-PDF set (725 questions): binary metrics
  weighted by n_gt_q, E/ metrics by n_evaluable (load_results.to_wide).
- Best value per metric column rendered \\textbf{...}.
- "$\\dagger$" appended to the R@10 cell where the paired t-test vs Arm 1 is
  significant (the significance protocol runs on binary Recall@k, STYLEGUIDE
  §13.3); the p column carries that p-value.
- siunitx S columns; the anchor row's p column is "--".

Caption must state: Effective GT (E/) primary; micro average over 5 PDFs
(725 q); pool depths differ (Arm 1 = 200, Arm 2 = 100) so R@100 favours Arm 1.

Usage:
  .venv/Scripts/python Report/scripts/build_headline_table.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPORT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import load_arm2c as la
import load_results as lr
from thesis_style import (
    SIGNIFICANCE_MARKER,
    fmt_metric,
    fmt_pvalue,
    get_variant,
    save_table,
)

OUTPUT_NAME = "tab_rq2_headline"
ANCHOR = "T03_arm1_naive"
ROWS = ["T03_arm1_naive", "T04_summary_node_hybrid", "ARM2C_agentic", "T05_pageindex"]
METRICS = ["E/R@10", "R@10", "Hit@10", "E/R@100", "R@100", "MRR@10", "nDCG@10"]
# Metrics where a higher value is better (all of the above).
PRIMARY_SIG_METRIC = "R@10"   # the metric the paired test is computed on


def _metric_header(m: str) -> str:
    return m.replace("E/R@", "E/R@").replace("nDCG", "nDCG")


def _render(rows: list[dict], best: dict[str, float]) -> str:
    n = len(METRICS)
    col_spec = "l " + " ".join(["S[table-format=1.3]"] * n) + " r"
    header = " & ".join(f"{{{_metric_header(m)}}}" for m in METRICS)

    out = [r"\begin{tabular}{" + col_spec + "}", r"\toprule",
           rf"Arm & {header} & {{$p$ vs Arm 1}} \\", r"\midrule"]
    for r in rows:
        cells = [r["display"]]
        for m in METRICS:
            val = r.get(m)
            text = fmt_metric(m, val)
            if val is not None and best.get(m) is not None and abs(val - best[m]) < 1e-9:
                text = rf"\textbf{{{text}}}"
            if m == PRIMARY_SIG_METRIC and r.get("sig"):
                text = text + SIGNIFICANCE_MARKER
            if "\\" in text:
                text = "{" + text + "}"
            cells.append(text)
        cells.append("--" if r["is_anchor"] else fmt_pvalue(r.get("p")))
        out.append(" & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


def main() -> None:
    long = lr.load_long(include_arm2c=True)
    sig = lr.load_significance()
    wide = lr.to_wide(long, METRICS, weighting="micro").set_index("method")

    rows: list[dict] = []
    for method in ROWS:
        if method not in wide.index:
            print(f"[warn] {method} absent from long frame; skipped")
            continue
        w = wide.loc[method]
        p = lr.p_vs_anchor(sig, method, ANCHOR, 10)
        # disambiguate the agentic arm now that CNR shares the table
        display = ("Arm 2C (simple nav)" if method == "ARM2C_agentic"
                   else get_variant(method).display)
        rows.append({
            "display": display,
            "is_anchor": method == ANCHOR,
            "p": p,
            "sig": p is not None and p < 0.05,
            **{m: float(w[m]) for m in METRICS},
        })

    # Arm-2C CNR row (e5 + corrective): full battery now available -- MRR@10/nDCG@10
    # recovered by re-running the e5 rerank over the saved pools and capturing the
    # ordered list (corrective/e2_metrics_full.py).
    cnr = la.arm2c_cnr_metrics()
    rows.append({
        "display": "Arm 2C (CNR)",
        "is_anchor": False, "p": None, "sig": False,
        **{m: float(cnr.get(m, float("nan"))) for m in METRICS},
    })

    rows.sort(key=lambda r: r["E/R@10"], reverse=True)

    best = {m: max(r[m] for r in rows if r[m] == r[m]) for m in METRICS}
    tex = _render(rows, best)
    out_path = save_table(tex, OUTPUT_NAME)
    print(f"[done] Wrote {out_path.relative_to(REPORT_DIR.parent)}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
