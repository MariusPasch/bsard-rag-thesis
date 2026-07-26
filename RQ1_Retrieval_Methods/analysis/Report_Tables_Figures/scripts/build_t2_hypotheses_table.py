"""
Build the Tier 2 pre-registered hypothesis verdict table.

Output: Report/tables/tab_rq1_t2_hypotheses.tex

What this is.
  Tier 2 was the only RQ1 tier with eight pre-registered hypotheses (H1-H8 in
  TIER2_DENSE_RETRIEVAL_PLAN.md section 15). They are listed here with their
  verdicts so the report acknowledges them transparently rather than reporting
  only the confirmed ones.

Content source.
  Hypothesis claims, verdicts, and evidence are transcribed from TIER2 sec 15
  and lightly compressed for the table. Numeric evidence is sourced from the
  same Tier 2 result JSONs that feed tab_rq1_t2_dense.

This builder emits LaTeX text only -- it does not load result JSONs. Update
the HYPOTHESES list below if the plan file's evidence is refreshed.

Usage:
  .venv/Scripts/python Report/scripts/build_t2_hypotheses_table.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPORT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPORT_DIR))

from thesis_style import save_table


# (label, claim, verdict, evidence) -- transcribed from TIER2 section 15.
HYPOTHESES: list[tuple[str, str, str, str]] = [
    (
        "H1",
        "Only fine-tuned dense models will close the semantic-paraphrase gap; "
        "zero-shot dense will be near-random.",
        "Refuted",
        "Zero-shot mE5-large, BGE-M3, and Qwen3-instruct each roughly double "
        "the best sparse R@10 on the paraphrase stratum (0.14 vs 0.07). "
        "Zero-shot CamemBERT-base does collapse (R@100 = 1.69\\%) -- that half holds.",
    ),
    (
        "H2",
        "French-specific \\texttt{sentence-camembert-large} may underperform "
        "multilingual mE5-large despite domain specificity.",
        "Confirmed",
        "mE5-large R@100 = 0.594 $>$ CamemBERT-lg R@100 = 0.583. Multilingual "
        "retrieval pretraining wins over French-only LM pretraining.",
    ),
    (
        "H3",
        "mE5-large may not consistently outperform mE5-base on statutory text.",
        "Refuted",
        "mE5-large R@100 = 0.594 vs mE5-base R@100 = 0.491 -- a $\\sim$10 pp gap. "
        "Scale matters here.",
    ),
    (
        "H4",
        "Paper checkpoint (if released) would significantly outperform all "
        "zero-shot models.",
        "Not tested",
        "Authors' fine-tuned checkpoint (EXP-D5) was never released.",
    ),
    (
        "H5",
        "Dense will outperform BM25 on semantically paraphrased queries "
        "while underperforming on lexically aligned queries.",
        "Mixed",
        "Paraphrased: dense $>$ sparse confirmed. Lexically aligned: dense "
        "\\emph{also} wins (Qwen3-instruct R@10 = 0.590 vs best sparse $\\sim$0.52) -- "
        "the second half is refuted.",
    ),
    (
        "H6",
        "BGE-M3 will outperform mE5 on long articles due to its 1024-token "
        "window; comparable on short articles.",
        "Mixed",
        "Overall: BGE-M3 R@100 = 0.592 essentially tied with mE5-large 0.594 "
        "despite less truncation. BGE-M3 \\emph{does} lead on R@500 -- the "
        "long-context advantage shows up only in deeper recall.",
    ),
    (
        "H7",
        "\\texttt{concat\\_2x} field weighting will give a modest "
        "($\\sim$0.5--1 pp) R@100 improvement on the winning encoder.",
        "Confirmed",
        "mE5-large concat-2x R@100 = 0.622 vs text-only 0.594 ($+$2.7 pp, "
        "larger than predicted; $p = 0.0003$ vs sparse anchor).",
    ),
    (
        "H8",
        "Qwen3-0.6B (decoder-only, last-token pooling) will outperform same-scale "
        "encoder-only bi-encoders; D10i $>$ D10.",
        "Confirmed",
        "Qwen3-plain R@100 = 0.582, Qwen3-instruct R@100 = 0.597 -- both above "
        "mE5-base (0.491) at similar scale. Instruct $>$ plain ($+$1.5 pp).",
    ),
]


def _escape(s: str) -> str:
    """Pass-through -- the strings above use raw LaTeX deliberately."""
    return s


def _render() -> str:
    out: list[str] = []
    out.append(r"\begin{tabularx}{\linewidth}{@{}l X l X@{}}")
    out.append(r"\toprule")
    out.append(r"H\# & {Claim} & {Verdict} & {Evidence} \\")
    out.append(r"\midrule")
    for label, claim, verdict, evidence in HYPOTHESES:
        # Emphasise the verdict so the column scans top-to-bottom.
        if verdict.lower() == "confirmed":
            verdict_tex = r"\textbf{Confirmed}"
        elif verdict.lower() == "refuted":
            verdict_tex = r"\textit{Refuted}"
        elif verdict.lower() == "mixed":
            verdict_tex = r"Mixed"
        elif verdict.lower() == "not tested":
            verdict_tex = r"\textit{Not tested}"
        else:
            verdict_tex = verdict
        out.append(f"{label} & {_escape(claim)} & {verdict_tex} & {_escape(evidence)} \\\\")
    out.append(r"\bottomrule")
    out.append(r"\end{tabularx}")
    return "\n".join(out) + "\n"


def main() -> None:
    tex = _render()
    out_path = save_table(tex, "tab_rq1_t2_hypotheses")
    print(f"[done] Wrote {out_path.relative_to(REPORT_DIR.parent)}")
    print(f"[done] {len(HYPOTHESES)} hypotheses.")
    # Quick verdict tally.
    from collections import Counter
    counts = Counter(h[2] for h in HYPOTHESES)
    print("        verdicts:", dict(counts))


if __name__ == "__main__":
    main()
