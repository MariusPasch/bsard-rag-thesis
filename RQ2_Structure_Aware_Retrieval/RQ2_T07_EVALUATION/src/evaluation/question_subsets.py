"""Build curated ground-truth files from the BSARD question extraction-status JSONL.

The JSONL at `ground_truth/question_extraction_status/questions_by_extraction_status.jsonl`
categorises each of the 1,108 BSARD questions by the PDF-extraction status of
its ground-truth articles. This module turns filter specs into per-run GT
files that the evaluator can load directly.

Typical workflow:
  1. `build` a curated GT file (one or more filter passes).
  2. Hand-edit it if needed (drop questions, restrict per-question GT).
  3. Point the evaluator at the file via config[ground_truth_file].

CLI:
  python -m evaluation.question_subsets build \\
      --status exact partial \\
      --split test \\
      --restrict-per-question-to FOUND \\
      --output ground_truth/runs/exact_partial_test.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Source-of-truth JSONL location relative to this file.
_DEFAULT_JSONL = (
    Path(__file__).resolve().parents[2]
    / "ground_truth"
    / "question_extraction_status"
    / "questions_by_extraction_status.jsonl"
)

EXTRACTION_STATUSES = ("exact", "partial", "not_present", "mixed")
VERIFICATION_STATUSES = ("FOUND", "PARTIAL", "NOT FOUND")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_extraction_status(path: str | Path | None = None) -> pd.DataFrame:
    """Load the question extraction-status JSONL into a DataFrame.

    One row per question; columns:
      question_id (int), split (str), extraction_status (str),
      n_relevant_bsard_articles (int), pdf_filenames (list[str]),
      bsard_articles (list[dict]).
    """
    p = Path(path) if path else _DEFAULT_JSONL
    if not p.exists():
        raise FileNotFoundError(
            f"Extraction-status JSONL not found at {p}. "
            "Re-run analysis/question_extraction_analysis.ipynb to regenerate."
        )
    return pd.read_json(p, lines=True)


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def select_question_ids(
    df: pd.DataFrame,
    *,
    extraction_status: list[str] | None = None,
    split: str | None = None,
    n_relevant_min: int | None = None,
    n_relevant_max: int | None = None,
    pdf_filename: str | None = None,
    include_question_ids: set[int] | None = None,
    exclude_question_ids: set[int] | None = None,
) -> set[int]:
    """Filter the question DataFrame and return the matching question_ids.

    All filters are AND-combined. Pass None to skip a filter.

    Args:
        extraction_status:        Subset of EXTRACTION_STATUSES.
        split:                    'train' or 'test'.
        n_relevant_min/max:       Inclusive bounds on n_relevant_bsard_articles.
        pdf_filename:             Substring; matches any of the question's PDFs.
        include_question_ids:     If set, intersect with this allowlist.
        exclude_question_ids:     If set, drop any question in this denylist.
    """
    out = df
    if extraction_status:
        unknown = set(extraction_status) - set(EXTRACTION_STATUSES)
        if unknown:
            raise ValueError(f"Unknown extraction_status values: {unknown}")
        out = out[out["extraction_status"].isin(extraction_status)]
    if split is not None:
        out = out[out["split"] == split]
    if n_relevant_min is not None:
        out = out[out["n_relevant_bsard_articles"] >= n_relevant_min]
    if n_relevant_max is not None:
        out = out[out["n_relevant_bsard_articles"] <= n_relevant_max]
    if pdf_filename is not None:
        out = out[out["pdf_filenames"].apply(
            lambda lst: any(pdf_filename in p for p in lst)
        )]

    qids = {int(q) for q in out["question_id"]}
    if include_question_ids is not None:
        qids &= {int(q) for q in include_question_ids}
    if exclude_question_ids is not None:
        qids -= {int(q) for q in exclude_question_ids}
    return qids


# ---------------------------------------------------------------------------
# GT construction
# ---------------------------------------------------------------------------

def build_ground_truth(
    df: pd.DataFrame,
    question_ids: set[int],
    *,
    restrict_per_question_to_status: list[str] | None = None,
) -> dict[str, list[int]]:
    """Assemble {query_id_str: [bsard_id, ...]} for the selected questions.

    Args:
        df:                              Output of load_extraction_status().
        question_ids:                    Selected question_ids (e.g. from select_question_ids).
        restrict_per_question_to_status: If set, e.g. ['FOUND'], keep only
                                         bsard_articles with verification_status
                                         in this allowlist. Useful for mixed-bucket
                                         questions where you want the article-level
                                         GT trimmed to extractable ones.
    """
    if restrict_per_question_to_status:
        unknown = set(restrict_per_question_to_status) - set(VERIFICATION_STATUSES)
        if unknown:
            raise ValueError(f"Unknown verification_status values: {unknown}")
        allowed = set(restrict_per_question_to_status)
    else:
        allowed = None

    selected = df[df["question_id"].isin(question_ids)]

    gt: dict[str, list[int]] = {}
    for _, row in selected.iterrows():
        articles = row["bsard_articles"]
        if allowed is not None:
            bsard_ids = [int(a["bsard_id"]) for a in articles
                         if a["verification_status"] in allowed]
        else:
            bsard_ids = [int(a["bsard_id"]) for a in articles]
        if bsard_ids:
            gt[str(int(row["question_id"]))] = sorted(set(bsard_ids))
        # Questions whose GT becomes empty under the restriction are dropped:
        # an empty GT yields recall=0 unconditionally and would skew aggregates.

    return gt


def build_ground_truth_with_cosines(
    df: pd.DataFrame,
    question_ids: set[int],
    *,
    restrict_per_question_to_status: list[str] | None = None,
) -> dict[str, dict[int, float | None]]:
    """Same selection as build_ground_truth, but preserves per-article cosines.

    Returns {query_id_str: {bsard_id: extraction_cosine | None}} for the selected
    questions. The cosine field is read from `bsard_articles[*].extraction_cosine`
    in the upstream JSONL (populated by analysis/question_extraction_analysis.ipynb
    by merging T03's per-PDF question_relevance.json under CN-T03-002). Rows where
    the field is missing or null come through as None, so the accessor degrades
    gracefully on un-regenerated upstream data.
    """
    if restrict_per_question_to_status:
        unknown = set(restrict_per_question_to_status) - set(VERIFICATION_STATUSES)
        if unknown:
            raise ValueError(f"Unknown verification_status values: {unknown}")
        allowed = set(restrict_per_question_to_status)
    else:
        allowed = None

    selected = df[df["question_id"].isin(question_ids)]

    gt: dict[str, dict[int, float | None]] = {}
    for _, row in selected.iterrows():
        articles = row["bsard_articles"]
        per_q: dict[int, float | None] = {}
        for a in articles:
            if allowed is not None and a["verification_status"] not in allowed:
                continue
            bid = int(a["bsard_id"])
            cos = a.get("extraction_cosine")
            per_q[bid] = float(cos) if cos is not None else None
        if per_q:
            gt[str(int(row["question_id"]))] = per_q

    return gt


def write_ground_truth(gt: dict[str, list[int]], path: str | Path) -> None:
    """Write {query_id_str: [bsard_id, ...]} to a JSON file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(gt, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote curated ground truth to %s (%d queries)", p, len(gt))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cmd_build(args: argparse.Namespace) -> int:
    df = load_extraction_status(args.jsonl)

    include_qids: set[int] | None = None
    if args.from_question_ids:
        with open(args.from_question_ids, encoding="utf-8") as f:
            include_qids = {int(line.strip()) for line in f if line.strip()}

    qids = select_question_ids(
        df,
        extraction_status=args.status,
        split=args.split,
        n_relevant_min=args.n_min,
        n_relevant_max=args.n_max,
        pdf_filename=args.pdf,
        include_question_ids=include_qids,
    )

    gt = build_ground_truth(
        df, qids,
        restrict_per_question_to_status=args.restrict_per_question_to,
    )
    write_ground_truth(gt, args.output)

    print(f"Selected {len(qids)} questions; wrote {len(gt)} GT entries to {args.output}")
    if len(gt) < len(qids):
        print(f"  ({len(qids) - len(gt)} questions dropped — empty GT under restriction)")
    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    df = load_extraction_status(args.jsonl)
    by_split = df.groupby(["split", "extraction_status"]).size().unstack(fill_value=0)
    for col in EXTRACTION_STATUSES:
        if col not in by_split.columns:
            by_split[col] = 0
    by_split = by_split[list(EXTRACTION_STATUSES)]
    by_split["total"] = by_split.sum(axis=1)
    print(by_split)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m evaluation.question_subsets",
        description="Curate per-run ground-truth files from the BSARD question extraction-status JSONL.",
    )
    parser.add_argument(
        "--jsonl", default=None,
        help=f"Path to questions_by_extraction_status.jsonl (default: {_DEFAULT_JSONL})",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build", help="Build a curated GT file.")
    p_build.add_argument("--status", nargs="+", choices=EXTRACTION_STATUSES,
                         help="Filter by extraction_status (multiple allowed).")
    p_build.add_argument("--split", choices=("train", "test"), help="Restrict to one split.")
    p_build.add_argument("--n-min", type=int, help="Minimum n_relevant_bsard_articles.")
    p_build.add_argument("--n-max", type=int, help="Maximum n_relevant_bsard_articles.")
    p_build.add_argument("--pdf", help="PDF filename substring filter.")
    p_build.add_argument("--from-question-ids", help="Path to a file with one question_id per line.")
    p_build.add_argument("--restrict-per-question-to", nargs="+", choices=VERIFICATION_STATUSES,
                         help="Keep only bsard_articles with these verification_statuses inside each question's GT.")
    p_build.add_argument("--output", "-o", required=True, help="Output GT JSON path.")
    p_build.set_defaults(func=_cmd_build)

    p_inspect = sub.add_parser("inspect", help="Print bucket counts.")
    p_inspect.set_defaults(func=_cmd_inspect)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
