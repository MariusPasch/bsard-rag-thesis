"""Stamp the most recent successful T07 evaluation pass for one PDF.

Writes ``<T07>/data/<doc_id>/eval/last_run.json`` (PLAN_per_pdf_pipeline §B.8).
The stamp is independent of the weights/projection caches; it lets the
orchestrator status command tell whether T07 has actually evaluated this
PDF without inferring it from the more granular per-config caches.

CLI:

    python -m evaluation.eval_stamp --doc-id 1804_03_21_1804032150
    python -m evaluation.eval_stamp --doc-id 1804_03_21_1804032150 \\
        --gt-run-label headline_exact_test --n-methods 6
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1


def _resolve_t00_paths():
    # eval_stamp.py → evaluation/ → src/ → RQ2_T07_EVALUATION/ → RQ2_ROOT/
    rq2_root = Path(__file__).resolve().parents[3]
    t00_src = rq2_root / "RQ2_T00_ORCHESTRATOR" / "src"
    if str(t00_src) not in sys.path:
        sys.path.insert(0, str(t00_src))
    from orchestrator import paths
    return paths


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_stamp(
    doc_id: str,
    *,
    out_path: Path,
    gt_run_label: str | None = None,
    n_methods_evaluated: int | None = None,
) -> dict:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "doc_id": doc_id,
        "last_eval_at": _now_utc(),
        "gt_run_label": gt_run_label,
        "n_methods_evaluated": n_methods_evaluated,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info("Wrote %s", out_path)
    return payload


def main() -> int:
    paths = _resolve_t00_paths()

    parser = argparse.ArgumentParser(prog="evaluation.eval_stamp",
                                     description=__doc__.split("\n", 1)[0])
    parser.add_argument("--doc-id", required=True, help="PDF stem (e.g. 1804_03_21_1804032150)")
    parser.add_argument("--gt-run-label", default=None,
                        help="Free-form label of the GT subset used (e.g. 'headline_exact_test')")
    parser.add_argument("--n-methods", type=int, default=None,
                        help="Number of retrieval methods evaluated in this pass")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    out_path = paths.project_data_dir("T07", args.doc_id) / "eval" / "last_run.json"
    write_stamp(
        args.doc_id,
        out_path=out_path,
        gt_run_label=args.gt_run_label,
        n_methods_evaluated=args.n_methods,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
