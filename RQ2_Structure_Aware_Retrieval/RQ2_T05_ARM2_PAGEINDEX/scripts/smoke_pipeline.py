"""End-to-end smoke for run_subset + retriever.

Runs 3 sample queries against the cached doc-2 tree using the stub LLM,
saves per-query JSON + manifest to a temp dir, reloads them, and
validates:

  * Each ranked_item has a non-null integer ``metadata['bsard_id']``
    (CN-T05-001).
  * Re-running with the same out_dir is idempotent (cache hit logged).
  * Manifest aggregates totals correctly.

Usage:
    python scripts/smoke_pipeline.py
    python scripts/smoke_pipeline.py --out /tmp/t05_pipeline
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

from arm2_pageindex.pipeline import load_subset_results, run_subset
from arm2_pageindex.tree_builder import load_law_tree

# Re-use the StubLLMClient from the navigator smoke.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from smoke_navigator import StubLLMClient  # noqa: E402


SAMPLE_QUERIES = [
    {
        "query_id": "11",
        "query_text": (
            "Quels sont les principes fondamentaux du droit de "
            "l'environnement en Région wallonne ?"
        ),
    },
    {
        "query_id": "22",
        "query_text": (
            "Comment se déroule la procédure de consultation publique "
            "pour un projet à incidence environnementale ?"
        ),
    },
    {
        "query_id": "33",
        "query_text": (
            "Quelles sont les obligations de l'exploitant en cas de "
            "dommage environnemental ?"
        ),
    },
]


def main() -> int:
    here = Path(__file__).resolve().parent
    project_root = here.parent

    parser = argparse.ArgumentParser(prog="smoke_pipeline")
    parser.add_argument(
        "--tree", type=Path,
        default=project_root / "data" / "2004_05_27_2004A27101" / "tree.json",
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path("C:/tmp/t05_pipeline_smoke"),
    )
    parser.add_argument("--clean", action="store_true",
                        help="Wipe out_dir before running (tests cold path).")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s - %(message)s",
    )
    log = logging.getLogger("smoke_pipeline")

    if args.clean and args.out.exists():
        log.info("Wiping %s", args.out)
        shutil.rmtree(args.out)

    if not args.tree.exists():
        log.error("Tree missing: %s. Run scripts/smoke_tree.py first.", args.tree)
        return 2

    tree, manifest = load_law_tree(args.tree)
    log.info("Loaded tree doc_id=%s articles=%s",
             manifest.get("document_id"), manifest.get("n_articles"))

    llm = StubLLMClient()

    print("\n=== first run (cold) ===")
    results1 = run_subset(
        tree=tree, queries=SAMPLE_QUERIES, llm=llm,
        out_dir=args.out, progress=False, log_every=1,
    )

    print("\n=== second run (idempotent reload) ===")
    llm2 = StubLLMClient()
    results2 = run_subset(
        tree=tree, queries=SAMPLE_QUERIES, llm=llm2,
        out_dir=args.out, progress=False, log_every=1,
    )
    assert llm2.calls == 0, (
        f"second run should be 100% cache hits, but stub fired {llm2.calls} times"
    )

    print("\n=== load_subset_results ===")
    reloaded = load_subset_results(args.out)
    print(f"Reloaded {len(reloaded)} results from {args.out}")

    # CN-T05-001 contract check
    cn_violations = 0
    for r in reloaded:
        for item in r.ranked_items:
            bid = (item.get("metadata") or {}).get("bsard_id")
            if bid is None or not isinstance(bid, int):
                cn_violations += 1
    if cn_violations:
        log.error(
            "CN-T05-001 violation: %d ranked_items missing/invalid bsard_id",
            cn_violations,
        )
    else:
        print("CN-T05-001 contract: OK (every ranked_item has metadata.bsard_id:int)")

    print(f"\nFiles in {args.out}:")
    for p in sorted(args.out.iterdir()):
        print(f"  {p.name:>20}  {p.stat().st_size:>6} B")

    manifest_p = args.out / "manifest.json"
    if manifest_p.exists():
        m = json.loads(manifest_p.read_text(encoding="utf-8"))
        print("\nManifest summary:")
        for k in ("method", "n_queries", "n_cached_skip", "tree_root_id",
                  "elapsed_s", "exit_reasons"):
            print(f"  {k}: {m.get(k)}")
        print(f"  totals: {m.get('totals')}")

    return 0 if cn_violations == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
