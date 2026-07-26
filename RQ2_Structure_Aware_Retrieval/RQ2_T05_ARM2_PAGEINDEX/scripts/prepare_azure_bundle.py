"""Prepare an Azure-side bundle for T05.

Joins a curated GT file (built by ``evaluation.question_subsets``) with
question texts from ``bsard_corpus.db``, packages the per-doc trees, and
emits a single ZIP ready to upload to Azure Blob.

Bundle contents (within the zip):

    trees/doc_<doc_id>.json    one per built document
    queries.json               [{"query_id": str, "query_text": str}, ...]
    gt.json                    copy of the input GT file
    manifest.json              run metadata + sha256s

Usage (one bundle per PDF — matches the per-PDF Azure runs):
    python scripts/prepare_azure_bundle.py \\
        --gt ../RQ2_T07_EVALUATION/ground_truth/runs/t04_1804_03_21_1804032150.json \\
        --doc-id 1804_03_21_1804032150 \\
        --data-root data \\
        --out data/azure_bundles/t04_1804_03_21_1804032150

Usage (multi-PDF bundle — every tree under data/):
    python scripts/prepare_azure_bundle.py \\
        --gt ../RQ2_T07_EVALUATION/ground_truth/runs/headline_exact_test.json \\
        --data-root data \\
        --out data/azure_bundles/headline_exact_test
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import sqlite3
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_question_texts(
    db_path: Path, question_ids: list[int],
) -> dict[int, str]:
    if not question_ids:
        return {}
    conn = sqlite3.connect(str(db_path))
    try:
        # Chunk in batches of 500 to stay under SQLite's parameter limit.
        out: dict[int, str] = {}
        for i in range(0, len(question_ids), 500):
            batch = question_ids[i : i + 500]
            placeholders = ",".join("?" * len(batch))
            rows = conn.execute(
                f"SELECT question_id, question_text FROM questions "
                f"WHERE question_id IN ({placeholders})",
                batch,
            ).fetchall()
            for qid, text in rows:
                out[int(qid)] = str(text)
        return out
    finally:
        conn.close()


def main() -> int:
    here = Path(__file__).resolve().parent
    project_root = here.parent

    parser = argparse.ArgumentParser(prog="prepare_azure_bundle")
    parser.add_argument(
        "--gt", type=Path, required=True,
        help="GT JSON file ({question_id_str: [bsard_id, ...]}).",
    )
    parser.add_argument(
        "--data-root", type=Path,
        default=project_root / "data",
        help="T05 data root; trees are read from <data_root>/<doc_id>/tree.json.",
    )
    parser.add_argument(
        "--bsard-db", type=Path,
        # BSARD corpus DB. Resolution order (RQ2 convention):
        #   $RQ2_BSARD_DB  →  <RQ2_DATA_DIR or repo>/data/bsard_corpus.db
        # Pull it with scripts/download_data.py, or point at the corpus project.
        default=Path(os.environ["RQ2_BSARD_DB"]).expanduser()
        if os.environ.get("RQ2_BSARD_DB")
        else project_root.parent / "data" / "bsard_corpus.db",
    )
    parser.add_argument(
        "--out", type=Path, required=True,
        help="Output directory (will be created); zipped to <out>.zip unless --no-zip.",
    )
    parser.add_argument(
        "--run-name", type=str, default=None,
        help="Logical run name. Defaults to the GT filename stem.",
    )
    parser.add_argument(
        "--doc-id", type=str, default=None,
        help="Restrict the bundle to a single PDF (BSARD stem, e.g. "
             "1804_03_21_1804032150). Default: bundle every <data_root>/<doc_id>/tree.json.",
    )
    parser.add_argument(
        "--no-zip", action="store_true",
        help="Skip writing the .zip — only produce the directory.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s - %(message)s",
    )
    log = logging.getLogger("prepare_azure_bundle")

    # 1. Load GT, extract question_ids.
    if not args.gt.exists():
        log.error("GT file not found: %s", args.gt)
        return 2
    gt = json.loads(args.gt.read_text(encoding="utf-8"))
    if not isinstance(gt, dict) or not gt:
        log.error("GT file %s is not a non-empty {qid: [bsard_ids]} dict", args.gt)
        return 2
    question_ids = sorted(int(qid) for qid in gt.keys())
    log.info("GT has %d questions", len(question_ids))

    # 2. Resolve question texts.
    if not args.bsard_db.exists():
        log.error("bsard_db not found: %s", args.bsard_db)
        return 2
    text_map = _load_question_texts(args.bsard_db, question_ids)
    missing = [qid for qid in question_ids if qid not in text_map]
    if missing:
        log.warning(
            "%d question_ids missing question_text in DB (sample: %s)",
            len(missing), missing[:5],
        )
    queries = [
        {"query_id": str(qid), "query_text": text_map[qid]}
        for qid in question_ids if qid in text_map
    ]
    log.info(
        "Resolved %d/%d question texts", len(queries), len(question_ids),
    )

    # 3. Collect trees from <data_root>/<doc_id>/tree.json.
    if not args.data_root.exists():
        log.error(
            "Data root missing: %s. Run scripts/build_tree.py first.",
            args.data_root,
        )
        return 2
    tree_files = sorted(args.data_root.glob("*/tree.json"))
    if args.doc_id is not None:
        tree_files = [tf for tf in tree_files if tf.parent.name == args.doc_id]
        if not tree_files:
            log.error(
                "No tree at %s/%s/tree.json. Build it first with "
                "scripts/build_tree.py --doc-id %s.",
                args.data_root, args.doc_id, args.doc_id,
            )
            return 2
    if not tree_files:
        log.error("No <doc_id>/tree.json trees in %s", args.data_root)
        return 2
    log.info("Found %d tree files", len(tree_files))

    # 4. Assemble bundle dir. Inside the zip, trees keep the legacy
    # `trees/doc_<doc_id>.json` naming so the cloud-side reader doesn't
    # break (PLAN_per_pdf_pipeline §B.6) — but <doc_id> is now the BSARD stem.
    args.out.mkdir(parents=True, exist_ok=True)
    bundle_trees = args.out / "trees"
    bundle_trees.mkdir(exist_ok=True)
    for tf in tree_files:
        doc_id = tf.parent.name
        shutil.copy2(tf, bundle_trees / f"doc_{doc_id}.json")

    (args.out / "queries.json").write_text(
        json.dumps(queries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    shutil.copy2(args.gt, args.out / "gt.json")

    run_name = args.run_name or args.gt.stem
    manifest = {
        "run_name": run_name,
        "n_questions_in_gt": len(question_ids),
        "n_questions_resolved": len(queries),
        "n_questions_missing_text": len(missing),
        "missing_question_ids_sample": missing[:20],
        "trees": [
            {
                "file": f"doc_{tf.parent.name}.json",
                "doc_id": tf.parent.name,
                "sha256": _sha256(tf),
                "bytes": tf.stat().st_size,
            }
            for tf in tree_files
        ],
        "gt_file": str(args.gt),
        "gt_sha256": _sha256(args.gt),
        "bsard_db_sha256": _sha256(args.bsard_db),
        "saved_at": _now_utc(),
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 5. Zip.
    zip_path = args.out.with_suffix(".zip")
    if not args.no_zip:
        if zip_path.exists():
            zip_path.unlink()
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for path in args.out.rglob("*"):
                if path.is_file():
                    zf.write(path, path.relative_to(args.out))
        log.info(
            "Wrote %s (%.1f KB)", zip_path, zip_path.stat().st_size / 1024,
        )

    print()
    print(f"Bundle directory: {args.out}")
    if not args.no_zip:
        print(f"Bundle zip:       {zip_path}")
    print(f"Run name:         {run_name}")
    print(
        f"Questions:        {len(queries)} resolved / {len(question_ids)} in GT"
    )
    print(f"Trees:            {len(tree_files)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
