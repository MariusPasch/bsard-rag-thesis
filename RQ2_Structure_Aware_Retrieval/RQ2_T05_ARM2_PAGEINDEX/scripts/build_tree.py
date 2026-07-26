"""Build per-PDF ToC trees and write them under ``data/<doc_id>/tree.json``.

Loads the AzureDI dump (via T04), runs the BSARD linker, aggregates
articles, builds a per-PDF tree, prints stats, and saves
``data/<doc_id>/tree.json`` (PLAN_per_pdf_pipeline §B.6) with the manifest
header pinning ``pdf_sha256``, ``tree_builder_version``, ``doc_id`` (BSARD
PDF stem), and ``azure_document_id`` (AzureDI integer).

Idempotent re-runs: a tree is rebuilt only when the cached manifest's
``pdf_sha256`` no longer matches the PDF on disk, or
``tree_builder_version`` has bumped, or ``--force-rebuild`` is passed.

Usage (from project root, with venv active and sibling packages installed):

    python scripts/build_tree.py                                   # all docs, cached
    python scripts/build_tree.py --doc-id 2004_05_27_2004A27101    # single PDF
    python scripts/build_tree.py --force-rebuild                   # ignore cache
    python scripts/build_tree.py --out /tmp/trees                  # alt output root
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

from arm2_metadata.azuredi_loader import load_azuredi_corpus
from arm2_metadata.bsard_link import link_corpus_to_bsard
from arm2_metadata.enricher import build_articles_from_nodes, build_definitions_map

from arm2_pageindex.tree_builder import (
    build_law_tree,
    is_cache_fresh,
    load_law_tree,
    save_law_tree,
    tree_stats,
)


def main() -> int:
    here = Path(__file__).resolve().parent
    project_root = here.parent
    rq2_root = project_root.parent

    parser = argparse.ArgumentParser(prog="build_tree")
    parser.add_argument(
        "--azuredi", type=Path,
        default=rq2_root / "RQ2_T04_ARM2_METADATA" / "azuredi",
        help="Path to azuredi/ directory (MyDocuments.csv + VectorDB*.json).",
    )
    parser.add_argument(
        "--pdf-map", type=Path,
        default=(
            rq2_root / "RQ2_T02_DATA_LOADER" / "data" / "csv"
            / "pdf_document_map.csv"
        ),
    )
    parser.add_argument(
        "--bsard-db", type=Path,
        # BSARD corpus DB. Resolution order (RQ2 convention):
        #   $RQ2_BSARD_DB  →  <RQ2_DATA_DIR or repo>/data/bsard_corpus.db
        # Pull it with scripts/download_data.py, or point at the corpus project.
        default=Path(os.environ["RQ2_BSARD_DB"]).expanduser()
        if os.environ.get("RQ2_BSARD_DB")
        else rq2_root / "data" / "bsard_corpus.db",
    )
    parser.add_argument(
        "--pdf-dir", type=Path,
        default=rq2_root / "RQ2_T02_DATA_LOADER" / "data" / "pdfs",
    )
    parser.add_argument(
        "--out", type=Path,
        default=project_root / "data",
        help="Output root; trees land at <out>/<doc_id>/tree.json.",
    )
    parser.add_argument(
        "--doc-id", type=str, default=None,
        help="Restrict to a single PDF (BSARD stem, e.g. 2004_05_27_2004A27101). "
             "Default: all docs in MyDocuments that have a pdf_filename mapping.",
    )
    parser.add_argument(
        "--force-rebuild", action="store_true",
        help="Ignore cache freshness and rebuild every targeted doc.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s - %(message)s",
    )
    log = logging.getLogger("build_tree")

    if not args.out.exists():
        log.error(
            "Output dir missing: %s\n"
            "Did you create the data/ junction? See README.md (Step 1).",
            args.out,
        )
        return 2

    log.info("Loading AzureDI corpus from %s", args.azuredi)
    nodes, my_documents, definitions_df = load_azuredi_corpus(
        args.azuredi, args.pdf_map,
    )
    log.info(
        "Loaded %d nodes across %d documents",
        len(nodes), len(my_documents),
    )

    log.info("Linking nodes to BSARD via %s", args.bsard_db)
    link_corpus_to_bsard(nodes, args.bsard_db)

    definitions_map = build_definitions_map(definitions_df)
    articles = build_articles_from_nodes(nodes, my_documents, definitions_map)
    log.info("Aggregated %d articles", len(articles))

    nodes_by_doc: dict[int, list] = defaultdict(list)
    for n in nodes:
        nodes_by_doc[n.doc_id].append(n)

    articles_by_doc: dict[int, list] = defaultdict(list)
    for a in articles:
        articles_by_doc[a.document_id].append(a)

    pdf_filename_by_doc: dict[int, str] = {}
    for n in nodes:
        if n.pdf_filename and n.doc_id not in pdf_filename_by_doc:
            pdf_filename_by_doc[n.doc_id] = n.pdf_filename

    # Build the (azure_doc_id → bsard_stem) reverse map. Only docs that have a
    # pdf_filename mapping can be saved under the new layout — flag and skip
    # any without one.
    stem_by_azure_doc: dict[int, str] = {}
    for azure_doc_id, pdf_filename in pdf_filename_by_doc.items():
        stem_by_azure_doc[azure_doc_id] = Path(pdf_filename).stem

    if args.doc_id is not None:
        # User passed a BSARD stem; resolve it to an azure_doc_id.
        match = [a for a, s in stem_by_azure_doc.items() if s == args.doc_id]
        if not match:
            log.error(
                "doc_id %r not in pdf_document_map.csv (no AzureDI mapping for this PDF). "
                "Known stems: %s",
                args.doc_id, sorted(stem_by_azure_doc.values()),
            )
            return 2
        target_azure_ids = match
    else:
        target_azure_ids = sorted(stem_by_azure_doc.keys())

    print()
    print(f"{'doc_id':>30}  {'azure':>5}  {'arts':>5}  {'chaps':>5}  {'derivable':>10}  {'bsard%':>7}  {'cache':>7}  title")
    print("-" * 110)
    for azure_doc_id in target_azure_ids:
        if azure_doc_id not in my_documents:
            log.warning("azure_doc_id=%d not in MyDocuments - skipping", azure_doc_id)
            continue
        doc_articles = articles_by_doc.get(azure_doc_id, [])
        if not doc_articles:
            log.warning(
                "azure_doc_id=%d has 0 aggregated articles - skipping", azure_doc_id,
            )
            continue

        doc_id = stem_by_azure_doc[azure_doc_id]
        pdf_filename = pdf_filename_by_doc.get(azure_doc_id)
        pdf_path = (
            args.pdf_dir / pdf_filename
            if pdf_filename and (args.pdf_dir / pdf_filename).exists()
            else None
        )

        cache_path = args.out / doc_id / "tree.json"
        if not args.force_rebuild and is_cache_fresh(cache_path, pdf_path=pdf_path):
            cached_node, cached_manifest = load_law_tree(cache_path)
            stats = {
                "n_articles": cached_manifest.get("n_articles", 0),
                "n_chapters": cached_manifest.get("n_chapters", 0),
                "bsard_id_coverage": cached_manifest.get("bsard_id_coverage", 0.0),
            }
            chapter_derivable = cached_manifest.get("chapter_derivable", False)
            cache_state = "hit"
        else:
            law_node, chapter_derivable = build_law_tree(
                doc_id=azure_doc_id,
                doc_meta=my_documents[azure_doc_id],
                articles_for_doc=doc_articles,
                nodes_for_doc=nodes_by_doc.get(azure_doc_id, []),
            )
            stats = tree_stats(law_node)
            save_law_tree(
                law_node,
                doc_id=doc_id,
                azure_document_id=azure_doc_id,
                pdf_filename=pdf_filename,
                pdf_path=pdf_path,
                chapter_derivable=chapter_derivable,
                dest_dir=args.out,
            )
            cache_state = "rebuilt"

        title = (my_documents[azure_doc_id].get("document_title", "") or "")[:35]
        print(
            f"{doc_id:>30}  {azure_doc_id:>5}  {stats['n_articles']:>5}  "
            f"{stats['n_chapters']:>5}  {str(chapter_derivable):>10}  "
            f"{stats['bsard_id_coverage']:>7.1%}  {cache_state:>7}  {title!r}"
        )

    print("-" * 110)
    print(f"\nTrees in: {args.out}/<doc_id>/tree.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
