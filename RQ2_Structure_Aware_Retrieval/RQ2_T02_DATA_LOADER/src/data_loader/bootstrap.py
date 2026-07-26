"""Per-PDF discovery stamp.

Writes ``RQ2_T02_DATA_LOADER/data/<doc_id>/discovery.json`` so other
projects (and the orchestrator status command) can answer "is this PDF
discoverable, and what is its canonical pdf_sha256?" without re-running
auto-discovery against the AzureDI dump.

The status registry treats ``discovery.json`` as the single source of
truth for ``pdf_sha256``. Every other project that records a sha256
copies the value from here rather than recomputing it.

CLI:

    python -m data_loader.bootstrap --doc-id 1804_03_21_1804032150
    python -m data_loader.bootstrap --doc-id 1804_03_21_1804032150 --force
    python -m data_loader.bootstrap --all
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1


def _resolve_t00_paths():
    """Late-import the T00 path resolver. T02 is a base layer; we only
    need ``project_dir`` / ``selected_pdf_doc_ids`` and don't want
    a hard package-level dependency on T00."""
    # bootstrap.py → data_loader/ → src/ → RQ2_T02_DATA_LOADER/ → RQ2_ROOT/
    rq2_root = Path(__file__).resolve().parents[3]
    t00_src = rq2_root / "RQ2_T00_ORCHESTRATOR" / "src"
    if str(t00_src) not in sys.path:
        sys.path.insert(0, str(t00_src))
    from orchestrator import paths
    return paths


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _pdf_n_pages(path: Path) -> int:
    import fitz
    with fitz.open(path) as doc:
        return doc.page_count


def _load_pdf_document_map(map_path: Path) -> dict[str, int]:
    df = pd.read_csv(map_path, encoding="utf-8")
    return {
        str(row["pdf_filename"]): int(row["document_id"])
        for _, row in df.iterrows()
    }


def _count_azure_articles(
    azuredi_dir: Path,
    pdf_document_map_csv: Path,
    azure_document_id: int,
) -> int | None:
    """Best-effort: load the AzureDI dump via T04's loader and count nodes
    for ``azure_document_id``. Returns None if T04 is not importable."""
    try:
        sys.path.insert(0, str(azuredi_dir.parent.parent / "RQ2_T04_ARM2_METADATA" / "src"))
        from arm2_metadata.azuredi_loader import load_azuredi_corpus
    except ImportError as exc:
        logger.warning(
            "arm2_metadata.azuredi_loader not importable; n_articles_in_azure will be null (%s)",
            exc,
        )
        return None
    nodes, _my_docs, _defs = load_azuredi_corpus(azuredi_dir, pdf_document_map_csv)
    return sum(1 for n in nodes if n.doc_id == azure_document_id)


def build_discovery(
    doc_id: str,
    *,
    pdf_dir: Path,
    pdf_document_map_csv: Path,
    azuredi_dir: Path | None,
    out_path: Path,
    force: bool,
) -> dict | None:
    """Compute and write ``discovery.json`` for one PDF. Returns the payload,
    or ``None`` when the PDF file is missing (no file is written)."""
    pdf_filename = f"{doc_id}.pdf"
    pdf_path = pdf_dir / pdf_filename
    if not pdf_path.exists():
        logger.error("PDF not found, skipping: %s", pdf_path)
        return None

    if out_path.exists() and not force:
        logger.info("discovery.json already exists, skipping (use --force to overwrite): %s", out_path)
        return json.loads(out_path.read_text(encoding="utf-8"))

    pdf_map = _load_pdf_document_map(pdf_document_map_csv)
    azure_document_id = pdf_map.get(pdf_filename)

    n_articles_in_azure: int | None = None
    if azure_document_id is not None and azuredi_dir is not None and azuredi_dir.exists():
        n_articles_in_azure = _count_azure_articles(
            azuredi_dir, pdf_document_map_csv, azure_document_id,
        )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "doc_id": doc_id,
        "pdf_filename": pdf_filename,
        "pdf_sha256": _sha256(pdf_path),
        "n_pages": _pdf_n_pages(pdf_path),
        "has_azure_extraction": azure_document_id is not None,
        "azure_document_id": azure_document_id,
        "n_articles_in_azure": n_articles_in_azure,
        "generated_at": _now_utc(),
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

    t02_dir = paths.project_dir("T02")
    default_pdf_dir = t02_dir / "data" / "pdfs"
    default_map = t02_dir / "data" / "csv" / "pdf_document_map.csv"
    default_azuredi = paths.project_dir("T04") / "azuredi"

    parser = argparse.ArgumentParser(prog="data_loader.bootstrap", description=__doc__.split("\n", 1)[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--doc-id", help="Single PDF stem (e.g. 1804_03_21_1804032150)")
    group.add_argument("--all", action="store_true",
                       help="Loop over RQ2_T00_ORCHESTRATOR/data/selected_pdfs.json")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite an existing discovery.json")
    parser.add_argument("--pdf-dir", type=Path, default=default_pdf_dir)
    parser.add_argument("--pdf-document-map", type=Path, default=default_map)
    parser.add_argument("--azuredi-dir", type=Path, default=default_azuredi,
                        help="Path to AzureDI dump (or 'none' to skip Azure article count)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    azuredi_dir: Path | None = args.azuredi_dir
    if str(args.azuredi_dir).lower() == "none":
        azuredi_dir = None

    doc_ids = paths.selected_pdf_doc_ids() if args.all else [args.doc_id]

    n_written = 0
    n_skipped = 0
    for doc_id in doc_ids:
        out_path = paths.project_data_dir("T02", doc_id) / "discovery.json"
        result = build_discovery(
            doc_id,
            pdf_dir=args.pdf_dir,
            pdf_document_map_csv=args.pdf_document_map,
            azuredi_dir=azuredi_dir,
            out_path=out_path,
            force=args.force,
        )
        if result is None:
            n_skipped += 1
        else:
            n_written += 1

    logger.info("Done: %d written, %d skipped", n_written, n_skipped)
    return 0 if n_skipped == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
