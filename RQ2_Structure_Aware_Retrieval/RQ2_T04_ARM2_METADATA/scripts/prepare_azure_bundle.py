"""Build the input bundle the Azure notebook expects.

Produces a single ``bundle.zip`` (~430 MB) containing every input the
T04 precompute needs, with a flat layout the notebook unpacks into
``/home/azureuser/bundle/``:

    azuredi/
        MyDocuments.csv
        DocumentDefinitions.csv
        VectorDB_Documents.json
        VectorDBdev_Documents.json
    dataset_creation_output/
        bsard_corpus.db
    pdfs/
        1967_10_10_1967101056.pdf
        1867_06_08_1867060850.pdf
        1804_03_21_1804032150.pdf
        1967_10_10_1967101055.pdf
    pdf_document_map.csv

After running this script:
    1. ``azcopy copy bundle.zip "<container-sas-url>/t04_azure_bundles/v4_remaining4.zip"``
    2. Open ``notebooks/azure_t04_precompute_run.ipynb`` on the Azure VM
    3. Set ``GITHUB_TOKEN``, ``AZURE_CONTAINER_SAS_URL`` in cell 1 and run.

Usage::

    python RQ2_T04_ARM2_METADATA/scripts/prepare_azure_bundle.py
    python RQ2_T04_ARM2_METADATA/scripts/prepare_azure_bundle.py --out /tmp/bundle.zip
    python RQ2_T04_ARM2_METADATA/scripts/prepare_azure_bundle.py --stems doc5,doc6
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent

# Defaults mirror the local data-root layout.
DEFAULT_AZUREDI_DIR = _REPO / "RQ2_T04_ARM2_METADATA" / "azuredi"
DEFAULT_BSARD_DB = _REPO / "dataset_creation_output" / "bsard_corpus.db"
DEFAULT_PDF_DOC_MAP = _REPO / "RQ2_T02_DATA_LOADER" / "data" / "csv" / "pdf_document_map.csv"
DEFAULT_PDF_DIR = _REPO / "RQ2_T02_DATA_LOADER" / "data" / "pdfs"

# AzureDI files to ship. CSVs are tiny; VectorDB JSONs are the bulk.
AZUREDI_FILES = (
    "MyDocuments.csv",
    "DocumentDefinitions.csv",
    "VectorDB_Documents.json",
    "VectorDBdev_Documents.json",
)

# Default stems = the 4 that need v4 re-precompute (doc 8 already done).
DEFAULT_STEMS = [
    "1967_10_10_1967101056",   # doc 5
    "1867_06_08_1867060850",   # doc 6
    "1804_03_21_1804032150",   # doc 9
    "1967_10_10_1967101055",   # doc 7
]


def _add(zf: zipfile.ZipFile, src: Path, arcname: str) -> int:
    """Add a file to the zip; return its uncompressed size in bytes."""
    if not src.exists():
        raise FileNotFoundError(f"Missing input: {src}")
    zf.write(src, arcname)
    return src.stat().st_size


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--out", type=Path,
        default=_REPO / "RQ2_T04_ARM2_METADATA" / "bundle.zip",
        help="Output zip path. Default: %(default)s.",
    )
    parser.add_argument(
        "--azuredi-dir", type=Path, default=DEFAULT_AZUREDI_DIR,
        help="AzureDI dump directory.",
    )
    parser.add_argument(
        "--bsard-db", type=Path, default=DEFAULT_BSARD_DB,
        help="bsard_corpus.db path.",
    )
    parser.add_argument(
        "--pdf-doc-map", type=Path, default=DEFAULT_PDF_DOC_MAP,
        help="pdf_document_map.csv path.",
    )
    parser.add_argument(
        "--pdf-dir", type=Path, default=DEFAULT_PDF_DIR,
        help="Directory containing the source PDFs.",
    )
    parser.add_argument(
        "--stems", type=str, default=",".join(DEFAULT_STEMS),
        help="Comma-separated PDF stems to include. Default: the 4 remaining "
             "stems for the v4 re-precompute.",
    )
    args = parser.parse_args(argv)

    stems = [s.strip() for s in args.stems.split(",") if s.strip()]
    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Building bundle at: {args.out}")
    total = 0
    n_files = 0
    with zipfile.ZipFile(args.out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        # 1. AzureDI files
        for name in AZUREDI_FILES:
            size = _add(zf, args.azuredi_dir / name, f"azuredi/{name}")
            total += size
            n_files += 1
            print(f"  + azuredi/{name}  ({size:>12,} B)")

        # 2. BSARD corpus DB
        size = _add(zf, args.bsard_db, f"dataset_creation_output/{args.bsard_db.name}")
        total += size
        n_files += 1
        print(f"  + dataset_creation_output/{args.bsard_db.name}  ({size:>12,} B)")

        # 3. pdf_document_map.csv (small)
        size = _add(zf, args.pdf_doc_map, "pdf_document_map.csv")
        total += size
        n_files += 1
        print(f"  + pdf_document_map.csv  ({size:>12,} B)")

        # 4. The 4 PDFs (needed for pdf_sha256 fingerprint so v4 hashes match the local build).
        for stem in stems:
            pdf = args.pdf_dir / f"{stem}.pdf"
            size = _add(zf, pdf, f"pdfs/{pdf.name}")
            total += size
            n_files += 1
            print(f"  + pdfs/{pdf.name}  ({size:>12,} B)")

    zip_size = args.out.stat().st_size
    print()
    print(f"  files:               {n_files}")
    print(f"  uncompressed total:  {total/1024/1024:.1f} MB")
    print(f"  zip size:            {zip_size/1024/1024:.1f} MB ({100*zip_size/total:.1f}% of raw)")
    print()
    print("Upload to Azure Blob:")
    print(f'  azcopy copy "{args.out}" "<container-sas-url>/t04_azure_bundles/v4_remaining4.zip"')
    return 0


if __name__ == "__main__":
    sys.exit(main())
