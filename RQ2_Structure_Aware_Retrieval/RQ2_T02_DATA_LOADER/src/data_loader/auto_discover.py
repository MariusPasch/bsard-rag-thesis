"""PDF auto-discovery and matching to MyDocuments.csv via pdf_document_map.csv."""

import logging
from glob import glob
from pathlib import Path

import pandas as pd

from .models import DocumentBundle

logger = logging.getLogger(__name__)


def discover_documents(config: dict) -> list[DocumentBundle]:
    """
    Scan pdf_dir, match each PDF to MyDocuments.csv via pdf_document_map.csv,
    join definitions from DocumentDefinitions.csv via DocumentId.

    PDFs not listed in pdf_document_map.csv get has_azure_extraction=False
    and will be processed by Arm 1 only.
    """
    pdf_dir = config["data"]["pdf_dir"]
    docs_csv = pd.read_csv(config["data"]["documents_csv"], encoding="utf-8", encoding_errors="replace")
    defs_csv = pd.read_csv(config["data"]["definitions_csv"], encoding="utf-8", encoding_errors="replace")
    pdf_map = pd.read_csv(config["data"]["pdf_document_map"])

    filename_to_doc_id: dict[str, int] = dict(
        zip(pdf_map["pdf_filename"], pdf_map["document_id"].astype(int))
    )
    doc_id_to_row: dict[int, dict] = {
        int(row["DocumentId"]): row.to_dict()
        for _, row in docs_csv.iterrows()
    }

    bundles: list[DocumentBundle] = []
    for pdf_path in sorted(glob(f"{pdf_dir}/*.pdf")):
        filename = Path(pdf_path).name

        if filename in filename_to_doc_id:
            doc_id = filename_to_doc_id[filename]
            doc_row = doc_id_to_row.get(doc_id)
            if doc_row is None:
                logger.error(f"{filename} mapped to DocumentId={doc_id} but row not found in MyDocuments.csv")
                continue

            doc_defs = defs_csv[defs_csv["DocumentId"] == doc_id].copy()
            bundle = DocumentBundle(
                document_id=doc_id,
                pdf_path=pdf_path,
                pdf_filename=filename,
                metadata=doc_row,
                definitions=doc_defs,
                articles=[],
                raw_text="",
                has_azure_extraction=True,
            )
            logger.info(f"Matched {filename} → DocumentId={doc_id}, {len(doc_defs)} definitions")
        else:
            bundle = DocumentBundle(
                document_id=-1,
                pdf_path=pdf_path,
                pdf_filename=filename,
                metadata={},
                definitions=pd.DataFrame(),
                articles=[],
                raw_text="",
                has_azure_extraction=False,
            )
            logger.debug(f"No Azure extraction for {filename}, Arm 1 only")

        bundles.append(bundle)

    logger.info(f"Discovered {len(bundles)} PDFs, {sum(b.has_azure_extraction for b in bundles)} with Azure extraction")
    return bundles
