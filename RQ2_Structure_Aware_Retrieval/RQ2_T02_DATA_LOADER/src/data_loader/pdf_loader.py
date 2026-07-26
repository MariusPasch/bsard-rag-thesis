"""PyMuPDF raw text extraction — page-by-page, joined with '\\n\\n'."""

import fitz  # PyMuPDF

from .models import DocumentBundle


def extract_raw_text(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    return "\n\n".join(page.get_text() for page in doc)


def populate_raw_text(bundle: DocumentBundle) -> DocumentBundle:
    bundle.raw_text = extract_raw_text(bundle.pdf_path)
    return bundle
