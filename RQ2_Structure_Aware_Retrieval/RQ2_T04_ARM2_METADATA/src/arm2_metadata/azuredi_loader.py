"""Read the AzureDI dump (MyDocuments.csv, DocumentDefinitions.csv,
VectorDB_Documents.json, VectorDBdev_Documents.json) into a list of AzureNode.

The two JSON files are MongoDB extended-JSON dumps: the outer file is *almost*
JSON but contains tokens like ``{"$numberDouble": "NaN"}`` and stray commas
between top-level objects, so ``json.load(f)`` on the whole file fails. We walk
top-level objects with a balanced-brace scanner and parse each one independently
with ``json.loads``.

Filter rules (from IMPLEMENTATION_PROMPT.md §2 + §6 Step B):
  * ``DocumentId == 1`` is dropped unconditionally (EU decision left over from a
    different process; not a BSARD source).
  * ``MyDocuments.csv`` at the top level of ``azuredi/`` is the source of truth
    for which document_ids exist this run. Anything under ``azuredi/backups/``
    is ignored.
  * Records in ``VectorDB*.json`` whose ``payload.metadata.document_id`` is not
    declared in the surviving MyDocuments set are skipped silently.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

logger = logging.getLogger(__name__)

# ── Files we read at the top level of azuredi/ ───────────────────────────────

_MYDOCS_FILE = "MyDocuments.csv"
_DEFINITIONS_FILE = "DocumentDefinitions.csv"
_VECTORDB_FILES = ("VectorDB_Documents.json", "VectorDBdev_Documents.json")

# ── Article marker — duplicated so this module is callable without importing T03.
# Same compiled regex is also used through arm1_naive in bsard_link, so this is
# only a fallback for inspection / sanity checks.

import re

_ART_PREFIX_RE = re.compile(r"^\s*art\.?\s+", re.IGNORECASE)


# ── Data model ───────────────────────────────────────────────────────────────


@dataclass
class AzureNode:
    """One AzureDI tree node, post-filter, pre-BSARD-link.

    All fields are populated from the raw JSON record's ``payload.metadata`` and
    ``payload.page_content`` (plus the document-level fields inherited from
    MyDocuments.csv). BSARD linkage (``bsard_ids``, ``matched_article_numbers``)
    is filled in by ``bsard_link.link_corpus_to_bsard``.
    """

    # ── Identity ────────────────────────────────────────────────────
    node_id: int
    doc_id: int

    # ── Content ─────────────────────────────────────────────────────
    page_content: str               # primary FR text
    content_summary: str            # English summary (gpt-4o)
    keywords: str                   # comma-separated English topic tags
    level_summary: str

    # ── Structure ───────────────────────────────────────────────────
    parent_id: int | None
    node_level: int
    node_source: str                # e.g. "Art. D15", "PARTIE III. - …"
    page_number: int
    is_header: bool
    has_requirements: bool
    path_to_item: str
    bounding_boxes: list[dict]

    # ── Document-level passthrough (from MyDocuments.csv) ───────────
    pdf_filename: str | None        # via pdf_document_map (None if not BSARD-mapped)
    document_title: str
    jurisdiction: str
    issuing_authority: str
    document_number: str
    issue_date: str
    entry_into_force: str
    regulatory_status: str
    document_type: str
    document_status: str            # per-node: "Effective" | …

    # ── Filled in by bsard_link / definition inversion ──────────────
    bsard_ids: list[int] = field(default_factory=list)
    matched_article_numbers: list[str] = field(default_factory=list)
    defined_terms: list[str] = field(default_factory=list)


# ── Balanced-brace iterator ──────────────────────────────────────────────────


def iter_objects(path: str | Path) -> Iterator[dict]:
    """Yield top-level JSON objects from a MongoDB-style dump.

    The file may contain stray commas between objects and MongoDB extended-JSON
    tokens like ``{"$numberDouble": "NaN"}``. Each top-level object is parsed
    independently with ``json.loads`` after extracting it via brace counting
    that respects strings and escapes.
    """
    raw = Path(path).read_text(encoding="utf-8")
    n = len(raw)
    i = 0
    while i < n:
        # find next opening brace at depth 0
        while i < n and raw[i] != "{":
            i += 1
        if i >= n:
            return

        start = i
        depth = 0
        in_str = False
        esc = False
        while i < n:
            c = raw[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            else:
                if c == '"':
                    in_str = True
                elif c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        i += 1
                        break
            i += 1

        chunk = raw[start:i]
        # MongoDB extended-JSON: replace the only NaN form we've actually seen
        # in this dump so json.loads succeeds. Other extended forms (ObjectId,
        # numberDouble with a real value) round-trip as nested dicts which is
        # fine — we never read _id and ignore vector_full_embeddings entirely.
        chunk_clean = chunk.replace(
            '{"$numberDouble": "NaN"}', "NaN"
        ).replace(
            '{ "$numberDouble": "NaN" }', "NaN"
        )
        try:
            yield json.loads(chunk_clean)
        except json.JSONDecodeError:
            # Last-resort cleanup: NaN → null so json.loads accepts it. This
            # only fires for records the targeted replacement above missed.
            try:
                yield json.loads(chunk_clean.replace("NaN", "null"))
            except json.JSONDecodeError as exc:
                logger.warning(
                    "Skipping unparseable record at offset %d: %s",
                    start, exc,
                )


# ── Helpers ──────────────────────────────────────────────────────────────────


def _coerce_int(v: Any) -> int | None:
    """Best-effort int coercion from the messy AzureDI JSON values."""
    if v is None:
        return None
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, float)):
        if isinstance(v, float) and (v != v):  # NaN
            return None
        return int(v)
    if isinstance(v, dict):
        # MongoDB extended JSON wrappers
        for key in ("$numberInt", "$numberLong", "$numberDouble"):
            if key in v:
                try:
                    return int(float(v[key]))
                except (TypeError, ValueError):
                    return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _coerce_str(v: Any, default: str = "") -> str:
    if v is None:
        return default
    if isinstance(v, str):
        return v
    return str(v)


def _coerce_bbox(v: Any) -> list[dict]:
    """bounding_boxes may be a JSON-stringified list, a list of dicts, or None."""
    if v is None:
        return []
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            return []
    if isinstance(v, list):
        return [b for b in v if isinstance(b, dict)]
    return []


# ── Top-level loader ─────────────────────────────────────────────────────────


def _load_my_documents(azuredi_dir: Path) -> dict[int, dict]:
    path = azuredi_dir / _MYDOCS_FILE
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    df = pd.read_csv(path, encoding="utf-8")
    out: dict[int, dict] = {}
    for _, row in df.iterrows():
        doc_id = _coerce_int(row.get("DocumentId"))
        if doc_id is None or doc_id == 1:
            # Drop the EU decision unconditionally — see PROJECT_CONTEXT / memory
            continue
        out[doc_id] = {
            "document_title": _coerce_str(row.get("DocumentTitle")),
            "document_type": _coerce_str(row.get("DocumentType")),
            "regulatory_status": _coerce_str(row.get("RegulatoryStatus"), "Effective"),
            "jurisdiction": _coerce_str(row.get("Jurisdiction")),
            "issue_date": _coerce_str(row.get("IssueDate")),
            "entry_into_force": _coerce_str(row.get("EntryIntoForceDate")),
            "version_number": _coerce_str(row.get("VersionNumber")),
            "document_number": _coerce_str(row.get("DocumentNumber")),
            "issuing_authority": _coerce_str(row.get("IssuingAuthority")),
            "summary": _coerce_str(row.get("Summary")),
            "language": _coerce_str(row.get("Language")),
            "document_pdf_url": _coerce_str(row.get("DocumentPdfUrl")),
        }
    return out


def _load_definitions(
    azuredi_dir: Path, kept_doc_ids: set[int],
) -> tuple[pd.DataFrame, dict[tuple[int, int], list[str]]]:
    """Return (filtered_definitions_df, {(doc_id, node_id): [Term, …]})."""
    path = azuredi_dir / _DEFINITIONS_FILE
    if not path.exists():
        logger.warning("DocumentDefinitions.csv missing at %s", path)
        return pd.DataFrame(), {}
    df = pd.read_csv(path, encoding="utf-8")
    df = df[df["DocumentId"].apply(lambda v: _coerce_int(v) in kept_doc_ids)].copy()

    invert: dict[tuple[int, int], list[str]] = {}
    for _, row in df.iterrows():
        doc_id = _coerce_int(row.get("DocumentId"))
        if doc_id is None:
            continue
        term = _coerce_str(row.get("Term"))
        used_in_raw = row.get("UsedIn")
        try:
            used_in = json.loads(used_in_raw) if isinstance(used_in_raw, str) else (used_in_raw or [])
        except json.JSONDecodeError:
            used_in = []
        for nid in used_in:
            n = _coerce_int(nid)
            if n is None:
                continue
            invert.setdefault((doc_id, n), []).append(term)
    return df, invert


def _load_pdf_document_map(map_path: Path | None) -> dict[int, str]:
    """Read pdf_document_map.csv → {document_id: pdf_filename}.

    When the map is missing or empty, AzureDI nodes simply carry
    ``pdf_filename=None`` (no BSARD coverage).
    """
    if map_path is None or not Path(map_path).exists():
        return {}
    df = pd.read_csv(map_path, encoding="utf-8")
    out: dict[int, str] = {}
    for _, row in df.iterrows():
        doc_id = _coerce_int(row.get("document_id"))
        pdf_filename = _coerce_str(row.get("pdf_filename"))
        if doc_id is not None and pdf_filename:
            out[doc_id] = pdf_filename
    return out


def load_azuredi_corpus(
    azuredi_dir: str | Path,
    pdf_document_map_csv: str | Path | None = None,
) -> tuple[
    list[AzureNode],
    dict[int, dict],
    pd.DataFrame,
]:
    """Load and filter the AzureDI dump.

    Returns:
      (nodes, my_documents, definitions_df) — ready for downstream linkage,
      enrichment, and indexing.

      * ``nodes``: list[AzureNode] — only records whose ``document_id`` survives
        the MyDocuments filter (which always drops doc_id=1).
      * ``my_documents``: {doc_id: doc-level metadata dict} for the kept docs.
      * ``definitions_df``: DataFrame from DocumentDefinitions.csv, restricted
        to kept docs. Used by both the term-match boost and the article-level
        ``term_definitions`` field.
    """
    azuredi_dir = Path(azuredi_dir)
    if not azuredi_dir.exists():
        raise FileNotFoundError(f"AzureDI directory missing: {azuredi_dir}")

    my_documents = _load_my_documents(azuredi_dir)
    kept_doc_ids = set(my_documents)
    if not kept_doc_ids:
        raise RuntimeError(
            f"No usable documents in {azuredi_dir/_MYDOCS_FILE} after dropping doc_id=1."
        )
    logger.info("Kept document_ids from MyDocuments.csv: %s", sorted(kept_doc_ids))

    definitions_df, term_index = _load_definitions(azuredi_dir, kept_doc_ids)
    pdf_map = _load_pdf_document_map(
        Path(pdf_document_map_csv) if pdf_document_map_csv else None
    )

    nodes: list[AzureNode] = []
    seen_per_doc: dict[int, int] = {}
    skipped_doc_ids: dict[int, int] = {}

    for vfile in _VECTORDB_FILES:
        vpath = azuredi_dir / vfile
        if not vpath.exists():
            logger.info("VectorDB file not present, skipping: %s", vpath)
            continue
        for record in iter_objects(vpath):
            payload = record.get("payload") or {}
            metadata = payload.get("metadata") or {}
            doc_id = _coerce_int(metadata.get("document_id"))
            if doc_id is None or doc_id not in kept_doc_ids:
                if doc_id is not None:
                    skipped_doc_ids[doc_id] = skipped_doc_ids.get(doc_id, 0) + 1
                continue

            node_id = _coerce_int(metadata.get("id")) or _coerce_int(record.get("id"))
            if node_id is None:
                continue

            doc_meta = my_documents[doc_id]
            page_content = _coerce_str(payload.get("page_content"))
            node_source = _coerce_str(metadata.get("node_source"))
            path_to_item = _coerce_str(metadata.get("path_to_item"))
            content_summary = _coerce_str(metadata.get("content_summary"))
            keywords = _coerce_str(metadata.get("keywords"))
            level_summary = _coerce_str(metadata.get("level_summary"))

            node = AzureNode(
                node_id=node_id,
                doc_id=doc_id,
                page_content=page_content,
                content_summary=content_summary,
                keywords=keywords,
                level_summary=level_summary,
                parent_id=_coerce_int(metadata.get("parent_id")),
                node_level=_coerce_int(metadata.get("node_level")) or 0,
                node_source=node_source,
                page_number=_coerce_int(metadata.get("page_number")) or 0,
                is_header=bool(metadata.get("is_header", False)),
                has_requirements=bool(metadata.get("has_requirements", False)),
                path_to_item=path_to_item,
                bounding_boxes=_coerce_bbox(metadata.get("bounding_boxes")),
                pdf_filename=pdf_map.get(doc_id),
                document_title=doc_meta["document_title"],
                jurisdiction=doc_meta["jurisdiction"],
                issuing_authority=doc_meta["issuing_authority"],
                document_number=doc_meta["document_number"],
                issue_date=doc_meta["issue_date"],
                entry_into_force=doc_meta["entry_into_force"],
                regulatory_status=doc_meta["regulatory_status"],
                document_type=_coerce_str(metadata.get("document_type"))
                              or doc_meta["document_type"],
                document_status=_coerce_str(metadata.get("document_status"), "Effective"),
                defined_terms=list(term_index.get((doc_id, node_id), [])),
            )
            nodes.append(node)
            seen_per_doc[doc_id] = seen_per_doc.get(doc_id, 0) + 1

    logger.info(
        "Loaded %d AzureDI nodes (per doc: %s); skipped %d records from non-kept doc_ids: %s",
        len(nodes), seen_per_doc, sum(skipped_doc_ids.values()), skipped_doc_ids,
    )
    return nodes, my_documents, definitions_df


# ── --inspect CLI ────────────────────────────────────────────────────────────


def _inspect(azuredi_dir: Path, pdf_document_map_csv: Path | None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")
    nodes, my_documents, definitions_df = load_azuredi_corpus(
        azuredi_dir, pdf_document_map_csv
    )

    by_doc: dict[int, list[AzureNode]] = {}
    for n in nodes:
        by_doc.setdefault(n.doc_id, []).append(n)

    print(f"AzureDI dump: {azuredi_dir}")
    print(f"Documents (post-filter):           {sorted(my_documents)}")
    print(f"Definitions rows (kept):           {len(definitions_df)}")
    print(f"Total nodes:                        {len(nodes)}")
    for doc_id, group in sorted(by_doc.items()):
        n_header = sum(1 for n in group if n.is_header)
        n_content = len(group) - n_header
        n_with_text = sum(1 for n in group if n.page_content.strip())
        n_node_source_art = sum(1 for n in group if _ART_PREFIX_RE.match(n.node_source or ""))
        token_counts = [len(n.page_content.split()) for n in group if n.page_content.strip()]
        mean_tokens = statistics.mean(token_counts) if token_counts else 0.0
        median_tokens = statistics.median(token_counts) if token_counts else 0.0
        max_tokens = max(token_counts) if token_counts else 0
        print(f"\n  doc_id={doc_id}  title={my_documents[doc_id]['document_title'][:60]!r}")
        print(f"    nodes:                     {len(group)}")
        print(f"    headers / content split:   {n_header} / {n_content}")
        print(f"    nodes with non-empty text: {n_with_text}")
        print(f"    node_source matches Art.:  {n_node_source_art}  ({n_node_source_art/len(group):.1%})")
        print(f"    page_content words:        mean={mean_tokens:.1f}  median={median_tokens:.0f}  max={max_tokens}")
        n_with_terms = sum(1 for n in group if n.defined_terms)
        print(f"    nodes carrying defined_terms: {n_with_terms}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m arm2_metadata.azuredi_loader",
        description="Inspect the AzureDI dump (counts, headers, token stats).",
    )
    parser.add_argument(
        "--azuredi", type=Path,
        default=Path(__file__).resolve().parents[2] / "azuredi",
        help="Path to azuredi/ (default: ../../azuredi relative to this file).",
    )
    parser.add_argument(
        "--pdf-map", type=Path,
        default=Path(__file__).resolve().parents[3]
                / "RQ2_T02_DATA_LOADER" / "data" / "csv" / "pdf_document_map.csv",
        help="Path to pdf_document_map.csv (BSARD doc-id ↔ pdf_filename).",
    )
    parser.add_argument("--inspect", action="store_true",
                        help="Print summary counts and exit.")
    args = parser.parse_args(argv)

    if args.inspect:
        return _inspect(args.azuredi, args.pdf_map)
    parser.error("--inspect is required (no other subcommands yet).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
