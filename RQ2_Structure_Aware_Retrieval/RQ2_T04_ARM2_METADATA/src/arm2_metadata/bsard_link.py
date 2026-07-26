"""Map AzureDI nodes → BSARD bsard_ids.

Bridge (IMPLEMENTATION_PROMPT.md §4 + extensions noted below):
  1. Run T03's ``_ART_RE`` over ``page_content`` to collect every "Art. X"
     marker the node carries.
  2. Fall back to ``node_source`` when it starts with "Art. …".
  3. **Parent-chain inheritance** for orphan content nodes: walk up
     ``parent_id`` looking for an ancestor with article anchors. Useful for
     dumps where article-anchored nodes are *parents* of their body paragraphs.
  4. **Preceding-content-node inheritance** for remaining orphans: walk content
     nodes in ``(page_number, node_id)`` order and inherit from the most
     recent article-anchored content node. This catches the common AzureDI
     layout where body paragraphs (``§ 2.``, ``1°``, ``a.`` …) are emitted
     as siblings of the article-anchor node rather than as its children, so
     parent-chain inheritance fails. Without this pass, ~38% of content nodes
     in the doc-2 dump get ``bsard_ids=[]`` and ~131 KB of body text never
     reach the index.
  5. Resolve each article number → bsard_id via the BSARD corpus DB. Reuses
     T03's normalisation + leading-integer fallback verbatim.

Both inheritance passes are scoped per ``doc_id`` so a node never inherits
across documents. The preceding-content-node anchor never crosses a header
node either: when an orphan inherits, it inherits from the *content* node
that most recently anchored an article, regardless of any structural headers
(PARTIE / LIVRE) that appeared between them.

Sanity-check entry point: ``python -m arm2_metadata.bsard_link --inspect``.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

from arm1_naive.chunker import _ART_RE, _normalise_key

from .azuredi_loader import AzureNode, _ART_PREFIX_RE, load_azuredi_corpus

logger = logging.getLogger(__name__)


# AzureDI emits hierarchy headers with Markdown prefixes ("##### Art. 219.[1 ...").
# T03's ``_ART_RE`` anchors at start-of-line and accepts only whitespace before
# ``Art``, so those headers never produce a Pass-1 match and the body
# paragraphs that should inherit from them via parent-chain inheritance fall
# back to whatever preceding *content* anchor exists. On doc 8 that's
# catastrophic — 110 article anchors are emitted only in Markdown-prefixed
# form, so ~25 % of all body text on the PDF inherits the wrong bsard_id
# (115 KB of rental-law paragraphs all swallowed by bsard_id 814).
#
# Same body as ``_ART_RE`` but ``[\s#]*`` lets the line start with any mix of
# whitespace and ``#`` markers, fixing the inheritance chain on Markdown-style
# header dumps without affecting T03 (PyMuPDF flat-text extraction never emits
# ``#`` prefixes). Promoted to LINKER_VERSION 4 in retriever.py.
_ART_RE_AZURE = re.compile(
    r"^[\s#]*Art\.?\s+([A-Z0-9][A-Z0-9_./-]*)",
    re.MULTILINE | re.IGNORECASE,
)


# ── DB query (mirrors T03 chunker.locate_articles) ───────────────────────────


def _load_article_lookup(
    conn: sqlite3.Connection, pdf_filename: str,
) -> tuple[dict[str, int], dict[str, list[int]]]:
    """Return (exact, lead) maps for the given pdf_filename.

    Tries the bare basename first, then the legacy ``img_l_pdf_<stem>_F.pdf``
    form (DB stores both shapes for different PDFs).
    """
    rows = conn.execute(
        "SELECT bsard_id, article_number FROM articles "
        "WHERE pdf_filename = ? AND bsard_id IS NOT NULL",
        (pdf_filename,),
    ).fetchall()
    if not rows:
        stem = Path(pdf_filename).stem
        db_filename = f"img_l_pdf_{stem}_F.pdf"
        rows = conn.execute(
            "SELECT bsard_id, article_number FROM articles "
            "WHERE pdf_filename = ? AND bsard_id IS NOT NULL",
            (db_filename,),
        ).fetchall()

    exact: dict[str, int] = {}
    lead: dict[str, list[int]] = defaultdict(list)
    for bsard_id, art_no in rows:
        if not art_no:
            continue
        exact[_normalise_key(art_no)] = int(bsard_id)
        m = re.match(r"^(\d+)", art_no)
        if m:
            lead[m.group(1)].append(int(bsard_id))
    return exact, dict(lead)


def article_no_to_bsard(
    art_no: str,
    exact: dict[str, int],
    lead: dict[str, list[int]],
) -> int | None:
    nk = _normalise_key(art_no)
    if nk in exact:
        return exact[nk]
    m = re.match(r"^(\d+)", art_no)
    if m and len(lead.get(m.group(1), [])) == 1:
        return lead[m.group(1)][0]
    return None


# ── Per-node article-number extraction ───────────────────────────────────────


def _node_own_article_numbers(node: AzureNode) -> list[str]:
    """Article numbers that fire on the node's own text (page_content + node_source).

    page_content is primary; node_source is a fallback (only fires on this dump
    when the AzureDI extraction stamped an article-anchor as the node label,
    which empirically does not happen for doc_id=2 — kept for future dumps).

    Uses ``_ART_RE_AZURE`` (Markdown-prefix-tolerant) on page_content so dumps
    that emit ``##### Art. NNN`` headers participate in Pass-1 anchoring.
    Falls back to ``_ART_RE`` on node_source where the bare form is the only
    shape AzureDI emits.
    """
    found: list[str] = []
    if node.page_content:
        for m in _ART_RE_AZURE.finditer(node.page_content):
            found.append(m.group(1).rstrip("."))
    if not found and node.node_source and _ART_PREFIX_RE.match(node.node_source):
        m = _ART_RE.search(node.node_source)
        if m:
            found.append(m.group(1).rstrip("."))
    # de-dup while preserving order
    seen: set[str] = set()
    uniq: list[str] = []
    for a in found:
        if a not in seen:
            uniq.append(a)
            seen.add(a)
    return uniq


def resolve_bsard_ids(
    art_numbers: list[str],
    exact: dict[str, int],
    lead: dict[str, list[int]],
) -> tuple[list[int], list[str]]:
    """Map a list of article numbers → (bsard_ids, matched_article_numbers).

    Article numbers that don't resolve to a bsard_id are dropped from the
    matched_article_numbers list as well (so the two lists are aligned).
    Duplicates in the bsard_ids output are collapsed while preserving order.
    """
    bids: list[int] = []
    matched: list[str] = []
    seen: set[int] = set()
    for an in art_numbers:
        bid = article_no_to_bsard(an, exact, lead)
        if bid is None:
            continue
        if bid not in seen:
            bids.append(bid)
            seen.add(bid)
        matched.append(an)
    return bids, matched


# ── Linker ───────────────────────────────────────────────────────────────────


def link_corpus_to_bsard(
    nodes: list[AzureNode],
    bsard_db_path: str | Path,
    pdf_document_map: dict[int, str] | None = None,
) -> list[AzureNode]:
    """Fill ``node.bsard_ids`` and ``node.matched_article_numbers`` in place.

    Process per-node:
      1. Look up article numbers in the node's own text.
      2. If none AND the node has a parent → walk up the parent chain
         (within the same doc) and inherit the closest ancestor's
         matched_article_numbers. This catches orphan paragraphs that
         start with ``§ N.`` rather than ``Art. X``.

    Args:
        nodes:              Output of ``load_azuredi_corpus``.
        bsard_db_path:      Path to ``bsard_corpus.db``.
        pdf_document_map:   {doc_id: pdf_filename}. When omitted, derived from
                            ``node.pdf_filename``. Nodes whose doc_id has no
                            mapped PDF get ``bsard_ids=[]`` (no BSARD coverage).
    """
    if pdf_document_map is None:
        pdf_document_map = {
            n.doc_id: n.pdf_filename
            for n in nodes
            if n.pdf_filename
        }

    conn = sqlite3.connect(str(bsard_db_path))
    try:
        conn.row_factory = sqlite3.Row

        # Build per-pdf lookup tables once.
        lookups: dict[int, tuple[dict[str, int], dict[str, list[int]]]] = {}
        for doc_id, pdf_filename in pdf_document_map.items():
            if pdf_filename:
                lookups[doc_id] = _load_article_lookup(conn, pdf_filename)

        # First pass: own-text article numbers.
        nodes_by_id: dict[tuple[int, int], AzureNode] = {(n.doc_id, n.node_id): n for n in nodes}
        for node in nodes:
            own = _node_own_article_numbers(node)
            node.matched_article_numbers = list(own)
            if node.doc_id in lookups:
                exact, lead = lookups[node.doc_id]
                bids, matched = resolve_bsard_ids(own, exact, lead)
                node.bsard_ids = bids
                node.matched_article_numbers = matched

        # Second pass: orphan inheritance via parent chain.
        for node in nodes:
            if node.bsard_ids or node.doc_id not in lookups:
                continue
            exact, lead = lookups[node.doc_id]
            cur_parent = node.parent_id
            depth = 0
            while cur_parent is not None and depth < 32:
                parent = nodes_by_id.get((node.doc_id, cur_parent))
                if parent is None:
                    break
                if parent.matched_article_numbers:
                    bids, matched = resolve_bsard_ids(
                        parent.matched_article_numbers, exact, lead,
                    )
                    if bids:
                        node.bsard_ids = bids
                        node.matched_article_numbers = matched
                        break
                cur_parent = parent.parent_id
                depth += 1

        # Third pass: preceding-content-node inheritance. For each doc, walk
        # content nodes in (page_number, node_id) order, tracking the most
        # recently article-anchored content node. Orphans (still no
        # bsard_ids after passes 1+2) inherit from that anchor. Header nodes
        # are skipped — they don't reset the anchor, but they don't anchor
        # either.
        nodes_by_doc: dict[int, list[AzureNode]] = defaultdict(list)
        for node in nodes:
            if node.doc_id in lookups:
                nodes_by_doc[node.doc_id].append(node)

        for doc_id, doc_nodes in nodes_by_doc.items():
            doc_nodes.sort(key=lambda n: (n.page_number, n.node_id))
            exact, lead = lookups[doc_id]
            anchor_arts: list[str] = []
            anchor_bids: list[int] = []
            for node in doc_nodes:
                if node.is_header or not node.page_content.strip():
                    continue
                if node.matched_article_numbers and node.bsard_ids:
                    # Update tracker — this node anchors subsequent orphans.
                    anchor_arts = list(node.matched_article_numbers)
                    anchor_bids = list(node.bsard_ids)
                    continue
                if not node.bsard_ids and anchor_bids:
                    node.bsard_ids = list(anchor_bids)
                    node.matched_article_numbers = list(anchor_arts)

    finally:
        conn.close()

    return nodes


# ── --inspect CLI ────────────────────────────────────────────────────────────


def _inspect(
    azuredi_dir: Path,
    pdf_document_map_csv: Path,
    bsard_db_path: Path,
    t03_spans_path: Path | None,
) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")
    nodes, my_documents, _ = load_azuredi_corpus(azuredi_dir, pdf_document_map_csv)
    nodes = link_corpus_to_bsard(nodes, bsard_db_path)

    by_doc: dict[int, list[AzureNode]] = defaultdict(list)
    for n in nodes:
        by_doc[n.doc_id].append(n)

    print("\nBSARD coverage per document:")
    for doc_id, group in sorted(by_doc.items()):
        n_total = len(group)
        n_covered = sum(1 for n in group if n.bsard_ids)
        n_content_covered = sum(
            1 for n in group if not n.is_header and n.bsard_ids
        )
        n_content = sum(1 for n in group if not n.is_header)
        bsard_set = {b for n in group for b in n.bsard_ids}
        per_node = [len(n.bsard_ids) for n in group if n.bsard_ids]
        avg_per_node = sum(per_node) / len(per_node) if per_node else 0.0
        title = my_documents.get(doc_id, {}).get("document_title", "?")
        print(f"\n  doc_id={doc_id}  title={title[:60]!r}")
        print(f"    nodes total / with bsard_ids:    {n_total} / {n_covered}  ({n_covered/n_total:.1%})")
        print(f"    content nodes (is_header=False): {n_content_covered} / {n_content}  ({n_content_covered/max(n_content,1):.1%})")
        print(f"    distinct bsard_ids reached:      {len(bsard_set)}")
        print(f"    mean bsard_ids per covered node: {avg_per_node:.2f}")
        if t03_spans_path and t03_spans_path.exists():
            payload = json.loads(t03_spans_path.read_text(encoding="utf-8"))
            t03_bsard = {int(s["bsard_id"]) for s in payload.get("spans", [])}
            overlap = bsard_set & t03_bsard
            t03_only = t03_bsard - bsard_set
            t04_only = bsard_set - t03_bsard
            print(f"    T03 article_spans bsard_ids:     {len(t03_bsard)}")
            print(f"    T03 & T04 overlap:               {len(overlap)}")
            print(f"    T03 only (missed by T04):        {len(t03_only)}")
            print(f"    T04 only (extra over T03):       {len(t04_only)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m arm2_metadata.bsard_link",
        description="Link AzureDI nodes to BSARD bsard_ids and inspect coverage.",
    )
    here = Path(__file__).resolve()
    parser.add_argument(
        "--azuredi", type=Path,
        default=here.parents[2] / "azuredi",
    )
    parser.add_argument(
        "--pdf-map", type=Path,
        default=here.parents[3] / "RQ2_T02_DATA_LOADER" / "data" / "csv" / "pdf_document_map.csv",
    )
    parser.add_argument(
        "--db", type=Path,
        default=here.parents[3] / "dataset_creation_output" / "bsard_corpus.db",
    )
    parser.add_argument(
        "--t03-spans", type=Path,
        default=here.parents[3]
                / "RQ2_T03_ARM1_NAIVE" / "data" / "2004_05_27_2004A27101"
                / "article_spans.json",
        help="Optional T03 article_spans.json for coverage comparison.",
    )
    parser.add_argument("--inspect", action="store_true", required=True)
    args = parser.parse_args(argv)

    return _inspect(args.azuredi, args.pdf_map, args.db, args.t03_spans)


if __name__ == "__main__":
    sys.exit(main())
