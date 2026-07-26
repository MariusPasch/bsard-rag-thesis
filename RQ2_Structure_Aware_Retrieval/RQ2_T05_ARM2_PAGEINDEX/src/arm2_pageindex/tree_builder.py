"""Build a deterministic ToC tree for Arm 2B navigation.

The tree mirrors the corpus's structural hierarchy
(Belgian Statutory Corpus -> Law -> Chapter -> Article) without any
LLM calls at build time, so it is fast, reproducible, and free of
model drift.

Inputs (from T04):
  * ``nodes``: ``list[AzureNode]``, post-link
    (``arm2_metadata.bsard_link.link_corpus_to_bsard``).
  * ``articles``: ``list[Article]``, post-aggregation
    (``arm2_metadata.enricher.build_articles_from_nodes``).
  * ``my_documents``: ``{doc_id: meta_dict}`` from
    ``arm2_metadata.azuredi_loader.load_azuredi_corpus``.

Per-doc tree shape:
  * 3 levels (Law -> Chapter -> Article) when at least one article has
    a header ancestor in the AzureDI structure.
  * 2 levels (Law -> Article) otherwise. Recorded as
    ``manifest['chapter_derivable']`` so callers can branch on it.

Cache layout (PLAN_per_pdf_pipeline §B.6): ``<cache_dir>/<doc_id>/tree.json``
where ``<doc_id>`` is the BSARD PDF stem. The integer AzureDI ``DocumentId``
is recorded inside the manifest header as ``azure_document_id``. Each file
pins ``pdf_sha256`` and ``tree_builder_version`` so a new AzureDI
dump or a builder change cleanly invalidates the cache.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Iterator

if TYPE_CHECKING:
    # AzureNode and Article are referenced only as type annotations
    # (which `from __future__ import annotations` keeps as strings).
    # Putting the imports under TYPE_CHECKING means tree_builder.py can
    # be imported on a host without T02/T04 installed - useful on the
    # Azure side, where the tree is loaded from JSON and never built.
    from arm2_metadata.azuredi_loader import AzureNode
    from data_loader.models import Article

TREE_BUILDER_VERSION = 1
SUMMARY_MAX_CHARS = 240

logger = logging.getLogger(__name__)


# ── Tree node ────────────────────────────────────────────────────────────────


@dataclass
class ToCNode:
    node_id: str
    title: str
    summary: str = ""
    metadata: dict = field(default_factory=dict)
    sub_nodes: list["ToCNode"] = field(default_factory=list)
    article_id: str | None = None
    bsard_id: int | None = None
    text: str | None = None
    cross_references: list[str] = field(default_factory=list)

    @property
    def is_leaf(self) -> bool:
        return self.article_id is not None

    def to_dict(self) -> dict:
        d: dict = {"node_id": self.node_id, "title": self.title}
        if self.summary:
            d["summary"] = self.summary
        if self.metadata:
            d["metadata"] = self.metadata
        if self.is_leaf:
            d["article_id"] = self.article_id
            d["bsard_id"] = self.bsard_id
            d["text"] = self.text
            if self.cross_references:
                d["cross_references"] = self.cross_references
        if self.sub_nodes:
            d["sub_nodes"] = [n.to_dict() for n in self.sub_nodes]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ToCNode":
        return cls(
            node_id=d["node_id"],
            title=d["title"],
            summary=d.get("summary", ""),
            metadata=d.get("metadata", {}),
            sub_nodes=[cls.from_dict(c) for c in d.get("sub_nodes", [])],
            article_id=d.get("article_id"),
            bsard_id=d.get("bsard_id"),
            text=d.get("text"),
            cross_references=d.get("cross_references", []),
        )

    def to_prompt_dict(
        self, depth: int = 0, include_summary: bool = True,
    ) -> dict:
        """Compact form for LLM prompts. Drops leaf text, crops summary.

        ``depth`` controls how many sub-node levels are inlined:
          ``0`` -> only this node, with ``sub_node_count``.
          ``k`` -> recurse k levels.
        """
        d: dict = {"node_id": self.node_id, "title": self.title}
        if include_summary and self.summary:
            d["summary"] = self.summary
        if self.metadata:
            keep = {
                k: v for k, v in self.metadata.items()
                if k in (
                    "article_count", "jurisdiction", "issuing_authority",
                    "article_number", "defined_terms", "header_chain",
                )
            }
            if keep:
                d["metadata"] = keep
        if self.is_leaf:
            return d
        if depth > 0:
            d["sub_nodes"] = [
                c.to_prompt_dict(depth - 1, include_summary)
                for c in self.sub_nodes
            ]
        else:
            d["sub_node_count"] = len(self.sub_nodes)
        return d


# ── Helpers ──────────────────────────────────────────────────────────────────


_SENT_SPLIT_RE = re.compile(r"(?<=[\.\!\?])\s+(?=[A-ZÀ-Ý])")
_NODE_ID_BAD_RE = re.compile(r"[^A-Za-z0-9]+")
_MD_HEADER_PREFIX_RE = re.compile(r"^\s*#+\s*")
_HEADER_LABEL_MAX_CHARS = 180


def first_sentence(
    text: str,
    max_chars: int = SUMMARY_MAX_CHARS,
    min_chars: int = 80,
) -> str:
    """Compact summary line. Accumulates sentences until ``min_chars`` is
    reached so a leading abbreviation like ``"Art."`` does not strand the
    summary at 4 characters."""
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text.replace("\r", " ").replace("\n", " ").strip())
    parts = _SENT_SPLIT_RE.split(text)
    head = parts[0]
    i = 1
    while len(head) < min_chars and i < len(parts):
        head = f"{head} {parts[i]}"
        i += 1
    if len(head) > max_chars:
        head = head[:max_chars].rsplit(" ", 1)[0] + "…"
    return head


def _sanitize_id(s: str, max_len: int = 40) -> str:
    safe = _NODE_ID_BAD_RE.sub("_", s).strip("_")
    return safe[:max_len] or "X"


def _compute_pdf_sha256(pdf_path: str | Path | None) -> str | None:
    if not pdf_path:
        return None
    p = Path(pdf_path)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ── Header-chain derivation ──────────────────────────────────────────────────


def _members_for_articles(
    nodes: Iterable[AzureNode],
) -> dict[tuple[int, int], list[AzureNode]]:
    """Index content nodes by (doc_id, bsard_id) — same key the article
    aggregator uses. Headers are excluded (no body text)."""
    out: dict[tuple[int, int], list[AzureNode]] = {}
    for n in nodes:
        if n.is_header or not n.bsard_ids:
            continue
        for bid in n.bsard_ids:
            out.setdefault((n.doc_id, bid), []).append(n)
    return out


def _header_label(node: AzureNode) -> str:
    """Clean a header node's ``page_content`` into a structural label.

    AzureDI emits header nodes whose ``page_content`` carries the
    structural label prefixed with markdown header markers, e.g.
    ``"#### CHAPITRE II. - Publicité…"``. The ``node_source`` field is
    uniformly the document title in this dump (per RQ2 §4.3) and is not
    usable. We strip the leading ``#`` markers, collapse whitespace, and
    truncate; an empty result means the node carries no usable label.
    """
    text = (node.page_content or "").strip()
    if not text:
        return ""
    text = _MD_HEADER_PREFIX_RE.sub("", text, count=1)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > _HEADER_LABEL_MAX_CHARS:
        text = text[:_HEADER_LABEL_MAX_CHARS].rsplit(" ", 1)[0] + "…"
    return text


def _header_chain_for_member(
    member: AzureNode,
    nodes_by_id: dict[tuple[int, int], AzureNode],
    max_depth: int = 16,
) -> list[str]:
    """Walk parent_id chain from ``member`` and collect header labels.

    Returned list runs closest-header-first. Duplicate consecutive labels
    are collapsed (some AzureDI dumps stack the same label multiple
    times in the parent chain). Empty list = the article has no header
    ancestor.
    """
    chain: list[str] = []
    cur = member.parent_id
    depth = 0
    while cur is not None and depth < max_depth:
        parent = nodes_by_id.get((member.doc_id, cur))
        if parent is None:
            break
        if parent.is_header:
            label = _header_label(parent)
            if label and (not chain or chain[-1] != label):
                chain.append(label)
        cur = parent.parent_id
        depth += 1
    return chain


# ── Builders ─────────────────────────────────────────────────────────────────


def _build_article_node(art: Article, header_chain: list[str]) -> ToCNode:
    title = (
        f"Article {art.article_number}"
        if art.article_number
        else f"Article ({art.article_id})"
    )
    metadata: dict = {}
    if art.article_number:
        metadata["article_number"] = art.article_number
    if art.defined_terms:
        metadata["defined_terms"] = list(art.defined_terms)[:5]
    if header_chain:
        metadata["header_chain"] = list(header_chain)
    return ToCNode(
        node_id=f"ART_{art.article_id}",
        title=title,
        summary=first_sentence(art.text),
        metadata=metadata,
        article_id=art.article_id,
        bsard_id=art.bsard_id,
        text=art.text,
        cross_references=list(art.cross_references) if art.cross_references else [],
    )


def build_law_tree(
    *,
    doc_id: int,
    doc_meta: dict,
    articles_for_doc: list[Article],
    nodes_for_doc: list[AzureNode],
) -> tuple[ToCNode, bool]:
    """Build a Law subtree (Law -> Chapter -> Article).

    Returns ``(law_node, chapter_derivable)``. When ``chapter_derivable``
    is False, the tree has two levels (Law -> Article) and chapter
    grouping is empty.
    """
    title = doc_meta.get("document_title") or f"Document {doc_id}"
    summary = first_sentence(doc_meta.get("summary", ""))
    law_metadata = {
        k: v for k, v in {
            "document_id": doc_id,
            "jurisdiction": doc_meta.get("jurisdiction"),
            "issuing_authority": doc_meta.get("issuing_authority"),
            "document_number": doc_meta.get("document_number"),
            "entry_into_force": doc_meta.get("entry_into_force"),
            "article_count": len(articles_for_doc),
        }.items()
        if v not in (None, "", [])
    }

    law_node = ToCNode(
        node_id=f"LAW_{doc_id}",
        title=title,
        summary=summary,
        metadata=law_metadata,
    )

    if not articles_for_doc:
        return law_node, False

    nodes_by_id = {(n.doc_id, n.node_id): n for n in nodes_for_doc}
    members_by_key = _members_for_articles(nodes_for_doc)

    article_chapter: dict[str, tuple[str, list[str]]] = {}
    for art in articles_for_doc:
        key = (art.document_id, art.bsard_id) if art.bsard_id is not None else None
        members = list(members_by_key.get(key, [])) if key else []
        members.sort(key=lambda n: (n.page_number, n.node_id))
        chain = _header_chain_for_member(members[0], nodes_by_id) if members else []
        chapter_label = chain[0] if chain else ""
        article_chapter[art.article_id] = (chapter_label, chain)

    chapter_derivable = any(label for label, _ in article_chapter.values())

    if not chapter_derivable:
        for art in articles_for_doc:
            law_node.sub_nodes.append(
                _build_article_node(art, article_chapter[art.article_id][1])
            )
        return law_node, False

    grouped: "OrderedDict[str, list[Article]]" = OrderedDict()
    for art in articles_for_doc:
        label, _ = article_chapter[art.article_id]
        grouped.setdefault(label or "_ungrouped", []).append(art)

    for chapter_key, chapter_articles in grouped.items():
        is_real = chapter_key != "_ungrouped"
        ch_title = chapter_key if is_real else "Articles non rattachés à un chapitre"
        first_no = chapter_articles[0].article_number or "?"
        last_no = chapter_articles[-1].article_number or "?"
        ch_summary = (
            f"{len(chapter_articles)} articles "
            f"(de {first_no} à {last_no})"
        )
        chapter_node = ToCNode(
            node_id=f"LAW_{doc_id}_CH_{_sanitize_id(chapter_key)}",
            title=ch_title,
            summary=ch_summary,
            metadata={"article_count": len(chapter_articles)},
        )
        for art in chapter_articles:
            chapter_node.sub_nodes.append(
                _build_article_node(art, article_chapter[art.article_id][1])
            )
        law_node.sub_nodes.append(chapter_node)
    return law_node, True


def compose_corpus_tree(law_subtrees: list[ToCNode]) -> ToCNode:
    """Wrap per-doc Law subtrees under a single corpus root."""
    n_articles = sum(_count_leaves(law) for law in law_subtrees)
    return ToCNode(
        node_id="ROOT",
        title="Belgian Statutory Corpus",
        summary=(
            f"{n_articles} articles across {len(law_subtrees)} legislative documents"
        ),
        metadata={
            "document_count": len(law_subtrees),
            "total_articles": n_articles,
        },
        sub_nodes=list(law_subtrees),
    )


# ── Tree utilities ───────────────────────────────────────────────────────────


def find_node(root: ToCNode, node_id: str) -> ToCNode | None:
    if root.node_id == node_id:
        return root
    for child in root.sub_nodes:
        hit = find_node(child, node_id)
        if hit is not None:
            return hit
    return None


def iter_leaves(root: ToCNode) -> Iterator[ToCNode]:
    if root.is_leaf:
        yield root
        return
    for child in root.sub_nodes:
        yield from iter_leaves(child)


def _count_leaves(node: ToCNode) -> int:
    if node.is_leaf:
        return 1
    return sum(_count_leaves(c) for c in node.sub_nodes)


def tree_stats(law_node: ToCNode) -> dict:
    n_chapters = sum(1 for c in law_node.sub_nodes if not c.is_leaf)
    n_articles = _count_leaves(law_node)
    bsard_filled = sum(
        1 for leaf in iter_leaves(law_node) if leaf.bsard_id is not None
    )
    return {
        "n_chapters": n_chapters,
        "n_articles": n_articles,
        "bsard_id_coverage": (bsard_filled / n_articles) if n_articles else 0.0,
    }


# ── Persistence ──────────────────────────────────────────────────────────────


def save_law_tree(
    law_node: ToCNode,
    *,
    doc_id: str,
    azure_document_id: int,
    pdf_filename: str | None,
    pdf_path: str | Path | None,
    chapter_derivable: bool,
    dest_dir: Path,
) -> Path:
    """Write the law tree to ``<dest_dir>/<doc_id>/tree.json``.

    The header records the BSARD PDF stem as ``doc_id`` and the AzureDI
    integer as ``azure_document_id`` (PLAN_per_pdf_pipeline §B.6).
    """
    dest_dir = Path(dest_dir)
    out_dir = dest_dir / doc_id
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "manifest": {
            "tree_builder_version": TREE_BUILDER_VERSION,
            "doc_id": doc_id,
            "azure_document_id": azure_document_id,
            "pdf_filename": pdf_filename,
            "pdf_sha256": _compute_pdf_sha256(pdf_path),
            "built_at": int(time.time()),
            "chapter_derivable": chapter_derivable,
            **tree_stats(law_node),
        },
        "tree": law_node.to_dict(),
    }
    out = out_dir / "tree.json"
    out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(
        "Wrote %s - %d articles in %d chapters (chapter_derivable=%s)",
        out,
        payload["manifest"]["n_articles"],
        payload["manifest"]["n_chapters"],
        chapter_derivable,
    )
    return out


def load_law_tree(path: Path) -> tuple[ToCNode, dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return ToCNode.from_dict(payload["tree"]), payload["manifest"]


def is_cache_fresh(
    cache_path: Path,
    *,
    pdf_path: str | Path | None,
) -> bool:
    if not Path(cache_path).exists():
        return False
    try:
        manifest = json.loads(
            Path(cache_path).read_text(encoding="utf-8")
        )["manifest"]
    except (KeyError, json.JSONDecodeError):
        return False
    if manifest.get("tree_builder_version") != TREE_BUILDER_VERSION:
        return False
    if pdf_path:
        cached = manifest.get("pdf_sha256")
        if cached and cached != _compute_pdf_sha256(pdf_path):
            return False
    return True
