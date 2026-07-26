"""Arm-2C deep-tree builder.

Builds a *full multi-level* ToC tree per PDF from the raw AzureDI nodes, nesting
articles by their COMPLETE parent_id->is_header chain (LIVRE > TITRE > CHAPITRE >
Section > Sous-section > Article) instead of collapsing to the single closest
header the way T05's flat builder does (tree_builder.build_law_tree, chain[0]).

Why the full parent-walk chain and not path_to_item:
  The 2026-06-04 probe (experiments/arm2c probe) showed path_to_item recovers 0%
  of parent_id orphans and is strictly shallower (caps ~Partie/Titre, drops
  Chapitre/Section). The parent-walk header_chain is the richer raw-AzureDI
  signal; T05's only mistake was collapsing it. See PROJECT memory arm2c.

Orphans (articles with no is_header ancestor, ~7-8%, unrecoverable by any signal)
are routed to an explicit catch-all branch under the law root so they stay
reachable by descent (a hierarchical navigator can never reach a truly detached
article otherwise). This is the structural recall ceiling, reported in stats.

Output: experiments/arm2c/data/<doc_id>/deep_tree.json, same ToCNode schema as
T05 tree.json (drop-in for a navigator), with a manifest carrying fan-out/depth
stats. Re-uses T05's ToCNode / header-chain helpers (T05 venv has arm2_pageindex).

Run (from repo root, T05 venv):
    <T05>/.venv/Scripts/python.exe RQ2_T04_ARM2_METADATA/experiments/arm2c/deep_tree_builder.py
    ... --doc-id 1804_03_21_1804032150      # single PDF
    ... --out <dir>                          # alt output root
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

# ── path injection so the script runs under any sibling venv ─────────────────
_HERE = Path(__file__).resolve()
_RQ2 = _HERE.parents[3]  # .../RQ2_Structure_Aware_Retrieval
for _sub in ("RQ2_T04_ARM2_METADATA", "RQ2_T02_DATA_LOADER", "RQ2_T03_ARM1_NAIVE",
             "RQ2_T01_SHARED", "RQ2_T05_ARM2_PAGEINDEX"):
    _p = _RQ2 / _sub / "src"
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from arm2_metadata.azuredi_loader import load_azuredi_corpus
from arm2_metadata.bsard_link import link_corpus_to_bsard
from arm2_metadata.enricher import build_articles_from_nodes, build_definitions_map
from arm2_pageindex.tree_builder import (
    ToCNode,
    _build_article_node,
    _header_label,
    _members_for_articles,
    _sanitize_id,
    first_sentence,
    iter_leaves,
)

DEEP_TREE_BUILDER_VERSION = 1
logger = logging.getLogger("arm2c.deep_tree_builder")


# ── core build ───────────────────────────────────────────────────────────────


def _node_id(doc_id: int, prefix: tuple[str, ...]) -> str:
    """Deterministic, unique, human-legible id for an internal node."""
    last = _sanitize_id(prefix[-1], 24)
    h = hashlib.md5("|".join(prefix).encode("utf-8")).hexdigest()[:6]
    return f"LAW_{doc_id}_{last}_{h}"


def _header_stack_with_nodes(member, nodes_by_id, max_depth: int = 16):
    """Like arm2_pageindex._header_chain_for_member but returns (label, header_node)
    pairs leaf->root, so the builder can copy each header node's native metadata
    (English `level_summary`). Collapses consecutive duplicate labels (keeps the
    closest node)."""
    out = []
    cur = member.parent_id
    depth = 0
    while cur is not None and depth < max_depth:
        parent = nodes_by_id.get((member.doc_id, cur))
        if parent is None:
            break
        if parent.is_header:
            label = _header_label(parent)
            if label and (not out or out[-1][0] != label):
                out.append((label, parent))
        cur = parent.parent_id
        depth += 1
    return out


def _attach_level_summary(node: ToCNode, header_node) -> None:
    """Carry the header node's NATIVE English `level_summary` (a description of
    its subtree) onto an internal tree node, for the navigator's *enriched*
    variant. Language stays native (English) by design — the multilingual LLM
    bridges; see PROJECT memory metadata-enrichment design. Skips empties."""
    ls = (getattr(header_node, "level_summary", "") or "").strip()
    if ls and not ls.startswith("#"):  # '#'-prefixed == echo of the FR label
        node.metadata["level_summary_en"] = ls[:400]


def _build_leaf(art, header_labels: list[str], member) -> ToCNode:
    """Article leaf with the French first-sentence summary (bare) PLUS the member
    node's native English `content_summary` + `keywords` (enriched). Native
    language preserved on purpose."""
    leaf = _build_article_node(art, header_labels)
    if member is not None:
        cs = (getattr(member, "content_summary", "") or "").strip()
        if cs and not cs.startswith(("#", "<")):  # skip FR-label / table echoes
            leaf.metadata["content_summary_en"] = cs[:400]
        kw = (getattr(member, "keywords", "") or "").strip()
        if kw and kw.lower() != "header - no keywords":
            leaf.metadata["keywords_en"] = kw[:200]
    return leaf


def build_deep_law_tree(
    *,
    doc_id: int,
    doc_meta: dict,
    articles_for_doc: list,
    nodes_for_doc: list,
) -> ToCNode:
    """Nest articles under the FULL header chain. Returns the law root."""
    law_node = ToCNode(
        node_id=f"LAW_{doc_id}",
        title=doc_meta.get("document_title") or f"Document {doc_id}",
        summary=first_sentence(doc_meta.get("summary", "")),
        metadata={
            k: v for k, v in {
                "document_id": doc_id,
                "jurisdiction": doc_meta.get("jurisdiction"),
                "issuing_authority": doc_meta.get("issuing_authority"),
                "document_number": doc_meta.get("document_number"),
                "article_count": len(articles_for_doc),
            }.items() if v not in (None, "", [])
        },
    )
    if not articles_for_doc:
        return law_node

    nodes_by_id = {(n.doc_id, n.node_id): n for n in nodes_for_doc}
    members_by_key = _members_for_articles(nodes_for_doc)

    # index of internal nodes by full prefix tuple; () == law root
    index: dict[tuple[str, ...], ToCNode] = {(): law_node}
    orphan_leaves: list[ToCNode] = []

    for art in articles_for_doc:
        key = (art.document_id, art.bsard_id) if art.bsard_id is not None else None
        members = sorted(
            members_by_key.get(key, []),
            key=lambda n: (n.page_number, n.node_id),
        ) if key else []
        member0 = members[0] if members else None
        stack = _header_stack_with_nodes(member0, nodes_by_id) if member0 else []
        leaf = _build_leaf(art, [lab for lab, _ in stack], member0)

        if not stack:                       # empty header stack -> Unfiled
            orphan_leaves.append(leaf)
            continue

        parent = law_node
        prefix: tuple[str, ...] = ()
        for label, header_node in reversed(stack):  # root -> deepest
            prefix = prefix + (label,)
            node = index.get(prefix)
            if node is None:
                node = ToCNode(
                    node_id=_node_id(doc_id, prefix),
                    title=label,
                    metadata={"depth": len(prefix)},
                )
                _attach_level_summary(node, header_node)
                index[prefix] = node
                parent.sub_nodes.append(node)
            parent = node
        parent.sub_nodes.append(leaf)

    # catch-all orphan branch, appended LAST so it never shadows real structure
    if orphan_leaves:
        orphan = ToCNode(
            node_id=f"LAW_{doc_id}_ORPHAN",
            title="Articles non rattachés à un chapitre",
            metadata={"orphan": True, "depth": 1},
        )
        orphan.sub_nodes.extend(orphan_leaves)
        law_node.sub_nodes.append(orphan)

    _annotate(law_node)
    return law_node


def _annotate(node: ToCNode) -> int:
    """Post-order: give each internal node a deterministic summary
    (article_count + article-number span) and return its leaf count."""
    if node.is_leaf:
        return 1
    leaves = list(iter_leaves(node))
    n = len(leaves)
    nums = [lf.metadata.get("article_number") for lf in leaves
            if lf.metadata.get("article_number")]
    span = f" (de {nums[0]} à {nums[-1]})" if nums else ""
    if not node.summary:  # keep law-root's own summary if present
        node.summary = f"{n} articles{span}"
    node.metadata["article_count"] = n
    for c in node.sub_nodes:
        _annotate(c)
    return n


# ── stats (validate against the header_chain audit) ──────────────────────────


def deep_stats(law_node: ToCNode) -> dict:
    leaves = list(iter_leaves(law_node))
    fanouts, depths = [], []

    def walk(node: ToCNode, depth: int):
        internal = [c for c in node.sub_nodes if not c.is_leaf]
        if node.sub_nodes:  # a decision point (has children at all)
            if not node.is_leaf:
                fanouts.append(len(node.sub_nodes))
        if node.is_leaf:
            depths.append(depth)
            return
        for c in node.sub_nodes:
            walk(c, depth + 1)

    walk(law_node, 0)
    orphan = next((c for c in law_node.sub_nodes
                   if c.metadata.get("orphan")), None)
    n_orphan = len(list(iter_leaves(orphan))) if orphan else 0
    bsard_filled = sum(1 for lf in leaves if lf.bsard_id is not None)
    return {
        "n_articles": len(leaves),
        "n_orphan_articles": n_orphan,
        "orphan_pct": round(100 * n_orphan / max(len(leaves), 1), 1),
        "max_depth": max(depths) if depths else 0,
        "n_internal_nodes": len(fanouts),
        "fanout_mean": round(statistics.mean(fanouts), 2) if fanouts else 0,
        "fanout_median": statistics.median(fanouts) if fanouts else 0,
        "fanout_max": max(fanouts) if fanouts else 0,
        "bsard_id_coverage": round(bsard_filled / max(len(leaves), 1), 3),
    }


def save_deep_tree(law_node: ToCNode, *, doc_id: str, azure_document_id: int,
                   pdf_filename: str | None, dest_dir: Path) -> Path:
    out_dir = Path(dest_dir) / doc_id
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "manifest": {
            "deep_tree_builder_version": DEEP_TREE_BUILDER_VERSION,
            "doc_id": doc_id,
            "azure_document_id": azure_document_id,
            "pdf_filename": pdf_filename,
            "built_at": int(time.time()),
            "structure_source": "parent_walk_header_chain_full",
            **deep_stats(law_node),
        },
        "tree": law_node.to_dict(),
    }
    out = out_dir / "deep_tree.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    return out


# ── CLI ──────────────────────────────────────────────────────────────────────

CURATED_STEMS = {
    "1804_03_21_1804032150", "1867_06_08_1867060850",
    "1967_10_10_1967101055", "1967_10_10_1967101056", "2003_07_17_2013A31614",
}


def main() -> int:
    ap = argparse.ArgumentParser(prog="arm2c-deep-tree-builder")
    ap.add_argument("--azuredi", type=Path,
                    default=_RQ2 / "RQ2_T04_ARM2_METADATA" / "azuredi")
    ap.add_argument("--pdf-map", type=Path,
                    default=_RQ2 / "RQ2_T02_DATA_LOADER" / "data" / "csv" / "pdf_document_map.csv")
    ap.add_argument("--bsard-db", type=Path,
                    # $RQ2_BSARD_DB  →  <RQ2_DATA_DIR or repo>/data/bsard_corpus.db
                    default=Path(os.environ["RQ2_BSARD_DB"]).expanduser()
                    if os.environ.get("RQ2_BSARD_DB")
                    else _RQ2 / "data" / "bsard_corpus.db")
    ap.add_argument("--out", type=Path, default=_HERE.parent / "data")
    ap.add_argument("--doc-id", type=str, default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING,
                        format="%(levelname)s %(name)s - %(message)s")

    print(f"Loading AzureDI from {args.azuredi} ...", flush=True)
    nodes, my_documents, definitions_df = load_azuredi_corpus(args.azuredi, args.pdf_map)
    link_corpus_to_bsard(nodes, args.bsard_db)
    defmap = build_definitions_map(definitions_df)
    articles = build_articles_from_nodes(nodes, my_documents, defmap)
    print(f"  nodes={len(nodes)} articles={len(articles)}\n", flush=True)

    nodes_by_doc = defaultdict(list)
    for n in nodes:
        nodes_by_doc[n.doc_id].append(n)
    arts_by_doc = defaultdict(list)
    for a in articles:
        arts_by_doc[a.document_id].append(a)
    stem_by_doc = {}
    pdfname_by_doc = {}
    for n in nodes:
        if n.pdf_filename and n.doc_id not in stem_by_doc:
            stem_by_doc[n.doc_id] = Path(n.pdf_filename).stem
            pdfname_by_doc[n.doc_id] = n.pdf_filename

    print(f"{'doc_id':>30} {'arts':>5} {'orph%':>6} {'depth':>5} {'fan mean/med/max':>17}  out")
    print("-" * 100)
    for azure_id, arts in sorted(arts_by_doc.items()):
        stem = stem_by_doc.get(azure_id)
        if stem is None or (args.doc_id and stem != args.doc_id) or stem not in CURATED_STEMS:
            continue
        law = build_deep_law_tree(
            doc_id=azure_id, doc_meta=my_documents[azure_id],
            articles_for_doc=arts, nodes_for_doc=nodes_by_doc.get(azure_id, []),
        )
        st = deep_stats(law)
        out = save_deep_tree(law, doc_id=stem, azure_document_id=azure_id,
                             pdf_filename=pdfname_by_doc.get(azure_id), dest_dir=args.out)
        print(f"{stem:>30} {st['n_articles']:>5} {st['orphan_pct']:>6} "
              f"{st['max_depth']:>5} {st['fanout_mean']:>5}/{st['fanout_median']}/{st['fanout_max']:<7} "
              f"{out.parent.name}/deep_tree.json")
    print("-" * 100)
    print(f"\nDeep trees in: {args.out}/<doc_id>/deep_tree.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
