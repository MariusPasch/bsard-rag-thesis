"""ToC tree explorer for T05 (PageIndex / 2B) — read-only hierarchy view.

Renders the LAW → CHAPTER → ARTICLE structure of a T05 tree JSON
(``RQ2_T05_ARM2_PAGEINDEX/data/trees/doc_<id>.json``) inside a Streamlit
expander, with visited-node markers driven by the active query's trace
``summary``:

  * Selected laws       → 🟦 prefix
  * Selected chapters   → 🟧 prefix
  * Final candidate arts → 🟢 prefix (+ bold title)

Read-only on first ship: clicking a node does not change the PDF
selection. The point is to make the navigator's path visible so users
can see "the LLM looked here, picked this chapter, and chose / failed
to choose these articles."
"""

from __future__ import annotations

import streamlit as st


_LAW_MARK = "🟦 "
_CHAPTER_MARK = "🟧 "
_ARTICLE_MARK = "🟢 "


def _truncate(s: str, limit: int = 90) -> str:
    s = (s or "").strip()
    return s if len(s) <= limit else s[: limit - 1] + "…"


def render_tree_explorer(
    tree_data: dict | None,
    trace_summary: dict | None,
    expanded: bool = False,
) -> None:
    """Render the T05 ToC tree as nested expanders.

    Args:
        tree_data:     Parsed tree JSON. Top level has ``manifest`` + ``tree``.
                       Pass ``None`` if no tree is available — emits a hint.
        trace_summary: ``trace.summary`` dict from the active query's result.
                       Used to mark visited / selected nodes. Pass ``None`` for
                       a plain hierarchy view.
        expanded:      Default expanded-state of the outer panel.
    """
    if not tree_data:
        return

    tree = tree_data.get("tree") or tree_data
    if not isinstance(tree, dict):
        return

    selected_laws = set((trace_summary or {}).get("selected_law_ids") or [])
    selected_chapters = set((trace_summary or {}).get("selected_chapter_ids") or [])
    final_articles = set((trace_summary or {}).get("candidate_article_ids") or [])

    n_chapters = len(tree.get("sub_nodes") or [])
    n_articles = sum(
        len(ch.get("sub_nodes") or []) for ch in tree.get("sub_nodes") or []
    )

    with st.expander(
        f"ToC tree · {n_chapters} chapters · {n_articles} articles",
        expanded=expanded,
    ):
        # ── Law node header ──────────────────────────────────────────────
        law_id = tree.get("node_id", "?")
        law_title = tree.get("title", law_id)
        law_marker = _LAW_MARK if law_id in selected_laws else "·  "
        st.markdown(f"{law_marker}**{_truncate(law_title, 110)}**")
        meta = tree.get("metadata") or {}
        sub_bits: list[str] = []
        if meta.get("jurisdiction"):
            sub_bits.append(f"`{meta['jurisdiction']}`")
        if meta.get("issuing_authority"):
            sub_bits.append(_truncate(meta["issuing_authority"], 60))
        if meta.get("document_number"):
            sub_bits.append(f"#{meta['document_number']}")
        if sub_bits:
            st.caption(" · ".join(sub_bits))

        st.divider()

        # ── Chapters ─────────────────────────────────────────────────────
        for ch in tree.get("sub_nodes") or []:
            ch_id = ch.get("node_id", "?")
            ch_title = ch.get("title", ch_id)
            ch_summary = ch.get("summary", "")
            ch_articles = ch.get("sub_nodes") or []
            visited_ch = ch_id in selected_chapters
            n_final_in_ch = sum(
                1 for a in ch_articles if a.get("node_id") in final_articles
            )

            mark = _CHAPTER_MARK if visited_ch else "·  "
            label_bits = [f"{mark}**{_truncate(ch_title, 80)}**"]
            label_bits.append(f"({len(ch_articles)} arts)")
            if n_final_in_ch:
                label_bits.append(f"· _{n_final_in_ch} final_")

            # Auto-expand visited chapters so the user immediately sees
            # which articles the LLM was choosing between.
            with st.expander(" ".join(label_bits), expanded=visited_ch):
                if ch_summary:
                    st.caption(_truncate(ch_summary, 200))
                for art in ch_articles:
                    art_id = art.get("node_id", "?")
                    art_title = art.get("title", art_id)
                    art_summary = art.get("summary", "")
                    bs_id = art.get("bsard_id")
                    is_final = art_id in final_articles
                    art_mark = _ARTICLE_MARK if is_final else "    "
                    title_md = (
                        f"{art_mark}**{art_title}**" if is_final
                        else f"{art_mark}{art_title}"
                    )
                    bs_label = f" · BSARD {bs_id}" if bs_id is not None else ""
                    st.markdown(f"{title_md}{bs_label}")
                    if art_summary:
                        st.caption(_truncate(art_summary, 160))


# ── Arm2C deep-tree explorer (recursive, depth-7) ────────────────────────────
#
# T05's render_tree_explorer above walks exactly LAW → chapter → article. The
# Arm2C deep tree (RQ2_T04_ARM2_METADATA/data/_arm2c/<stem>/deep_tree.json) is
# recursive up to depth 7 (LAW → LIVRE → TITRE → CHAPITRE → SECTION →
# SOUS-SECTION → ARTICLE), so it needs an arbitrary-depth renderer.
#
# Highlight model (driven by trace.summary written by export_for_t08.py):
#   * node_id ∈ visited_node_ids   → 🔍 marker + auto-expand along descent path
#   * leaf with node_id ∈ selected_node_ids  → 🟢 marker + bold
#   * leaf with bsard_id ∈ selected_bsard_ids → 🟢 marker + bold
# Read-only — clicks do not change the PDF selection.

_VISITED_MARK = "🔍 "
_SELECTED_MARK = "🟢 "


def _node_is_leaf(node: dict) -> bool:
    return node.get("bsard_id") is not None or not (node.get("sub_nodes") or [])


def _render_deep_node(
    node: dict,
    visited: set[str],
    selected_nodes: set[str],
    selected_bsard: set[int],
    depth: int,
) -> None:
    """Recursively render one deep-tree node and its descendants."""
    node_id = str(node.get("node_id", ""))
    title = node.get("title", node_id) or node_id
    meta = node.get("metadata") or {}
    on_path = node_id in visited
    children = node.get("sub_nodes") or []

    bs_id = node.get("bsard_id")
    is_leaf = _node_is_leaf(node)
    is_selected = (
        node_id in selected_nodes
        or (bs_id is not None and int(bs_id) in selected_bsard)
    )

    if is_leaf:
        # Leaves render inline (no nested expander needed).
        if is_selected:
            marker = _SELECTED_MARK
            title_md = f"{marker}**{_truncate(title, 90)}**"
        else:
            marker = _VISITED_MARK if on_path else "    "
            title_md = f"{marker}{_truncate(title, 90)}"
        bs_label = f" · BSARD {bs_id}" if bs_id is not None else ""
        st.markdown(f"{title_md}{bs_label}")
        summary = meta.get("content_summary_en") or node.get("summary") or ""
        if summary:
            st.caption(_truncate(summary, 160))
        return

    # Internal node → expander. Auto-expand when it lies on the descent path.
    marker = _VISITED_MARK if on_path else "·  "
    art_count = meta.get("article_count")
    count_label = f" · {art_count} arts" if art_count else f" · {len(children)} children"
    label = f"{marker}{_truncate(title, 80)}{count_label}"
    with st.expander(label, expanded=on_path):
        summary = meta.get("level_summary_en") or node.get("summary") or ""
        if summary:
            st.caption(_truncate(summary, 200))
        for child in children:
            _render_deep_node(
                child, visited, selected_nodes, selected_bsard, depth + 1
            )


def render_deep_tree_explorer(
    tree_data: dict | None,
    trace_summary: dict | None,
    expanded: bool = False,
) -> None:
    """Render the Arm2C depth-7 deep tree as nested expanders.

    Args:
        tree_data:     Parsed deep_tree.json. Top level has ``manifest`` + ``tree``.
        trace_summary: ``trace.summary`` from the active query — supplies
                       ``visited_node_ids`` / ``selected_node_ids`` /
                       ``selected_bsard_ids`` used to mark the descent path.
        expanded:      Default expanded-state of the outer panel.
    """
    if not tree_data:
        return
    tree = tree_data.get("tree") or tree_data
    if not isinstance(tree, dict):
        return

    summary = trace_summary or {}
    visited = {str(x) for x in (summary.get("visited_node_ids") or [])}
    selected_nodes = {str(x) for x in (summary.get("selected_node_ids") or [])}
    selected_bsard = {
        int(x) for x in (summary.get("selected_bsard_ids") or [])
        if x is not None
    }

    manifest = tree_data.get("manifest") or {}
    n_articles = manifest.get("n_articles")
    max_depth = manifest.get("max_depth")
    hdr_bits = ["Deep tree"]
    if n_articles:
        hdr_bits.append(f"{n_articles} articles")
    if max_depth:
        hdr_bits.append(f"depth {max_depth}")
    if visited:
        hdr_bits.append(f"{len(visited)} nodes visited")

    with st.expander(" · ".join(hdr_bits), expanded=expanded):
        _render_deep_node(tree, visited, selected_nodes, selected_bsard, depth=0)
