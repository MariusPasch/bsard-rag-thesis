"""Right-hand text inspection panels (question, selected item, GT articles, trace/graph)."""

from __future__ import annotations

import streamlit as st

from visualization.region_adapter import HighlightRegion
from visualization.trace_renderer import (
    format_arm2c_trace,
    format_pageindex_trace,
    format_t04_trace,
    format_trace,
)


def render_selected_panel(region: HighlightRegion | None) -> None:
    """Show metadata and text for the currently selected HighlightRegion."""
    st.markdown("#### Selected")
    if region is None:
        st.caption("Click an item in the sidebar or on the PDF to inspect it.")
        return

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"**ID:** `{region.region_id}`")
        if region.score is not None:
            st.markdown(f"**Score:** {region.score:.4f}")
        if region.rank is not None:
            st.markdown(f"**Rank:** {region.rank}")
    with col_b:
        st.markdown(f"**Arm:** `{region.arm}`")
        st.markdown(f"**Chars:** {region.start_char}–{region.end_char}")
        if region.start_token is not None:
            st.markdown(f"**Tokens:** {region.start_token}–{region.end_token}")

    meta = region.metadata or {}
    if meta.get("bsard_ids"):
        st.markdown(f"**BSARD IDs:** {meta['bsard_ids']}")
    if meta.get("bsard_id") is not None:
        st.markdown(f"**BSARD ID:** {meta['bsard_id']}")
    if meta.get("chapter"):
        st.markdown(f"**Chapter:** {meta['chapter']}")
    if meta.get("unresolved"):
        st.warning("Position could not be resolved — showing whole-page fallback.")

    with st.expander("Full text", expanded=True):
        st.text(region.text or "_(empty)_")


def render_question_panel(
    query_text: str | None,
    relevant_article_ids: list[int] | None,
    query_id: str | None = None,
    gt_article_data: dict | None = None,
) -> None:
    """Show question text, question id, and ground-truth BSARD ids with article numbers."""
    header = "#### Question"
    if query_id:
        header += f" {query_id}"
    st.markdown(header)
    if not query_text:
        st.caption("No question active.")
        return

    st.markdown(query_text)

    if relevant_article_ids:
        gt_article_data = gt_article_data or {}
        parts: list[str] = []
        for bs in relevant_article_ids:
            try:
                key_int = int(bs)
            except (TypeError, ValueError):
                key_int = bs
            entry = gt_article_data.get(key_int) or gt_article_data.get(str(key_int))
            art_num = (entry or {}).get("article_number", "") if entry else ""
            if art_num:
                parts.append(f"`{bs}` (Art. {art_num})")
            else:
                parts.append(f"`{bs}`")
        st.markdown("**GT BSARD IDs:** " + ", ".join(parts))


def render_gt_panel(
    gt_regions: list[HighlightRegion],
    selected_id: str | None = None,
    ranked_items: list[dict] | None = None,
    gt_article_data: dict | None = None,
) -> str | None:
    """Show clickable ground-truth article buttons with their associated retrieved chunks.

    Returns the region_id of a clicked item (GT article or chunk), or None.
    """
    if not gt_regions:
        return None
    st.markdown("#### Ground-truth articles")
    new_selected: str | None = None

    ranked_items = ranked_items or []
    gt_article_data = gt_article_data or {}

    for reg in gt_regions:
        bs_id = reg.metadata.get("bsard_id")
        if bs_id is None:
            # legacy fallback — region_id is "gt_<id>"
            bs_id = reg.region_id.removeprefix("gt_")
        try:
            bs_int = int(bs_id)
        except (TypeError, ValueError):
            bs_int = bs_id

        is_sel = reg.region_id == selected_id
        prefix = "▶ " if is_sel else "   "

        db_entry = gt_article_data.get(bs_int) or gt_article_data.get(str(bs_int))
        art_number = db_entry.get("article_number", "") if db_entry else ""
        btn_label = (
            f"{prefix}Art. {art_number} (BSARD {bs_int})"
            if art_number else f"{prefix}BSARD {bs_int}"
        )

        if st.button(btn_label, key=f"gt_btn_{reg.region_id}", use_container_width=True):
            new_selected = reg.region_id
        if is_sel or new_selected == reg.region_id:
            with st.expander("Text", expanded=True):
                st.text(reg.text or "_(no text)_")

        if db_entry:
            law_title = db_entry.get("law_title", "")
            header = (
                f"Art. {art_number} — {law_title}"
                if art_number or law_title else f"BSARD {bs_int}"
            )
            with st.expander(f"BSARD: {header}", expanded=False):
                st.text(db_entry["text"] or "_(no text in DB)_")

        # Show chunks from ranked_items that contain this article
        art_chunks = [
            item for item in ranked_items
            if bs_int in [
                int(b) for b in (
                    item.get("bsard_ids") or item.get("metadata", {}).get("bsard_ids") or []
                )
            ]
        ]
        if art_chunks:
            with st.expander(
                f"Retrieved chunks ({len(art_chunks)})", expanded=False
            ):
                for item in art_chunks:
                    chunk_id = item.get("id", "?")
                    rank = item.get("rank") or next(
                        (i + 1 for i, r in enumerate(ranked_items) if r.get("id") == chunk_id), None
                    )
                    rank_label = f"#{rank}" if rank else "#?"
                    is_chunk_sel = chunk_id == selected_id
                    chunk_prefix = "▶ " if is_chunk_sel else "   "
                    if st.button(
                        f"{chunk_prefix}  {rank_label} {chunk_id}",
                        key=f"gt_chunk_btn_{chunk_id}_{bs_int}",
                        use_container_width=True,
                    ):
                        new_selected = chunk_id
        else:
            st.caption(f"  No retrieved chunks for BSARD {bs_int}")

    return new_selected


def render_trace_panel(trace: dict | None, method: str) -> None:
    """Show arm-specific retrieval trace.

    PageIndex (2B): full LLM navigation trace.
    Metadata (2A / arm2_metadata): variant, unit, boosts, query signals.
    """
    if not trace:
        return
    if method.startswith("2B") or method.startswith("arm2_pageindex"):
        st.markdown("#### Navigation trace (2B)")
        st.markdown(format_pageindex_trace(trace))
    elif method.startswith("2A") or method.startswith("arm2_metadata"):
        st.markdown("#### Retrieval signals (2A)")
        st.markdown(format_t04_trace(trace))
    elif method.startswith("2C"):
        st.markdown("#### ReAct navigation trace (2C)")
        st.markdown(format_arm2c_trace(trace))


def render_right_panel(
    selected_region: HighlightRegion | None,
    query_text: str | None = None,
    relevant_article_ids: list[int] | None = None,
    gt_regions: list[HighlightRegion] | None = None,
    trace: dict | None = None,
    ranked_items: list[dict] | None = None,
    method: str = "",
    mode: str = "review",
    gt_article_data: dict | None = None,
    query_id: str | None = None,
) -> None:
    """Orchestrate all right-panel sections for the current mode."""
    if mode == "retrieval":
        render_question_panel(
            query_text, relevant_article_ids,
            query_id=query_id, gt_article_data=gt_article_data,
        )
        st.divider()

    render_selected_panel(selected_region)

    if mode == "retrieval":
        st.divider()
        current_sel = st.session_state.get("selected_id")
        gt_click = render_gt_panel(gt_regions or [], selected_id=current_sel, ranked_items=ranked_items, gt_article_data=gt_article_data)
        if gt_click:
            st.session_state["selected_id"] = gt_click
        render_trace_panel(trace, method)
