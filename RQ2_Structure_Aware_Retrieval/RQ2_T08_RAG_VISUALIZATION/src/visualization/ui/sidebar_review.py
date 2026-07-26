"""Sidebar panel for Review mode: lists chunks or articles and handles selection."""

from __future__ import annotations

import streamlit as st

from visualization.region_adapter import HighlightRegion


def render_chunk_controls(
    strategy: str,
    window_size: int,
    stride: int,
    max_tokens: int,
) -> tuple[str, int, int, int]:
    """Render chunking strategy controls. Returns updated (strategy, window_size, stride, max_tokens)."""
    strategy = st.selectbox(
        "Strategy",
        ["sliding_window", "recursive"],
        index=0 if strategy == "sliding_window" else 1,
        key="strategy",
    )
    if strategy == "sliding_window":
        window_size = st.number_input("Window size (tokens)", 64, 2048, window_size, 64, key="window_size")
        stride = st.number_input("Stride (tokens)", 16, window_size, min(stride, window_size), 16, key="stride")
        max_tokens = max_tokens
    else:
        max_tokens = st.number_input("Max tokens", 64, 2048, max_tokens, 64, key="max_tokens")
        window_size = window_size
        stride = stride

    return strategy, int(window_size), int(stride), int(max_tokens)


def render_item_list(
    regions: list[HighlightRegion],
    selected_id: str | None,
    page_info: dict[str, int] | None = None,
) -> str | None:
    """Render scrollable item list; return the region_id of the clicked item or None."""
    st.markdown(f"**Items ({len(regions)})**")

    new_selected: str | None = None

    for reg in regions:
        page_hint = ""
        if page_info and reg.region_id in page_info:
            p = page_info[reg.region_id]
            page_hint = f" p.{p + 1}"

        label = f"{reg.region_id}{page_hint}"
        is_selected = reg.region_id == selected_id

        # Highlight the selected item with a ▶ marker
        prefix = "▶ " if is_selected else "   "
        if st.button(f"{prefix}{label}", key=f"btn_{reg.region_id}", use_container_width=True):
            new_selected = reg.region_id

    return new_selected


def render_review_sidebar(
    arm: str,
    regions: list[HighlightRegion],
    selected_id: str | None,
    strategy: str = "sliding_window",
    window_size: int = 512,
    stride: int = 256,
    max_tokens: int = 512,
    page_info: dict[str, int] | None = None,
) -> tuple[str | None, str, int, int, int]:
    """Render the full review-mode sidebar.

    Returns:
        (newly_selected_id, strategy, window_size, stride, max_tokens)
        newly_selected_id is None if no item was clicked this rerun.
    """
    new_selected: str | None = None

    if arm == "arm1":
        st.markdown("### Chunking")
        strategy, window_size, stride, max_tokens = render_chunk_controls(
            strategy, window_size, stride, max_tokens
        )
        rechunk = st.button("Re-chunk", type="primary", use_container_width=True)
        if rechunk:
            # Signal re-chunk by setting a session flag; app.py handles the actual call
            st.session_state["rechunk_requested"] = True

    st.divider()

    if regions:
        new_selected = render_item_list(regions, selected_id, page_info)
    elif arm == "arm1":
        st.caption(
            "No chunks yet — set the strategy / size above and click **Re-chunk** "
            "to generate them. Every chunk will then be overlaid on the PDF "
            "(faintly) so you can inspect coverage and overlap."
        )
    else:
        st.caption(
            "No articles available for this arm. Articles for arm2 are sourced "
            "from the bundle JSON (or live DB lookup) — load one to see them."
        )

    return new_selected, strategy, window_size, stride, max_tokens
