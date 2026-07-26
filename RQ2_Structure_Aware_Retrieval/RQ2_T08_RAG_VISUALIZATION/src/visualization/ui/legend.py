"""Color legend widget for the RAG visualization viewer."""

from __future__ import annotations

import streamlit as st

from visualization.annotation_builder import DEFAULT_COLORS, GRAPHRAG_SOURCE_COLORS


_CATEGORY_LABELS = {
    "gt":             "Ground truth (blue)",
    "retrieved":      "Correctly retrieved (green)",
    "wrong_retrieved":"Wrongly retrieved (red)",
    "retrieved_node_page":       "Correct node — page band (T04)",
    "wrong_retrieved_node_page": "Wrong node — page band (T04)",
    "overlap":        "Overlap GT ∩ Retrieved (teal)",
    "selected":       "Selected item (amber)",
    "chunk":          "Chunk (review mode)",
    "article":        "Article (review mode)",
    "search_hit":     "Search match (yellow)",
    "search_active":  "Active search match (dark yellow)",
}


def render_legend(mode: str, arm: str, graphrag_color_by_source: bool = False) -> None:
    """Render a compact color legend appropriate for the current mode and arm."""
    with st.expander("Legend", expanded=False):
        if mode == "review":
            categories = ["chunk", "article", "selected", "search_hit", "search_active"]
        else:
            categories = ["gt", "retrieved", "wrong_retrieved", "overlap", "search_hit", "search_active"]
            # Show node-page-band swatches only when a T04 2A arm is active —
            # other arms don't emit them.
            if arm.startswith("2A-") or arm.startswith("arm2_metadata"):
                categories.insert(3, "retrieved_node_page")
                categories.insert(4, "wrong_retrieved_node_page")

        for cat in categories:
            color = DEFAULT_COLORS.get(cat, "#9CA3AF")
            label = _CATEGORY_LABELS.get(cat, cat)
            st.markdown(
                f'<span style="display:inline-block;width:16px;height:16px;'
                f'background:{color};border-radius:3px;margin-right:6px;'
                f'vertical-align:middle;"></span> {label}',
                unsafe_allow_html=True,
            )

        # Extra GraphRAG hop-source legend when 2C arm is selected
        if arm.startswith("2C") and graphrag_color_by_source:
            st.markdown("**GraphRAG hop source:**")
            for src, color in GRAPHRAG_SOURCE_COLORS.items():
                st.markdown(
                    f'<span style="display:inline-block;width:16px;height:16px;'
                    f'background:{color};border-radius:3px;margin-right:6px;'
                    f'vertical-align:middle;"></span> {src}',
                    unsafe_allow_html=True,
                )
