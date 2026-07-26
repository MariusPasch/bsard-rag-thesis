"""Three-column Streamlit layout: sidebar | PDF viewer | text panels."""

from __future__ import annotations

import streamlit as st

from visualization.annotation_builder import build_annotations
from visualization.coord_mapper import WordSpan, get_first_page
from visualization.region_adapter import HighlightRegion, text_matches_to_regions
from visualization.ui.legend import render_legend
from visualization.ui.text_panels import render_right_panel


# Human-readable arm names for the selector. Keys are the *internal* arm
# identifiers used everywhere for dispatch (``arm == "arm1"``,
# ``arm.startswith("2C")`` …); only the displayed text changes, so all
# downstream logic keeps working on the stable internal value.
_ARM_DISPLAY_LABELS = {
    "arm1": "ARM1-Naive-Chunking",
    "2A-summary-node": "ARM2A-Metadata-Enrichment-Summary",
    "2B-pageindex": "ARM2B-PageIndex",
    "2C-enriched-rerank": "ARM2C-AzureDI-Agentic-Tree-Navigation-CNR",
}


def _arm_display_label(arm: str) -> str:
    return _ARM_DISPLAY_LABELS.get(arm, arm)


def render_top_bar(current_mode: str, current_arm: str, arms: list[str]) -> tuple[str, str]:
    """Render mode toggle and arm selector in the top bar. Returns (mode, arm)."""
    col_mode, col_arm, col_legend = st.columns([2, 2, 1])

    with col_mode:
        mode = st.radio(
            "Mode",
            ["review", "retrieval"],
            index=0 if current_mode == "review" else 1,
            horizontal=True,
            key="mode_radio",
            label_visibility="collapsed",
        )

    with col_arm:
        arm = st.selectbox(
            "Arm",
            arms,
            index=arms.index(current_arm) if current_arm in arms else 0,
            key="arm_select",
            format_func=_arm_display_label,
            label_visibility="collapsed",
        )

    with col_legend:
        render_legend(mode, arm)  # type: ignore[arg-type]

    return mode, arm  # type: ignore[return-value]


def render_page_nav(current_page: int, total_pages: int) -> int:
    """Render prev / text-input / next page controls. Returns the new current page."""
    if "page_number_input" not in st.session_state:
        st.session_state["page_number_input"] = str(current_page)

    # on_click callbacks run before the next rerun, so they may safely write
    # to session state before any widget is instantiated.
    def _prev():
        p = max(1, st.session_state.get("current_page", 1) - 1)
        st.session_state["current_page"] = p
        st.session_state["page_number_input"] = str(p)

    def _next():
        p = min(total_pages, st.session_state.get("current_page", 1) + 1)
        st.session_state["current_page"] = p
        st.session_state["page_number_input"] = str(p)

    def _on_change():
        val = st.session_state.get("page_number_input", "")
        try:
            p = int(val)
            if 1 <= p <= total_pages:
                st.session_state["current_page"] = p
            else:
                st.session_state["page_number_input"] = str(st.session_state.get("current_page", current_page))
        except (ValueError, TypeError):
            st.session_state["page_number_input"] = str(st.session_state.get("current_page", current_page))

    col_prev, col_input, col_next = st.columns([1, 2, 1])

    with col_prev:
        st.button("← Prev", disabled=current_page <= 1, on_click=_prev, use_container_width=True)

    with col_input:
        st.text_input(
            "Page",
            key="page_number_input",
            label_visibility="collapsed",
            on_change=_on_change,
        )

    with col_next:
        st.button("Next →", disabled=current_page >= total_pages, on_click=_next, use_container_width=True)

    st.caption(f"Page {st.session_state.get('current_page', current_page)} of {total_pages}")
    return st.session_state.get("current_page", current_page)


def render_search_bar(
    raw_text: str,
    word_index: list[WordSpan],
    total_pages: int,
) -> list[HighlightRegion]:
    """Render Ctrl-F-style search above the PDF panel.

    Returns a list of HighlightRegion (category ``"search_hit"`` plus one
    ``"search_active"`` for the focused match) which the caller should append
    to the regions list passed to ``build_annotations``.

    Side effects: when the focused match changes, the PDF jumps to its page
    by writing ``current_page`` / ``page_number_input`` into session state —
    same mechanism as the existing selection-driven auto-navigation.
    """
    # Session-state defaults
    st.session_state.setdefault("search_query", "")
    st.session_state.setdefault("search_active_idx", 0)
    st.session_state.setdefault("_search_last_query", "")

    def _clear() -> None:
        st.session_state["search_query"] = ""
        st.session_state["search_active_idx"] = 0
        st.session_state["_search_last_query"] = ""
        st.session_state.pop("_search_jump_to_page", None)

    c_input, c_clear = st.columns([5, 1], gap="small")
    with c_input:
        st.text_input(
            "Find in PDF",
            key="search_query",
            placeholder="🔍 Find in PDF…",
            label_visibility="collapsed",
        )
    with c_clear:
        st.button(
            "Clear",
            key="search_clear_btn",
            on_click=_clear,
            use_container_width=True,
        )

    query = (st.session_state.get("search_query") or "").strip()
    if not query:
        return []

    # Reset the active match whenever the query text changes.
    if query != st.session_state.get("_search_last_query"):
        st.session_state["search_active_idx"] = 0
        st.session_state["_search_last_query"] = query
        # Trigger a one-shot page jump on this rerun.
        st.session_state["_search_jump_pending"] = True

    active_idx = int(st.session_state.get("search_active_idx", 0))
    matches = text_matches_to_regions(raw_text, query, active_idx=active_idx)

    if not matches:
        st.caption(f"No matches for `{query}`.")
        return []

    n = len(matches)
    # Clamp active index in case the user just shrunk the result set.
    if active_idx >= n:
        active_idx = 0
        st.session_state["search_active_idx"] = 0

    def _prev() -> None:
        st.session_state["search_active_idx"] = (
            int(st.session_state.get("search_active_idx", 0)) - 1
        ) % max(1, n)
        st.session_state["_search_jump_pending"] = True

    def _next() -> None:
        st.session_state["search_active_idx"] = (
            int(st.session_state.get("search_active_idx", 0)) + 1
        ) % max(1, n)
        st.session_state["_search_jump_pending"] = True

    c_count, c_prev, c_next = st.columns([3, 1, 1], gap="small")
    with c_count:
        st.caption(f"Match {active_idx + 1} of {n} — `{query}`")
    with c_prev:
        st.button(
            "← Prev",
            key="search_prev_btn",
            on_click=_prev,
            use_container_width=True,
            disabled=n <= 1,
        )
    with c_next:
        st.button(
            "Next →",
            key="search_next_btn",
            on_click=_next,
            use_container_width=True,
            disabled=n <= 1,
        )

    # If a navigation just fired (or the query just changed), jump the PDF
    # viewer to the active match's page. Done here — after the click handlers
    # ran via on_click — so current_page is consistent with the rerun.
    if st.session_state.pop("_search_jump_pending", False):
        active_idx = int(st.session_state.get("search_active_idx", 0))
        active = matches[active_idx] if 0 <= active_idx < n else None
        if active is not None:
            page = get_first_page(active, word_index)
            if page is not None:
                target = page + 1  # WordSpan.page_num is 0-based
                target = max(1, min(total_pages, target))
                st.session_state["current_page"] = target
                st.session_state["page_number_input"] = str(target)

    return matches


def render_pdf_panel(
    pdf_bytes: bytes | None,
    annotations: list[dict],
    current_page: int,
    width: int = 900,
) -> None:
    """Render the central PDF viewer showing only current_page."""
    if pdf_bytes is None:
        st.info("Select a PDF from the sidebar to begin.")
        return

    try:
        from streamlit_pdf_viewer import pdf_viewer
    except ImportError:
        st.error(
            "streamlit-pdf-viewer is not installed. "
            "Run: pip install streamlit-pdf-viewer==0.0.19"
        )
        return

    # Filter annotations to only those on the current page so the viewer
    # doesn't try to render highlights for pages not being displayed.
    page_annotations = [a for a in annotations if a.get("page") == current_page]

    pdf_viewer(
        input=pdf_bytes,
        annotations=page_annotations,
        pages_to_render=[current_page],
        width=width,
    )

    # NOTE: filled translucent backgrounds for the annotation rectangles, the
    # mix-blend-mode that keeps text readable beneath them, and the bounding
    # box around the selected chunk are all delivered via a stylesheet
    # patched into streamlit_pdf_viewer's own dist/index.html — see
    # ``annotation_builder.patch_pdf_viewer_index_html``, called once at app
    # startup from ``app.py``. Patching the package's own asset is the only
    # mechanism that reliably reaches into the component's iframe (CSS in
    # the parent page does not propagate; component iframes are sandboxed
    # such that JS shims can't always cross the boundary).


def render_main_layout(
    pdf_bytes: bytes | None,
    sidebar_content_fn,
    regions: list[HighlightRegion],
    word_index: list[WordSpan],
    selected_id: str | None,
    mode: str,
    arm: str,
    query_text: str | None = None,
    relevant_article_ids: list[int] | None = None,
    gt_regions: list[HighlightRegion] | None = None,
    trace: dict | None = None,
    ranked_items: list[dict] | None = None,
    pdf_viewer_width: int = 900,
    graphrag_color_by_source: bool = False,
    total_pages: int = 1,
    gt_article_data: dict | None = None,
    raw_text: str = "",
    query_id: str | None = None,
) -> None:
    """Assemble the three-column layout and render all panels."""
    left, center, right = st.columns([1, 2, 1], gap="small")

    with left:
        sidebar_content_fn()

    # Re-read selected_id AFTER sidebar ran — sidebar may have updated it this rerun
    selected_id = st.session_state.get("selected_id")

    # Search bar lives in the central column, above the PDF nav. Side-effect:
    # may write `current_page` to session state, which we read below.
    with center:
        search_regions = render_search_bar(raw_text, word_index, total_pages)

    # Build annotations from all regions + mark selected
    # Selected-overlay annotations were previously appended here to paint an
    # amber tint over whichever region the user clicked. We removed that
    # because (a) the package's renderer overrides our `id` field with a
    # sequential `annotation-${i}`, so a `[id*="_sel_"]` CSS hook never
    # matched, and (b) in focus mode the selected chunk is the only chunk
    # visible on the page, so its own green / red fill already communicates
    # "this is what you picked" — no extra overlay needed.
    all_regions = list(regions) + list(search_regions)

    annotations = build_annotations(
        all_regions, word_index,
        graphrag_color_by_source=graphrag_color_by_source,
        selected_region_id=selected_id,
        mode=mode,
    )

    # Auto-navigate to the selected item's page only when the selection changes.
    # Guarding on _last_auto_nav_id prevents Prev/Next clicks from being
    # overridden on every rerun while a chunk is already selected. We match
    # any annotation whose id starts with the selected region's id, since the
    # builder appends per-line suffixes (``_ret0``, ``_gt0``, ``_0``, …).
    if selected_id and selected_id != st.session_state.get("_last_auto_nav_id"):
        for ann in annotations:
            aid = ann.get("id", "")
            if aid == selected_id or aid.startswith(f"{selected_id}_"):
                target_page = ann.get("page")   # 1-based
                if target_page:
                    st.session_state["current_page"] = target_page
                    st.session_state["page_number_input"] = str(target_page)
                break
        st.session_state["_last_auto_nav_id"] = selected_id

    with center:
        current_page = st.session_state.get("current_page", 1)
        current_page = render_page_nav(current_page, total_pages)
        render_pdf_panel(pdf_bytes, annotations, current_page, pdf_viewer_width)

    selected_region = next((r for r in regions if r.region_id == selected_id), None)

    with right:
        render_right_panel(
            selected_region=selected_region,
            query_text=query_text,
            relevant_article_ids=relevant_article_ids,
            gt_regions=gt_regions,
            trace=trace,
            ranked_items=ranked_items,
            method=arm,
            mode=mode,
            gt_article_data=gt_article_data,
            query_id=query_id,
        )
