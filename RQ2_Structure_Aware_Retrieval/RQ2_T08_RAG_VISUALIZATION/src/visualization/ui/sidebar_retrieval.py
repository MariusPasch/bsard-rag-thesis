"""Sidebar panel for Retrieval mode: query input and ranked result list."""

from __future__ import annotations

import streamlit as st

from visualization.query_loader import Query
from visualization.region_adapter import HighlightRegion


def _handle_retrieve_click() -> None:
    """Streamlit on_click callback for the Retrieve button.

    Runs *before* the script reruns, so by the time ``main()`` reaches its
    ``if _ss('retrieve_requested'):`` check at the top of retrieval mode, the
    flag is already set. Without this callback, the click would only set the
    flag *after* main()'s dispatch had already passed, and the user would
    see an empty sidebar until they triggered a second rerun.

    Reads ``_pending_query_text`` / ``_pending_top_k`` mirrors that the
    sidebar render keeps fresh on every rerun.
    """
    st.session_state["retrieve_requested"] = True
    st.session_state["active_query_text"] = st.session_state.get("_pending_query_text", "")
    st.session_state["top_k"] = int(st.session_state.get("_pending_top_k", 10))


def render_retrieval_sidebar(
    queries: list[Query],
    regions: list[HighlightRegion],
    gt_article_ids: list[int],
    selected_id: str | None,
    article_data: dict | None = None,
    cached_query_ids: set[str] | None = None,
    bundle_bsard_ids: set[int] | None = None,
    bsard_to_pdf: dict[int, str] | None = None,
) -> tuple[str, str | None, int]:
    """Render the retrieval-mode sidebar.

    Args:
        queries: full list of BSARD test queries from the DB.
        bundle_bsard_ids: BSARD ids of the articles in the current PDF.
            When the "Limit to questions relevant to this PDF" toggle is on
            (default), only questions whose GT bsard_ids overlap this set are
            shown.
        bsard_to_pdf: ``{bsard_id: pdf_filename}`` map, used to tag entries
            with their relevant PDF stem when the scope toggle is off.
        cached_query_ids: query_ids that have at least one cached retrieval
            result for the current PDF (T04 + T05 union). Shown as a ``✓``
            after the question id so the user knows which arms can answer
            without warming an index live.

    Returns:
        (query_text, newly_selected_id, top_k)
    """
    st.markdown("### Question")

    query_text = ""

    cached_set = cached_query_ids or set()
    bundle_bsard_ids = bundle_bsard_ids or set()
    bsard_to_pdf = bsard_to_pdf or {}

    # ── Scope toggle ────────────────────────────────────────────────────────
    # ON  → only questions whose GT articles intersect this PDF.
    # OFF → all BSARD test questions (ordered by question_id) with a PDF tag
    #       on each entry so the user knows which document they belong to.
    scope_to_pdf = st.checkbox(
        "Limit to questions relevant to this PDF",
        key="scope_to_pdf",
        value=st.session_state.get("scope_to_pdf", True),
        help=(
            "On: only show BSARD questions whose GT articles appear in the "
            "currently-loaded PDF.\n"
            "Off: show every BSARD question, ordered by question_id, with the "
            "relevant PDF stem next to each entry."
        ),
    )

    if scope_to_pdf and bundle_bsard_ids:
        displayed_queries: list[Query] = [
            q for q in queries
            if any(int(b) in bundle_bsard_ids for b in q.relevant_article_ids)
        ]
    else:
        displayed_queries = sorted(
            queries,
            key=lambda q: int(q.query_id) if str(q.query_id).isdigit() else q.query_id,
        )

    def _pdf_stem_for_query(q: Query) -> str:
        """Pick a representative PDF stem for ``q`` from its first GT article."""
        for bs in q.relevant_article_ids:
            try:
                pdf = bsard_to_pdf.get(int(bs))
            except (TypeError, ValueError):
                pdf = None
            if pdf:
                stem = pdf.removesuffix(".pdf")
                # Strip the BSARD-DB ``img_l_pdf_`` prefix and trailing ``_F``
                # so the label stays short — e.g.
                # ``img_l_pdf_1804_03_21_1804032150_F.pdf`` →
                # ``1804_03_21_1804032150``.
                stem = stem.removeprefix("img_l_pdf_").removesuffix("_F")
                return stem
        return "?"

    def _label(q: Query) -> str:
        cache_mark = " ✓" if q.query_id in cached_set else ""
        if scope_to_pdf:
            return f"[{q.query_id}]{cache_mark} {q.query_text[:80]}"
        return (
            f"[{q.query_id}]{cache_mark} {q.query_text[:60]} "
            f"— PDF: {_pdf_stem_for_query(q)}"
        )

    # ── Selectbox keys differ for scoped vs all so each preserves its state.
    select_key = "query_select_scoped" if scope_to_pdf else "query_select_all"

    if displayed_queries:
        options = ["(free text)"] + [_label(q) for q in displayed_queries]
        # Default index: 1 (first preset) when scoped, 0 (free text) when
        # showing the full set so we don't auto-pick something arbitrary.
        default_idx = 1 if scope_to_pdf else 0
        # If a question is already active (e.g. pre-loaded from a launched
        # bundle), default the dropdown to *that* question instead of the
        # first scoped preset — otherwise we'd silently override the bundle's
        # query_id, and arms whose cache lacks the overridden id error out.
        active_qid = st.session_state.get("active_query_id")
        if active_qid is not None:
            for i, q in enumerate(displayed_queries):
                if str(q.query_id) == str(active_qid):
                    default_idx = i + 1
                    break
        if select_key not in st.session_state:
            st.session_state[select_key] = default_idx
        sel_idx = st.selectbox(
            "Preset question",
            range(len(options)),
            format_func=lambda i: options[i],
            key=select_key,
        )
        if sel_idx > 0 and sel_idx <= len(displayed_queries):
            q = displayed_queries[sel_idx - 1]
            query_text = q.query_text
            # Side-effects: keep session state in sync with the picked query
            # so Retrieve has a query_id (T04/T05 lookups need it) and the
            # GT panel reflects the right article set.
            if q.relevant_article_ids != gt_article_ids:
                st.session_state["relevant_article_ids"] = q.relevant_article_ids
            st.session_state["active_query_id"] = q.query_id
        else:
            query_text = st.text_area("Question", value=query_text, key="query_freetext", height=80)
    else:
        query_text = st.text_area("Question", key="query_freetext", height=80)

    col_btn, col_k = st.columns([2, 1])
    with col_btn:
        # Mirror current values into session state so the on_click callback
        # (which runs before the rerun) can read them — local variables in
        # this function aren't accessible from a callback.
        st.session_state["_pending_query_text"] = query_text
        st.session_state["_pending_top_k"] = int(
            st.session_state.get("top_k_input", 10)
        )
        st.button(
            "Retrieve",
            type="primary",
            use_container_width=True,
            on_click=_handle_retrieve_click,
        )
    with col_k:
        st.number_input("Top-K", min_value=1, max_value=200, step=1, key="top_k_input")
    top_k = int(st.session_state.get("top_k_input", 10))

    st.divider()

    # Ranked results list
    new_selected: str | None = None

    all_retrieved = [r for r in regions if r.category in ("retrieved", "wrong_retrieved")]
    all_retrieved.sort(key=lambda r: r.rank if r.rank is not None else 9999)
    retrieved = all_retrieved[:top_k]
    gt_ids_set = set(gt_article_ids)

    article_data = article_data or {}

    # ── Inject CSS to colour hit/miss buttons (both sections) ───────────────
    if retrieved:
        css_rules = []
        for reg in retrieved:
            for key in (f"ret_btn_{reg.region_id}", f"hit_btn_{reg.region_id}"):
                if reg.category == "retrieved":
                    css_rules.append(
                        f"div.st-key-{key} button{{"
                        "background-color:#d4edda!important;"
                        "border-color:#28a745!important;"
                        "color:#155724!important;"
                        "}"
                    )
                else:
                    css_rules.append(
                        f"div.st-key-{key} button{{"
                        "background-color:#f8d7da!important;"
                        "border-color:#dc3545!important;"
                        "color:#721c24!important;"
                        "}"
                    )
        st.markdown(f"<style>{''.join(css_rules)}</style>", unsafe_allow_html=True)

    def _render_article_entries(reg: HighlightRegion) -> None:
        bs_ids = reg.metadata.get("bsard_ids") or []
        if not bs_ids:
            return
        # Outer expander — collapses the whole article list under the chunk.
        with st.expander(f"Articles in this chunk ({len(bs_ids)})", expanded=False):
            for bs_id in bs_ids:
                try:
                    key = int(bs_id)
                except (TypeError, ValueError):
                    key = bs_id
                entry = article_data.get(key) or article_data.get(str(key))
                if entry:
                    art_num = entry.get("article_number", "")
                    law = entry.get("law_title", "")
                    if art_num:
                        header = f"Art. {art_num} (BSARD {key})"
                        if law:
                            header += f" — {law}"
                    else:
                        header = f"BSARD {key}"
                        if law:
                            header += f" — {law}"
                    with st.expander(header, expanded=False):
                        st.text(entry["text"] or "_(no text)_")
                else:
                    st.caption(f"BSARD {key} — _(no DB entry)_")

    def _render_rank(rank: int | None) -> None:
        rank_str = str(rank) if rank else "—"
        st.markdown(
            f"<div style='text-align:center;font-size:1.3em;font-weight:700;"
            f"line-height:2.1em;color:#374151'>{rank_str}</div>",
            unsafe_allow_html=True,
        )

    # ── Correctly retrieved summary ───────────────────────────────────────────
    hits = [r for r in retrieved if r.category == "retrieved"]
    if hits:
        st.markdown(f"**Correctly Retrieved ({len(hits)} / {len(retrieved)})**")
        for reg in hits:
            score_label = f"{reg.score:.3f}" if reg.score is not None else ""
            is_selected = reg.region_id == selected_id
            prefix = "▶ " if is_selected else "   "
            label = f"{prefix}{reg.region_id} ({score_label}) ✓"
            c_rank, c_btn = st.columns([1, 9], gap="small")
            with c_rank:
                _render_rank(reg.rank)
            with c_btn:
                if st.button(label, key=f"hit_btn_{reg.region_id}", use_container_width=True):
                    new_selected = reg.region_id
            _render_article_entries(reg)
        st.divider()

    # ── Full ranked list ──────────────────────────────────────────────────────
    if retrieved:
        st.markdown(f"**Top {len(retrieved)} results**")
        for reg in retrieved:
            score_label = f"{reg.score:.3f}" if reg.score is not None else ""
            if reg.category == "retrieved":
                gt_mark = " ✓"
            elif gt_ids_set:
                gt_mark = " ✗"
            else:
                gt_mark = ""
            is_selected = reg.region_id == selected_id
            prefix = "▶ " if is_selected else "   "

            label = f"{prefix}{reg.region_id} ({score_label}){gt_mark}"
            c_rank, c_btn = st.columns([1, 9], gap="small")
            with c_rank:
                _render_rank(reg.rank)
            with c_btn:
                if st.button(label, key=f"ret_btn_{reg.region_id}", use_container_width=True):
                    new_selected = reg.region_id
            _render_article_entries(reg)

    elif st.session_state.get("active_query_text"):
        st.caption("No results yet — click Retrieve.")
    else:
        st.caption("Enter a query and click Retrieve.")

    return query_text, new_selected, top_k
