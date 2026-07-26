"""Build streamlit-pdf-viewer annotation dicts from HighlightRegion lists.

Annotation format expected by streamlit-pdf-viewer:
    {
        "page": int,           # 1-based
        "x": float,            # top-left x in PDF points
        "y": float,            # top-left y in PDF points
        "width": float,
        "height": float,
        "color": "#RRGGBB",
        "border": "solid",
        "id": str,
    }

Overlap layering:
    GT and retrieved regions share the PDF canvas. Where they intersect, a
    separate "overlap" annotation is emitted. The intersecting area is then
    subtracted from both the GT and retrieved rectangles to avoid additive
    opacity stacking (which would produce visual noise on semi-transparent
    annotations).

    Rectangle difference (A minus B) is computed by splitting A into up to
    four axis-aligned strips around B's intersection with A.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from visualization.coord_mapper import WordSpan, regions_to_page_bboxes
from visualization.region_adapter import HighlightCategory, HighlightRegion

if TYPE_CHECKING:
    pass

# ── Default colour map ────────────────────────────────────────────────────────

DEFAULT_COLORS: dict[str, str] = {
    "gt":             "#3B82F6",   # blue
    "retrieved":      "#10B981",   # green  (correctly retrieved)
    "wrong_retrieved":"#EF4444",   # red    (retrieved but not GT)
    "retrieved_node_page":       "#059669",   # darker green (T04 node page band)
    "wrong_retrieved_node_page": "#DC2626",   # darker red
    "overlap":        "#0F766E",   # darker teal (gt ∩ retrieved — distinct from both blue and green)
    "selected":       "#F59E0B",   # amber
    "chunk":          "#9CA3AF",   # gray
    "article":        "#9CA3AF",   # gray
    "search_hit":     "#FACC15",   # yellow (Ctrl-F match)
    "search_active":  "#EAB308",   # darker yellow (currently-focused match)
}

DEFAULT_OPACITIES: dict[str, float] = {
    # Halved twice (from the original palette) so text stays readable even
    # when 3–4 chunk rectangles stack on the same line. mix-blend-mode would
    # solve overlap stacking cleanly, but it breaks rendering in this
    # package's iframe — so we pay the legibility cost via lower alpha.
    "gt":             0.08,
    "retrieved":      0.08,
    "wrong_retrieved":0.08,
    # Page-band markers sit in the top margin (no text under them) so they
    # can use a much higher opacity without hurting legibility.
    "retrieved_node_page":       0.55,
    "wrong_retrieved_node_page": 0.55,
    "overlap":        0.18,        # still the strongest, so gt∩retrieved still reads
    "selected":       0.18,        # the 3px amber bounding box does the heavy lifting
    "chunk":          0.04,
    "article":        0.04,
    "search_hit":     0.15,
    "search_active":  0.25,
}

# Dim variants — used when something is selected. The selected region keeps
# its full DEFAULT_COLOR / DEFAULT_OPACITY (and gains the amber overlay on
# top), while every other dimmable region is repainted with these paler hexes
# at much lower opacity. The hex values are deliberately distinct from the
# DEFAULT_COLORS palette so the CSS-by-outline-color matcher can give them a
# different fill rule.
_DIM_COLORS: dict[str, str] = {
    "gt":             "#93C5FD",   # blue-300
    "retrieved":      "#6EE7B7",   # green-300
    "wrong_retrieved":"#FCA5A5",   # red-300
    "chunk":          "#D1D5DB",   # gray-300
    "article":        "#D1D5DB",
}

_DIM_OPACITIES: dict[str, float] = {
    "gt":             0.03,
    "retrieved":      0.03,
    "wrong_retrieved":0.03,
    "chunk":          0.02,
    "article":        0.02,
}

# GraphRAG hop-source colours for retrieved items when arm starts with "2C"
GRAPHRAG_SOURCE_COLORS: dict[str, str] = {
    "seed":  "#10B981",   # green
    "hop_1": "#F59E0B",   # amber
    "hop_2": "#F97316",   # orange
}

# ── Geometry helpers ──────────────────────────────────────────────────────────

Rect = tuple[float, float, float, float]  # (x0, y0, x1, y1)


def _intersection(a: Rect, b: Rect) -> Rect | None:
    """Return the intersection of two rects, or None if they don't overlap."""
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    if x0 < x1 and y0 < y1:
        return (x0, y0, x1, y1)
    return None


def _subtract(a: Rect, b: Rect) -> list[Rect]:
    """Return the parts of rect a that don't overlap with rect b.

    Splits a into up to four strips around the intersection.
    """
    inter = _intersection(a, b)
    if inter is None:
        return [a]

    ix0, iy0, ix1, iy1 = inter
    ax0, ay0, ax1, ay1 = a

    parts: list[Rect] = []

    if ay0 < iy0:             # top strip
        parts.append((ax0, ay0, ax1, iy0))
    if iy1 < ay1:             # bottom strip
        parts.append((ax0, iy1, ax1, ay1))
    if ax0 < ix0:             # left strip (clipped vertically to intersection y-range)
        parts.append((ax0, iy0, ix0, iy1))
    if ix1 < ax1:             # right strip
        parts.append((ix1, iy0, ax1, iy1))

    return parts


# ── Annotation dict builder ───────────────────────────────────────────────────


def _page_extents(
    word_index: list[WordSpan],
) -> dict[int, tuple[float, float, float, float]]:
    """Return ``{0-based page_num: (min_x0, min_y0, max_x1, max_y1)}`` from the word index.

    Used to position page-band markers in the top margin without needing the
    PDF's actual page dimensions — the band spans the text's horizontal extent
    on that page, and sits just above the topmost line of words.
    """
    out: dict[int, list[float]] = {}
    for ws in word_index:
        x0, y0, x1, y1 = ws.bbox
        cur = out.get(ws.page_num)
        if cur is None:
            out[ws.page_num] = [x0, y0, x1, y1]
        else:
            cur[0] = min(cur[0], x0)
            cur[1] = min(cur[1], y0)
            cur[2] = max(cur[2], x1)
            cur[3] = max(cur[3], y1)
    return {k: (v[0], v[1], v[2], v[3]) for k, v in out.items()}


def _make_annotation(
    region_id: str,
    page_num: int,   # 0-based internally
    rect: Rect,
    category: str,
    color_map: dict[str, str] | None = None,
    color_override: str | None = None,
    suffix: str = "",
    dim: bool = False,
) -> dict:
    if dim and category in _DIM_COLORS and color_override is None:
        color = _DIM_COLORS[category]
        opacity = _DIM_OPACITIES.get(category, 0.10)
    else:
        colors = color_map or DEFAULT_COLORS
        color = color_override or colors.get(category, "#9CA3AF")
        opacity = DEFAULT_OPACITIES.get(category, 0.20)

    x0, y0, x1, y1 = rect
    return {
        "page": page_num + 1,         # streamlit-pdf-viewer is 1-based
        "x": x0,
        "y": y0,
        "width": max(x1 - x0, 1.0),
        "height": max(y1 - y0, 1.0),
        "color": color,
        "opacity": opacity,
        "border": "solid",
        "id": f"{region_id}{suffix}",
    }


def build_annotations(
    regions: list[HighlightRegion],
    word_index: list[WordSpan],
    color_map: dict[str, str] | None = None,
    graphrag_color_by_source: bool = False,
    selected_region_id: str | None = None,
    mode: str = "retrieval",
) -> list[dict]:
    """Full pipeline: regions → per-page bboxes → annotation dicts with overlap layer.

    Steps:
      1. Map all regions to page bboxes.
      2. Per page: compute geometric intersections of gt and retrieved rects.
      3. Subtract intersection area from gt and retrieved rects; emit overlap rects.
      4. Emit one annotation dict per (region, line-bbox) slice.

    When ``selected_region_id`` is set, every dimmable region whose ``region_id``
    differs from it is painted with a paler hex / lower opacity so the focused
    chunk visually dominates. The selected region itself keeps its default
    colour and additionally carries the amber ``selected`` overlay.

    ``mode`` controls focus behaviour:
      - ``"retrieval"`` — when nothing is selected, *hide* every retrieved /
        chunk / article rectangle so only GT context shows; clicking a
        sidebar row reveals just that one chunk on the page. This is the
        original "one chunk at a time" behaviour for retrieval inspection.
      - ``"review"`` — every chunk / article is always rendered (faintly
        when something else is selected, full opacity for the selection),
        because Review mode's whole purpose is to *see all* chunks /
        articles tile the page.
    """
    if not regions or not word_index:
        return []

    # In retrieval mode: hide every non-selected retrieved / wrong-retrieved /
    # chunk / article region so the user sees a single highlight at a time
    # (the one they clicked). Ground-truth spans, the selected overlay, and
    # search hits are always preserved as context. With no selection active,
    # this effectively hides every retrieved chunk — GT alone is shown until
    # the user picks a row from the sidebar.
    #
    # In review mode: keep everything visible. Selected dimming below still
    # emphasises the focused region.
    if mode != "review":
        _HIDEABLE_WHEN_UNSELECTED = ("retrieved", "wrong_retrieved", "chunk", "article")
        regions = [
            r for r in regions
            if r.category not in _HIDEABLE_WHEN_UNSELECTED
            or r.region_id == selected_region_id
        ]
    if not regions:
        return []

    cmap = color_map or DEFAULT_COLORS

    # Split regions by category for overlap computation.
    gt_regions = [r for r in regions if r.category == "gt"]
    ret_regions = [r for r in regions if r.category == "retrieved"]
    # Search hits must paint on TOP of everything else; emit them last.
    search_regions = [r for r in regions if r.category in ("search_hit", "search_active")]
    # The selected overlay is split out so it can be emitted AFTER gt/retrieved
    # (and after overlap) — guarantees the focused chunk's amber fill and its
    # bounding-box outline aren't hidden by neighbouring rectangles.
    selected_regions = [r for r in regions if r.category == "selected"]
    other_regions = [
        r for r in regions
        if r.category not in ("gt", "retrieved", "search_hit", "search_active", "selected")
    ]

    page_bboxes_gt = regions_to_page_bboxes(gt_regions, word_index)
    page_bboxes_ret = regions_to_page_bboxes(ret_regions, word_index)
    page_bboxes_other = regions_to_page_bboxes(other_regions, word_index)
    page_bboxes_selected = regions_to_page_bboxes(selected_regions, word_index)
    page_bboxes_search = regions_to_page_bboxes(search_regions, word_index)

    def _is_dim(region: HighlightRegion) -> bool:
        # In focus mode, GT is the only non-selected thing left on the
        # page; render it at full colour so it's clearly visible as
        # context. Selected / overlap / search are also never dimmed.
        if region.category in ("gt", "selected", "overlap", "search_hit", "search_active"):
            return False
        if not selected_region_id:
            return False
        return region.region_id != selected_region_id

    # ── Subtract the selected chunk's geometry from every other dimmable
    # rect on the same page. Reasoning: sliding-window chunks share roughly
    # half their tokens with their neighbours, so non-selected chunks would
    # otherwise stack inside the selected area, painting that region with
    # multiple translucent layers and making it look much darker than its
    # configured opacity. Subtracting here means the selected chunk's
    # rectangle is the *only* highlight inside its area, so the colour is
    # uniform and matches DEFAULT_OPACITIES exactly.
    if selected_region_id:
        sel_rects_by_page: dict[int, list[Rect]] = {}
        for _bboxes in (page_bboxes_ret, page_bboxes_other):
            for page_num, items in _bboxes.items():
                for region, rect in items:
                    if region.region_id == selected_region_id:
                        sel_rects_by_page.setdefault(page_num, []).append(rect)

        def _subtract_selected(items: list, page_num: int) -> list:
            sel_rects = sel_rects_by_page.get(page_num, [])
            if not sel_rects:
                return items
            new_items = []
            for region, rect in items:
                if region.region_id == selected_region_id:
                    new_items.append((region, rect))
                    continue
                if region.category in ("selected", "overlap", "search_hit", "search_active"):
                    new_items.append((region, rect))
                    continue
                remaining = [rect]
                for s_rect in sel_rects:
                    next_rem: list[Rect] = []
                    for r in remaining:
                        next_rem.extend(_subtract(r, s_rect))
                    remaining = next_rem
                for sub_rect in remaining:
                    new_items.append((region, sub_rect))
            return new_items

        for _bboxes in (page_bboxes_ret, page_bboxes_gt, page_bboxes_other):
            for page_num in list(_bboxes.keys()):
                _bboxes[page_num] = _subtract_selected(_bboxes[page_num], page_num)

    annotations: list[dict] = []

    # ── Other regions (chunk, article, wrong_retrieved) ───────────────────
    _other_line_counter: dict[str, int] = {}
    for page_num, items in page_bboxes_other.items():
        for region, rect in items:
            i = _other_line_counter.get(region.region_id, 0)
            _other_line_counter[region.region_id] = i + 1
            color = None
            if region.category == "retrieved" and graphrag_color_by_source and region.arm.startswith("2C"):
                src = region.metadata.get("source", "")
                color = GRAPHRAG_SOURCE_COLORS.get(src)
            annotations.append(
                _make_annotation(
                    region.region_id, page_num, rect, region.category, cmap, color,
                    suffix=f"_{i}", dim=_is_dim(region),
                )
            )

    # ── GT + retrieved with overlap subtraction (per page) ─────────────────
    all_pages = set(page_bboxes_gt) | set(page_bboxes_ret)

    for page_num in all_pages:
        gt_items = page_bboxes_gt.get(page_num, [])
        ret_items = page_bboxes_ret.get(page_num, [])

        # Collect all rect pairs to compute intersections
        gt_rects: list[tuple[HighlightRegion, Rect]] = list(gt_items)
        ret_rects: list[tuple[HighlightRegion, Rect]] = list(ret_items)

        overlap_rects: list[Rect] = []

        # Find intersections between every gt × retrieved pair, then
        # deduplicate. Sliding-window chunks frequently share GT spans, so
        # the same (gt_line × ret_line) intersection can be produced by
        # multiple retrieved chunks. Without dedup these identical (or
        # near-identical) rects all get emitted as separate overlap
        # annotations and stack as multiple translucent teal layers,
        # producing 2-3× the configured opacity in those areas.
        _seen_overlaps: set[tuple[float, float, float, float]] = set()
        for _gt_region, g_rect in gt_rects:
            for _ret_region, r_rect in ret_rects:
                inter = _intersection(g_rect, r_rect)
                if not inter:
                    continue
                key = (
                    round(inter[0], 1), round(inter[1], 1),
                    round(inter[2], 1), round(inter[3], 1),
                )
                if key in _seen_overlaps:
                    continue
                _seen_overlaps.add(key)
                overlap_rects.append(inter)

        # Subtract overlap from each GT rect
        for gt_region, g_rect in gt_rects:
            gt_dim = _is_dim(gt_region)
            remaining = [g_rect]
            for o_rect in overlap_rects:
                next_remaining: list[Rect] = []
                for r in remaining:
                    next_remaining.extend(_subtract(r, o_rect))
                remaining = next_remaining
            for i, sub_rect in enumerate(remaining):
                annotations.append(
                    _make_annotation(
                        gt_region.region_id, page_num, sub_rect, "gt", cmap,
                        suffix=f"_gt{i}", dim=gt_dim,
                    )
                )

        # Subtract overlap from each retrieved rect
        for ret_region, r_rect in ret_rects:
            ret_dim = _is_dim(ret_region)
            remaining = [r_rect]
            for o_rect in overlap_rects:
                next_remaining = []
                for r in remaining:
                    next_remaining.extend(_subtract(r, o_rect))
                remaining = next_remaining
            color = None
            if graphrag_color_by_source and ret_region.arm.startswith("2C"):
                src = ret_region.metadata.get("source", "")
                color = GRAPHRAG_SOURCE_COLORS.get(src)
            for i, sub_rect in enumerate(remaining):
                annotations.append(
                    _make_annotation(
                        ret_region.region_id, page_num, sub_rect, "retrieved", cmap, color,
                        suffix=f"_ret{i}", dim=ret_dim,
                    )
                )

        # Emit overlap annotations on top
        for i, o_rect in enumerate(overlap_rects):
            annotations.append(
                _make_annotation(f"overlap_p{page_num}_{i}", page_num, o_rect, "overlap", cmap)
            )

    # ── Selected overlay — emitted after gt/retrieved/overlap so its amber
    #    fill and bounding-box outline sit on top of the regular layers.
    _selected_line_counter: dict[str, int] = {}
    for page_num, items in page_bboxes_selected.items():
        for region, rect in items:
            i = _selected_line_counter.get(region.region_id, 0)
            _selected_line_counter[region.region_id] = i + 1
            annotations.append(
                _make_annotation(region.region_id, page_num, rect, "selected", cmap, suffix=f"_{i}")
            )

    # ── Search hits — last, so they paint on top of every other layer ─────
    _search_line_counter: dict[str, int] = {}
    for page_num, items in page_bboxes_search.items():
        for region, rect in items:
            i = _search_line_counter.get(region.region_id, 0)
            _search_line_counter[region.region_id] = i + 1
            annotations.append(
                _make_annotation(region.region_id, page_num, rect, region.category, cmap, suffix=f"_{i}")
            )

    # ── T04 page-band markers ────────────────────────────────────────────
    # These are siblings of node-level retrieved items: they don't carry
    # char offsets — instead, ``metadata.page_number`` (1-based, as stored
    # in T04's compare_*.jsonl) tells us which page the node lives on.
    # We render a thin saturated band in that page's top margin so the user
    # can see the specific page even when the article fill paints across
    # many pages. They live outside the focus-mode hide rule (the band IS
    # the precise locator) and are emitted at the END of the list so they
    # always paint on top of any existing top-margin content.
    band_regions = [
        r for r in regions
        if r.category in ("retrieved_node_page", "wrong_retrieved_node_page")
    ]
    if band_regions:
        extents = _page_extents(word_index)
        for region in band_regions:
            page_1b = region.metadata.get("page_number")
            if page_1b is None:
                continue
            page_0b = int(page_1b) - 1
            ext = extents.get(page_0b)
            if ext is None:
                continue
            min_x0, min_y0, max_x1, _max_y1 = ext
            band_height = 12.0
            gap = 4.0
            y_top = max(min_y0 - band_height - gap, 4.0)
            y_bot = y_top + band_height
            rect = (min_x0, y_top, max_x1, y_bot)
            annotations.append(
                _make_annotation(
                    region.region_id, page_0b, rect, region.category, cmap,
                    suffix="_pgband",
                )
            )

    return annotations


def annotation_id_for_region(region_id: str) -> str:
    """Return the primary annotation id for a given region_id (for scroll_to_annotation)."""
    return region_id


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _annotation_fill_css_rules() -> str:
    """Return the CSS rules (no <style> wrapper) that fill annotation rectangles.

    Minimal baseline: only background-color fills (matched on inline outline
    colour) and a thicker outline on the selected overlay. We deliberately
    avoid ``mix-blend-mode`` and any layout-affecting properties because the
    streamlit-pdf-viewer iframe paints annotations with a fragile DOM/CSS
    interaction — adding properties beyond ``background-color`` can cause
    the package's JS to fail to paint outlines at all, leaving the canvas
    looking like the annotation list was empty.
    """
    rules: list[str] = []
    seen: set[str] = set()

    def _emit(category_or_label: str, hex_color: str, opacity: float) -> None:
        if hex_color in seen:
            return
        seen.add(hex_color)
        r, g, b = _hex_to_rgb(hex_color)
        rgba = f"rgba({r}, {g}, {b}, {opacity})"
        # Use descendant selector (not direct-child) so we still match if the
        # package wraps annotations in an extra container in some version.
        selectors = [
            f'#pdfAnnotations div[style*="rgb({r}, {g}, {b})"]',
            f'#pdfAnnotations div[style*="{hex_color.upper()}"]',
            f'#pdfAnnotations div[style*="{hex_color.lower()}"]',
        ]
        rules.append(
            f"/* {category_or_label} */ "
            f"{','.join(selectors)} {{ background-color: {rgba} !important; }}"
        )

    for cat, hex_color in DEFAULT_COLORS.items():
        _emit(cat, hex_color, DEFAULT_OPACITIES.get(cat, 0.25))
    for cat, hex_color in _DIM_COLORS.items():
        _emit(f"{cat}_dim", hex_color, _DIM_OPACITIES.get(cat, 0.10))
    for src, hex_color in GRAPHRAG_SOURCE_COLORS.items():
        _emit(f"graphrag {src}", hex_color, DEFAULT_OPACITIES.get("retrieved", 0.25))

    # NOTE: a previous version added a ``[id*="_sel_"]`` rule for a thicker
    # outline on the selected chunk. It turned out to be a no-op because
    # streamlit-pdf-viewer overrides the annotation div's DOM id with a
    # sequential ``annotation-${i}`` and ignores the id field we pass.
    # We dropped the selected-overlay annotation entirely (see
    # ui/layout.render_main_layout), so the focused chunk's own green / red
    # fill is now the sole indicator of selection.

    return "\n".join(rules)


def patch_pdf_viewer_index_html() -> bool:
    """Inject our fill / blend / selection-box CSS into ``streamlit_pdf_viewer``'s
    ``dist/index.html`` so the rules load *inside* the component's iframe.

    Why this approach: the package's iframe paints annotations as hollow
    rectangles (``outline`` only, no fill). CSS in the parent Streamlit page
    cannot cross the iframe boundary, and a JS shim trying to walk into the
    iframe via ``window.parent.document`` is unreliable across Streamlit
    versions. Patching the package's own served HTML once per app start is
    the most boring, robust mechanism — the iframe loads index.html, browser
    parses our injected ``<style>``, and CSS rules apply to every annotation
    div the viewer ever creates inside that iframe.

    The patch is idempotent: re-runs replace the marker block in place, so
    palette / opacity tweaks in this file propagate on the next app boot.

    Returns True if the file was modified, False otherwise.
    """
    try:
        import streamlit_pdf_viewer as _spv
    except ImportError:
        return False

    from pathlib import Path
    pkg_dir = Path(_spv.__file__).resolve().parent
    index_html = pkg_dir / "frontend" / "dist" / "index.html"
    if not index_html.exists():
        return False

    START = "<!--rag-vis-fill-css-START-->"
    END = "<!--rag-vis-fill-css-END-->"

    css_rules = _annotation_fill_css_rules()
    new_block = f"{START}\n<style>\n{css_rules}\n</style>\n{END}"

    try:
        content = index_html.read_text(encoding="utf-8")
    except OSError:
        return False

    if START in content and END in content:
        import re
        pattern = re.escape(START) + r".*?" + re.escape(END)
        new_content = re.sub(pattern, new_block, content, count=1, flags=re.DOTALL)
    elif "</head>" in content:
        new_content = content.replace("</head>", f"{new_block}\n</head>", 1)
    else:
        return False

    if new_content == content:
        return False

    try:
        index_html.write_text(new_content, encoding="utf-8")
        return True
    except OSError:
        return False


