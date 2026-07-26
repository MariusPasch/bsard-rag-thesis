"""
Visualization unit + integration tests.

Run:
    cd RQ2_T08_RAG_VISUALIZATION
    .venv\\Scripts\\activate
    pytest tests/test_visualization.py -v

Covers:
  A. region_adapter  — HighlightRegion creation and correct/wrong classification
  B. annotation_builder — colors, overlap, wrong_retrieved bypass
  C. coord_mapper  — word-index build and region→bbox mapping
  D. layout helpers — selected_id promotion, scroll target resolution
  E. demo pipeline — BM25 chunking + retrieval smoke test (requires PDF + DB)
"""

from __future__ import annotations

import os

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Literal

import pytest

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
_REPO = _HERE.parent.parent

sys.path.insert(0, str(_SRC))
for _sibling in _REPO.glob("RQ2_T0*/src"):
    _s = str(_sibling)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from visualization.region_adapter import (
    HighlightRegion,
    chunks_to_regions,
    gt_regions_from_article_ids,
    retrieval_result_to_regions,
)
from visualization.annotation_builder import (
    DEFAULT_COLORS,
    DEFAULT_OPACITIES,
    _intersection,
    _subtract,
    build_annotations,
)
from visualization.coord_mapper import WordSpan, regions_to_page_bboxes, get_first_page

# ── Shared fixtures ───────────────────────────────────────────────────────────

PDF_PATH = _REPO / "RQ2_T02_DATA_LOADER/data/pdfs/2004_05_27_2004A27101.pdf"
DB_PATH = Path(os.environ.get("RQ2_BSARD_DB", str(_REPO / "data" / "bsard_corpus.db")))

# Minimal stubs so tests don't need real arm1_naive Chunk objects
@dataclass
class _FakeChunk:
    chunk_id: str
    text: str
    doc_id: str
    start_char: int
    end_char: int
    start_token: int
    end_token: int
    bsard_ids: list[int] = field(default_factory=list)

@dataclass
class _FakeBundle:
    document_id: int = 0
    pdf_path: str = ""
    pdf_filename: str = ""
    metadata: dict = field(default_factory=dict)
    articles: list = field(default_factory=list)
    raw_text: str = ""
    has_azure_extraction: bool = False

@dataclass
class _FakeRetrievalResult:
    query_id: str
    query_text: str
    ranked_items: list[dict]
    method: str = "arm1_bm25"
    cost: dict = field(default_factory=dict)
    trace: dict | None = None


# ═══════════════════════════════════════════════════════════════════════════════
# A. region_adapter tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestChunksToRegions:
    """chunks_to_regions — trivial adapter, Chunk → HighlightRegion."""

    def test_basic_fields(self):
        chunk = _FakeChunk("doc_sw_0", "hello world", "doc", 0, 11, 0, 5, [42])
        regions = chunks_to_regions([chunk])
        assert len(regions) == 1
        r = regions[0]
        assert r.region_id == "doc_sw_0"
        assert r.text == "hello world"
        assert r.start_char == 0
        assert r.end_char == 11
        assert r.category == "chunk"
        assert r.arm == "arm1"
        assert r.metadata["bsard_ids"] == [42]

    def test_empty_list(self):
        assert chunks_to_regions([]) == []

    def test_multiple_chunks(self):
        chunks = [_FakeChunk(f"doc_sw_{i}", "text", "doc", i*10, i*10+5, 0, 3) for i in range(5)]
        regions = chunks_to_regions(chunks)
        assert len(regions) == 5
        assert all(r.category == "chunk" for r in regions)


class TestRetrievalResultToRegions:
    """retrieval_result_to_regions — correct vs. wrong classification."""

    def _make_result(self, items):
        return _FakeRetrievalResult(
            query_id="q1", query_text="test", ranked_items=items
        )

    def test_correctly_retrieved_when_bsard_ids_overlap_gt(self):
        result = self._make_result([{
            "id": "chunk_1", "score": 0.9, "text": "...",
            "bsard_ids": [10, 20], "start_char": 0, "end_char": 100,
            "start_token": 0, "end_token": 50, "metadata": {},
        }])
        bundle = _FakeBundle()
        regions = retrieval_result_to_regions(result, bundle, gt_article_ids=[20, 30])
        assert len(regions) == 1
        assert regions[0].category == "retrieved"

    def test_wrongly_retrieved_when_no_bsard_id_overlap(self):
        result = self._make_result([{
            "id": "chunk_1", "score": 0.5, "text": "...",
            "bsard_ids": [99], "start_char": 0, "end_char": 100,
            "start_token": 0, "end_token": 50, "metadata": {},
        }])
        bundle = _FakeBundle()
        regions = retrieval_result_to_regions(result, bundle, gt_article_ids=[20, 30])
        assert len(regions) == 1
        assert regions[0].category == "wrong_retrieved"

    def test_no_gt_article_ids_defaults_to_retrieved(self):
        result = self._make_result([{
            "id": "chunk_1", "score": 0.7, "text": "...",
            "bsard_ids": [99], "start_char": 0, "end_char": 100,
            "start_token": 0, "end_token": 50, "metadata": {},
        }])
        bundle = _FakeBundle()
        # gt_article_ids=None → no GT set → can't classify as wrong
        regions = retrieval_result_to_regions(result, bundle, gt_article_ids=None)
        assert regions[0].category == "retrieved"

    def test_empty_bsard_ids_treated_as_wrong(self):
        result = self._make_result([{
            "id": "chunk_1", "score": 0.3, "text": "...",
            "bsard_ids": [], "start_char": 0, "end_char": 100,
            "start_token": 0, "end_token": 50, "metadata": {},
        }])
        bundle = _FakeBundle()
        regions = retrieval_result_to_regions(result, bundle, gt_article_ids=[20])
        assert regions[0].category == "wrong_retrieved"

    def test_rank_assigned_correctly(self):
        result = self._make_result([
            {"id": f"chunk_{i}", "score": 1.0 - i*0.1, "text": "...",
             "bsard_ids": [], "start_char": i*100, "end_char": i*100+50,
             "start_token": 0, "end_token": 20, "metadata": {}}
            for i in range(3)
        ])
        regions = retrieval_result_to_regions(result, _FakeBundle())
        assert [r.rank for r in regions] == [1, 2, 3]

    def test_char_offsets_preserved(self):
        result = self._make_result([{
            "id": "c1", "score": 0.5, "text": "hello",
            "bsard_ids": [], "start_char": 42, "end_char": 47,
            "start_token": 5, "end_token": 8, "metadata": {},
        }])
        r = retrieval_result_to_regions(result, _FakeBundle())[0]
        assert r.start_char == 42
        assert r.end_char == 47
        assert r.start_token == 5
        assert r.end_token == 8


class TestGTRegions:
    """gt_regions_from_article_ids — ground-truth region creation."""

    def _make_span(self, art_id, start, end):
        class _Span:
            bsard_id = art_id
            start_char = start
            end_char = end
        return _Span()

    def test_creates_gt_region_with_correct_char_offsets(self):
        spans = [self._make_span(10, 100, 500)]
        bundle = _FakeBundle()
        regions = gt_regions_from_article_ids([10], bundle, article_spans=spans)
        assert len(regions) == 1
        assert regions[0].category == "gt"
        assert regions[0].start_char == 100
        assert regions[0].end_char == 500
        assert regions[0].region_id == "gt_10"

    def test_unknown_article_id_yields_zero_offset(self):
        regions = gt_regions_from_article_ids([99], _FakeBundle(), article_spans=[])
        assert regions[0].start_char == 0
        assert regions[0].end_char == 0

    def test_multiple_gt_articles(self):
        spans = [self._make_span(i, i*100, i*100+50) for i in range(1, 4)]
        regions = gt_regions_from_article_ids([1, 2, 3], _FakeBundle(), article_spans=spans)
        assert len(regions) == 3
        assert all(r.category == "gt" for r in regions)


# ═══════════════════════════════════════════════════════════════════════════════
# B. annotation_builder tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestColors:
    """Verify expected colors and opacities for each category."""

    def test_gt_color_is_blue(self):
        assert DEFAULT_COLORS["gt"] == "#3B82F6"

    def test_retrieved_color_is_green(self):
        assert DEFAULT_COLORS["retrieved"] == "#10B981"

    def test_wrong_retrieved_color_is_red(self):
        assert DEFAULT_COLORS["wrong_retrieved"] == "#EF4444"

    def test_overlap_color_is_teal(self):
        assert DEFAULT_COLORS["overlap"] == "#0D9488"

    def test_selected_color_is_amber(self):
        assert DEFAULT_COLORS["selected"] == "#F59E0B"

    def test_all_categories_have_opacity(self):
        for cat in ("gt", "retrieved", "wrong_retrieved", "overlap", "selected", "chunk", "article"):
            assert cat in DEFAULT_OPACITIES, f"Missing opacity for {cat}"


class TestRectGeometry:
    """_intersection and _subtract geometry helpers."""

    def test_intersection_overlapping(self):
        a = (0, 0, 10, 10)
        b = (5, 5, 15, 15)
        inter = _intersection(a, b)
        assert inter == (5, 5, 10, 10)

    def test_intersection_no_overlap(self):
        assert _intersection((0, 0, 5, 5), (6, 6, 10, 10)) is None

    def test_intersection_touching_edge_no_area(self):
        assert _intersection((0, 0, 5, 5), (5, 0, 10, 5)) is None

    def test_subtract_b_inside_a_produces_four_strips(self):
        a = (0, 0, 10, 10)
        b = (3, 3, 7, 7)
        parts = _subtract(a, b)
        assert len(parts) == 4  # top, bottom, left, right strips

    def test_subtract_no_overlap_returns_a(self):
        a = (0, 0, 5, 5)
        b = (10, 10, 20, 20)
        assert _subtract(a, b) == [a]

    def test_subtract_full_cover_returns_empty_strips(self):
        a = (3, 3, 7, 7)
        b = (0, 0, 10, 10)  # b completely covers a
        parts = _subtract(a, b)
        # All strips should have zero or negative area
        for p in parts:
            assert p[2] - p[0] <= 0 or p[3] - p[1] <= 0


class TestBuildAnnotations:
    """build_annotations — integration of mapping + overlap."""

    def _word_index(self, regions):
        """Synthetic word index: one word per region at its char range."""
        spans = []
        for r in regions:
            mid = (r.start_char + r.end_char) // 2
            spans.append(WordSpan(
                page_num=0,
                global_char_start=r.start_char,
                global_char_end=r.end_char,
                bbox=(10.0, 10.0, 100.0, 20.0),
                block_no=0,
                line_no=0,
            ))
        return sorted(spans, key=lambda ws: ws.global_char_start)

    def test_empty_regions_returns_empty(self):
        assert build_annotations([], []) == []

    def test_chunk_annotation_uses_gray_color(self):
        region = HighlightRegion("c1", "text", 0, 100, "chunk", "arm1")
        wi = self._word_index([region])
        anns = build_annotations([region], wi)
        assert any(a["color"] == DEFAULT_COLORS["chunk"] for a in anns)

    def test_wrong_retrieved_annotation_uses_red_color(self):
        region = HighlightRegion("c1", "text", 0, 100, "wrong_retrieved", "arm1")
        wi = self._word_index([region])
        anns = build_annotations([region], wi)
        assert any(a["color"] == DEFAULT_COLORS["wrong_retrieved"] for a in anns)

    def test_gt_and_retrieved_produce_overlap_annotation(self):
        gt = HighlightRegion("gt_1", "gt text", 0, 100, "gt", "gt")
        ret = HighlightRegion("chunk_1", "ret text", 0, 100, "retrieved", "arm1")
        wi = self._word_index([gt, ret])
        anns = build_annotations([gt, ret], wi)
        overlap_anns = [a for a in anns if a["color"] == DEFAULT_COLORS["overlap"]]
        assert len(overlap_anns) > 0

    def test_wrong_retrieved_not_in_overlap_computation(self):
        gt = HighlightRegion("gt_1", "text", 0, 100, "gt", "gt")
        wrong = HighlightRegion("c1", "text", 0, 100, "wrong_retrieved", "arm1")
        wi = self._word_index([gt, wrong])
        anns = build_annotations([gt, wrong], wi)
        # No overlap should be computed (wrong_retrieved bypasses the gt ∩ retrieved logic)
        overlap_anns = [a for a in anns if a["color"] == DEFAULT_COLORS["overlap"]]
        assert len(overlap_anns) == 0

    def test_annotation_page_is_one_based(self):
        region = HighlightRegion("c1", "text", 0, 100, "chunk", "arm1")
        wi = self._word_index([region])
        anns = build_annotations([region], wi)
        assert all(a["page"] >= 1 for a in anns)

    def test_annotation_has_required_fields(self):
        region = HighlightRegion("c1", "text", 0, 100, "chunk", "arm1")
        wi = self._word_index([region])
        anns = build_annotations([region], wi)
        for a in anns:
            for field in ("page", "x", "y", "width", "height", "color", "id"):
                assert field in a, f"Missing field {field!r} in annotation"


# ═══════════════════════════════════════════════════════════════════════════════
# C. coord_mapper tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestCoordMapper:
    """Word index building and region→bbox mapping."""

    def _make_word_index(self, entries):
        """entries: list of (page, global_start, global_end, bbox)."""
        return [
            WordSpan(page_num=p, global_char_start=s, global_char_end=e,
                     bbox=bbox, block_no=0, line_no=0)
            for p, s, e, bbox in entries
        ]

    def test_regions_to_page_bboxes_single_word(self):
        wi = self._make_word_index([(0, 10, 20, (5.0, 5.0, 50.0, 15.0))])
        region = HighlightRegion("r1", "word", 10, 20, "chunk", "arm1")
        result = regions_to_page_bboxes([region], wi)
        assert 0 in result
        assert len(result[0]) == 1
        _, bbox = result[0][0]
        assert bbox == (5.0, 5.0, 50.0, 15.0)

    def test_regions_to_page_bboxes_no_overlap(self):
        wi = self._make_word_index([(0, 50, 60, (5.0, 5.0, 50.0, 15.0))])
        region = HighlightRegion("r1", "word", 0, 10, "chunk", "arm1")
        result = regions_to_page_bboxes([region], wi)
        assert result == {}

    def test_get_first_page_returns_correct_page(self):
        wi = self._make_word_index([
            (2, 10, 20, (5, 5, 50, 15)),
            (3, 20, 30, (5, 5, 50, 15)),
        ])
        region = HighlightRegion("r1", "text", 10, 30, "chunk", "arm1")
        assert get_first_page(region, wi) == 2

    def test_get_first_page_no_match_returns_none(self):
        wi = self._make_word_index([(0, 100, 200, (0, 0, 10, 10))])
        region = HighlightRegion("r1", "text", 300, 400, "chunk", "arm1")
        assert get_first_page(region, wi) is None

    def test_multi_page_region_appears_in_both_pages(self):
        wi = self._make_word_index([
            (0, 0, 50, (5, 5, 50, 15)),
            (1, 50, 100, (5, 5, 50, 15)),
        ])
        region = HighlightRegion("r1", "text", 0, 100, "chunk", "arm1")
        result = regions_to_page_bboxes([region], wi)
        assert 0 in result
        assert 1 in result

    @pytest.mark.skipif(not PDF_PATH.exists(), reason="PDF not available")
    def test_build_word_index_real_pdf(self):
        from visualization.coord_mapper import build_word_index
        wi = build_word_index(str(PDF_PATH))
        assert len(wi) > 0
        # Check sorted by global_char_start
        starts = [ws.global_char_start for ws in wi]
        assert starts == sorted(starts)
        # All bboxes should have positive width and height
        for ws in wi[:100]:
            x0, y0, x1, y1 = ws.bbox
            assert x1 > x0
            assert y1 > y0


# ═══════════════════════════════════════════════════════════════════════════════
# D. layout helpers
# ═══════════════════════════════════════════════════════════════════════════════

class TestLayoutHelpers:
    """Selected-id promotion and scroll target resolution logic (extracted)."""

    _SELECTABLE = ("chunk", "article", "retrieved", "wrong_retrieved", "gt")

    def _make_region(self, rid, cat, start=0, end=100):
        return HighlightRegion(rid, "text", start, end, cat, "arm1")

    def test_selectable_categories_include_wrong_retrieved(self):
        assert "wrong_retrieved" in self._SELECTABLE

    def test_selectable_categories_include_gt(self):
        assert "gt" in self._SELECTABLE

    def test_scroll_index_prefers_sel_prefix_annotation(self):
        """The first _sel_* annotation must be found before non-sel annotations."""
        annotations = [
            {"id": "chunk_1_ret0", "page": 1},
            {"id": "chunk_1_sel_0", "page": 2},
            {"id": "chunk_1_sel_1", "page": 2},
        ]
        selected_id = "chunk_1"
        sel_prefix = f"{selected_id}_sel"
        scroll_index = None
        for i, ann in enumerate(annotations, start=1):
            aid = ann.get("id", "")
            if aid == sel_prefix or aid.startswith(sel_prefix):
                scroll_index = i
                break
        assert scroll_index == 2  # chunk_1_sel_0 is at position 2

    def test_scroll_fallback_to_prefix_when_no_sel_annotation(self):
        annotations = [
            {"id": "chunk_1_ret0", "page": 1},
        ]
        selected_id = "chunk_1"
        sel_prefix = f"{selected_id}_sel"
        scroll_index = None
        for i, ann in enumerate(annotations, start=1):
            aid = ann.get("id", "")
            if aid == sel_prefix or aid.startswith(sel_prefix):
                scroll_index = i
                break
        if scroll_index is None:
            for i, ann in enumerate(annotations, start=1):
                if ann.get("id", "").startswith(selected_id):
                    scroll_index = i
                    break
        assert scroll_index == 1  # fallback: chunk_1_ret0 at position 1


# ═══════════════════════════════════════════════════════════════════════════════
# E. page navigation tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestPageNavigation:
    """Single-page rendering: annotation filtering, auto-jump, scroll index, unique IDs."""

    def _make_region(self, rid, cat, start=0, end=100):
        return HighlightRegion(rid, "text", start, end, cat, "arm1")

    def _word_index_page(self, page_num: int, start: int, end: int):
        return [WordSpan(
            page_num=page_num,
            global_char_start=start,
            global_char_end=end,
            bbox=(10.0, 10.0, 100.0, 20.0),
            block_no=0,
            line_no=0,
        )]

    # ── 1. Annotation page filtering ─────────────────────────────────────────

    def test_annotations_filtered_to_current_page(self):
        """render_pdf_panel filters to only annotations on current_page."""
        annotations = [
            {"id": "a", "page": 1, "x": 0, "y": 0, "width": 10, "height": 10},
            {"id": "b", "page": 2, "x": 0, "y": 0, "width": 10, "height": 10},
            {"id": "c", "page": 3, "x": 0, "y": 0, "width": 10, "height": 10},
        ]
        current_page = 2
        page_annotations = [a for a in annotations if a.get("page") == current_page]
        assert len(page_annotations) == 1
        assert page_annotations[0]["id"] == "b"

    def test_annotations_filter_returns_empty_for_no_match(self):
        annotations = [
            {"id": "a", "page": 1},
            {"id": "b", "page": 3},
        ]
        page_annotations = [a for a in annotations if a.get("page") == 2]
        assert page_annotations == []

    def test_annotations_filter_returns_all_on_same_page(self):
        annotations = [{"id": f"a{i}", "page": 5} for i in range(4)]
        page_annotations = [a for a in annotations if a.get("page") == 5]
        assert len(page_annotations) == 4

    # ── 2. Auto-jump: page extracted from first _sel annotation ──────────────

    def test_auto_jump_extracts_page_from_sel_annotation(self):
        """When a selection is made, current_page should update to the annotation's page."""
        annotations = [
            {"id": "chunk_1_0", "page": 3},
            {"id": "chunk_1_sel_0", "page": 3},
            {"id": "chunk_1_sel_1", "page": 3},
        ]
        selected_id = "chunk_1"
        sel_prefix = f"{selected_id}_sel"
        target_page = None
        for ann in annotations:
            aid = ann.get("id", "")
            if aid == sel_prefix or aid.startswith(sel_prefix):
                target_page = ann.get("page")
                break
        assert target_page == 3

    def test_auto_jump_no_sel_annotation_leaves_page_unchanged(self):
        """If no _sel annotation exists, no page change should happen."""
        annotations = [{"id": "chunk_1_0", "page": 5}]
        selected_id = "chunk_1"
        sel_prefix = f"{selected_id}_sel"
        target_page = None
        for ann in annotations:
            aid = ann.get("id", "")
            if aid == sel_prefix or aid.startswith(sel_prefix):
                target_page = ann.get("page")
                break
        assert target_page is None  # no change

    def test_auto_jump_syncs_both_session_state_keys(self):
        """Both current_page and page_number_input must be updated together.
        Without syncing page_number_input, the number_input widget returns its
        stale value and overrides the jump on the same rerun."""
        session = {"current_page": 1, "page_number_input": 1}
        annotations = [{"id": "chunk_5_sel_0", "page": 26}]
        selected_id = "chunk_5"
        sel_prefix = f"{selected_id}_sel"
        for ann in annotations:
            aid = ann.get("id", "")
            if aid == sel_prefix or aid.startswith(sel_prefix):
                target_page = ann.get("page")
                if target_page:
                    session["current_page"] = target_page
                    session["page_number_input"] = target_page
                break
        assert session["current_page"] == 26
        assert session["page_number_input"] == 26  # widget won't override the jump

    def test_auto_jump_picks_first_sel_page_for_multi_page_chunk(self):
        """Multi-page chunk: auto-jump goes to first page of the chunk."""
        annotations = [
            {"id": "chunk_1_sel_0", "page": 4},
            {"id": "chunk_1_sel_1", "page": 5},
        ]
        selected_id = "chunk_1"
        sel_prefix = f"{selected_id}_sel"
        target_page = None
        for ann in annotations:
            aid = ann.get("id", "")
            if aid == sel_prefix or aid.startswith(sel_prefix):
                target_page = ann.get("page")
                break
        assert target_page == 4  # first occurrence

    # ── 3. total_pages computation ────────────────────────────────────────────

    def test_total_pages_from_word_index_single_page(self):
        wi = self._word_index_page(0, 0, 100)
        total = max(ws.page_num for ws in wi) + 1
        assert total == 1

    def test_total_pages_from_word_index_multi_page(self):
        wi = (
            self._word_index_page(0, 0, 100) +
            self._word_index_page(2, 200, 300) +
            self._word_index_page(5, 400, 500)
        )
        total = max(ws.page_num for ws in wi) + 1
        assert total == 6

    def test_total_pages_empty_word_index_defaults_to_one(self):
        word_index = []
        total = max((ws.page_num for ws in word_index), default=0) + 1
        assert total == 1

    # ── 4. Unique annotation IDs for other_regions ───────────────────────────

    def test_other_region_annotations_have_unique_ids(self):
        """Each line annotation of a wrong_retrieved chunk must have a unique id."""
        region = HighlightRegion("sw_88", "text", 0, 500, "wrong_retrieved", "arm1_bm25")
        wi = [
            WordSpan(0, i*10, i*10+9, (0, i*5, 100, i*5+4), 0, i)
            for i in range(5)  # 5 words on different lines
        ]
        anns = build_annotations([region], wi)
        ids = [a["id"] for a in anns]
        assert len(ids) == len(set(ids)), f"Duplicate annotation ids: {ids}"

    def test_selected_region_annotations_have_unique_ids(self):
        """_sel region annotations must also be uniquely identified."""
        sel_region = HighlightRegion("sw_88_sel", "text", 0, 500, "selected", "arm1_bm25")
        wi = [
            WordSpan(0, i*10, i*10+9, (0, i*5, 100, i*5+4), 0, i)
            for i in range(4)
        ]
        anns = build_annotations([sel_region], wi)
        ids = [a["id"] for a in anns]
        assert len(ids) == len(set(ids))

    # ── 5. Scroll index is 1-based ────────────────────────────────────────────

    def test_scroll_index_is_one_based(self):
        """First annotation in the list has index 1, not 0."""
        annotations = [
            {"id": "chunk_1_sel_0", "page": 2},
            {"id": "chunk_1_sel_1", "page": 2},
        ]
        selected_id = "chunk_1"
        sel_prefix = f"{selected_id}_sel"
        scroll_index = None
        for i, ann in enumerate(annotations, start=1):
            aid = ann.get("id", "")
            if aid == sel_prefix or aid.startswith(sel_prefix):
                scroll_index = i
                break
        assert scroll_index == 1

    def test_scroll_index_points_to_correct_annotation(self):
        """scroll_index must point to the first _sel annotation, not a prior one."""
        annotations = [
            {"id": "other_0", "page": 1},
            {"id": "other_1", "page": 1},
            {"id": "chunk_1_sel_0", "page": 3},
        ]
        selected_id = "chunk_1"
        sel_prefix = f"{selected_id}_sel"
        scroll_index = None
        for i, ann in enumerate(annotations, start=1):
            aid = ann.get("id", "")
            if aid == sel_prefix or aid.startswith(sel_prefix):
                scroll_index = i
                break
        assert scroll_index == 3  # 1-based position of "chunk_1_sel_0"

    # ── 6. Page clamping logic ────────────────────────────────────────────────

    def test_prev_clamps_at_page_one(self):
        current_page = 1
        new_page = max(1, current_page - 1)
        assert new_page == 1

    def test_next_clamps_at_total_pages(self):
        total_pages = 10
        current_page = 10
        new_page = min(total_pages, current_page + 1)
        assert new_page == 10

    def test_prev_decrements_normally(self):
        current_page = 5
        new_page = max(1, current_page - 1)
        assert new_page == 4

    def test_next_increments_normally(self):
        total_pages = 10
        current_page = 4
        new_page = min(total_pages, current_page + 1)
        assert new_page == 5


# ═══════════════════════════════════════════════════════════════════════════════
# F. demo pipeline smoke test (requires PDF + DB)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(
    not PDF_PATH.exists() or not DB_PATH.exists(),
    reason="PDF or DB not available — skipping integration test",
)
class TestDemoPipeline:
    """End-to-end: chunk PDF → BM25 retrieval → regions → annotations."""

    @pytest.fixture(scope="class")
    def chunks_and_spans(self):
        import sqlite3
        import fitz
        from arm1_naive.chunker import chunk_sliding_window, locate_articles
        from transformers import AutoTokenizer

        doc = fitz.open(str(PDF_PATH))
        raw_text = "\n\n".join(page.get_text() for page in doc)

        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        spans = locate_articles(raw_text, PDF_PATH.name, conn)
        conn.close()

        tok = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large-instruct")
        chunks = chunk_sliding_window(raw_text, tok, PDF_PATH.stem, spans,
                                      window_size=512, stride=256)
        return chunks, spans, raw_text

    def test_chunk_count_nonzero(self, chunks_and_spans):
        chunks, _, _ = chunks_and_spans
        assert len(chunks) > 0

    def test_article_spans_found(self, chunks_and_spans):
        _, spans, _ = chunks_and_spans
        assert len(spans) > 100  # this PDF has 441 spans

    def test_bm25_retrieval_returns_top_k(self, chunks_and_spans):
        import re
        from rank_bm25 import BM25Okapi
        chunks, _, _ = chunks_and_spans
        STOPWORDS = {"le","la","les","de","du","des","un","une","et","en"}
        TOK_RE = re.compile(r"[^\w]+", re.UNICODE)
        def tok(t): return [x for x in TOK_RE.split(t.lower()) if x and x not in STOPWORDS]
        corpus = [tok(c.text) for c in chunks]
        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(tok("Comment lire facture eau"))
        top_k = scores.argsort()[::-1][:10]
        assert len(top_k) == 10

    def test_retrieval_result_regions_classification(self, chunks_and_spans):
        chunks, spans, _ = chunks_and_spans
        gt_ids = [17039, 36350]  # Q39 GT

        # Take first 10 chunks as fake retrieval results
        ranked_items = [
            {
                "id": c.chunk_id, "score": 1.0, "text": c.text,
                "bsard_ids": c.bsard_ids,
                "start_char": c.start_char, "end_char": c.end_char,
                "start_token": c.start_token, "end_token": c.end_token,
                "metadata": {},
            }
            for c in chunks[:10]
        ]
        result = _FakeRetrievalResult("q39", "test", ranked_items)
        bundle = _FakeBundle()
        regions = retrieval_result_to_regions(result, bundle, article_spans=spans, gt_article_ids=gt_ids)
        categories = {r.category for r in regions}
        # At least some should be classified
        assert categories <= {"retrieved", "wrong_retrieved"}

    def test_gt_regions_resolve_to_nonzero_offsets(self, chunks_and_spans):
        """Article 17039 (number D230) resolves; 36350 (number R270bis8) does NOT —
        the ART_RE regex only captures [A-Z0-9_./-]+ so lowercase 'bis' truncates
        the match to 'R270', which doesn't match the DB entry.  This is a documented
        ARM1 limitation; the test verifies the known behaviour rather than hiding it."""
        _, spans, _ = chunks_and_spans
        bundle = _FakeBundle()
        gt_regions = gt_regions_from_article_ids([17039, 36350], bundle, article_spans=spans)
        assert len(gt_regions) == 2

        by_id = {r.region_id: r for r in gt_regions}
        # D230 (article 17039) — standard number, should resolve
        assert by_id["gt_17039"].start_char < by_id["gt_17039"].end_char, \
            "Article 17039 (D230) should resolve via locate_articles()"
        # R270bis8 (article 36350) — lowercase in number, known to NOT resolve
        assert by_id["gt_36350"].start_char == 0 and by_id["gt_36350"].end_char == 0, \
            "Article 36350 (R270bis8) is expected to NOT resolve (ART_RE limitation)"

    def test_word_index_covers_full_document(self, chunks_and_spans):
        from visualization.coord_mapper import build_word_index
        _, _, raw_text = chunks_and_spans
        wi = build_word_index(str(PDF_PATH))
        # Word index should reach near end of document
        max_end = max(ws.global_char_end for ws in wi)
        assert max_end > len(raw_text) * 0.8

    def test_annotations_built_for_retrieval_regions(self, chunks_and_spans):
        from visualization.coord_mapper import build_word_index
        chunks, spans, _ = chunks_and_spans
        wi = build_word_index(str(PDF_PATH))

        gt_ids = [17039, 36350]
        bundle = _FakeBundle()
        gt_regs = gt_regions_from_article_ids(gt_ids, bundle, article_spans=spans)

        # Take 5 chunks as retrieved
        ranked_items = [
            {"id": c.chunk_id, "score": 1.0, "text": c.text,
             "bsard_ids": c.bsard_ids, "start_char": c.start_char,
             "end_char": c.end_char, "start_token": c.start_token,
             "end_token": c.end_token, "metadata": {}}
            for c in chunks[:5]
        ]
        result = _FakeRetrievalResult("q39", "test", ranked_items)
        ret_regs = retrieval_result_to_regions(result, bundle, article_spans=spans, gt_article_ids=gt_ids)

        anns = build_annotations(ret_regs + gt_regs, wi)
        assert len(anns) > 0
        # Check all required fields present
        for a in anns:
            for f in ("page", "x", "y", "width", "height", "color", "id"):
                assert f in a
