# 08 — RAG VISUALIZATION CONTEXT
## Interactive PDF Viewer for Chunks, Articles, and Retrieval Inspection

---

## 1. PURPOSE

A local, single-user tool for visually inspecting what chunkers, article locators, and retrievers actually produce **on the original PDF page layout**. It consumes outputs from every retrieval arm (Arm 1 chunks, Arm 2A/B/C articles) through a uniform interface and overlays them on the real PDF, with side panels showing the question text, the selected span's text, and any reasoning traces.

The viewer is a **debugging and thesis-figure tool**, not part of the evaluation pipeline. It imports from the arms; nothing imports from it. It never writes to the evaluation DB, results directories, or cached indices.

Two modes:
1. **Review mode** — browse every chunk (Arm 1) or article (Arm 2) of a document on the PDF page.
2. **Retrieval mode** — for a given query, overlay ground-truth article spans (blue), retrieved items (green), and their overlap (teal), with the ranked list in a sidebar.

## 2. DIRECTORY STRUCTURE

```
visualization/
├── __init__.py
├── app.py                   # Streamlit entry point (streamlit run app.py)
├── launcher.py              # Public launch() function for orchestrator / CLI
├── coord_mapper.py          # char-offset → (page, bbox) mapping via PyMuPDF words index
├── region_adapter.py        # Unifies Chunk / Article / RetrievalResult → HighlightRegion
├── annotation_builder.py    # Builds streamlit-pdf-viewer annotation dicts + overlap layering
├── query_loader.py          # Loads queries + GT article_ids from the ground-truth store
├── trace_renderer.py        # Formats PageIndex trace for side panel
├── index_cache.py           # Loads cached FAISS+BM25 bundles from data/indices/<arm>/
├── t03_loader.py            # Adapt T03 Arm 1 chunks/results for the viewer
├── t04_loader.py            # Adapt T04 Arm 2A node/article results for the viewer
├── t05_loader.py            # Adapt T05 Arm 2B tree + navigation trace for the viewer
└── ui/
    ├── __init__.py
    ├── layout.py            # Three-column layout (sidebar | PDF | text panels)
    ├── sidebar_review.py    # Chunk / article list panel
    ├── sidebar_retrieval.py # Query box + ranked results
    ├── text_panels.py       # Question, selected item, GT, trace panels
    ├── tree_explorer.py     # T05 ToC-tree explorer panel
    └── legend.py            # Color legend
```

## 3. DEPENDENCIES

- Uses: `shared.embeddings`, `shared.faiss_store`, `shared.bm25_store` (transitively, via arm retrievers)
- Uses: `data_loader.models` (`DocumentBundle`, `Article`, `Chunk`, `RetrievalResult`)
- Uses: `arm1_naive.chunker` for `chunk_sliding_window` / `chunk_recursive` / `locate_articles`
- Uses: `arm1_naive.retriever`, `arm2_metadata`, `arm2_pageindex` — only to replay a query against a cached index. The viewer never builds an index itself.
- External: `streamlit`, `streamlit-pdf-viewer` (PDF.js-backed, pinned), `PyMuPDF` (already a project dependency), `pyyaml`, `pandas`
- **No new system-level dependencies.** No LibreOffice, no pdf2htmlEX, no Qt, no separate JS build step.

## 4. SCOPE

| In scope | Out of scope |
|---|---|
| Rendering the original PDF page by page | Embedding-space scatterplots (UMAP/t-SNE) |
| Colored bbox overlays for chunks, articles, GT, retrieved items | Cluster visualizations |
| Side panels: question text, selected item text, GT articles, PageIndex trace | LLM trace dashboards, span profilers |
| Multi-selection in review mode (e.g., inspect sliding-window overlap) | Metric dashboards (that's 07_EVALUATION) |
| Per-arm color coding | Answer generation or chatbot UI |
| Interactive re-chunking (change strategy/params → see result) | Mutating pipeline artifacts |

The PDF page layout is the primary visual surface. Every other UI element is secondary. If a tool can't put colored bboxes on the rendered PDF page, it doesn't belong.

## 5. TOOL SURVEY & STACK DECISION

Restricted to tools that overlay regions on the **original PDF page layout**. Embedding-space and trace-view tools (RAGxplorer, Arize Phoenix, LangSmith, Langfuse, rag-visualizer) are excluded by definition and not considered further.

### 5.1 Considered contenders

| Tool | Category | Multi-color | Reactive | Notes |
|---|---|---|---|---|
| `rag-document-viewer` (preprocess-co, MIT) | Purpose-built for RAG PDF highlights | ❌ single color per viewer | ❌ static HTML bundle generation | Closest shape to the target but fundamentally offline: a bundle is regenerated for every chunking change. Requires LibreOffice + pdf2htmlEX as OS-level deps. |
| Verba (Weaviate, BSD-3) | Full RAG chatbot w/ source highlighting | ✅ | ✅ | Owns the whole pipeline (Weaviate, its own chunkers/embedders). Cannot plug in this project's arms. Rejected. |
| `streamlit-pdf-viewer` (lfoppiano, Apache-2.0) + PyMuPDF + custom Streamlit app | Building block | ✅ per-annotation `color` | ✅ native Streamlit rerun | De-facto Python standard. Supports `on_annotation_click`, `scroll_to_annotation`, page filtering. **Chosen.** |
| `react-pdf-viewer` / `react-pdf-highlighter-extended` | Building block | ✅ | ✅ | Require a full React frontend; disproportionate plumbing. |
| `gradio-pdf` | Building block | ❌ no native bbox overlay | — | Would force pixmap rasterization; inferior UX. |

### 5.2 Decision

Build on **Streamlit + streamlit-pdf-viewer + PyMuPDF**, with all arm retrievers accessed through the existing sub-project interfaces. This is the only combination that satisfies four hard requirements: (a) multi-color overlays for GT/retrieved/overlap, (b) reactive chunker parameter tuning, (c) compatibility with the existing `DocumentBundle` / `Article` / `Chunk` / `RetrievalResult` data model, and (d) no new system-level dependencies.

The `rag-document-viewer` path is retained as a documented fallback (§ 11.7) in case streamlit-pdf-viewer breaks mid-thesis — accepting its single-color, static-bundle limitations.

## 6. DATA MODEL

The viewer introduces a single internal type that every arm's output gets mapped into before rendering.

```python
from dataclasses import dataclass, field
from typing import Literal, Optional

HighlightCategory = Literal[
    "chunk",          # Arm 1 chunk (review mode)
    "article",        # Arm 2 article (review mode)
    "gt",             # Ground-truth article span (retrieval mode)
    "retrieved",      # Retrieved item from any arm (retrieval mode)
    "overlap",        # Geometric intersection of gt ∩ retrieved
    "selected",       # User-selected item in review mode
]

@dataclass
class HighlightRegion:
    """
    Uniform representation across all arms. Everything the viewer renders
    starts as a HighlightRegion, regardless of whether it originated as a
    Chunk or an Article.
    """
    region_id: str                         # display id (e.g., "chunk_42", "art_D.35")
    text: str                              # the span text (shown in side panel)
    start_char: int                        # char offset in bundle.raw_text
    end_char: int
    category: HighlightCategory
    arm: str                               # "arm1", "2A-full", "2A-filtered", "2B", ...
    score: Optional[float] = None          # retrieval score if applicable
    rank: Optional[int] = None             # 1-based rank if from a ranked list
    start_token: Optional[int] = None      # when available (Arm 1 chunks)
    end_token: Optional[int] = None
    metadata: dict = field(default_factory=dict)  # anything else (e.g., source='hop_1', trace_id)
```

### 6.1 Adapters (region_adapter.py)

```python
def chunks_to_regions(chunks: list[Chunk], arm: str = "arm1") -> list[HighlightRegion]:
    """Arm 1: Chunk objects already carry start_char/end_char — trivial map."""
    ...

def articles_to_regions(articles: list[Article],
                        bundle: DocumentBundle,
                        arm: str) -> list[HighlightRegion]:
    """
    Arm 2: Article objects have no char offsets. Locate each article in
    bundle.raw_text via arm1_naive.chunker.locate_articles (which returns
    ArticleSpan with start_char/end_char). Cached per bundle.
    """
    ...

def retrieval_result_to_regions(result: RetrievalResult,
                                bundle: DocumentBundle) -> list[HighlightRegion]:
    """
    Generic adapter for any arm's output. Walks result.ranked_items and
    resolves each id back to a Chunk or Article in the bundle (or in the
    arm's cached index metadata).
    """
    ...
```

The viewer never needs to know which arm produced a result — the adapter erases that distinction. The `arm` string is kept only for legend/trace display.

## 7. CORE TECHNICAL DESIGN

### 7.1 Char-offset → page-coordinate mapping (coord_mapper.py)

The raw text the pipeline uses is `"\n\n".join(page.get_text() for page in doc)` (see `02_DATA_LOADER/pdf_loader.py`). `HighlightRegion` carries `start_char`/`end_char` into that flat string. The viewer maps these back to `(page_number, bbox)` via a **word-level index** built once per PDF.

```python
@dataclass
class WordSpan:
    page_num: int                  # 0-based
    global_char_start: int         # offset into bundle.raw_text
    global_char_end: int
    bbox: tuple[float, float, float, float]  # (x0, y0, x1, y1) in PDF points
    block_no: int
    line_no: int

def build_word_index(pdf_path: str) -> list[WordSpan]:
    """
    One-time build. For each page: page.get_text("words") returns
    (x0, y0, x1, y1, word, block_no, line_no, word_no). In parallel,
    walk the page's get_text() output with a moving pointer to find
    each word's char span in the flat text. Offset by cumulative
    per-page starts (plus "\n\n" separator length) to produce global
    char positions.

    Returns word_spans sorted by global_char_start. Cached via
    @st.cache_resource keyed by (pdf_path, mtime).
    """
    ...

def regions_to_page_bboxes(regions: list[HighlightRegion],
                           word_index: list[WordSpan]
                           ) -> dict[int, list[tuple[HighlightRegion, Rect]]]:
    """
    For each region, binary-search word_index by global_char_start/end,
    group matching words by page_num, then merge adjacent word bboxes
    within the same block_no/line_no (one rectangle per visual line,
    not per word). Multi-page regions emit bboxes on multiple pages.
    """
    ...
```

**Why not `page.search_for(region.text)`?** Unreliable on chunks with hyphenation, whitespace-normalisation drift, column breaks, or repeated phrases. The word-index approach is deterministic and has no false matches.

**Why not `get_text("rawdict")` (per-character)?** Strictly more data than needed — word granularity suffices and is ~10× cheaper. Reserve `rawdict` as fallback for pathological PDFs only.

### 7.2 Annotation builder (annotation_builder.py)

`streamlit-pdf-viewer` accepts Grobid-style dicts:

```python
{
    "page": int,      # 1-based (not 0-based)
    "x": float, "y": float,              # top-left, PDF points
    "width": float, "height": float,
    "color": "#RRGGBB",
    "border": "solid",
    "id": str,
}
```

```python
def build_annotations(regions: list[HighlightRegion],
                      word_index: list[WordSpan],
                      color_map: dict[HighlightCategory, str]
                      ) -> list[dict]:
    """
    Full pipeline:
    1. Map regions → page bboxes via coord_mapper.
    2. Compute overlap layer (see 7.3).
    3. Emit one annotation dict per (region, page_slice, line_bbox).
    4. Attach region.region_id to annotation.id for click callbacks.
    """
    ...
```

### 7.3 Color scheme and overlap layering

| Category | Color | Opacity | When |
|---|---|---|---|
| `gt` | `#3B82F6` blue | 0.25 | retrieval mode |
| `retrieved` | `#10B981` green | 0.25 | retrieval mode |
| `overlap` | `#0D9488` teal | 0.35 | retrieval mode (computed) |
| `selected` | `#F59E0B` amber | 0.25 | review mode |
| `chunk` / `article` | `#9CA3AF` gray | 0.12 | review mode (all items, faint) |

**Overlap is computed geometrically at build time, not via CSS blending.** For each page: compute the intersection of `gt` rects and `retrieved` rects, emit it as a separate `overlap` annotation, and subtract it from both the `gt` and `retrieved` sets. This prevents translucent rectangles from additively stacking into visual noise.

**Arm-specific extensions**:
- **PageIndex (2B)**: retrieved items have a parent chapter/section from the trace. The annotation tooltip includes the trace path (e.g., `"Book II / Title 3 / Chapter 2"`).
- (A previous GraphRAG / 2C source-coloring extension exists as guarded dead code in `annotation_builder.py` and `trace_renderer.py`; never executes because 2C was dropped from RQ2. Removal is a cleanup pass; leaving it in is harmless.)

### 7.4 Text side panels (text_panels.py)

Plain Streamlit widgets, populated on every rerun from session state:

- **Question panel** (retrieval mode): the query text verbatim + `relevant_article_ids` from ground truth.
- **Selected item panel** (both modes): `region.text`, `region.region_id`, char + token ranges, page range, retrieval score + rank if applicable.
- **GT article panel** (retrieval mode): each GT article's id, title (if in `bundle.metadata`), and full text.
- **Trace panel** (retrieval mode, 2B only): rendered PageIndex trace — law selection → chapter selection → final result, via `trace_renderer.format_pageindex_trace(result.trace)`.

### 7.5 Performance

- **Word-index build**: once per PDF, `@st.cache_resource` keyed by `(pdf_path, mtime)`. < 1 s for a 400-page BSARD law.
- **Chunking / article location**: on demand when strategy or parameters change, `@st.cache_data` keyed by `(pdf_path, strategy, params)`.
- **Retrieval**: uses cached indices from `data/indices/<arm>/<doc_id>/`. Loaded once at app start via `@st.cache_resource`. Queries are sub-second after warm-up.
- **No re-rendering of PDF pages on selection change** — `streamlit-pdf-viewer` updates only its annotation layer; PDF.js keeps rendered pages cached in the browser. This is the main reason to prefer it over any pixmap approach.

## 8. UI LAYOUT

Three-column Streamlit layout: sidebar (controls + list), main panel (PDF viewer), right panel (text inspection).

```
┌──────────────────────────────────────────────────────────────────────────┐
│  Mode: ( • Review ) ( ○ Retrieval )  Arm: [ arm1 ▼ ]   [ Legend ▸ ]      │
├────────────────────────┬──────────────────────────┬──────────────────────┤
│  PDF: [ dropdown ▼ ]   │                          │  ── Question ──      │
│  Strategy: [ ▼ ]       │                          │  [ question text ]   │
│  window: 512           │                          │                      │
│  stride: 256           │                          │  GT articles:        │
│  [ Re-chunk ]          │       PDF VIEWER         │  • art. D.35         │
│                        │   (streamlit-pdf-viewer, │  • art. D.49         │
│  ─── Items (147) ───── │    scrollable, zoomable, │                      │
│  ▸ chunk_0   p.1–2     │    annotated in place)   │  ── Selected ──      │
│  ▸ chunk_1   p.2       │                          │  chunk_42 (rank 1)   │
│  ▸ chunk_2   p.2–3     │                          │  score: 0.912        │
│                        │                          │  tokens: 12288–12800 │
│  [ Retrieval mode ]    │                          │  pages: 47–48        │
│  Query: [__________]   │                          │  [ full text ]       │
│  ── Ranked top-10 ──   │                          │                      │
│  1. chunk_42 (0.91) ✓  │                          │  ── Trace (2B) ──    │
│  2. chunk_17 (0.87)    │                          │  Book II > Title 3   │
│  3. chunk_91 (0.82) ✓  │                          │  > Chapter 2         │
└────────────────────────┴──────────────────────────┴──────────────────────┘
```

- Clicking an item in the sidebar toggles selection; annotations are recomputed and PDF.js scrolls to the first via `scroll_to_annotation=<id>`. The right panel fills with that item's text and metadata.
- ✓ next to a retrieved item means it overlaps a GT span.
- The right panel is collapsible for maximum PDF real-estate.
- When `method` starts with `2B-`, the trace panel is shown; when it's `arm1` or `2A-*`, neither.

## 9. INTERFACE FOR ORCHESTRATOR

The viewer is primarily a **standalone Streamlit app** (`streamlit run app.py`). The orchestrator interacts with it in two ways.

### 9.1 Launch helper (launcher.py)

```python
def launch(
    bundle: DocumentBundle,
    mode: Literal["review", "retrieval"] = "review",
    arm: str = "arm1",                         # "arm1" | "2A-full" | "2A-filtered" | "2B" | ...
    retrieval_results: list[RetrievalResult] | None = None,
    queries: list[Query] | None = None,        # pre-populate dropdown
    arm_index_dir: Path | None = None,         # required if retrieving fresh queries
    config: dict | None = None,
    port: int = 8501,
    open_browser: bool = True,
) -> subprocess.Popen:
    """
    Spawn streamlit run app.py in a subprocess. Bundle and options are
    passed via a temp JSON side-file (standard Streamlit arg pattern):
        streamlit run app.py -- --bundle-json=<tmp>.json
    Returns the Popen handle so the caller can terminate it.
    """
    ...
```

### 9.2 Running the viewer

The shipped entry points are:

- **Standalone app:** `streamlit run src/visualization/app.py` (optionally
  `-- --bundle-json <path>` to load a T03-baked per-question bundle).
- **`launch()`** (§9.1) from another script.

> **Not implemented:** there is no `python -m visualization` CLI (no `__main__.py`) and no
> orchestrator `--inspect` flag — the `run_experiment.py` `--inspect` integration described
> in earlier drafts was never wired in. The viewer is launched standalone or via `launch()`.

## 10. CONFIGURATION

> **Status:** not implemented as a file. There is no `config.yaml` in this project and no
> `visualization:` block in the orchestrator config; `launch(config=...)` accepts a dict but
> the viewer currently ignores it (reserved). Colors/defaults are set in code. The schema
> below is the intended shape if/when config is externalised.

```yaml
# 08_VISUALIZATION
visualization:
  # Colors (hex, no alpha — opacity is applied by the viewer at render time)
  color_gt: "#3B82F6"
  color_retrieved: "#10B981"
  color_overlap: "#0D9488"
  color_selected: "#F59E0B"
  color_background_item: "#9CA3AF"

  # UI defaults
  default_mode: "review"           # or "retrieval"
  pdf_viewer_width: 900
  show_trace_panel_for: ["2B"]     # method prefixes that get the trace panel

  # Performance
  use_lightweight_models: false    # if true, swap to bge-small for viewer-only retrieval

  # Streamlit runtime
  port: 8501
  open_browser: true

  # streamlit-pdf-viewer pin — do not float
  streamlit_pdf_viewer_version: "0.0.19"
```

## 11. IMPORTANT NOTES

- **Read-only wrt the pipeline.** The viewer must never mutate chunks, indices, the ground-truth DB, or the bundle. All state is ephemeral Streamlit session state.
- **Reuse, don't reimplement.** Chunking, article location, retrieval — all imported from the arms via `arm1_naive.*`, `arm2_metadata.*`, `arm2_pageindex.*`. If logic changes in an arm, the viewer picks it up automatically.
- **Self-consistent coord mapping.** The viewer rebuilds its `word_spans` table from the PDF itself, not from a serialized offset table. A bundle produced by an older extractor version still renders correctly at the cost of a ~1 s rebuild per PDF.
- **No LLM calls introduced by the viewer.** Retrieval mode calls the same stacks the arms use. For Arm 1 and Arm 2A, cost remains 0. For Arm 2B (PageIndex), the viewer can *either* load pre-computed `RetrievalResult.trace` objects (cost 0) *or* replay a query live (incurs the arm's LLM cost). Live replay is opt-in via a UI toggle; the default is to display cached results.
- **Not part of evaluation.** The source of truth for metrics remains `07_EVALUATION`. The viewer imports from the arms; the evaluator imports from the arms; the viewer and the evaluator do not import each other.
- **Arm 2 article positions are resolved on-demand**, using `arm1_naive.chunker.locate_articles(bundle.raw_text, articles)`. If an article cannot be located (e.g., Azure extraction text doesn't match the raw PDF text exactly), it is marked with a warning icon in the sidebar and rendered as a "whole-page" annotation on its best-guess page as a fallback.
- **Performance budget.** Opening a 400-page PDF + chunking + rendering < 3 s on the thesis laptop (CPU). Query retrieval (cached index, no LLM) < 2 s after warm-up. Live PageIndex replay is unbounded but typically 10–30 s.
- **Fallback plan.** If `streamlit-pdf-viewer` breaks in an unrecoverable way, switch to `rag-document-viewer` (preprocess-co, MIT) with two acceptances: (a) chunking parameters become fixed at bundle-generation time, and (b) multi-color overlays require generating one bundle per color category and layering them in an HTML wrapper. Ugly but functional.
