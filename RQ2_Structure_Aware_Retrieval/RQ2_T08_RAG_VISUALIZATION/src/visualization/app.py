"""Streamlit entry point for the RAG Visualization viewer.

Launch:
    streamlit run src/visualization/app.py
    streamlit run src/visualization/app.py -- --bundle-json=/tmp/bundle.json
    streamlit run src/visualization/app.py -- --pdf data/pdfs/2.pdf --db data/bsard_corpus.db
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="RAG Visualization",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Resolve project root so sibling packages are importable ──────────────────
_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent
_PROJECT_ROOT = _SRC.parent.parent  # .../RQ2_Structure_Aware_Retrieval

for _p in [str(_SRC), str(_PROJECT_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Also add sibling src directories for arm packages
for _sibling in _PROJECT_ROOT.glob("RQ2_T0*/src"):
    _s = str(_sibling)
    if _s not in sys.path:
        sys.path.insert(0, _s)

# ── Internal imports ──────────────────────────────────────────────────────────
from visualization.coord_mapper import build_word_index
from visualization.query_loader import (
    Query,
    load_bsard_id_to_pdf,
    load_queries,
    queries_for_document,
)
from visualization.region_adapter import (
    HighlightRegion,
    article_spans_to_regions,
    articles_to_regions,
    chunks_to_regions,
    gt_regions_from_bsard_ids,
    retrieval_result_to_regions,
)
from visualization.t03_loader import (
    T03Config,
    discover_t03_configs,
    doc_dir_for_pdf as t03_doc_dir_for_pdf,
    pick_default_config as pick_default_t03_config,
)
from visualization.t04_loader import (
    T04Variant,
    discover_t04_variants,
    doc_dir_for_pdf,
    load_t04_results,
    t04_dict_to_retrieval_result,
)
from visualization.t05_loader import (
    T05Coverage,
    discover_t05_coverage,
    load_t05_result,
    load_tree as load_t05_tree,
    t05_dict_to_retrieval_result,
)
from visualization.t04c_loader import (
    Arm2cCoverage,
    arm2c_dict_to_retrieval_result,
    discover_arm2c_coverage,
    load_arm2c_result,
    load_deep_tree,
)
from visualization.ui.layout import render_main_layout, render_top_bar
from visualization.ui.sidebar_review import render_review_sidebar
from visualization.ui.sidebar_retrieval import render_retrieval_sidebar
from visualization.ui.tree_explorer import render_deep_tree_explorer, render_tree_explorer


# ── CLI arg parsing (Streamlit passes extra args after --) ───────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--bundle-json", default=None)
    parser.add_argument("--pdf", default=None)
    parser.add_argument("--db", default=None)
    parser.add_argument("--mode", default="retrieval")
    parser.add_argument("--arm", default="arm1")
    parser.add_argument("--index-dir", default=None)
    parser.add_argument(
        "--t03-data-dir",
        default=None,
        help="Override path to RQ2_T03_ARM1_NAIVE/data (per-doc cached indices).",
    )
    parser.add_argument(
        "--t04-data-dir",
        default=None,
        help="Override path to RQ2_T04_ARM2_METADATA/data (per-doc cached results).",
    )
    parser.add_argument(
        "--t05-data-dir",
        default=None,
        help="Override path to RQ2_T05_ARM2_PAGEINDEX/data (per-doc cached results + trees).",
    )
    parser.add_argument(
        "--arm2c-data-dir",
        default=None,
        help="Override path to RQ2_T04_ARM2_METADATA/data/_arm2c (exported Arm2C results + deep trees).",
    )
    parser.add_argument("--port", type=int, default=8501)
    # Streamlit injects its own args before the -- separator; ignore unknown
    known, _ = parser.parse_known_args(sys.argv[1:])
    return known


# ── Cached resource builders ──────────────────────────────────────────────────

@st.cache_resource(show_spinner="Building word index …")
def _get_word_index(pdf_path: str, _mtime: float):
    return build_word_index(pdf_path)


@st.cache_data(show_spinner="Chunking …")
def _get_chunks(pdf_path: str, raw_text: str, strategy: str, window_size: int, stride: int, max_tokens: int, doc_stem: str):
    try:
        from transformers import AutoTokenizer
    except ImportError:
        st.error("transformers not installed — cannot chunk.")
        return []

    try:
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    except Exception as exc:
        st.error(f"Could not load tokenizer: {exc}")
        return []

    from arm1_naive.chunker import chunk_recursive, chunk_sliding_window

    if strategy == "sliding_window":
        return chunk_sliding_window(raw_text, tokenizer, doc_stem, window_size=window_size, stride=stride)
    else:
        return chunk_recursive(raw_text, tokenizer, doc_stem, max_tokens=max_tokens)


@st.cache_data(show_spinner="Mapping BSARD ids → PDFs …")
def _get_bsard_to_pdf(db_path: str) -> dict[int, str]:
    return load_bsard_id_to_pdf(db_path) if db_path else {}


@st.cache_data(show_spinner="Loading article texts …")
def _get_gt_article_data(db_path: str, bsard_ids: tuple) -> dict:
    """Return {bsard_id: {text, article_number, law_title}} for the given bsard ids.

    Multiple `articles` rows may share the same bsard_id (canonical→raw fan-out);
    we pick the first row per bsard_id, which is sufficient for display.
    """
    if not db_path or not bsard_ids:
        return {}
    import sqlite3
    placeholders = ",".join("?" * len(bsard_ids))
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            f"SELECT bsard_id, article_text, article_number, law_title_text "
            f"FROM articles WHERE bsard_id IN ({placeholders})",
            list(bsard_ids),
        ).fetchall()
        out: dict = {}
        for r in rows:
            bs = r["bsard_id"]
            if bs in out:
                continue
            out[bs] = {
                "text": r["article_text"] or "",
                "article_number": r["article_number"] or "",
                "law_title": r["law_title_text"] or "",
            }
        return out
    finally:
        conn.close()


@st.cache_data(show_spinner="Locating articles …")
def _get_article_spans(raw_text: str, pdf_filename: str, db_path: str):
    if not db_path or not Path(db_path).exists():
        return []
    import sqlite3
    from arm1_naive.chunker import locate_articles
    conn = sqlite3.connect(db_path)
    try:
        return locate_articles(raw_text, pdf_filename, conn)
    except Exception:
        return []
    finally:
        conn.close()


def _article_spans_from_bundle(bundle_json_data: dict | None) -> list:
    """Build ArticleSpan objects from a bundle JSON's `article_spans` field.

    Used when running off a saved preprocessed bundle (no live DB needed).
    """
    if not bundle_json_data:
        return []
    raw = bundle_json_data.get("article_spans") or []
    if not raw:
        return []
    try:
        from arm1_naive.chunker import ArticleSpan
    except ImportError:
        return []
    out = []
    for s in raw:
        try:
            out.append(
                ArticleSpan(
                    bsard_id=int(s["bsard_id"]),
                    start_char=int(s["start_char"]),
                    end_char=int(s["end_char"]),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return out


# ── Bundle loading helpers ────────────────────────────────────────────────────

def _load_bundle_from_json(json_path: str):
    """Load a DocumentBundle serialised by launcher.py."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    try:
        import pandas as pd
        from data_loader.models import Article, DocumentBundle
    except ImportError:
        st.error("data_loader package not installed.")
        return None

    articles = [Article(**a) for a in data.get("articles", [])]
    bundle = DocumentBundle(
        document_id=data["document_id"],
        pdf_path=data["pdf_path"],
        pdf_filename=data["pdf_filename"],
        metadata=data.get("metadata", {}),
        definitions=pd.DataFrame(data.get("definitions", [])),
        articles=articles,
        raw_text=data.get("raw_text", ""),
        has_azure_extraction=data.get("has_azure_extraction", False),
    )
    return bundle


def _read_pdf_bytes(pdf_path: str) -> bytes | None:
    p = Path(pdf_path)
    if not p.exists():
        return None
    return p.read_bytes()


# ── Session-state helpers ─────────────────────────────────────────────────────

def _ss(key: str, default=None):
    return st.session_state.get(key, default)


def _init_state():
    defaults = {
        "mode": "retrieval",
        "arm": "arm1",
        "selected_id": None,
        "regions": [],
        "gt_regions": [],
        "active_query_text": "",
        "active_query_id": None,
        "relevant_article_ids": [],
        "ranked_items": [],
        "trace": None,
        "rechunk_requested": False,
        "retrieve_requested": False,
        "strategy": "sliding_window",
        "window_size": 512,
        "stride": 256,
        "max_tokens": 512,
        "current_page": 1,
        "top_k": 10,
        "top_k_input": 10,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ── Sidebar: PDF selector ─────────────────────────────────────────────────────

def _render_pdf_selector(args: argparse.Namespace, bundle_json_data) -> tuple[str | None, str | None]:
    """Returns (pdf_path, db_path) from sidebar widgets or pre-loaded args."""
    with st.sidebar:
        st.markdown("## RAG Visualization")
        st.divider()

        pdf_path = args.pdf or (bundle_json_data.get("pdf_path") if bundle_json_data else None)
        db_path = args.db or (bundle_json_data.get("db_path") if bundle_json_data else None)

        pdf_input = st.text_input("PDF path", value=pdf_path or "", key="pdf_path_input")
        db_input = st.text_input("DB path (optional)", value=db_path or "", key="db_path_input")

        # Quick PDF scanner — look for PDFs in data/ relative to CWD
        data_dir = Path.cwd() / "data" / "pdfs"
        if data_dir.exists():
            pdf_files = sorted(data_dir.glob("*.pdf"))
            if pdf_files:
                names = [p.name for p in pdf_files]
                picked = st.selectbox("…or pick from data/pdfs/", ["(enter path above)"] + names, key="pdf_picker")
                if picked != "(enter path above)":
                    pdf_input = str(data_dir / picked)

        return (pdf_input or None, db_input or None)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    _init_state()
    args = _parse_args()

    # Patch streamlit-pdf-viewer's bundled index.html so our highlight CSS
    # (translucent fills, multiply blend, selected bounding box) loads
    # *inside* the component's iframe. Idempotent — re-runs replace a
    # marker block, so palette tweaks in annotation_builder.py propagate
    # on the next app start.
    from visualization.annotation_builder import patch_pdf_viewer_index_html
    patch_pdf_viewer_index_html()

    # Load bundle from JSON side-file if launcher passed one
    bundle_json_data = None
    bundle = None
    if args.bundle_json and Path(args.bundle_json).exists():
        bundle = _load_bundle_from_json(args.bundle_json)
        bundle_json_data = json.loads(Path(args.bundle_json).read_text(encoding="utf-8"))

        # Pre-populate session state from the bundle JSON (demo / launcher path)
        # Only do this once per session (guard with a flag)
        if not st.session_state.get("_bundle_loaded"):
            if bundle_json_data.get("initial_mode"):
                st.session_state["mode"] = bundle_json_data["initial_mode"]
            if bundle_json_data.get("initial_arm"):
                st.session_state["arm"] = bundle_json_data["initial_arm"]
            if bundle_json_data.get("active_query_text"):
                st.session_state["active_query_text"] = bundle_json_data["active_query_text"]
            if bundle_json_data.get("active_query_id"):
                st.session_state["active_query_id"] = bundle_json_data["active_query_id"]
            if bundle_json_data.get("gt_article_ids"):
                st.session_state["relevant_article_ids"] = bundle_json_data["gt_article_ids"]
            if bundle_json_data.get("top_k"):
                st.session_state["top_k"] = bundle_json_data["top_k"]
                st.session_state["top_k_input"] = bundle_json_data["top_k"]
            st.session_state["_bundle_loaded"] = True

    pdf_path, db_path = _render_pdf_selector(args, bundle_json_data)

    if not pdf_path or not Path(pdf_path).exists():
        st.info("Enter a valid PDF path in the sidebar to start.")
        return

    # PDF bytes for the viewer
    pdf_bytes = _read_pdf_bytes(pdf_path)
    if pdf_bytes is None:
        st.error(f"Cannot read PDF: {pdf_path}")
        return

    # Word index (cached by path + mtime)
    mtime = Path(pdf_path).stat().st_mtime
    word_index = _get_word_index(pdf_path, mtime)

    # Determine raw_text — from bundle or by extracting it ourselves
    if bundle is None or not bundle.raw_text:
        try:
            import fitz
            doc = fitz.open(pdf_path)
            raw_text = "\n\n".join(page.get_text() for page in doc)
            pdf_filename = Path(pdf_path).name
        except Exception as exc:
            st.error(f"Could not extract text: {exc}")
            return
    else:
        raw_text = bundle.raw_text
        pdf_filename = bundle.pdf_filename

    doc_stem = Path(pdf_filename).stem

    # Arm list — always include arm1; append arm2 variants discovered from
    # cached results, then any remaining placeholder stubs.
    available_arms = ["arm1"]

    # ── Discover T03 (Arm 1 naive) cached index configs for this PDF ─────────
    t03_root = (
        Path(args.t03_data_dir) if args.t03_data_dir
        else _PROJECT_ROOT / "RQ2_T03_ARM1_NAIVE" / "data"
    )
    t03_doc_dir = t03_doc_dir_for_pdf(t03_root, pdf_filename)
    t03_configs: list[T03Config] = (
        discover_t03_configs(t03_doc_dir) if t03_doc_dir else []
    )
    t03_active_config: T03Config | None = pick_default_t03_config(t03_configs)

    # ── Discover T04 (Arm 2A metadata) variants for this PDF ─────────────────
    t04_root = (
        Path(args.t04_data_dir) if args.t04_data_dir
        else _PROJECT_ROOT / "RQ2_T04_ARM2_METADATA" / "data"
    )
    t04_doc_dir = doc_dir_for_pdf(t04_root, pdf_filename)
    t04_variants: list[T04Variant] = (
        discover_t04_variants(t04_doc_dir) if t04_doc_dir else []
    )
    # Only surface the canonical Arm-2A variant (summary / node). The other
    # T04 variants are ablations we don't expose in the viewer's arm selector.
    t04_arm_to_variant: dict[str, T04Variant] = {}
    for v in t04_variants:
        if v.arm_label != "2A-summary-node":
            continue
        if v.arm_label not in available_arms:
            available_arms.append(v.arm_label)
        t04_arm_to_variant[v.arm_label] = v

    # ── Discover T05 (Arm 2B PageIndex) coverage for this PDF ───────────────
    t05_root = (
        Path(args.t05_data_dir) if args.t05_data_dir
        else _PROJECT_ROOT / "RQ2_T05_ARM2_PAGEINDEX" / "data"
    )
    t05_coverage: T05Coverage | None = discover_t05_coverage(t05_root, pdf_filename)
    if t05_coverage and t05_coverage.has_results:
        if "2B-pageindex" not in available_arms:
            available_arms.append("2B-pageindex")

    # ── Discover Arm2C (agentic ReAct tree-navigation) coverage for this PDF ──
    arm2c_root = (
        Path(args.arm2c_data_dir) if args.arm2c_data_dir
        else _PROJECT_ROOT / "RQ2_T04_ARM2_METADATA" / "data" / "_arm2c"
    )
    arm2c_coverage: Arm2cCoverage | None = discover_arm2c_coverage(arm2c_root, pdf_filename)
    if arm2c_coverage and arm2c_coverage.has_results:
        if "2C-enriched-rerank" not in available_arms:
            available_arms.append("2C-enriched-rerank")

    # ── Top bar: mode + arm ───────────────────────────────────────────────────
    mode, arm = render_top_bar(
        current_mode=_ss("mode", args.mode),
        current_arm=_ss("arm", args.arm),
        arms=available_arms,
    )
    st.session_state["mode"] = mode
    st.session_state["arm"] = arm

    st.divider()

    # ── Article spans (needed by both review and retrieval modes) ─────────────
    # Prefer spans embedded in the bundle JSON (preprocessed pipeline output);
    # fall back to a live DB lookup via locate_articles only when no bundle is
    # given.
    article_spans = _article_spans_from_bundle(bundle_json_data)
    if not article_spans and db_path:
        article_spans = _get_article_spans(raw_text, pdf_filename, db_path)

    # ── Load queries (for retrieval mode) ─────────────────────────────────────
    # Always load the full set; the sidebar exposes a toggle to scope down to
    # questions whose GT articles intersect this PDF (default on).
    queries: list[Query] = []
    bundle_bsard_ids: set[int] = set()
    bsard_to_pdf: dict[int, str] = {}
    if db_path and mode == "retrieval":
        queries = load_queries(db_path)
        # Saved t08 bundles ship with article_spans populated but articles=[]
        # (the heavyweight Article rows are dropped to keep the JSON small).
        # Use whichever source is non-empty so the "Limit to questions
        # relevant to this PDF" filter actually has something to filter on.
        if bundle and bundle.articles:
            bundle_bsard_ids = {
                int(a.bsard_id) for a in bundle.articles if getattr(a, "bsard_id", None) is not None
            }
        elif article_spans:
            bundle_bsard_ids = {
                int(s.bsard_id) for s in article_spans
                if getattr(s, "bsard_id", None) is not None
            }
        bsard_to_pdf = _get_bsard_to_pdf(db_path)

    # ── Build regions depending on mode + arm ─────────────────────────────────
    regions: list[HighlightRegion] = []

    # Track mode purely for telemetry / future use — we deliberately do NOT
    # clear session_state["regions"] / ["ranked_items"] on mode switch.
    # Doing so forced a live re-retrieval (cold-loading the embedding model +
    # FAISS) every time the user toggled back to Retrieval — multi-second
    # pause for what should be an instant view change. Instead, each mode's
    # branch below filters the cached regions to the categories it cares
    # about, so review/retrieval state can coexist in session_state.
    st.session_state["_loaded_mode"] = mode

    if mode == "review":
        if arm == "arm1":
            # Chunking is opt-in: only fires when the user clicks Re-chunk.
            # First-load auto-chunking is too costly on large legal PDFs
            # (600+ pages → 30 s+ block before the layout renders).
            if _ss("rechunk_requested"):
                chunks = _get_chunks(
                    pdf_path, raw_text,
                    _ss("strategy"), _ss("window_size"), _ss("stride"), _ss("max_tokens"),
                    doc_stem,
                )
                regions = chunks_to_regions(chunks, arm="arm1")
                st.session_state["regions"] = regions
                st.session_state["rechunk_requested"] = False
            else:
                # Only re-use cached regions if they're chunks (left over
                # from an earlier Re-chunk in this mode). Anything else is
                # stale state from a prior retrieval / arm switch.
                existing = _ss("regions") or []
                regions = [r for r in existing if r.category == "chunk"]
        else:
            if bundle and bundle.articles:
                regions = articles_to_regions(bundle.articles, bundle, arm, article_spans)
                st.session_state["regions"] = regions
            elif article_spans:
                # Saved t08 bundles drop the heavy Article rows but keep
                # article_spans, so paint the document's article segmentation
                # (the unit Arm-2 indexes) straight from the spans. Hydrate
                # text/number from the DB when one is loaded.
                span_bsard_ids = tuple(sorted({
                    int(s.bsard_id) for s in article_spans
                    if getattr(s, "bsard_id", None) is not None
                }))
                art_data = (
                    _get_gt_article_data(db_path, span_bsard_ids)
                    if db_path else {}
                )
                regions = article_spans_to_regions(article_spans, arm, art_data)
                st.session_state["regions"] = regions
                st.caption(
                    f"Showing {len(regions)} article units (the segmentation "
                    f"this arm indexes). "
                    + ("Article text from the DB." if db_path
                       else "Pass --db to show article text on click.")
                )
            else:
                st.info(
                    "Review mode here shows the document's **article segmentation** "
                    "— the unit the Arm-2 retrievers index over. It needs article "
                    "spans, which come from a saved bundle.\n\n"
                    "Launch with `--bundle-json <…/t08_bundles/…/q<id>.json>` "
                    "(recommended), or with `--db <bsard_corpus.db>` so spans can "
                    "be located live from the PDF text."
                )
                regions = []

    else:  # retrieval mode
        # ── Auto-rerun on arm change ─────────────────────────────────────────
        # When the user switches the arm dropdown with a query already active,
        # flush the previous arm's results and auto-fire the dispatch below so
        # they don't have to click Retrieve again. First entry into retrieval
        # mode (no prior arm recorded) is a no-op so the bundle pre-load can
        # still populate the initial state.
        last_loaded_arm = st.session_state.get("_loaded_arm")
        if last_loaded_arm is not None and last_loaded_arm != arm:
            st.session_state["regions"] = []
            st.session_state["gt_regions"] = []
            st.session_state["ranked_items"] = []
            st.session_state["trace"] = None
            if _ss("active_query_id") or _ss("active_query_text"):
                st.session_state["retrieve_requested"] = True
        st.session_state["_loaded_arm"] = arm

        # Pre-populate regions from bundle JSON on first load (demo path)
        if (
            not _ss("regions")
            and bundle_json_data
            and bundle_json_data.get("retrieval_result")
            and not st.session_state.get("_retrieval_loaded")
        ):
            from data_loader.models import RetrievalResult as _RR
            pre_result = bundle_json_data["retrieval_result"]
            pre_obj = _RR(
                query_id=pre_result.get("query_id", ""),
                query_text=pre_result.get("query_text", ""),
                ranked_items=pre_result.get("ranked_items", []),
                method=pre_result.get("method", "arm1_bm25"),
                cost=pre_result.get("cost", {}),
            )
            gt_bs_ids = _ss("relevant_article_ids") or []
            _dummy = bundle if bundle is not None else _dummy_bundle(pdf_path, pdf_filename, raw_text)
            ret_regions = retrieval_result_to_regions(pre_obj, _dummy, article_spans, gt_bsard_ids=gt_bs_ids)
            gt_regs = gt_regions_from_bsard_ids(gt_bs_ids, _dummy, article_spans)
            regions = ret_regions + gt_regs
            st.session_state["regions"] = regions
            st.session_state["gt_regions"] = gt_regs
            st.session_state["ranked_items"] = pre_result.get("ranked_items", [])
            st.session_state["_retrieval_loaded"] = True

        # Use pre-loaded results if they exist in session state (from retrieve
        # button or bundle pre-load). Drop any chunk/article overlays left
        # over from a prior Review session so they don't paint on the
        # retrieval view.
        regions = [
            r for r in (_ss("regions") or [])
            if r.category not in ("chunk", "article")
        ]
        if _ss("retrieve_requested"):
            q_text = _ss("active_query_text", "")
            if q_text:
                gt_bs_ids = _ss("relevant_article_ids") or []
                result = _run_retrieval(
                    arm, q_text, raw_text, pdf_path, doc_stem, db_path,
                    bundle, article_spans, args,
                    top_k=_ss("top_k", 10),
                    query_id=_ss("active_query_id", ""),
                    t03_active_config=t03_active_config,
                    t04_arm_to_variant=t04_arm_to_variant,
                    t05_coverage=t05_coverage,
                    arm2c_coverage=arm2c_coverage,
                )
                if result is not None:
                    ret_regions = retrieval_result_to_regions(result, bundle or _dummy_bundle(pdf_path, pdf_filename, raw_text), article_spans, gt_bsard_ids=gt_bs_ids)
                    gt_regs = gt_regions_from_bsard_ids(gt_bs_ids, bundle or _dummy_bundle(pdf_path, pdf_filename, raw_text), article_spans)
                    regions = ret_regions + gt_regs
                    st.session_state["regions"] = regions
                    st.session_state["gt_regions"] = gt_regs
                    st.session_state["trace"] = result.trace
                    st.session_state["ranked_items"] = result.ranked_items
            st.session_state["retrieve_requested"] = False

    # ── Build per-region first-page lookup for sidebar hints ─────────────────
    from visualization.coord_mapper import get_first_page
    page_info = {r.region_id: get_first_page(r, word_index) or 0 for r in regions if r.category in ("chunk", "article")}

    # ── Compute the union of query_ids that have cached results for this PDF.
    # These are the queries the dropdown should default to (instead of the
    # full 250-question BSARD set) since they're the only ones with usable
    # cached data on T04 / T05. arm1 is excluded — its index supports any
    # query, so the "show all queries" toggle in the sidebar covers that.
    cached_query_ids: set[str] = _collect_cached_query_ids(
        tuple(str(v.results_path) for v in t04_variants),
        tuple(t05_coverage.query_files) if t05_coverage else (),
        tuple(arm2c_coverage.query_files) if arm2c_coverage else (),
    )

    # ── Sidebar content function (injected into layout) ───────────────────────
    def sidebar_fn():
        nonlocal regions
        if mode == "review":
            # The chunk-control widgets all bind to session_state via key=,
            # so writing those keys back here would raise StreamlitAPIException
            # ("cannot be modified after the widget … is instantiated"). The
            # returned values are kept only for type compatibility.
            new_sel, _strat, _ws, _stride, _mt = render_review_sidebar(
                arm=arm,
                regions=regions,
                selected_id=_ss("selected_id"),
                strategy=_ss("strategy"),
                window_size=_ss("window_size"),
                stride=_ss("stride"),
                max_tokens=_ss("max_tokens"),
                page_info=page_info,
            )
            if new_sel:
                st.session_state["selected_id"] = new_sel
        else:
            gt_art_ids = _ss("relevant_article_ids") or []
            # render_retrieval_sidebar owns the dropdown filtering + writes
            # active_query_id / relevant_article_ids into session state when
            # the user picks a preset.
            q_text, new_sel, top_k = render_retrieval_sidebar(
                queries=queries,
                regions=regions,
                gt_article_ids=gt_art_ids,
                selected_id=_ss("selected_id"),
                article_data=gt_article_data,
                cached_query_ids=cached_query_ids,
                bundle_bsard_ids=bundle_bsard_ids,
                bsard_to_pdf=bsard_to_pdf,
            )
            st.session_state["top_k"] = top_k
            if new_sel:
                st.session_state["selected_id"] = new_sel

            # ── ToC tree explorer for T05 (PageIndex / 2B) ──────────────
            # Loaded once per doc and cached; render only when the active
            # arm is the PageIndex one and a tree is available.
            if (
                arm == "2B-pageindex"
                and t05_coverage is not None
                and t05_coverage.tree_path is not None
            ):
                tree_data = _read_t05_tree_cached(str(t05_coverage.tree_path))
                trace_summary = (_ss("trace") or {}).get("summary") or {}
                render_tree_explorer(tree_data, trace_summary, expanded=False)

            # ── Deep-tree explorer for Arm2C (recursive, depth-7) ───────
            if (
                arm.startswith("2C")
                and arm2c_coverage is not None
                and arm2c_coverage.tree_path is not None
            ):
                deep_tree = _read_arm2c_tree_cached(str(arm2c_coverage.tree_path))
                trace_summary = (_ss("trace") or {}).get("summary") or {}
                render_deep_tree_explorer(deep_tree, trace_summary, expanded=False)

    total_pages = max((ws.page_num for ws in word_index), default=0) + 1 if word_index else 1

    gt_bs_ids_for_lookup = _ss("relevant_article_ids") or []
    # Include bsard_ids from all ranked chunks so the sidebar can show them too
    all_bsard_ids: set = {int(b) for b in gt_bs_ids_for_lookup}
    for _item in (_ss("ranked_items") or []):
        for b in (_item.get("bsard_ids") or _item.get("metadata", {}).get("bsard_ids") or []):
            all_bsard_ids.add(int(b))
    gt_article_data = (
        _get_gt_article_data(db_path or "", tuple(sorted(all_bsard_ids)))
        if db_path else {}
    )

    render_main_layout(
        pdf_bytes=pdf_bytes,
        sidebar_content_fn=sidebar_fn,
        regions=regions,
        word_index=word_index,
        selected_id=_ss("selected_id"),
        mode=mode,
        arm=arm,
        query_text=_ss("active_query_text"),
        relevant_article_ids=_ss("relevant_article_ids"),
        gt_regions=_ss("gt_regions"),
        trace=_ss("trace"),
        ranked_items=_ss("ranked_items"),
        total_pages=total_pages,
        gt_article_data=gt_article_data,
        raw_text=raw_text,
        query_id=_ss("active_query_id"),
    )


# ── Retrieval dispatcher ──────────────────────────────────────────────────────

def _dummy_bundle(pdf_path: str, pdf_filename: str, raw_text: str):
    """Create a minimal DocumentBundle when the full one is not available."""
    try:
        import pandas as pd
        from data_loader.models import DocumentBundle
        return DocumentBundle(
            document_id=0,
            pdf_path=pdf_path,
            pdf_filename=pdf_filename,
            metadata={},
            definitions=pd.DataFrame(),
            articles=[],
            raw_text=raw_text,
            has_azure_extraction=False,
        )
    except ImportError:
        return None


def _run_retrieval(
    arm, query_text, raw_text, pdf_path, doc_stem, db_path,
    bundle, article_spans, args,
    top_k: int = 10,
    query_id: str = "",
    t03_active_config: T03Config | None = None,
    t04_arm_to_variant: dict | None = None,
    t05_coverage: T05Coverage | None = None,
    arm2c_coverage: Arm2cCoverage | None = None,
):
    """Run retrieval for the given arm and query. Returns a RetrievalResult or None."""
    index_dir = Path(args.index_dir) if args.index_dir else Path.cwd() / "data" / "indices"

    if arm == "arm1":
        return _run_arm1_retrieval(
            query_text, doc_stem, index_dir,
            top_k=top_k,
            t03_config=t03_active_config,
        )

    if t04_arm_to_variant and arm in t04_arm_to_variant:
        return _run_t04_retrieval(arm, query_id, t04_arm_to_variant[arm], top_k=top_k)

    if arm == "2B-pageindex" and t05_coverage is not None:
        return _run_t05_retrieval(query_id, t05_coverage, top_k=top_k)

    if arm.startswith("2C") and arm2c_coverage is not None:
        return _run_arm2c_retrieval(query_id, arm2c_coverage, top_k=top_k)

    st.warning(
        f"Live retrieval for arm {arm!r} not yet wired in the viewer. "
        "Load pre-computed RetrievalResult objects via --bundle-json, or run "
        "the arm's pipeline to populate its results cache."
    )
    return None


@st.cache_data(show_spinner="Reading T05 result …")
def _read_t05_result_cached(path_str: str) -> dict | None:
    return load_t05_result(path_str)


@st.cache_data(show_spinner=False)
def _collect_cached_query_ids(
    t04_result_paths: tuple[str, ...],
    t05_query_ids: tuple[str, ...],
    arm2c_query_ids: tuple[str, ...] = (),
) -> set[str]:
    """Return the union of query_ids that have cached results across T04 + T05 + Arm2C.

    Cached on the tuple of paths/ids so the scan only runs once per (PDF,
    discovered-cache) combination.
    """
    qids: set[str] = set(t05_query_ids) | set(arm2c_query_ids)
    for path in t04_result_paths:
        # We only need the query_ids here — skip BM25 text hydration so this
        # scan doesn't load (and log) every variant's BM25 store. Text is
        # hydrated lazily later, only for the one variant actually retrieved.
        results = load_t04_results(path, hydrate_text=False)
        qids.update(str(qid) for qid in results)
    return qids


@st.cache_data(show_spinner="Loading T05 ToC tree …")
def _read_t05_tree_cached(path_str: str) -> dict | None:
    return load_t05_tree(path_str)


def _run_t05_retrieval(query_id: str, coverage: T05Coverage, top_k: int = 10):
    """Look up a cached T05 PageIndex result by query_id."""
    if not query_id:
        st.warning(
            "T05 PageIndex requires a preset query (results are keyed by query_id). "
            "Pick a query from the sidebar dropdown."
        )
        return None
    path = coverage.query_files.get(str(query_id))
    if path is None:
        available = sorted(coverage.query_files)
        st.warning(
            f"No cached T05 result for query {query_id!r}. "
            f"Available: {available or '(none)'}."
        )
        return None
    d = _read_t05_result_cached(str(path))
    if d is None:
        st.error(f"Could not read T05 result file: {path}")
        return None
    result = t05_dict_to_retrieval_result(d)
    result.ranked_items = list(result.ranked_items)[:top_k]
    return result


@st.cache_data(show_spinner="Reading Arm2C result …")
def _read_arm2c_result_cached(path_str: str) -> dict | None:
    return load_arm2c_result(path_str)


@st.cache_data(show_spinner="Loading Arm2C deep tree …")
def _read_arm2c_tree_cached(path_str: str) -> dict | None:
    return load_deep_tree(path_str)


def _run_arm2c_retrieval(query_id: str, coverage: Arm2cCoverage, top_k: int = 10):
    """Look up an exported Arm2C result by query_id."""
    if not query_id:
        st.warning(
            "Arm2C requires a preset query (results are keyed by query_id). "
            "Pick a query from the sidebar dropdown."
        )
        return None
    path = coverage.query_files.get(str(query_id))
    if path is None:
        available = sorted(coverage.query_files)
        st.warning(
            f"No cached Arm2C result for query {query_id!r}. "
            f"Available: {available or '(none)'}."
        )
        return None
    d = _read_arm2c_result_cached(str(path))
    if d is None:
        st.error(f"Could not read Arm2C result file: {path}")
        return None
    result = arm2c_dict_to_retrieval_result(d)
    result.ranked_items = list(result.ranked_items)[:top_k]
    return result


@st.cache_data(show_spinner="Reading T04 results …")
def _load_t04_jsonl(jsonl_path: str) -> dict:
    return load_t04_results(jsonl_path)


def _run_t04_retrieval(arm: str, query_id: str, variant: T04Variant, top_k: int = 10):
    """Look up a cached T04 (Arm 2A metadata) result by query_id."""
    if not query_id:
        st.warning(
            f"T04 arm {arm!r} requires a preset query (results are keyed by query_id). "
            "Pick a query from the sidebar dropdown."
        )
        return None

    results = _load_t04_jsonl(str(variant.results_path))
    d = results.get(str(query_id))
    if not d:
        st.warning(
            f"No cached T04 result for query {query_id!r} in variant {arm!r} "
            f"(file: {variant.results_path.name})."
        )
        return None

    result = t04_dict_to_retrieval_result(d)
    result.ranked_items = list(result.ranked_items)[:top_k]
    return result


def _run_arm1_retrieval(
    query_text: str,
    doc_stem: str,
    index_dir: Path,
    top_k: int = 10,
    t03_config: T03Config | None = None,
):
    """Retrieve using the cached Arm 1 FAISS + BM25 index.

    Prefers ``t03_config.source_dir`` (T03's cache-aware layout under
    ``RQ2_T03_ARM1_NAIVE/data/<doc_stem>/configs/<hash>/``) when available,
    and falls back to the legacy ``<index_dir>/arm1/<doc_stem>/`` layout.
    """
    try:
        from arm1_naive.indexer import load_arm1_index
        from arm1_naive.retriever import retrieve_arm1
        from shared.embeddings import EmbeddingModel
    except ImportError as exc:
        st.error(f"arm1_naive / shared packages not installed: {exc}")
        return None

    if t03_config is not None:
        source_dir = t03_config.source_dir
        with st.spinner(
            f"Loading T03 index ({t03_config.embedding_model}, "
            f"hash={t03_config.config_hash}) …"
        ):
            faiss_store, bm25_store = load_arm1_index(
                doc_stem, source_dir.parent, source_dir=source_dir,
            )
    else:
        from visualization.index_cache import index_exists
        arm1_index_dir = index_dir / "arm1"
        if not index_exists("arm1", doc_stem, index_dir):
            st.error(
                f"No Arm 1 index found for {doc_stem!r}. Looked in:\n"
                f"  • T03 cache: {(_PROJECT_ROOT / 'RQ2_T03_ARM1_NAIVE' / 'data' / doc_stem / 'configs')}\n"
                f"  • Legacy:    {arm1_index_dir / doc_stem}\n"
                "Run the arm1 pipeline first or pass --t03-data-dir."
            )
            return None
        with st.spinner("Loading index …"):
            faiss_store, bm25_store = load_arm1_index(doc_stem, arm1_index_dir)

    embed_model_name = (
        t03_config.embedding_model if t03_config is not None
        else "intfloat/multilingual-e5-large-instruct"
    )
    with st.spinner("Loading models …"):
        embed_model = _get_embed_model(embed_model_name)

    if embed_model is None:
        return None

    with st.spinner("Retrieving …"):
        ranked = retrieve_arm1(query_text, faiss_store, bm25_store, embed_model)
        ranked = ranked[:top_k]

    from data_loader.models import RetrievalResult
    ranked_items = [
        {
            "id": r.id,
            "score": r.score,
            "text": r.metadata.get("text", ""),
            "bsard_ids": r.metadata.get("bsard_ids", []),
            "start_char": r.metadata.get("start_char"),
            "end_char": r.metadata.get("end_char"),
            "start_token": r.metadata.get("start_token"),
            "end_token": r.metadata.get("end_token"),
            "metadata": r.metadata,
        }
        for r in ranked
    ]
    return RetrievalResult(
        query_id="viewer",
        query_text=query_text,
        ranked_items=ranked_items,
        method="arm1",
        cost={},
    )


@st.cache_resource(show_spinner="Loading embedding model …")
def _get_embed_model(model_name: str):
    try:
        from shared.embeddings import EmbeddingModel
        return EmbeddingModel(model_name=model_name)
    except Exception as exc:
        st.error(f"Could not load embedding model {model_name!r}: {exc}")
        return None


if __name__ == "__main__":
    main()
