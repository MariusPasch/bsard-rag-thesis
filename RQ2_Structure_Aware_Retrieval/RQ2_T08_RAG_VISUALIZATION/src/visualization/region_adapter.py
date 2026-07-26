"""Uniform HighlightRegion type and adapters for all arm outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Optional

if TYPE_CHECKING:
    from data_loader.models import Article, DocumentBundle, RetrievalResult
    from arm1_naive.chunker import Chunk as Arm1Chunk

HighlightCategory = Literal[
    "chunk",           # Arm 1 chunk in review mode
    "article",         # Arm 2 article in review mode
    "gt",              # Ground-truth article span in retrieval mode
    "retrieved",       # Correctly retrieved item (article_ids overlap GT)
    "wrong_retrieved", # Retrieved item whose article_ids do NOT overlap GT
    "retrieved_node_page",        # T04 node-level hit: page-band marker (correct)
    "wrong_retrieved_node_page",  # T04 node-level hit: page-band marker (incorrect)
    "overlap",         # Geometric intersection of gt ∩ retrieved
    "selected",        # User-selected item in review mode
    "search_hit",      # Ctrl-F-style text search match (overlaid on top)
    "search_active",   # The currently-focused search match (Prev/Next)
]


@dataclass
class HighlightRegion:
    """Uniform representation across all arms for rendering on the PDF."""

    region_id: str
    text: str
    start_char: int
    end_char: int
    category: HighlightCategory
    arm: str
    score: Optional[float] = None
    rank: Optional[int] = None
    start_token: Optional[int] = None
    end_token: Optional[int] = None
    metadata: dict = field(default_factory=dict)


# ── Arm 1: Chunk → HighlightRegion ───────────────────────────────────────────


def chunks_to_regions(chunks: list, arm: str = "arm1") -> list[HighlightRegion]:
    """Map Arm 1 Chunk objects to HighlightRegion list (trivial — Chunks already carry char offsets)."""
    regions = []
    for chunk in chunks:
        regions.append(
            HighlightRegion(
                region_id=chunk.chunk_id,
                text=chunk.text,
                start_char=chunk.start_char,
                end_char=chunk.end_char,
                category="chunk",
                arm=arm,
                start_token=chunk.start_token,
                end_token=chunk.end_token,
                metadata={"bsard_ids": getattr(chunk, "bsard_ids", [])},
            )
        )
    return regions


# ── Arm 2: Article → HighlightRegion ─────────────────────────────────────────


def articles_to_regions(
    articles: list,
    bundle: "DocumentBundle",
    arm: str,
    article_spans: list | None = None,
) -> list[HighlightRegion]:
    """Map Arm 2 Article objects to HighlightRegion list.

    article_spans: pre-computed ArticleSpan list from arm1_naive.chunker.locate_articles().
    If None, articles without resolvable positions are emitted with start_char=end_char=0
    and flagged in metadata.
    """
    span_by_id: dict[int, tuple[int, int]] = {}
    if article_spans:
        for span in article_spans:
            span_by_id[span.bsard_id] = (span.start_char, span.end_char)

    regions = []
    for article in articles:
        bs_id = getattr(article, "bsard_id", None)
        span = span_by_id.get(bs_id) if bs_id is not None else None

        if span:
            start_char, end_char = span
            unresolved = False
        else:
            start_char, end_char = 0, 0
            unresolved = True

        label = bs_id if bs_id is not None else article.article_id

        regions.append(
            HighlightRegion(
                region_id=f"art_{label}",
                text=article.text,
                start_char=start_char,
                end_char=end_char,
                category="article",
                arm=arm,
                metadata={
                    "bsard_id": bs_id,
                    "chapter": getattr(article, "chapter", ""),
                    "section": getattr(article, "section", ""),
                    "article_number": getattr(article, "article_number", ""),
                    "unresolved": unresolved,
                },
            )
        )
    return regions


def article_spans_to_regions(
    article_spans: list,
    arm: str,
    article_data: dict | None = None,
) -> list[HighlightRegion]:
    """Build article-unit review regions straight from ``ArticleSpan`` objects.

    Used for the Arm-2 arms (2A/2B/2C) in review mode: saved t08 bundles drop
    the heavyweight ``Article`` rows but keep ``article_spans`` (bsard_id →
    char range), so we paint the document's article segmentation — the unit
    these arms index over — directly from the spans. ``article_data`` (the
    ``{bsard_id: {text, article_number, law_title}}`` map from the DB) hydrates
    each region's text/number when available; without it the spans still paint,
    just without per-article text in the Selected panel.
    """
    article_data = article_data or {}
    regions: list[HighlightRegion] = []
    used_ids: set[str] = set()
    for span in article_spans:
        bs = getattr(span, "bsard_id", None)
        if bs is None:
            continue
        bs_int = int(bs)
        start, end = int(span.start_char), int(span.end_char)
        # A bsard_id can map to several distinct PDF ranges (article-span
        # fan-out: the same article's text appears in more than one place).
        # Keep each physical range as its own selectable region, but make the
        # id unique — clean ``art_<bs>`` for the common single-span case,
        # suffixed with the offset for extras. Skip exact-duplicate ranges.
        region_id = f"art_{bs_int}"
        if region_id in used_ids:
            region_id = f"art_{bs_int}_{start}"
            if region_id in used_ids:
                continue
        used_ids.add(region_id)

        entry = article_data.get(bs_int) or article_data.get(str(bs_int)) or {}
        regions.append(
            HighlightRegion(
                region_id=region_id,
                text=entry.get("text", ""),
                start_char=start,
                end_char=end,
                category="article",
                arm=arm,
                metadata={
                    "bsard_id": bs_int,
                    "article_number": entry.get("article_number", ""),
                    "law_title": entry.get("law_title", ""),
                },
            )
        )
    return regions


# ── Any arm: RetrievalResult → HighlightRegion ───────────────────────────────


def retrieval_result_to_regions(
    result: "RetrievalResult",
    bundle: "DocumentBundle",
    article_spans: list | None = None,
    chunk_map: dict | None = None,
    gt_bsard_ids: list[int] | None = None,
) -> list[HighlightRegion]:
    """Convert ranked_items in a RetrievalResult to HighlightRegion list.

    Each ranked item carries start_char/end_char and a bsard_ids list (the
    canonical BSARD article ids overlapping the chunk). Items are classified
    as 'retrieved' (correct) or 'wrong_retrieved' based on bsard_ids ∩ GT.
    """
    span_by_bsard_id: dict = {}
    if article_spans:
        for span in article_spans:
            span_by_bsard_id[span.bsard_id] = (span.start_char, span.end_char)

    gt_id_set: set = {int(b) for b in gt_bsard_ids} if gt_bsard_ids else set()

    regions = []
    for rank, item in enumerate(result.ranked_items, start=1):
        item_id = item.get("id", f"item_{rank}")
        text = item.get("text", "")
        meta = item.get("metadata", {}) if isinstance(item.get("metadata"), dict) else {}
        score = item.get("score")

        # Resolve char offsets — arm1 stores them directly in the item dict
        start_char = item.get("start_char") or meta.get("start_char")
        end_char = item.get("end_char") or meta.get("end_char")

        if start_char is None or end_char is None:
            # Try resolving via bsard_id (single-article items)
            bs_id = item.get("bsard_id") or meta.get("bsard_id")
            if bs_id is not None:
                span = span_by_bsard_id.get(int(bs_id))
                if span:
                    start_char, end_char = span

        if start_char is None:
            start_char, end_char = 0, 0

        start_token = item.get("start_token") or meta.get("start_token")
        end_token = item.get("end_token") or meta.get("end_token")

        hop_source = meta.get("source", "")  # "seed" / "hop_1" / "hop_2" for 2C

        # Classify as correctly or wrongly retrieved based on GT bsard_id overlap
        item_bsard_ids = {
            int(b) for b in (item.get("bsard_ids") or meta.get("bsard_ids") or [])
        }
        if gt_id_set and not (item_bsard_ids & gt_id_set):
            category: HighlightCategory = "wrong_retrieved"
        else:
            category = "retrieved"

        regions.append(
            HighlightRegion(
                region_id=item_id,
                text=text,
                start_char=int(start_char),
                end_char=int(end_char),
                category=category,
                arm=result.method,
                score=float(score) if score is not None else None,
                rank=rank,
                start_token=int(start_token) if start_token is not None else None,
                end_token=int(end_token) if end_token is not None else None,
                metadata={
                    **meta,
                    "source": hop_source,
                    "bsard_ids": list(item.get("bsard_ids") or meta.get("bsard_ids") or []),
                },
            )
        )

        # T04 node-level items carry both ``node_id`` and ``page_number``. The
        # bsard fallback above resolves their span to the parent article (no
        # finer-grained char offsets exist in the cached results), which paints
        # the whole article on every page it spans. Emit a sibling page-band
        # region so the user can also see the *specific* page the node lives
        # on. Same region_id so a sidebar click selects both the article fill
        # and the band together.
        node_id_meta = meta.get("node_id")
        page_number_meta = meta.get("page_number")
        if node_id_meta is not None and page_number_meta is not None:
            band_category: HighlightCategory = (
                "wrong_retrieved_node_page" if category == "wrong_retrieved"
                else "retrieved_node_page"
            )
            regions.append(
                HighlightRegion(
                    region_id=item_id,
                    text="",
                    start_char=0,
                    end_char=0,
                    category=band_category,
                    arm=result.method,
                    score=float(score) if score is not None else None,
                    rank=rank,
                    metadata={
                        "is_page_band": True,
                        "page_number": int(page_number_meta),
                        "node_id": node_id_meta,
                        "bsard_ids": list(
                            item.get("bsard_ids") or meta.get("bsard_ids") or []
                        ),
                    },
                )
            )
    return regions


def gt_regions_from_bsard_ids(
    bsard_ids: list,
    bundle: "DocumentBundle",
    article_spans: list | None = None,
) -> list[HighlightRegion]:
    """Build ground-truth HighlightRegion list from a list of BSARD article ids."""
    span_by_id: dict = {}
    if article_spans:
        for span in article_spans:
            span_by_id[int(span.bsard_id)] = (span.start_char, span.end_char)

    # Build article lookup from bundle (bundle Articles carry both ids)
    art_by_bsard_id: dict = {}
    for art in bundle.articles:
        bs = getattr(art, "bsard_id", None)
        if bs is not None:
            art_by_bsard_id[int(bs)] = art

    regions = []
    for bs_id in bsard_ids:
        bs_int = int(bs_id)
        span = span_by_id.get(bs_int)
        art = art_by_bsard_id.get(bs_int)
        text = art.text if art else ""

        start_char, end_char = span if span else (0, 0)

        regions.append(
            HighlightRegion(
                region_id=f"gt_{bs_int}",
                text=text,
                start_char=start_char,
                end_char=end_char,
                category="gt",
                arm="gt",
                metadata={"bsard_id": bs_int},
            )
        )
    return regions


# Legacy alias kept so any external import continues to work; new code should
# call gt_regions_from_bsard_ids directly.
gt_regions_from_article_ids = gt_regions_from_bsard_ids


# ── Ctrl-F-style PDF text search ─────────────────────────────────────────────


def text_matches_to_regions(
    raw_text: str,
    query: str,
    *,
    max_hits: int = 500,
    active_idx: int | None = None,
) -> list[HighlightRegion]:
    """Find ``query`` inside ``raw_text`` and return one HighlightRegion per match.

    Search is case-insensitive and tolerates whitespace runs (so a phrase that
    spans a soft line break or page separator still matches). The match at
    ``active_idx`` (if provided) is emitted with category ``"search_active"``
    so the viewer can paint it differently from the rest.
    """
    import re

    query = (query or "").strip()
    if not query or not raw_text:
        return []

    # Tokenise on runs of whitespace, then join with \s+ so the pattern matches
    # the same phrase whether the underlying text inserts a newline, a "\n\n"
    # page separator, or extra spaces inside the phrase.
    tokens = [re.escape(tok) for tok in query.split() if tok]
    if not tokens:
        return []
    pattern = re.compile(r"\s+".join(tokens), re.IGNORECASE)

    regions: list[HighlightRegion] = []
    for i, m in enumerate(pattern.finditer(raw_text)):
        if i >= max_hits:
            break
        cat: HighlightCategory = "search_active" if i == active_idx else "search_hit"
        regions.append(
            HighlightRegion(
                region_id=f"search_{i}",
                text=m.group(0),
                start_char=m.start(),
                end_char=m.end(),
                category=cat,
                arm="search",
                metadata={"match_idx": i, "query": query},
            )
        )
    return regions
