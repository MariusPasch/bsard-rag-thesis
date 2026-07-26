"""Chunk-bsard overlap weight precomputation for weighted partial-relevance metrics.

Reference: Kekäläinen (2005), "Using graded relevance assessments in IR evaluation."

w(c, B) = |chars of B covered by c| / |chars of B|

The cached path (``precompute_weights_cached``) computes this from T03's
*authoritative* article spans (``article_spans.json``) and the matching chunk
char spans, so weights align exactly with what T03 indexed — no re-location, no
tokenizer. (The legacy in-memory ``precompute_weights`` still uses the
token-anchor kernel for callers that pass raw Article objects without spans.)

Historically the cached path re-located each article in ``raw_text`` via an
``article.text[:100]`` anchor search; that matched <11% of articles when the
BSARD canonical text diverged from the PDF extraction (CN-T07-010). Reusing
T03's spans fixes coverage.

Keys are (chunk_id, bsard_id) — articles without a bsard_id (non-BSARD) are
skipped, since the GT is bsard-id-keyed and they cannot match.

Cached entry point
------------------
``precompute_weights_cached(...)`` (added in Phase 2) reads chunks + raw_text
from T03's PDF cache, fetches BSARD-eligible articles for the PDF from the
corpus DB, and persists weights + manifest under
``<cache>/<doc_id>/eval/<arm1_config_hash>/`` (PLAN_per_pdf_pipeline §B.8 —
T07-owned subtree, sibling to T03's ``configs/``).
The legacy ``precompute_weights(...)`` signature is preserved for direct
in-memory callers.
"""

from __future__ import annotations

import csv
import logging
import pickle
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    # Type-only imports to keep T07's import graph lean for non-cache call paths.
    from arm1_naive.cache import ConfigCache, PdfCache


@dataclass
class ChunkArticleWeight:
    """Despite the legacy name, the article identifier is a bsard_id."""
    chunk_id: str
    bsard_id: int
    weight: float


@dataclass
class _ArticleRow:
    """Minimal article shape consumed by ``_compute_weights_and_ranges``."""
    bsard_id: int
    text: str


# ── Public, in-memory entry point (legacy signature preserved) ───────────────


def precompute_weights(
    chunks: list,       # list[Chunk] from data_loader.models
    articles: list,     # list[Article] from data_loader.models
    tokenizer_name: str,
    raw_text: str | None = None,
) -> list[ChunkArticleWeight]:
    """Compute overlap weights between chunks and BSARD articles.

    Articles without a bsard_id (``a.bsard_id is None``) are skipped: they
    cannot appear in bsard-id-keyed ground truth.
    """
    bsard_articles = [a for a in articles if a.bsard_id is not None]
    skipped = len(articles) - len(bsard_articles)
    if skipped:
        logger.info("Skipping %d non-BSARD articles in weight precomputation", skipped)

    weights, _ranges = _compute_weights_and_ranges(
        chunks, bsard_articles, tokenizer_name, raw_text
    )
    _validate_weight_coverage(weights, bsard_articles)
    return weights


# ── Cached entry point ───────────────────────────────────────────────────────


def precompute_weights_cached(
    *,
    doc_id: str,
    arm1_config_hash: str,
    t07_cache_root,                  # T07CacheRoot
    t03_cache_root,                  # arm1_naive.cache.CacheRoot
    db_conn: sqlite3.Connection,
    tokenizer_name: str,
    pdf_filename: str,
    force: bool = False,
) -> list[ChunkArticleWeight]:
    """Return chunk-bsard weights for one PDF, hitting the cache when valid.

    Cache layout (CACHE_DESIGN.md §4): per ``arm1_config_hash`` the weights
    live alongside the chunks/indices that produced them, so a chunking-param
    change moves to a new sibling folder rather than poisoning this one.

    On miss / staleness:
      1. Load chunks + raw_text from T03's PdfCache + ConfigCache.
      2. Fetch BSARD-eligible articles for the PDF from the DB (in document
         order so ``_compute_bsard_token_ranges`` advances forward through
         ``raw_text`` correctly).
      3. Run ``_compute_weights_and_ranges`` → (weights, ranges).
      4. Persist CSV + PKL + ranges JSON + manifest with coverage stats.

    Returns the freshly built or freshly loaded weight rows.
    """
    # Local imports — keep arm1_naive optional for purely in-memory callers.
    from arm1_naive.cache import ConfigCache, PdfCache

    from .cache import ChunkWeightsCache, compute_bsard_db_fingerprint

    pdf_cache = PdfCache(t03_cache_root, doc_id)
    cfg_cache = ConfigCache(pdf_cache, arm1_config_hash)

    if not cfg_cache.exists():
        raise FileNotFoundError(
            f"T03 manifest missing at {cfg_cache.manifest_path()} — "
            "populate the T03 cache first (scripts/precompute_pdf.py)."
        )
    t03_manifest_resolved = str(cfg_cache.manifest_path().resolve(strict=True))

    db_fingerprint = compute_bsard_db_fingerprint(db_conn)
    weights_cache = ChunkWeightsCache(t07_cache_root, doc_id, arm1_config_hash)

    if not force and weights_cache.exists() and weights_cache.validate(
        tokenizer_name=tokenizer_name,
        bsard_db_fingerprint=db_fingerprint,
        t03_manifest_resolved_path=t03_manifest_resolved,
    ):
        logger.info(
            "[cache hit] doc=%s arm1_config=%s — loading weights",
            doc_id, arm1_config_hash,
        )
        return weights_cache.load_weights()

    logger.info(
        "[cache miss] doc=%s arm1_config=%s — recomputing weights",
        doc_id, arm1_config_hash,
    )

    chunks = cfg_cache.load_chunks()

    # Reuse T03's authoritative article spans (char ranges) instead of
    # re-locating articles from DB text via fragile anchor matching. T03 already
    # located + drift-filtered every in-PDF article when it chunked, and the
    # chunks carry the matching char spans, so char-overlap weights align
    # exactly with what was indexed (CN-T07-010 fix).
    article_spans = pdf_cache.load_article_spans()
    if not article_spans:
        logger.warning(
            "No article spans cached for doc_id=%s — weights will be empty. "
            "Run T03's scripts/precompute_pdf.py first.", doc_id,
        )

    weights, ranges = _compute_weights_from_spans(chunks, article_spans)
    coverage_stats = compute_weight_coverage(weights, article_spans)
    _log_coverage_warnings(weights, article_spans)

    weights_cache.save(
        weights=weights,
        ranges=ranges,
        tokenizer_name=tokenizer_name,
        bsard_db_fingerprint=db_fingerprint,
        t03_manifest_resolved_path=t03_manifest_resolved,
        n_chunks=len(chunks),
        n_bsard_in_pdf=len(article_spans),
        coverage_stats=coverage_stats,
    )
    return weights


def _fetch_bsard_articles_for_pdf(
    db_conn: sqlite3.Connection,
    pdf_filename: str,
) -> list[_ArticleRow]:
    """Fetch (bsard_id, article_text) for a PDF, ordered for token-range computation.

    Returns articles in document order (``article_id`` ASC) so the prefix-token
    scan in ``_compute_bsard_token_ranges`` advances monotonically through
    ``raw_text``. Mirrors the filename fallback in
    ``arm1_naive.chunker.locate_articles`` so the same DB schema variants work.
    """
    rows = db_conn.execute(
        "SELECT bsard_id, article_text FROM articles "
        "WHERE pdf_filename = ? AND bsard_id IS NOT NULL "
        "ORDER BY article_id",
        (pdf_filename,),
    ).fetchall()

    if not rows:
        stem = Path(pdf_filename).stem
        db_filename = f"img_l_pdf_{stem}_F.pdf"
        rows = db_conn.execute(
            "SELECT bsard_id, article_text FROM articles "
            "WHERE pdf_filename = ? AND bsard_id IS NOT NULL "
            "ORDER BY article_id",
            (db_filename,),
        ).fetchall()

    return [
        _ArticleRow(bsard_id=int(r[0]), text=str(r[1] or ""))
        for r in rows
    ]


# ── Coverage stats (used by both legacy logger and cached manifest) ──────────


def compute_weight_coverage(
    weights: list[ChunkArticleWeight],
    articles: list,
    tolerance: float = 0.05,
) -> dict[str, int]:
    """Bucket articles by per-bsard weight-sum quality.

    Returns ``{ok, warning, missing}`` counts:
      ok       — sum within ``1.0 ± tolerance`` (the chunker fully covered the article)
      warning  — sum present but outside tolerance (truncation, drift, near-misses)
      missing  — no overlap rows at all (article not located, or chunks skipped it)
    """
    coverage: dict[int, float] = defaultdict(float)
    for w in weights:
        coverage[w.bsard_id] += w.weight

    counts = {"ok": 0, "warning": 0, "missing": 0}
    for article in articles:
        if article.bsard_id is None:
            continue
        total = coverage.get(int(article.bsard_id), 0.0)
        if total == 0.0:
            counts["missing"] += 1
        elif abs(total - 1.0) > tolerance:
            counts["warning"] += 1
        else:
            counts["ok"] += 1
    return counts


def _log_coverage_warnings(
    weights: list[ChunkArticleWeight],
    articles: list,
    tolerance: float = 0.05,
) -> None:
    coverage: dict[int, float] = defaultdict(float)
    for w in weights:
        coverage[w.bsard_id] += w.weight
    for article in articles:
        if article.bsard_id is None:
            continue
        total = coverage.get(int(article.bsard_id), 0.0)
        if abs(total - 1.0) > tolerance:
            logger.warning(
                "bsard_id %s: weight sum = %.4f (expected ~1.0)",
                article.bsard_id, total,
            )


def _validate_weight_coverage(
    weights: list[ChunkArticleWeight],
    articles: list,
    tolerance: float = 0.05,
) -> None:
    """Backwards-compatible name; kept so any external test harness keeps working."""
    _log_coverage_warnings(weights, articles, tolerance)


# ── Core algorithm ───────────────────────────────────────────────────────────


def _compute_weights_from_spans(
    chunks: list,
    article_spans: list,
) -> tuple[list[ChunkArticleWeight], dict[int, tuple[int, int]]]:
    """Char-overlap weights from T03's authoritative article spans.

    ``w(c, B) = |chars of B covered by c| / |chars of B|`` using the SAME char
    coordinates T03 used to chunk and to tag bsard_ids, so coverage matches what
    was actually indexed. No re-location and no tokenizer required.

    ``article_spans`` is a list of objects exposing ``bsard_id``, ``start_char``
    and ``end_char`` (T03's ``ArticleSpan``); ``chunks`` expose ``chunk_id``,
    ``start_char`` and ``end_char``. An article may have several spans (it
    appears in more than one place in the PDF); overlap and length are summed
    across all of its spans. Returns ``(weights, ranges)`` where
    ``ranges[bsard_id] = (min_start_char, max_end_char)`` (debug envelope).
    """
    spans_by_bsard: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for s in article_spans:
        if getattr(s, "bsard_id", None) is None or s.end_char <= s.start_char:
            continue
        spans_by_bsard[int(s.bsard_id)].append((int(s.start_char), int(s.end_char)))

    art_length = {
        b: sum(e - st for st, e in spans) for b, spans in spans_by_bsard.items()
    }

    weights: list[ChunkArticleWeight] = []
    for chunk in chunks:
        c_start, c_end = chunk.start_char, chunk.end_char
        for bsard_id, spans in spans_by_bsard.items():
            overlap = sum(
                max(0, min(c_end, e) - max(c_start, st)) for st, e in spans
            )
            if overlap > 0:
                weights.append(ChunkArticleWeight(
                    chunk_id=chunk.chunk_id,
                    bsard_id=bsard_id,
                    weight=overlap / art_length[bsard_id],
                ))

    ranges = {
        b: (min(st for st, _ in spans), max(e for _, e in spans))
        for b, spans in spans_by_bsard.items()
    }
    return weights, ranges


def _compute_weights_and_ranges(
    chunks: list,
    bsard_articles: list,
    tokenizer_name: str,
    raw_text: str | None,
) -> tuple[list[ChunkArticleWeight], dict[int, tuple[int, int]]]:
    """Shared kernel for ``precompute_weights`` and ``precompute_weights_cached``.

    Returns ``(weights, ranges)`` so the cached wrapper can persist ranges as
    ``article_token_ranges.json`` without recomputing them.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    if raw_text is None:
        raw_text = _reconstruct_raw_text_from_chunks(chunks)

    bsard_ranges = _compute_bsard_token_ranges(bsard_articles, raw_text, tokenizer)

    weights: list[ChunkArticleWeight] = []
    for chunk in chunks:
        for bsard_id, (art_start, art_end) in bsard_ranges.items():
            art_length = art_end - art_start
            if art_length == 0:
                continue

            overlap_start = max(chunk.start_token, art_start)
            overlap_end = min(chunk.end_token, art_end)
            overlap = max(0, overlap_end - overlap_start)

            if overlap > 0:
                weights.append(ChunkArticleWeight(
                    chunk_id=chunk.chunk_id,
                    bsard_id=bsard_id,
                    weight=overlap / art_length,
                ))

    return weights, bsard_ranges


def _compute_bsard_token_ranges(
    articles: list,
    raw_text: str,
    tokenizer,
) -> dict[int, tuple[int, int]]:
    """Locate each article in raw_text and return its token range, keyed by bsard_id."""
    ranges: dict[int, tuple[int, int]] = {}
    search_start = 0

    for article in articles:
        anchor = article.text[:100]
        char_pos = raw_text.find(anchor, search_start)

        if char_pos == -1:
            char_pos = _fuzzy_find(raw_text, anchor, search_start)

        if char_pos >= 0:
            prefix_tokens = len(tokenizer.tokenize(raw_text[:char_pos]))
            article_tokens = len(tokenizer.tokenize(article.text))
            ranges[int(article.bsard_id)] = (prefix_tokens, prefix_tokens + article_tokens)
            search_start = char_pos + len(article.text)
        else:
            logger.warning(
                "Could not locate article bsard_id=%s in raw text",
                article.bsard_id,
            )

    return ranges


def _fuzzy_find(raw_text: str, snippet: str, start: int) -> int:
    """Retry find after normalizing whitespace."""
    normalized_raw = " ".join(raw_text[start:].split())
    normalized_snippet = " ".join(snippet.split())
    pos = normalized_raw.find(normalized_snippet)
    if pos == -1:
        return -1
    return start + pos


def _reconstruct_raw_text_from_chunks(chunks: list) -> str:
    """Build raw text by joining chunks sorted by start_token."""
    sorted_chunks = sorted(chunks, key=lambda c: c.start_token)
    return " ".join(c.text for c in sorted_chunks)


def weights_to_lookup(
    weights: list[ChunkArticleWeight],
) -> dict[tuple[str, int], float]:
    """Convert to dict for O(1) lookup during metric computation."""
    return {(w.chunk_id, w.bsard_id): w.weight for w in weights}


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_weights_csv(weights: list[ChunkArticleWeight], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["chunk_id", "bsard_id", "weight"])
        for w in weights:
            writer.writerow([w.chunk_id, w.bsard_id, w.weight])


def load_weights_csv(path: str | Path) -> list[ChunkArticleWeight]:
    weights = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            weights.append(ChunkArticleWeight(
                chunk_id=row["chunk_id"],
                bsard_id=int(row["bsard_id"]),
                weight=float(row["weight"]),
            ))
    return weights


def save_weights_pickle(weights: list[ChunkArticleWeight], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(weights, f)


def load_weights_pickle(path: str | Path) -> list[ChunkArticleWeight]:
    with open(path, "rb") as f:
        return pickle.load(f)
