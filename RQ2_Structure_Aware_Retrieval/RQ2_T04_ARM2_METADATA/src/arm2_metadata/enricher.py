"""Variant generators for T04 retrieval units (AzureNode or aggregated Article).

Six variants. The first five mirror PROJECT_CONTEXT.md §4 / §7; the sixth
(``summary``) is unique to T04 and exists because AzureDI provides per-node
English summaries + keyword tags for free.

Token-budget handling
---------------------
Truncation order when the assembled text exceeds ``max_tokens`` (Step D
spec): (1) drop ``content_summary``, (2) drop ``terms_block``, (3) drop the
``path_to_item`` breadcrumb, (4) trim the tail of the body text. Every event
is logged to ``logger`` and counted on the per-call ``EnrichmentStats``.

Deliberate divergence from RQ1: ``concat_2x``
---------------------------------------------
RQ1's canonical "best hybrid" runner (`run_hybrid_experiments.py`) prepends
the embedding input with a ``concat_2x`` field-weighting trick — the article
text is concatenated with the law-code text repeated 2x. This boosts dense
retrieval on the BSARD test split but conflates the metadata-enrichment
signal that RQ2 wants to measure cleanly. **We deliberately do NOT support
``concat_2x`` here**; T04's variants prepend explicit, named structural
fields (``[Authority: ...]``, ``[Reference: ...]``, ``[Defined terms]``,
etc.) that an ablation can switch on/off. Any future enrichment variant
should follow the same pattern; do not add a hand-tuned text-repetition
trick. See ``RQ1_STACK_FOR_T04.md`` (§K) for the rationale.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from data_loader.models import Article

from .azuredi_loader import AzureNode

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


VARIANTS_NODE = ("raw", "enriched", "summary", "filtered", "full", "terms")
# Article-level shares everything except the node-only "summary" variant.
# Compute, don't restate, so the two sets cannot drift.
VARIANTS_ARTICLE = tuple(v for v in VARIANTS_NODE if v != "summary")

# Variants whose retrieval pipeline runs the boost / filter stage at query
# time. Imported by retriever.run_arm2_metadata to decide `apply_boost`.
# Promote here so a new variant can't silently miss the boost.
BOOST_VARIANTS: frozenset[str] = frozenset({"filtered", "full", "terms"})


# ── Stats ────────────────────────────────────────────────────────────────────


@dataclass
class EnrichmentStats:
    """Per-batch enrichment telemetry. Inspected by retriever.py for log lines."""

    n_units: int = 0
    n_truncated: int = 0
    n_dropped_summary: int = 0
    n_dropped_terms: int = 0
    n_dropped_path: int = 0
    n_tail_trimmed: int = 0
    truncation_rate: float = field(init=False, default=0.0)

    def finalise(self) -> "EnrichmentStats":
        if self.n_units > 0:
            self.truncation_rate = self.n_truncated / self.n_units
        return self


# ── Helpers ──────────────────────────────────────────────────────────────────


def _count_tokens(text: str, tokenizer: "PreTrainedTokenizerBase") -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def _truncate_to_budget(
    text: str, tokenizer: "PreTrainedTokenizerBase", max_tokens: int,
) -> str:
    """Trim ``text`` from the end to fit ``max_tokens`` tokens.

    Decodes the truncated token sequence back to text. Loses the few characters
    after the last kept token boundary, which is acceptable for embedding /
    BM25 input.
    """
    encoding = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = encoding["input_ids"]
    if len(ids) <= max_tokens:
        return text
    offsets = encoding["offset_mapping"]
    keep_end = offsets[max_tokens - 1][1]
    return text[:keep_end]


def _build_header(unit: AzureNode | Article, *, with_in_force: bool = True) -> str:
    """Common doc-context header used by enriched / full / terms variants."""
    if isinstance(unit, AzureNode):
        title = unit.document_title
        authority = unit.issuing_authority
        jurisdiction = unit.jurisdiction
        ref = unit.document_number
        in_force = unit.entry_into_force
    else:
        title = unit.document_title
        authority = unit.issuing_authority
        jurisdiction = unit.jurisdiction
        ref = unit.document_number
        in_force = unit.entry_into_force

    lines = [
        f"[Document: {title}]",
        f"[Authority: {authority}]",
        f"[Jurisdiction: {jurisdiction}]",
        f"[Reference: {ref}]",
    ]
    if with_in_force:
        lines.append(f"[In force: {in_force}]")
    return "\n".join(lines)


def _terms_block(unit: AzureNode | Article, definitions_map: dict[str, str] | None) -> str:
    """Build the "[Defined terms in this article]" block.

    For Article units we use the dataclass ``term_definitions`` dict directly
    (already populated upstream). For AzureNode units we look each defined term
    up in the joined ``definitions_map`` (Term → Definition) so the block
    survives even when the node was built from the raw dump.
    """
    if isinstance(unit, Article):
        items = list(unit.term_definitions.items())
    else:
        defs_map = definitions_map or {}
        items = [(t, defs_map[t]) for t in unit.defined_terms if t in defs_map]
    if not items:
        return ""
    body = "\n".join(f"- {term}: {defn}" for term, defn in items)
    return f"[Defined terms in this article]\n{body}"


# ── Top-level enrichment ─────────────────────────────────────────────────────


def _join_nonempty(*parts: str, sep: str = "\n\n") -> str:
    """Join parts with ``sep``, skipping empty / falsy entries."""
    return sep.join(p for p in parts if p)


def enrich_unit(
    unit: AzureNode | Article,
    variant: str,
    *,
    tokenizer: "PreTrainedTokenizerBase | None" = None,
    max_tokens: int = 512,
    definitions_map: dict[str, str] | None = None,
    stats: EnrichmentStats | None = None,
    apply_truncation: bool = True,
    single_doc_corpus: bool = False,
) -> str:
    """Generate the embedding/BM25 input text for one unit + variant.

    Variant semantics (PROJECT_CONTEXT.md §4 / §7):
      raw       — body text only
      enriched  — doc-context header + body
      summary   — node-level only: header + AzureDI English summary + keywords
                  + section breadcrumb + body
      filtered  — body only (filtering happens at query time)
      full      — doc-context header + body (header text + query-time filter)
      terms     — doc-context header + body + defined-terms block

    Truncation cascade (only applied when ``apply_truncation=True`` AND
    ``tokenizer`` is set AND ``max_tokens > 0``). Cascade order, applied
    only as needed when assembled text exceeds ``max_tokens``:
      (1) ``summary`` variant: drop the [Section] breadcrumb (path_to_item)
          — review #8-B: path is the bigger and more redundant component
          (~132 tok median vs ~88 for summary), so dropping it first frees
          more budget while losing less unique signal.
      (2) ``terms`` variant: drop the terms block.
      (3) ``summary`` variant: drop the [Summary] block (the variant's
          *raison d'être*; sacrificed only after path is gone).
      (4) Universal: trim the body tail to fit.

    Parameters added in review #8 (#8-A and #8-C):
      apply_truncation:    When False, the cascade is skipped entirely; the
                           assembled text is returned as-is. Pass False under
                           sparse-only retrieval — BM25 has no sequence-length
                           cap, so the truncation is wasted (and lossy) work.
      single_doc_corpus:   When True, the doc-context header is omitted from
                           every variant that would otherwise prepend it
                           (`enriched`, `summary`, `full`, `terms`). On a
                           single-doc corpus the header is constant noise
                           — same ~39 tokens on every entry. Detected at
                           indexer level via `len(set(doc_ids)) == 1`.
    """
    if isinstance(unit, AzureNode):
        body = unit.page_content or ""
        path_to_item = unit.path_to_item or ""
        content_summary = unit.content_summary or ""
        keywords = unit.keywords or ""
    else:
        body = unit.text or ""
        path_to_item = ""
        content_summary = unit.summary_sentence or ""
        keywords = ""

    if not body and isinstance(unit, AzureNode):
        # Pure header / structural nodes still get *something* embeddable so
        # the dense index can locate them (we filter them out before indexing
        # in retriever.py, but allow the variant build for logging).
        body = unit.node_source or ""

    # Header is computed once; "" on a single-doc corpus (review #8-C).
    header_full = "" if single_doc_corpus else _build_header(unit)
    header_no_inforce = (
        "" if single_doc_corpus else _build_header(unit, with_in_force=False)
    )

    # Component dict, keyed for deterministic drop-step rebuilds. Empty
    # values are kept here intentionally so each rebuild reads them by name
    # without conditional logic per drop-step (review #15-A2/A3).
    components: dict[str, str] = {}

    def _summary_text() -> str:
        return _join_nonempty(
            header_full,
            f"[Summary] {components['summary']}" if components.get("summary") else "",
            f"[Keywords] {components['keywords']}" if components.get("keywords") else "",
            f"[Section] {components['path']}" if components.get("path") else "",
            body,
        )

    if variant == "raw":
        text = body
    elif variant == "filtered":
        text = body
    elif variant == "enriched":
        text = _join_nonempty(header_full, body)
    elif variant == "full":
        text = _join_nonempty(header_no_inforce, body)
    elif variant == "summary":
        if not isinstance(unit, AzureNode):
            raise ValueError("variant='summary' is node-level only (T04 unique).")
        components.update({
            "summary": content_summary,
            "keywords": keywords,
            "path": path_to_item,
        })
        text = _summary_text()
    elif variant == "terms":
        terms = _terms_block(unit, definitions_map)
        components["terms_block"] = terms
        text = _join_nonempty(header_full, body, terms)
    else:
        raise ValueError(f"Unknown variant: {variant!r}")

    if tokenizer is None or max_tokens <= 0 or not apply_truncation:
        if stats is not None:
            stats.n_units += 1
        return text

    # ── Token-budget cascade ──────────────────────────────────────────────
    # Steps run in fixed order. Each step is a no-op when its target
    # component wasn't present (review #15-A3: was incrementing n_dropped_*
    # spuriously). After the cascade, if still over budget, the body tail
    # is trimmed.
    n_tok = _count_tokens(text, tokenizer)
    dropped: list[str] = []

    # Step 1 — summary variant only: drop the [Section] breadcrumb FIRST
    # (review #8-B; it's the bigger and more redundant component).
    if n_tok > max_tokens and variant == "summary" and components.get("path"):
        components["path"] = ""
        text = _summary_text()
        n_tok = _count_tokens(text, tokenizer)
        if stats is not None: stats.n_dropped_path += 1
        dropped.append("path")

    # Step 2 — terms variant only: drop terms block (skip if it was empty)
    if n_tok > max_tokens and variant == "terms" and components.get("terms_block"):
        text = _join_nonempty(header_full, body)
        n_tok = _count_tokens(text, tokenizer)
        if stats is not None: stats.n_dropped_terms += 1
        dropped.append("terms")

    # Step 3 — summary variant only: drop the [Summary] block (the variant's
    # *raison d'être*; only after path is gone). Review #8-B reorder; review
    # #15-A2 — keywords are kept here (previous implementation also dropped
    # them silently).
    if n_tok > max_tokens and variant == "summary" and components.get("summary"):
        components["summary"] = ""
        text = _summary_text()
        n_tok = _count_tokens(text, tokenizer)
        if stats is not None: stats.n_dropped_summary += 1
        dropped.append("summary")

    # Step 4 — universal: trim body tail to fit
    if n_tok > max_tokens:
        text = _truncate_to_budget(text, tokenizer, max_tokens)
        if stats is not None: stats.n_tail_trimmed += 1
        dropped.append("tail")

    if stats is not None:
        stats.n_units += 1
        if dropped:
            stats.n_truncated += 1
            logger.debug(
                "enrich_unit truncated unit (variant=%s dropped=%s)", variant, dropped,
            )
    return text


# ── Article aggregation (article-level corpus) ───────────────────────────────


def build_articles_from_nodes(
    nodes: list[AzureNode],
    my_documents: dict[int, dict],
    definitions_map: dict[str, str],
) -> list[Article]:
    """Aggregate AzureDI nodes into one Article per (doc_id, bsard_id).

    Concatenates the content of all nodes whose ``bsard_ids`` contains the
    target bsard_id, ordered by ``(page_number, node_id)``. Only content
    nodes (``is_header=False``) contribute body text; header nodes are
    skipped because they pollute embeddings.

    Each Article carries:
      - text                — concatenated FR page_content
      - article_id          — string ``"doc_<doc_id>_bsard_<bsard_id>"``
      - bsard_id            — int (the BSARD canonical id)
      - document_*, jurisdiction, etc. — inherited from MyDocuments
      - defined_terms       — sorted union across contributing nodes
      - term_definitions    — same union, with definitions joined in
      - article_number      — first matched_article_number, when known
    """
    by_bid: dict[tuple[int, int], list[AzureNode]] = {}
    for n in nodes:
        if n.is_header or not n.bsard_ids:
            continue
        for bid in n.bsard_ids:
            by_bid.setdefault((n.doc_id, bid), []).append(n)

    articles: list[Article] = []
    for (doc_id, bsard_id), members in sorted(by_bid.items()):
        # Sort by reading order through the source PDF. Two reasons this is the
        # right choice and not just the obvious one:
        #   1. BM25 (T04's default sparse-only path) is bag-of-words → order is
        #      provably irrelevant for ranking. Verified empirically: reversing
        #      paragraph order produces bit-identical top-k.
        #   2. The truncation cascade in enrich_unit() trims the *tail* when
        #      the assembled text exceeds max_tokens. Reading order puts the
        #      article-anchor node ("Art. D230. ...") first, so truncation
        #      drops continuation paragraphs (least informative) before the
        #      article opening. For all 9 GT articles in the doc-2 subset the
        #      anchor IS first under this sort.
        # A `parent_id` tree traversal would only revisit the same nodes; the
        # post-fix linker emits orphan continuations as *siblings* of the
        # anchor, not children, so a tree walk wouldn't find them. See #13 in
        # the review log.
        members.sort(key=lambda n: (n.page_number, n.node_id))
        body = "\n\n".join(m.page_content for m in members if m.page_content)
        defined_terms = sorted({t for m in members for t in m.defined_terms})
        term_defs = {t: definitions_map[t] for t in defined_terms if t in definitions_map}
        first_match = next(
            (an for m in members for an in m.matched_article_numbers if an), "",
        )

        meta = my_documents.get(doc_id, {})
        articles.append(Article(
            article_id=f"doc_{doc_id}_bsard_{bsard_id}",
            text=body,
            document_id=doc_id,
            document_title=meta.get("document_title", ""),
            jurisdiction=meta.get("jurisdiction", ""),
            issuing_authority=meta.get("issuing_authority", ""),
            document_number=meta.get("document_number", ""),
            issue_date=meta.get("issue_date", ""),
            entry_into_force=meta.get("entry_into_force", ""),
            regulatory_status=meta.get("regulatory_status", "Effective"),
            language=[],
            summary_sentence="",
            defined_terms=defined_terms,
            term_definitions=term_defs,
            cross_references=[],
            token_count=0,
            article_number=first_match,
            bsard_id=bsard_id,
        ))
    return articles


def build_definitions_map(definitions_df) -> dict[str, str]:
    """Term → Definition map for use by ``enrich_unit(variant='terms')``."""
    if definitions_df is None or len(definitions_df) == 0:
        return {}
    out: dict[str, str] = {}
    for _, row in definitions_df.iterrows():
        term = row.get("Term")
        defn = row.get("Definition")
        if isinstance(term, str) and isinstance(defn, str):
            out[term] = defn
    return out
