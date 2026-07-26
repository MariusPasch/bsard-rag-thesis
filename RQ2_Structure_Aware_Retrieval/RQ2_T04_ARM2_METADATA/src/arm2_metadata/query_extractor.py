"""Query-time signal extraction (PROJECT_CONTEXT.md §5).

Three methods, applied in sequence and short-circuiting LLM calls when the
deterministic methods already produced a useful signal.

  1. Term-dictionary matching       — substring scan against
     ``DocumentDefinitions.csv`` (Term + TranslatedTerm).
  2. Rule-based regex               — French legal patterns for
     document_number / article_ref / date_ref / jurisdiction.
  3. LLM classification             — only when methods 1+2 yielded nothing.
     Result is cached by query-text hash so re-runs are free.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from shared.llm import LLMClient

logger = logging.getLogger(__name__)


# ── Signal container ─────────────────────────────────────────────────────────


@dataclass
class QuerySignals:
    matched_terms: list[str] = field(default_factory=list)
    term_node_ids: set[str] = field(default_factory=set)        # entry ids of T04 index
    term_article_ids: set[str] = field(default_factory=set)     # alias kept for spec parity
    article_refs: list[str] = field(default_factory=list)
    document_numbers: list[str] = field(default_factory=list)
    date_ref: str | None = None
    jurisdiction: str | None = None
    legal_domain: str | None = None
    temporal_scope: str | None = None

    def has_any_filter(self) -> bool:
        return bool(
            self.matched_terms
            or self.article_refs
            or self.document_numbers
            or self.date_ref
            or self.jurisdiction
        )


# ── Method 1: term-dictionary matching ──────────────────────────────────────


def _extract_term_signals(
    query: str,
    definitions_df: pd.DataFrame,
    signals: QuerySignals,
    node_to_bsard: dict[tuple[int, int], list[int]] | None = None,
) -> QuerySignals:
    """Substring-match the query against Term + TranslatedTerm columns.

    ``node_to_bsard``: optional `{(doc_id, node_id): [bsard_id, ...]}` map.
    When provided, ``signals.term_node_ids`` is populated with BOTH the
    node-level entry id (`node_<doc>_<id>`) AND the article-level entry id
    (`doc_<doc>_bsard_<bid>`) — so the ``used_in`` boost in `boost.py`
    fires regardless of whether the running index is at node or article
    granularity. Without this map, only the node-level shape is emitted
    (the article-level boost is dead, see review #15-A1).
    """
    if definitions_df is None or len(definitions_df) == 0:
        return signals
    q_lower = query.lower()
    seen_terms: set[str] = set()

    for _, row in definitions_df.iterrows():
        term = row.get("Term") if isinstance(row.get("Term"), str) else None
        translated = (
            row.get("TranslatedTerm")
            if isinstance(row.get("TranslatedTerm"), str) else None
        )
        if not term and not translated:
            continue
        match = (term and term.lower() in q_lower) or (
            translated and translated.lower() in q_lower
        )
        if not match:
            continue
        if term not in seen_terms and term:
            signals.matched_terms.append(term)
            seen_terms.add(term)

        used_in_raw = row.get("UsedIn")
        try:
            used_in = json.loads(used_in_raw) if isinstance(used_in_raw, str) else (
                used_in_raw or []
            )
        except json.JSONDecodeError:
            used_in = []
        doc_id_raw = row.get("DocumentId")
        try:
            doc_id = int(doc_id_raw) if doc_id_raw is not None else None
        except (TypeError, ValueError):
            doc_id = None

        for nid in used_in:
            try:
                nid_int = int(nid)
            except (TypeError, ValueError):
                continue
            if doc_id is not None:
                node_id = f"node_{doc_id}_{nid_int}"
                signals.term_node_ids.add(node_id)
                signals.term_article_ids.add(node_id)
                # Also emit the article-level entry-id shape when the linker's
                # node→bsard map is provided — otherwise the used_in boost
                # silently never fires for unit="article".
                if node_to_bsard:
                    for bid in node_to_bsard.get((doc_id, nid_int), []):
                        article_id = f"doc_{doc_id}_bsard_{bid}"
                        signals.term_node_ids.add(article_id)
                        signals.term_article_ids.add(article_id)
            else:
                fallback_id = str(nid_int)
                signals.term_node_ids.add(fallback_id)
                signals.term_article_ids.add(fallback_id)

    return signals


# ── Method 2: rule-based regex ───────────────────────────────────────────────


_PATTERNS = {
    "document_number": [
        re.compile(
            r"(?:loi|décret|decret|arrêté|arrete|decision|règlement|reglement)"
            r"\s+(?:du\s+)?\d{1,2}[-/]\d{1,2}[-/]\d{2,4}",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?:DECISION|REGULATION|DIRECTIVE)"
            r"\s*\(?(?:EU|EC|EEC)\)?\s*(?:No\.?\s*)?\d{4}[/-]\d+",
            re.IGNORECASE,
        ),
    ],
    "article_ref": [
        re.compile(r"art(?:icle)?\.?\s*[D\.]?\s*\d+(?:[.\-]\d+)*", re.IGNORECASE),
    ],
    "date_ref": [
        re.compile(
            r"\d{1,2}\s+(?:janvier|février|fevrier|mars|avril|mai|juin|juillet|"
            r"août|aout|septembre|octobre|novembre|décembre|decembre)"
            r"\s+\d{4}",
            re.IGNORECASE,
        ),
    ],
    "jurisdiction_signal": [
        # Note: EU acronym is matched case-SENSITIVELY (separate pattern below)
        # because lowercase "eu" is the French verb "had" — would false-fire
        # on "qu'il a eu". Review #15-C5.
        re.compile(
            r"(?P<wal>wallon(?:ne)?)|(?P<fla>flamand(?:e)?)|"
            r"(?P<bru>bruxell(?:ois|es))|(?P<fed>fédéral(?:e)?|federal(?:e)?)|"
            r"(?P<eu>européen(?:ne)?|europ(?:e|éenne))|(?P<be>belge)",
            re.IGNORECASE,
        ),
        # Case-sensitive EU/UE acronym match (no IGNORECASE flag).
        re.compile(r"(?P<eu>\b(?:EU|UE)\b)"),
    ],
}

_JURISDICTION_GROUP_TO_LABEL = {
    "wal": "Walloon",
    "fla": "Flemish",
    "bru": "Brussels",
    "fed": "Belgian Federal",
    "eu": "EU",
    "be": "Belgian Federal",
}


def _extract_regex_signals(query: str, signals: QuerySignals) -> QuerySignals:
    for pattern in _PATTERNS["document_number"]:
        signals.document_numbers.extend(m.group(0) for m in pattern.finditer(query))
    for pattern in _PATTERNS["article_ref"]:
        signals.article_refs.extend(m.group(0) for m in pattern.finditer(query))
    for pattern in _PATTERNS["date_ref"]:
        m = pattern.search(query)
        if m and signals.date_ref is None:
            signals.date_ref = m.group(0)
    for pattern in _PATTERNS["jurisdiction_signal"]:
        m = pattern.search(query)
        if m and signals.jurisdiction is None:
            for grp, label in _JURISDICTION_GROUP_TO_LABEL.items():
                if m.groupdict().get(grp):
                    signals.jurisdiction = label
                    break
    return signals


# ── Method 3: LLM classification ─────────────────────────────────────────────


_LLM_CLASSIFY_PROMPT = """Given this legal question: "{query}"

Extract the following if identifiable:
- jurisdiction: EU / Belgian Federal / Walloon / Flemish / Brussels / Unknown
- legal_domain: environmental / civil / criminal / commercial / administrative / Unknown
- temporal_scope: specific date or "current"

Respond in JSON only, no other text."""


def _query_cache_key(query: str) -> str:
    return hashlib.sha256(query.encode("utf-8")).hexdigest()[:16]


def _extract_llm_signals(
    query: str,
    llm_client: "LLMClient",
    signals: QuerySignals,
    cache: dict[str, dict] | None = None,
) -> QuerySignals:
    key = _query_cache_key(query)
    if cache is not None and key in cache:
        payload = cache[key]
    else:
        try:
            payload = llm_client.generate_json(_LLM_CLASSIFY_PROMPT.format(query=query))
        except Exception as exc:
            logger.warning("LLM classification failed for query %r: %s", query[:60], exc)
            payload = {}
        if cache is not None:
            cache[key] = payload

    juris = payload.get("jurisdiction")
    if isinstance(juris, str) and juris.strip().lower() != "unknown":
        signals.jurisdiction = juris.strip()

    domain = payload.get("legal_domain")
    if isinstance(domain, str) and domain.strip().lower() != "unknown":
        signals.legal_domain = domain.strip()

    temporal = payload.get("temporal_scope")
    if isinstance(temporal, str) and temporal.strip():
        signals.temporal_scope = temporal.strip()

    return signals


# ── Top-level entry ──────────────────────────────────────────────────────────


def extract_query_signals(
    query: str,
    definitions_df: pd.DataFrame,
    llm_client: "LLMClient | None" = None,
    *,
    llm_cache: dict[str, dict] | None = None,
    use_llm: bool = True,
    node_to_bsard: dict[tuple[int, int], list[int]] | None = None,
) -> QuerySignals:
    """Apply methods 1–3 in order, short-circuiting the LLM call.

    Args:
        query:           Natural-language question text.
        definitions_df:  DocumentDefinitions DataFrame from the AzureDI loader.
        llm_client:      Optional LLMClient for method 3.
        llm_cache:       External cache dict ``{query_hash: payload}`` for the
                         LLM JSON output. Pass the same dict across queries to
                         deduplicate calls.
        use_llm:         When False, skip method 3 even if llm_client is set
                         (used by ``filtered`` variant when we only want
                         deterministic signals).
        node_to_bsard:   Linker output mapping `(doc_id, node_id) -> bsard_ids`.
                         When provided, the ``used_in`` boost fires for both
                         node-level AND article-level indices. Without it,
                         article-level retrieval silently misses the boost.
    """
    signals = QuerySignals()
    signals = _extract_term_signals(query, definitions_df, signals, node_to_bsard)
    signals = _extract_regex_signals(query, signals)

    if use_llm and llm_client is not None and not signals.has_any_filter():
        signals = _extract_llm_signals(query, llm_client, signals, cache=llm_cache)

    return signals
