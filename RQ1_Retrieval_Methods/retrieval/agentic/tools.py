"""
Tool execution functions for the ReAct agent (Tier 4.2).

Tools
-----
search[requête]  — retrieve top_k_shown articles from the backbone retriever
lookup[id]       — fetch full article text by numeric ID
finish[done]     — pure stopping signal (no observation text, no IDs)

Pre-flight gate (T1)
--------------------
validate_tool_call(tool_name, tool_args) -> (ok: bool, error_msg: str)
Called before any tool execution; rejects malformed or unknown tool calls.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Optional, Protocol

import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALLOWED_TOOLS = {"search", "lookup", "finish"}

# Default characters of article text shown in a search snippet.
# Round-1 used 200 chars (raw prefix); Round-2 §16.3 (Block C) cuts this to
# ~80 chars and prefers a "title / first-sentence" extraction so the agent
# is forced into a search → lookup → search workflow. The value is now a
# parameter on execute_search; this constant is just its default.
_SEARCH_SNIPPET_CHARS = 80

# Maximum characters shown in a lookup result (keeps observations bounded)
_LOOKUP_MAX_CHARS = 1500

# Sentence-boundary regex used by Block C's first-sentence snippet extractor.
# Matches end-of-sentence punctuation followed by whitespace OR end-of-string.
_SENTENCE_END_RE = re.compile(r"(?:[\.!?]+)(?:\s|$)")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ToolResult:
    """Structured result from a single tool invocation."""
    tool_name:   str          # "search" | "lookup" | "finish"
    raw_text:    str          # Observation text to append to the scratchpad
    article_ids: list[int]    # IDs seen in this result (empty for finish)
    latency_ms:  float        # Wall-clock time for this call


class RetrieverProtocol(Protocol):
    def retrieve(self, query: str, top_k: int) -> tuple[list[int], float]: ...


# ---------------------------------------------------------------------------
# Pre-flight validation gate (T1)
# ---------------------------------------------------------------------------

def validate_tool_call(
    tool_name: Optional[str],
    tool_args: Optional[dict],
) -> tuple[bool, str]:
    """
    Validate a parsed tool call before execution.

    Returns (True, "") on success or (False, reason) on failure.
    """
    if tool_name is None or tool_name not in ALLOWED_TOOLS:
        valid = "', '".join(sorted(ALLOWED_TOOLS))
        return False, f"Outil inconnu : '{tool_name}'. Outils valides : '{valid}'."

    if tool_args is None:
        return False, "Arguments manquants."

    if tool_name == "search":
        q = tool_args.get("query", "")
        if not isinstance(q, str) or not q.strip():
            return False, "search[] requiert une requête non vide."

    elif tool_name == "lookup":
        aid = tool_args.get("article_id")
        if not isinstance(aid, int):
            return False, "lookup[] requiert un identifiant numérique entier."

    return True, ""


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def _first_sentence_snippet(text: str, max_chars: int) -> str:
    """
    Return the first sentence of `text` if one ends within `max_chars + 50`
    characters; otherwise hard-cut at `max_chars` with an ellipsis.

    `max_chars` is a soft cap: a short first sentence (e.g. an article title
    "Article un.") is returned in full even if it is shorter than `max_chars`.
    The design goal (§16.3 Block C) is to push body text behind `lookup`.
    """
    text = (text or "").strip()
    if not text:
        return ""
    window_end = min(len(text), max_chars + 50)
    m = _SENTENCE_END_RE.search(text, 0, window_end)
    if m:
        return text[:m.end()].rstrip()
    if len(text) > max_chars:
        return text[:max_chars].rstrip() + "…"
    return text


def execute_search(
    query: str,
    retriever: RetrieverProtocol,
    df_articles: pd.DataFrame,
    top_k_shown: int = 3,
    snippet_chars: int = _SEARCH_SNIPPET_CHARS,
) -> ToolResult:
    """
    Retrieve top_k_shown articles and format them as a numbered observation.

    Round-1 default: top_k_shown=5, snippet=200-char raw prefix.
    Round-2 (§16.3) default: top_k_shown=3, snippet ≈ first sentence
    (snippet_chars ≈ 80) so body text only becomes visible via `lookup`.

    Parameters
    ----------
    snippet_chars : char budget for each per-article snippet. Used by
                    `_first_sentence_snippet` which prefers a sentence cut
                    over a hard truncation.
    """
    id_to_text: dict[int, str] = dict(
        zip(df_articles["article_id"], df_articles["article_text"])
    )

    t0 = time.perf_counter()
    article_ids, _ = retriever.retrieve(query, top_k=top_k_shown)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    lines = ["\nObservation :"]
    for i, aid in enumerate(article_ids, start=1):
        text    = id_to_text.get(aid, "(texte non disponible)")
        snippet = _first_sentence_snippet(text, snippet_chars)
        lines.append(f"  [{i}] ID={aid} — {snippet}")

    return ToolResult(
        tool_name="search",
        raw_text="\n".join(lines),
        article_ids=list(article_ids),
        latency_ms=latency_ms,
    )


def execute_lookup(
    article_id: int,
    df_articles: pd.DataFrame,
) -> ToolResult:
    """
    Return the full text for article_id.

    Truncates to _LOOKUP_MAX_CHARS characters.
    Returns an empty article_ids list if the ID is not found.
    """
    id_to_text: dict[int, str] = dict(
        zip(df_articles["article_id"], df_articles["article_text"])
    )

    t0   = time.perf_counter()
    text = id_to_text.get(article_id)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    if text is None:
        return ToolResult(
            tool_name="lookup",
            raw_text=f"\nObservation :\n  ID={article_id} — Article non trouvé.",
            article_ids=[],
            latency_ms=latency_ms,
        )

    if len(text) > _LOOKUP_MAX_CHARS:
        text = text[:_LOOKUP_MAX_CHARS].rstrip() + "…"

    return ToolResult(
        tool_name="lookup",
        raw_text=f"\nObservation :\n  ID={article_id} — Texte complet : {text}",
        article_ids=[article_id],
        latency_ms=latency_ms,
    )


def execute_finish() -> ToolResult:
    """
    Pure stopping signal — no observation text, no article IDs.

    ID consolidation is handled deterministically by the finalize node.
    """
    return ToolResult(
        tool_name="finish",
        raw_text="",
        article_ids=[],
        latency_ms=0.0,
    )
