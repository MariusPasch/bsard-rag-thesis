"""
ReAct — Reasoning + Acting retriever (Tier 4.2).

Architecture
------------
ReActRetriever
    Public retriever matching the project RetrieverProtocol:
        retrieve(query, top_k=100) -> (ranked_article_ids, latency_ms)

    Internally runs a LangGraph StateGraph with three nodes:
        generate_step — LLM generates Thought + Action; action is parsed
        execute_tool  — T1 pre-flight gate, D2 overlap guard, tool dispatch,
                        C1 sliding-window scratchpad truncation
        finalize      — D1 post-hoc LLaMA re-ranking of all observed IDs

Design contracts (from TIER42_AGENTIC_REACT_PLAN.md)
------------------------------------------------------
finish[done] semantics
    finish[done] is a PURE stopping signal. Article IDs are extracted
    deterministically from the observation trace; the LLM never lists IDs
    in the finish action.

C1 sliding-window budget
    MAX_SCRATCHPAD_TOKENS = 5500 (whitespace-split word count as proxy).
    When exceeded, the oldest Observation block is replaced with a truncation
    marker; its IDs are archived in observed_ids_archive, not discarded.

D1 post-hoc re-ranking
    After finish[done] or step >= max_steps, the union of observed_ids and
    observed_ids_archive is re-scored against the ORIGINAL query using
    LLM_JUDGE_BINARY_PROMPT. The original query is immutable throughout.

D2 semantic overlap guard
    Prevents redundant search iterations by checking Jaccard similarity
    between new result IDs and all previously seen IDs. Fires if
    Jaccard >= overlap_threshold.
"""

from __future__ import annotations

import re
import time
from typing import Optional, TypedDict

import numpy as np
import pandas as pd

from retrieval.agentic.llm_client import OllamaClient
from retrieval.agentic.llm_eval_prompts import (
    LLM_JUDGE_BINARY_PROMPT,
    format_fewshot_block,
    parse_binary_judgment,
)
from retrieval.agentic.prompts import (
    REACT_FEWSHOT_TRAJECTORIES_BLOCK,
    REACT_GAP_PROMPT,
    REACT_INVALID_ACTION_OBSERVATION,
    REACT_INVALID_ACTION_OBSERVATION_TEMPLATE,
    REACT_SYSTEM_PROMPT,
    REACT_USER_TEMPLATE,
)
from retrieval.agentic.tools import (
    execute_finish,
    execute_lookup,
    execute_search,
    validate_tool_call,
)


# ---------------------------------------------------------------------------
# Tool schemas for Ollama native function calling (Round-2 Block A)
# ---------------------------------------------------------------------------

REACT_TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Recherche les articles de loi belge les plus pertinents "
                           "pour une requête en français.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Requête de recherche en français.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Affiche le texte complet d'un article par son identifiant numérique.",
            "parameters": {
                "type": "object",
                "properties": {
                    "article_id": {
                        "type": "integer",
                        "description": "Identifiant numérique de l'article.",
                    },
                },
                "required": ["article_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Termine la recherche une fois les articles pertinents trouvés.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_SCRATCHPAD_TOKENS = 5500

# Regex to extract Action line — Round-1 strict canonical form:
#   Action : search[...] | lookup[...] | finish[done]
_ACTION_RE = re.compile(
    r"Action\s*:\s*(search|lookup|finish)\[([^\]]*)\]",
    re.IGNORECASE,
)

# Regex to extract article IDs from observation text
_ID_RE = re.compile(r"\bID=(\d+)\b")

# ---------------------------------------------------------------------------
# Round-2 Block A — broadened action regex
# ---------------------------------------------------------------------------
# The Round-1 strict regex misses common Llama 3.1 8B emission patterns
# (53 % null-action rate per §13.1). The broadened parser accepts:
#   search[query]         search(query)         search:"query"        — bracket / paren / colon openers
#   search["query"]       search(« query »)     search « query »      — French / ASCII quotes
#   Action: search query  search query          search : query        — opener-less, lookahead-bounded
# It then closes the argument on the first of: ], ), ", », newline, or end-of-string,
# and returns the canonical (verb, args) tuple.
#
# When use_action_regex_v2=True is passed to ReActRetriever, the parser tries
# the strict regex first (so v1 traces remain reproducible) and only falls
# back to the permissive variants on miss. v1 default keeps strict-only.

# Helpers split out so unit tests can target each variant independently.
_RE_PAREN  = re.compile(
    r"Action\s*:\s*(search|lookup|finish)\s*\(\s*[«\"']?\s*(.*?)\s*[»\"']?\s*\)",
    re.IGNORECASE | re.DOTALL,
)
_RE_COLON  = re.compile(
    r"Action\s*:\s*(search|lookup|finish)\s*:\s*[«\"']?\s*([^\n\"»']+?)\s*[»\"']?\s*(?:\n|$)",
    re.IGNORECASE,
)
_RE_FRENCH = re.compile(
    r"Action\s*:\s*(search|lookup|finish)\s*«\s*(.+?)\s*»",
    re.IGNORECASE | re.DOTALL,
)
_RE_BARE   = re.compile(
    r"Action\s*:\s*(search|lookup|finish)\s+(.+?)\s*(?:\n|$)",
    re.IGNORECASE,
)


def _normalise_args_for_verb(verb: str, raw_args: str) -> tuple[str, dict | None]:
    """Convert a raw arg string to the canonical {query} | {article_id} | {} dict."""
    raw_args = (raw_args or "").strip().strip("«»\"'")
    if verb == "search":
        return ("search", {"query": raw_args})
    if verb == "lookup":
        try:
            return ("lookup", {"article_id": int(raw_args)})
        except (ValueError, TypeError):
            return ("lookup", {"article_id": None})
    if verb == "finish":
        return ("finish", {})
    return (verb, None)


def _parse_action_v2(llm_output: str) -> tuple[Optional[str], Optional[dict]]:
    """
    Permissive parser tried after the strict regex misses.

    Order: parens → French quotes → colon → bare-word fallback.
    First match wins. Returns (None, None) only when no opener variant matches.
    """
    for rx in (_RE_PAREN, _RE_FRENCH, _RE_COLON, _RE_BARE):
        m = rx.search(llm_output)
        if m:
            verb = m.group(1).lower()
            return _normalise_args_for_verb(verb, m.group(2))
    return None, None


# ---------------------------------------------------------------------------
# State definition
# ---------------------------------------------------------------------------

class ReActState(TypedDict):
    # ── Input (set once at invocation, immutable) ────────────────────────────
    query: str

    # ── Loop state ───────────────────────────────────────────────────────────
    scratchpad:        str
    scratchpad_tokens: int
    step:              int
    done:              bool

    # ── Parsed action (from generate_step, consumed by execute_tool) ─────────
    tool_name: Optional[str]
    tool_args: Optional[dict]
    last_raw_output: str   # raw LLM output of the most recent generate_step
                           # — used by Round-2 templated invalid-action echo

    # ── ID accumulation ──────────────────────────────────────────────────────
    observed_ids:         list[int]   # IDs in current scratchpad window
    observed_ids_archive: list[int]   # IDs dropped by C1 truncation

    # ── Round-2 Block B — reformulation thrash detection ─────────────────────
    prior_search_queries: list[str]   # all search query strings issued so far
    pending_gap_facets:   str         # gap-prompt output for the next generate_step

    # ── Output ───────────────────────────────────────────────────────────────
    final_article_ids: list[int]
    pool_topped_up:    bool   # Round-2 Block D — set in finalize_node when
                              # the first stage's top-K had to backfill the
                              # pool because the agent's exploration was too
                              # narrow.

    # ── Diagnostics ──────────────────────────────────────────────────────────
    latency_breakdown_ms: dict         # llm_generate, retrieval, rerank_posthoc
    agent_loop_trace:     list[dict]   # one entry per step
    preflight_failures:   int
    overlap_guard_fires:  int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_action(
    llm_output: str,
    use_v2_fallback: bool = False,
) -> tuple[Optional[str], Optional[dict]]:
    """
    Parse tool_name and tool_args from LLM output.

    The strict regex (`_ACTION_RE`) is tried first to keep Round-1 traces
    reproducible. If `use_v2_fallback=True` and the strict regex misses, the
    permissive parser (`_parse_action_v2`) is tried as a backstop.

    Returns (None, None) if no recognised Action line is found.
    """
    match = _ACTION_RE.search(llm_output)
    if match:
        tool_name = match.group(1).lower()
        args_str  = match.group(2).strip()
        return _normalise_args_for_verb(tool_name, args_str)

    if use_v2_fallback:
        return _parse_action_v2(llm_output)
    return None, None


# ---------------------------------------------------------------------------
# Round-2 Block B — query-token Jaccard for D2 reformulation guard
# ---------------------------------------------------------------------------

# Conservative French stopword set used as the lightweight fallback when spaCy
# is not initialised. Kept explicit so unit tests do not need the corpus.
_FALLBACK_STOPWORDS = frozenset({
    "le", "la", "les", "l", "un", "une", "des", "du", "de", "d",
    "à", "au", "aux", "et", "ou", "où", "que", "qui", "quoi", "dont",
    "ce", "cet", "cette", "ces", "mon", "ma", "mes", "ton", "ta", "tes",
    "son", "sa", "ses", "notre", "votre", "leur", "leurs",
    "je", "tu", "il", "elle", "on", "nous", "vous", "ils", "elles",
    "me", "te", "se", "lui", "y", "en",
    "est", "sont", "été", "être", "avoir", "a", "ai", "ont",
    "pour", "par", "avec", "sur", "dans", "sous", "entre", "vers",
    "comme", "si", "pas",   # 'pas' is in legal-keep elsewhere but here we want it filtered for queries
    "tout", "tous", "toute", "toutes",
    "plus", "moins", "tres", "très",
})

_TOKEN_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+", re.UNICODE)


def _query_tokens_fallback(text: str) -> list[str]:
    """Lowercase regex tokenization + small stopword filter — no spaCy."""
    return [
        t for t in (m.group(0).lower() for m in _TOKEN_RE.finditer(text or ""))
        if t not in _FALLBACK_STOPWORDS and len(t) >= 2
    ]


def _query_tokens(text: str) -> list[str]:
    """
    Lemmatised query tokens for the Round-2 D2 guard. Tries spaCy via
    `retrieval.preprocessing.tokenize_lemmatize` (already loaded by the BM25
    backbone in production) and falls back to a regex+stopword tokenizer if
    spaCy is unavailable (unit-test path).
    """
    try:
        from retrieval.preprocessing import tokenize_lemmatize
        return tokenize_lemmatize(text or "")
    except Exception:
        return _query_tokens_fallback(text or "")


def _query_token_jaccard(a: str, b: str) -> float:
    """Jaccard similarity over the (lemmatised) token sets of two query strings."""
    sa, sb = set(_query_tokens(a)), set(_query_tokens(b))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _extract_malformed_action_line(llm_output: str) -> str:
    """
    Pull the first line containing 'Action' from the LLM output for echo-back.
    Falls back to the first non-empty stripped line, capped at 200 chars.
    """
    for line in (llm_output or "").splitlines():
        stripped = line.strip()
        if stripped and "action" in stripped.lower():
            return stripped[:200]
    for line in (llm_output or "").splitlines():
        stripped = line.strip()
        if stripped:
            return stripped[:200]
    return "(sortie vide)"


def _extract_step_fragment(llm_output: str) -> str:
    """
    Extract the Pensée + Action fragment from LLM output.
    Takes all lines up to and including the first Action line.
    """
    lines    = llm_output.split("\n")
    fragment = []
    for line in lines:
        fragment.append(line)
        if re.match(r"^\s*Action\s*:", line):
            break
    return "\n".join(fragment).strip()


def _count_tokens(text: str) -> int:
    """Estimate token count as whitespace-split word count (fast proxy)."""
    return len(text.split())


def _truncate_scratchpad(
    scratchpad: str,
    observed_ids: list[int],
) -> tuple[str, list[int], list[int]]:
    """
    C1 sliding-window truncation.

    Finds the oldest Observation block, archives its article IDs, and
    replaces the block text with a truncation marker.

    Returns
    -------
    (new_scratchpad, new_observed_ids, newly_archived_ids)
    """
    MARKER = "\nObservation :\n  [Observation tronquée — IDs archivés]\n"

    obs_start = scratchpad.find("\nObservation :")
    if obs_start == -1:
        return scratchpad, observed_ids, []

    next_pensee = scratchpad.find("\nPensée :", obs_start + 1)
    obs_end     = next_pensee if next_pensee != -1 else len(scratchpad)

    obs_block       = scratchpad[obs_start:obs_end]
    newly_archived  = [int(m) for m in _ID_RE.findall(obs_block)]
    archived_set    = set(newly_archived)
    new_observed    = [aid for aid in observed_ids if aid not in archived_set]
    new_scratchpad  = scratchpad[:obs_start] + MARKER + scratchpad[obs_end:]

    return new_scratchpad, new_observed, newly_archived


# ---------------------------------------------------------------------------
# ReActRetriever
# ---------------------------------------------------------------------------

class ReActRetriever:
    """
    Drop-in retriever wrapping any Tier 1–3 backbone with the ReAct loop.

    Parameters
    ----------
    retriever                  : any object with .retrieve(query, top_k) -> (list[int], float)
    df_articles                : DataFrame with columns [article_id, article_text]
    llm_client                 : OllamaClient (llama3.1:8b)
    top_k_shown                : docs shown per search call (Round-1 used 5,
                                 Round-2 §16.3 default 3)
    search_snippet_chars       : per-article snippet budget for search results.
                                 Round-1 used 200 (raw prefix); Round-2 §16.3
                                 default 80 with a first-sentence cut, forcing
                                 body text through `lookup`.
    max_steps                  : hard step limit (tuned on val, default 6)
    overlap_threshold          : D2 Jaccard threshold to suppress redundant searches (default 0.7)
    fewshot_examples           : dict from load_fewshot_examples(); None = no few-shot injection
    max_article_tokens         : article text truncation for D1 re-ranking (whitespace tokens)
    max_tokens_generate        : max_tokens for the per-step generation call
                                 (Round-2 Block A — was 64 in Round-1, default 128)
    use_function_calling       : if True, use Ollama's native /api/chat + tools schema
                                 for action emission. Falls back to regex on parse miss.
                                 Default False = Round-1 behaviour.
    use_action_regex_v2        : if True, broaden the action parser to accept
                                 search(...), search:"...", French «…», bare prose.
                                 Default False = Round-1 strict regex only.
    use_invalid_action_echo    : if True, on parse fail use the templated rejection
                                 that echoes the malformed line back. Default False
                                 = Round-1 generic rejection.
    use_fewshot_trajectories   : if True, prepend REACT_FEWSHOT_TRAJECTORIES_BLOCK
                                 to REACT_SYSTEM_PROMPT. Default False = zero-shot.
    overlap_metric             : "ids" (Round-1) or "query_tokens" (Round-2). The
                                 D2 guard's similarity metric. Default "ids".
    use_gap_prompt             : if True, before each search after the first,
                                 issue a short LLM call asking the model to list
                                 facets missing from prior observations and
                                 inject the result into the next prompt's
                                 context. Default False.
    inject_step_budget         : if True, prepend a French step-budget line to
                                 every generate_step prompt. Default False.
    seed_search_k              : if > 0, run an initial first-stage retrieval
                                 (top-`seed_search_k`) before the agent's
                                 first generate_step and seed observed_ids /
                                 the scratchpad with an "Observation initiale"
                                 block. Default 0 (off).
    topup_threshold            : if > 0 and len(observed ∪ archive) <
                                 topup_threshold at finalize time, top up the
                                 pool with the first stage's top-`topup_k`
                                 before D1 re-ranks. Default 0 (off).
    topup_k                    : pool top-up depth (§16.4). Default 50.
    """

    def __init__(
        self,
        retriever,
        df_articles: pd.DataFrame,
        llm_client: OllamaClient,
        top_k_shown: int = 3,
        search_snippet_chars: int = 80,
        max_steps: int = 6,
        overlap_threshold: float = 0.7,
        fewshot_examples: Optional[dict] = None,
        max_article_tokens: int = 300,
        # ── Round-2 Block A — parsing layer ──────────────────────────────────
        max_tokens_generate: int = 128,
        use_function_calling: bool = False,
        use_action_regex_v2: bool = False,
        use_invalid_action_echo: bool = False,
        use_fewshot_trajectories: bool = False,
        # ── Round-2 Block B — reformulation thrash detection ─────────────────
        overlap_metric: str = "ids",
        use_gap_prompt: bool = False,
        inject_step_budget: bool = False,
        # ── Round-2 Block D — pool ceiling ───────────────────────────────────
        seed_search_k: int = 0,
        topup_threshold: int = 0,
        topup_k: int = 50,
    ):
        if overlap_metric not in {"ids", "query_tokens"}:
            raise ValueError(
                f"overlap_metric must be 'ids' or 'query_tokens', got {overlap_metric!r}"
            )
        self._retriever                = retriever
        self._llm                      = llm_client
        self.top_k_shown               = top_k_shown
        self.search_snippet_chars      = search_snippet_chars
        self.max_steps                 = max_steps
        self.overlap_threshold         = overlap_threshold
        self._fewshot_block            = ""  # zero-shot D1: fewshot adds ~600 tokens per call with no quality gain
        self._max_article_tokens       = max_article_tokens
        self._df_articles              = df_articles
        # Round-2 Block A
        self._max_tokens_generate      = max_tokens_generate
        self._use_function_calling     = use_function_calling
        self._use_action_regex_v2      = use_action_regex_v2
        self._use_invalid_action_echo  = use_invalid_action_echo
        self._use_fewshot_trajectories = use_fewshot_trajectories
        # Round-2 Block B
        self._overlap_metric           = overlap_metric
        self._use_gap_prompt           = use_gap_prompt
        self._inject_step_budget       = inject_step_budget
        # Round-2 Block D
        self._seed_search_k            = seed_search_k
        self._topup_threshold          = topup_threshold
        self._topup_k                  = topup_k

        self._id_to_text: dict[int, str] = dict(
            zip(df_articles["article_id"], df_articles["article_text"])
        )

        # Stats accumulators (reset across retrieve() calls)
        self._all_traces:           list[list[dict]] = []
        self._all_breakdowns:       list[dict]       = []
        self._all_pool_topup_flags: list[bool]       = []   # Round-2 Block D
        self._last_breakdown:       dict             = {}

        self._graph = self._build_graph()

    # ── Public interface ─────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 100) -> tuple[list[int], float]:
        """Run the ReAct loop and return (ranked_article_ids[:top_k], total_latency_ms)."""
        # Round-2 Block D — mandatory seed search before graph invoke. Pre-
        # populates observed_ids and seeds the scratchpad with an
        # "Observation initiale" block so the agent starts from the first
        # stage's top-K rather than having to discover the pool from scratch.
        seed_scratchpad     = ""
        seed_observed_ids:  list[int] = []
        seed_retrieval_lat: float     = 0.0
        if self._seed_search_k > 0:
            t_seed = time.perf_counter()
            seed_ids, _ = self._retriever.retrieve(query, top_k=self._seed_search_k)
            seed_retrieval_lat = (time.perf_counter() - t_seed) * 1000.0

            seed_observed_ids = [int(aid) for aid in seed_ids]
            id_text = self._id_to_text
            lines = ["Observation initiale :"]
            for i, aid in enumerate(seed_observed_ids, start=1):
                from retrieval.agentic.tools import _first_sentence_snippet
                txt = id_text.get(aid, "(texte non disponible)")
                snip = _first_sentence_snippet(txt, self.search_snippet_chars)
                lines.append(f"  [{i}] ID={aid} — {snip}")
            seed_scratchpad = "\n".join(lines)

        initial: ReActState = {
            "query":                query,
            "scratchpad":           seed_scratchpad,
            "scratchpad_tokens":    _count_tokens(seed_scratchpad),
            "step":                 0,
            "done":                 False,
            "tool_name":            None,
            "tool_args":            None,
            "last_raw_output":      "",
            "observed_ids":         seed_observed_ids,
            "observed_ids_archive": [],
            "prior_search_queries": [],
            "pending_gap_facets":   "",
            "final_article_ids":    [],
            "pool_topped_up":       False,
            "latency_breakdown_ms": {
                "llm_generate":   0.0,
                "retrieval":      seed_retrieval_lat,
                "rerank_posthoc": 0.0,
            },
            "agent_loop_trace":   [],
            "preflight_failures": 0,
            "overlap_guard_fires": 0,
        }

        t0     = time.perf_counter()
        result = self._graph.invoke(initial)
        total_ms = (time.perf_counter() - t0) * 1000.0

        self._last_breakdown = result["latency_breakdown_ms"]
        self._all_breakdowns.append(dict(result["latency_breakdown_ms"]))
        self._all_traces.append(result["agent_loop_trace"])
        self._all_pool_topup_flags.append(bool(result.get("pool_topped_up", False)))

        return result["final_article_ids"][:top_k], total_ms

    def get_pool_topup_flags(self) -> list[bool]:
        """Per-query pool_topped_up flags in processing order (Round-2 Block D)."""
        return list(self._all_pool_topup_flags)

    def get_all_traces(self) -> list[list[dict]]:
        """Return per-query loop traces in processing order (one list per query)."""
        return list(self._all_traces)

    def get_latency_breakdown(self) -> dict:
        """Return latency breakdown from the last retrieve() call."""
        return dict(self._last_breakdown)

    def get_latency_breakdown_mean(self) -> dict:
        """Return mean per-component latency (ms) across all retrieve() calls."""
        if not self._all_breakdowns:
            return {"llm_generate": 0.0, "retrieval": 0.0, "rerank_posthoc": 0.0}
        keys = ("llm_generate", "retrieval", "rerank_posthoc")
        return {
            k: round(float(np.mean([b.get(k, 0.0) for b in self._all_breakdowns])), 1)
            for k in keys
        }

    def get_loop_stats(self) -> dict:
        """Aggregate loop statistics across all retrieve() calls."""
        traces = self._all_traces
        if not traces:
            return {}

        n = len(traces)
        steps_per_query = [len(t) for t in traces]

        finished = sum(
            1 for t in traces
            if any(e.get("tool_name") == "finish" for e in t)
        )
        terminated_by_limit = sum(
            1 for t in traces
            if t
            and not any(e.get("tool_name") == "finish" for e in t)
            and len(t) >= self.max_steps
        )
        total_overlap   = sum(
            sum(1 for e in t if e.get("overlap_guard_fired", False)) for t in traces
        )
        total_preflight = sum(
            sum(1 for e in t if e.get("preflight_failed", False)) for t in traces
        )

        return {
            "n_queries":                            n,
            "mean_steps_per_query":                 float(np.mean(steps_per_query)),
            "fraction_queries_converged_finish":    finished / n,
            "fraction_queries_terminated_by_limit": terminated_by_limit / n,
            "total_overlap_guard_fires":            total_overlap,
            "total_preflight_failures":             total_preflight,
        }

    # ── Graph construction ────────────────────────────────────────────────────

    def _build_graph(self):
        from langgraph.graph import END, START, StateGraph

        # Capture configuration in closures (graph compiled once per instance)
        retriever                = self._retriever
        llm                      = self._llm
        df_articles              = self._df_articles
        id_to_text               = self._id_to_text
        top_k_shown              = self.top_k_shown
        search_snippet_chars     = self.search_snippet_chars
        max_steps                = self.max_steps
        overlap_threshold        = self.overlap_threshold
        fewshot_block            = self._fewshot_block
        max_article_tokens       = self._max_article_tokens
        # Round-2 Block A
        max_tokens_generate      = self._max_tokens_generate
        use_function_calling     = self._use_function_calling
        use_action_regex_v2      = self._use_action_regex_v2
        use_invalid_action_echo  = self._use_invalid_action_echo
        use_fewshot_trajectories = self._use_fewshot_trajectories
        # Round-2 Block B
        overlap_metric           = self._overlap_metric
        use_gap_prompt           = self._use_gap_prompt
        inject_step_budget       = self._inject_step_budget
        # Round-2 Block D
        topup_threshold          = self._topup_threshold
        topup_k                  = self._topup_k

        system_prompt = (
            REACT_FEWSHOT_TRAJECTORIES_BLOCK + "\n" + REACT_SYSTEM_PROMPT
            if use_fewshot_trajectories
            else REACT_SYSTEM_PROMPT
        )

        # ── Node: generate_step ──────────────────────────────────────────────
        def generate_step_node(state: ReActState) -> dict:
            # Round-2 Block B: optional preamble injected ahead of the user turn
            preamble_lines = []
            if inject_step_budget:
                step_idx     = int(state["step"])
                remaining    = max(max_steps - step_idx - 1, 0)
                preamble_lines.append(
                    f"Étape {step_idx + 1}/{max_steps} — il vous reste "
                    f"{remaining} étape(s) après celle-ci."
                )
            pending = state.get("pending_gap_facets") or ""
            if use_gap_prompt and pending.strip():
                preamble_lines.append("Facettes encore manquantes :\n" + pending.strip())
            preamble = ("\n\n".join(preamble_lines) + "\n\n") if preamble_lines else ""

            user_turn = preamble + REACT_USER_TEMPLATE.format(
                question=state["query"],
                scratchpad=state["scratchpad"],
            )

            t0 = time.perf_counter()
            tool_name: Optional[str] = None
            tool_args: Optional[dict] = None
            llm_output: str           = ""

            if use_function_calling:
                # Native Ollama /api/chat + tools — Llama 3.1 supports it.
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_turn},
                ]
                msg, _ = llm.chat_with_tools(
                    messages=messages,
                    tools=REACT_TOOL_SCHEMAS,
                    temperature=0.0,
                    max_tokens=max_tokens_generate,
                )
                llm_output = msg.get("content", "") or ""
                tool_calls = msg.get("tool_calls") or []
                if tool_calls:
                    fn   = tool_calls[0].get("function", {}) or {}
                    name = (fn.get("name") or "").lower()
                    args = fn.get("arguments") or {}
                    if isinstance(args, str):
                        try:
                            import json as _json
                            args = _json.loads(args)
                        except Exception:
                            args = {}
                    if name == "search":
                        tool_name, tool_args = "search", {"query": str(args.get("query", "")).strip()}
                    elif name == "lookup":
                        try:
                            tool_name, tool_args = "lookup", {"article_id": int(args.get("article_id"))}
                        except (ValueError, TypeError):
                            tool_name, tool_args = "lookup", {"article_id": None}
                    elif name == "finish":
                        tool_name, tool_args = "finish", {}

                # Fall back to regex on the textual content if no tool_calls.
                if tool_name is None and llm_output:
                    tool_name, tool_args = _parse_action(
                        llm_output, use_v2_fallback=use_action_regex_v2
                    )
            else:
                prompt = system_prompt + "\n\n" + user_turn
                llm_output, _ = llm.generate(
                    prompt, temperature=0.0, max_tokens=max_tokens_generate,
                )
                tool_name, tool_args = _parse_action(
                    llm_output, use_v2_fallback=use_action_regex_v2
                )

            llm_lat = (time.perf_counter() - t0) * 1000.0
            fragment = _extract_step_fragment(llm_output)

            scratchpad = state["scratchpad"]
            if fragment:
                scratchpad = scratchpad + ("\n" if scratchpad else "") + fragment

            breakdown = dict(state["latency_breakdown_ms"])
            breakdown["llm_generate"] = breakdown.get("llm_generate", 0.0) + llm_lat

            trace_entry = {
                "step":                state["step"],
                "tool_name":           tool_name,
                "tool_args":           tool_args,
                "llm_latency_ms":      round(llm_lat, 1),
                "ids_returned":        [],
                "preflight_failed":    False,
                "overlap_guard_fired": False,
            }

            return {
                "scratchpad":           scratchpad,
                "scratchpad_tokens":    _count_tokens(scratchpad),
                "tool_name":            tool_name,
                "tool_args":            tool_args,
                "last_raw_output":      llm_output,
                "pending_gap_facets":   "",   # consumed; clear for next turn
                "latency_breakdown_ms": breakdown,
                "agent_loop_trace":     state["agent_loop_trace"] + [trace_entry],
            }

        # ── Node: execute_tool ───────────────────────────────────────────────
        def execute_tool_node(state: ReActState) -> dict:
            tool_name            = state["tool_name"]
            tool_args            = state["tool_args"]
            scratchpad           = state["scratchpad"]
            observed_ids         = list(state["observed_ids"])
            observed_ids_archive = list(state["observed_ids_archive"])
            breakdown            = dict(state["latency_breakdown_ms"])
            preflight_failures   = state["preflight_failures"]
            overlap_guard_fires  = state["overlap_guard_fires"]

            trace       = list(state["agent_loop_trace"])
            last_entry  = dict(trace[-1]) if trace else {}

            # ── T1 Pre-flight gate ───────────────────────────────────────────
            ok, err_msg = validate_tool_call(tool_name, tool_args)
            if not ok:
                if use_invalid_action_echo:
                    malformed = _extract_malformed_action_line(state.get("last_raw_output", ""))
                    scratchpad += "\n" + REACT_INVALID_ACTION_OBSERVATION_TEMPLATE.format(
                        malformed_line=malformed
                    )
                else:
                    scratchpad += "\n" + REACT_INVALID_ACTION_OBSERVATION
                preflight_failures += 1
                last_entry["preflight_failed"] = True
                if trace:
                    trace[-1] = last_entry
                return {
                    "scratchpad":           scratchpad,
                    "scratchpad_tokens":    _count_tokens(scratchpad),
                    "step":                 state["step"] + 1,
                    "preflight_failures":   preflight_failures,
                    "agent_loop_trace":     trace,
                    "prior_search_queries": list(state.get("prior_search_queries") or []),
                    "pending_gap_facets":   state.get("pending_gap_facets") or "",
                }

            new_ids: list[int] = []
            prior_search_queries = list(state.get("prior_search_queries") or [])
            pending_gap_facets   = state.get("pending_gap_facets") or ""

            # ── Tool dispatch ────────────────────────────────────────────────
            if tool_name == "search":
                new_query = tool_args["query"]

                # D2 overlap guard — Round-2: query-token Jaccard against prior
                # search queries (catches "rephrase the same query" loops that
                # ID-Jaccard misses). Round-1 ID-Jaccard preserved when
                # overlap_metric == "ids".
                if overlap_metric == "query_tokens":
                    jaccard = (
                        max((_query_token_jaccard(new_query, q) for q in prior_search_queries),
                            default=0.0)
                        if prior_search_queries else 0.0
                    )
                    suppress = jaccard >= overlap_threshold

                    if suppress:
                        scratchpad += (
                            "\nObservation : Requête trop proche d'une recherche "
                            "précédente. Reformulez en ciblant un autre angle."
                        )
                        overlap_guard_fires += 1
                        last_entry["overlap_guard_fired"] = True
                        prior_search_queries.append(new_query)
                    else:
                        tool_result = execute_search(
                            query=new_query,
                            retriever=retriever,
                            df_articles=df_articles,
                            top_k_shown=top_k_shown,
                            snippet_chars=search_snippet_chars,
                        )
                        breakdown["retrieval"] = breakdown.get("retrieval", 0.0) + tool_result.latency_ms
                        scratchpad += tool_result.raw_text
                        observed_ids.extend(tool_result.article_ids)
                        new_ids = tool_result.article_ids
                        prior_search_queries.append(new_query)
                else:
                    # Round-1 path — ID-level Jaccard against all observed IDs
                    tool_result = execute_search(
                        query=new_query,
                        retriever=retriever,
                        df_articles=df_articles,
                        top_k_shown=top_k_shown,
                        snippet_chars=search_snippet_chars,
                    )
                    breakdown["retrieval"] = breakdown.get("retrieval", 0.0) + tool_result.latency_ms

                    all_seen    = set(observed_ids + observed_ids_archive)
                    new_ids_set = set(tool_result.article_ids)
                    if all_seen and new_ids_set:
                        jaccard = len(new_ids_set & all_seen) / len(new_ids_set | all_seen)
                    else:
                        jaccard = 0.0

                    if jaccard >= overlap_threshold:
                        scratchpad += (
                            "\nObservation : Résultats similaires déjà observés. "
                            "Essayez une reformulation différente."
                        )
                        overlap_guard_fires += 1
                        last_entry["overlap_guard_fired"] = True
                    else:
                        scratchpad += tool_result.raw_text
                        observed_ids.extend(tool_result.article_ids)
                        new_ids = tool_result.article_ids
                    prior_search_queries.append(new_query)

                # Gap-prompt sub-step (Round-2 Block B): fire AFTER each search
                # whose results were observed (skip if this was the first
                # search — there is nothing to compare against). Output is
                # stashed and consumed by the NEXT generate_step prompt.
                if use_gap_prompt and len(prior_search_queries) >= 2 and not last_entry["overlap_guard_fired"]:
                    obs_for_gap = scratchpad[-2000:]   # last 2k chars of scratchpad as context
                    gap_prompt  = REACT_GAP_PROMPT.format(
                        question=state["query"],
                        observations_so_far=obs_for_gap,
                    )
                    t_gap = time.perf_counter()
                    gap_text, _ = llm.generate(
                        gap_prompt, temperature=0.0, max_tokens=80,
                    )
                    breakdown["llm_generate"] = (
                        breakdown.get("llm_generate", 0.0)
                        + (time.perf_counter() - t_gap) * 1000.0
                    )
                    pending_gap_facets = gap_text.strip()

            elif tool_name == "lookup":
                tool_result = execute_lookup(
                    article_id=tool_args["article_id"],
                    df_articles=df_articles,
                )
                breakdown["retrieval"] = breakdown.get("retrieval", 0.0) + tool_result.latency_ms
                scratchpad  += tool_result.raw_text
                observed_ids.extend(tool_result.article_ids)
                new_ids = tool_result.article_ids

            elif tool_name == "finish":
                # Pure stopping signal — no observation text appended
                last_entry["ids_returned"] = []
                if trace:
                    trace[-1] = last_entry
                return {
                    "scratchpad":           scratchpad,
                    "scratchpad_tokens":    _count_tokens(scratchpad),
                    "step":                 state["step"] + 1,
                    "done":                 True,
                    "observed_ids":         observed_ids,
                    "observed_ids_archive": observed_ids_archive,
                    "prior_search_queries": prior_search_queries,
                    "pending_gap_facets":   pending_gap_facets,
                    "latency_breakdown_ms": breakdown,
                    "agent_loop_trace":     trace,
                    "overlap_guard_fires":  overlap_guard_fires,
                    "preflight_failures":   preflight_failures,
                }

            # ── C1 Sliding-window truncation ─────────────────────────────────
            scratchpad_tokens = _count_tokens(scratchpad)
            while scratchpad_tokens > MAX_SCRATCHPAD_TOKENS:
                scratchpad, observed_ids, archived = _truncate_scratchpad(
                    scratchpad, observed_ids
                )
                if not archived:
                    break
                observed_ids_archive.extend(archived)
                scratchpad_tokens = _count_tokens(scratchpad)

            last_entry["ids_returned"] = new_ids
            if trace:
                trace[-1] = last_entry

            return {
                "scratchpad":           scratchpad,
                "scratchpad_tokens":    scratchpad_tokens,
                "step":                 state["step"] + 1,
                "observed_ids":         observed_ids,
                "observed_ids_archive": observed_ids_archive,
                "prior_search_queries": prior_search_queries,
                "pending_gap_facets":   pending_gap_facets,
                "latency_breakdown_ms": breakdown,
                "agent_loop_trace":     trace,
                "overlap_guard_fires":  overlap_guard_fires,
                "preflight_failures":   preflight_failures,
            }

        # ── Node: finalize ───────────────────────────────────────────────────
        def finalize_node(state: ReActState) -> dict:
            # Union pool — deduplicated, order preserved
            all_ids = list(
                dict.fromkeys(state["observed_ids"] + state["observed_ids_archive"])
            )

            # Round-2 Block D — pool top-up fallback. Bounds the worst case
            # to ≥ first-stage's top-K recall when the agent's exploration
            # was too narrow (per §15.1.4, ~55 % of queries on the hybrid
            # backbone never observed a single gold ID).
            pool_topped_up = False
            breakdown = dict(state["latency_breakdown_ms"])
            if topup_threshold > 0 and len(all_ids) < topup_threshold:
                t_topup = time.perf_counter()
                topup_ids, _ = retriever.retrieve(state["query"], top_k=topup_k)
                breakdown["retrieval"] = (
                    breakdown.get("retrieval", 0.0)
                    + (time.perf_counter() - t_topup) * 1000.0
                )
                # Append, preserving the agent's order first then the first
                # stage's top-K — keeps any agent-discovered novel IDs in
                # front when D1 ranks ties.
                seen = set(all_ids)
                for aid in topup_ids:
                    aid = int(aid)
                    if aid not in seen:
                        all_ids.append(aid)
                        seen.add(aid)
                pool_topped_up = True

            if not all_ids:
                return {
                    "final_article_ids":    [],
                    "pool_topped_up":       pool_topped_up,
                    "latency_breakdown_ms": breakdown,
                }

            # D1 Post-hoc LLaMA re-ranking against ORIGINAL (immutable) query
            t0     = time.perf_counter()
            scored: list[tuple[int, float]] = []
            for aid in all_ids:
                text  = id_to_text.get(aid, "")
                words = text.split()
                if len(words) > max_article_tokens:
                    text = " ".join(words[:max_article_tokens]) + " …"

                prompt = LLM_JUDGE_BINARY_PROMPT.format(
                    fewshot_block=fewshot_block,
                    question=state["query"],
                    article_text_truncated=text,
                )
                response, _ = llm.generate(prompt, temperature=0.0, max_tokens=8)
                _, score    = parse_binary_judgment(response)
                scored.append((aid, score))

            rerank_lat = (time.perf_counter() - t0) * 1000.0
            breakdown["rerank_posthoc"] = (
                breakdown.get("rerank_posthoc", 0.0) + rerank_lat
            )

            # Sort descending by relevance score; stable sort preserves observation order on ties
            scored.sort(key=lambda x: x[1], reverse=True)
            final_ids = [aid for aid, _ in scored]

            return {
                "final_article_ids":    final_ids,
                "pool_topped_up":       pool_topped_up,
                "latency_breakdown_ms": breakdown,
            }

        # ── Routing ──────────────────────────────────────────────────────────
        def route_after_execute(state: ReActState) -> str:
            if state["done"] or state["step"] >= max_steps:
                return "finalize"
            return "generate_step"

        # ── Wire the graph ────────────────────────────────────────────────────
        builder = StateGraph(ReActState)
        builder.add_node("generate_step", generate_step_node)
        builder.add_node("execute_tool",  execute_tool_node)
        builder.add_node("finalize",      finalize_node)

        builder.add_edge(START,             "generate_step")
        builder.add_edge("generate_step",   "execute_tool")
        builder.add_conditional_edges(
            "execute_tool",
            route_after_execute,
            {
                "finalize":      "finalize",
                "generate_step": "generate_step",
            },
        )
        builder.add_edge("finalize", END)

        return builder.compile()
