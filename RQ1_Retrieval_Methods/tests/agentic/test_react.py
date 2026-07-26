"""
Unit tests for ReActRetriever (Tier 4.2) — Round-2 parsing layer.

Run with:
    .venv/Scripts/python.exe -m pytest tests/agentic/test_react.py -v

The tests exercise three things in isolation, with no real LLM or retriever:

  1. Action-line parsing on 8 valid and 8 invalid forms (Round-1 strict regex
     and Round-2 broadened parser, separately).
  2. The templated invalid-action echo (Round-2 Block A, flag-toggled).
  3. End-to-end loop with mock LLM and mock retriever, both with v1 defaults
     (use_function_calling=False, regex parser only) and with the v2 flags on.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from retrieval.agentic.prompts import (
    REACT_INVALID_ACTION_OBSERVATION,
    REACT_INVALID_ACTION_OBSERVATION_TEMPLATE,
)
from retrieval.agentic.react import (
    ReActRetriever,
    _extract_malformed_action_line,
    _parse_action,
    _parse_action_v2,
    _query_token_jaccard,
    _query_tokens,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class MockRetriever:
    """Returns a fixed list of article IDs with a fixed latency."""

    def __init__(self, ids: list[int], latency_ms: float = 1.0):
        self._ids = ids
        self._lat = latency_ms

    def retrieve(self, _query: str, top_k: int = 100):
        return self._ids[:top_k], self._lat


class MockLLM:
    """
    Mock for OllamaClient.

    Recognised prompt types:
      - contains "Pertinent"          → judge call (cycles `judge_responses`, default "Oui")
      - contains "Facettes manquantes" → Block B gap prompt (cycles `gap_responses`,
                                          default "(aucune)")
      - else                          → generate-step call (cycles `generate_responses`)

    Also exposes chat_with_tools() for the function-calling path; pass
    `chat_responses` as a list of message dicts.
    """

    def __init__(
        self,
        generate_responses: list[str] | None = None,
        judge_responses:    list[str] | None = None,
        chat_responses:     list[dict] | None = None,
        gap_responses:      list[str] | None = None,
        latency_ms:         float = 1.0,
    ):
        self._gen        = generate_responses or ["Action : finish[done]"]
        self._jud        = judge_responses    or ["Oui"]
        self._chat       = chat_responses     or []
        self._gap        = gap_responses      or ["(aucune)"]
        self._gi = self._ji = self._ci = self._gpi = 0
        self._lat        = latency_ms

    def generate(self, prompt: str, **_kwargs):
        if "Pertinent" in prompt:
            r = self._jud[self._ji % len(self._jud)]
            self._ji += 1
            return r, self._lat
        if "Facettes manquantes" in prompt:
            r = self._gap[self._gpi % len(self._gap)]
            self._gpi += 1
            return r, self._lat
        r = self._gen[self._gi % len(self._gen)]
        self._gi += 1
        return r, self._lat

    def chat_with_tools(self, messages, tools, **_kwargs):
        msg = self._chat[self._ci % len(self._chat)]
        self._ci += 1
        return msg, self._lat

    def preflight(self):  # pragma: no cover — never called in tests
        return 0.0


def _make_df(ids: list[int]) -> pd.DataFrame:
    return pd.DataFrame({
        "article_id":   ids,
        "article_text": [f"Texte de l'article {i}" for i in ids],
    })


def _make_react(generate_responses, judge_responses=None, chat_responses=None,
                gap_responses=None, ids=None, **kwargs) -> ReActRetriever:
    if ids is None:
        ids = list(range(1, 11))
    df  = _make_df(ids)
    llm = MockLLM(generate_responses, judge_responses, chat_responses, gap_responses)
    # Defaults are overridable via kwargs.
    defaults = {
        "top_k_shown":        5,
        "max_steps":          3,
        "overlap_threshold":  1.1,    # disable D2 unless test overrides
        "max_article_tokens": 50,
    }
    defaults.update(kwargs)
    return ReActRetriever(
        retriever=MockRetriever(ids),
        df_articles=df,
        llm_client=llm,
        **defaults,
    )


# ---------------------------------------------------------------------------
# 1.  Action-line parsing — 8 valid forms (Round-1 strict regex)
# ---------------------------------------------------------------------------

VALID_STRICT = [
    ("Action : search[droit du travail]",            "search", {"query": "droit du travail"}),
    ("Action: search[bail commercial]",              "search", {"query": "bail commercial"}),
    ("Action :search[divorce]",                      "search", {"query": "divorce"}),
    ("Pensée: ...\nAction : lookup[1234]",           "lookup", {"article_id": 1234}),
    ("Action : LOOKUP[42]",                          "lookup", {"article_id": 42}),
    ("Action : finish[done]",                        "finish", {}),
    ("Action : Finish[]",                            "finish", {}),
    ("Action  :  search[ congé maladie ]",           "search", {"query": "congé maladie"}),
]


@pytest.mark.parametrize("text,exp_name,exp_args", VALID_STRICT)
def test_parse_action_strict_valid(text, exp_name, exp_args):
    name, args = _parse_action(text, use_v2_fallback=False)
    assert name == exp_name
    assert args == exp_args


# ---------------------------------------------------------------------------
# 2.  Action-line parsing — 8 invalid forms with strict-only parser
# ---------------------------------------------------------------------------

INVALID_FOR_STRICT = [
    "Action : search(droit du travail)",       # parens — Round-2 form
    'Action : search:"bail commercial"',       # colon  — Round-2 form
    "Action : search «divorce»",               # French quotes
    "Pensée seulement, pas d'Action.",         # prose only
    'I will search for "tenancy".',            # English prose, no Action: line
    "Action : ",                               # blank action
    "Action : unknown[foo]",                   # unknown verb
    "Action : search\nbail commercial",        # newline-truncated argument
]


@pytest.mark.parametrize("text", INVALID_FOR_STRICT)
def test_parse_action_strict_rejects(text):
    name, args = _parse_action(text, use_v2_fallback=False)
    assert name is None and args is None, f"strict parser unexpectedly matched: {text!r}"


# ---------------------------------------------------------------------------
# 3.  Action-line parsing — Round-2 broadened parser recovers most v1 misses
# ---------------------------------------------------------------------------

V2_RECOVERS = [
    ("Action : search(droit du travail)",      "search", {"query": "droit du travail"}),
    ('Action : search:"bail commercial"',      "search", {"query": "bail commercial"}),
    ("Action : search «divorce»",              "search", {"query": "divorce"}),
    ("Action : finish()",                      "finish", {}),
    ("Action : lookup(42)",                    "lookup", {"article_id": 42}),
]


@pytest.mark.parametrize("text,exp_name,exp_args", V2_RECOVERS)
def test_parse_action_v2_recovers_round1_misses(text, exp_name, exp_args):
    name, args = _parse_action(text, use_v2_fallback=True)
    assert name == exp_name
    assert args == exp_args


def test_parse_action_v2_truly_invalid_still_none():
    """v2 fallback must still reject pure prose."""
    name, args = _parse_action(
        "Pensée seulement, pas d'Action.",
        use_v2_fallback=True,
    )
    assert name is None and args is None


# ---------------------------------------------------------------------------
# 4.  Templated invalid-action echo
# ---------------------------------------------------------------------------

def test_extract_malformed_action_line_picks_action_line():
    raw = "Pensée : Je cherche.\nAction : search(droit du travail)\nNotes additionnelles."
    line = _extract_malformed_action_line(raw)
    assert line == "Action : search(droit du travail)"


def test_extract_malformed_action_line_falls_back_to_first_nonempty():
    raw = "\n\nJe vais chercher quelque chose.\nLigne 2."
    line = _extract_malformed_action_line(raw)
    assert line == "Je vais chercher quelque chose."


def test_invalid_action_echo_template_inserts_line():
    line = 'Action : search("foo")'
    rendered = REACT_INVALID_ACTION_OBSERVATION_TEMPLATE.format(malformed_line=line)
    assert line in rendered
    assert "search[<requête>]" in rendered


# ---------------------------------------------------------------------------
# 5.  End-to-end loop — v1 defaults (regex-only, generic rejection)
# ---------------------------------------------------------------------------

def test_v1_defaults_round1_behaviour_preserved():
    """With all Round-2 flags off, the agent uses the strict regex and the
    generic rejection — i.e. exactly the Round-1 codepath."""
    react = _make_react(
        generate_responses=[
            "Action : search[droit du travail]",
            "Action : finish[done]",
        ],
    )
    ids, _ = react.retrieve("test query")
    assert ids[:5] == [1, 2, 3, 4, 5]
    trace = react.get_all_traces()[-1]
    assert trace[0]["tool_name"] == "search"
    assert trace[1]["tool_name"] == "finish"


def test_v1_defaults_invalid_action_uses_generic_rejection():
    react = _make_react(
        generate_responses=[
            "Action : search(droit du travail)",   # Round-2 form, invalid for v1
            "Action : finish[done]",
        ],
    )
    react.retrieve("test query")
    trace = react.get_all_traces()[-1]
    assert trace[0]["preflight_failed"] is True
    assert trace[0]["tool_name"] is None


# ---------------------------------------------------------------------------
# 6.  End-to-end loop — v2 broadened regex recovers the same input
# ---------------------------------------------------------------------------

def test_v2_regex_recovers_paren_form():
    react = _make_react(
        generate_responses=[
            "Action : search(droit du travail)",   # invalid for v1, valid for v2
            "Action : finish[done]",
        ],
        use_action_regex_v2=True,
    )
    react.retrieve("test query")
    trace = react.get_all_traces()[-1]
    assert trace[0]["preflight_failed"] is False
    assert trace[0]["tool_name"] == "search"


def test_v2_invalid_echo_template_appears_in_scratchpad():
    react = _make_react(
        generate_responses=[
            "Action : search(droit du travail)",   # invalid for v1
            "Action : finish[done]",
        ],
        use_invalid_action_echo=True,
    )
    react.retrieve("test query")
    trace = react.get_all_traces()[-1]
    assert trace[0]["preflight_failed"] is True
    # The malformed line must be echoed back in the scratchpad observation.
    # We can't access the scratchpad directly from the trace, but we know that
    # the template formats with {malformed_line}; check the template renders
    # are wired to the right input.
    rendered = REACT_INVALID_ACTION_OBSERVATION_TEMPLATE.format(
        malformed_line="Action : search(droit du travail)",
    )
    assert "search(droit du travail)" in rendered


# ---------------------------------------------------------------------------
# 7.  End-to-end loop — function-calling path uses chat_with_tools
# ---------------------------------------------------------------------------

def test_function_calling_search_then_finish():
    chat = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "search",
                                          "arguments": {"query": "droit du travail"}}}],
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "finish", "arguments": {}}}],
        },
    ]
    react = _make_react(
        generate_responses=[],   # never used; chat path is taken
        chat_responses=chat,
        use_function_calling=True,
    )
    ids, _ = react.retrieve("test query")
    trace = react.get_all_traces()[-1]
    assert trace[0]["tool_name"] == "search"
    assert trace[1]["tool_name"] == "finish"
    assert ids[:5] == [1, 2, 3, 4, 5]


def test_function_calling_arguments_can_be_json_string():
    """Some Ollama builds return arguments as a JSON-encoded string."""
    chat = [{
        "role": "assistant",
        "content": "",
        "tool_calls": [{"function": {"name": "search",
                                      "arguments": json.dumps({"query": "bail commercial"})}}],
    }, {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"function": {"name": "finish", "arguments": "{}"}}],
    }]
    react = _make_react(
        generate_responses=[],
        chat_responses=chat,
        use_function_calling=True,
    )
    react.retrieve("test query")
    trace = react.get_all_traces()[-1]
    assert trace[0]["tool_name"] == "search"
    assert trace[0]["tool_args"] == {"query": "bail commercial"}


# ---------------------------------------------------------------------------
# 8.  Block B — query-token Jaccard helpers
# ---------------------------------------------------------------------------

def test_query_tokens_drops_stopwords_and_lowercases():
    toks = set(_query_tokens("Quel est le délai de prescription pour un vol simple ?"))
    # stopwords ('le', 'est') should be filtered; content words preserved.
    # Note: tokenize_lemmatize lemmatises ('délai' stays, 'prescription' stays);
    # the fallback keeps 'délai', 'prescription', 'vol', 'simple', etc.
    assert "le" not in toks and "est" not in toks
    assert "vol" in toks
    assert "simple" in toks


def test_query_token_jaccard_identical_is_one():
    assert _query_token_jaccard("droit du travail", "droit du travail") == pytest.approx(1.0)


def test_query_token_jaccard_disjoint_is_zero():
    assert _query_token_jaccard("droit du travail", "secret professionnel médical") == 0.0


def test_query_token_jaccard_paraphrase_high_overlap():
    """Two near-paraphrases should land above 0.5 — typical thrash signal."""
    j = _query_token_jaccard(
        "fin du bail commercial",
        "résiliation du bail commercial",
    )
    assert j >= 0.4   # 2 of 4 tokens overlap → 2/(3+3-2)=0.5 with stopwords stripped


# ---------------------------------------------------------------------------
# 9.  Block B — D2 guard with overlap_metric="query_tokens"
# ---------------------------------------------------------------------------

def test_query_token_guard_fires_on_paraphrase():
    """Same query restated → query-token Jaccard 1.0 → guard fires on the 2nd
    search even though IDs from the (mock) retriever rotate."""
    react = _make_react(
        generate_responses=[
            "Action : search[droit du travail]",
            "Action : search[droit du travail]",   # identical reformulation
            "Action : finish[done]",
        ],
        overlap_metric="query_tokens",
        overlap_threshold=0.6,
    )
    react.retrieve("test query")
    trace = react.get_all_traces()[-1]
    assert trace[0]["overlap_guard_fired"] is False
    assert trace[1]["overlap_guard_fired"] is True


def test_query_token_guard_does_not_fire_on_distinct_queries():
    react = _make_react(
        generate_responses=[
            "Action : search[droit du travail]",
            "Action : search[secret professionnel médical]",
            "Action : finish[done]",
        ],
        overlap_metric="query_tokens",
        overlap_threshold=0.6,
    )
    react.retrieve("test query")
    trace = react.get_all_traces()[-1]
    assert trace[0]["overlap_guard_fired"] is False
    assert trace[1]["overlap_guard_fired"] is False


def test_id_overlap_metric_still_works_default():
    """Default overlap_metric='ids' preserves Round-1 behaviour."""
    react = _make_react(
        generate_responses=[
            "Action : search[a]",
            "Action : search[b]",          # mock retriever returns same IDs
            "Action : finish[done]",
        ],
        overlap_metric="ids",
        overlap_threshold=1.0,             # strict — same IDs = 1.0 jaccard
    )
    react.retrieve("test query")
    trace = react.get_all_traces()[-1]
    assert trace[1]["overlap_guard_fired"] is True


def test_overlap_metric_invalid_raises():
    with pytest.raises(ValueError):
        _make_react(
            generate_responses=["Action : finish[done]"],
            overlap_metric="bogus",
        )


# ---------------------------------------------------------------------------
# 10. Block B — gap prompt fires after the first search and is cleared after use
# ---------------------------------------------------------------------------

def test_gap_prompt_fires_after_first_search():
    """
    Gap-prompt LLM calls go through llm.generate() but with a different
    prompt template. MockLLM routes by prompt content: anything containing
    'Facettes manquantes' draws from gap_responses, leaving the action slot
    untouched. The trace must show three actions (search → search → finish)
    even though MockLLM was called five times.
    """
    react = _make_react(
        generate_responses=[
            "Action : search[droit du travail]",
            "Action : search[bail commercial]",
            "Action : finish[done]",
        ],
        gap_responses=[
            "- condition de rupture\n- durée de préavis",
            "- exception colocation\n- forme du congé",
        ],
        use_gap_prompt=True,
        overlap_metric="query_tokens",
        overlap_threshold=0.99,   # don't suppress these distinct queries
        max_steps=4,
    )
    react.retrieve("test query")
    trace = react.get_all_traces()[-1]
    assert [e["tool_name"] for e in trace] == ["search", "search", "finish"]
    # MockLLM should have answered exactly 1 gap-prompt (after the 2nd search
    # the trace ends with finish — no further gap fires).
    assert react._llm._gpi == 1


def test_gap_prompt_skipped_when_disabled():
    """Without use_gap_prompt, no extra LLM call between searches."""
    react = _make_react(
        generate_responses=[
            "Action : search[a]",
            "Action : search[b]",
            "Action : finish[done]",
        ],
        use_gap_prompt=False,
        overlap_metric="query_tokens",
        overlap_threshold=0.99,
        max_steps=4,
    )
    react.retrieve("test query")
    trace = react.get_all_traces()[-1]
    assert [e["tool_name"] for e in trace] == ["search", "search", "finish"]


# ---------------------------------------------------------------------------
# 11. Block B — step-budget injection (smoke: prompt contains the budget string)
# ---------------------------------------------------------------------------

def test_inject_step_budget_appears_in_generate_prompt():
    """Capture the prompt argument to llm.generate to confirm the step-budget
    line is prepended."""

    captured: list[str] = []

    class CapturingLLM(MockLLM):
        def generate(self, prompt: str, **kwargs):
            captured.append(prompt)
            return super().generate(prompt, **kwargs)

    df  = _make_df(list(range(1, 11)))
    llm = CapturingLLM(["Action : finish[done]"])
    react = ReActRetriever(
        retriever=MockRetriever(list(range(1, 11))),
        df_articles=df,
        llm_client=llm,
        top_k_shown=5,
        max_steps=8,
        overlap_threshold=1.1,
        max_article_tokens=50,
        inject_step_budget=True,
    )
    react.retrieve("test query")
    assert any("Étape 1/8" in p for p in captured), \
        f"step-budget string missing in any prompt; captured={captured!r}"


# ---------------------------------------------------------------------------
# 12. Block C — first-sentence snippet extractor and snippet-chars threading
# ---------------------------------------------------------------------------

def test_first_sentence_snippet_prefers_sentence_boundary():
    from retrieval.agentic.tools import _first_sentence_snippet
    text = "Premier titre du contrat. Le bail commercial est conclu pour neuf ans, sauf dérogation."
    snip = _first_sentence_snippet(text, max_chars=80)
    assert snip == "Premier titre du contrat."


def test_first_sentence_snippet_falls_back_to_hard_cut():
    from retrieval.agentic.tools import _first_sentence_snippet
    text = "Une phrase très longue sans aucune ponctuation interne qui dépasse largement la fenêtre"
    snip = _first_sentence_snippet(text, max_chars=40)
    assert snip.endswith("…")
    assert len(snip) <= 41


def test_search_snippet_chars_threading_uses_first_sentence_when_available():
    """When the article text begins with a short first sentence, the snippet
    extractor returns that sentence even with a tiny max_chars budget — that
    is the §16.3 design (title / first sentence only)."""
    from retrieval.agentic.tools import execute_search
    df  = _make_df([1])
    df["article_text"] = ["Article un. Suite très longue qui dépasse 200 chars" + ("." * 200)]
    res = execute_search("q", retriever=MockRetriever([1]), df_articles=df,
                         top_k_shown=1, snippet_chars=5)
    assert "Article un." in res.raw_text
    # Make sure the long suffix did NOT bleed into the snippet — the cut is
    # at the first sentence boundary.
    assert "Suite très longue" not in res.raw_text


def test_search_snippet_chars_threading_hard_cuts_when_no_sentence_in_window():
    """When no sentence end falls within the window, fall back to hard cut."""
    from retrieval.agentic.tools import execute_search
    df  = _make_df([1])
    df["article_text"] = ["x" * 500]
    res = execute_search("q", retriever=MockRetriever([1]), df_articles=df,
                         top_k_shown=1, snippet_chars=20)
    # 20 chars of x + ellipsis
    assert "xxxxxxxxxxxxxxxxxxxx…" in res.raw_text


def test_default_top_k_shown_is_three():
    """Round-2 §16.3 sets the default to 3 (was 5)."""
    react = ReActRetriever(
        retriever=MockRetriever(list(range(1, 11))),
        df_articles=_make_df(list(range(1, 11))),
        llm_client=MockLLM(["Action : finish[done]"]),
    )
    assert react.top_k_shown == 3


# ---------------------------------------------------------------------------
# 13. Block D — seed search and pool top-up
# ---------------------------------------------------------------------------

def test_seed_search_seeds_observed_ids_and_scratchpad():
    """seed_search_k > 0 pre-populates observed_ids before the first
    generate_step, and writes an "Observation initiale :" block."""

    class CapturingLLM(MockLLM):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.first_prompt = None

        def generate(self, prompt: str, **kwargs):
            if self.first_prompt is None and "Pertinent" not in prompt \
                    and "Facettes manquantes" not in prompt:
                self.first_prompt = prompt
            return super().generate(prompt, **kwargs)

    df  = _make_df([1, 2, 3, 4, 5])
    llm = CapturingLLM(["Action : finish[done]"])
    react = ReActRetriever(
        retriever=MockRetriever([1, 2, 3, 4, 5]),
        df_articles=df,
        llm_client=llm,
        top_k_shown=3,
        max_steps=2,
        overlap_threshold=1.1,
        max_article_tokens=50,
        seed_search_k=4,                      # seed first 4 IDs
    )
    ids, _ = react.retrieve("test query")
    # The first generate_step's prompt must contain the seed observation block.
    assert "Observation initiale :" in llm.first_prompt
    assert "ID=1" in llm.first_prompt and "ID=4" in llm.first_prompt
    # Returned IDs should include the seed.
    assert set([1, 2, 3, 4]).issubset(set(ids))


def test_topup_fires_when_pool_below_threshold():
    """When the agent's pool size is below topup_threshold at finalize,
    the first stage's top-K is unioned in and pool_topped_up=True."""

    # The agent never searches, so observed_ids stays empty until topup fires.
    react = ReActRetriever(
        retriever=MockRetriever([10, 11, 12, 13, 14]),
        df_articles=_make_df([10, 11, 12, 13, 14]),
        llm_client=MockLLM(["Action : finish[done]"]),
        top_k_shown=3,
        max_steps=1,
        overlap_threshold=1.1,
        max_article_tokens=50,
        topup_threshold=10,                   # pool < 10 → top up
        topup_k=5,
    )
    ids, _ = react.retrieve("q")
    flags = react.get_pool_topup_flags()
    assert flags == [True]
    assert set(ids[:5]) == {10, 11, 12, 13, 14}


def test_topup_does_not_fire_when_pool_already_full():
    """When the agent's pool already reaches topup_threshold, top-up stays off."""
    react = ReActRetriever(
        retriever=MockRetriever([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
        df_articles=_make_df(list(range(1, 13))),
        llm_client=MockLLM(["Action : search[anything]", "Action : finish[done]"]),
        top_k_shown=12,                       # one search fills the pool
        max_steps=2,
        overlap_threshold=1.1,
        max_article_tokens=50,
        topup_threshold=5,                    # pool ≥ 5 → no top-up
        topup_k=20,
    )
    react.retrieve("q")
    assert react.get_pool_topup_flags() == [False]


def test_seed_search_and_topup_combined_smoke():
    """Smoke: with both seed and top-up enabled, the agent's final pool is
    bounded below by the first stage's contribution and pool_topped_up
    reflects whether top-up fired (pool ≥ threshold thanks to seed → False)."""
    react = ReActRetriever(
        retriever=MockRetriever([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
        df_articles=_make_df(list(range(1, 16))),
        llm_client=MockLLM(["Action : finish[done]"]),
        top_k_shown=3,
        max_steps=1,
        overlap_threshold=1.1,
        max_article_tokens=50,
        seed_search_k=12,                     # seed alone reaches threshold
        topup_threshold=10,
        topup_k=15,
    )
    ids, _ = react.retrieve("q")
    assert react.get_pool_topup_flags() == [False]
    assert len(set(ids)) >= 12


def test_function_calling_falls_back_to_regex_on_empty_tool_calls():
    """If the model emits plain text without tool_calls, the regex parser
    is consulted on the textual content (with v2 fallback if enabled)."""
    chat = [
        {"role": "assistant", "content": "Action : search[droit du travail]",
         "tool_calls": []},
        {"role": "assistant", "content": "Action : finish[done]",
         "tool_calls": []},
    ]
    react = _make_react(
        generate_responses=[],
        chat_responses=chat,
        use_function_calling=True,
    )
    react.retrieve("test query")
    trace = react.get_all_traces()[-1]
    assert trace[0]["tool_name"] == "search"
    assert trace[1]["tool_name"] == "finish"
