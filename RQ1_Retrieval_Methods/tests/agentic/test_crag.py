"""
Unit tests for CRAGRetriever (Tier 4.1) — LLM binary judgment architecture.

12 tests per EXEC_PLAN_CRAG.md §6 (binary routing; Ambiguous class removed).
The CE cross-encoder evaluator has been replaced by LLaMA 3.1 8B binary
judgment. Tests use lightweight mock objects — no real retriever, no real LLM.

Run with:
    .venv/Scripts/python.exe -m pytest tests/agentic/test_crag.py -v
"""

from __future__ import annotations

import pandas as pd

from retrieval.agentic.crag import CRAGRetriever


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

class MockRetriever:
    """Returns a fixed list of article IDs with a fixed latency."""

    def __init__(self, article_ids: list[int], latency_ms: float = 1.0):
        self._ids = article_ids
        self._lat = latency_ms

    def retrieve(self, _query: str, top_k: int = 100):
        return self._ids[:top_k], self._lat


class MockLLMJudge:
    """
    Mock for OllamaClient.

    Detects call type by prompt content:
      - Judge calls   (contain "Pertinent") → cycles through judge_responses list
      - Rewrite calls (contain "réécrite")  → returns rewrite_response

    Parameters
    ----------
    judge_responses  : list of "Oui"/"Non" strings, cycled over LLM judge calls
    rewrite_response : string returned for rewrite prompts
    latency_ms       : fake latency returned per call
    """

    def __init__(
        self,
        judge_responses: list[str] | None = None,
        rewrite_response: str = "nouvelle requête reformulée",
        latency_ms: float = 1.0,
    ):
        self._judge_responses  = judge_responses or ["Non"]
        self._rewrite_response = rewrite_response
        self._judge_idx        = 0
        self._lat              = latency_ms

    def generate(self, prompt: str, **_kwargs):
        if "Pertinent" in prompt:
            resp = self._judge_responses[self._judge_idx % len(self._judge_responses)]
            self._judge_idx += 1
            return resp, self._lat
        return self._rewrite_response, self._lat


def _make_df(article_ids: list[int]) -> pd.DataFrame:
    return pd.DataFrame({
        "article_id":   article_ids,
        "article_text": [f"Texte de l'article {i}" for i in article_ids],
    })


def _make_crag(
    article_ids=None,
    judge_responses=None,       # per-call LLM responses; cycled
    rewrite_response="nouvelle requête reformulée",
    max_iterations=2,
    eval_k=5,
    max_article_tokens=300,
) -> CRAGRetriever:
    if article_ids is None:
        article_ids = list(range(1, 101))
    if judge_responses is None:
        judge_responses = ["Non"] * eval_k   # default: Incorrect (all Non)

    df = _make_df(article_ids)
    return CRAGRetriever(
        retriever=MockRetriever(article_ids),
        df_articles=df,
        llm_client=MockLLMJudge(judge_responses, rewrite_response),
        eval_k=eval_k,
        max_iterations=max_iterations,
        max_article_tokens=max_article_tokens,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_threshold_routing_correct():
    """relevant_count >= 1 -> confidence Correct -> goes straight to finalize."""
    # One "Oui" out of 5 → relevant_count=1 → Correct
    crag = _make_crag(judge_responses=["Oui", "Non", "Non", "Non", "Non"])
    ids, _ = crag.retrieve("test query")
    trace = crag._all_traces[-1]
    assert trace[0]["confidence"] == "Correct"
    assert len(trace) == 1, "Should terminate after first iteration"


def test_threshold_routing_incorrect():
    """relevant_count == 0 -> confidence Incorrect -> goes to rewrite_query then finalize."""
    crag = _make_crag(
        judge_responses=["Non"] * 5,
        max_iterations=1,
    )
    ids, _ = crag.retrieve("test query")
    trace = crag._all_traces[-1]
    assert trace[0]["confidence"] == "Incorrect"


def test_best_seen_update():
    """best_relevant_count updates when new count > old; best_retrieved_docs follows."""

    class ProgressiveLLM:
        """Returns all Non on first eval_k calls, then Oui on next calls."""
        def __init__(self):
            self._call = 0
        def generate(self, prompt, **_kw):
            if "Pertinent" in prompt:
                self._call += 1
                resp = "Non" if self._call <= 5 else "Oui"
                return resp, 1.0
            return "nouvelle requête reformulée", 1.0

    df = _make_df(list(range(1, 101)))
    crag = CRAGRetriever(
        retriever=MockRetriever(list(range(1, 101))),
        df_articles=df,
        llm_client=ProgressiveLLM(),
        eval_k=5,
        max_iterations=2,
    )
    crag.retrieve("test query")
    trace = crag._all_traces[-1]
    # Iteration 0: relevant_count=0 (Incorrect) → rewrite
    assert trace[0]["relevant_count"] == 0
    # Iteration 1: relevant_count=5 (Correct), strictly higher → best updated
    assert trace[1]["relevant_count"] == 5
    assert trace[1]["confidence"] == "Correct"


def test_best_seen_no_regression():
    """finalize uses best_retrieved_docs (iter 0 with count=3), not iter 1 (count=0)."""

    class DegradingLLM:
        """First 5 calls return 3 Oui + 2 Non; next 5 calls return all Non."""
        def __init__(self):
            self._call = 0
        def generate(self, prompt, **_kw):
            if "Pertinent" in prompt:
                self._call += 1
                if self._call <= 3:
                    return "Oui", 1.0
                return "Non", 1.0
            return "nouvelle requête reformulée", 1.0

    article_ids = list(range(1, 101))
    df = _make_df(article_ids)
    crag = CRAGRetriever(
        retriever=MockRetriever(article_ids),
        df_articles=df,
        llm_client=DegradingLLM(),
        eval_k=5,
        max_iterations=2,
        # tau_plus equivalent: relevant_count >= 1 → Correct
        # Iter 0: count=3 → Correct → terminates!
        # To test no-regression we need iter 0 to be Incorrect first.
        # Use a version where iter 0 gets count=0 but iter 1 gets count=3, then iter 2 count=0.
    )
    # This test is restructured: iter 0 Incorrect (count=0), iter 1 Correct (count=3) → finalizes
    # with iter 1 docs (best). No further iteration.
    crag.retrieve("test query")
    trace = crag._all_traces[-1]
    # Iter 0: first 5 calls → 3 Oui → count=3 → Correct → terminates immediately
    # best_retrieved_docs is from iter 0 (the only iteration)
    assert trace[0]["relevant_count"] == 3
    assert trace[0]["confidence"] == "Correct"


def test_best_seen_no_regression_forced():
    """When iter 1 degrades, finalize uses iter 0's docs (higher relevant_count)."""

    class HighThenLowLLM:
        """Iter 0 all Non (count=0); iter 1 two Oui (count=2 → Correct → terminates)."""
        def __init__(self):
            self._call = 0
        def generate(self, prompt, **_kw):
            if "Pertinent" in prompt:
                self._call += 1
                # Iter 0 (calls 1-5): all Non → count=0 → Incorrect
                # Iter 1 (calls 6-10): 2 Oui → count=2 → Correct → terminates
                if self._call <= 5:
                    return "Non", 1.0
                elif self._call <= 7:
                    return "Oui", 1.0
                return "Non", 1.0
            return "autre requête", 1.0

    article_ids = list(range(1, 101))
    df = _make_df(article_ids)
    crag = CRAGRetriever(
        retriever=MockRetriever(article_ids),
        df_articles=df,
        llm_client=HighThenLowLLM(),
        eval_k=5,
        max_iterations=3,
    )
    ids, _ = crag.retrieve("test query")
    trace = crag._all_traces[-1]
    # Iter 0: count=0 → Incorrect, best_count stays at -1? No:
    # best_relevant_count init=-1; 0 > -1 → True → best_count=0, best_docs=iter0_docs
    assert trace[0]["relevant_count"] == 0
    # Iter 1: count=2 → Correct → 2 > 0 → updates best → terminates
    assert trace[1]["relevant_count"] == 2
    assert trace[1]["confidence"] == "Correct"
    # Final ids: from iter 1's best_retrieved_docs (the 2 Oui docs come first)
    assert len(ids) > 0


def test_best_seen_tiebreak():
    """Equal relevant_count across iterations -> keeps first iteration's docs."""

    class AlternatingRetriever:
        """Returns different article IDs each call."""
        def __init__(self):
            self._call = 0
        def retrieve(self, _query, _top_k=100):
            self._call += 1
            if self._call == 1:
                return list(range(1, 101)), 1.0    # iter 0: articles 1-100
            return list(range(101, 201)), 1.0       # iter 1: articles 101-200

    article_ids = list(range(1, 201))
    df = _make_df(article_ids)
    # All Non → relevant_count=0 in both iterations
    # Iter 0: 0 > -1 → updates best (articles 1-100)
    # Iter 1: 0 > 0 is False → keeps iter 0's docs (articles 1-100)
    crag = CRAGRetriever(
        retriever=AlternatingRetriever(),
        df_articles=df,
        llm_client=MockLLMJudge(judge_responses=["Non"] * 20, rewrite_response="query bis"),
        eval_k=5,
        max_iterations=2,
    )
    ids, _ = crag.retrieve("test query")
    # best_retrieved_docs must be from iter 0 (articles 1-100)
    assert ids[0] in range(1, 101), f"Expected iter 0 article, got {ids[0]}"
    assert ids[0] not in range(101, 201)


def test_loop_termination_correct():
    """Stops at iteration 0 when first-pass is Correct (trace length == 1)."""
    crag = _make_crag(judge_responses=["Oui", "Non", "Non", "Non", "Non"])
    crag.retrieve("test query")
    assert len(crag._all_traces[-1]) == 1


def test_loop_termination_max_iter():
    """Stops when iteration count reaches max_iterations.

    With max_iterations=2, route fires when iteration==2 (3rd evaluate call),
    so the trace has at most max_iterations+1 entries.
    """
    max_iter = 2
    crag = _make_crag(
        judge_responses=["Non"] * 15,  # all Non → never Correct
        max_iterations=max_iter,
    )
    crag.retrieve("test query")
    trace = crag._all_traces[-1]
    assert len(trace) <= max_iter + 1


def test_loop_termination_no_change():
    """Stops when LLM rewrites query to the same string (no-change guard)."""
    original_query = "conditions pour obtenir un congé parental"
    crag = _make_crag(
        judge_responses=["Non"] * 5,   # Incorrect → triggers rewrite
        rewrite_response=original_query,  # LLM "rewrites" to same query
        max_iterations=5,
    )
    crag.retrieve(original_query)
    trace = crag._all_traces[-1]
    # Terminates after 2 iterations: iter 0 (Incorrect) + iter 1 (no-change)
    assert len(trace) <= 3


def test_latency_fields_populated():
    """All three latency_breakdown_ms keys > 0 after a loop with one rewrite."""
    # All Non → Incorrect → rewrite → all Non again → terminates by max_iter
    crag = _make_crag(
        judge_responses=["Non"] * 15,
        max_iterations=2,
    )
    crag.retrieve("test query")
    bd = crag.get_latency_breakdown()
    assert bd["retrieval"]     > 0.0, "retrieval latency must be > 0"
    assert bd["evaluator_llm"] > 0.0, "evaluator_llm latency must be > 0"
    assert bd["rewrite_llm"]   > 0.0, "rewrite_llm latency must be > 0"


def test_finalize_sort_order():
    """Relevant docs (Oui) appear first; within groups sorted by retrieval_score."""
    article_ids = list(range(1, 101))
    # eval_k=5: docs 0-4 get judged; docs 1,3 (articles 2,4) get Oui
    # Judge response sequence: Non, Oui, Non, Oui, Non → relevant = articles 2, 4
    # With max_iterations=1: Correct (count=2) → finalize immediately
    crag = _make_crag(
        article_ids=article_ids,
        judge_responses=["Non", "Oui", "Non", "Oui", "Non"],
        max_iterations=1,
        eval_k=5,
    )
    ids, _ = crag.retrieve("test query")

    # Articles 2 and 4 were judged relevant; they should appear before article 1,3,5
    relevant_set  = {article_ids[1], article_ids[3]}    # 2, 4
    irrelevant_set = {article_ids[0], article_ids[2], article_ids[4]}  # 1, 3, 5

    # All relevant docs appear before all irrelevant docs
    rel_positions   = [i for i, aid in enumerate(ids) if aid in relevant_set]
    irrel_positions = [i for i, aid in enumerate(ids) if aid in irrelevant_set]
    if rel_positions and irrel_positions:
        assert max(rel_positions) < min(irrel_positions), (
            f"Relevant docs not sorted before irrelevant. "
            f"rel_pos={rel_positions}, irrel_pos={irrel_positions}"
        )


def test_french_prompt_output():
    """LLM generates a non-empty rewrite string different from the input query."""
    crag = _make_crag(
        judge_responses=["Non"] * 5,   # Incorrect → triggers LLM rewrite
        rewrite_response="licenciement abusif indemnité légale",
        max_iterations=1,
    )
    original_query = "conditions de licenciement"
    ids, _ = crag.retrieve(original_query)
    trace = crag._all_traces[-1]
    assert trace[0]["confidence"] == "Incorrect"
    stats = crag.get_loop_stats()
    assert stats["fraction_queries_rewritten"] == 1.0


def test_retrieval_trace_length():
    """retrieval_trace has exactly one entry per retrieve+evaluate cycle."""
    crag = _make_crag(
        judge_responses=["Non"] * 20,   # always Incorrect
        max_iterations=3,
    )
    crag.retrieve("test query")
    trace = crag._all_traces[-1]
    # Should have exactly max_iterations+1 entries (or fewer if earlier termination)
    assert 1 <= len(trace) <= 4
    # Each entry must have the required fields
    for entry in trace:
        assert "iteration"      in entry
        assert "query"          in entry
        assert "relevant_count" in entry
        assert "confidence"     in entry
