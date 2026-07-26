"""
CRAG — Corrective Retrieval-Augmented Generation (Tier 4.1).

Architecture
------------
CRAGRetriever
    Public retriever matching the project RetrieverProtocol:
        retrieve(query, top_k=100) -> (ranked_article_ids, latency_ms)

    Internally runs a LangGraph StateGraph with four nodes:
        retrieve      — backbone.retrieve(query, top_k=100)
        evaluate      — LLM binary judgment on top-eval_k docs (Oui/Non per doc)
        rewrite_query — LLM rewrite for the Incorrect path
        finalize      — sort best_retrieved_docs by relevance then retrieval_score

State contract (CRAGState)
--------------------------
See §1.5.3 of TIER41_AGENTIC_CRAG_PLAN.md for the full read/write table.

Best-seen tracking
------------------
best_retrieved_docs and best_relevant_count are only updated when the current
iteration's relevant_count strictly exceeds the historical best. This prevents
the loop timeout from returning a degraded result (§1.5.4).
Tie-break: equal relevant_count keeps the first iteration's docs (iteration 0
is the unbiased retrieval).

Loop-termination conditions (checked in route_after_evaluate)
--------------------------------------------------------------
  1. confidence == "Correct"      (relevant_count >= 1)
  2. iteration >= max_iterations
  3. query == previous_query      (LLM produced an unchanged rewrite)

Binary routing (LLM-based)
--------------------------
  relevant_count  = sum("Oui" judgments over top eval_k docs)
  relevant_count >= 1 → Correct   → finalize
  relevant_count == 0 → Incorrect → rewrite_query

  eval_k is calibrated on the val split via
  scripts/agentic/calibrate_crag_llm.py before experiments are run.

Evaluator: LLaMA 3.1 8B via OllamaClient — same binary prompt and few-shot
  examples as T4.0 (LLM_JUDGE_BINARY_PROMPT from llm_eval_prompts.py). This
  eliminates the evaluator confound across all Tier 4 experiments.
"""

from __future__ import annotations

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
from retrieval.agentic.prompts import CRAG_REWRITE_PROMPT


# ---------------------------------------------------------------------------
# State definition
# ---------------------------------------------------------------------------

class CRAGState(TypedDict):
    # ── Input (set once at invocation) ──────────────────────────────────────
    query:              str           # Current query (mutated by rewrite)
    original_query:     str           # Immutable — first query; used in prompts
    previous_query:     str           # Query from the previous iteration (loop guard)
    iteration:          int           # 0-based iteration counter

    # ── Retrieval output ────────────────────────────────────────────────────
    retrieved_docs: list[dict]
    # Each doc: {article_id: int, retrieval_score: float,
    #            is_relevant: bool|None, llm_response: str|None}

    # ── Best-seen tracking ──────────────────────────────────────────────────
    best_retrieved_docs:  list[dict]  # Docs from the best-scoring iteration
    best_relevant_count:  int         # Highest relevant_count seen (-1 sentinel)

    # ── Routing ─────────────────────────────────────────────────────────────
    confidence: str                   # "Correct" | "Incorrect" | ""

    # ── Diagnostics ─────────────────────────────────────────────────────────
    retrieval_trace:      list[dict]  # One entry per iteration
    latency_breakdown_ms: dict        # {retrieval, evaluator_llm, rewrite_llm}

    # ── Output ──────────────────────────────────────────────────────────────
    final_article_ids:  list[int]


# ---------------------------------------------------------------------------
# CRAGRetriever
# ---------------------------------------------------------------------------

class CRAGRetriever:
    """
    Drop-in retriever wrapping any Tier 1–3 backbone with the CRAG loop.

    Parameters
    ----------
    retriever          : any object with .retrieve(query, top_k) -> (list[int], float)
    df_articles        : DataFrame with columns [article_id, article_text]
    llm_client         : OllamaClient (llama3.1:8b)
    eval_k             : number of top docs evaluated by LLM per iteration (default 5)
                         Set eval_k == backbone_top_k to evaluate all retrieved docs
                         (required for direct T4.0 comparability).
    max_iterations     : hard iteration cap (default 2)
    backbone_top_k     : docs retrieved from backbone per iteration (default 100)
                         Set to 20 together with eval_k=20 to match T4.0's candidate pool.
    fewshot_examples   : dict returned by load_fewshot_examples(); None = no few-shot
    max_article_tokens : article text truncation in whitespace tokens (default 300)
    """

    def __init__(
        self,
        retriever,
        df_articles: pd.DataFrame,
        llm_client: OllamaClient,
        eval_k: int = 5,
        max_iterations: int = 2,
        backbone_top_k: int = 100,
        fewshot_examples: Optional[dict] = None,
        max_article_tokens: int = 300,
    ):
        self._retriever        = retriever
        self._llm              = llm_client
        self.eval_k            = eval_k
        self.max_iterations    = max_iterations
        self._backbone_top_k   = backbone_top_k
        self._fewshot_block    = format_fewshot_block(fewshot_examples or {}, "binary")
        self._max_article_tokens = max_article_tokens

        # Article text lookup
        self._id_to_text: dict[int, str] = dict(
            zip(df_articles["article_id"], df_articles["article_text"])
        )

        # Stats accumulator (reset per retrieve() call, aggregated by get_loop_stats)
        self._all_traces:     list[list[dict]] = []
        self._all_breakdowns: list[dict]       = []
        self._last_breakdown: dict             = {}

        # Build graph once
        self._graph = self._build_graph()

    # ── Public interface ─────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 100) -> tuple[list[int], float]:
        """
        Run the CRAG loop and return (ranked_article_ids[:top_k], total_latency_ms).
        Internal retrieval uses backbone_top_k per iteration; top_k slices the output.
        """
        initial_state: CRAGState = {
            "query":               query,
            "original_query":      query,
            "previous_query":      "",
            "iteration":           0,
            "retrieved_docs":      [],
            "best_retrieved_docs": [],
            "best_relevant_count": -1,   # -1 sentinel ensures first iteration always updates
            "confidence":          "",
            "retrieval_trace":     [],
            "latency_breakdown_ms": {
                "retrieval":     0.0,
                "evaluator_llm": 0.0,
                "rewrite_llm":   0.0,
            },
            "final_article_ids":  [],
        }

        t0 = time.perf_counter()
        result = self._graph.invoke(initial_state)
        total_ms = (time.perf_counter() - t0) * 1000.0

        self._last_breakdown = result["latency_breakdown_ms"]
        self._all_breakdowns.append(dict(result["latency_breakdown_ms"]))
        self._all_traces.append(result["retrieval_trace"])

        return result["final_article_ids"][:top_k], total_ms

    def get_latency_breakdown(self) -> dict:
        """Return latency breakdown from the last retrieve() call."""
        return dict(self._last_breakdown)

    def get_latency_breakdown_mean(self) -> dict:
        """Return mean per-component latency (ms) across all retrieve() calls."""
        if not self._all_breakdowns:
            return {"retrieval": 0.0, "evaluator_llm": 0.0, "rewrite_llm": 0.0}
        keys = ("retrieval", "evaluator_llm", "rewrite_llm")
        return {
            k: round(float(np.mean([b.get(k, 0.0) for b in self._all_breakdowns])), 1)
            for k in keys
        }

    def get_loop_stats(self) -> dict:
        """
        Aggregate loop statistics over all retrieve() calls made so far.
        Returns fractions and means across queries.
        """
        traces = self._all_traces
        if not traces:
            return {}

        n = len(traces)
        iterations_per_query = [len(t) for t in traces]

        correct_first  = sum(1 for t in traces if t and t[0]["confidence"] == "Correct")
        rewritten      = sum(1 for t in traces if any(e["confidence"] == "Incorrect" for e in t))
        terminated_lim = sum(
            1 for t in traces
            if t and t[-1]["confidence"] != "Correct"
               and len(t) >= self.max_iterations
        )
        no_change_term = sum(
            1 for t in traces if any(e.get("no_change_termination") for e in t)
        )

        all_entries = [e for t in traces for e in t]
        n_entries   = max(len(all_entries), 1)

        return {
            "n_queries":                              n,
            "mean_iterations_per_query":              float(np.mean(iterations_per_query)),
            "fraction_queries_correct_first_pass":    correct_first / n,
            "fraction_queries_rewritten":             rewritten / n,
            "fraction_queries_terminated_by_limit":   terminated_lim / n,
            "fraction_queries_no_change_termination": no_change_term / n,
            "distribution_Correct":   sum(
                e["confidence"] == "Correct" for e in all_entries
            ) / n_entries,
            "distribution_Incorrect": sum(
                e["confidence"] == "Incorrect" for e in all_entries
            ) / n_entries,
        }

    # ── Graph construction ────────────────────────────────────────────────────

    def _build_graph(self):
        from langgraph.graph import END, START, StateGraph

        # Capture self in closures
        retriever          = self._retriever
        llm                = self._llm
        id_to_text         = self._id_to_text
        eval_k             = self.eval_k
        max_iter           = self.max_iterations
        backbone_top_k     = self._backbone_top_k
        fewshot_block      = self._fewshot_block
        max_article_tokens = self._max_article_tokens

        # ── Node: retrieve ───────────────────────────────────────────────────
        def retrieve_node(state: CRAGState) -> dict:
            t0 = time.perf_counter()
            article_ids, _ = retriever.retrieve(state["query"], top_k=backbone_top_k)
            lat = (time.perf_counter() - t0) * 1000.0

            retrieved_docs = [
                {
                    "article_id":      aid,
                    "retrieval_score": 1.0 / (i + 1),  # reciprocal rank proxy
                    "is_relevant":     None,
                    "llm_response":    None,
                }
                for i, aid in enumerate(article_ids)
            ]

            breakdown = dict(state["latency_breakdown_ms"])
            breakdown["retrieval"] = breakdown.get("retrieval", 0.0) + lat

            return {
                "retrieved_docs":      retrieved_docs,
                "latency_breakdown_ms": breakdown,
            }

        # ── Node: evaluate ───────────────────────────────────────────────────
        def evaluate_node(state: CRAGState) -> dict:
            docs     = state["retrieved_docs"]
            top_docs = docs[:eval_k]

            t0 = time.perf_counter()
            all_docs = list(docs)
            relevant_count = 0

            for i, doc in enumerate(top_docs):
                article_text = id_to_text.get(doc["article_id"], "")

                # Truncate to max_article_tokens whitespace-delimited tokens
                words = article_text.split()
                if len(words) > max_article_tokens:
                    article_text = " ".join(words[:max_article_tokens]) + " …"

                prompt = LLM_JUDGE_BINARY_PROMPT.format(
                    fewshot_block=fewshot_block,
                    question=state["query"],
                    article_text_truncated=article_text,
                )
                response, _ = llm.generate(prompt, temperature=0.0, max_tokens=5)
                is_relevant, _ = parse_binary_judgment(response)

                if is_relevant:
                    relevant_count += 1

                all_docs[i] = {
                    **all_docs[i],
                    "is_relevant":  is_relevant,
                    "llm_response": response.strip(),
                }

            llm_lat = (time.perf_counter() - t0) * 1000.0

            # Best-seen update: only when strictly better (tie → keep iteration 0)
            if relevant_count > state["best_relevant_count"]:
                best_docs  = all_docs
                best_count = relevant_count
            else:
                best_docs  = state["best_retrieved_docs"]
                best_count = state["best_relevant_count"]

            # Binary routing: relevant_count >= 1 → Correct
            confidence = "Correct" if relevant_count >= 1 else "Incorrect"

            trace_entry = {
                "iteration":              state["iteration"],
                "query":                  state["query"],
                "relevant_count":         relevant_count,
                "confidence":             confidence,
                "no_change_termination":  False,
                "retrieved_article_ids":  [d["article_id"] for d in docs[:eval_k]],
            }

            breakdown = dict(state["latency_breakdown_ms"])
            breakdown["evaluator_llm"] = breakdown.get("evaluator_llm", 0.0) + llm_lat

            return {
                "retrieved_docs":      all_docs,
                "best_retrieved_docs": best_docs,
                "best_relevant_count": best_count,
                "confidence":          confidence,
                "retrieval_trace":     state["retrieval_trace"] + [trace_entry],
                "latency_breakdown_ms": breakdown,
            }

        # ── Node: rewrite_query ──────────────────────────────────────────────
        def rewrite_query_node(state: CRAGState) -> dict:
            prompt = CRAG_REWRITE_PROMPT.format(
                original_query=state["original_query"],
                current_query=state["query"],
            )
            new_query, llm_lat = llm.generate(
                prompt, temperature=0.0, max_tokens=128
            )
            new_query = new_query.strip()

            breakdown = dict(state["latency_breakdown_ms"])
            breakdown["rewrite_llm"] = breakdown.get("rewrite_llm", 0.0) + llm_lat

            return {
                "previous_query":       state["query"],
                "query":                new_query,
                "iteration":            state["iteration"] + 1,
                "latency_breakdown_ms": breakdown,
            }

        # ── Node: finalize ───────────────────────────────────────────────────
        def finalize_node(state: CRAGState) -> dict:
            docs = state["best_retrieved_docs"]

            # Relevant docs (is_relevant=True) sorted by retrieval_score descending
            relevant_docs = sorted(
                [d for d in docs if d.get("is_relevant") is True],
                key=lambda d: d["retrieval_score"],
                reverse=True,
            )
            # Non-relevant or unscored docs sorted by retrieval_score descending
            other_docs = sorted(
                [d for d in docs if d.get("is_relevant") is not True],
                key=lambda d: d["retrieval_score"],
                reverse=True,
            )

            final_ids = [d["article_id"] for d in relevant_docs + other_docs]
            return {"final_article_ids": final_ids}

        # ── Routing function ─────────────────────────────────────────────────
        def route_after_evaluate(state: CRAGState) -> str:
            no_change = (
                state["query"] == state["previous_query"]
                and state["previous_query"] != ""  # ignore the very first iteration
            )
            if no_change:
                # Mark trace entry so get_loop_stats() can count it
                if state["retrieval_trace"]:
                    state["retrieval_trace"][-1]["no_change_termination"] = True
                return "finalize"
            if (
                state["confidence"] == "Correct"
                or state["iteration"] >= max_iter
            ):
                return "finalize"
            return "rewrite_query"   # Incorrect

        # ── Wire the graph ───────────────────────────────────────────────────
        builder = StateGraph(CRAGState)
        builder.add_node("retrieve",      retrieve_node)
        builder.add_node("evaluate",      evaluate_node)
        builder.add_node("rewrite_query", rewrite_query_node)
        builder.add_node("finalize",      finalize_node)

        builder.add_edge(START,            "retrieve")
        builder.add_edge("retrieve",       "evaluate")
        builder.add_conditional_edges(
            "evaluate",
            route_after_evaluate,
            {
                "finalize":      "finalize",
                "rewrite_query": "rewrite_query",
            },
        )
        builder.add_edge("rewrite_query",  "retrieve")
        builder.add_edge("finalize",       END)

        return builder.compile()
