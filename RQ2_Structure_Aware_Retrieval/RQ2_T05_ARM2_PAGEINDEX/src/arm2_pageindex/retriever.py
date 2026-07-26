"""Single-query public API for Arm 2B.

Wraps ``navigator.navigate`` and converts the resulting state into the
RQ2-canonical ``RetrievalResult`` dataclass (``data_loader.models``).
The result is read by:

  * T07's evaluation adapter, via ``result.ranked_items[*]['metadata']
    ['bsard_id']`` (CN-T05-001).
  * T07's cost tracker, via ``result.cost`` -> ``llm_calls``,
    ``tokens_in``, ``tokens_out``, ``latency_ms``.
  * T08's PDF viewer, via ``result.trace`` for the navigation overlay.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Iterable

from data_loader.models import RetrievalResult

from .navigator import NavigationState, NavigatorConfig, navigate
from .tree_builder import ToCNode, find_node

if TYPE_CHECKING:
    from shared.llm import LLMClient

logger = logging.getLogger(__name__)

DEFAULT_METHOD_NAME = "arm2b_pageindex"


def run_arm2b(
    *,
    query: str,
    query_id: str,
    tree: ToCNode,
    llm: "LLMClient",
    cfg: NavigatorConfig | None = None,
    method_name: str = DEFAULT_METHOD_NAME,
) -> RetrievalResult:
    """Navigate the tree for one query and return the RQ2 RetrievalResult."""
    state = navigate(
        query=query, query_id=query_id, tree=tree, llm=llm, cfg=cfg,
    )
    return state_to_retrieval_result(state, method_name=method_name)


def state_to_retrieval_result(
    state: NavigationState,
    *,
    method_name: str = DEFAULT_METHOD_NAME,
) -> RetrievalResult:
    """Convert a NavigationState into a serialisable RetrievalResult."""
    ranked_items = list(_ranked_items_from_state(state))

    cost = dict(state.cost)
    cost["exit_reason"] = state.exit_reason
    cost["iterations"] = state.iteration

    trace_payload = {
        "summary": {
            "exit_reason": state.exit_reason,
            "iteration": state.iteration,
            "selected_law_ids": list(state.selected_law_ids),
            "selected_chapter_ids": list(state.selected_chapter_ids),
            "candidate_article_ids": list(state.candidate_article_ids),
            "n_candidates": len(state.candidate_article_ids),
        },
        "steps": list(state.trace),
    }
    return RetrievalResult(
        query_id=state.query_id,
        query_text=state.query,
        ranked_items=ranked_items,
        method=method_name,
        cost=cost,
        trace=trace_payload,
    )


def _ranked_items_from_state(state: NavigationState) -> Iterable[dict]:
    seen: set[int] = set()
    for node_id, score in state.final_ranked:
        leaf = find_node(state.tree, node_id)
        if leaf is None:
            continue
        if leaf.bsard_id is None:
            logger.warning(
                "Dropping ranked item %s: no bsard_id "
                "(violates CN-T05-001 contract)", node_id,
            )
            continue
        # T07 evaluator collapses duplicate bsard_ids to max score, so
        # emitting them is harmless; we skip exact duplicates only to
        # keep the saved bundle compact.
        if leaf.bsard_id in seen:
            continue
        seen.add(leaf.bsard_id)

        meta = dict(leaf.metadata) if leaf.metadata else {}
        meta["bsard_id"] = int(leaf.bsard_id)
        if leaf.text and "text" not in meta:
            meta["text"] = leaf.text  # T08 viewer reads metadata["text"]

        yield {
            "id": node_id,
            "score": float(score),
            "text": leaf.text or "",
            "metadata": meta,
        }
