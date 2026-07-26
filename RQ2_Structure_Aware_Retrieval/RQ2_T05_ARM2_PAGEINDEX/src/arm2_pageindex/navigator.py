"""Per-query navigation loop for Arm 2B.

Plain Python pipeline (no LangGraph) — the control flow is linear with
one feedback loop, so a graph compiler would only obscure it.

Pipeline:

    select_laws  ->  select_chapters  ->  select_articles  ->  evaluate
                                                                  |
                                                                  v
                                                          sufficient -> END
                                                                  |
                                                                  v
                                                          need_more (iter < max)
                                                                  |
                                                                  v
                                                          follow_refs -> evaluate
                                                                  |
                                                          iter >= max -> force END

State is a single dataclass that accumulates the trace, cost
(``llm_calls``, ``tokens_in``, ``tokens_out``, ``latency_ms``), and the
ranked-article list.

The navigator is deliberately tolerant of LLM JSON-parse failures:
a failed step records ``parse_ok=False`` in the trace, propagates an
empty selection downstream, and lets later steps recover or the
``evaluate`` step terminate with ``exit_reason='parse_fail'``. This
matches the pattern used by the RQ1 azure notebooks.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # avoid loading shared.__init__'s heavy deps (faiss, transformers)
    from shared.llm import LLMClient

from .prompts import (
    ARTICLE_SELECTION_PROMPT,
    CHAPTER_SELECTION_PROMPT,
    EVALUATE_PROMPT,
    LAW_SELECTION_PROMPT,
)
from .tree_builder import ToCNode, find_node, iter_leaves

logger = logging.getLogger(__name__)

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)
_ARTICLE_NUM_NORMALISE_RE = re.compile(r"[\.\s\-_/]+")


# ── Configuration & state ────────────────────────────────────────────────────


@dataclass
class NavigatorConfig:
    max_laws: int = 3
    max_chapters_per_law: int = 5
    max_articles_per_chapter: int = 6
    max_iterations: int = 3
    skip_law_selection_if_single_doc: bool = True
    score_threshold: int = 1   # leaves with eval_score >= threshold survive
    # Pad the final ranking with leaves of selected laws (score 0.0)
    # until len(final_ranked) >= pad_to_k. Makes R@K for K in {20, 100}
    # comparable with vector retrievers (T03/T04) that always emit ~100
    # candidates. Set to 0 to disable padding.
    pad_to_k: int = 100


@dataclass
class NavigationState:
    query: str
    query_id: str
    tree: ToCNode

    cfg: NavigatorConfig
    iteration: int = 0

    selected_law_ids: list[str] = field(default_factory=list)
    selected_chapter_ids: list[str] = field(default_factory=list)
    candidate_article_ids: list[str] = field(default_factory=list)
    article_scores: dict[str, int] = field(default_factory=dict)

    exit_reason: str = "incomplete"
    final_ranked: list[tuple[str, float]] = field(default_factory=list)

    cost: dict = field(default_factory=lambda: {
        "llm_calls": 0,
        "tokens_in": 0,
        "tokens_out": 0,
        "latency_ms": 0.0,
        "parse_failures": 0,
    })
    trace: list[dict] = field(default_factory=list)


# ── LLM helper with telemetry ────────────────────────────────────────────────


def _extract_json_text(text: str) -> str:
    m = _JSON_FENCE_RE.search(text)
    return (m.group(1) if m else text).strip()


def _llm_json_call(
    llm: LLMClient,
    prompt: str,
    *,
    max_attempts: int = 2,
) -> tuple[dict, dict]:
    """Generate + parse JSON. Returns ``(parsed_or_empty, metrics)``.

    metrics = {tokens_in, tokens_out, latency_ms, attempts, parse_ok, raw_text}.

    On parse failure ``parsed`` is ``{}`` and ``parse_ok=False`` so the
    caller can degrade gracefully rather than crash mid-query.
    """
    metrics: dict[str, Any] = {
        "tokens_in": 0, "tokens_out": 0, "latency_ms": 0.0,
        "attempts": 0, "parse_ok": False, "raw_text": "",
    }
    for _ in range(max_attempts):
        resp = llm.generate(prompt)
        metrics["tokens_in"] += resp.input_tokens
        metrics["tokens_out"] += resp.output_tokens
        metrics["latency_ms"] += resp.latency_ms
        metrics["attempts"] += 1
        metrics["raw_text"] = resp.text
        try:
            parsed = json.loads(_extract_json_text(resp.text))
            metrics["parse_ok"] = True
            return parsed, metrics
        except json.JSONDecodeError:
            continue
    return {}, metrics


def _extract_selected_ids(parsed: Any, key: str) -> list[str]:
    """Pull ``key`` from a parsed JSON response.

    Accepts both shapes the 8B model produces in practice:
      * ``{"<key>": [...], "reason": "..."}`` (prompted form)
      * ``[{"<key>": [...], ...}, {"<key>": [...], ...}, ...]`` —
        an array of per-section objects that the model emits when it
        decides to iterate internally (observed on every PDF 1804
        chapter-selection response pre-fix). Without flattening,
        ``parsed.get(...)`` raises AttributeError on lists and the
        original code defensively dropped the whole result.
    """
    if isinstance(parsed, dict):
        val = parsed.get(key, [])
        if isinstance(val, list):
            return [s for s in val if isinstance(s, str)]
        return []
    if isinstance(parsed, list):
        out: list[str] = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            val = item.get(key, [])
            if isinstance(val, list):
                out.extend(s for s in val if isinstance(s, str))
        return out
    return []


def _record(
    state: NavigationState,
    step: str,
    prompt: str,
    parsed: dict,
    metrics: dict,
) -> None:
    state.cost["llm_calls"] += 1
    state.cost["tokens_in"] += metrics["tokens_in"]
    state.cost["tokens_out"] += metrics["tokens_out"]
    state.cost["latency_ms"] += metrics["latency_ms"]
    if not metrics["parse_ok"]:
        state.cost["parse_failures"] += 1
    state.trace.append({
        "step": step,
        "iteration": state.iteration,
        "prompt_chars": len(prompt),
        "parsed": parsed,
        "parse_ok": metrics["parse_ok"],
        "tokens_in": metrics["tokens_in"],
        "tokens_out": metrics["tokens_out"],
        "latency_ms": round(metrics["latency_ms"], 1),
        "raw_text": (
            None if metrics["parse_ok"] else metrics["raw_text"][:500]
        ),
    })


# ── Tree helpers ─────────────────────────────────────────────────────────────


def _looks_single_doc(tree: ToCNode) -> bool:
    if tree.node_id.startswith("LAW_"):
        return True
    if tree.node_id == "ROOT" and len(tree.sub_nodes) == 1:
        return True
    return False


def _laws_in_tree(tree: ToCNode) -> list[ToCNode]:
    if tree.node_id.startswith("LAW_"):
        return [tree]
    return [n for n in tree.sub_nodes if n.node_id.startswith("LAW_")]


def _normalise_article_number(s: str) -> str:
    return _ARTICLE_NUM_NORMALISE_RE.sub("", s or "").upper()


def _build_article_number_index(tree: ToCNode) -> dict[str, list[ToCNode]]:
    """Map normalised article_number -> list of leaf nodes (typically size 1)."""
    idx: dict[str, list[ToCNode]] = {}
    for leaf in iter_leaves(tree):
        an = (leaf.metadata or {}).get("article_number", "") if leaf.metadata else ""
        if an:
            key = _normalise_article_number(an)
            idx.setdefault(key, []).append(leaf)
    return idx


# ── Steps ────────────────────────────────────────────────────────────────────


def select_laws(state: NavigationState, llm: LLMClient) -> None:
    laws = _laws_in_tree(state.tree)
    if state.cfg.skip_law_selection_if_single_doc and _looks_single_doc(state.tree):
        state.selected_law_ids = [law.node_id for law in laws]
        state.trace.append({
            "step": "law_selection",
            "iteration": 0,
            "skipped": True,
            "reason": "single_doc_corpus",
            "selected": list(state.selected_law_ids),
        })
        return

    laws_json = json.dumps(
        [law.to_prompt_dict(depth=0) for law in laws],
        ensure_ascii=False, indent=2,
    )
    prompt = LAW_SELECTION_PROMPT.format(
        query=state.query,
        max_laws=state.cfg.max_laws,
        laws_json=laws_json,
        example_law_id=laws[0].node_id,
    )
    parsed, metrics = _llm_json_call(llm, prompt)
    raw_selected = _extract_selected_ids(parsed, "selected_law_ids")
    valid = {law.node_id for law in laws}
    selected = [s for s in raw_selected if s in valid][: state.cfg.max_laws]
    if not selected:
        # fallback: take all laws (matches "structure-agnostic" floor)
        logger.warning(
            "Law selection produced no valid IDs (parse_ok=%s); "
            "falling back to all %d laws", metrics["parse_ok"], len(valid),
        )
        selected = sorted(valid)
    state.selected_law_ids = selected
    _record(state, "law_selection", prompt, parsed, metrics)


def select_chapters(state: NavigationState, llm: LLMClient) -> None:
    selected: list[str] = []
    for law_id in state.selected_law_ids:
        law_node = find_node(state.tree, law_id)
        if law_node is None:
            continue

        # 2-level fallback: this law has no chapters, only direct articles.
        children_are_leaves = bool(law_node.sub_nodes) and all(
            c.is_leaf for c in law_node.sub_nodes
        )
        if children_are_leaves:
            # Synthesise a single virtual "chapter" id so select_articles
            # processes the full law in one shot.
            selected.append(law_node.node_id)
            state.trace.append({
                "step": "chapter_selection",
                "iteration": 0,
                "law_id": law_id,
                "skipped": True,
                "reason": "law_is_2_level",
                "n_articles": len(law_node.sub_nodes),
            })
            continue

        chapters_json = json.dumps(
            [ch.to_prompt_dict(depth=0) for ch in law_node.sub_nodes],
            ensure_ascii=False, indent=2,
        )
        prompt = CHAPTER_SELECTION_PROMPT.format(
            query=state.query,
            law_title=law_node.title,
            max_chapters=state.cfg.max_chapters_per_law,
            chapters_json=chapters_json,
            example_chapter_id=law_node.sub_nodes[0].node_id,
        )
        parsed, metrics = _llm_json_call(llm, prompt)
        raw = _extract_selected_ids(parsed, "selected_chapter_ids")
        valid = {ch.node_id for ch in law_node.sub_nodes}
        chosen = [c for c in raw if c in valid][
            : state.cfg.max_chapters_per_law
        ]
        if not chosen and metrics["parse_ok"]:
            logger.info("Law %s: chapter selection picked none — skipping law", law_id)
        selected.extend(chosen)
        _record(state, f"chapter_selection[{law_id}]", prompt, parsed, metrics)

    state.selected_chapter_ids = selected


def select_articles(state: NavigationState, llm: LLMClient) -> None:
    candidates: list[str] = []
    for chapter_id in state.selected_chapter_ids:
        chapter_node = find_node(state.tree, chapter_id)
        if chapter_node is None:
            continue
        # Leaves of this chapter (or all-articles when chapter_id is a 2-level law).
        leaves = (
            chapter_node.sub_nodes
            if chapter_node.sub_nodes and all(c.is_leaf for c in chapter_node.sub_nodes)
            else [n for n in iter_leaves(chapter_node)]
        )
        if not leaves:
            continue

        articles_json = json.dumps(
            [leaf.to_prompt_dict(depth=0) for leaf in leaves],
            ensure_ascii=False, indent=2,
        )
        prompt = ARTICLE_SELECTION_PROMPT.format(
            query=state.query,
            chapter_title=chapter_node.title,
            articles_json=articles_json,
            example_article_id=leaves[0].node_id,
        )
        parsed, metrics = _llm_json_call(llm, prompt)
        raw = _extract_selected_ids(parsed, "selected_article_ids")
        valid = {leaf.node_id for leaf in leaves}
        chosen = [a for a in raw if a in valid][
            : state.cfg.max_articles_per_chapter
        ]
        candidates.extend(chosen)
        _record(state, f"article_selection[{chapter_id}]", prompt, parsed, metrics)

    # Deduplicate while preserving order.
    seen: set[str] = set()
    deduped: list[str] = []
    for aid in candidates:
        if aid not in seen:
            seen.add(aid)
            deduped.append(aid)
    state.candidate_article_ids = deduped


def _format_articles_for_evaluation(
    state: NavigationState, article_ids: list[str],
) -> str:
    pieces: list[str] = []
    for aid in article_ids:
        leaf = find_node(state.tree, aid)
        if leaf is None or not leaf.text:
            continue
        pieces.append(f"[{aid}]\n{leaf.text}")
    return "\n\n---\n\n".join(pieces)


def evaluate(state: NavigationState, llm: LLMClient) -> dict:
    """Run the evaluate prompt; update ``article_scores``; return parsed payload."""
    if not state.candidate_article_ids:
        state.exit_reason = "no_candidates"
        return {"status": "sufficient", "scores": {}, "additional_article_numbers": []}

    article_texts = _format_articles_for_evaluation(
        state, state.candidate_article_ids,
    )
    prompt = EVALUATE_PROMPT.format(
        query=state.query,
        article_texts=article_texts,
        example_article_id=state.candidate_article_ids[0],
    )
    parsed, metrics = _llm_json_call(llm, prompt)
    _record(state, f"evaluate[iter={state.iteration}]", prompt, parsed, metrics)

    if not metrics["parse_ok"] or not isinstance(parsed, dict):
        state.exit_reason = "parse_fail"
        return {"status": "sufficient", "scores": {}, "additional_article_numbers": []}

    raw_scores = parsed.get("scores", {}) or {}
    if isinstance(raw_scores, dict):
        for aid, sc in raw_scores.items():
            if not isinstance(aid, str):
                continue
            try:
                score = int(sc)
            except (TypeError, ValueError):
                continue
            # Highest score wins across iterations.
            prior = state.article_scores.get(aid, -1)
            if score > prior:
                state.article_scores[aid] = score
    return parsed


def follow_refs(
    state: NavigationState, eval_payload: dict, art_num_index: dict[str, list[ToCNode]],
) -> int:
    """Resolve ``additional_article_numbers`` to leaf node_ids and append.

    Returns the number of new articles added (post-dedup).
    """
    raw_refs = eval_payload.get("additional_article_numbers", []) or []
    if not isinstance(raw_refs, list):
        return 0
    seen = set(state.candidate_article_ids)
    added = 0
    for ref in raw_refs:
        if not isinstance(ref, str):
            continue
        key = _normalise_article_number(ref)
        if not key:
            continue
        for leaf in art_num_index.get(key, []):
            if leaf.node_id not in seen:
                seen.add(leaf.node_id)
                state.candidate_article_ids.append(leaf.node_id)
                added += 1
    state.trace.append({
        "step": f"follow_refs[iter={state.iteration}]",
        "requested": raw_refs,
        "added": added,
    })
    return added


# ── Top-level ────────────────────────────────────────────────────────────────


def navigate(
    *,
    query: str,
    query_id: str,
    tree: ToCNode,
    llm: LLMClient,
    cfg: NavigatorConfig | None = None,
) -> NavigationState:
    """Run the full navigation loop for one query. Returns the final state."""
    state = NavigationState(
        query=query, query_id=query_id, tree=tree, cfg=cfg or NavigatorConfig(),
    )

    select_laws(state, llm)
    select_chapters(state, llm)
    select_articles(state, llm)

    art_num_index = _build_article_number_index(tree)
    eval_payload = evaluate(state, llm)

    while True:
        status = eval_payload.get("status", "sufficient")
        if state.exit_reason in {"parse_fail", "no_candidates"}:
            break
        if status == "sufficient":
            state.exit_reason = "sufficient"
            break
        if state.iteration + 1 >= state.cfg.max_iterations:
            state.exit_reason = "force_output"
            break
        added = follow_refs(state, eval_payload, art_num_index)
        if added == 0:
            # LLM asked for refs but none resolved — no point looping.
            state.exit_reason = "no_refs_resolved"
            break
        state.iteration += 1
        eval_payload = evaluate(state, llm)

    # Build final ranking. Articles with no recorded score get
    # ``score_threshold`` so they survive when the LLM forgot to score them.
    ranked: list[tuple[str, float]] = []
    for aid in state.candidate_article_ids:
        score = state.article_scores.get(aid)
        if score is None:
            score = state.cfg.score_threshold
        if score >= state.cfg.score_threshold:
            ranked.append((aid, float(score)))
    ranked.sort(key=lambda x: -x[1])

    # Two-tier padding so R@K for K > #LLM-picks is comparable with vector
    # retrievers (T03/T04) that always emit ~100 candidates.
    #   tier 1: leaves of selected_chapter_ids (concentrated — the LLM is
    #           good at picking chapters even when it picks too few articles
    #           inside them)
    #   tier 2: leaves of selected_law_ids (broader fallback to fill the tail)
    # The local PDF 1804 ablation (data/.../experiments/REPORT.md) showed this
    # lifts R@10 from 0.124 to 0.203 (+63% rel) and R@100 from 0.247 to 0.393
    # (+59% rel) versus the previous law-only padding. Fully deterministic
    # post-processing — no extra LLM calls.
    # If no laws were selected (degenerate fallback), pad from every law in
    # the tree so the metric still has something to score.
    if state.cfg.pad_to_k and len(ranked) < state.cfg.pad_to_k:
        seen_ids = {nid for nid, _ in ranked}
        source_lists = (
            list(state.selected_chapter_ids),
            state.selected_law_ids or [
                law.node_id for law in _laws_in_tree(state.tree)
            ],
        )
        for source_ids in source_lists:
            for src_id in source_ids:
                src_node = find_node(state.tree, src_id)
                if src_node is None:
                    continue
                for leaf in iter_leaves(src_node):
                    if leaf.node_id in seen_ids:
                        continue
                    ranked.append((leaf.node_id, 0.0))
                    seen_ids.add(leaf.node_id)
                    if len(ranked) >= state.cfg.pad_to_k:
                        break
                if len(ranked) >= state.cfg.pad_to_k:
                    break
            if len(ranked) >= state.cfg.pad_to_k:
                break

    state.final_ranked = ranked
    return state
