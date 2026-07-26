"""Tier (b) — CRAG x ReAct corrective navigator (the "new" architecture).

Wraps the single-pass descent in a corrective outer loop WITHOUT modifying
react_navigator.py (additive constraint). The inner descent here is a faithful
re-implementation of react_navigator.navigate()'s body, parameterised so it can be
SEEDED from an arbitrary frontier and SHARE a visited-set / NavState across rounds
(navigate() always starts from [root] and resets state, so it can't be looped).
Round 1 with seed=[root] is byte-for-byte the old behaviour — so the smoke harness
compares old-vs-new on equal footing.

Outer loop (refines the brief):
  seed = [root]
  for round in 1..max_rounds:
      descend(seed)                      # inner ReAct pass, shared visited/state
      collect pruned sections at visited nodes
      reach = leaf-children of visited nodes
      stop if reach stopped growing (NOT an LLM "sufficient?" grader -- that 8B
        grader was 98% false-positive on Arm-2B; we use reach growth + force >=1
        corrective round instead)
      seed = rank_pruned_sections(query, pruned)[:m]   # the DISCRIMINATOR the
        first pass lacked -- without it, re-seeding all pruned branches = an
        exhaustive scan that defeats the tree, or re-seeding on the same titles
        that already pruned them = inert.
  return rerank(union reached pool) + pad

Two re-seed strategies:
  "ranked" (default) one LLM call ranks the pruned sections by relevance, top-m.
  "all"              re-seed first m pruned sections (document order), no LLM -- the
                     brute-force control the ranked mode must beat.

Needs Ollama only for live runs (LlamaClient is lazy + mockable); the control flow
is exercised by test_corrective_mock.py with a scripted MockLLM, no model.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))           # experiments/arm2c
from navigator_tools import (                            # noqa: E402
    DeepTree, NavState, clean_title, expand, preflight, render_children,
)
# Reuse the exact prompts / parsing / pool+rerank machinery so round 1 == navigate().
from react_navigator import (                            # noqa: E402
    SYSTEM_PROMPT, USER_TEMPLATE, StepTrace, _extract, _thought,
    _rerank, _to_bsard_and_pad,
)

# Prompt for ranking the pruned sections to re-seed (sections, not articles).
RESEED_SYSTEM = (
    "Vous êtes un assistant juridique. La première exploration de l'arbre du texte "
    "de loi n'a pas trouvé assez d'articles pertinents. On vous donne la question et "
    "une liste numérotée de SECTIONS qui ont été écartées lors de cette première "
    "exploration. Classez ces sections, de la PLUS à la MOINS susceptible de contenir "
    "des articles pertinents — il faut décider lesquelles ré-explorer. Préférez "
    'l\'inclusion. Répondez STRICTEMENT par un seul objet JSON : {"ranking": [numéros]}'
)
RESEED_USER = (
    "Question : {query}\n\nSections écartées à ré-explorer :\n{menu}\n\n"
    'Classez-les. JSON : {{"ranking": [numéros]}}'
)


@dataclass
class RoundInfo:
    round: int
    seed_size: int
    nodes_this_round: int
    visited_total: int
    reach_total: int
    reseed_call: bool = False


@dataclass
class CorrectiveResult:
    query: str
    selected_bsard_ids: list[int]
    ranked_bsard_ids: list[int]
    ranked_bsard_ids_prererank: list[int] | None
    nodes_visited: int
    llm_calls: int
    rounds: list[RoundInfo] = field(default_factory=list)
    steps: list[StepTrace] = field(default_factory=list)
    exit_reason: str = "rounds_done"
    # ── round-1 snapshot = the matched "old single-pass" baseline. Round 1 of the
    # loop IS navigate() (verified), so comparing round1_* vs the top-level fields is
    # a CONTROLLED ablation off ONE navigation — no run-to-run LLM-noise confound and
    # half the cost of navigating twice. Reach is derived from round1_visit_order.
    round1_ranked_bsard_ids: list[int] = field(default_factory=list)
    round1_selected_bsard_ids: list[int] = field(default_factory=list)
    round1_visit_order: list[str] = field(default_factory=list)
    # reached-pool leaf node_ids (the candidate set the rerank operates on), for an
    # external reranker (e.g. the e5 embedding rerank, E2). Final loop vs round-1.
    pool_node_ids: list[str] = field(default_factory=list)
    round1_pool_node_ids: list[str] = field(default_factory=list)


# ── inner descent: a faithful, seedable copy of navigate()'s body ────────────

def _descend(query, tree, llm, state: NavState, seed: list[str], *, mode: str,
             max_nodes: int, max_branch: int, summary_chars: int):
    """Run one ReAct descent from `seed`, mutating shared `state`. Returns
    (steps, llm_calls, budget_hit). Mirrors react_navigator.navigate() exactly:
    DFS frontier, one JSON call per node, preflight-gated descend/select."""
    frontier = list(seed)
    queued = set(seed)
    steps: list[StepTrace] = []
    llm_calls = 0
    budget_hit = False
    while frontier:
        if state.steps >= max_nodes:
            budget_hit = True
            break
        node_id = frontier.pop()                 # DFS (stack)
        if node_id in state.visited:
            continue
        state.visited.add(node_id)
        state.steps += 1
        node = tree.node(node_id)
        views = expand(tree, node_id, mode=mode)
        menu, num2id = render_children(views, mode=mode, summary_chars=summary_chars)
        prompt = USER_TEMPLATE.format(query=query, title=node.title, menu=menu)
        st = StepTrace(node_id=node_id, title=node.title, n_children=len(views))
        t0 = time.perf_counter()
        try:
            data = llm.generate_json(prompt, system_prompt=SYSTEM_PROMPT)
        except Exception as exc:
            st.parse_ok = False
            st.thought = f"<parse_fail: {exc}>"
            st.latency_ms = (time.perf_counter() - t0) * 1000
            llm_calls += 1
            steps.append(st)
            continue
        st.latency_ms = (time.perf_counter() - t0) * 1000
        llm_calls += 1
        st.thought = _thought(data)
        new_kids = []
        for n in _extract(data, "sections"):
            cid = num2id.get(n)
            if cid and preflight(tree, "expand", cid) is None \
                    and cid not in queued and cid not in state.visited:
                new_kids.append(cid)
                queued.add(cid)
                st.sections.append(cid)
        for cid in reversed(new_kids[:max_branch]):
            frontier.append(cid)
        for n in _extract(data, "articles"):
            cid = num2id.get(n)
            if cid and preflight(tree, "select", cid) is None:
                state.select(cid)
                st.articles.append(cid)
        steps.append(st)
    return steps, llm_calls, budget_hit


# ── pruned-section collection + reached pool ─────────────────────────────────

def _collect_pruned(tree: DeepTree, state: NavState, pruned: dict[str, str]) -> None:
    """Pruned sections = internal children of visited nodes never themselves
    visited. Accumulates across rounds into `pruned` (node_id -> title)."""
    for vid in state.visited:
        node = tree.node(vid)
        if not node:
            continue
        for c in node.sub_nodes:
            if (not c.is_leaf) and c.node_id not in state.visited:
                pruned[c.node_id] = c.title


def _reach_pool(tree: DeepTree, state: NavState,
                visit_order: list[str] | None = None) -> list[str]:
    """Unique leaf node_ids reached: selected first, then leaf children of every
    visited node. Same construction as navigate()'s pool.

    `visit_order` MUST be passed when the result feeds the ranking/padding (the
    final pool): it preserves visit order, exactly like navigate()'s
    `expanded_order`. Iterating the `state.visited` SET instead gives hash-arbitrary
    order, which scrambles the rerank head[:cap] + the padding tail and can push
    already-reached gold out of the top-10 (observed: Civil q243 0.333->0.000).
    Omit it only for the per-round reach COUNT, where order is irrelevant."""
    order = visit_order if visit_order is not None else list(state.visited)
    pool, seen = [], set()
    for nid in state.selected:
        if tree.is_leaf(nid) and nid not in seen:
            seen.add(nid); pool.append(nid)
    for vid in order:
        node = tree.node(vid)
        if node:
            for c in node.sub_nodes:
                if c.is_leaf and c.node_id not in seen:
                    seen.add(c.node_id); pool.append(c.node_id)
    return pool


def _rank_pruned(query, cand: list[str], tree, llm, mode: str, m: int,
                 strategy: str) -> tuple[list[str], bool]:
    """Choose up to m pruned sections to re-seed. Returns (seed, made_llm_call)."""
    if not cand:
        return [], False
    if strategy == "all":
        return cand[:m], False
    # "ranked": one inclusion-biased LLM call orders the pruned sections.
    head = cand[:max(m * 6, 30)]                 # cap prompt size
    lines, num2node = [], {}
    for i, nid in enumerate(head, 1):
        n = tree.node(nid)
        num2node[i] = nid
        meta = n.metadata or {}
        summ = (meta.get("level_summary_en") if mode == "enriched" else "") or ""
        cnt = (meta.get("article_count") if meta else None)
        tail = f" — (EN) {summ[:140]}" if summ else ""
        cntp = f" ({cnt} art.)" if cnt else ""
        lines.append(f"  [{i}] {clean_title(n.title)}{cntp}{tail}")
    try:
        data = llm.generate_json(RESEED_USER.format(query=query, menu="\n".join(lines)),
                                 system_prompt=RESEED_SYSTEM)
    except Exception:
        return head[:m], True
    out, used = [], set()
    for num in _extract(data, "ranking"):
        nid = num2node.get(num)
        if nid and nid not in used:
            used.add(nid); out.append(nid)
    for nid in head:                             # ranked-omitted keep order
        if nid not in used:
            out.append(nid)
    return out[:m], True


# ── the corrective loop ──────────────────────────────────────────────────────

def navigate_corrective(query: str, tree: DeepTree, llm, *, mode: str = "enriched",
                        max_nodes: int = 40, max_branch: int = 5, summary_chars: int = 160,
                        rerank: bool = True, rerank_pool: int = 60,
                        max_rounds: int = 2, reseed_m: int = 5,
                        reseed_strategy: str = "ranked",
                        min_reach_growth: int = 1) -> CorrectiveResult:
    state = NavState()
    all_steps: list[StepTrace] = []
    pruned: dict[str, str] = {}
    rounds: list[RoundInfo] = []
    llm_calls = 0
    exit_reason = "rounds_done"
    seed = [tree.root.node_id]
    reseeded: set[str] = set()
    prev_reach = -1
    r1_visit_order: list[str] = []        # round-1 snapshot (the matched baseline)
    r1_selected: list[str] = []

    for rnd in range(1, max_rounds + 1):
        steps, calls, budget_hit = _descend(
            query, tree, llm, state, seed, mode=mode, max_nodes=max_nodes,
            max_branch=max_branch, summary_chars=summary_chars)
        all_steps += steps
        llm_calls += calls
        if rnd == 1:                       # freeze the round-1 state = "old navigate()"
            r1_visit_order = [s.node_id for s in all_steps]
            r1_selected = list(state.selected)
        _collect_pruned(tree, state, pruned)
        reach = _reach_pool(tree, state)
        info = RoundInfo(round=rnd, seed_size=len(seed),
                         nodes_this_round=len(steps), visited_total=len(state.visited),
                         reach_total=len(reach))
        # termination — reach growth (force >=1 corrective round: never stop after rnd 1)
        if budget_hit:
            rounds.append(info); exit_reason = "budget"; break
        if rnd > 1 and (len(reach) - prev_reach) < min_reach_growth:
            rounds.append(info); exit_reason = "no_growth"; break
        prev_reach = len(reach)
        if rnd == max_rounds:
            rounds.append(info); exit_reason = "rounds_done"; break
        # corrective re-seed
        cand = [nid for nid in pruned
                if nid not in state.visited and nid not in reseeded]
        new_seed, made_call = _rank_pruned(query, cand, tree, llm, mode, reseed_m,
                                            reseed_strategy)
        if made_call:
            llm_calls += 1
            info.reseed_call = True
        rounds.append(info)
        if not new_seed:
            exit_reason = "no_pruned_left"; break
        reseeded.update(new_seed)
        seed = new_seed

    # final pool + rerank (identical tail to navigate()): build in VISIT order so
    # round-1-only is byte-for-byte navigate(), and re-navigated leaves append after
    # the round-1 head rather than scrambling it.
    pool = _reach_pool(tree, state, visit_order=[s.node_id for s in all_steps])
    ranked_pre = _to_bsard_and_pad(pool, tree, k=100)
    if rerank and pool:
        reranked = _rerank(query, pool, tree, llm, mode, rerank_pool)
        llm_calls += 1
        ranked = _to_bsard_and_pad(reranked, tree, k=100)
    else:
        ranked = ranked_pre

    # ── round-1 baseline (matched ablation) ──────────────────────────────────
    # Same pool+rerank tail over ONLY the round-1 trace = what the old single pass
    # produced this very navigation. If just one round ran, it equals the final.
    r1_state = NavState(); r1_state.selected = r1_selected
    r1_state.visited = set(r1_visit_order)
    r1_pool = _reach_pool(tree, r1_state, visit_order=r1_visit_order)
    if len(rounds) <= 1:
        round1_ranked = ranked
    else:
        if rerank and r1_pool:
            r1_reranked = _rerank(query, r1_pool, tree, llm, mode, rerank_pool)
            llm_calls += 1
            round1_ranked = _to_bsard_and_pad(r1_reranked, tree, k=100)
        else:
            round1_ranked = _to_bsard_and_pad(r1_pool, tree, k=100)

    return CorrectiveResult(
        query=query, selected_bsard_ids=state.selected_bsard_ids(tree),
        ranked_bsard_ids=ranked,
        ranked_bsard_ids_prererank=(ranked_pre if rerank else None),
        nodes_visited=state.steps, llm_calls=llm_calls, rounds=rounds,
        steps=all_steps, exit_reason=exit_reason,
        round1_ranked_bsard_ids=round1_ranked,
        round1_selected_bsard_ids=r1_state.selected_bsard_ids(tree),
        round1_visit_order=r1_visit_order,
        pool_node_ids=pool, round1_pool_node_ids=r1_pool)
