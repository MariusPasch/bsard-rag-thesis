"""Arm-2C ReAct navigator: variable-depth frontier descent over the deep tree.

Control flow (PageIndex-faithful, call-efficient — ~1 LLM call per visited node,
not per move, because LLaMA-8B calls are ~tens of seconds each):

  frontier = [root]
  while frontier and budget left:
      node = frontier.pop(0)
      show its numbered children to the LLM (bare or enriched)
      LLM returns {"sections": [...], "articles": [...]}   # subset to act on
      sections -> enqueue (descend later);  articles -> select (keep)
  rank = selected (in selection order) + Arm-2B-style padding to k

Robustness (ported from RQ1 Tier42): JSON mode w/ retry (shared.LLMClient),
out-of-range / wrong-kind picks are dropped via preflight (never crash), a
visited-set repeat guard, and a hard node budget. Per-query trace logged for the
eval analysis (path, picks, finish reason), à la Tier42 agent_loop_stats.

bare = controlled-vs-Arm-2B core; enriched (native-EN summaries) behind --mode.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))
from navigator_tools import (  # noqa: E402
    DeepTree, NavState, clean_title, expand, preflight, render_children,
)

import json as _json  # noqa: E402
import re as _re      # noqa: E402

_JSON_FENCE = _re.compile(r"```(?:json)?\s*([\s\S]*?)```", _re.IGNORECASE)


class LlamaClient:
    """Minimal self-contained Ollama/LLaMA-3.1-8B JSON client.

    Replicates shared.llm.LLMClient (temp 0, num_ctx 16384, fence-strip + 1 retry)
    without importing the `shared` package (whose __init__ eagerly pulls rank_bm25,
    absent in the vectorless T05 venv). `ollama` is imported lazily so this module
    loads — and the loop is testable with a mock LLM — even where Ollama is absent.
    """

    def __init__(self, model: str = "llama3.1:8b", base_url: str = "http://localhost:11434",
                 temperature: float = 0.0, num_ctx: int = 16384,
                 timeout: int = 120, num_predict: int = 1024) -> None:
        import ollama  # lazy: only needed for live runs
        # timeout bounds every call (a wedged/slow Ollama can't freeze the run forever);
        # num_predict caps generation so a runaway loop can't generate to the full context
        # (our JSON answers are <200 tokens — 1024 is ample headroom).
        self._client = ollama.Client(host=base_url, timeout=timeout)
        self._model = model
        self._opts = {"temperature": temperature, "num_ctx": num_ctx,
                      "num_predict": num_predict}

    def generate(self, prompt: str, system_prompt: str | None = None,
                 fmt: str | None = None) -> str:
        msgs = ([{"role": "system", "content": system_prompt}] if system_prompt else [])
        msgs.append({"role": "user", "content": prompt})
        kwargs = {"options": self._opts}
        if fmt:                       # Ollama constrains output to valid JSON when fmt="json"
            kwargs["format"] = fmt
        # keep_alive=-1 pins the model in VRAM for the whole run on EVERY call, so it
        # never idle-unloads (the "Stopping..." wedge that kept stalling the run). No
        # sudo / service config needed — the client guarantees it.
        r = self._client.chat(model=self._model, messages=msgs, keep_alive=-1, **kwargs)
        return r["message"]["content"]

    def generate_json(self, prompt: str, system_prompt: str | None = None) -> dict:
        # Retry up to 3x on EITHER a timeout/connection error (Ollama hiccup) OR a JSON
        # parse failure. format="json" forces valid JSON; num_predict caps runaway output.
        last_exc: Exception | None = None
        for _ in range(3):
            try:
                text = self.generate(prompt, system_prompt, fmt="json").strip()
            except Exception as exc:          # timeout / connection — retry
                last_exc = exc
                continue
            m = _JSON_FENCE.search(text)
            if m:
                text = m.group(1).strip()
            try:
                return _json.loads(text)
            except _json.JSONDecodeError as exc:
                last_exc = exc
                continue
        raise last_exc if last_exc else RuntimeError("generate_json failed")

# Inclusion-biased prompt — mirrors T05/Arm-2B fix #4: the conservative
# "uniquement ce qui est pertinent" framing made the 8B model return empty
# selections 65-84% of the time, collapsing recall to ~0. Preferring inclusion
# (let padding/ranking do the fine sort) is what lifted Arm-2B's recall.
SYSTEM_PROMPT = (
    "Vous êtes un assistant juridique qui navigue dans la structure hiérarchique "
    "d'un texte de loi belge afin de trouver les articles pertinents à une question. "
    "À chaque étape, on vous montre une section et ses sous-éléments numérotés "
    "(sous-sections et/ou articles).\n"
    "PRÉFÉREZ L'INCLUSION : explorez toute sous-section dont le thème pourrait, même "
    "partiellement, concerner la question, et retenez tout article qui paraît même "
    "partiellement lié. Mieux vaut explorer une section de trop que manquer les bons "
    "articles. Ne renvoyez des listes vides que si vraiment aucun élément n'a de lien "
    "thématique plausible avec la question.\n"
    "Les libellés de structure sont en français ; lorsqu'un résumé « (EN) » apparaît, "
    "il est en anglais — raisonnez à travers les deux langues.\n"
    "Répondez STRICTEMENT par UN SEUL objet JSON (pas un tableau d'objets), sans texte "
    'autour :\n{"thought": "...", "sections": [numéros], "articles": [numéros]}'
)

USER_TEMPLATE = (
    "Question : {query}\n\n"
    "Section actuelle : {title}\n"
    "Sous-éléments :\n{menu}\n\n"
    "Quels sous-éléments pourraient concerner la question ? Préférez l'inclusion. "
    'Répondez par un seul objet JSON : {{"thought": "...", "sections": [numéros de '
    'sous-sections à explorer], "articles": [numéros d\'articles à retenir]}}'
)

# Optional final re-rank: navigation surfaces the right SECTIONS (recall@100 0.515)
# but leaves gold ranked low (recall@10 0.214). One call orders the navigated pool
# by relevance, converting reachable gold into the top-k.
RERANK_SYSTEM = (
    "Vous êtes un assistant juridique. On vous donne une question et une liste "
    "numérotée d'articles candidats avec un court résumé. Classez les articles du "
    "PLUS au MOINS pertinent pour répondre à la question. Préférez l'inclusion : "
    "placez en tête tous les articles plausiblement pertinents. Répondez STRICTEMENT "
    'par un seul objet JSON : {"ranking": [numéros, du plus au moins pertinent]}'
)
RERANK_USER = (
    "Question : {query}\n\nArticles candidats :\n{menu}\n\n"
    'Classez-les par pertinence. JSON : {{"ranking": [numéros]}}'
)


@dataclass
class StepTrace:
    node_id: str
    title: str
    n_children: int
    sections: list[str] = field(default_factory=list)
    articles: list[str] = field(default_factory=list)
    thought: str = ""
    parse_ok: bool = True
    latency_ms: float = 0.0


@dataclass
class NavResult:
    query: str
    selected_bsard_ids: list[int]
    ranked_bsard_ids: list[int]          # final ranking (after rerank if enabled), to k
    nodes_visited: int
    llm_calls: int
    exit_reason: str                     # "frontier_empty" | "budget"
    ranked_bsard_ids_prererank: list[int] | None = None   # ranking BEFORE rerank (None if rerank off)
    steps: list[StepTrace] = field(default_factory=list)


def _coerce_nums(v) -> list[int]:
    out = []
    if isinstance(v, list):
        for x in v:
            try:
                out.append(int(x))
            except (TypeError, ValueError):
                continue
    return out


def _extract(data, key: str) -> list[int]:
    """Pull `key` as ints, tolerating BOTH {key:[...]} and [{key:[...]}, ...].
    Mirrors T05/Arm-2B fix #3: the 8B model sometimes wraps its answer in a list,
    which silently zeroed selections (and here would crash data.get)."""
    if isinstance(data, dict):
        return _coerce_nums(data.get(key))
    if isinstance(data, list):
        out: list[int] = []
        for item in data:
            if isinstance(item, dict):
                out += _coerce_nums(item.get(key))
        return out
    return []


def _thought(data) -> str:
    return str(data.get("thought", ""))[:200] if isinstance(data, dict) else ""


def navigate(query: str, tree: DeepTree, llm, *, mode: str = "bare",
             max_nodes: int = 40, max_branch: int = 5, summary_chars: int = 160,
             rerank: bool = False, rerank_pool: int = 60) -> NavResult:
    # max_branch caps sections descended per node (T05's max_chapters_per_law=5):
    # the inclusion prompt proposes generously, but uncapped descent would explode
    # the frontier and the LLM-call count (1 call per visited node). max_nodes is
    # the global budget. Article selection is uncapped (it costs no extra call).
    state = NavState()
    frontier: list[str] = [tree.root.node_id]
    queued = {tree.root.node_id}
    steps: list[StepTrace] = []
    llm_calls = 0
    exit_reason = "frontier_empty"

    while frontier:
        if state.steps >= max_nodes:
            exit_reason = "budget"
            break
        node_id = frontier.pop()        # DFS (stack): reach leaf-parent sections fast,
                                         # so their articles enter the padded ranking
                                         # within the node budget (BFS would burn the
                                         # budget on shallow levels — esp. now that the
                                         # inclusion bias selects more branches).
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
        except Exception as exc:               # parse/connection failure -> prune node
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
        for n in _extract(data, "sections"):                # descend
            cid = num2id.get(n)
            if cid and preflight(tree, "expand", cid) is None \
                    and cid not in queued and cid not in state.visited:
                new_kids.append(cid)
                queued.add(cid)
                st.sections.append(cid)
        for cid in reversed(new_kids[:max_branch]):   # cap branches; first pops first (DFS)
            frontier.append(cid)
        for n in _extract(data, "articles"):                # keep
            cid = num2id.get(n)
            if cid and preflight(tree, "select", cid) is None:
                state.select(cid)
                st.articles.append(cid)
        steps.append(st)

    expanded_order = [s.node_id for s in steps]              # visit order = priority
    # navigated pool (ordered unique leaf node_ids): selected first, then leaf
    # children of every expanded section — the recall@100-bearing set.
    pool: list[str] = []
    seen_n: set[str] = set()
    for nid in state.selected:
        if tree.is_leaf(nid) and nid not in seen_n:
            seen_n.add(nid); pool.append(nid)
    for nid in expanded_order:
        n = tree.node(nid)
        if n:
            for c in n.sub_nodes:
                if c.is_leaf and c.node_id not in seen_n:
                    seen_n.add(c.node_id); pool.append(c.node_id)

    selected = state.selected_bsard_ids(tree)
    ranked_pre = _to_bsard_and_pad(pool, tree, k=100)     # ranking BEFORE rerank
    if rerank and pool:                  # one extra call: order the pool by relevance
        reranked = _rerank(query, pool, tree, llm, mode, rerank_pool)
        llm_calls += 1
        ranked = _to_bsard_and_pad(reranked, tree, k=100)
    else:
        ranked = ranked_pre
    return NavResult(query=query, selected_bsard_ids=selected, ranked_bsard_ids=ranked,
                     nodes_visited=state.steps, llm_calls=llm_calls, exit_reason=exit_reason,
                     ranked_bsard_ids_prererank=(ranked_pre if rerank else None), steps=steps)


def _rerank(query: str, pool_nodes: list[str], tree: DeepTree, llm, mode: str,
            cap: int) -> list[str]:
    """LLM orders the navigated pool by relevance (one call). Caps the prompt at
    `cap` candidates; unranked/overflow keep their navigation order at the tail.
    Returns original order on any failure (json-mode makes that rare)."""
    head, tail = pool_nodes[:cap], pool_nodes[cap:]
    lines, num2node = [], {}
    for i, nid in enumerate(head, 1):
        n = tree.node(nid)
        num2node[i] = nid
        meta = n.metadata or {}
        summ = (meta.get("content_summary_en")
                if mode == "enriched" and meta.get("content_summary_en")
                else (n.summary or ""))
        lines.append(f"  [{i}] {clean_title(n.title)} — {summ[:160]}")
    try:
        data = llm.generate_json(RERANK_USER.format(query=query, menu="\n".join(lines)),
                                 system_prompt=RERANK_SYSTEM)
    except Exception:
        return pool_nodes
    used, out = set(), []
    for num in _extract(data, "ranking"):
        nid = num2node.get(num)
        if nid and nid not in used:
            used.add(nid); out.append(nid)
    for nid in head:                     # ranked-omitted head items keep nav order
        if nid not in used:
            out.append(nid)
    return out + tail


def _to_bsard_and_pad(pool_nodes: list[str], tree: DeepTree, k: int) -> list[int]:
    """Pool (node_ids) -> bsard_ids (dedup), then breadth-fill over remaining
    leaves (document order) to reach k."""
    ranked: list[int] = []
    seen = set()

    def add(bid):
        if bid is not None and bid not in seen:
            seen.add(bid); ranked.append(bid)

    for nid in pool_nodes:
        n = tree.node(nid)
        if n:
            add(n.bsard_id)

    def walk(n):
        if n.is_leaf:
            add(n.bsard_id); return
        for c in n.sub_nodes:
            walk(c)
    walk(tree.root)
    return ranked[:k]


# ── smoke: one real query on Code Civil ──────────────────────────────────────


def _smoke(mode: str, max_nodes: int) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    tree = DeepTree.load(_HERE.parent / "data" / "1804_03_21_1804032150" / "deep_tree.json")
    # T05 q183: étranger reconnaissant son enfant; gold incl. bsard 930/940/941...
    query = ("Je vis avec ma famille en Belgique. J'ai un enfant en Belgique. "
             "Je suis étranger. Quels documents dois-je déposer à la commune "
             "pour reconnaître mon enfant ?")
    gold = {930, 940, 941, 942}
    print(f"mode={mode} max_nodes={max_nodes}\nquery: {query}\n")
    llm = LlamaClient(model="llama3.1:8b", temperature=0.0, num_ctx=16384)
    t0 = time.perf_counter()
    res = navigate(query, tree, llm, mode=mode, max_nodes=max_nodes)
    secs = time.perf_counter() - t0
    print(f"--- visited {res.nodes_visited} nodes, {res.llm_calls} LLM calls, "
          f"exit={res.exit_reason}, {secs:.0f}s ---")
    for s in res.steps:
        picks = f"sections={len(s.sections)} articles={len(s.articles)}"
        print(f"  [{s.node_id[:32]:32}] {s.title[:34]:34} {picks}  ({s.latency_ms:.0f}ms)"
              + ("" if s.parse_ok else "  PARSE_FAIL"))
    print(f"\nselected bsard_ids ({len(res.selected_bsard_ids)}): {res.selected_bsard_ids}")
    hit = [b for b in res.selected_bsard_ids if b in gold]
    print(f"gold={sorted(gold)}  selected∩gold={hit}  "
          f"R@10={len(set(res.ranked_bsard_ids[:10]) & gold)}/{len(gold)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["bare", "enriched"], default="bare")
    ap.add_argument("--max-nodes", type=int, default=40)
    args = ap.parse_args()
    return _smoke(args.mode, args.max_nodes)


if __name__ == "__main__":
    sys.exit(main())
