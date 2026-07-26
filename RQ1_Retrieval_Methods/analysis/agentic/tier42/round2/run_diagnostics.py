"""
Round-2 cheap diagnostics for Tier 4.2 ReAct (§15.1.3, §15.1.4, §15.1.5).

Inputs (offline)
----------------
- Trace files:
    output/results/agentic/ReAct/react_bm25_test_traces.json
    output/results/agentic/ReAct/react_hybrid_rrf_k60_test_traces.json
- Result files (for stratified comparison and cached first-stage runs):
    output/results/agentic/ReAct/react_bm25_test.json
    output/results/agentic/ReAct/react_hybrid_rrf_k60_test.json
    output/results/hybrid/hybrid_rrf_k60_test.json
    output/results/agentic/llm_judge/llm_rerank/llm_rerank_binary_top50_hybrid_rrf_k60_test.json
- evaluation/data/query_strata.json (loaded via evaluation.stratify.load_strata)

Outputs
-------
- output/results/agentic/ReAct/round2/diagnostics_15_1_3_per_stratum_delta.json
- output/results/agentic/ReAct/round2/diagnostics_15_1_4_gold_presence.json
- output/results/agentic/ReAct/round2/diagnostics_15_1_5_d1_on_top50.json
- output/cache/bm25_tuned_top100_test.json (first-run only — needs the corpus)

Notes
-----
15.1.5 issues LLM_JUDGE_BINARY_PROMPT calls against local Ollama. To avoid a
multi-hour run, the script accepts --d1-n-queries N (default 10) for a smoke
sample; the full Azure run sets N=222. The output JSON records which mode it
ran in so the analysis notebook can disambiguate.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

ROUND2_DIR     = ROOT / "output" / "results" / "agentic" / "ReAct" / "round2"
CACHE_DIR      = ROOT / "output" / "cache"
REACT_DIR      = ROOT / "output" / "results" / "agentic" / "ReAct"
HYBRID_FILE    = ROOT / "output" / "results" / "hybrid" / "hybrid_rrf_k60_test.json"
T40_HYBRID_FILE = (
    ROOT / "output" / "results" / "agentic" / "llm_judge" / "llm_rerank"
    / "llm_rerank_binary_top50_hybrid_rrf_k60_test.json"
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _trec_run_to_top_ids(trec_run: dict, qid: str | int, k: int) -> list[int]:
    """Pull top-k article IDs from a TREC run dict (sorted by score desc)."""
    docs = trec_run.get(str(qid), {}) or {}
    if not docs:
        return []
    ranked = sorted(docs.items(), key=lambda x: -x[1])
    return [int(aid) for aid, _ in ranked[:k]]


def _r_at_k(predicted: list[int], gold: list[int], k: int) -> float:
    if not gold:
        return 0.0
    pred_set = set(predicted[:k])
    gold_set = set(gold)
    return len(pred_set & gold_set) / len(gold_set)


# ---------------------------------------------------------------------------
# 15.1.3 — Per-stratum R@10 delta vs first stage
# ---------------------------------------------------------------------------

def diag_15_1_3() -> dict:
    """
    Compare the ReAct agent (R@10 stratified) to the first stage alone (R@10
    stratified) along single_article/multi_article and lexically_aligned/
    semantically_paraphrased axes. Per-query delta histograms.
    """
    from evaluation.stratify import load_strata

    strata = load_strata()
    if strata is None:
        raise FileNotFoundError("evaluation/data/query_strata.json missing")

    react_hybrid = _load_json(REACT_DIR / "react_hybrid_rrf_k60_test.json")
    hybrid_first = _load_json(HYBRID_FILE)
    react_bm25   = _load_json(REACT_DIR / "react_bm25_test.json")

    qrels = react_hybrid["_trec_qrels"]
    qids  = sorted(qrels.keys(), key=int)

    # Gold IDs per query
    gold = {qid: list(qrels[qid].keys()) for qid in qids}
    gold = {qid: [int(g) for g in g_list] for qid, g_list in gold.items()}

    # First-stage hybrid top-10 from cached run
    hybrid_top10 = {
        qid: _trec_run_to_top_ids(hybrid_first["_trec_run"], qid, 10)
        for qid in qids
    }

    # Agent top-10 from each variant's _trec_run
    agent_top10_hybrid = {
        qid: _trec_run_to_top_ids(react_hybrid["_trec_run"], qid, 10)
        for qid in qids
    }
    agent_top10_bm25 = {
        qid: _trec_run_to_top_ids(react_bm25["_trec_run"], qid, 10)
        for qid in qids
    }

    def _per_query_r10(top_map: dict[str, list[int]]) -> dict[str, float]:
        return {qid: _r_at_k(top_map[qid], gold[qid], 10) for qid in qids}

    r10 = {
        "agent_hybrid":  _per_query_r10(agent_top10_hybrid),
        "first_hybrid":  _per_query_r10(hybrid_top10),
        "agent_bm25":    _per_query_r10(agent_top10_bm25),
    }

    # We do not have a cached BM25 first-stage run; the hybrid case is the
    # primary comparison anyway (the agent's hybrid vs first hybrid gap is
    # the largest per §13).
    def _aggregate_by(field: str, value: str, source: str) -> dict:
        sel = [
            qid for qid in qids
            if strata.get(int(qid), {}).get(field) == value
        ]
        if not sel:
            return {"n": 0, "mean_r10": None}
        vals = [r10[source][qid] for qid in sel]
        return {
            "n": len(sel),
            "mean_r10": sum(vals) / len(vals),
            "qids": sel,
        }

    # Build delta tables per stratum: (agent - first_stage) on hybrid backbone
    strata_axes = {
        "article_count": ["single_article", "multi_article"],
        "lex_align":     ["lexically_aligned", "semantically_paraphrased"],
    }

    stratum_results = {}
    for axis, levels in strata_axes.items():
        stratum_results[axis] = {}
        for lvl in levels:
            agent  = _aggregate_by(axis, lvl, "agent_hybrid")
            first  = _aggregate_by(axis, lvl, "first_hybrid")
            agent_b = _aggregate_by(axis, lvl, "agent_bm25")
            if agent["n"] == 0:
                continue
            stratum_results[axis][lvl] = {
                "n_queries":            agent["n"],
                "agent_hybrid_R10":     agent["mean_r10"],
                "first_hybrid_R10":     first["mean_r10"],
                "agent_bm25_R10":       agent_b["mean_r10"],
                "delta_agent_minus_first_hybrid": agent["mean_r10"] - first["mean_r10"],
            }

    # Per-query deltas (for the analysis notebook)
    per_query_delta = {
        qid: {
            "agent_hybrid_R10": r10["agent_hybrid"][qid],
            "first_hybrid_R10": r10["first_hybrid"][qid],
            "delta":            r10["agent_hybrid"][qid] - r10["first_hybrid"][qid],
            "stratum": {
                "article_count": strata.get(int(qid), {}).get("article_count"),
                "lex_align":     strata.get(int(qid), {}).get("lex_align"),
            },
        }
        for qid in qids
    }

    # Niche detection: where does the agent beat (or match) the first stage?
    n_total       = len(qids)
    n_agent_wins  = sum(1 for v in per_query_delta.values() if v["delta"] > 0)
    n_agent_ties  = sum(1 for v in per_query_delta.values() if v["delta"] == 0)
    n_agent_loses = sum(1 for v in per_query_delta.values() if v["delta"] < 0)

    return {
        "diagnostic": "15.1.3 per-stratum R@10 delta vs first stage (hybrid)",
        "n_queries": n_total,
        "overall": {
            "agent_hybrid_R10":      sum(r10["agent_hybrid"].values()) / n_total,
            "first_hybrid_R10":      sum(r10["first_hybrid"].values()) / n_total,
            "agent_bm25_R10":        sum(r10["agent_bm25"].values()) / n_total,
            "wins/ties/losses":      [n_agent_wins, n_agent_ties, n_agent_loses],
        },
        "stratified": stratum_results,
        "per_query":  per_query_delta,
    }


# ---------------------------------------------------------------------------
# 15.1.4 — Gold-presence audit
# ---------------------------------------------------------------------------

def _build_or_load_bm25_top100() -> dict[str, list[int]]:
    """
    Cache the BM25 first-stage top-100 IDs per test query at
    output/cache/bm25_tuned_top100_test.json. Build only if missing.
    Returns a dict mapping str(qid) -> list[int] of top-100 article IDs.
    """
    cache_path = CACHE_DIR / "bm25_tuned_top100_test.json"
    if cache_path.exists():
        return _load_json(cache_path)

    print("  Building BM25 first-stage top-100 cache (one-time)...", flush=True)
    import pandas as pd
    from evaluation.split import load_questions
    from retrieval.sparse import BM25Retriever

    df = pd.read_parquet(ROOT / "output" / "bsard_articles_dedup.parquet")
    bm25 = BM25Retriever(
        df,
        normalization="lemmatize",
        field_weighting="text_only",
        variant="okapi",
        k1=1.5,
        b=0.25,
    )
    questions = load_questions(subset="test")
    cache: dict[str, list[int]] = {}
    t0 = time.perf_counter()
    for q in questions:
        ids, _ = bm25.retrieve(q["question_text"], top_k=100)
        cache[str(q["question_id"])] = list(map(int, ids))
    dt = time.perf_counter() - t0
    print(f"  BM25 222 queries done in {dt:.1f}s; caching to {cache_path.name}")
    _save_json(cache_path, cache)
    return cache


def diag_15_1_4() -> dict:
    """
    For each query, was any gold ID present in (observed_ids ∪ archive) at any
    step? Compare to "is any gold ID in the first stage's top-100?"
    """
    react_hybrid = _load_json(REACT_DIR / "react_hybrid_rrf_k60_test.json")
    react_bm25   = _load_json(REACT_DIR / "react_bm25_test.json")
    hybrid_first = _load_json(HYBRID_FILE)

    traces_hybrid = _load_json(REACT_DIR / "react_hybrid_rrf_k60_test_traces.json")
    traces_bm25   = _load_json(REACT_DIR / "react_bm25_test_traces.json")

    qrels = {qid: [int(g) for g in row.keys()]
             for qid, row in react_hybrid["_trec_qrels"].items()}

    def _agent_pool(traces) -> dict[str, set[int]]:
        """Union of all ids_returned across all steps per query."""
        pool: dict[str, set[int]] = {}
        for entry in traces:
            qid = str(entry["question_id"])
            ids: set[int] = set()
            for step in entry["trace"]:
                ids.update(step.get("ids_returned") or [])
            pool[qid] = ids
        return pool

    pool_hybrid = _agent_pool(traces_hybrid)
    pool_bm25   = _agent_pool(traces_bm25)

    bm25_top100 = _build_or_load_bm25_top100()

    def _gold_in(pool: dict[str, set[int] | list[int]], qid: str) -> bool:
        gold = set(qrels.get(qid, []))
        seen = set(pool.get(qid, []))
        return bool(gold & seen)

    qids = sorted(qrels.keys(), key=int)

    rows = []
    n_agent_hybrid_has_gold = 0
    n_first_hybrid_has_gold = 0
    n_agent_bm25_has_gold   = 0
    n_first_bm25_has_gold   = 0

    for qid in qids:
        first_hybrid_ids = _trec_run_to_top_ids(hybrid_first["_trec_run"], qid, 100)
        first_bm25_ids   = bm25_top100.get(qid, [])

        ah = _gold_in(pool_hybrid, qid)
        fh = bool(set(qrels[qid]) & set(first_hybrid_ids))
        ab = _gold_in(pool_bm25, qid)
        fb = bool(set(qrels[qid]) & set(first_bm25_ids))

        n_agent_hybrid_has_gold += ah
        n_first_hybrid_has_gold += fh
        n_agent_bm25_has_gold   += ab
        n_first_bm25_has_gold   += fb

        rows.append({
            "qid": qid,
            "n_gold": len(qrels[qid]),
            "agent_hybrid_pool_size": len(pool_hybrid.get(qid, [])),
            "agent_bm25_pool_size":   len(pool_bm25.get(qid, [])),
            "agent_hybrid_has_any_gold": ah,
            "first_hybrid_has_any_gold": fh,
            "agent_bm25_has_any_gold":   ab,
            "first_bm25_has_any_gold":   fb,
        })

    n = len(qids)
    return {
        "diagnostic": "15.1.4 gold-presence audit (any gold in pool / first-stage top-100)",
        "n_queries": n,
        "summary": {
            "fraction_agent_hybrid_pool_has_gold":  n_agent_hybrid_has_gold / n,
            "fraction_first_hybrid_top100_has_gold": n_first_hybrid_has_gold / n,
            "fraction_agent_bm25_pool_has_gold":    n_agent_bm25_has_gold / n,
            "fraction_first_bm25_top100_has_gold":  n_first_bm25_has_gold / n,
            "fraction_lost_by_agent_hybrid":  (n_first_hybrid_has_gold - n_agent_hybrid_has_gold) / n,
            "fraction_lost_by_agent_bm25":    (n_first_bm25_has_gold - n_agent_bm25_has_gold) / n,
        },
        "per_query": rows,
    }


# ---------------------------------------------------------------------------
# 15.1.5 — D1-on-top-50 isolation
# ---------------------------------------------------------------------------

def diag_15_1_5(d1_n_queries: int, max_article_tokens: int) -> dict:
    """
    Re-run only the D1 finalize step (LLM_JUDGE_BINARY_PROMPT) over the
    hybrid first stage's top-50 and compare to T4.0's saved hybrid R@10.
    Uses the agent's D1 config (max_article_tokens — default 200) to isolate
    "does D1 alone on a sane pool reach T4.0?".

    For local smoke runs, --d1-n-queries=10 is the default. Set to 222 on
    Azure for the full run.
    """
    import pandas as pd
    from retrieval.agentic.llm_client import OllamaClient
    from retrieval.agentic.llm_eval_prompts import (
        LLM_JUDGE_BINARY_PROMPT,
        parse_binary_judgment,
    )

    hybrid_first = _load_json(HYBRID_FILE)
    react_hybrid = _load_json(REACT_DIR / "react_hybrid_rrf_k60_test.json")
    qrels        = react_hybrid["_trec_qrels"]

    df = pd.read_parquet(ROOT / "output" / "bsard_articles_dedup.parquet")
    id_to_text = dict(zip(df["article_id"].astype(int), df["article_text"]))

    qids = sorted(qrels.keys(), key=int)
    if d1_n_queries < len(qids):
        # Deterministic prefix — preserves comparability across reruns.
        qids = qids[:d1_n_queries]
        mode = "smoke"
    else:
        mode = "full"

    llm = OllamaClient()
    llm.preflight()

    per_query: list[dict] = []
    sum_r10 = 0.0
    t_total = time.perf_counter()

    for i, qid in enumerate(qids, 1):
        # Top-50 from hybrid first stage
        top50 = _trec_run_to_top_ids(hybrid_first["_trec_run"], qid, 50)
        gold  = [int(g) for g in qrels[qid].keys()]

        # Pull the original question text out of the result row's first item
        # — but we don't have it. Use evaluation.split to reload.
        # (lazy import to avoid loading questions twice)
        from evaluation.split import load_questions
        if i == 1:
            questions = {q["question_id"]: q for q in load_questions(subset="test")}
        question_text = questions[int(qid)]["question_text"]

        scored = []
        for aid in top50:
            text = id_to_text.get(aid, "")
            words = text.split()
            if len(words) > max_article_tokens:
                text = " ".join(words[:max_article_tokens]) + " …"
            prompt = LLM_JUDGE_BINARY_PROMPT.format(
                fewshot_block="",
                question=question_text,
                article_text_truncated=text,
            )
            response, _ = llm.generate(prompt, temperature=0.0, max_tokens=8)
            _, score = parse_binary_judgment(response)
            scored.append((aid, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        ranked = [aid for aid, _ in scored]
        r10 = _r_at_k(ranked, gold, 10)
        sum_r10 += r10

        per_query.append({
            "qid": qid,
            "n_gold": len(gold),
            "n_top50": len(top50),
            "ranked_top10": ranked[:10],
            "R@10": r10,
        })

        if i % 5 == 0 or i == len(qids):
            elapsed = time.perf_counter() - t_total
            print(f"  [{i}/{len(qids)}] elapsed={elapsed:.1f}s  rolling R@10={sum_r10/i:.4f}",
                  flush=True)

    mean_r10 = sum_r10 / len(qids) if qids else 0.0
    t40_anchor = _load_json(T40_HYBRID_FILE)["metrics"]["Recall@10"]

    return {
        "diagnostic": "15.1.5 D1-on-top-50 (agent D1 config) over hybrid first stage",
        "mode": mode,
        "n_queries": len(qids),
        "max_article_tokens": max_article_tokens,
        "mean_R10": mean_r10,
        "t40_hybrid_R10_full_anchor": t40_anchor,
        "elapsed_s": round(time.perf_counter() - t_total, 1),
        "per_query": per_query,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-d1", action="store_true",
                        help="Skip 15.1.5 (Ollama LLM_JUDGE rerun)")
    parser.add_argument("--d1-n-queries", type=int, default=10,
                        help="N queries for 15.1.5 (default 10 for smoke, set 222 on Azure)")
    parser.add_argument("--d1-max-article-tokens", type=int, default=200,
                        help="max_article_tokens for D1 (matches Round-1 ReAct config)")
    args = parser.parse_args()

    ROUND2_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("§15.1.3 — per-stratum R@10 delta vs first stage")
    print("=" * 60)
    out_15_1_3 = diag_15_1_3()
    _save_json(ROUND2_DIR / "diagnostics_15_1_3_per_stratum_delta.json", out_15_1_3)
    print(f"  overall: agent_hybrid={out_15_1_3['overall']['agent_hybrid_R10']:.4f}  "
          f"first_hybrid={out_15_1_3['overall']['first_hybrid_R10']:.4f}  "
          f"wins/ties/losses={out_15_1_3['overall']['wins/ties/losses']}")

    print("\n" + "=" * 60)
    print("§15.1.4 — gold-presence audit")
    print("=" * 60)
    out_15_1_4 = diag_15_1_4()
    _save_json(ROUND2_DIR / "diagnostics_15_1_4_gold_presence.json", out_15_1_4)
    s = out_15_1_4["summary"]
    print(f"  agent_hybrid_pool: {s['fraction_agent_hybrid_pool_has_gold']:.3f}  "
          f"vs first_hybrid_top100: {s['fraction_first_hybrid_top100_has_gold']:.3f}")
    print(f"  agent_bm25_pool:   {s['fraction_agent_bm25_pool_has_gold']:.3f}  "
          f"vs first_bm25_top100:   {s['fraction_first_bm25_top100_has_gold']:.3f}")

    if args.skip_d1:
        print("\n§15.1.5 — skipped (--skip-d1)")
        return

    print("\n" + "=" * 60)
    print(f"§15.1.5 — D1-on-top-50 (n={args.d1_n_queries}, "
          f"max_article_tokens={args.d1_max_article_tokens})")
    print("=" * 60)
    out_15_1_5 = diag_15_1_5(args.d1_n_queries, args.d1_max_article_tokens)
    _save_json(ROUND2_DIR / "diagnostics_15_1_5_d1_on_top50.json", out_15_1_5)
    print(f"  mean R@10 = {out_15_1_5['mean_R10']:.4f}  "
          f"(T4.0 anchor = {out_15_1_5['t40_hybrid_R10_full_anchor']:.4f}, "
          f"mode = {out_15_1_5['mode']})")


if __name__ == "__main__":
    main()
