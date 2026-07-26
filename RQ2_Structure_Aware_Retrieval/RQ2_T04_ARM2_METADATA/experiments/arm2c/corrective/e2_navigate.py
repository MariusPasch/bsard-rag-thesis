"""E2 phase 1 — corrective NAVIGATION only (Ollama), save the reached pools.

Decoupled from the e5 rerank (phase 2 = e2_rerank.py) so the navigator runs ALONE
on the GPU: loading e5 alongside llama on the 16 GB T4 caused VRAM contention that
perturbed llama's greedy path and collapsed the corrective reach. Run nav first
(llama only), save the pools, then rerank separately (e5 only).

For each query it saves both reached pools (round-1 single-pass + full corrective),
both 8B-reranked rankings, and the visit orders (for reach) — everything phase 2
needs, with NO embedding model. Resumable: re-running skips queries already saved.

    python corrective/e2_navigate.py --stem 1804_03_21_1804032150 \\
        --qids 243,244,252,181,290,1057,158,159,1043,302 --max-rounds 2

Writes runs/_corrective_e2/<stem>_nav.json (the input to phase 2).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve()
_ARM2C = _HERE.parent.parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_ARM2C))
from corrective_navigator import navigate_corrective         # noqa: E402
from smoke_corrective import load_stem, _reached_recall, _recall  # noqa: E402

OUT_DIR = _ARM2C / "runs" / "_corrective_e2"


def _nav_path(stem: str) -> Path:
    return OUT_DIR / f"{stem}_nav.json"


def _load_existing(stem: str) -> dict:
    p = _nav_path(stem)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"stem": stem, "rows": []}


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", required=True)
    ap.add_argument("--qids", default="", help="comma-separated; omit with --all")
    ap.add_argument("--all", action="store_true", help="navigate every query in the bundle")
    ap.add_argument("--mode", choices=["bare", "enriched"], default="enriched")
    ap.add_argument("--max-nodes", type=int, default=40)
    ap.add_argument("--max-rounds", type=int, default=2)
    ap.add_argument("--reseed-strategy", choices=["ranked", "all"], default="ranked")
    ap.add_argument("--reseed-m", type=int, default=5)
    args = ap.parse_args()

    tree, b2l, queries = load_stem(args.stem)
    if args.all:
        qids = list(queries.keys())
    else:
        qids = [q.strip() for q in args.qids.split(",") if q.strip()]
    if not qids:
        print("give --qids or --all"); return 1

    from react_navigator import LlamaClient                  # lazy: ollama
    llm = LlamaClient(model="llama3.1:8b", temperature=0.0, num_ctx=16384)

    store = _load_existing(args.stem)
    store.update({"mode": args.mode, "max_rounds": args.max_rounds,
                  "reseed_strategy": args.reseed_strategy})
    done = {str(r["qid"]) for r in store["rows"]}
    todo = [q for q in qids if str(q) not in done]
    print(f"stem={args.stem} | {len(todo)} to navigate, {len(done)} already saved\n")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for i, qid in enumerate(todo, 1):
        q = queries.get(str(qid))
        if not q:
            print(f"{qid:>6}  (not in bundle)"); continue
        gold = set(int(x) for x in q["gold_bsard_ids"]); text = q["query_text"]
        t0 = time.perf_counter()
        r = navigate_corrective(text, tree, llm, mode=args.mode, max_nodes=args.max_nodes,
                                max_rounds=args.max_rounds, reseed_strategy=args.reseed_strategy,
                                reseed_m=args.reseed_m, rerank=True)
        secs = time.perf_counter() - t0
        reach1 = _reached_recall(tree, b2l, gold, set(r.round1_visit_order))
        reachC = _reached_recall(tree, b2l, gold, {s.node_id for s in r.steps})
        store["rows"].append({
            "qid": str(qid), "gold_bsard_ids": sorted(gold), "query_text": text,
            "round1_pool": r.round1_pool_node_ids, "corr_pool": r.pool_node_ids,
            "round1_ranked": r.round1_ranked_bsard_ids, "corr_ranked": r.ranked_bsard_ids,
            "round1_visit": r.round1_visit_order,
            "corr_visit": [s.node_id for s in r.steps],
            "nodes": r.nodes_visited, "rounds": len(r.rounds), "llm_calls": r.llm_calls,
            "secs": round(secs, 1),
        })
        # persist after every query (resumable / crash-safe)
        _nav_path(args.stem).write_text(json.dumps(store, ensure_ascii=False, indent=1),
                                        encoding="utf-8")
        print(f"[{i:>3}/{len(todo)}] q{qid:>5} ng={len(gold):>2} "
              f"reach {reach1:.2f}->{reachC:.2f}  8B R@10 {_recall(gold, r.round1_ranked_bsard_ids,10):.2f}"
              f"->{_recall(gold, r.ranked_bsard_ids,10):.2f}  ({secs:.0f}s)", flush=True)

    print(f"\n[{len(store['rows'])} queries saved -> {_nav_path(args.stem)}]")
    print("next: bash corrective/launch_e2_rerank.sh --stem", args.stem)
    return 0


if __name__ == "__main__":
    sys.exit(main())
