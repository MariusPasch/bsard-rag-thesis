"""Headless full run — runs DETACHED on the VM so it survives kernel/connection
drops, laptop sleep, VSCode disconnects (everything that kept killing the
notebook mid-run). Same logic as notebook cells 6/10/11, resume-friendly.

On the VM (model already pinned at num_ctx 16384, keep_alive=-1):

    cd ~/repos/RQ2_T04_ARM2_METADATA/experiments/arm2c && git pull
    nohup python run_all.py --mode enriched --rerank > run_enriched_rerank.log 2>&1 &
    tail -f run_enriched_rerank.log          # watch; Ctrl-C just stops watching, not the run

It picks up any q<qid>.json already on disk (so it resumes the 60 you have), runs
the rest, then prints + writes the analysis. Disconnecting your IDE does nothing
to it — it's a VM-side background process.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))
from navigator_tools import DeepTree          # noqa: E402
from react_navigator import LlamaClient, navigate   # noqa: E402
import analyze_results                          # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc-id", default="1804_03_21_1804032150")
    ap.add_argument("--mode", default="enriched", choices=["bare", "enriched"])
    ap.add_argument("--rerank", action="store_true")
    ap.add_argument("--max-nodes", type=int, default=40)
    ap.add_argument("--max-branch", type=int, default=5)
    ap.add_argument("--results-root", default="/home/azureuser/results")
    ap.add_argument("--model", default="llama3.1:8b")
    args = ap.parse_args()

    bundle = _HERE.parent / "bundles" / args.doc_id
    tree = DeepTree.load(bundle / "deep_tree.json")
    queries = json.loads((bundle / "queries.json").read_text(encoding="utf-8"))
    run_name = f"arm2c_{args.doc_id}_{args.mode}" + ("_rerank" if args.rerank else "")
    results = Path(args.results_root) / run_name
    results.mkdir(parents=True, exist_ok=True)
    done = {p.stem for p in results.glob("q*.json")}
    print(f"[run_all] {run_name}: resuming {len(done)}/{len(queries)} "
          f"(mode={args.mode} rerank={args.rerank})", flush=True)

    llm = LlamaClient(model=args.model, num_ctx=16384)
    t0 = time.perf_counter()
    for i, q in enumerate(queries, 1):
        qid = str(q["query_id"])
        if f"q{qid}" in done:
            continue
        r = navigate(q["query_text"], tree, llm, mode=args.mode, max_nodes=args.max_nodes,
                     max_branch=args.max_branch, rerank=args.rerank)
        rec = {
            "query_id": qid, "query_text": q["query_text"], "gold_bsard_ids": q["gold_bsard_ids"],
            "selected_bsard_ids": r.selected_bsard_ids, "ranked_bsard_ids": r.ranked_bsard_ids,
            "ranked_bsard_ids_prererank": r.ranked_bsard_ids_prererank,
            "nodes_visited": r.nodes_visited, "llm_calls": r.llm_calls,
            "exit_reason": r.exit_reason, "mode": args.mode,
            "steps": [asdict(s) for s in r.steps],
        }
        (results / f"q{qid}.json").write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
        if i % 10 == 0:
            n = len(list(results.glob("q*.json")))
            print(f"[run_all] {n}/{len(queries)}  "
                  f"({(time.perf_counter()-t0)/max(i-len(done),1):.1f}s/q)", flush=True)

    n = len(list(results.glob("q*.json")))
    print(f"[run_all] DONE {n}/{len(queries)} in {results}", flush=True)
    print("\n[run_all] analysis:\n", flush=True)
    analyze_results.analyze(results, bundle, bundle / "deep_tree.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
