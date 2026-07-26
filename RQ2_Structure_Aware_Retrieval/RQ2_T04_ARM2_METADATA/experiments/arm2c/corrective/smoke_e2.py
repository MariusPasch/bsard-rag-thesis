"""E2 — the full synthesis: Arm-2C corrective NAVIGATION (reach) + Arm-2A e5 RANKING.

For each query, ONE corrective navigation produces two reached pools (round-1 =
single-pass, and the full corrective pool), and we rank each with both rerankers —
so four R@10 numbers come off the same navigation, isolating both levers:

                       single-pass pool        corrective pool
  8B list-rerank       8B/single (shipped 2C)  8B/corr (corrective loop as-is)
  e5 embed-rerank      e5/single (= E1)        e5/corr  ← THE SYNTHESIS

Hypothesis: the corrective loop raises reach (0.47->~0.80 on hard queries) and the
e5 ranker converts ~83% of reach into the top-10 (E1) — so e5/corr should clear all
three others and approach Arm-2A territory.

Needs BOTH Ollama+llama3.1:8b (navigation) and the e5 encoder (rerank); both fit a
16 GB T4. Both are injectable, so the control flow is mock-testable.

    python corrective/smoke_e2.py --stem 1804_03_21_1804032150 --qids 243,158,1043,290 --max-rounds 2
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
from navigator_tools import DeepTree                         # noqa: E402
from corrective_navigator import navigate_corrective         # noqa: E402
from embed_rerank import embed_rerank_pool                   # noqa: E402
from smoke_corrective import load_stem, _reached_recall, _recall  # noqa: E402


def run_e2(tree, b2l, queries: dict, qids: list[str], llm, encoder, *,
           mode: str = "enriched", max_nodes: int = 40, max_rounds: int = 2,
           reseed_strategy: str = "ranked", reseed_m: int = 5, text_field: str = "text",
           emit=print) -> list[dict]:
    hdr = (f"{'qid':>6} {'ng':>3} | {'reach1':>6} {'reachC':>6} | "
           f"{'8B/1':>5} {'8B/C':>5} {'e5/1':>5} {'e5/C':>5} | rnds nodes  s")
    emit(hdr); emit("-" * len(hdr))
    rows = []
    for qid in qids:
        q = queries.get(str(qid))
        if not q:
            emit(f"{qid:>6}  (not in bundle)"); continue
        gold = set(int(x) for x in q["gold_bsard_ids"]); text = q["query_text"]

        t0 = time.perf_counter()
        r = navigate_corrective(text, tree, llm, mode=mode, max_nodes=max_nodes,
                                max_rounds=max_rounds, reseed_strategy=reseed_strategy,
                                reseed_m=reseed_m, rerank=True)
        # e5 rerank over each pool (same encoder)
        e5_single = embed_rerank_pool(text, r.round1_pool_node_ids, tree, encoder,
                                      k=100, text_field=text_field)
        e5_corr = embed_rerank_pool(text, r.pool_node_ids, tree, encoder,
                                    k=100, text_field=text_field)
        secs = time.perf_counter() - t0

        reach1 = _reached_recall(tree, b2l, gold, set(r.round1_visit_order))
        reachC = _reached_recall(tree, b2l, gold, {s.node_id for s in r.steps})
        m = {
            "r8_single": _recall(gold, r.round1_ranked_bsard_ids, 10),
            "r8_corr":   _recall(gold, r.ranked_bsard_ids, 10),
            "e5_single": _recall(gold, e5_single, 10),
            "e5_corr":   _recall(gold, e5_corr, 10),
        }
        emit(f"{qid:>6} {len(gold):>3} | {reach1:>6.2f} {reachC:>6.2f} | "
             f"{m['r8_single']:>5.2f} {m['r8_corr']:>5.2f} {m['e5_single']:>5.2f} "
             f"{m['e5_corr']:>5.2f} | {len(r.rounds):>4} {r.nodes_visited:>5} {secs:>3.0f}")
        rows.append({"qid": str(qid), "n_gold": len(gold), "gold": sorted(gold),
                     "reach_single": reach1, "reach_corr": reachC,
                     "nodes": r.nodes_visited, "rounds": len(r.rounds),
                     "llm_calls": r.llm_calls, "secs": round(secs, 1), **m})
    if rows:
        mean = lambda k: sum(r[k] for r in rows) / len(rows)
        emit("-" * len(hdr))
        emit(f"  mean reach {mean('reach_single'):.2f}->{mean('reach_corr'):.2f}  |  "
             f"R@10  8B/single {mean('r8_single'):.3f}  8B/corr {mean('r8_corr'):.3f}  "
             f"e5/single {mean('e5_single'):.3f}  e5/corr {mean('e5_corr'):.3f}  (n={len(rows)})")
    return rows


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", required=True)
    ap.add_argument("--qids", required=True)
    ap.add_argument("--mode", choices=["bare", "enriched"], default="enriched")
    ap.add_argument("--max-nodes", type=int, default=40)
    ap.add_argument("--max-rounds", type=int, default=2)
    ap.add_argument("--reseed-strategy", choices=["ranked", "all"], default="ranked")
    ap.add_argument("--reseed-m", type=int, default=5)
    ap.add_argument("--text-field", choices=["text", "summary"], default="text")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    tree, b2l, queries = load_stem(args.stem)
    qids = [q.strip() for q in args.qids.split(",") if q.strip()]

    from react_navigator import LlamaClient                  # lazy: ollama
    from embed_rerank import E5Encoder                       # lazy: sentence-transformers
    llm = LlamaClient(model="llama3.1:8b", temperature=0.0, num_ctx=16384)
    print("loading e5 encoder (intfloat/multilingual-e5-large-instruct)…")
    encoder = E5Encoder(device=args.device)

    print(f"stem={args.stem} mode={args.mode} max_rounds={args.max_rounds} "
          f"text_field={args.text_field}\n")
    rows = run_e2(tree, b2l, queries, qids, llm, encoder, mode=args.mode,
                  max_nodes=args.max_nodes, max_rounds=args.max_rounds,
                  reseed_strategy=args.reseed_strategy, reseed_m=args.reseed_m,
                  text_field=args.text_field)
    if rows:
        outdir = _ARM2C / "runs" / "_corrective_e2"
        outdir.mkdir(parents=True, exist_ok=True)
        p = outdir / f"{args.stem}_e2.json"
        p.write_text(json.dumps({"stem": args.stem, "mode": args.mode,
                                 "max_rounds": args.max_rounds, "text_field": args.text_field,
                                 "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[wrote {p}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
