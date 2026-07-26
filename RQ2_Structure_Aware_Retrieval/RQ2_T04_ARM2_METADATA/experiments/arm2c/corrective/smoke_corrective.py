"""Tier (c) — minimal REAL smoke: OLD navigate() vs NEW navigate_corrective() on a
handful of saved queries. Needs Ollama + llama3.1:8b (NOT available on the dev box;
run on the Azure T4 / wherever the navigator runs).

Runs both navigators on the SAME queries with the SAME LlamaClient and reports, per
query, the PADDING-FREE navigator capabilities (reached / selected recall) AND the
padded headline R@10 — old vs new — so you can see whether the corrective loop
actually reaches/selects more gold, not just whether padding moved.

Two entry points, one engine:
  * CLI / launch_smoke.sh  -> main()  (argparse)
  * notebook on the kernel -> load_stem() + run_smoke()

Test order (per the plan): depth/fine-prune docs first, housing as negative control.
  Code Penal   1867_06_08_1867060850   (depth; backtrack should help)
  Code Civil   1804_03_21_1804032150   (select-gap; re-select should help)
  Housing      2003_07_17_2013A31614   (NEGATIVE CONTROL; expect little)

Example (T05 venv has arm2_metadata + ollama; or use launch_smoke.sh):
  python corrective/smoke_corrective.py --stem 1867_06_08_1867060850 \\
    --qids 1048,202,240 --max-rounds 2 --mode enriched

Writes (additive, optional): runs/_corrective_smoke/<stem>_smoke.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve()
_ARM2C = _HERE.parent.parent
sys.path.insert(0, str(_HERE.parent))                 # corrective/
sys.path.insert(0, str(_ARM2C))                       # experiments/arm2c
from navigator_tools import DeepTree                  # noqa: E402
from corrective_navigator import navigate_corrective  # noqa: E402
# OLD (single-pass) baseline is derived from the loop's round-1 snapshot — see run_smoke.


# ── metric helpers ───────────────────────────────────────────────────────────

def _bsard_to_leaves(tree: DeepTree) -> dict[int, list[str]]:
    out: dict[int, list[str]] = {}
    for n in tree.by_id.values():
        if n.is_leaf and n.bsard_id is not None:
            out.setdefault(n.bsard_id, []).append(n.node_id)
    return out


def _reached_recall(tree, b2l, gold: set[int], visited: set[str]) -> float:
    """Gold whose parent section was expanded (padding-free navigation reach)."""
    if not gold:
        return 0.0
    hit = sum(any(tree.parent.get(lf) in visited for lf in b2l.get(g, [])) for g in gold)
    return hit / len(gold)


def _recall(gold: set[int], ranked: list[int], k: int) -> float:
    return len(gold & set(ranked[:k])) / len(gold) if gold else 0.0


# ── shared engine (used by CLI, launcher, notebook) ──────────────────────────

def load_stem(stem: str, arm2c_dir: Path = _ARM2C):
    """Return (tree, bsard_to_leaves, {qid: query}) for a stem's committed bundle."""
    tree_path = arm2c_dir / "bundles" / stem / "deep_tree.json"
    if not tree_path.exists():
        tree_path = arm2c_dir / "data" / stem / "deep_tree.json"
    tree = DeepTree.load(tree_path)
    queries = json.loads((arm2c_dir / "bundles" / stem / "queries.json").read_text("utf-8"))
    return tree, _bsard_to_leaves(tree), {str(q["query_id"]): q for q in queries}


def run_smoke(tree, b2l, queries: dict, qids: list[str], llm, *, mode: str = "enriched",
              max_nodes: int = 40, max_rounds: int = 2, reseed_strategy: str = "ranked",
              reseed_m: int = 5, emit=print) -> list[dict]:
    """OLD vs NEW on each qid from ONE navigation per query: OLD = the corrective
    loop's round-1 snapshot (== old navigate(), verified), NEW = the full loop.

    This is a CONTROLLED ablation — identical round-1 trace, so the only difference
    is the corrective rounds. Running navigate() and navigate_corrective() as two
    independent passes (the previous design) confounded the loop's effect with the
    8B's run-to-run selection noise (temp-0 llama.cpp isn't bit-reproducible), which
    on borderline queries swamped the signal (e.g. Civil q243 flipping select 4->0).
    """
    hdr = (f"{'qid':>6} {'ngold':>5} | {'OLD reach':>9} {'OLD sel':>7} {'OLD R@10':>8} "
           f"| {'NEW reach':>9} {'NEW sel':>7} {'NEW R@10':>8} | rounds nodes calls  s")
    emit(hdr); emit("-" * len(hdr))
    rows: list[dict] = []
    for qid in qids:
        q = queries.get(str(qid))
        if not q:
            emit(f"{qid:>6}  (not in bundle)")
            continue
        gold = set(int(x) for x in q["gold_bsard_ids"])
        text = q["query_text"]

        t0 = time.perf_counter()
        run = navigate_corrective(text, tree, llm, mode=mode, max_nodes=max_nodes,
                                  max_rounds=max_rounds, reseed_strategy=reseed_strategy,
                                  reseed_m=reseed_m, rerank=True)
        secs = time.perf_counter() - t0

        old_vis = set(run.round1_visit_order)
        old_reach = _reached_recall(tree, b2l, gold, old_vis)
        old_sel = len(gold & set(run.round1_selected_bsard_ids)) / len(gold) if gold else 0.0
        old_r10 = _recall(gold, run.round1_ranked_bsard_ids, 10)

        new_vis = {s.node_id for s in run.steps}
        new_reach = _reached_recall(tree, b2l, gold, new_vis)
        new_sel = len(gold & set(run.selected_bsard_ids)) / len(gold) if gold else 0.0
        new_r10 = _recall(gold, run.ranked_bsard_ids, 10)

        emit(f"{qid:>6} {len(gold):>5} | {old_reach:>9.3f} {old_sel:>7.3f} {old_r10:>8.3f} "
             f"| {new_reach:>9.3f} {new_sel:>7.3f} {new_r10:>8.3f} "
             f"| {len(run.rounds):>6} {run.nodes_visited:>5} {run.llm_calls:>5} {secs:>4.0f}")
        rows.append({
            "qid": str(qid), "n_gold": len(gold), "gold": sorted(gold),
            "old": {"reach": old_reach, "selected": old_sel, "r10": old_r10,
                    "nodes": len(run.round1_visit_order)},
            "new": {"reach": new_reach, "selected": new_sel, "r10": new_r10,
                    "nodes": run.nodes_visited, "calls": run.llm_calls, "secs": round(secs, 1),
                    "rounds": len(run.rounds), "exit": run.exit_reason,
                    "round_reach": [ri.reach_total for ri in run.rounds]},
        })
    if rows:
        def mean(key, sub): return sum(r[sub][key] for r in rows) / len(rows)
        emit("-" * len(hdr))
        emit(f"  mean   reach {mean('reach','old'):.3f}->{mean('reach','new'):.3f}   "
             f"selected {mean('selected','old'):.3f}->{mean('selected','new'):.3f}   "
             f"R@10 {mean('r10','old'):.3f}->{mean('r10','new'):.3f}   (n={len(rows)})")
    return rows


def write_smoke(stem: str, mode: str, max_rounds: int, reseed_strategy: str,
                rows: list[dict], arm2c_dir: Path = _ARM2C) -> Path:
    outdir = arm2c_dir / "runs" / "_corrective_smoke"
    outdir.mkdir(parents=True, exist_ok=True)
    payload = {"stem": stem, "mode": mode, "max_rounds": max_rounds,
               "reseed_strategy": reseed_strategy, "rows": rows}
    p = outdir / f"{stem}_smoke.json"
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", required=True)
    ap.add_argument("--qids", required=True, help="comma-separated query ids")
    ap.add_argument("--mode", choices=["bare", "enriched"], default="enriched")
    ap.add_argument("--max-nodes", type=int, default=40)
    ap.add_argument("--max-rounds", type=int, default=2)
    ap.add_argument("--reseed-strategy", choices=["ranked", "all"], default="ranked")
    ap.add_argument("--reseed-m", type=int, default=5)
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()

    from react_navigator import LlamaClient          # lazy: imports ollama
    tree, b2l, queries = load_stem(args.stem)
    qids = [q.strip() for q in args.qids.split(",") if q.strip()]
    llm = LlamaClient(model="llama3.1:8b", temperature=0.0, num_ctx=16384)
    print(f"stem={args.stem} mode={args.mode} max_rounds={args.max_rounds} "
          f"reseed={args.reseed_strategy}\n")
    rows = run_smoke(tree, b2l, queries, qids, llm, mode=args.mode, max_nodes=args.max_nodes,
                     max_rounds=args.max_rounds, reseed_strategy=args.reseed_strategy,
                     reseed_m=args.reseed_m)
    if not args.no_write and rows:
        p = write_smoke(args.stem, args.mode, args.max_rounds, args.reseed_strategy, rows)
        print(f"\n[wrote {p}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
