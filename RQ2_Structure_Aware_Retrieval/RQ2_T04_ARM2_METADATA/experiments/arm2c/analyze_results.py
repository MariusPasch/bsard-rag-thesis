"""Arm-2C results analysis — not just pass/fail, but WHY, and WHAT TO DO NEXT.

Reads the per-query q<qid>.json + the deep tree + the bundle and produces a
decision-support report:

  1. Headline      recall@{1,5,10,20,100}, hit@10, MRR@10 vs Arm-1/2A/2B baselines
  2. Navigation    nodes visited, LLM calls, exit reasons, parse-fail, depth, Unfiled use
  3. Selection     #selected/q, precision, empty-selection queries
  4. MISS DECOMP   the key diagnostic — every gold article classified:
        HIT               selected
        SEEN_NOT_SELECTED its section was expanded (gold shown) but not picked  -> selection/prompt
        NOT_REACHED       its section was never expanded (branch pruned)        -> tree/navigation (Arm-2B-like)
        ORPHAN_UNREACHED  gold in the Unfiled branch, never explored            -> coverage ceiling
  5. By cardinality   recall@10 for single- vs multi-gold questions
  6. Extremes         best / worst queries to eyeball
  7. RECOMMENDATION   rule-based: scale to other PDFs, or change what

Writes analysis_report.md + per_query.csv next to the results. Stdlib only.

    python analyze_results.py --results <dir> --bundle bundles/<doc>/ --tree data/<doc>/deep_tree.json
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))
from navigator_tools import DeepTree  # noqa: E402


# ── metrics ──────────────────────────────────────────────────────────────────

def _recall(gold, ranked, k):
    return len(set(gold) & set(ranked[:k])) / len(set(gold)) if gold else 0.0

def _hit(gold, ranked, k):
    return int(bool(set(gold) & set(ranked[:k])))

def _mrr(gold, ranked, k):
    gs = set(gold)
    for i, b in enumerate(ranked[:k], 1):
        if b in gs:
            return 1.0 / i
    return 0.0


# ── tree helpers for the miss decomposition ──────────────────────────────────

def _tree_maps(tree: DeepTree):
    """bsard_id -> [leaf node_ids]; set of leaf node_ids under the Unfiled branch."""
    bsard_to_leaves: dict[int, list[str]] = {}
    orphan_leaves: set[str] = set()

    def walk(node, under_orphan):
        is_orphan = under_orphan or bool(node.metadata.get("orphan"))
        if node.is_leaf:
            if node.bsard_id is not None:
                bsard_to_leaves.setdefault(node.bsard_id, []).append(node.node_id)
            if is_orphan:
                orphan_leaves.add(node.node_id)
            return
        for c in node.sub_nodes:
            walk(c, is_orphan)

    walk(tree.root, False)
    return bsard_to_leaves, orphan_leaves


def _classify_gold(g, leaves_map, orphan_leaves, tree, visited, selected_bsard):
    leaves = leaves_map.get(g, [])
    if not leaves:
        return "not_in_tree"
    if g in selected_bsard:
        return "HIT"
    parent_seen = any(tree.parent.get(lf) in visited for lf in leaves)
    if parent_seen:
        return "SEEN_NOT_SELECTED"
    if all(lf in orphan_leaves for lf in leaves):
        return "ORPHAN_UNREACHED"
    return "NOT_REACHED"


# ── main analysis ────────────────────────────────────────────────────────────

def analyze(results_dir: Path, bundle_dir: Path, tree_path: Path) -> dict:
    queries = json.loads((bundle_dir / "queries.json").read_text(encoding="utf-8"))
    base = json.loads((bundle_dir / "baselines.json").read_text(encoding="utf-8")).get("baselines_recall@10", {})
    tree = DeepTree.load(tree_path)
    leaves_map, orphan_leaves = _tree_maps(tree)

    results = {}
    for p in results_dir.glob("q*.json"):
        try:
            r = json.loads(p.read_text(encoding="utf-8"))
            results[str(r["query_id"])] = r
        except Exception:
            pass

    Ks = [1, 5, 10, 20, 100]
    agg = {k: 0.0 for k in Ks}
    hit10 = mrr10 = 0.0
    r10_pre = r100_pre = 0.0
    n_pre = 0          # queries that carry a pre-rerank ranking
    nsel = []
    prec = []
    exits = Counter()
    visited_depth = []
    used_unfiled = 0
    pf_calls = tot_calls = 0
    gold_cls = Counter()
    missing = []
    empties = 0
    per_query_rows = []
    by_card = {"single": [], "multi": []}

    for q in queries:
        qid = str(q["query_id"])
        gold = q["gold_bsard_ids"]
        r = results.get(qid)
        if r is None:
            missing.append(qid)
            continue
        ranked = r.get("ranked_bsard_ids") or []
        prer = r.get("ranked_bsard_ids_prererank")
        if prer is not None:
            r10_pre += _recall(gold, prer, 10)
            r100_pre += _recall(gold, prer, 100)
            n_pre += 1
        selected_bsard = set(r.get("selected_bsard_ids") or [])
        if not selected_bsard:
            empties += 1
        rec = {k: _recall(gold, ranked, k) for k in Ks}
        for k in Ks:
            agg[k] += rec[k]
        hit10 += _hit(gold, ranked, 10)
        mrr10 += _mrr(gold, ranked, 10)
        nsel.append(len(selected_bsard))
        prec.append(len(selected_bsard & set(gold)) / len(selected_bsard) if selected_bsard else 0.0)
        exits[r.get("exit_reason", "?")] += 1
        steps = r.get("steps", [])
        visited = {s["node_id"] for s in steps}
        visited_depth.append(len(steps))
        if any("ORPHAN" in s["node_id"] for s in steps):
            used_unfiled += 1
        for s in steps:
            tot_calls += 1
            if not s.get("parse_ok", True):
                pf_calls += 1
        cls = Counter(_classify_gold(g, leaves_map, orphan_leaves, tree, visited, selected_bsard) for g in gold)
        gold_cls.update(cls)
        (by_card["multi"] if len(gold) > 1 else by_card["single"]).append(rec[10])
        per_query_rows.append({
            "qid": qid, "n_gold": len(gold), "recall@10": round(rec[10], 3),
            "recall@10_pre": round(_recall(gold, prer, 10), 3) if prer is not None else "",
            "recall@100": round(rec[100], 3), "hit@10": _hit(gold, ranked, 10),
            "mrr@10": round(_mrr(gold, ranked, 10), 3), "n_selected": len(selected_bsard),
            "nodes_visited": r.get("nodes_visited"), "llm_calls": r.get("llm_calls"),
            "exit": r.get("exit_reason"),
            **{f"gold_{c}": cls.get(c, 0) for c in
               ("HIT", "SEEN_NOT_SELECTED", "NOT_REACHED", "ORPHAN_UNREACHED")},
        })

    n = len(per_query_rows)
    if n == 0:
        print("No scored queries — run the notebook first.")
        return {}

    mean = {k: agg[k] / n for k in Ks}
    mhit10, mmrr10 = hit10 / n, mrr10 / n
    pf_rate = pf_calls / max(tot_calls, 1)
    arm2b = base.get("arm2b", {}).get("mean_recall@10")
    total_gold = sum(gold_cls.values())
    miss = total_gold - gold_cls["HIT"]

    # ── build report lines ────────────────────────────────────────────────
    L = []
    def P(s=""):
        L.append(s)

    P("=" * 70)
    P(f"ARM-2C ANALYSIS — {results_dir.name}  ({n}/{len(queries)} queries scored)")
    P("=" * 70)
    P("\n[1] HEADLINE — recall@k / hit / MRR")
    P("  " + "  ".join(f"R@{k}={mean[k]:.3f}" for k in Ks))
    P(f"  hit@10={mhit10:.3f}   MRR@10={mmrr10:.3f}")
    P("  baselines recall@10:  "
      f"Arm-1={base.get('arm1',{}).get('mean_recall@10')}  "
      f"Arm-2A={base.get('arm2a',{}).get('mean_recall@10')}  "
      f"Arm-2B={arm2b}")
    if arm2b:
        d = mean[10] - arm2b
        P(f"  Arm-2C vs Arm-2B: {d:+.3f} ({100*d/arm2b:+.0f}% rel)  "
          f"{'BEATS' if d > 0 else 'BELOW'} the bar")

    if n_pre:
        pre10, pre100 = r10_pre / n_pre, r100_pre / n_pre
        P("\n[1b] RERANK EFFECT (same navigation, ranking before vs after the re-rank call)")
        P(f"  recall@10 : {pre10:.3f} (pre) -> {mean[10]:.3f} (post)   delta {mean[10]-pre10:+.3f}")
        P(f"  recall@100: {pre100:.3f} (pre) -> {mean[100]:.3f} (post)  "
          "(rerank reorders the pool; R@100 ~unchanged)")

    P("\n[2] NAVIGATION")
    P(f"  nodes visited/q: mean={statistics.mean(visited_depth):.1f} "
      f"median={int(statistics.median(visited_depth))} max={max(visited_depth)}")
    P(f"  exit reasons: {dict(exits)}   (budget-hits => raise MAX_NODES)")
    P(f"  LLM calls={tot_calls}  parse-fail={pf_calls} ({100*pf_rate:.1f}%)")
    P(f"  queries that explored the Unfiled branch: {used_unfiled}/{n}")

    P("\n[3] SELECTION")
    P(f"  selected/q: mean={statistics.mean(nsel):.1f} median={int(statistics.median(nsel))}  "
      f"empty-selection queries={empties}")
    P(f"  precision (selected that are gold): {statistics.mean(prec):.3f}")

    P("\n[4] MISS DECOMPOSITION  (the key diagnostic — total gold = "
      f"{total_gold}, misses = {miss})")
    for c in ("HIT", "SEEN_NOT_SELECTED", "NOT_REACHED", "ORPHAN_UNREACHED", "not_in_tree"):
        v = gold_cls.get(c, 0)
        if v or c in ("HIT", "NOT_REACHED", "SEEN_NOT_SELECTED"):
            of_miss = f"  ({100*v/miss:.0f}% of misses)" if (miss and c != "HIT") else ""
            P(f"  {c:18}: {v:4}  ({100*v/total_gold:.0f}% of gold){of_miss}")

    P("\n[5] BY CARDINALITY — recall@10")
    for card in ("single", "multi"):
        xs = by_card[card]
        if xs:
            P(f"  {card}-gold ({len(xs)} q): {statistics.mean(xs):.3f}")

    P("\n[6] EXTREMES")
    rows_sorted = sorted(per_query_rows, key=lambda r: r["recall@10"])
    P("  worst (recall@10=0, gold reachable):")
    for r in [r for r in rows_sorted if r["recall@10"] == 0][:5]:
        P(f"    q{r['qid']:>5}  gold={r['n_gold']}  reached_miss={r['gold_NOT_REACHED']} "
          f"seen_miss={r['gold_SEEN_NOT_SELECTED']}  visited={r['nodes_visited']} exit={r['exit']}")
    P("  best (recall@10=1):")
    for r in [r for r in rows_sorted if r["recall@10"] == 1][-3:]:
        P(f"    q{r['qid']:>5}  gold={r['n_gold']}  selected={r['n_selected']}")

    # ── [7] recommendation ────────────────────────────────────────────────
    op_ok = (not missing) and pf_rate <= 0.10
    sci_ok = arm2b is not None and mean[10] > arm2b
    nr_share = gold_cls["NOT_REACHED"] / miss if miss else 0
    sn_share = gold_cls["SEEN_NOT_SELECTED"] / miss if miss else 0
    orph_share = gold_cls["ORPHAN_UNREACHED"] / miss if miss else 0
    rank_gap = mean[100] - mean[10]

    recs = []
    if pf_rate > 0.10:
        recs.append(f"FIX ROBUSTNESS FIRST — parse-fail {100*pf_rate:.0f}%. Results unreliable; "
                    "inspect the JSON prompt / add a stricter retry before trusting anything.")
    if exits.get("budget", 0) > 0.2 * n:
        recs.append(f"{exits['budget']} queries hit the node budget — raise MAX_NODES "
                    "(agent is still exploring when cut off).")
    if rank_gap > 0.10:
        recs.append(f"RANKING: recall@100 ({mean[100]:.2f}) >> recall@10 ({mean[10]:.2f}). "
                    "Gold is found but ranked low (padding-propped, like Arm-2B). "
                    "Add an article-scoring/re-rank step.")
    if miss:
        if nr_share >= 0.5:
            recs.append(f"NAVIGATION/TREE is the main gap — {100*nr_share:.0f}% of misses are NOT_REACHED "
                        "(the agent pruned the gold's branch — the Arm-2B failure mode). "
                        "Before other PDFs: try MODE=enriched (richer branch summaries), raise MAX_NODES, "
                        "or improve node summaries. The deep tree alone isn't closing it.")
        elif sn_share >= 0.5:
            recs.append(f"SELECTION is the main gap — {100*sn_share:.0f}% of misses are SEEN_NOT_SELECTED "
                        "(agent saw the gold in a menu but didn't pick it). This is a prompt/model issue, "
                        "NOT the tree: tune the selection instruction, lower the bar to keep an article, "
                        "or add scoring. The tree is doing its job.")
        if orph_share >= 0.25:
            recs.append(f"{100*orph_share:.0f}% of misses are ORPHAN_UNREACHED (gold in the Unfiled branch). "
                        "Enable proximity-reattach in the builder, or prompt the agent to check Unfiled.")
    if op_ok and sci_ok and nr_share < 0.5:
        recs.append("HEALTHY SUCCESS — beats Arm-2B and misses aren't dominated by navigation. "
                    "SCALE: rebuild trees+bundles for the other 4 PDFs and run each.")
    elif sci_ok:
        recs.append("Beats Arm-2B but with caveats above — scaling is reasonable; addressing the top "
                    "recommendation first will likely lift it further.")
    elif not recs:
        recs.append("Below Arm-2B with no single dominant failure mode — inspect the worst queries "
                    "above by hand before deciding.")

    P("\n[7] RECOMMENDATION")
    P(f"  OPERATIONAL={'PASS' if op_ok else 'FAIL'}  SCIENTIFIC={'PASS' if sci_ok else 'FAIL'}  "
      f">>> {'SUCCESS' if (op_ok and sci_ok) else 'NEEDS WORK'} <<<")
    for i, rc in enumerate(recs, 1):
        P(f"  ({i}) {rc}")

    report = "\n".join(L)
    print(report)

    # ── write artifacts ───────────────────────────────────────────────────
    (results_dir / "analysis_report.md").write_text(report, encoding="utf-8")
    with (results_dir / "per_query.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_query_rows[0].keys()))
        w.writeheader()
        w.writerows(per_query_rows)
    print(f"\n[wrote {results_dir/'analysis_report.md'} + per_query.csv]")
    return {"recall@10": mean[10], "success": op_ok and sci_ok,
            "miss_decomp": dict(gold_cls)}


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--tree", type=Path, required=True)
    args = ap.parse_args()
    analyze(args.results, args.bundle, args.tree)
    return 0


if __name__ == "__main__":
    sys.exit(main())
