"""Tier (a) — ZERO-LLM counterfactual ceiling for the Arm-2C corrective loop.

Reads the SAVED single-pass run traces (runs/arm2c_<stem>_enriched_rerank/q*.json)
+ the deep tree (bundles/<stem>/deep_tree.json), and — without any model call —
estimates how much each corrective mode COULD recover, before we spend a GPU run.

For every gold article in every query we reconstruct, from the trace alone:
  visited      : node_ids the agent expanded (steps[].node_id)
  reached pool : leaf-children of visited nodes (the recall@100-bearing set)
  top10        : ranked_bsard_ids[:10]   (the shipped headline)
and classify the gold's fate + recoverability:

  HIT       gold already in top10                          -> nothing to recover
  RESELECT  reached (its section was expanded) but not top10
            -> RE-SELECT mode upside (re-rank/commit the pool)
  FINE      NOT reached, but its TOP branch WAS entered and its branch hangs off a
            node the agent committed to (pruned mid-tree)
            -> RE-NAVIGATE upside, HIGH confidence (the discriminator only has to
               beat the few pruned siblings at a node we already chose correctly).
            Annotated `deep` when the gold sits >=3 edges below the deepest visited
            ancestor (the brief's "N-round": reachable, but the inner pass must make
            several more correct choices and may re-prune).
  COARSE    NOT reached, and the gold's TOP branch was never entered (the agent
            rejected the right Livre/Titre at the first multi-way node)
            -> RE-NAVIGATE upside, LOW confidence: a re-rank sees the SAME titles
               that already lost, so it likely re-rejects. The brief's
               "unreachable-without-reformulation".
  ORPHAN    gold sits in the Unfiled catch-all branch
            -> the loop CANNOT fix this (needs a tree reattach). Negative control.

IMPORTANT — these are ORACLE ceilings (assume perfect targeting / perfect inner
descent). A real corrective loop recovers <= the ceiling, discounted by grader and
pruned-section-ranking calibration. The `deep` annotation and the COARSE/FINE split
are exactly the levers that discount it.

Reuses navigator_tools.DeepTree (stdlib-only) and mirrors the gold->tree mapping in
analyze_results._classify_gold / load_arm2c so the taxonomy stays consistent with
the published miss-anatomy. Writes are additive only:
  runs/_corrective_ceiling/ceiling_<stem>.csv   (per-query)
  runs/_corrective_ceiling/CEILING_SUMMARY.md   (per-PDF table + predictions)

    python corrective/corrective_ceiling.py
"""
from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_ARM2C = _HERE.parent.parent                       # experiments/arm2c
sys.path.insert(0, str(_ARM2C))
from navigator_tools import DeepTree                # noqa: E402  (stdlib-only)

# Documents in the brief's test order: depth/fine-prune docs first (where
# backtracking should help most), housing flagged as the negative control.
DOCS: list[tuple[str, str, str]] = [
    # (stem, label, role)
    ("1867_06_08_1867060850", "Code Penal",                "depth / fine-prune"),
    ("1804_03_21_1804032150", "Code Civil",                "depth / select-gap"),
    ("1967_10_10_1967101055", "Code Judiciaire (larger)",  "deep clean code"),
    ("1967_10_10_1967101056", "Code Judiciaire (smaller)", "oversized section"),
    ("2003_07_17_2013A31614", "Housing/leases (mislabel)", "NEGATIVE CONTROL"),
]

REGISTER = "enriched_rerank"
OUT_DIR = _ARM2C / "runs" / "_corrective_ceiling"


# ── tree maps (mirror analyze_results._tree_maps + the first-multiway logic) ──

def _build_maps(tree: DeepTree):
    """Return per-tree lookups used by the classifier.

    bsard_to_leaves : bsard_id -> [leaf node_ids]
    orphan_leaves   : set of leaf node_ids under an Unfiled/orphan branch
    depth           : node_id -> depth in edges from root
    multiway_id     : node_id of the first node with >1 child (collapsing the
                      single-child root chain) -- the top-level branch decision
    top_branch_of   : node_id -> the ancestor that is a direct child of multiway
                      (i.e. which top branch this node lives under); None if it IS
                      at/above the multiway node
    """
    bsard_to_leaves: dict[int, list[str]] = {}
    orphan_leaves: set[str] = set()
    depth: dict[str, int] = {}

    def walk(node, d, under_orphan):
        depth[node.node_id] = d
        is_orphan = under_orphan or bool((node.metadata or {}).get("orphan"))
        if node.is_leaf:
            if node.bsard_id is not None:
                bsard_to_leaves.setdefault(node.bsard_id, []).append(node.node_id)
            if is_orphan:
                orphan_leaves.add(node.node_id)
            return
        for c in node.sub_nodes:
            walk(c, d + 1, is_orphan)

    walk(tree.root, 0, False)

    # first multi-way node (collapse single-child chain from the root)
    m = tree.root
    while len(m.sub_nodes) == 1:
        m = m.sub_nodes[0]
    multiway_id = m.node_id
    top_children = {c.node_id for c in m.sub_nodes}

    # top_branch_of: walk up until the node whose parent is the multiway node
    top_branch_of: dict[str, str | None] = {}
    for nid in depth:
        cur = nid
        tb = None
        while cur is not None:
            if cur in top_children:
                tb = cur
                break
            cur = tree.parent.get(cur)
        top_branch_of[nid] = tb

    return {
        "bsard_to_leaves": bsard_to_leaves, "orphan_leaves": orphan_leaves,
        "depth": depth, "multiway_id": multiway_id, "top_branch_of": top_branch_of,
    }


def _ancestors(tree: DeepTree, nid: str) -> list[str]:
    """root..nid (inclusive), top-down."""
    out = []
    cur = nid
    while cur is not None:
        out.append(cur)
        cur = tree.parent.get(cur)
    return out[::-1]


# ── per-gold classification ──────────────────────────────────────────────────

FATES = ["HIT", "RESELECT", "FINE", "COARSE", "ORPHAN", "NOT_IN_TREE"]


def _classify_one_leaf(tree, maps, lf, visited, top10_has, selected_has):
    """Classify a single gold leaf; returns (fate, depth_below_d, descended_dummy).
    depth_below_d = edges from the deepest visited ancestor to the leaf (the inner
    descent difficulty for the FINE bucket); 0 for non-FINE."""
    parent = tree.parent.get(lf)
    reached = parent in visited
    if top10_has(lf):
        return "HIT", 0
    if reached:
        return "RESELECT", 0
    # NOT reached
    if lf in maps["orphan_leaves"]:
        return "ORPHAN", 0
    tb = maps["top_branch_of"].get(lf)
    if tb is None or tb not in visited:
        return "COARSE", 0
    # FINE: top branch entered, gold pruned somewhere below it.
    anc = _ancestors(tree, lf)
    d = None
    for a in anc[:-1]:                       # exclude the leaf itself
        if a in visited:
            d = a
    depth_below = maps["depth"][lf] - maps["depth"][d] if d is not None else maps["depth"][lf]
    return "FINE", depth_below


def _classify_gold(tree, maps, g, visited, top10, selected):
    """Best-case fate over a gold's leaves (HIT>RESELECT>FINE>COARSE>ORPHAN)."""
    leaves = maps["bsard_to_leaves"].get(g, [])
    if not leaves:
        return "NOT_IN_TREE", 0
    top10_has = lambda lf: tree.node(lf).bsard_id in top10
    selected_has = lambda lf: tree.node(lf).bsard_id in selected
    order = {"HIT": 0, "RESELECT": 1, "FINE": 2, "COARSE": 3, "ORPHAN": 4}
    best, best_depth = "ORPHAN", 0
    for lf in leaves:
        fate, depth_below = _classify_one_leaf(tree, maps, lf, visited, top10_has, selected_has)
        if order[fate] < order.get(best, 99) or best is None:
            best, best_depth = fate, depth_below
    return best, best_depth


# ── per-PDF pass ─────────────────────────────────────────────────────────────

DEEP_THRESH = 3      # FINE gold >=3 edges below deepest visited ancestor = "N-round"


def process_stem(stem: str) -> dict | None:
    tree_path = _ARM2C / "bundles" / stem / "deep_tree.json"
    if not tree_path.exists():
        tree_path = _ARM2C / "data" / stem / "deep_tree.json"
    run_dir = _ARM2C / "runs" / f"arm2c_{stem}_{REGISTER}"
    qfiles = sorted(run_dir.glob("q*.json"))
    if not tree_path.exists() or not qfiles:
        print(f"  [skip {stem}] tree or run missing")
        return None
    tree = DeepTree.load(tree_path)
    maps = _build_maps(tree)

    # gold-article-weighted accumulators
    gold_fate = {f: 0 for f in FATES}
    fine_deep = 0
    # question-weighted projected recalls (oracle)
    rec_now, rec_renav, rec_renav_opt = [], [], []
    rec_reselect, rec_both, rec_ceiling = [], [], []
    reached_recall, selected_recall = [], []
    per_query_rows = []

    for f in qfiles:
        d = json.loads(f.read_text(encoding="utf-8"))
        gold = [int(x) for x in d.get("gold_bsard_ids", [])]
        if not gold:
            continue
        visited = {s.get("node_id") for s in d.get("steps", [])}
        top10 = set(int(x) for x in d.get("ranked_bsard_ids", [])[:10])
        selected = set(int(x) for x in d.get("selected_bsard_ids", []))

        hit, reselect, fine, coarse, orphan = set(), set(), set(), set(), set()
        ng = len(gold)
        deep_here = 0
        for g in gold:
            fate, depth_below = _classify_gold(tree, maps, g, visited, top10, selected)
            gold_fate[fate] += 1
            if fate == "HIT":
                hit.add(g)
            elif fate == "RESELECT":
                reselect.add(g)
            elif fate == "FINE":
                fine.add(g)
                if depth_below >= DEEP_THRESH:
                    fine_deep += 1
                    deep_here += 1
            elif fate == "COARSE":
                coarse.add(g)
            elif fate == "ORPHAN":
                orphan.add(g)
        G = set(gold)
        r = lambda s: len(s) / ng
        recall_now = r(hit)
        recall_renav = r(hit | fine)
        recall_renav_opt = r(hit | fine | coarse)
        recall_reselect = r(hit | reselect)
        recall_both = r(hit | fine | reselect)
        recall_ceil = r(hit | fine | coarse | reselect)          # all but orphan
        # padding-free cross-checks
        reached = {g for g in gold if tree.parent.get(
            maps["bsard_to_leaves"].get(g, [None])[0]) in visited} \
            if all(maps["bsard_to_leaves"].get(g) for g in gold) else set()
        reached_q = sum(any(tree.parent.get(lf) in visited
                            for lf in maps["bsard_to_leaves"].get(g, []))
                        for g in gold) / ng
        selected_q = len(G & selected) / ng

        rec_now.append(recall_now); rec_renav.append(recall_renav)
        rec_renav_opt.append(recall_renav_opt); rec_reselect.append(recall_reselect)
        rec_both.append(recall_both); rec_ceiling.append(recall_ceil)
        reached_recall.append(reached_q); selected_recall.append(selected_q)
        per_query_rows.append({
            "qid": d.get("query_id"), "n_gold": ng,
            "hit": len(hit), "reselect": len(reselect), "fine": len(fine),
            "fine_deep": deep_here, "coarse": len(coarse), "orphan": len(orphan),
            "recall_now": round(recall_now, 3),
            "recall_renav_fine": round(recall_renav, 3),
            "recall_reselect": round(recall_reselect, 3),
            "recall_both": round(recall_both, 3),
            "recall_ceiling": round(recall_ceil, 3),
            "reached_recall": round(reached_q, 3),
            "selected_recall": round(selected_q, 3),
        })

    n = len(per_query_rows)
    total_gold = sum(gold_fate[f] for f in ("HIT", "RESELECT", "FINE", "COARSE", "ORPHAN"))
    miss = total_gold - gold_fate["HIT"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / f"ceiling_{stem}.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_query_rows[0].keys()))
        w.writeheader(); w.writerows(per_query_rows)

    M = statistics.mean
    return {
        "stem": stem, "n": n, "total_gold": total_gold, "miss": miss,
        "gold_fate": gold_fate, "fine_deep": fine_deep,
        # question-weighted oracle recall projections
        "R@10_now": M(rec_now), "R10_renav_fine": M(rec_renav),
        "R10_renav_opt": M(rec_renav_opt), "R10_reselect": M(rec_reselect),
        "R10_both": M(rec_both), "R10_ceiling": M(rec_ceiling),
        "reached_recall": M(reached_recall), "selected_recall": M(selected_recall),
    }


# ── summary report ───────────────────────────────────────────────────────────

def _pct(x, d): return f"{100*x/d:.0f}%" if d else "—"


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    results = []
    print("=== Arm-2C corrective-loop ORACLE ceiling (zero-LLM) ===\n")
    for stem, label, role in DOCS:
        print(f"[{label}] ({role})")
        r = process_stem(stem)
        if r:
            r["label"], r["role"] = label, role
            results.append(r)
            gf = r["gold_fate"]
            print(f"  n={r['n']} gold={r['total_gold']}  "
                  f"R@10_now={r['R@10_now']:.3f}  reached={r['reached_recall']:.3f}  "
                  f"selected={r['selected_recall']:.3f}")
            print(f"  fate: HIT={gf['HIT']} RESELECT={gf['RESELECT']} "
                  f"FINE={gf['FINE']}(deep {r['fine_deep']}) COARSE={gf['COARSE']} "
                  f"ORPHAN={gf['ORPHAN']}")
            print(f"  oracle R@10:  re-navigate(FINE) {r['R@10_now']:.3f}->{r['R10_renav_fine']:.3f}"
                  f"   re-select {r['R@10_now']:.3f}->{r['R10_reselect']:.3f}"
                  f"   both {r['R10_both']:.3f}   ceiling {r['R10_ceiling']:.3f}\n")

    # markdown
    L = ["# Arm-2C corrective-loop ceiling — zero-LLM counterfactual",
         "",
         "**Oracle upper bounds** on what each corrective mode could recover from the "
         "saved single-pass `enriched_rerank` traces. A real loop recovers ≤ these, "
         "discounted by pruned-section-ranking and grader calibration. Fractions over "
         "gold articles; recall projections are question-weighted (comparable to the "
         "published R@10).",
         "",
         "## Per-PDF gold fate (article-weighted)",
         "",
         "| PDF | role | n_q | gold | R@10 now | HIT | RESELECT | FINE (deep) | COARSE | ORPHAN |",
         "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for r in results:
        gf = r["gold_fate"]; tg = r["total_gold"]
        L.append(f"| {r['label']} | {r['role']} | {r['n']} | {tg} | "
                 f"{r['R@10_now']:.3f} | {_pct(gf['HIT'],tg)} | {_pct(gf['RESELECT'],tg)} | "
                 f"{_pct(gf['FINE'],tg)} ({r['fine_deep']}) | {_pct(gf['COARSE'],tg)} | "
                 f"{_pct(gf['ORPHAN'],tg)} |")

    L += ["",
          "## Oracle Recall@10 projections (question-weighted)",
          "",
          "- **re-nav FINE** = recover all FINE gold (high-confidence backtrack).",
          "- **re-nav opt** = + COARSE (optimistic; needs reformulation, likely won't hold).",
          "- **re-select** = recover all reached-but-not-top10 gold.",
          "- **both** = FINE + re-select. **ceiling** = everything but ORPHAN.",
          "",
          "| PDF | now | re-nav FINE | re-nav opt | re-select | both | ceiling | reached | selected |",
          "|---|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for r in results:
        L.append(f"| {r['label']} | {r['R@10_now']:.3f} | {r['R10_renav_fine']:.3f} | "
                 f"{r['R10_renav_opt']:.3f} | {r['R10_reselect']:.3f} | {r['R10_both']:.3f} | "
                 f"{r['R10_ceiling']:.3f} | {r['reached_recall']:.3f} | {r['selected_recall']:.3f} |")

    L += ["",
          "## Reading the predictions",
          "",
          "- **FINE share / re-nav lift** = the re-navigate (backtrack) upside. High on "
          "depth/fine-prune docs where the agent entered the right top branch then pruned.",
          "- **RESELECT share / re-select lift** = the re-select upside (gold reached, "
          "ranked out of top-10). High where reach ≫ select.",
          "- **COARSE + ORPHAN** = the loop's blind spot: the right top branch was never "
          "entered (needs reformulation) or the gold is orphaned (needs a tree fix). A doc "
          "dominated by these is a **negative control** for the corrective loop.",
          "- `deep` FINE = gold ≥3 edges below the deepest committed node: reachable by "
          "backtrack but the inner pass must make several more correct choices (the brief's "
          "\"N-round\"; the oracle ceiling ignores inner re-pruning, so discount these).",
          ""]
    (OUT_DIR / "CEILING_SUMMARY.md").write_text("\n".join(L), encoding="utf-8")
    print(f"[wrote {OUT_DIR/'CEILING_SUMMARY.md'} + ceiling_<stem>.csv x{len(results)}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
