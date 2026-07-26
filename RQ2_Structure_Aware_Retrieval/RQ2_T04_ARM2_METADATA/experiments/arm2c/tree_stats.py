"""Honest navigation-cost stats for the Arm-2C deep trees.

Reads each experiments/arm2c/data/<doc_id>/deep_tree.json (NO corpus reload),
separates the catch-all orphan bucket from real navigation nodes, and reports
the decision sizes a top-down navigator actually faces:

  * coarse_L1     - #top-level groups (excl. orphan) = the first prune
  * real fan-out  - per internal node, EXCLUDING the orphan bucket
  * orphan_bucket - size of the catch-all fallback branch
  * max_depth     - law root -> deepest leaf

Compared against Arm-2B's flat single-call chapter set (~95). Optionally
rewrites each manifest's stats block in place (--write).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[3] / "RQ2_T05_ARM2_PAGEINDEX" / "src"))
from arm2_pageindex.tree_builder import ToCNode, iter_leaves  # noqa: E402

CURATED = ["1804_03_21_1804032150", "1867_06_08_1867060850",
           "1967_10_10_1967101055", "1967_10_10_1967101056", "2003_07_17_2013A31614"]
ARM2B_FLAT = {  # Arm-2B's flat chapter pick-set size (from header_chain audit)
    "1804_03_21_1804032150": 108, "1867_06_08_1867060850": 110,
    "1967_10_10_1967101055": 128, "1967_10_10_1967101056": 40,
    "2003_07_17_2013A31614": 91,
}


def _pct(xs, p):
    if not xs:
        return 0
    s = sorted(xs)
    return s[min(len(s) - 1, int(round((p / 100) * (len(s) - 1))))]


def honest_stats(root: ToCNode) -> dict:
    leaves = list(iter_leaves(root))
    orphan = next((c for c in root.sub_nodes if c.metadata.get("orphan")), None)
    orphan_size = len(list(iter_leaves(orphan))) if orphan else 0

    real_fanouts, depths = [], []

    def walk(n: ToCNode, depth: int, in_orphan: bool):
        if n.is_leaf:
            depths.append(depth)
            return
        is_orphan = in_orphan or bool(n.metadata.get("orphan"))
        if not is_orphan:
            real_fanouts.append(len(n.sub_nodes))
        for c in n.sub_nodes:
            walk(c, depth + 1, is_orphan)

    walk(root, 0, False)
    coarse_L1 = len([c for c in root.sub_nodes if not c.metadata.get("orphan")])
    return {
        "n_articles": len(leaves),
        "max_depth": max(depths) if depths else 0,
        "coarse_L1": coarse_L1,
        "real_fanout_mean": round(statistics.mean(real_fanouts), 1) if real_fanouts else 0,
        "real_fanout_median": statistics.median(real_fanouts) if real_fanouts else 0,
        "real_fanout_p90": _pct(real_fanouts, 90),
        "real_fanout_max": max(real_fanouts) if real_fanouts else 0,
        "orphan_bucket": orphan_size,
        "orphan_pct": round(100 * orphan_size / max(len(leaves), 1), 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=_HERE.parent / "data")
    ap.add_argument("--write", action="store_true", help="refresh manifest stats in place")
    args = ap.parse_args()

    print(f"{'doc_id':>30} {'arts':>4} {'depth':>5} {'L1':>3} "
          f"{'real fan mean/med/p90/max':>26} {'orphan':>9}  vs Arm-2B flat")
    print("-" * 108)
    rows = []
    for stem in CURATED:
        p = args.data / stem / "deep_tree.json"
        if not p.exists():
            print(f"{stem:>30}  (missing)")
            continue
        payload = json.loads(p.read_text(encoding="utf-8"))
        root = ToCNode.from_dict(payload["tree"])
        st = honest_stats(root)
        rows.append(st)
        print(f"{stem:>30} {st['n_articles']:>4} {st['max_depth']:>5} {st['coarse_L1']:>3} "
              f"{st['real_fanout_mean']:>6}/{st['real_fanout_median']}/{st['real_fanout_p90']}/{st['real_fanout_max']:<6} "
              f"{st['orphan_bucket']:>3} ({st['orphan_pct']:>4}%)  {ARM2B_FLAT.get(stem,'?'):>4}-way 1-shot")
        if args.write:
            payload["manifest"].update({f"stat_{k}": v for k, v in st.items()})
            p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("-" * 108)
    if rows:
        print(f"\nAGG  coarse L1: mean {round(statistics.mean(r['coarse_L1'] for r in rows),1)} "
              f"(vs Arm-2B flat mean {round(statistics.mean(ARM2B_FLAT.values()),1)})")
        print(f"     real per-node fan-out: median {round(statistics.mean(r['real_fanout_median'] for r in rows),1)}, "
              f"worst section {max(r['real_fanout_max'] for r in rows)} articles")
        print(f"     orphan fallback: {sum(r['orphan_bucket'] for r in rows)} articles "
              f"({round(100*sum(r['orphan_bucket'] for r in rows)/sum(r['n_articles'] for r in rows),1)}%)")
    if args.write:
        print("\n(manifests refreshed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
