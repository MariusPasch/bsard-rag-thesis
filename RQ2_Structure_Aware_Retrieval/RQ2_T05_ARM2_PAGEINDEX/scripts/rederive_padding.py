"""Rewrite per-query q*.json files in place with a new padding strategy.

Pure post-processing: the LLM-scored picks (score > 0) are kept; only the
padding tail (score == 0) is regenerated from the tree under the new
two-tier `selected_chapter THEN selected_law` policy. No LLM calls.

Use when you've patched the navigator's padding block and want to apply the
same change to existing results without paying for an Azure re-run.

Usage:
    python scripts/rederive_padding.py 1804_03_21_1804032150
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
T05_BASE = ROOT / "RQ2_T05_ARM2_PAGEINDEX/data"


def find_node(root: dict, node_id: str) -> dict | None:
    if root.get("node_id") == node_id:
        return root
    for ch in root.get("sub_nodes", []):
        hit = find_node(ch, node_id)
        if hit is not None:
            return hit
    return None


def iter_leaves(node: dict):
    if node.get("article_id"):
        yield node
        return
    for ch in node.get("sub_nodes", []):
        yield from iter_leaves(ch)


def laws_in_tree(root: dict) -> list[dict]:
    if root.get("node_id", "").startswith("LAW_"):
        return [root]
    return [n for n in root.get("sub_nodes", []) if n.get("node_id", "").startswith("LAW_")]


def leaf_to_ranked_item(leaf: dict, score: float) -> dict:
    """Mirror navigator/retriever._ranked_items_from_state output shape."""
    md = dict(leaf.get("metadata") or {})
    md["bsard_id"] = int(leaf["bsard_id"])
    if leaf.get("text") and "text" not in md:
        md["text"] = leaf["text"]
    return {
        "id": leaf["node_id"],
        "score": float(score),
        "text": leaf.get("text") or "",
        "metadata": md,
    }


def rederive_one(qpath: Path, tree_root: dict, pad_to_k: int = 100) -> dict:
    d = json.loads(qpath.read_text(encoding="utf-8"))
    summary = (d.get("trace") or {}).get("summary", {})
    selected_chapter_ids = summary.get("selected_chapter_ids") or []
    selected_law_ids = summary.get("selected_law_ids") or [
        law["node_id"] for law in laws_in_tree(tree_root)
    ]

    # Preserve the LLM-scored picks exactly (top of the ranking).
    llm_items = [it for it in d.get("ranked_items", []) if (it.get("score") or 0) > 0]
    seen_bsard = {int(it["metadata"]["bsard_id"]) for it in llm_items}

    new_items = list(llm_items)

    # Two-tier pad: chapters first, then laws — same source/order as the
    # patched navigator block in src/arm2_pageindex/navigator.py.
    for source_ids in (selected_chapter_ids, selected_law_ids):
        for src_id in source_ids:
            node = find_node(tree_root, src_id)
            if node is None:
                continue
            for leaf in iter_leaves(node):
                bid = leaf.get("bsard_id")
                if bid is None:
                    continue
                bid = int(bid)
                if bid in seen_bsard:
                    continue
                seen_bsard.add(bid)
                new_items.append(leaf_to_ranked_item(leaf, 0.0))
                if len(new_items) >= pad_to_k:
                    break
            if len(new_items) >= pad_to_k:
                break
        if len(new_items) >= pad_to_k:
            break

    d["ranked_items"] = new_items
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stem", help="PDF stem, e.g. 1804_03_21_1804032150")
    ap.add_argument("--pad-to-k", type=int, default=100)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    stem_dir = T05_BASE / args.stem
    tree_root = json.loads((stem_dir / "tree.json").read_text(encoding="utf-8"))["tree"]
    results_dir = stem_dir / "results"

    qfiles = sorted(results_dir.glob("q*.json"))
    print(f"Rewriting {len(qfiles)} files in {results_dir} (pad_to_k={args.pad_to_k})")
    if args.dry_run:
        print("DRY RUN — no files will be modified.")

    n_changed = 0
    n_unchanged = 0
    delta_sizes = []
    for fp in qfiles:
        before = json.loads(fp.read_text(encoding="utf-8"))
        after = rederive_one(fp, tree_root, args.pad_to_k)
        before_bids = [int(it["metadata"]["bsard_id"]) for it in before["ranked_items"]]
        after_bids = [int(it["metadata"]["bsard_id"]) for it in after["ranked_items"]]
        if before_bids == after_bids:
            n_unchanged += 1
            continue
        delta_sizes.append(len(set(after_bids) - set(before_bids)))
        n_changed += 1
        if not args.dry_run:
            fp.write_text(
                json.dumps(after, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    print(f"\nChanged: {n_changed}/{len(qfiles)}  Unchanged: {n_unchanged}/{len(qfiles)}")
    if delta_sizes:
        import statistics
        print(f"  new-bsard_ids per changed file: "
              f"mean={statistics.mean(delta_sizes):.1f}  "
              f"median={int(statistics.median(delta_sizes))}  "
              f"max={max(delta_sizes)}")


if __name__ == "__main__":
    main()
