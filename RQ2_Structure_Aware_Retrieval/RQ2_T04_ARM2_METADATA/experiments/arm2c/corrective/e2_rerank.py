"""E2 phase 2 — e5 rerank over the saved corrective pools (no Ollama).

Reads runs/_corrective_e2/<stem>_nav.json (phase 1) and ranks each saved pool
(round-1 single-pass + full corrective) with the e5 encoder, producing the four
R@10 off one navigation:

                       single-pass pool        corrective pool
  8B list-rerank       8B/single (saved)       8B/corr (saved)
  e5 embed-rerank      e5/single (= E1)        e5/corr  ← THE SYNTHESIS

Only e5 is loaded, so no VRAM contention with llama (the phase-1/phase-2 split is
exactly to avoid that). Cheap to re-run for text-field / encoder sweeps.

    python corrective/e2_rerank.py --stem 1804_03_21_1804032150 [--text-field text|summary] [--mock]

Writes runs/_corrective_e2/<stem>_e2_<field>.{csv,md}
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_ARM2C = _HERE.parent.parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_ARM2C))
from navigator_tools import DeepTree                          # noqa: E402
from embed_rerank import encode_nodes, embed_rerank_pool_cached  # noqa: E402
from eval_embed_rerank import MockEncoder                     # noqa: E402
from smoke_corrective import _reached_recall, _recall         # noqa: E402

OUT_DIR = _ARM2C / "runs" / "_corrective_e2"


def _b2l(tree: DeepTree) -> dict[int, list[str]]:
    out: dict[int, list[str]] = {}
    for n in tree.by_id.values():
        if n.is_leaf and n.bsard_id is not None:
            out.setdefault(n.bsard_id, []).append(n.node_id)
    return out


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", required=True)
    ap.add_argument("--text-field", choices=["text", "summary"], default="text")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    nav_path = OUT_DIR / f"{args.stem}_nav.json"
    if not nav_path.exists():
        print(f"no nav file {nav_path} — run e2_navigate.py first"); return 1
    store = json.loads(nav_path.read_text(encoding="utf-8"))
    rows_in = store["rows"]

    tree_path = _ARM2C / "bundles" / args.stem / "deep_tree.json"
    if not tree_path.exists():
        tree_path = _ARM2C / "data" / args.stem / "deep_tree.json"
    tree = DeepTree.load(tree_path)
    b2l = _b2l(tree)

    if args.mock:
        encoder = MockEncoder()
    else:
        from embed_rerank import E5Encoder
        print("loading e5 encoder (intfloat/multilingual-e5-large-instruct)…")
        encoder = E5Encoder(device=args.device)

    # encode every unique pool article + every query ONCE (dedup → CPU-feasible)
    all_nodes = [n for d in rows_in for n in (d.get("round1_pool", []) + d.get("corr_pool", []))]
    print(f"encoding {len(set(all_nodes))} unique articles + {len(rows_in)} queries once…")
    vec_by_node = encode_nodes(all_nodes, tree, encoder, text_field=args.text_field)
    q_vecs = encoder.encode_queries([d["query_text"] for d in rows_in])

    hdr = (f"{'qid':>6} {'ng':>3} | {'reach1':>6} {'reachC':>6} | "
           f"{'8B/1':>5} {'8B/C':>5} {'e5/1':>5} {'e5/C':>5}")
    print(hdr); print("-" * len(hdr))
    rows = []
    for i, d in enumerate(rows_in):
        gold = set(int(x) for x in d["gold_bsard_ids"])
        if not gold:
            continue
        qv = q_vecs[i]
        e5_single = embed_rerank_pool_cached(qv, d["round1_pool"], tree, vec_by_node, k=100)
        e5_corr = embed_rerank_pool_cached(qv, d["corr_pool"], tree, vec_by_node, k=100)
        reach1 = _reached_recall(tree, b2l, gold, set(d["round1_visit"]))
        reachC = _reached_recall(tree, b2l, gold, set(d["corr_visit"]))
        m = {
            "reach_single": reach1, "reach_corr": reachC,
            "r8_single": _recall(gold, d["round1_ranked"], 10),
            "r8_corr":   _recall(gold, d["corr_ranked"], 10),
            "e5_single": _recall(gold, e5_single, 10),
            "e5_corr":   _recall(gold, e5_corr, 10),
        }
        print(f"{d['qid']:>6} {len(gold):>3} | {reach1:>6.2f} {reachC:>6.2f} | "
              f"{m['r8_single']:>5.2f} {m['r8_corr']:>5.2f} {m['e5_single']:>5.2f} {m['e5_corr']:>5.2f}")
        rows.append({"qid": d["qid"], "n_gold": len(gold), **m})

    if not rows:
        print("no scored rows"); return 1
    mean = lambda k: statistics.mean(r[k] for r in rows)
    print("-" * len(hdr))
    summ = {k: mean(k) for k in ("reach_single", "reach_corr", "r8_single",
                                 "r8_corr", "e5_single", "e5_corr")}
    print(f"  mean reach {summ['reach_single']:.3f}->{summ['reach_corr']:.3f}  |  "
          f"R@10  8B/1 {summ['r8_single']:.3f}  8B/C {summ['r8_corr']:.3f}  "
          f"e5/1 {summ['e5_single']:.3f}  e5/C {summ['e5_corr']:.3f}  (n={len(rows)})")

    tag = "mock" if args.mock else args.text_field
    with (OUT_DIR / f"{args.stem}_e2_{tag}.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    L = [f"# E2 — corrective nav + e5 rerank — {args.stem} ({len(rows)} q, {tag})", "",
         "Four R@10 off one corrective navigation (phase 1) — only the pool×reranker changes.",
         "e5/C = the synthesis (Arm-2C reach + Arm-2A ranking).", "",
         f"- reach     single→corr : {summ['reach_single']:.3f} → {summ['reach_corr']:.3f}",
         f"- R@10  8B/single       : {summ['r8_single']:.3f}",
         f"- R@10  8B/corr         : {summ['r8_corr']:.3f}",
         f"- R@10  e5/single (E1)  : {summ['e5_single']:.3f}",
         f"- R@10  e5/corr  (E2)   : {summ['e5_corr']:.3f}"]
    (OUT_DIR / f"{args.stem}_e2_{tag}.md").write_text("\n".join(L), encoding="utf-8")
    print(f"\n[wrote {OUT_DIR / f'{args.stem}_e2_{tag}.csv'} + .md]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
