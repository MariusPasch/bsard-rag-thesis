"""E2 phase-2b — recover the FULL metric battery for CNR (e5_corr).

Phase 2 (e2_rerank.py) e5-ranked the saved corrective pool but persisted only the
R@10 scalar (e2_text.csv), so MRR@10 / nDCG@10 / R@5 / R@20 were unavailable for the
CNR config and the recall curve could only draw a 2-point (k=10,100) dash.

This re-runs ONLY the e5 rerank over the saved pools in <stem>_nav.json (corr_pool =
CNR, round1_pool = e5/single) — deterministic, no Ollama — and now CAPTURES the
ordered list, computing the same binary battery the Report's load_arm2c uses for
simple-nav (the metric defs are inlined verbatim below so this runs on the light
azureml env with no Report/T06 stack). e5/corr R@10 reproduces the published
e2_text.csv value exactly.

Self-contained deps: navigator_tools + embed_rerank (this package) + torch +
sentence-transformers + numpy. Needs <stem>_nav.json (phase-1 pools) and the deep
trees (bundles|data/<stem>/deep_tree.json) present.

Writes (per stem, under runs/_corrective_e2/):
  <stem>_e2_ranked.json   {qid: {"gold":[...], "e5_corr":[top-100], "e5_single":[top-100]}}
  <stem>_e2_metrics.csv   qid,n_gold,R@5,R@10,R@20,R@100,Hit@10,MRR@10,nDCG@10,P@10  (CNR=e5_corr)

    python corrective/e2_metrics_full.py [--device cuda] [--stem <stem>] [--text-field text|summary]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
_ARM2C = _HERE.parent.parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_ARM2C))

from navigator_tools import DeepTree                                   # noqa: E402
from embed_rerank import encode_nodes, embed_rerank_pool_cached        # noqa: E402

OUT = _ARM2C / "runs" / "_corrective_e2"

# (stem, display label) — the curated 5-PDF set, large-first display order.
DOCS = [
    ("1804_03_21_1804032150", "Code Civil"),
    ("1967_10_10_1967101055", "Code Judiciaire (larger)"),
    ("2003_07_17_2013A31614", "Code du Logement"),
    ("1967_10_10_1967101056", "Code Judiciaire (smaller)"),
    ("1867_06_08_1867060850", "Code Pénal"),
]
STEM_LABEL = {s: lab for s, lab in DOCS}

CURVE_K = [5, 10, 20, 100]
BATTERY = ["R@5", "R@10", "R@20", "R@100", "Hit@10", "MRR@10", "nDCG@10", "P@10"]


def _query_metrics(gold: set[int], ranked: list[int]) -> dict:
    """Binary IR metrics for one query (verbatim from Report/load_arm2c._query_metrics)."""
    g = gold
    m: dict = {}
    for k in CURVE_K:
        topk = ranked[:k]
        m[f"R@{k}"] = len(g.intersection(topk)) / len(g)
    top10 = ranked[:10]
    hit = g.intersection(top10)
    m["Hit@10"] = 1.0 if hit else 0.0
    m["P@10"] = len(hit) / 10.0
    rr = 0.0
    for i, a in enumerate(top10):
        if a in g:
            rr = 1.0 / (i + 1)
            break
    m["MRR@10"] = rr
    dcg = sum((1.0 / math.log2(i + 2)) for i, a in enumerate(top10) if a in g)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(g), 10)))
    m["nDCG@10"] = (dcg / idcg) if idcg > 0 else 0.0
    return m


def _tree(stem: str) -> DeepTree:
    for p in (_ARM2C / "bundles" / stem / "deep_tree.json",
              _ARM2C / "data" / stem / "deep_tree.json"):
        if p.exists():
            return DeepTree.load(p)
    raise FileNotFoundError(f"no deep_tree for {stem}")


def run_stem(stem: str, encoder, text_field: str = "text") -> list[dict]:
    nav = json.loads((OUT / f"{stem}_nav.json").read_text(encoding="utf-8"))
    rows_in = nav["rows"]
    tree = _tree(stem)
    all_nodes = [n for d in rows_in for n in (d.get("round1_pool", []) + d.get("corr_pool", []))]
    # Disk-cache the (deterministic) article vectors so this never re-encodes.
    cache = OUT / f"{stem}_e5vec_{text_field}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        vec = {str(i): v for i, v in zip(z["ids"], z["mat"])}
        print(f"  [{stem}] loaded {len(vec)} cached article vectors", flush=True)
    else:
        vec = encode_nodes(all_nodes, tree, encoder, text_field=text_field)
        if vec:
            np.savez(cache, ids=np.array(list(vec)), mat=np.vstack(list(vec.values())))
        print(f"  [{stem}] encoded {len(vec)} article vectors -> cached", flush=True)
    qvecs = encoder.encode_queries([d["query_text"] for d in rows_in])

    per_q, ranked = [], {}
    for i, d in enumerate(rows_in):
        gold = set(int(x) for x in d["gold_bsard_ids"])
        if not gold:
            continue
        e5_corr = [int(x) for x in embed_rerank_pool_cached(qvecs[i], d["corr_pool"], tree, vec, k=100)]
        e5_single = [int(x) for x in embed_rerank_pool_cached(qvecs[i], d["round1_pool"], tree, vec, k=100)]
        m = _query_metrics(gold, e5_corr)                   # CNR = e5 over corrective pool
        per_q.append({"qid": d["qid"], "n_gold": len(gold), **{k: m[k] for k in BATTERY}})
        ranked[str(d["qid"])] = {"gold": sorted(gold), "e5_corr": e5_corr, "e5_single": e5_single}

    with (OUT / f"{stem}_e2_metrics.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["qid", "n_gold", *BATTERY]); w.writeheader(); w.writerows(per_q)
    (OUT / f"{stem}_e2_ranked.json").write_text(json.dumps(ranked), encoding="utf-8")
    return per_q


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", default=None, help="one stem; default = all 5")
    ap.add_argument("--text-field", choices=["text", "summary"], default="text")
    ap.add_argument("--device", default=None, help="cuda | cpu | None(auto)")
    args = ap.parse_args()

    try:
        import torch
        torch.set_num_threads(os.cpu_count() or 8)
    except Exception:
        pass
    stems = [args.stem] if args.stem else [s for s, _ in DOCS]
    from embed_rerank import E5Encoder
    print("loading e5 encoder (intfloat/multilingual-e5-large-instruct)…", flush=True)
    enc = E5Encoder(device=args.device)

    grand_n = 0
    grand = {k: 0.0 for k in BATTERY}
    for stem in stems:
        if not (OUT / f"{stem}_nav.json").exists():
            print(f"  skip {stem} (no nav.json)", flush=True); continue
        per_q = run_stem(stem, enc, text_field=args.text_field)
        n = len(per_q)
        means = {k: sum(r[k] for r in per_q) / n for k in BATTERY}
        for k in BATTERY:
            grand[k] += means[k] * n
        grand_n += n
        print(f"  {STEM_LABEL.get(stem, stem):26s} n={n:4d}  "
              f"R@5 {means['R@5']:.3f}  R@10 {means['R@10']:.3f}  R@20 {means['R@20']:.3f}  "
              f"R@100 {means['R@100']:.3f}  MRR {means['MRR@10']:.3f}  nDCG {means['nDCG@10']:.3f}", flush=True)

    if grand_n:
        print("-" * 70)
        print(f"  {'CORPUS (micro)':26s} n={grand_n:4d}  " +
              "  ".join(f"{k} {grand[k]/grand_n:.3f}" for k in ["R@5", "R@10", "R@20", "R@100", "MRR@10", "nDCG@10"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
