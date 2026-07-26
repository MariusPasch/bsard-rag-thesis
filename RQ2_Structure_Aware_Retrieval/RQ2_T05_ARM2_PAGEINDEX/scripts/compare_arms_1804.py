"""Cross-arm comparison on one PDF — T03 vs every T04 variant vs T05.

Loads each arm's per-query rankings, computes R@K against the curated GT,
writes a CSV next to the local experiments output.

Note: T05 picks up whatever is currently in
`RQ2_T05_ARM2_PAGEINDEX/data/<stem>/results/` — so this script reflects
the live post-fix (and post-rederive-padding, if applied) ranking.

Usage:
    python scripts/compare_arms_1804.py                          # defaults to 1804
    python scripts/compare_arms_1804.py 1867_06_08_1867060850    # any selected PDF stem
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent.parent
KS = (1, 5, 10, 20, 100)


def dedupe_keep_order(seq):
    seen = set(); out = []
    for x in seq:
        if x in seen: continue
        seen.add(x); out.append(x)
    return out


def recall_at_k(ranked, gt, k):
    if not gt:
        return 0.0
    return len(set(ranked[:k]) & gt) / len(gt)


def score_arm(qid_to_ranked, gt):
    common = set(qid_to_ranked) & set(gt)
    rows = {f"R@{k}": [] for k in KS}
    for qid in common:
        r = qid_to_ranked[qid]; gtb = gt[qid]
        for k in KS:
            rows[f"R@{k}"].append(recall_at_k(r, gtb, k))
    return {"n_q": len(common),
            **{k: float(np.mean(v)) if v else 0.0 for k, v in rows.items()}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stem", nargs="?", default="1804_03_21_1804032150")
    args = ap.parse_args()
    STEM = args.stem
    RUN = f"t04_{STEM}"

    gt_raw = json.loads(
        (ROOT / "RQ2_T07_EVALUATION/ground_truth/runs" / f"{RUN}.json").read_text(encoding="utf-8")
    )
    gt = {str(k): {int(b) for b in v} for k, v in gt_raw.items()}

    rows: list[dict] = []

    # T03 fused (whatever hash is present under this PDF's results dir)
    t03_root = ROOT / f"RQ2_T03_ARM1_NAIVE/data/{STEM}/results"
    if t03_root.exists():
        for hash_dir in sorted(t03_root.iterdir()):
            pool_dir = hash_dir / "retrieval_pool"
            if not pool_dir.exists():
                continue
            t03 = {}
            for fp in sorted(pool_dir.glob("*.jsonl")):
                with fp.open(encoding="utf-8") as f:
                    for line in f:
                        d = json.loads(line)
                        qid = str(d["question_id"])
                        flat = []
                        for c in d.get("candidates", []):
                            flat.extend(int(b) for b in c.get("bsard_ids", []) or [])
                        t03[qid] = dedupe_keep_order(flat)
            if t03:
                rows.append({"arm": "T03 fused (dense+BM25, top-200 pool)",
                             "variant_hash": hash_dir.name, **score_arm(t03, gt)})
    else:
        print(f"  (T03 results not found for {STEM} — skipping)")

    # T04 — every variant
    t04_root = ROOT / f"RQ2_T04_ARM2_METADATA/data/{STEM}"
    for hash_dir in sorted((t04_root / "results").iterdir()):
        if not hash_dir.is_dir(): continue
        jsonls = sorted(hash_dir.glob("*.jsonl"))
        if not jsonls: continue
        cfg = json.loads(
            (t04_root / "configs" / hash_dir.name / "manifest.json").read_text(encoding="utf-8")
        )
        ch = cfg.get("chunking", {})
        label = (f"T04 {ch.get('variant')}/{ch.get('unit')}/"
                 f"{'sparse' if ch.get('sparse_only') else 'hybrid'}/"
                 f"lv{ch.get('linker_version')}")
        qid_to_ranked = {}
        with jsonls[-1].open(encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                qid = str(d["query_id"])
                flat = []
                for it in d.get("ranked_items", []):
                    md = it.get("metadata", {}) or {}
                    bids = md.get("bsard_ids") or md.get("bsard_id")
                    if bids is None: continue
                    if isinstance(bids, list):
                        flat.extend(int(b) for b in bids)
                    else:
                        flat.append(int(bids))
                qid_to_ranked[qid] = dedupe_keep_order(flat)
        rows.append({"arm": label, "variant_hash": hash_dir.name,
                     **score_arm(qid_to_ranked, gt)})

    # T05 — live results dir (= post-fix + chapter-then-law padding)
    t05_dir = ROOT / f"RQ2_T05_ARM2_PAGEINDEX/data/{STEM}/results"
    t05 = {}
    for fp in sorted(t05_dir.glob("q*.json")):
        d = json.loads(fp.read_text(encoding="utf-8"))
        qid = str(d["query_id"])
        flat = []
        for it in d.get("ranked_items", []):
            md = it.get("metadata", {}) or {}
            bid = md.get("bsard_id")
            if bid is None: continue
            flat.append(int(bid))
        t05[qid] = dedupe_keep_order(flat)
    rows.append({"arm": "T05 PageIndex (post-fix + chapter-then-law pad)",
                 "variant_hash": "-", **score_arm(t05, gt)})

    # Also score the pre-fix and post-fix-baseline snapshots for the report.
    # Snapshot dir names are per-PDF and date-stamped; scan for whichever
    # `results_pre_fix_*` and `results_baseline_post_fix_*` happen to exist.
    snap_root = ROOT / f"RQ2_T05_ARM2_PAGEINDEX/data/{STEM}"
    snap_specs: list[tuple[str, str]] = []
    for child in sorted(snap_root.iterdir()):
        if not child.is_dir(): continue
        if child.name.startswith("results_pre_fix_"):
            snap_specs.append((child.name, "T05 PageIndex (PRE-FIX, broken)"))
        elif child.name.startswith("results_baseline_post_fix_"):
            snap_specs.append((child.name, "T05 PageIndex (post-fix, law-only pad)"))
    for snap_name, label in snap_specs:
        snap_dir = snap_root / snap_name
        snap = {}
        for fp in sorted(snap_dir.glob("q*.json")):
            d = json.loads(fp.read_text(encoding="utf-8"))
            qid = str(d["query_id"])
            flat = []
            for it in d.get("ranked_items", []):
                md = it.get("metadata", {}) or {}
                bid = md.get("bsard_id")
                if bid is None: continue
                flat.append(int(bid))
            snap[qid] = dedupe_keep_order(flat)
        rows.append({"arm": label, "variant_hash": "-",
                     **score_arm(snap, gt)})

    df = pd.DataFrame(rows)
    for c in [f"R@{k}" for k in KS]:
        df[c] = df[c].round(4)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 70)
    print(df.to_string(index=False))

    out_path = ROOT / f"RQ2_T05_ARM2_PAGEINDEX/data/{STEM}/experiments/cross_arm_comparison.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
