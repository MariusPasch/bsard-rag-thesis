"""Build the Azure run bundle for one PDF: queries (qid->text+clipped gold) +
baselines (Arm-2A / Arm-2B mean recall@10 on this PDF, for the pass/fail check).

queries = test-split BSARD questions whose gold intersects this PDF's reachable
bsard_ids (the deep_tree leaves) — same GT-clipping the cross-arm eval uses.

Run locally (any venv): python experiments/arm2c/prepare_bundle.py --doc-id <stem>
Writes experiments/arm2c/bundles/<stem>/{queries.json, baselines.json}.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sqlite3
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_RQ2 = _HERE.parents[3]
# $RQ2_BSARD_DB  →  <RQ2_DATA_DIR or repo>/data/bsard_corpus.db
BSARD_DB = (Path(os.environ["RQ2_BSARD_DB"]).expanduser()
            if os.environ.get("RQ2_BSARD_DB")
            else _RQ2 / "data" / "bsard_corpus.db")
T06_TABLES = _RQ2 / "RQ2_T06_ARM_RESULTS" / "data" / "tables"


def reachable_bsard_ids(doc_id: str) -> set[int]:
    tree = json.loads((_HERE.parent / "data" / doc_id / "deep_tree.json").read_text(encoding="utf-8"))
    ids: set[int] = set()

    def walk(n):
        if n.get("article_id") is not None:
            if n.get("bsard_id") is not None:
                ids.add(n["bsard_id"])
        for c in n.get("sub_nodes", []):
            walk(c)
    walk(tree["tree"])
    return ids


def _parse_ids(raw) -> list[int]:
    if raw is None:
        return []
    if isinstance(raw, (int, float)):
        return [int(raw)]
    s = str(raw).strip()
    for loader in (json.loads,):
        try:
            v = loader(s)
            if isinstance(v, list):
                return [int(x) for x in v]
        except Exception:
            pass
    return [int(x) for x in s.replace("[", "").replace("]", "").split(",") if x.strip().lstrip("-").isdigit()]


def build_queries(doc_id: str, reachable: set[int], split: str = "all") -> list[dict]:
    """split='all' => train+test (matches the T06 baselines, which evaluated every
    question touching the PDF). 'test'/'train' restrict to that split."""
    con = sqlite3.connect(str(BSARD_DB))
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT question_id, question_text, relevant_bsard_ids, split FROM questions"
    ).fetchall()
    con.close()
    out = []
    for r in rows:
        if split != "all" and (r["split"] or "").lower() != split:
            continue
        gold_all = _parse_ids(r["relevant_bsard_ids"])
        gold = sorted(set(gold_all) & reachable)
        if not gold:
            continue                      # question doesn't touch this PDF
        out.append({"query_id": str(r["question_id"]),
                    "query_text": r["question_text"],
                    "gold_bsard_ids": gold,
                    "n_gold_full": len(set(gold_all))})
    out.sort(key=lambda q: int(q["query_id"]))
    return out


def baseline_recall10(doc_id: str, qids: set[str]) -> dict:
    """Mean recall@10 on this PDF for Arm-1/2A/2B over the SAME qids, canonical
    config only (Arm-2A has 3 variants in the CSV → keep summary_node; Arm-1/2B
    each have a single variant)."""
    CANON = {"arm2a": ("T04_summary_node_hybrid", "node")}
    res = {}
    for arm, fname in (("arm1", "arm1_per_question.csv"), ("arm2a", "arm2a_per_question.csv"),
                       ("arm2b", "arm2b_per_question.csv")):
        p = T06_TABLES / fname
        if not p.exists():
            continue
        vals = []
        for row in csv.DictReader(p.open(encoding="utf-8")):
            if row.get("stem") != doc_id or row.get("qid") not in qids:
                continue
            if arm in CANON and (row.get("variant"), row.get("unit")) != CANON[arm]:
                continue
            try:
                vals.append(float(row["recall@10"]))
            except (KeyError, ValueError):
                pass
        if vals:
            res[arm] = {"mean_recall@10": round(sum(vals) / len(vals), 4), "n_questions": len(vals)}
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc-id", default="1804_03_21_1804032150")
    ap.add_argument("--split", default="all", choices=["all", "test", "train"])
    args = ap.parse_args()

    reachable = reachable_bsard_ids(args.doc_id)
    queries = build_queries(args.doc_id, reachable, split=args.split)
    baselines = baseline_recall10(args.doc_id, {q["query_id"] for q in queries})

    out_dir = _HERE.parent / "bundles" / args.doc_id
    out_dir.mkdir(parents=True, exist_ok=True)
    # Copy the deep tree INTO the bundle so the run inputs are self-contained and
    # committable — experiments/arm2c/data/ is gitignored (data/ rule), bundles/ isn't.
    import shutil
    src_tree = _HERE.parent / "data" / args.doc_id / "deep_tree.json"
    shutil.copyfile(src_tree, out_dir / "deep_tree.json")
    (out_dir / "queries.json").write_text(json.dumps(queries, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "baselines.json").write_text(json.dumps(
        {"doc_id": args.doc_id, "split": args.split, "n_reachable_bsard": len(reachable),
         "n_queries": len(queries), "baselines_recall@10": baselines},
        ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"doc_id={args.doc_id}  reachable_bsard={len(reachable)}  "
          f"{args.split}-queries touching this PDF: {len(queries)}")
    tot_gold = sum(len(q["gold_bsard_ids"]) for q in queries)
    print(f"  total clipped gold across queries: {tot_gold}  "
          f"(mean {tot_gold/max(len(queries),1):.1f}/q)")
    print(f"  baselines recall@10: {baselines}")
    print(f"  wrote {out_dir}/queries.json + baselines.json")
    print(f"\n  sample query: {queries[0]['query_id']} — {queries[0]['query_text'][:70]!r}")
    print(f"    gold: {queries[0]['gold_bsard_ids']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
