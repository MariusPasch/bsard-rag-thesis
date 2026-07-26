"""Arm-2C results check — did the run succeed or fail?

Reads per-query result JSONs (q<qid>.json) produced by the Azure run, computes
recall@10 / recall@100 / hit@10 / MRR@10 against the clipped gold, aggregates,
and compares to the Arm-1/2A/2B baselines (same 252-question set). Prints a clear
PASS / FAIL verdict on two axes:

  OPERATIONAL  — every query produced a result, parse-fail rate low, no empties
  SCIENTIFIC   — mean recall@10 beats Arm-2B (the hypothesis: a better tree fixes
                 PageIndex's wrong-chapter miss)

Usable as the notebook's final cell (import + call `check`) or standalone:
    python check_results.py --results <dir> --bundle <bundle_dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _recall_at(gold: list[int], ranked: list[int], k: int) -> float:
    if not gold:
        return 0.0
    topk = set(ranked[:k])
    return len(set(gold) & topk) / len(set(gold))


def _hit_at(gold: list[int], ranked: list[int], k: int) -> int:
    return int(bool(set(gold) & set(ranked[:k])))


def _mrr_at(gold: list[int], ranked: list[int], k: int) -> float:
    gs = set(gold)
    for i, b in enumerate(ranked[:k], 1):
        if b in gs:
            return 1.0 / i
    return 0.0


def load_results(results_dir: Path) -> list[dict]:
    out = []
    for p in sorted(results_dir.glob("q*.json")):
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception as exc:
            print(f"  WARN unreadable {p.name}: {exc}")
    return out


def check(results_dir: Path, bundle_dir: Path,
          op_parsefail_max: float = 0.10, sci_margin: float = 0.0) -> bool:
    queries = json.loads((bundle_dir / "queries.json").read_text(encoding="utf-8"))
    baselines = json.loads((bundle_dir / "baselines.json").read_text(encoding="utf-8"))
    base = baselines.get("baselines_recall@10", {})
    n_expected = len(queries)

    results = load_results(results_dir)
    by_qid = {str(r.get("query_id")): r for r in results}

    # ── per-query metrics ────────────────────────────────────────────────
    r10 = r100 = hit10 = mrr10 = 0.0
    n_scored = 0
    empties = 0
    parse_fail_calls = total_calls = 0
    missing = []
    for q in queries:
        qid = str(q["query_id"])
        gold = q["gold_bsard_ids"]
        r = by_qid.get(qid)
        if r is None:
            missing.append(qid)
            continue
        ranked = r.get("ranked_bsard_ids") or r.get("selected_bsard_ids") or []
        if not ranked:
            empties += 1
        r10 += _recall_at(gold, ranked, 10)
        r100 += _recall_at(gold, ranked, 100)
        hit10 += _hit_at(gold, ranked, 10)
        mrr10 += _mrr_at(gold, ranked, 10)
        n_scored += 1
        # operational telemetry (if present)
        for s in r.get("steps", []):
            total_calls += 1
            if not s.get("parse_ok", True):
                parse_fail_calls += 1

    if n_scored == 0:
        print("FAIL — no scored queries (no result files found).")
        return False

    mR10 = r10 / n_scored
    mR100 = r100 / n_scored
    mHit10 = hit10 / n_scored
    mMRR10 = mrr10 / n_scored
    parse_fail_rate = parse_fail_calls / max(total_calls, 1)

    # ── verdicts ─────────────────────────────────────────────────────────
    arm2b = base.get("arm2b", {}).get("mean_recall@10")
    arm2a = base.get("arm2a", {}).get("mean_recall@10")
    arm1 = base.get("arm1", {}).get("mean_recall@10")

    op_ok = (not missing) and (parse_fail_rate <= op_parsefail_max)
    sci_ok = (arm2b is not None) and (mR10 > arm2b + sci_margin)

    # ── report ───────────────────────────────────────────────────────────
    print("=" * 64)
    print(f"Arm-2C results — {results_dir}")
    print("=" * 64)
    print(f"queries expected={n_expected}  scored={n_scored}  "
          f"missing={len(missing)}  empty-ranking={empties}")
    if total_calls:
        print(f"LLM calls={total_calls}  parse-fail={parse_fail_calls} "
              f"({100*parse_fail_rate:.1f}%)")
    print("-" * 64)
    print(f"  mean recall@10  : {mR10:.4f}")
    print(f"  mean recall@100 : {mR100:.4f}")
    print(f"  mean hit@10     : {mHit10:.4f}")
    print(f"  mean MRR@10     : {mMRR10:.4f}")
    print("-" * 64)
    print("  baselines (same 252-q set), recall@10:")
    print(f"    Arm-1  (naive)     : {arm1}")
    print(f"    Arm-2A (metadata)  : {arm2a}")
    print(f"    Arm-2B (pageindex) : {arm2b}   <- bar to clear")
    if arm2b:
        delta = mR10 - arm2b
        print(f"    Arm-2C - Arm-2B    : {delta:+.4f}  ({100*delta/arm2b:+.0f}% rel)")
    print("=" * 64)
    print(f"  OPERATIONAL : {'PASS' if op_ok else 'FAIL'}"
          + ("" if op_ok else f"  (missing={len(missing)}, parse-fail={100*parse_fail_rate:.1f}%)"))
    print(f"  SCIENTIFIC  : {'PASS' if sci_ok else 'FAIL'}"
          f"  (recall@10 {mR10:.3f} {'>' if sci_ok else '<='} Arm-2B {arm2b})")
    verdict = op_ok and sci_ok
    print(f"\n  >>> {'SUCCESS' if verdict else 'FAILURE'} <<<")
    if missing[:10]:
        print(f"  missing qids (first 10): {missing[:10]}")
    return verdict


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True, help="dir of q<qid>.json")
    ap.add_argument("--bundle", type=Path, required=True, help="bundles/<stem>/ dir")
    args = ap.parse_args()
    ok = check(args.results, args.bundle)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
