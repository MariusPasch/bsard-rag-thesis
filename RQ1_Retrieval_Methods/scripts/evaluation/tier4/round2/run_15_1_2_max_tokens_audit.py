"""
§15.1.2 — Re-run the same 50 null-action samples with max_tokens=256 and
compare the null-rate to the Round-1 max_tokens=64 budget.

Purpose
-------
Tests the sub-hypothesis from §13.1: that the strict-regex parser is failing
because the Action line was *truncated* by max_tokens=64, not because the
model emitted the wrong shape. If raising the budget to 256 drops the null
rate substantially, the budget itself is a primary cause.

This script is **not** intended to run locally — it must run inside the Azure
notebook (Llama 3.1 8B on a T4) so the latency / null-rate measurements are
comparable to the Round-1 production run.

Output
------
output/results/agentic/ReAct/round2/diagnostics_15_1_2_max_tokens_audit.json
{
  "n_sampled": 50,
  "n_null_at_64":   int,
  "n_null_at_256":  int,
  "delta_null_rate": float,    # (n_null_at_64 - n_null_at_256) / n_sampled
  "mean_lat_64":   float,
  "mean_lat_256":  float,
  "examples": [
    {"qid": ..., "step": ..., "raw_64": "...", "raw_256": "...",
     "matched_64": bool, "matched_256": bool}, ...
  ]
}

Usage (Azure)
-------------
  python scripts/evaluation/tier4/round2/run_15_1_2_max_tokens_audit.py \\
      --variant hybrid_rrf_k60 --n-sampled 50 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["bm25", "hybrid_rrf_k60"],
                        default="hybrid_rrf_k60")
    parser.add_argument("--n-sampled", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from evaluation.split import load_questions
    from retrieval.agentic.llm_client import OllamaClient
    from retrieval.agentic.prompts import REACT_SYSTEM_PROMPT, REACT_USER_TEMPLATE
    from retrieval.agentic.react import _parse_action

    traces_path = (
        ROOT / "output" / "results" / "agentic" / "ReAct"
        / f"react_{args.variant}_test_traces.json"
    )
    traces = json.loads(traces_path.read_text(encoding="utf-8"))

    null_targets: list[tuple[int, int]] = []
    for entry in traces:
        for step in entry["trace"]:
            if step["tool_name"] is None and step.get("preflight_failed"):
                null_targets.append((entry["question_id"], step["step"]))

    rng = random.Random(args.seed)
    rng.shuffle(null_targets)
    sample = null_targets[: args.n_sampled]

    questions = {q["question_id"]: q for q in load_questions(subset="test")}
    llm = OllamaClient()
    llm.preflight()

    examples = []
    sum_lat_64  = 0.0
    sum_lat_256 = 0.0
    n_null_64  = 0
    n_null_256 = 0

    for i, (qid, step_idx) in enumerate(sample, 1):
        q = questions[qid]
        prompt = (
            REACT_SYSTEM_PROMPT + "\n\n"
            + REACT_USER_TEMPLATE.format(question=q["question_text"], scratchpad="")
        )

        raw_64,  lat_64  = llm.generate(prompt, temperature=0.0, max_tokens=64)
        raw_256, lat_256 = llm.generate(prompt, temperature=0.0, max_tokens=256)
        sum_lat_64  += lat_64
        sum_lat_256 += lat_256

        tool_64,  _ = _parse_action(raw_64)
        tool_256, _ = _parse_action(raw_256)
        matched_64  = tool_64  is not None
        matched_256 = tool_256 is not None
        n_null_64  += 0 if matched_64  else 1
        n_null_256 += 0 if matched_256 else 1

        examples.append({
            "qid":        qid,
            "step":       step_idx,
            "matched_64":  matched_64,
            "matched_256": matched_256,
            "raw_64":   raw_64,
            "raw_256":  raw_256,
            "lat_64_ms":  round(lat_64,  1),
            "lat_256_ms": round(lat_256, 1),
        })
        print(f"  [{i}/{len(sample)}] matched_64={matched_64}  matched_256={matched_256}",
              flush=True)

    n = len(sample)
    out_payload = {
        "diagnostic":      "15.1.2 max_tokens=64 vs 256 null-action audit",
        "variant":         args.variant,
        "n_sampled":       n,
        "n_null_at_64":    n_null_64,
        "n_null_at_256":   n_null_256,
        "delta_null_rate": (n_null_64 - n_null_256) / n if n else 0.0,
        "mean_lat_64_ms":  sum_lat_64  / n if n else 0.0,
        "mean_lat_256_ms": sum_lat_256 / n if n else 0.0,
        "examples":        examples,
    }
    out = ROOT / "output" / "results" / "agentic" / "ReAct" / "round2" \
        / "diagnostics_15_1_2_max_tokens_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(out_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved → {out}")
    print(f"  null@64 = {n_null_64}/{n}   null@256 = {n_null_256}/{n}")


if __name__ == "__main__":
    main()
