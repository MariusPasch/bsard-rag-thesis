"""
§15.1.1 — Sample 50 queries with null actions, replay them with verbose=True
to capture the raw LLM output, and bucket the failures.

Buckets
-------
(a) wrong_bracket_shape   — `search("…")`, `search:"…"`, French «…», bare prose, etc.
(b) prose_only            — no `Action :` line at all
(c) max_tokens_truncation — Action line cut off mid-string; no closing `]`
(d) french_quotes         — `«…»` pair instead of `"…"` or `[…]`
(e) refusal_or_empty      — empty / "Je ne peux pas …" / safety refusal

This script is **not** intended to run locally — it must run inside the Azure
notebook with the Round-1 ReActRetriever (regex parser + max_tokens=64) so
that we reproduce the exact failure mode.

Output
------
output/results/agentic/ReAct/round2/diagnostics_15_1_1_null_action_buckets.json
{
  "n_sampled": 50,
  "n_total_steps_replayed": ...,
  "buckets": {
    "wrong_bracket_shape": int,
    "prose_only":          int,
    "max_tokens_truncation": int,
    "french_quotes":       int,
    "refusal_or_empty":    int,
  },
  "examples": [{"qid": ..., "step": ..., "raw_output": ..., "bucket": ...}, ...]
}

Usage (Azure)
-------------
  python scripts/evaluation/tier4/round2/run_15_1_1_null_action_buckets.py \\
      --variant hybrid_rrf_k60 --n-sampled 50 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Bucketing logic
# ---------------------------------------------------------------------------

# Round-1 strict regex (from retrieval/agentic/react.py:73 in v1).
_STRICT_ACTION_RE = re.compile(
    r"Action\s*:\s*(search|lookup|finish)\[([^\]]*)\]",
    re.IGNORECASE,
)

# Permissive matchers used purely for failure bucketing (NOT used by the agent).
_HAS_ACTION_LINE_RE        = re.compile(r"^\s*Action\s*:", re.IGNORECASE | re.MULTILINE)
_FRENCH_QUOTE_RE           = re.compile(r"«[^»]*»")
_PARENS_OR_COLON_FORM_RE   = re.compile(
    r"Action\s*:\s*(search|lookup|finish)\s*[:\(]",
    re.IGNORECASE,
)
_REFUSAL_PHRASES = (
    "je ne peux", "i cannot", "i'm sorry", "désolé", "ne peux pas",
)


def classify_null_output(raw: str, max_tokens_used: int) -> str:
    """
    Decide which bucket a failed step's raw LLM output falls into.

    Order matters: refusal/empty first, then truncation (the saved trace records
    the actual token count consumed), then shape problems.
    """
    text = (raw or "").strip()

    if not text or any(p in text.lower() for p in _REFUSAL_PHRASES):
        return "refusal_or_empty"

    if not _HAS_ACTION_LINE_RE.search(text):
        return "prose_only"

    # Was the closing `]` missing AND the model used the full token budget?
    has_action      = _HAS_ACTION_LINE_RE.search(text)
    last_action_pos = text.rfind("Action")
    tail            = text[last_action_pos:] if last_action_pos != -1 else ""
    if has_action and "]" not in tail and max_tokens_used >= 60:
        return "max_tokens_truncation"

    if _FRENCH_QUOTE_RE.search(text):
        return "french_quotes"

    if _PARENS_OR_COLON_FORM_RE.search(text):
        return "wrong_bracket_shape"

    # Default catch-all — text exists, has 'Action', but not the canonical form.
    return "wrong_bracket_shape"


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["bm25", "hybrid_rrf_k60"],
                        default="hybrid_rrf_k60")
    parser.add_argument("--n-sampled", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=64,
                        help="Match Round-1 budget for the bucket comparison")
    args = parser.parse_args()

    import pandas as pd
    from evaluation.split import load_questions
    from retrieval.agentic.llm_client import OllamaClient
    from retrieval.agentic.prompts import (
        REACT_SYSTEM_PROMPT,
        REACT_USER_TEMPLATE,
    )
    from retrieval.agentic.react import _parse_action

    df_articles = pd.read_parquet(ROOT / "output" / "bsard_articles_dedup.parquet")

    traces_path = (
        ROOT / "output" / "results" / "agentic" / "ReAct"
        / f"react_{args.variant}_test_traces.json"
    )
    traces = json.loads(traces_path.read_text(encoding="utf-8"))

    null_targets: list[tuple[int, int]] = []  # (question_id, step_idx)
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

    # NOTE: This stub re-prompts each null-action step in isolation (with an
    # empty scratchpad, so the bucket reflects the "first turn" failure). The
    # full Azure version should replay the full prefix scratchpad up to that
    # step. Kept simple here for review-ability — adjust on Azure if needed.
    buckets = {
        "wrong_bracket_shape":   0,
        "prose_only":            0,
        "max_tokens_truncation": 0,
        "french_quotes":         0,
        "refusal_or_empty":      0,
    }
    examples: list[dict] = []

    for i, (qid, step_idx) in enumerate(sample, 1):
        q = questions[qid]
        prompt = (
            REACT_SYSTEM_PROMPT + "\n\n"
            + REACT_USER_TEMPLATE.format(question=q["question_text"], scratchpad="")
        )
        raw, lat = llm.generate(prompt, temperature=0.0, max_tokens=args.max_tokens)
        bucket = classify_null_output(raw, args.max_tokens)
        # Sanity check: if the strict regex DOES match, we miscounted as null.
        if _STRICT_ACTION_RE.search(raw):
            bucket = "false_positive_strict_regex_now_matches"

        buckets[bucket] = buckets.get(bucket, 0) + 1
        examples.append({
            "qid": qid,
            "step": step_idx,
            "bucket": bucket,
            "latency_ms": round(lat, 1),
            "raw_output": raw,
        })
        print(f"  [{i}/{len(sample)}] qid={qid} step={step_idx} bucket={bucket}",
              flush=True)

    out = ROOT / "output" / "results" / "agentic" / "ReAct" / "round2" \
        / "diagnostics_15_1_1_null_action_buckets.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "diagnostic":            "15.1.1 null-action bucket sample",
        "variant":               args.variant,
        "n_sampled":             len(sample),
        "n_total_null_steps":    len(null_targets),
        "max_tokens":            args.max_tokens,
        "buckets":               buckets,
        "examples":              examples,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
