"""
Gold-ceiling Tier-3 run.

For each of the 48 T3-subset questions, treat the gold ``relevant_article_ids``
(ordered as listed, truncated to k=10) as the "retrieval" and run UMBRELA,
eRAG, and RAGAS-WA over those (q, d) pairs. The resulting scores establish
the ceiling each autonomous evaluator can produce against perfect retrieval —
separating "retrieval failure" from "evaluator failure" in RQ3 analysis.

Why this script reimplements the evaluators
───────────────────────────────────────────
The editable install of bsard_evaluation (sibling repo
RQ3_Autonomous_Evaluation) is currently in a state that does not expose
the {umbrela, erag, ragas_wa} components that produced the 19-system data —
its TierConfig validation rejects those names and the harness signature
does not accept ``contexts_with_ranks``. To produce a 20th comparable row
without modifying the sibling repo we reimplement the three evaluators
inline. Prompts are explicit (see UMBRELA_PROMPT / ERAG_PROMPT /
RAGAS_WA_VERDICT_PROMPT / HYDE_PROMPT below).

Output
──────
output/results/RQ3/gold_ceiling_per_query.json   — per-query sidecar
output/results/RQ3/gold_ceiling.json             — consolidated record
                                                   (family="oracle")

Usage
─────
  $env:OPENAI_API_KEY="sk-..."
  .venv/Scripts/python scripts/evaluation/RQ3/run_gold_ceiling.py
  .venv/Scripts/python scripts/evaluation/RQ3/run_gold_ceiling.py --dry-run
  .venv/Scripts/python scripts/evaluation/RQ3/run_gold_ceiling.py --concurrency 8

Cost on gpt-4o-mini: ~$0.30–0.50; wall time ~5–10 min.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ROOT        = Path(__file__).parent.parent.parent.parent
PARQUET_PATH = _ROOT / "output" / "bsard_articles_dedup.parquet"
SUBSET_FILE  = _ROOT / "evaluation" / "data" / "tier3_subset.json"
OUT_DIR      = _ROOT / "output" / "results" / "RQ3"
SIDECAR_PATH = OUT_DIR / "gold_ceiling_per_query.json"
RECORD_PATH  = OUT_DIR / "gold_ceiling.json"

K_TOP        = 10
DEFAULT_MODEL = "gpt-4o-mini"

# ---------------------------------------------------------------------------
# Prompts (explicit so the methodology is auditable)
# ---------------------------------------------------------------------------

UMBRELA_PROMPT = """\
Given a query and a passage, you must provide a score on an integer scale of 0 to 3 with the following meanings:

0 = represents that the passage has nothing to do with the query
1 = represents that the passage seems related to the query but does not answer it
2 = represents that the passage has some answer for the query, but the answer may be a bit unclear, or hidden amongst extraneous information
3 = represents that the passage is dedicated to the query and contains the exact answer.

Important Instruction: Assign category 1 if the passage is somewhat related to the topic but not completely, category 2 if passage presents something very important related to the entire topic but also has some extra information and category 3 if the passage only and entirely refers to the topic. If none of the above satisfies give it category 0.

Query: {query}
Passage: {passage}

Split this problem into steps:
Consider the underlying intent of the search.
Measure how well the content matches a likely intent of the query (M).
Measure how trustworthy the passage is (T).
Consider the aspects above and the relative importance of each, and decide on a final score (O).
Final score must be an integer value only.
Do not provide any code in result. Provide each score in the format of: ##final score: score without providing any reasoning.
"""

ERAG_PROMPT = """\
You are evaluating retrieved evidence for a Belgian-law question answering system.

Decide whether the passage by itself contains sufficient information to answer the query.
Reply 1 if yes (the passage alone is enough), or 0 if no.

Query: {query}
Passage: {passage}

Reply with only a single character: 0 or 1.
"""

RAGAS_WA_VERDICT_PROMPT = """\
You are verifying whether a retrieved passage was useful in arriving at a given answer.

Question: {question}
Answer:   {answer}
Passage:  {passage}

Was this passage useful for producing the answer? Reply with only a single character: 1 (useful) or 0 (not useful).
"""

HYDE_PROMPT = """\
You are a Belgian legal assistant. Write a concise hypothetical answer (3–5 sentences) to the question below, in French. Do not include any disclaimers; respond as if you knew the relevant articles. The answer is used only as a similarity reference for retrieval evaluation.

Question: {question}

Answer:"""

# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_subset() -> list[dict]:
    return json.loads(SUBSET_FILE.read_text(encoding="utf-8"))["questions"]


def load_articles() -> dict[int, str]:
    df = pd.read_parquet(PARQUET_PATH, columns=["article_id", "article_text"])
    df = df[df["article_text"].notna()]
    return dict(zip(df["article_id"].astype(int), df["article_text"].astype(str)))


# ---------------------------------------------------------------------------
# OpenAI client wrapper
# ---------------------------------------------------------------------------

class LLM:
    def __init__(self, model: str = DEFAULT_MODEL):
        from openai import OpenAI
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY not set in environment")
        self.client = OpenAI()
        self.model  = model

    def chat(self, prompt: str, *, max_tokens: int = 32, temperature: float = 0.0) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Evaluator primitives
# ---------------------------------------------------------------------------

_UMB_RE   = re.compile(r"##\s*final\s*score\s*:\s*([0-3])", re.IGNORECASE)
_DIGIT_RE = re.compile(r"[0-3]")
_BIN_RE   = re.compile(r"[01]")


def umbrela_grade(llm: LLM, query: str, passage: str) -> int:
    """0–3 grade following the published UMBRELA prompt."""
    out = llm.chat(UMBRELA_PROMPT.format(query=query, passage=passage[:3500]),
                   max_tokens=32)
    m = _UMB_RE.search(out)
    if m:
        return int(m.group(1))
    # Fallback: first 0–3 digit anywhere in the output
    m = _DIGIT_RE.search(out)
    return int(m.group(0)) if m else 0


def erag_score(llm: LLM, query: str, passage: str) -> int:
    """0/1: is this passage alone sufficient to answer the query?"""
    out = llm.chat(ERAG_PROMPT.format(query=query, passage=passage[:3500]),
                   max_tokens=4).strip()
    m = _BIN_RE.search(out)
    return int(m.group(0)) if m else 0


def hyde_answer(llm: LLM, question: str) -> str:
    return llm.chat(HYDE_PROMPT.format(question=question),
                    max_tokens=300, temperature=0.0).strip()


def ragas_wa_verdict(llm: LLM, question: str, answer: str, passage: str) -> int:
    """0/1 verdict: was this passage useful in arriving at the answer?"""
    out = llm.chat(
        RAGAS_WA_VERDICT_PROMPT.format(
            question=question, answer=answer, passage=passage[:3500]),
        max_tokens=4,
    ).strip()
    m = _BIN_RE.search(out)
    return int(m.group(0)) if m else 0


def context_precision_with_reference(verdicts: list[int]) -> float:
    """
    Replication of ragas LLMContextPrecisionWithReference aggregator.

    context_precision@K = sum_{k=1..K} (precision@k * v_k) / sum(v)
    where precision@k = sum_{j=1..k} v_j / k, v_k in {0, 1}.
    Returns 0.0 if no verdicts are positive.
    """
    if not verdicts or sum(verdicts) == 0:
        return 0.0
    cum = 0
    weighted_sum = 0.0
    for k, v in enumerate(verdicts, start=1):
        cum += v
        if v == 1:
            weighted_sum += cum / k
    return weighted_sum / sum(verdicts)


# ---------------------------------------------------------------------------
# Per-question evaluation
# ---------------------------------------------------------------------------

def evaluate_question(
    llm: LLM,
    qid: str,
    question_text: str,
    gold_doc_ids: list[str],
    corpus: dict[int, str],
) -> dict:
    """
    Returns a dict with keys: ranks, umbrela{doc->grade}, erag{doc->0/1},
    ragas_wa (float scalar), hyde (string).
    """
    docs = [(d, corpus.get(int(d), "")) for d in gold_doc_ids]
    # HyDE first (used as reference for RAGAS-WA)
    hyde = hyde_answer(llm, question_text)

    umb: dict[str, int] = {}
    erg: dict[str, int] = {}
    rag_verdicts: list[int] = []
    for d, text in docs:
        if not text:
            umb[d] = 0
            erg[d] = 0
            rag_verdicts.append(0)
            continue
        umb[d] = umbrela_grade(llm, question_text, text)
        erg[d] = erag_score(llm, question_text, text)
        rag_verdicts.append(ragas_wa_verdict(llm, question_text, hyde, text))

    return {
        "ranks":    [d for d, _ in docs],
        "umbrela":  umb,
        "erag":     erg,
        "ragas_wa": context_precision_with_reference(rag_verdicts),
        "hyde":     hyde,
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate(per_q: dict[str, dict]) -> dict[str, float]:
    """Compute T3/* metrics over the per-question results (matching RQ3 schema)."""
    umb_grades: list[int] = []
    erg_vals:   list[int] = []
    rag_vals:   list[float] = []
    for qid, q in per_q.items():
        umb_grades.extend(q["umbrela"].values())
        erg_vals.extend(q["erag"].values())
        rag_vals.append(q["ragas_wa"])

    n_umb = len(umb_grades)
    umb_mean = sum(umb_grades) / (3.0 * n_umb) if n_umb else 0.0  # mean grade / 3 → [0,1]
    umb_relfrac = sum(1 for g in umb_grades if g >= 2) / n_umb if n_umb else 0.0
    erg_mean = sum(erg_vals) / len(erg_vals) if erg_vals else 0.0
    rag_mean = sum(rag_vals) / len(rag_vals) if rag_vals else 0.0

    # Old AQS (7:6:3) — preserved for backward-compat consolidation/notebook code
    old_aqs = 0.4375 * umb_mean + 0.3750 * erg_mean + 0.1875 * rag_mean

    return {
        "T3/umbrela/mean":             umb_mean,
        "T3/umbrela/relevant_fraction": umb_relfrac,
        "T3/erag/mean":                erg_mean,
        "T3/ragas_wa/mean":            rag_mean,
        "T3/AQS":                      old_aqs,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip LLM calls, write a synthetic sidecar.")
    parser.add_argument("--concurrency", type=int, default=4,
                        help="Number of concurrent question evaluations.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=None,
                        help="Evaluate only the first N questions (for debugging).")
    args = parser.parse_args()

    print("=" * 70)
    print("RQ3 — GOLD-CEILING TIER-3 RUN  (variable-k, k <= 10, gold-only)")
    print("=" * 70)

    questions = load_subset()
    if args.limit:
        questions = questions[: args.limit]
    print(f"Questions: {len(questions)}")
    print(f"Loading corpus …")
    corpus = load_articles()
    print(f"Articles:  {len(corpus):,}")

    if args.dry_run:
        per_q = {}
        for q in questions:
            qid = str(q["question_id"])
            ids = [str(d) for d in q["relevant_article_ids"][:K_TOP]]
            per_q[qid] = {
                "ranks":    ids,
                "umbrela":  {d: 0 for d in ids},
                "erag":     {d: 0 for d in ids},
                "ragas_wa": 0.0,
                "hyde":     "[dry-run]",
            }
        agg = aggregate(per_q)
        print(f"[dry-run] aggregated: {agg}")
        return

    if not os.environ.get("OPENAI_API_KEY"):
        print("[ERROR] OPENAI_API_KEY is not set.")
        sys.exit(1)

    llm = LLM(model=args.model)

    print(f"Evaluating with model={args.model}, concurrency={args.concurrency}")
    per_q: dict[str, dict] = {}
    started = time.perf_counter()

    def _task(q):
        qid = str(q["question_id"])
        ids = [str(d) for d in q["relevant_article_ids"][:K_TOP]]
        return qid, evaluate_question(llm, qid, q["question_text"], ids, corpus)

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = [ex.submit(_task, q) for q in questions]
        for i, fut in enumerate(as_completed(futures), 1):
            qid, payload = fut.result()
            per_q[qid] = payload
            n_d = len(payload["ranks"])
            umb_mean = sum(payload["umbrela"].values()) / max(n_d, 1)
            print(f"  [{i:>2}/{len(questions)}] qid={qid:>4}  k={n_d:>2}  "
                  f"umb_avg={umb_mean:.2f}  erag_sum={sum(payload['erag'].values())}  "
                  f"rag_wa={payload['ragas_wa']:.3f}")

    elapsed = time.perf_counter() - started
    print(f"\nLLM evaluation done in {elapsed:.1f}s")

    # ── Write per-query sidecar (matching the per-system schema) ──────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sidecar = {
        "ranks":    {qid: r["ranks"] for qid, r in per_q.items()},
        "umbrela":  {qid: r["umbrela"] for qid, r in per_q.items()},
        "erag":     {qid: r["erag"] for qid, r in per_q.items()},
        "ragas_wa": {qid: r["ragas_wa"] for qid, r in per_q.items()},
        "hyde":     {qid: r["hyde"] for qid, r in per_q.items()},
    }
    SIDECAR_PATH.write_text(
        json.dumps(sidecar, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[wrote] {SIDECAR_PATH.relative_to(_ROOT).as_posix()}")

    # ── Build the consolidated record (matches RQ3 schema; family="oracle") ──
    metrics = aggregate(per_q)
    n = len(per_q)
    subset_metrics = {
        "subset_ids": [int(qid) for qid in per_q.keys()],
        "n":          n,
        "metrics":    metrics,
    }
    record = {
        "experiment_id":     "gold_ceiling",
        "family":            "oracle",
        "model_or_method":   "gold_ceiling",
        "hyperparameters":   {
            "k_top":          K_TOP,
            "judge_model":    args.model,
            "evaluators":     ["umbrela", "erag", "ragas_wa"],
            "ragas_wa_kind":  "LLMContextPrecisionWithReference (HyDE)",
        },
        "preprocessing":      None,
        "source_result":      None,
        "per_query_sidecar":  SIDECAR_PATH.relative_to(_ROOT).as_posix(),
        "subset_metrics":     subset_metrics,
    }
    RECORD_PATH.write_text(
        json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[wrote] {RECORD_PATH.relative_to(_ROOT).as_posix()}")
    print(f"\nMetrics:")
    for k, v in metrics.items():
        print(f"  {k:<32s} {v:.4f}")


if __name__ == "__main__":
    main()
