"""
Tier 4.0 — LLM-as-a-Judge Re-Ranking Experiment Orchestrator.

First-stage retriever: bm25_tuned_k11.5_b0.25 (BM25 Okapi, k1=1.5, b=0.25,
lemmatize, text_only) — selected for all Tier 4 methods.

Execution order:
  1. llm_rerank_binary_top50_val   — primary val experiment
  2. llm_rerank_binary_top20_val   — from cache (free)
  3. llm_rerank_binary_top10_val   — from cache (free)

Then test run with best (prompt_variant, top_n) selected from val:
  4. llm_rerank_binary_top50_test  — canonical Tier 4.0 result

Usage
-----
  # Run val experiments:
  python scripts/evaluation/tier3/run_llm_rerank_experiments.py --split val

  # Run specific experiment:
  python scripts/evaluation/tier3/run_llm_rerank_experiments.py --experiment llm_rerank_binary_top50_val

  # Run test with best config (set --best-variant and --best-top-n after reviewing val):
  python scripts/evaluation/tier3/run_llm_rerank_experiments.py --split test --best-variant binary --best-top-n 50

  # Include top-100 val variant (extra ~1h):
  python scripts/evaluation/tier3/run_llm_rerank_experiments.py --split val --include-top50 --skip-top100
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from evaluation.runner import run_experiment, add_significance, save_result
from evaluation.split import load_questions
from evaluation.stratify import load_strata
from evaluation.metrics import per_query_recall
from retrieval.sparse import BM25Retriever
from retrieval.llm_reranker import LLMJudgeReranker
from retrieval.agentic.llm_client import OllamaClient

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESULTS_DIR     = Path("output/results/agentic/llm_judge/llm_rerank")
CACHE_DIR       = Path("output")
CORPUS_PATH     = Path("output/bsard_articles_dedup.parquet")
FEWSHOT_PATH    = Path("evaluation/data/fewshot_examples.json")

ANCHOR_PATH = Path("output/results/sparse_retrieval/bm25_tuned_k11.5_b0.25_test.json")

SPARSE_CONFIG = {
    "variant":         "okapi",
    "normalization":   "lemmatize",
    "field_weighting": "text_only",
    "k1": 1.5,
    "b":  0.25,
}
FIRST_STAGE_LABEL  = "bm25_tuned_k11.5_b0.25"
MAX_ARTICLE_TOKENS = 1000   # full quality (GPU); use --max-article-tokens 300 for CPU fallback
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_anchor() -> dict | None:
    if not ANCHOR_PATH.exists():
        print(f"[warn] Anchor not found at {ANCHOR_PATH} — significance tests will be skipped.")
        return None
    with open(ANCHOR_PATH) as f:
        return json.load(f)


def _print_metrics(result: dict) -> None:
    m = result["metrics"]
    print(
        f"  R@1={m['Recall@1']:.4f}  R@5={m['Recall@5']:.4f}  "
        f"R@10={m['Recall@10']:.4f}  R@100={m['Recall@100']:.4f}  "
        f"MRR@10={m['MRR@10']:.4f}  lat={result['latency_ms_mean']:.0f}ms"
    )


def _cache_path_for(prompt_variant: str, split: str, max_article_tokens: int) -> Path:
    return CACHE_DIR / f"llm_judge_cache_{prompt_variant}_{split}_tok{max_article_tokens}.json"


def _build_first_stage(corpus: pd.DataFrame) -> BM25Retriever:
    return BM25Retriever(
        corpus,
        variant=SPARSE_CONFIG["variant"],
        normalization=SPARSE_CONFIG["normalization"],
        field_weighting=SPARSE_CONFIG["field_weighting"],
        k1=SPARSE_CONFIG["k1"],
        b=SPARSE_CONFIG["b"],
    )


def _run_single_experiment(
    experiment_id: str,
    questions: list[dict],
    first_stage,
    article_texts: dict[int, str],
    llm_client: OllamaClient,
    strata: dict,
    prompt_variant: str,
    top_n: int,
    split: str,
    max_article_tokens: int = MAX_ARTICLE_TOKENS,
    anchor: dict | None = None,
) -> dict:
    """Run one experiment and save result JSON."""

    out_path = RESULTS_DIR / f"{experiment_id}.json"
    if out_path.exists():
        print(f"  [skip] {experiment_id} — result already exists at {out_path}")
        return json.loads(out_path.read_text())

    q_id_map = {q["question_text"]: q["question_id"] for q in questions}
    cache_path = _cache_path_for(prompt_variant, split, max_article_tokens)

    reranker = LLMJudgeReranker(
        first_stage_retriever=first_stage,
        article_texts=article_texts,
        llm_client=llm_client,
        top_n=top_n,
        max_article_tokens=max_article_tokens,
        prompt_variant=prompt_variant,
        cache_path=cache_path,
        question_id_fn=lambda q: q_id_map.get(q, hash(q) & 0x7FFFFFFF),
        fewshot_path=FEWSHOT_PATH,
    )

    hyperparameters = {
        "fusion_method":        "llm_judge_reranker",
        "first_stage":          FIRST_STAGE_LABEL,
        "llm_backbone":         llm_client.model,
        "llm_temperature":      0.0,
        "top_n":                top_n,
        "max_article_tokens":   max_article_tokens,
        "scoring_method":       f"llm_judge_{prompt_variant}",
        "prompt_variant":       prompt_variant,
        "fewshot_examples_file": str(FEWSHOT_PATH),
        "sparse_config":        SPARSE_CONFIG,
    }
    preprocessing = {
        "normalization":   SPARSE_CONFIG["normalization"],
        "field_weighting": SPARSE_CONFIG["field_weighting"],
    }

    n_total = len(questions)
    t_wall0 = time.perf_counter()

    print(f"\n[{experiment_id}] Starting — {n_total} questions, top_n={top_n}, variant={prompt_variant}")

    # Monkey-patch retrieve() for live progress printing
    original_retrieve = reranker.retrieve
    call_count = [0]
    running_recall = [0.0]
    ground_truth_map = {q["question_id"]: q["relevant_article_ids"] for q in questions}

    def _retrieve_with_progress(query: str, top_k: int = 10):
        ranked, lat = original_retrieve(query, top_k=top_k)
        call_count[0] += 1
        i = call_count[0]

        # Update running R@10
        qid = q_id_map.get(query)
        if qid and qid in ground_truth_map:
            gt = set(ground_truth_map[qid])
            hit = sum(1 for aid in ranked[:10] if aid in gt)
            r10 = hit / len(gt) if gt else 0.0
            running_recall[0] = (running_recall[0] * (i - 1) + r10) / i

        if i % 10 == 0 or i == n_total:
            stats = reranker.get_stats()
            parse_fail = stats["parse_failure_rate"]
            elapsed = (time.perf_counter() - t_wall0) / 60.0
            eta = (elapsed / i * (n_total - i)) if i > 0 else 0.0
            print(
                f"  [Q {i:3d}/{n_total}] R@10 running={running_recall[0]:.3f} | "
                f"llm_calls={stats['total_llm_calls']} | parse_fail={parse_fail:.3f} | "
                f"elapsed={elapsed:.1f}min ETA={eta:.1f}min"
            )

        return ranked, lat

    reranker.retrieve = _retrieve_with_progress

    result = run_experiment(
        retriever=reranker,
        questions=questions,
        experiment_id=experiment_id,
        hyperparameters=hyperparameters,
        preprocessing=preprocessing,
        strata=strata,
        top_k=100,
    )

    wall_clock_s = time.perf_counter() - t_wall0

    # Save score cache
    reranker.save_cache()

    # Enrich result with LLM-specific fields
    stats = reranker.get_stats()
    bd    = reranker.get_latency_breakdown()

    result["latency_breakdown_ms_mean"] = {
        "first_stage": bd["first_stage_ms_mean"],
        "llm_scoring": bd["llm_scoring_ms_mean"],
    }
    result["llm_rerank_stats"] = {
        "mean_llm_calls_per_query":  top_n,
        "cache_size_after_run":      stats["cache_size"],
        "parse_failure_rate":        stats["parse_failure_rate"],
        "total_experiment_wall_clock_s": round(wall_clock_s, 1),
    }
    result["score_diagnostics"] = {
        **stats["score_diagnostics"],
        "llm_score_vs_ce_score_spearman_rho": None,
    }

    # Save per-query recall lists so significance can be computed post-hoc
    # (must happen before save_result() strips _raw_results)
    if "_raw_results" in result and "_raw_gt" in result:
        result["per_query_recalls"] = {
            "k10":  [float(x) for x in per_query_recall(result["_raw_results"], result["_raw_gt"], k=10)],
            "k100": [float(x) for x in per_query_recall(result["_raw_results"], result["_raw_gt"], k=100)],
        }

    # Significance vs anchor
    if anchor is not None and "_raw_results" in anchor:
        result = add_significance(result, anchor, k_values=[10, 100], primary_k=10)
    elif anchor is not None and "per_query_recalls" in anchor and "_raw_results" in result and "_raw_gt" in result:
        # Anchor has saved per-query recalls from a previous run — use them directly
        from scipy.stats import ttest_rel as _ttest
        gt = result["_raw_gt"]
        scores_exp_10  = per_query_recall(result["_raw_results"], gt, k=10)
        scores_anc_10  = anchor["per_query_recalls"]["k10"]
        scores_exp_100 = per_query_recall(result["_raw_results"], gt, k=100)
        scores_anc_100 = anchor["per_query_recalls"]["k100"]
        _, p10  = _ttest(scores_exp_10,  scores_anc_10)
        _, p100 = _ttest(scores_exp_100, scores_anc_100)
        result["significance_vs_anchor"] = {
            "anchor_experiment_id": ANCHOR_PATH.stem,
            "p_value_recall10":     round(float(p10),  4),
            "p_value_recall100":    round(float(p100), 4),
            "significant":          bool(p10 < 0.05),
        }
    else:
        result["significance_vs_anchor"] = {
            "anchor_experiment_id":   ANCHOR_PATH.stem,
            "p_value_recall10":       None,
            "p_value_recall100":      None,
            "significant":            None,
        }

    save_result(result, results_dir=RESULTS_DIR)
    print(f"  Saved → {RESULTS_DIR / experiment_id}.json")
    _print_metrics(result)

    # Verification gates
    frac_same = result["score_diagnostics"]["fraction_queries_all_same_score"]
    parse_fail = result["llm_rerank_stats"]["parse_failure_rate"]
    if frac_same >= 0.5:
        print(
            f"  [GATE WARN] fraction_queries_all_same_score={frac_same:.2f} >= 0.5 — "
            "LLM assigns same score to all candidates; re-ranking has no effect. "
            "Report as finding."
        )
    if parse_fail >= 0.05:
        print(
            f"  [GATE WARN] parse_failure_rate={parse_fail:.3f} >= 0.05 — "
            "Investigate prompt format."
        )

    return result


# ---------------------------------------------------------------------------
# Val experiments
# ---------------------------------------------------------------------------

def run_val_experiments(
    first_stage,
    article_texts: dict[int, str],
    llm_client: OllamaClient,
    strata: dict,
    anchor: dict | None,
    skip_top100: bool = False,
    include_top50: bool = False,
    max_article_tokens: int = MAX_ARTICLE_TOKENS,
) -> dict[str, dict]:
    questions = load_questions(subset="val")
    print(f"\n=== VAL EXPERIMENTS ({len(questions)} questions) ===")

    results = {}

    # Binary top-N grid.
    # Default (GPU): top-20 primary; top-10 free from cache.
    # --include-top50: adds top-50 and top-100 (use on GPU with extra time budget).
    # Cache strategy: always run largest N first so smaller N reuse cache.
    if include_top50:
        top_n_order = [100, 50, 20, 10] if not skip_top100 else [50, 20, 10]
    else:
        top_n_order = [20, 10]   # top-20 primary; top-10 free from cache

    for top_n in top_n_order:
        eid = f"llm_rerank_binary_top{top_n}_val"
        results[eid] = _run_single_experiment(
            experiment_id=eid,
            questions=questions,
            first_stage=first_stage,
            article_texts=article_texts,
            llm_client=llm_client,
            strata=strata,
            prompt_variant="binary",
            top_n=top_n,
            split="val",
            max_article_tokens=max_article_tokens,
            anchor=anchor,
        )

    # The 0-to-10 numeric-scoring variant was run on test only — see Tier 4.0
    # plan §3.1. No val grid for it.

    # Print val summary
    print("\n=== VAL SUMMARY ===")
    print(f"{'Experiment':<40}  R@10    R@100   MRR@10")
    for eid, res in results.items():
        m = res["metrics"]
        print(f"  {eid:<40}  {m['Recall@10']:.4f}  {m['Recall@100']:.4f}  {m['MRR@10']:.4f}")

    print("\nNext: review val results, select best (prompt_variant, top_n), update Decision Log,")
    print("then run: python scripts/hybrid/run_llm_rerank_experiments.py --split test --best-variant <v> --best-top-n <n>")

    return results


# ---------------------------------------------------------------------------
# Test experiment
# ---------------------------------------------------------------------------

def run_test_experiment(
    first_stage,
    article_texts: dict[int, str],
    llm_client: OllamaClient,
    strata: dict,
    anchor: dict | None,
    best_variant: str,
    best_top_n: int,
    max_article_tokens: int = MAX_ARTICLE_TOKENS,
) -> dict:
    questions = load_questions(subset="test")
    print(f"\n=== TEST EXPERIMENT ({len(questions)} questions) ===")
    print(f"  Best config: variant={best_variant}, top_n={best_top_n}")
    print("  IMPORTANT: cache hit rate = 0% — test questions were never scored.")

    return _run_single_experiment(
        experiment_id=f"llm_rerank_{best_variant}_top{best_top_n}_test",
        questions=questions,
        first_stage=first_stage,
        article_texts=article_texts,
        llm_client=llm_client,
        strata=strata,
        prompt_variant=best_variant,
        top_n=best_top_n,
        split="test",
        max_article_tokens=max_article_tokens,
        anchor=anchor,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tier 4.0 LLM-as-a-Judge experiment runner")
    p.add_argument("--split",       choices=["val", "test", "all"], default="val")
    p.add_argument("--experiment",  type=str, default=None,
                   help="Run a single experiment by ID (e.g. llm_rerank_binary_top20_val)")
    p.add_argument("--skip-top100", action="store_true",
                   help="(legacy) skip top-100 when --include-top50 is set")
    p.add_argument("--include-top50", action="store_true",
                   help="Also run top-50 and top-100 val variants (needs extra ~2h on T4)")
    p.add_argument("--best-variant", type=str, default="binary",
                   help="Best prompt_variant from val (for --split test)")
    p.add_argument("--best-top-n",  type=int,  default=20,
                   help="Best top_n from val (default 20; for --split test)")
    p.add_argument("--max-article-tokens", type=int, default=MAX_ARTICLE_TOKENS,
                   help=f"Word-level article truncation (default {MAX_ARTICLE_TOKENS}; use 300 for CPU)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # Pre-flight
    if not FEWSHOT_PATH.exists():
        print(f"ERROR: {FEWSHOT_PATH} not found.")
        print("Run scripts/agentic/select_fewshot_examples.py first.")
        sys.exit(1)

    print("[T4.0] Loading corpus …")
    corpus       = pd.read_parquet(CORPUS_PATH)
    article_texts: dict[int, str] = dict(
        zip(corpus["article_id"], corpus["article_text"].fillna(""))
    )

    print("[T4.0] Loading strata and anchor …")
    strata = load_strata()
    anchor = _load_anchor()

    print("[T4.0] Building first-stage retriever (bm25_tuned_k11.5_b0.25) …")
    first_stage = _build_first_stage(corpus)

    print("[T4.0] Initialising OllamaClient …")
    llm_client = OllamaClient(model="llama3.1:8b")
    try:
        lat = llm_client.preflight()
        print(f"  Ollama OK — preflight latency {lat:.0f}ms")
    except (ConnectionError, ValueError) as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    # Run
    max_article_tokens = args.max_article_tokens

    if args.experiment:
        # Single experiment by ID
        # Derive parameters from experiment ID
        eid = args.experiment
        if "binary" in eid:
            variant = "binary"
        elif "0to10" in eid:
            variant = "0to10"
        else:
            variant = args.best_variant

        top_n_match = [int(x) for x in eid.split("top") if x[:3].isdigit()]
        top_n = top_n_match[0] if top_n_match else args.best_top_n
        split = "test" if "_test" in eid else "val"
        questions = load_questions(subset=split)

        _run_single_experiment(
            experiment_id=eid,
            questions=questions,
            first_stage=first_stage,
            article_texts=article_texts,
            llm_client=llm_client,
            strata=strata,
            prompt_variant=variant,
            top_n=top_n,
            split=split,
            max_article_tokens=max_article_tokens,
            anchor=anchor,
        )

    elif args.split in ("val", "all"):
        run_val_experiments(
            first_stage=first_stage,
            article_texts=article_texts,
            llm_client=llm_client,
            strata=strata,
            anchor=anchor,
            skip_top100=args.skip_top100,
            include_top50=args.include_top50,
            max_article_tokens=max_article_tokens,
        )
        if args.split == "all":
            run_test_experiment(
                first_stage=first_stage,
                article_texts=article_texts,
                llm_client=llm_client,
                strata=strata,
                anchor=anchor,
                best_variant=args.best_variant,
                best_top_n=args.best_top_n,
                max_article_tokens=max_article_tokens,
            )

    elif args.split == "test":
        run_test_experiment(
            first_stage=first_stage,
            article_texts=article_texts,
            llm_client=llm_client,
            strata=strata,
            anchor=anchor,
            best_variant=args.best_variant,
            best_top_n=args.best_top_n,
            max_article_tokens=max_article_tokens,
        )


if __name__ == "__main__":
    main()
