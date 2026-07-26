"""
Post-hoc significance test: T4.0 LLM-as-a-Judge vs T3-C CE reranker (anchor).

Fast path (preferred):
  If llm_rerank_best_test.json already contains per_query_recalls (saved by
  run_llm_rerank_experiments.py ≥ post-fix runs), only T3-C needs to re-run.

Full path (fallback):
  Both retrievers are re-run. T4.0 uses the existing score cache (~2 min).
  T3-C uses the CE cross-encoder model (~30 min CPU, ~8 min T4 GPU).

Result: patches significance_vs_anchor in llm_rerank_best_test.json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from evaluation.split import load_questions, make_ground_truth
from evaluation.metrics import per_query_recall
from retrieval.dense import DenseRetriever
from retrieval.sparse import BM25Retriever
from retrieval.hybrid import HybridRetriever, CrossEncoderReranker
from retrieval.llm_reranker import LLMJudgeReranker
from retrieval.agentic.llm_client import OllamaClient

import pandas as pd

# ── Paths / constants (mirror run_llm_rerank_experiments.py) ──────────────────

RESULTS_DIR    = Path("evaluation/hybrid/results")
CORPUS_PATH    = Path("output/bsard_articles_dedup.parquet")
EMBEDDINGS_DIR = Path("output/embeddings")
DB_PATH        = Path("output/bsard_corpus.db")
FEWSHOT_PATH   = Path("evaluation/data/fewshot_examples.json")

T40_RESULT_PATH    = RESULTS_DIR / "llm_rerank_best_test.json"
ANCHOR_RESULT_PATH = RESULTS_DIR / "hybrid_rerank_minilm_best_test.json"

DENSE_MODEL  = "intfloat/multilingual-e5-large-instruct"
QUERY_PREFIX = (
    "Instruct: Given a legal question in French, "
    "retrieve the relevant Belgian statutory article\nQuery: "
)
SPARSE_CONFIG = {
    "variant": "okapi", "normalization": "lemmatize",
    "field_weighting": "concat_2x", "k1": 1.5, "b": 0.75,
}

# T4.0 config
T40_TOP_N           = 20
T40_MAX_TOKENS      = 1000
T40_PROMPT_VARIANT  = "binary"
T40_CACHE_PATH      = Path("output") / f"llm_judge_cache_{T40_PROMPT_VARIANT}_test_tok{T40_MAX_TOKENS}.json"

# T3-C config (mirror hybrid_rerank_minilm_best_test hyperparameters)
T3C_MODEL      = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
T3C_TOP_N      = 50
T3C_MAX_TOKENS = 400


def build_first_stage(corpus: pd.DataFrame) -> HybridRetriever:
    sparse = BM25Retriever(
        corpus,
        variant=SPARSE_CONFIG["variant"],
        normalization=SPARSE_CONFIG["normalization"],
        field_weighting=SPARSE_CONFIG["field_weighting"],
        k1=SPARSE_CONFIG["k1"],
        b=SPARSE_CONFIG["b"],
    )
    dense = DenseRetriever(
        corpus,
        model_name=DENSE_MODEL,
        field_weighting="concat_2x",
        query_prefix=QUERY_PREFIX,
        passage_prefix="",
        embeddings_dir=EMBEDDINGS_DIR,
    )
    return HybridRetriever(sparse_retriever=sparse, dense_retriever=dense, rrf_k=60)


def run_retriever(retriever, questions: list[dict], top_k: int = 10,
                  label: str = "", log_every: int = 20) -> dict[int, list[int]]:
    """Run retriever on all questions; return {question_id: [ranked article_ids]}."""
    results: dict[int, list[int]] = {}
    t0 = time.perf_counter()
    for i, q in enumerate(questions, 1):
        qid   = q["question_id"]
        text  = q["question_text"]
        ids, _ = retriever.retrieve(text, top_k=top_k)
        results[qid] = ids
        if i % log_every == 0 or i == len(questions):
            elapsed = (time.perf_counter() - t0) / 60
            eta     = elapsed / i * (len(questions) - i)
            print(f"  [{label}] {i}/{len(questions)}  elapsed={elapsed:.1f}min  ETA={eta:.1f}min")
    return results


def main() -> None:
    # ── Check fast path: per_query_recalls already saved in result JSON ────────
    t40_result = json.loads(T40_RESULT_PATH.read_text(encoding="utf-8"))
    if "per_query_recalls" in t40_result:
        recalls_t40_k10  = t40_result["per_query_recalls"]["k10"]
        recalls_t40_k100 = t40_result["per_query_recalls"]["k100"]
        print(f"Fast path: loaded per_query_recalls from {T40_RESULT_PATH.name}")
        need_t40_rerun = False
    else:
        print("per_query_recalls not in result JSON — will re-run T4.0 from score cache.")
        need_t40_rerun = True

    # ── Load data ─────────────────────────────────────────────────────────────
    print("Loading corpus...")
    corpus = pd.read_parquet(CORPUS_PATH)
    article_texts = dict(zip(corpus["article_id"], corpus["article_text"]))

    print("Loading test questions...")
    questions = load_questions(DB_PATH, split="test")
    gt = make_ground_truth(questions)
    print(f"  {len(questions)} test questions loaded.")

    # ── Build first-stage ────────────────────────────────────────────────────
    print("Building first-stage retriever...")
    first_stage = build_first_stage(corpus)

    # ── T4.0: LLM reranker from score cache (only if needed) ─────────────────
    if need_t40_rerun:
        if not T40_CACHE_PATH.exists():
            print(f"[ERROR] T4.0 score cache not found: {T40_CACHE_PATH}")
            print("  Run T4.0 test experiment in Colab (Cell 10) to generate the cache.")
            sys.exit(1)

    if need_t40_rerun:
        qid_map = {q["question_text"]: q["question_id"] for q in questions}
        print(f"Building T4.0 reranker (cache={T40_CACHE_PATH.name})...")
        llm_client = OllamaClient()
        t40_reranker = LLMJudgeReranker(
            first_stage_retriever=first_stage,
            article_texts=article_texts,
            llm_client=llm_client,
            top_n=T40_TOP_N,
            max_article_tokens=T40_MAX_TOKENS,
            prompt_variant=T40_PROMPT_VARIANT,
            cache_path=T40_CACHE_PATH,
            question_id_fn=lambda q: qid_map.get(q, hash(q) & 0x7FFFFFFF),
            fewshot_path=FEWSHOT_PATH,
        )
        print("Running T4.0 on test split (should be fully cached)...")
        t40_results = run_retriever(t40_reranker, questions, top_k=10, label="T4.0")
        recalls_t40_k10  = per_query_recall(t40_results, gt, k=10)
        recalls_t40_k100 = per_query_recall(t40_results, gt, k=100)

    # ── T3-C: CE reranker (always re-run) ────────────────────────────────────
    print(f"Building T3-C CE reranker ({T3C_MODEL})...")
    t3c_reranker = CrossEncoderReranker(
        first_stage_retriever=first_stage,
        model_name=T3C_MODEL,
        top_n=T3C_TOP_N,
        article_texts=article_texts,
        max_article_tokens=T3C_MAX_TOKENS,
    )
    print("Running T3-C on test split (~30 min CPU / ~8 min T4 GPU)...")
    t3c_results = run_retriever(t3c_reranker, questions, top_k=10, label="T3-C")
    recalls_t3c_k10  = per_query_recall(t3c_results, gt, k=10)
    recalls_t3c_k100 = per_query_recall(t3c_results, gt, k=100)

    # ── Paired t-test ─────────────────────────────────────────────────────────
    _, p10  = ttest_rel(recalls_t40_k10,  recalls_t3c_k10)
    _, p100 = ttest_rel(recalls_t40_k100, recalls_t3c_k100)

    print(f"\nSignificance test results:")
    print(f"  T4.0  mean R@10 = {float(np.mean(recalls_t40_k10)):.4f}")
    print(f"  T3-C  mean R@10 = {float(np.mean(recalls_t3c_k10)):.4f}")
    print(f"  p-value R@10    = {p10:.4f}  ({'significant' if p10 < 0.05 else 'not significant'})")
    print(f"  p-value R@100   = {p100:.4f}")

    # ── Patch the T4.0 result JSON ─────────────────────────────────────────
    result = json.loads(T40_RESULT_PATH.read_text(encoding="utf-8"))
    result["significance_vs_anchor"] = {
        "anchor_experiment_id": ANCHOR_RESULT_PATH.stem,
        "p_value_recall10":     round(float(p10),  4),
        "p_value_recall100":    round(float(p100), 4),
        "significant":          bool(p10 < 0.05),
    }
    T40_RESULT_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nPatched: {T40_RESULT_PATH}")
    print("Done.")


if __name__ == "__main__":
    main()
