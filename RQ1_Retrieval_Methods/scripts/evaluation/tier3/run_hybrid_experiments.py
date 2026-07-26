"""
Tier 3 — Hybrid Retrieval Experiment Orchestrator.

Execution order
---------------
1.  Load corpus, questions, strata, T3 anchor
2.  Instantiate sparse (BM25 lemmatize+concat_2x) and dense (mE5-large-instruct+concat_2x)
3.  T3-A : RRF k in {30, 60, 120} on val  ->  best k  ->  test
4.  T3-A2: SKIP (BGE-M3 not selected; documented below)
5.  T3-B : linear alpha grid 0.1-0.9 on val  ->  best alpha  ->  test
6.  T3-C : select best first stage (argmax R@100 val),
           CE top-N in {50, 100} on val  ->  best N  ->  test
7.  T3-D : HyDE + RAG-Fusion  (only if --include-augmented)

Usage
-----
  python scripts/hybrid/run_hybrid_experiments.py
  python scripts/hybrid/run_hybrid_experiments.py --include-augmented
  python scripts/hybrid/run_hybrid_experiments.py --stage t3a   # run only T3-A
  python scripts/hybrid/run_hybrid_experiments.py --stage t3b
  python scripts/hybrid/run_hybrid_experiments.py --stage t3c
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from evaluation.runner import run_experiment, add_significance, save_result
from evaluation.split import load_questions
from evaluation.stratify import load_strata
from retrieval.dense import DenseRetriever
from retrieval.sparse import BM25Retriever
from retrieval.hybrid import HybridRetriever, CrossEncoderReranker

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESULTS_DIR = Path("evaluation/hybrid/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CORPUS_PATH = Path("output/bsard_articles_dedup.parquet")
EMBEDDINGS_DIR = Path("output/embeddings")

DENSE_MODEL = "intfloat/multilingual-e5-large-instruct"
QUERY_PREFIX = (
    "Instruct: Given a legal question in French, "
    "retrieve the relevant Belgian statutory article\nQuery: "
)

SPARSE_CONFIG = {
    "variant": "okapi",
    "normalization": "lemmatize",
    "field_weighting": "concat_2x",
    "k1": 1.5,
    "b": 0.75,
}
DENSE_CONFIG = {
    "model_name": DENSE_MODEL,
    "field_weighting": "concat_2x",
    "query_prefix": QUERY_PREFIX,
    "passage_prefix": "",
}

T3_A2_SKIP_NOTE = {
    "t3_a2_skipped": True,
    "t3_a2_skip_reason": (
        "BGE-M3 ranked 3rd on val (R@100=0.5421); "
        "mE5-large-instruct+concat2x selected (R@100=0.5520)"
    ),
}

CE_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

ANCHOR_PATH = Path(
    "evaluation/dense_retrieval/results/"
    "dense_me5_large_instruct_concat2x_zeroshot_test.json"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_anchor() -> dict:
    if not ANCHOR_PATH.exists():
        raise FileNotFoundError(
            f"T3 anchor not found: {ANCHOR_PATH}\n"
            "Run Step 0 first: dense_me5_large_instruct_concat2x_zeroshot_test"
        )
    with open(ANCHOR_PATH) as f:
        return json.load(f)


def _print_metrics(result: dict) -> None:
    m = result["metrics"]
    print(
        f"  R@10={m['Recall@10']:.4f}  R@100={m['Recall@100']:.4f}"
        f"  MRR@10={m['MRR@10']:.4f}  lat={result['latency_ms_mean']:.1f}ms"
    )


def _add_sig_and_save(result: dict, anchor: dict, results_dir: Path) -> dict:
    """Add significance vs anchor (uses raw results if anchor has them)."""
    sig_block = result.get("significance_vs_anchor", {})
    # anchor loaded from JSON has no _raw_results; skip t-test in that case
    if "_raw_results" in anchor and "_raw_results" in result:
        result = add_significance(result, anchor, k_values=[10, 100], primary_k=10)
    else:
        result["significance_vs_anchor"] = {
            "p_value_recall10": None,
            "p_value_recall100": None,
            "significant": None,
            "note": "anchor loaded from JSON; raw results unavailable for t-test",
        }
    save_result(result, results_dir=results_dir)
    return result


def _run_one(
    retriever,
    questions: list[dict],
    experiment_id: str,
    hyperparameters: dict,
    strata: dict,
    anchor: dict,
    results_dir: Path,
    top_k: int = 500,
    extra_fields: dict | None = None,
) -> dict:
    """Run a single experiment, print metrics, add significance, save, return result."""
    print(f"\n[{experiment_id}]")
    result = run_experiment(
        retriever=retriever,
        questions=questions,
        experiment_id=experiment_id,
        hyperparameters=hyperparameters,
        preprocessing={
            "normalization": "lemmatize_sparse__none_dense",
            "field_weighting": "concat_2x_both",
        },
        strata=strata,
        top_k=top_k,
    )
    if extra_fields:
        result.update(extra_fields)
    result = _add_sig_and_save(result, anchor, results_dir)
    _print_metrics(result)
    return result


# ---------------------------------------------------------------------------
# T3-A — Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def run_t3a(sparse, dense, val_qs, test_qs, strata, anchor, results_dir):
    print("\n" + "=" * 60)
    print("T3-A  Reciprocal Rank Fusion (k in {30, 60, 120})")
    print("=" * 60)

    val_results = {}
    for rrf_k in [30, 60, 120]:
        retriever = HybridRetriever(
            sparse_retriever=sparse,
            dense_retriever=dense,
            fusion_method="rrf",
            rrf_k=rrf_k,
            first_stage_k=100,
        )
        exp_id = f"hybrid_rrf_k{rrf_k}_val"
        hp = {
            "fusion_method": "rrf",
            "rrf_k": rrf_k,
            "first_stage_k": 100,
            "sparse_config": SPARSE_CONFIG,
            "dense_config": DENSE_CONFIG,
            **T3_A2_SKIP_NOTE,
        }
        r = _run_one(retriever, val_qs, exp_id, hp, strata, anchor, results_dir)
        val_results[rrf_k] = r

    # Select best k by val Recall@10
    best_k = max(val_results, key=lambda k: val_results[k]["metrics"]["Recall@10"])
    best_k_r100 = val_results[best_k]["metrics"]["Recall@100"]
    print(f"\n  Best RRF k = {best_k}  (val R@10={val_results[best_k]['metrics']['Recall@10']:.4f}, "
          f"R@100={best_k_r100:.4f})")

    # Test run
    retriever = HybridRetriever(
        sparse_retriever=sparse,
        dense_retriever=dense,
        fusion_method="rrf",
        rrf_k=best_k,
        first_stage_k=100,
    )
    hp_test = {
        "fusion_method": "rrf",
        "rrf_k": best_k,
        "first_stage_k": 100,
        "rrf_k_selected_by": f"val_recall@10 (best k={best_k})",
        "sparse_config": SPARSE_CONFIG,
        "dense_config": DENSE_CONFIG,
        **T3_A2_SKIP_NOTE,
    }
    test_result = _run_one(
        retriever, test_qs, "hybrid_rrf_best_test", hp_test, strata, anchor, results_dir
    )
    return val_results, test_result, best_k


# ---------------------------------------------------------------------------
# T3-B — Linear score interpolation
# ---------------------------------------------------------------------------

def run_t3b(sparse, dense, val_qs, test_qs, strata, anchor, results_dir):
    print("\n" + "=" * 60)
    print("T3-B  Linear interpolation (alpha grid 0.1 - 0.9)")
    print("=" * 60)

    alphas = [round(a * 0.1, 1) for a in range(1, 10)]
    val_results = {}
    for alpha in alphas:
        retriever = HybridRetriever(
            sparse_retriever=sparse,
            dense_retriever=dense,
            fusion_method="linear",
            alpha=alpha,
            first_stage_k=100,
        )
        exp_id = f"hybrid_linear_alpha_{alpha}_val"
        hp = {
            "fusion_method": "linear",
            "alpha": alpha,
            "first_stage_k": 100,
            "sparse_config": SPARSE_CONFIG,
            "dense_config": DENSE_CONFIG,
        }
        r = _run_one(retriever, val_qs, exp_id, hp, strata, anchor, results_dir)
        val_results[alpha] = r

    best_alpha = max(val_results, key=lambda a: val_results[a]["metrics"]["Recall@10"])
    print(f"\n  Best alpha = {best_alpha}  "
          f"(val R@10={val_results[best_alpha]['metrics']['Recall@10']:.4f}, "
          f"R@100={val_results[best_alpha]['metrics']['Recall@100']:.4f})")

    # Test run
    retriever = HybridRetriever(
        sparse_retriever=sparse,
        dense_retriever=dense,
        fusion_method="linear",
        alpha=best_alpha,
        first_stage_k=100,
    )
    hp_test = {
        "fusion_method": "linear",
        "alpha": best_alpha,
        "first_stage_k": 100,
        "alpha_selected_by": f"val_recall@10 (best alpha={best_alpha})",
        "sparse_config": SPARSE_CONFIG,
        "dense_config": DENSE_CONFIG,
    }
    test_result = _run_one(
        retriever, test_qs, "hybrid_linear_best_test", hp_test, strata, anchor, results_dir
    )
    return val_results, test_result, best_alpha


# ---------------------------------------------------------------------------
# T3-C — Cross-encoder re-ranking
# ---------------------------------------------------------------------------

def run_t3c(
    sparse, dense, df,
    val_qs, test_qs, strata, anchor,
    rrf_val_results, rrf_best_k,
    linear_val_results, linear_best_alpha,
    results_dir,
):
    print("\n" + "=" * 60)
    print("T3-C  Cross-encoder re-ranking")
    print("=" * 60)

    # Select best first stage by R@100 on val
    rrf_r100   = rrf_val_results[rrf_best_k]["metrics"]["Recall@100"]
    lin_r100   = linear_val_results[linear_best_alpha]["metrics"]["Recall@100"]
    if rrf_r100 >= lin_r100:
        best_fs_name = f"hybrid_rrf_k{rrf_best_k}"
        first_stage_retriever = HybridRetriever(
            sparse_retriever=sparse,
            dense_retriever=dense,
            fusion_method="rrf",
            rrf_k=rrf_best_k,
            first_stage_k=100,
        )
    else:
        best_fs_name = f"hybrid_linear_alpha_{linear_best_alpha}"
        first_stage_retriever = HybridRetriever(
            sparse_retriever=sparse,
            dense_retriever=dense,
            fusion_method="linear",
            alpha=linear_best_alpha,
            first_stage_k=100,
        )
    best_fs_r100 = max(rrf_r100, lin_r100)
    print(f"\n  First stage: {best_fs_name}  (val R@100={best_fs_r100:.4f})")

    article_texts = dict(zip(df["article_id"].tolist(), df["article_text"].fillna("").tolist()))

    val_results = {}
    for top_n in [50, 100]:
        reranker = CrossEncoderReranker(
            first_stage_retriever=first_stage_retriever,
            model_name=CE_MODEL,
            top_n=top_n,
            article_texts=article_texts,
            max_article_tokens=400,
        )
        exp_id = f"hybrid_rerank_minilm_top{top_n}_val"
        hp = {
            "fusion_method": "cross_encoder_rerank",
            "reranker_model": CE_MODEL,
            "top_n": top_n,
            "max_article_tokens": 400,
            "first_stage_selected": best_fs_name,
            "first_stage_r100_val": best_fs_r100,
            "sparse_config": SPARSE_CONFIG,
            "dense_config": DENSE_CONFIG,
        }
        r = _run_one(reranker, val_qs, exp_id, hp, strata, anchor, results_dir)

        # Add CE latency breakdown and fraction_truncated
        r["hyperparameters"]["latency_first_stage_ms_mean"] = reranker.last_first_stage_ms
        r["hyperparameters"]["latency_rerank_ms_mean"]      = reranker.last_rerank_ms
        r["hyperparameters"]["fraction_truncated"]           = reranker.fraction_truncated
        save_result(r, results_dir=results_dir)

        val_results[top_n] = r

    best_n = max(val_results, key=lambda n: val_results[n]["metrics"]["Recall@10"])
    print(f"\n  Best top_n = {best_n}  "
          f"(val R@10={val_results[best_n]['metrics']['Recall@10']:.4f})")

    # Test run
    reranker = CrossEncoderReranker(
        first_stage_retriever=first_stage_retriever,
        model_name=CE_MODEL,
        top_n=best_n,
        article_texts=article_texts,
        max_article_tokens=400,
    )
    hp_test = {
        "fusion_method": "cross_encoder_rerank",
        "reranker_model": CE_MODEL,
        "top_n": best_n,
        "max_article_tokens": 400,
        "first_stage_selected": best_fs_name,
        "first_stage_r100_val": best_fs_r100,
        "top_n_selected_by": f"val_recall@10 (best top_n={best_n})",
        "sparse_config": SPARSE_CONFIG,
        "dense_config": DENSE_CONFIG,
    }
    test_result = _run_one(
        reranker, test_qs, "hybrid_rerank_minilm_best_test", hp_test, strata, anchor, results_dir
    )
    # Add latency breakdown
    test_result["hyperparameters"]["latency_first_stage_ms_mean"] = reranker.last_first_stage_ms
    test_result["hyperparameters"]["latency_rerank_ms_mean"]      = reranker.last_rerank_ms
    test_result["hyperparameters"]["fraction_truncated"]           = reranker.fraction_truncated
    save_result(test_result, results_dir=results_dir)

    # T3-C Decision Gate
    t3c_r10  = test_result["metrics"]["Recall@10"]
    anc_r10  = anchor.get("metrics", {}).get("Recall@10", 0.0)
    gate_pass = t3c_r10 > anc_r10 + 0.03
    print(f"\n  [T3-C Gate] T3-C R@10={t3c_r10:.4f}  anchor R@10={anc_r10:.4f}"
          f"  diff={t3c_r10 - anc_r10:+.4f}"
          f"  -> {'PASS  (proceed to T3-D)' if gate_pass else 'FAIL  (defer T3-D)'}")

    return val_results, test_result, best_n, best_fs_name, gate_pass


# ---------------------------------------------------------------------------
# T3-D — HyDE and RAG-Fusion (conditional)
# ---------------------------------------------------------------------------

def _make_ollama_llm(model: str = "llama3.1:8b"):
    """Returns a callable(prompt) -> str using Ollama REST API."""
    import requests

    def call(prompt: str) -> str:
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["response"]

    return call


def run_t3d(
    sparse, dense, df,
    val_qs, test_qs, strata, anchor,
    rrf_best_k, results_dir,
):
    print("\n" + "=" * 60)
    print("T3-D  HyDE + RAG-Fusion")
    print("=" * 60)

    from retrieval.hybrid import hyde_retrieve, ragfusion_retrieve

    # Check Ollama availability
    try:
        import requests
        requests.get("http://localhost:11434", timeout=3)
    except Exception as e:
        print(f"  Ollama not available ({e}). Skipping T3-D.")
        return None

    llm_fn = _make_ollama_llm("llama3.1:8b")

    article_texts = dict(zip(df["article_id"].tolist(), df["article_text"].fillna("").tolist()))

    # ── HyDE ─────────────────────────────────────────────────────────────────
    hyde_cache_path = Path(f"output/hyde_cache_llama3.1_val.json")
    hyde_cache: dict[str, str] = {}
    if hyde_cache_path.exists():
        with open(hyde_cache_path) as f:
            hyde_cache = json.load(f)

    class _HyDERetriever:
        """Wrap hyde_retrieve as a standard retriever for run_experiment."""
        def __init__(self, with_rrf: bool = False):
            self.with_rrf = with_rrf
            self._rrf_retriever = HybridRetriever(
                sparse_retriever=sparse, dense_retriever=dense,
                fusion_method="rrf", rrf_k=rrf_best_k, first_stage_k=100,
            ) if with_rrf else None
            self.audit = dense.audit

        def retrieve(self, query: str, top_k: int = 10):
            if not self.with_rrf:
                return hyde_retrieve(query, llm_fn, dense, top_k=top_k, hyde_cache=hyde_cache)
            # HyDE + RRF: fuse HyDE dense result with BM25 via RRF
            hyp_ids, _ = hyde_retrieve(query, llm_fn, dense, top_k=100, hyde_cache=hyde_cache)
            bm25_ids, _ = sparse.retrieve(query, top_k=100)
            from retrieval.hybrid import HybridRetriever as HR
            t0 = time.perf_counter()
            rrf_fused = HR(sparse_retriever=sparse, dense_retriever=dense,
                           fusion_method="rrf", rrf_k=rrf_best_k, first_stage_k=100)
            merged = rrf_fused._rrf(bm25_ids, hyp_ids)
            return merged[:top_k], (time.perf_counter() - t0) * 1000.0

    for with_rrf in [False, True]:
        suffix = "_rrf" if with_rrf else ""
        exp_id = f"hybrid_hyde{suffix}_llama_val"
        hp = {
            "fusion_method": f"hyde{'_rrf' if with_rrf else ''}",
            "llm_model": "llama3.1:8b",
            "n_hypothetical_docs": 1,
            "hyde_with_rrf": with_rrf,
            "hyde_cache_path": str(hyde_cache_path),
            "dense_config": DENSE_CONFIG,
        }
        r = _run_one(_HyDERetriever(with_rrf), val_qs, exp_id, hp, strata, anchor, results_dir)

    # Save cache
    hyde_cache_path.write_text(json.dumps(hyde_cache, ensure_ascii=False, indent=2))

    # Select best HyDE variant and run on test
    hyde_val = {}
    for with_rrf in [False, True]:
        suffix = "_rrf" if with_rrf else ""
        jf = results_dir / f"hybrid_hyde{suffix}_llama_val.json"
        with open(jf) as f:
            hyde_val[with_rrf] = json.load(f)
    best_hyde = max(hyde_val, key=lambda k: hyde_val[k]["metrics"]["Recall@10"])

    hyde_test_cache_path = Path("output/hyde_cache_llama3.1_test.json")
    hyde_test_cache: dict[str, str] = {}
    r_test = _run_one(
        _HyDERetriever(best_hyde), test_qs, "hybrid_hyde_best_test",
        {**hyde_val[best_hyde]["hyperparameters"], "hyde_cache_path": str(hyde_test_cache_path)},
        strata, anchor, results_dir,
    )
    hyde_test_cache_path.write_text(json.dumps(hyde_test_cache, ensure_ascii=False, indent=2))

    # ── RAG-Fusion ────────────────────────────────────────────────────────────
    ragfusion_cache_path = Path("output/ragfusion_cache_llama3.1_val.json")
    ragfusion_cache: dict[str, list[str]] = {}
    if ragfusion_cache_path.exists():
        with open(ragfusion_cache_path) as f:
            ragfusion_cache = json.load(f)

    from retrieval.hybrid import _parse_paraphrases, _RAGFUSION_PROMPT_TEMPLATE

    def paraphrase_fn(query: str) -> list[str]:
        prompt = _RAGFUSION_PROMPT_TEMPLATE.format(question=query)
        raw = llm_fn(prompt)
        return _parse_paraphrases(raw, n=4)

    rrf_hybrid = HybridRetriever(
        sparse_retriever=sparse, dense_retriever=dense,
        fusion_method="rrf", rrf_k=rrf_best_k, first_stage_k=100,
    )

    for use_hybrid in [False, True]:
        suffix = "hybrid" if use_hybrid else "dense"
        exp_id = f"hybrid_ragfusion_{suffix}_n4_val"
        retriever_for_rf = rrf_hybrid if use_hybrid else dense

        class _RFRetriever:
            def __init__(self, _r=retriever_for_rf):
                self._r = _r
                self.audit = dense.audit

            def retrieve(self, query: str, top_k: int = 10):
                return ragfusion_retrieve(
                    query, paraphrase_fn, self._r,
                    n_paraphrases=4, rrf_k=60, top_k=top_k,
                    include_original=True, paraphrase_cache=ragfusion_cache,
                )

        hp = {
            "fusion_method": f"ragfusion_{suffix}",
            "llm_model": "llama3.1:8b",
            "n_paraphrases": 4,
            "include_original": True,
            "retriever_per_list": suffix,
            "paraphrase_cache_path": str(ragfusion_cache_path),
        }
        _run_one(_RFRetriever(), val_qs, exp_id, hp, strata, anchor, results_dir)

    ragfusion_cache_path.write_text(json.dumps(ragfusion_cache, ensure_ascii=False, indent=2))

    # Best RAG-Fusion on test
    rf_val = {}
    for suf in ["dense", "hybrid"]:
        jf = results_dir / f"hybrid_ragfusion_{suf}_n4_val.json"
        with open(jf) as f:
            rf_val[suf] = json.load(f)
    best_rf = max(rf_val, key=lambda k: rf_val[k]["metrics"]["Recall@10"])

    ragfusion_test_cache = {}
    test_retriever = rrf_hybrid if best_rf == "hybrid" else dense

    class _BestRFRetriever:
        def __init__(self):
            self.audit = dense.audit

        def retrieve(self, query: str, top_k: int = 10):
            return ragfusion_retrieve(
                query, paraphrase_fn, test_retriever,
                n_paraphrases=4, rrf_k=60, top_k=top_k,
                include_original=True, paraphrase_cache=ragfusion_test_cache,
            )

    _run_one(
        _BestRFRetriever(), test_qs, "hybrid_ragfusion_best_test",
        {**rf_val[best_rf]["hyperparameters"]},
        strata, anchor, results_dir,
    )
    Path("output/ragfusion_cache_llama3.1_test.json").write_text(
        json.dumps(ragfusion_test_cache, ensure_ascii=False, indent=2)
    )


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_summary(results_dir: Path) -> None:
    import glob
    print("\n" + "=" * 80)
    print(f"{'Experiment':<50} {'R@10':>7} {'R@100':>7} {'MRR@10':>8} {'Lat(ms)':>8}")
    print("-" * 80)

    rows = []
    for jf in sorted(results_dir.glob("*.json")):
        with open(jf) as f:
            d = json.load(f)
        m = d.get("metrics", {})
        rows.append((
            d["experiment_id"],
            m.get("Recall@10", 0),
            m.get("Recall@100", 0),
            m.get("MRR@10", 0),
            d.get("latency_ms_mean", 0),
        ))
    rows.sort(key=lambda x: x[1], reverse=True)
    for exp, r10, r100, mrr, lat in rows:
        print(f"  {exp:<48} {r10:7.4f} {r100:7.4f} {mrr:8.4f} {lat:8.1f}")
    print("=" * 80)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Tier 3 Hybrid Retrieval Experiments")
    parser.add_argument(
        "--include-augmented", action="store_true",
        help="Also run T3-D (HyDE + RAG-Fusion) if T3-C gate passes",
    )
    parser.add_argument(
        "--stage", choices=["t3a", "t3b", "t3c", "t3d", "all"], default="all",
        help="Run only a specific stage (default: all)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Tier 3 — Hybrid Retrieval Experiments")
    print("=" * 60)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\nLoading corpus ...")
    df = pd.read_parquet(CORPUS_PATH)
    print(f"  {len(df):,} articles")

    val_qs  = load_questions(subset="val")
    test_qs = load_questions(subset="test")
    strata  = load_strata()
    anchor  = _load_anchor()
    print(f"  val={len(val_qs)}, test={len(test_qs)}")
    print(f"  T3 anchor: R@10={anchor['metrics']['Recall@10']:.4f}  "
          f"R@100={anchor['metrics']['Recall@100']:.4f}")

    # ── Retrievers ────────────────────────────────────────────────────────────
    print("\nInitialising BM25 (lemmatize + concat_2x) ...")
    t0 = time.perf_counter()
    sparse = BM25Retriever(
        df=df,
        normalization="lemmatize",
        field_weighting="concat_2x",
        variant="okapi",
        k1=1.5,
        b=0.75,
    )
    print(f"  done in {time.perf_counter() - t0:.1f}s")

    print(f"\nInitialising DenseRetriever ({DENSE_MODEL}, concat_2x) ...")
    t1 = time.perf_counter()
    dense = DenseRetriever(
        df=df,
        model_name=DENSE_MODEL,
        field_weighting="concat_2x",
        query_prefix=QUERY_PREFIX,
        passage_prefix="",
        batch_size=64,
        device="cpu",
        embeddings_dir=EMBEDDINGS_DIR,
    )
    print(f"  done in {time.perf_counter() - t1:.1f}s  "
          f"(dim={dense.embedding_dim}, n={dense.n_articles})")

    print("\n  [T3-A2 SKIP] BGE-M3 not selected as Tier 2 winner.")
    print("  Reason: mE5-large-instruct+concat2x achieved R@100=0.5520 vs BGE-M3 R@100=0.5421")

    # ── T3-A ─────────────────────────────────────────────────────────────────
    rrf_val_results, rrf_test_result, best_rrf_k = None, None, 60  # defaults
    if args.stage in ("t3a", "all"):
        rrf_val_results, rrf_test_result, best_rrf_k = run_t3a(
            sparse, dense, val_qs, test_qs, strata, anchor, RESULTS_DIR
        )

    # If we only re-run from t3b onwards, reload saved val results
    if args.stage in ("t3b", "t3c", "t3d") and rrf_val_results is None:
        rrf_val_results = {}
        for k in [30, 60, 120]:
            jf = RESULTS_DIR / f"hybrid_rrf_k{k}_val.json"
            if jf.exists():
                with open(jf) as f:
                    rrf_val_results[k] = json.load(f)
        if rrf_val_results:
            best_rrf_k = max(rrf_val_results, key=lambda k: rrf_val_results[k]["metrics"]["Recall@10"])

    # ── T3-B ─────────────────────────────────────────────────────────────────
    lin_val_results, lin_test_result, best_alpha = None, None, 0.7  # defaults
    if args.stage in ("t3b", "all"):
        lin_val_results, lin_test_result, best_alpha = run_t3b(
            sparse, dense, val_qs, test_qs, strata, anchor, RESULTS_DIR
        )

    if args.stage in ("t3c", "t3d") and lin_val_results is None:
        lin_val_results = {}
        for a in [round(x * 0.1, 1) for x in range(1, 10)]:
            jf = RESULTS_DIR / f"hybrid_linear_alpha_{a}_val.json"
            if jf.exists():
                with open(jf) as f:
                    lin_val_results[a] = json.load(f)
        if lin_val_results:
            best_alpha = max(lin_val_results, key=lambda a: lin_val_results[a]["metrics"]["Recall@10"])

    # ── T3-C ─────────────────────────────────────────────────────────────────
    gate_pass = False
    if args.stage in ("t3c", "all") and rrf_val_results and lin_val_results:
        _, _, _, _, gate_pass = run_t3c(
            sparse, dense, df,
            val_qs, test_qs, strata, anchor,
            rrf_val_results, best_rrf_k,
            lin_val_results, best_alpha,
            RESULTS_DIR,
        )

    # ── T3-D ─────────────────────────────────────────────────────────────────
    if args.stage in ("t3d", "all") and args.include_augmented and gate_pass:
        run_t3d(
            sparse, dense, df,
            val_qs, test_qs, strata, anchor,
            best_rrf_k, RESULTS_DIR,
        )
    elif args.include_augmented and not gate_pass:
        print("\n[T3-D] Gate did not pass — T3-D deferred to future work.")

    # ── Summary ───────────────────────────────────────────────────────────────
    print_summary(RESULTS_DIR)


if __name__ == "__main__":
    main()
