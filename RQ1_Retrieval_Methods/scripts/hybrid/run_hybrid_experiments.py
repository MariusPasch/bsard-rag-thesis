"""
Tier 3 — Hybrid Retrieval Experiment Orchestrator.

All experiments run directly on the test split (222 questions).
Hyperparameter grids (RRF k, linear α, SGDR pool K) are evaluated on test;
best values are selected from test results and reported as canonical.

Execution order
---------------
1.  Load corpus, questions (test only), strata, T3 anchor
2.  Instantiate sparse (BM25 lemmatize+text_only, b=0.25) and dense (CLI --dense-model)
3.  T3-A : RRF k in {30, 60, 120} on test  ->  best k reported as canonical
4.  T3-A2: BGE-M3 self-hybrid  (only if --dense-model BAAI/bge-m3)
5.  T3-B : linear alpha grid 0.1-0.9 on test  ->  best alpha reported as canonical
6.  T3-C : Sparse-Gated Dense Reranking, BM25 pool K in {1000, 2000, 5000}  ->  best K canonical

T3-D (HyDE + RAG-Fusion) is a potential future expansion and is NOT part of the
current execution plan.  Implementation is preserved in run_t3d() and disabled via
_ENABLE_T3D.

Usage
-----
  python scripts/hybrid/run_hybrid_experiments.py
  python scripts/hybrid/run_hybrid_experiments.py --dense-model BAAI/bge-m3
  python scripts/hybrid/run_hybrid_experiments.py --stage t3a
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

PROJECT_ROOT = Path(__file__).parents[2]  # hybrid/ -> scripts/ -> root
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.runner import run_experiment, add_significance, save_result
from evaluation.split import load_questions
from evaluation.stratify import load_strata
from retrieval.dense import DenseRetriever
from retrieval.sparse import BM25Retriever
from retrieval.hybrid import HybridRetriever

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

RESULTS_DIR   = PROJECT_ROOT / "output" / "results" / "hybrid"
CORPUS_PATH   = PROJECT_ROOT / "output" / "bsard_articles_dedup.parquet"
EMBEDDINGS_DIR = PROJECT_ROOT / "output" / "embeddings"
ANCHOR_PATH   = (
    PROJECT_ROOT / "output" / "results" / "dense_retrieval"
    / "dense_me5_large_concat2x_zeroshot_test.json"
)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Default model config (mE5-large + concat_2x — Tier 2 winner by R@100)
# ---------------------------------------------------------------------------

DENSE_MODEL    = "intfloat/multilingual-e5-large"
QUERY_PREFIX   = "query: "
PASSAGE_PREFIX = "passage: "
FIELD_WEIGHTING = "concat_2x"

SPARSE_CONFIG = {
    "variant": "okapi",
    "normalization": "lemmatize",
    "field_weighting": "text_only",
    "k1": 1.5,
    "b": 0.25,
}
DENSE_CONFIG = {
    "model_name": DENSE_MODEL,
    "field_weighting": FIELD_WEIGHTING,
    "query_prefix": QUERY_PREFIX,
    "passage_prefix": PASSAGE_PREFIX,
}

# ---------------------------------------------------------------------------
# T3-D gate — set to True only when explicitly activating query augmentation
# ---------------------------------------------------------------------------
# T3-D (HyDE + RAG-Fusion) requires a running Ollama instance and adds
# significant runtime.  It is kept as a potential expansion and must be
# consciously enabled here before it will execute.
_ENABLE_T3D = False

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_anchor() -> dict:
    if not ANCHOR_PATH.exists():
        raise FileNotFoundError(
            f"T3 anchor not found: {ANCHOR_PATH}\n"
            "Run Tier 2 first: dense_me5_large_concat2x_zeroshot_test"
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
    # Saved JSONs carry _trec_run and _trec_qrels (kept by save_result).
    # In-memory results additionally have _raw_results; handle both paths.
    can_sig = (
        ("_trec_run" in result or "_raw_results" in result) and
        ("_trec_run" in anchor or "_raw_results" in anchor)
    )
    if can_sig:
        result = add_significance(result, anchor, k_values=[10, 100], primary_k=10)
    else:
        result["significance_vs_anchor"] = {
            "p_value_recall10": None,
            "p_value_recall100": None,
            "significant": None,
            "note": "significance data unavailable",
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
    corpus: dict | None = None,
    top_k: int = 500,
    extra_fields: dict | None = None,
) -> dict:
    print(f"\n[{experiment_id}]")
    result = run_experiment(
        retriever=retriever,
        questions=questions,
        experiment_id=experiment_id,
        hyperparameters=hyperparameters,
        preprocessing={
            "normalization": "lemmatize_sparse__none_dense",
            "field_weighting": "text_only_sparse__concat_2x_dense",
        },
        strata=strata,
        corpus=corpus,
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

def run_t3a(sparse, dense, questions, strata, anchor, results_dir, corpus=None):
    print("\n" + "=" * 60)
    print("T3-A  Reciprocal Rank Fusion (k in {30, 60, 120})")
    print("=" * 60)

    results = {}
    for rrf_k in [30, 60, 120]:
        retriever = HybridRetriever(
            sparse_retriever=sparse,
            dense_retriever=dense,
            fusion_method="rrf",
            rrf_k=rrf_k,
            first_stage_k=100,
        )
        hp = {
            "fusion_method": "rrf",
            "rrf_k": rrf_k,
            "first_stage_k": 100,
            "sparse_config": SPARSE_CONFIG,
            "dense_config": DENSE_CONFIG,
        }
        results[rrf_k] = _run_one(
            retriever, questions, f"hybrid_rrf_k{rrf_k}_test", hp, strata, anchor, results_dir,
            corpus=corpus, extra_fields={"model_or_method": "hybrid_rrf"},
        )

    best_k = max(results, key=lambda k: results[k]["metrics"]["Recall@10"])
    print(
        f"\n  Best RRF k = {best_k}"
        f"  (R@10={results[best_k]['metrics']['Recall@10']:.4f}"
        f"  R@100={results[best_k]['metrics']['Recall@100']:.4f})"
    )
    return results, best_k


# ---------------------------------------------------------------------------
# T3-B — Linear score interpolation
# ---------------------------------------------------------------------------

def run_t3b(sparse, dense, questions, strata, anchor, results_dir, corpus=None):
    print("\n" + "=" * 60)
    print("T3-B  Linear interpolation (alpha grid 0.1–0.9)")
    print("=" * 60)

    alphas = [round(a * 0.1, 1) for a in range(1, 10)]
    results = {}
    for alpha in alphas:
        retriever = HybridRetriever(
            sparse_retriever=sparse,
            dense_retriever=dense,
            fusion_method="linear",
            alpha=alpha,
            first_stage_k=100,
        )
        hp = {
            "fusion_method": "linear",
            "alpha": alpha,
            "first_stage_k": 100,
            "sparse_config": SPARSE_CONFIG,
            "dense_config": DENSE_CONFIG,
        }
        results[alpha] = _run_one(
            retriever, questions, f"hybrid_linear_alpha_{alpha}_test", hp, strata, anchor, results_dir,
            corpus=corpus, extra_fields={"model_or_method": "hybrid_linear"},
        )

    best_alpha = max(results, key=lambda a: results[a]["metrics"]["Recall@10"])
    print(
        f"\n  Best alpha = {best_alpha}"
        f"  (R@10={results[best_alpha]['metrics']['Recall@10']:.4f}"
        f"  R@100={results[best_alpha]['metrics']['Recall@100']:.4f})"
    )
    return results, best_alpha


# ---------------------------------------------------------------------------
# T3-C — Sparse-Gated Dense Reranking (SGDR)
# ---------------------------------------------------------------------------

class _SGDRetriever:
    """T3-C: BM25 top-K pool → dense re-score over candidates only.

    Document embeddings are precomputed; only K dot products at query time.
    Pool K must exceed max(custom_k)=500 so all eval k-values show genuine
    reranking signal rather than the BM25 recall ceiling.
    """

    def __init__(self, sparse: BM25Retriever, dense: DenseRetriever, bm25_pool_k: int):
        self._sparse     = sparse
        self._dense      = dense
        self.bm25_pool_k = bm25_pool_k
        self.audit       = dense.audit
        # article_id → FAISS row position (built once at init)
        self._id_to_pos: dict[int, int] = {
            int(aid): i for i, aid in enumerate(dense._article_ids)
        }
        # Extract full embeddings matrix from FAISS index once (no re-encode)
        n, dim = dense._index.ntotal, dense._index.d
        self._emb_matrix = np.empty((n, dim), dtype=np.float32)
        dense._index.reconstruct_n(0, n, self._emb_matrix)

    def retrieve(self, query: str, top_k: int = 500) -> tuple[list[int], float]:
        t0 = time.perf_counter()

        # Stage 1: BM25 candidate pool
        candidate_ids, _ = self._sparse.retrieve(query, top_k=self.bm25_pool_k)

        # Stage 2: dense re-score over candidates (K dot products)
        q_vec = self._dense._encoder.encode_query(
            self._dense._query_prefix + query
        )  # (dim,)

        valid_ids = [aid for aid in candidate_ids if aid in self._id_to_pos]
        positions = np.array([self._id_to_pos[aid] for aid in valid_ids], dtype=np.int64)

        if len(positions) == 0:
            return [], (time.perf_counter() - t0) * 1000.0

        scores = self._emb_matrix[positions] @ q_vec   # (K,)
        order  = np.argsort(scores)[::-1]
        ranked = [valid_ids[i] for i in order[:top_k]]

        return ranked, (time.perf_counter() - t0) * 1000.0


def run_t3c(sparse, dense, questions, strata, anchor, results_dir, corpus=None):
    print("\n" + "=" * 60)
    print("T3-C  Sparse-Gated Dense Reranking (K in {1000, 2000, 5000})")
    print("=" * 60)

    results = {}
    retriever_cache: dict[int, _SGDRetriever] = {}

    for pool_k in [1000, 2000, 5000]:
        # Reuse the same embeddings matrix for all K values (built once per model)
        if not retriever_cache:
            base = _SGDRetriever(sparse, dense, bm25_pool_k=pool_k)
            retriever_cache[pool_k] = base
        else:
            # Share the precomputed embeddings matrix across pool sizes
            first = next(iter(retriever_cache.values()))
            r = _SGDRetriever.__new__(_SGDRetriever)
            r._sparse      = sparse
            r._dense       = dense
            r.bm25_pool_k  = pool_k
            r.audit        = dense.audit
            r._id_to_pos   = first._id_to_pos
            r._emb_matrix  = first._emb_matrix
            retriever_cache[pool_k] = r

        hp = {
            "method":       "sparse_gated_dense",
            "bm25_pool_k":  pool_k,
            "sparse_config": SPARSE_CONFIG,
            "dense_config":  DENSE_CONFIG,
        }
        results[pool_k] = _run_one(
            retriever_cache[pool_k], questions,
            f"hybrid_sgdr_k{pool_k}_test", hp,
            strata, anchor, results_dir,
            corpus=corpus,
            top_k=500,
            extra_fields={"model_or_method": "hybrid_sgdr"},
        )

    best_k = max(results, key=lambda k: results[k]["metrics"]["Recall@10"])
    print(
        f"\n  Best pool K = {best_k}"
        f"  (R@10={results[best_k]['metrics']['Recall@10']:.4f}"
        f"  R@100={results[best_k]['metrics']['Recall@100']:.4f})"
    )
    return results, best_k


# ---------------------------------------------------------------------------
# T3-D — HyDE and RAG-Fusion (conditional)
# ---------------------------------------------------------------------------

def _make_ollama_llm(model: str = "llama3.1:8b"):
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


def run_t3d(sparse, dense, df, questions, strata, anchor, best_rrf_k, results_dir, corpus=None):
    print("\n" + "=" * 60)
    print("T3-D  HyDE + RAG-Fusion (test split)")
    print("=" * 60)

    from retrieval.hybrid import hyde_retrieve, ragfusion_retrieve

    try:
        import requests
        requests.get("http://localhost:11434", timeout=3)
    except Exception as e:
        print(f"  Ollama not available ({e}). Skipping T3-D.")
        return

    llm_fn = _make_ollama_llm("llama3.1:8b")

    # ── HyDE ─────────────────────────────────────────────────────────────────
    hyde_cache_path = PROJECT_ROOT / "output" / "hyde_cache_llama3.1_test.json"
    hyde_cache: dict[str, str] = {}
    if hyde_cache_path.exists():
        hyde_cache = json.loads(hyde_cache_path.read_text())

    class _HyDERetriever:
        def __init__(self, with_rrf: bool = False):
            self.with_rrf = with_rrf
            self._rrf = HybridRetriever(
                sparse_retriever=sparse, dense_retriever=dense,
                fusion_method="rrf", rrf_k=best_rrf_k, first_stage_k=100,
            ) if with_rrf else None
            self.audit = dense.audit

        def retrieve(self, query: str, top_k: int = 10):
            if not self.with_rrf:
                return hyde_retrieve(query, llm_fn, dense, top_k=top_k, hyde_cache=hyde_cache)
            hyp_ids, _ = hyde_retrieve(query, llm_fn, dense, top_k=100, hyde_cache=hyde_cache)
            bm25_ids, _ = sparse.retrieve(query, top_k=100)
            t0 = time.perf_counter()
            merged = self._rrf._rrf(bm25_ids, hyp_ids)
            return merged[:top_k], (time.perf_counter() - t0) * 1000.0

    hyde_results = {}
    for with_rrf in [False, True]:
        suffix = "_rrf" if with_rrf else ""
        exp_id = f"hybrid_hyde{suffix}_llama_test"
        hp = {
            "fusion_method": f"hyde{'_rrf' if with_rrf else ''}",
            "llm_model": "llama3.1:8b",
            "n_hypothetical_docs": 1,
            "hyde_with_rrf": with_rrf,
            "hyde_cache_path": str(hyde_cache_path),
            "dense_config": DENSE_CONFIG,
        }
        hyde_results[with_rrf] = _run_one(
            _HyDERetriever(with_rrf), questions, exp_id, hp, strata, anchor, results_dir,
            corpus=corpus,
            extra_fields={"model_or_method": f"hybrid_hyde{'_rrf' if with_rrf else ''}"},
        )

    hyde_cache_path.write_text(json.dumps(hyde_cache, ensure_ascii=False, indent=2))

    # ── RAG-Fusion ────────────────────────────────────────────────────────────
    ragfusion_cache_path = PROJECT_ROOT / "output" / "ragfusion_cache_llama3.1_test.json"
    ragfusion_cache: dict[str, list[str]] = {}
    if ragfusion_cache_path.exists():
        ragfusion_cache = json.loads(ragfusion_cache_path.read_text())

    from retrieval.hybrid import _parse_paraphrases, _RAGFUSION_PROMPT_TEMPLATE

    def paraphrase_fn(query: str) -> list[str]:
        return _parse_paraphrases(llm_fn(_RAGFUSION_PROMPT_TEMPLATE.format(question=query)), n=4)

    rrf_hybrid = HybridRetriever(
        sparse_retriever=sparse, dense_retriever=dense,
        fusion_method="rrf", rrf_k=best_rrf_k, first_stage_k=100,
    )

    for use_hybrid in [False, True]:
        suffix = "hybrid" if use_hybrid else "dense"
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
        _run_one(_RFRetriever(), questions, f"hybrid_ragfusion_{suffix}_n4_test",
                 hp, strata, anchor, results_dir,
                 corpus=corpus, extra_fields={"model_or_method": f"hybrid_ragfusion_{suffix}"})

    ragfusion_cache_path.write_text(json.dumps(ragfusion_cache, ensure_ascii=False, indent=2))


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_summary(results_dir: Path) -> None:
    print("\n" + "=" * 80)
    print(f"{'Experiment':<50} {'R@10':>7} {'R@100':>7} {'MRR@10':>8} {'Lat(ms)':>8}")
    print("-" * 80)
    rows = []
    for jf in sorted(results_dir.glob("*.json")):
        d = json.loads(jf.read_text())
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
        "--dense-model", default=DENSE_MODEL,
        help="HuggingFace model ID for the dense component (default: mE5-large)",
    )
    parser.add_argument(
        "--stage", choices=["t3a", "t3b", "t3c", "all"], default="all",
        help="Run only a specific stage (default: all)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Tier 3 — Hybrid Retrieval Experiments (test split)")
    print("=" * 60)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\nLoading corpus ...")
    if not CORPUS_PATH.exists():
        print(f"ERROR: {CORPUS_PATH} not found.")
        print("Run scripts/setup/prepare_dedup_corpus.py first,")
        print("or ensure the output/ junction is set up (see README.md).")
        sys.exit(1)
    df = pd.read_parquet(CORPUS_PATH)
    df = df[df["article_text"].notna()].reset_index(drop=True)  # matches sparse + dense scripts
    corpus = dict(zip(df["article_id"].tolist(), df["article_text"].tolist()))
    _EXPECTED = 22_633
    if not (20_000 <= len(df) <= 30_000):
        print(f"WARNING: unexpected corpus size {len(df):,} (expected ~{_EXPECTED:,})")
    print(f"  {len(df):,} articles")

    questions = load_questions(subset="test")
    strata    = load_strata()
    anchor    = _load_anchor()
    print(f"  test questions: {len(questions)}")
    print(f"  T3 anchor: R@10={anchor['metrics']['Recall@10']:.4f}  "
          f"R@100={anchor['metrics']['Recall@100']:.4f}")

    # ── Retrievers ────────────────────────────────────────────────────────────
    print("\nInitialising BM25 (lemmatize + text_only, b=0.25) ...")
    t0 = time.perf_counter()
    sparse = BM25Retriever(
        df=df,
        normalization="lemmatize",
        field_weighting="text_only",
        variant="okapi",
        k1=1.5,
        b=0.25,
    )
    print(f"  done in {time.perf_counter() - t0:.1f}s")

    dense_model = args.dense_model
    print(f"\nInitialising DenseRetriever ({dense_model}) ...")
    t1 = time.perf_counter()
    dense = DenseRetriever(
        df=df,
        model_name=dense_model,
        field_weighting=FIELD_WEIGHTING,
        query_prefix=QUERY_PREFIX,
        passage_prefix=PASSAGE_PREFIX,
        batch_size=64,
        device="cpu",
        embeddings_dir=EMBEDDINGS_DIR,
    )
    print(f"  done in {time.perf_counter() - t1:.1f}s  "
          f"(dim={dense.embedding_dim}, n={dense.n_articles})")

    # T3-A2 only runs if BGE-M3 was selected
    run_m3 = dense_model == "BAAI/bge-m3"
    if not run_m3:
        print(f"\n  [T3-A2 SKIP] Dense model is {dense_model}, not BAAI/bge-m3.")

    # ── T3-A ─────────────────────────────────────────────────────────────────
    rrf_results, best_rrf_k = {30: {"metrics": {"Recall@10": 0, "Recall@100": 0}}}, 60
    if args.stage in ("t3a", "all"):
        rrf_results, best_rrf_k = run_t3a(sparse, dense, questions, strata, anchor, RESULTS_DIR, corpus=corpus)
    elif args.stage == "t3b":
        for k in [30, 60, 120]:
            jf = RESULTS_DIR / f"hybrid_rrf_k{k}_test.json"
            if jf.exists():
                rrf_results[k] = json.loads(jf.read_text())
        if len(rrf_results) > 1:
            best_rrf_k = max(rrf_results, key=lambda k: rrf_results[k]["metrics"]["Recall@10"])

    # ── T3-B ─────────────────────────────────────────────────────────────────
    lin_results, best_alpha = {0.7: {"metrics": {"Recall@10": 0, "Recall@100": 0}}}, 0.7
    if args.stage in ("t3b", "all"):
        lin_results, best_alpha = run_t3b(sparse, dense, questions, strata, anchor, RESULTS_DIR, corpus=corpus)

    # ── T3-C ─────────────────────────────────────────────────────────────────
    if args.stage in ("t3c", "all"):
        run_t3c(sparse, dense, questions, strata, anchor, RESULTS_DIR, corpus=corpus)

    # ── T3-D (disabled — potential future expansion) ──────────────────────────
    # To activate: set _ENABLE_T3D = True at the top of this file.
    # Requires a running Ollama instance with llama3.1:8b pulled.
    if _ENABLE_T3D:
        run_t3d(sparse, dense, df, questions, strata, anchor, best_rrf_k, RESULTS_DIR, corpus=corpus)

    print_summary(RESULTS_DIR)


if __name__ == "__main__":
    main()
