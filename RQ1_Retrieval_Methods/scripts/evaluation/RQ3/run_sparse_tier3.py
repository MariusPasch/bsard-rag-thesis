"""
Run Tier 3 autonomous evaluation for all 12 sparse retrieval experiments
on the fixed 48-question T3 subset (evaluation/tier3_subset.json).

Results are merged into the existing result JSONs under
  output/results/sparse_retrieval/{exp_id}_test.json
  → subset_metrics.metrics  (T3/* and T2-umbrela/* keys added)

Prerequisites
─────────────
  pip install "ragas>=0.2" datasets
  set OPENAI_API_KEY=sk-...          (Windows: $env:OPENAI_API_KEY="sk-...")

Usage (from project root)
─────────────────────────
  .venv/Scripts/python scripts/evaluation/RQ3/run_sparse_tier3.py
  .venv/Scripts/python scripts/evaluation/RQ3/run_sparse_tier3.py --dry-run
  .venv/Scripts/python scripts/evaluation/RQ3/run_sparse_tier3.py --exp bm25_anchor
  .venv/Scripts/python scripts/evaluation/RQ3/run_sparse_tier3.py --skip-done

Cost estimate: ~$0.13 per experiment × 12 = ~$1.50 total (gpt-4o-mini, k=10)

Per TIER1_SPARSE_RETRIEVAL_PLAN.md §10.6.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Callable

import pandas as pd

# Make the project root importable when running as a script
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from bsard_evaluation import EvaluationHarness
from bsard_evaluation.config import TierConfig
from evaluation.runner import (
    _to_trec_run, _to_trec_qrels, build_contexts_with_ranks,
    _trec_metrics_to_legacy,
)
from retrieval.sparse import BM25Retriever, FTS5Retriever, TFIDFRetriever

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_ROOT         = Path(__file__).parent.parent.parent.parent
PARQUET_PATH  = _ROOT / "output" / "bsard_articles_dedup.parquet"
DB_PATH       = _ROOT / "output" / "bsard_corpus.db"
SUBSET_FILE   = _ROOT / "evaluation" / "data" / "tier3_subset.json"
RESULTS_DIR   = _ROOT / "output" / "results" / "sparse_retrieval"
SIDECAR_DIR   = RESULTS_DIR / "tier3_per_query"

# ---------------------------------------------------------------------------
# Tier 3 TierConfig  (§10.6 of TIER1_SPARSE_RETRIEVAL_PLAN.md)
# ---------------------------------------------------------------------------

_K_FULL = [1, 5, 10, 20, 50, 100, 200, 500]

TIER3_CFG = TierConfig(
    tiers=[0, 1, 2, 3],
    custom_k=_K_FULL,
    tier3_components=["umbrela", "erag", "ragas_wa"],
    tier3_sample_size=None,          # subset already pre-filtered to 48 q
    tier3_use_api=True,
    tier3_model="gpt-4o-mini",
    umbrela_judge_model="gpt-4o-mini",
)

# ---------------------------------------------------------------------------
# Experiment registry
# Each entry: (short_name, retriever_factory(df) -> retriever)
# short_name must match the experiment_id in the result JSON (without _test).
# ---------------------------------------------------------------------------

def _experiments(df: pd.DataFrame) -> list[tuple[str, object]]:
    """Return list of (short_name, retriever) for all 12 experiments."""
    return [
        # ── BM25 experiments ────────────────────────────────────────────────
        ("bm25_anchor",
         BM25Retriever(df, normalization="none",      field_weighting="text_only",
                       variant="okapi", k1=1.5, b=0.75)),
        ("bm25_lemmatize_text_only",
         BM25Retriever(df, normalization="lemmatize", field_weighting="text_only",
                       variant="okapi", k1=1.5, b=0.75)),
        ("bm25_stem_text_only",
         BM25Retriever(df, normalization="stem",      field_weighting="text_only",
                       variant="okapi", k1=1.5, b=0.75)),
        ("bm25_none_concat_2x",
         BM25Retriever(df, normalization="none",      field_weighting="concat_2x",
                       variant="okapi", k1=1.5, b=0.75)),
        ("bm25_lemmatize_concat_2x",
         BM25Retriever(df, normalization="lemmatize", field_weighting="concat_2x",
                       variant="okapi", k1=1.5, b=0.75)),
        # Tuned: best k1=1.5, b=0.25 from val-split grid search
        ("bm25_tuned_k11.5_b0.25",
         BM25Retriever(df, normalization="lemmatize", field_weighting="text_only",
                       variant="okapi", k1=1.5, b=0.25)),
        # BM25 variants
        ("bm25_plus_lemmatize",
         BM25Retriever(df, normalization="lemmatize", field_weighting="text_only",
                       variant="plus")),
        ("bm25_l_lemmatize",
         BM25Retriever(df, normalization="lemmatize", field_weighting="text_only",
                       variant="l")),
        # ── FTS5 ────────────────────────────────────────────────────────────
        ("fts5_default",
         FTS5Retriever(DB_PATH)),
        # ── TF-IDF ──────────────────────────────────────────────────────────
        ("tfidf_none_text_only",
         TFIDFRetriever(df, normalization="none",      field_weighting="text_only")),
        ("tfidf_lemmatize_text_only",
         TFIDFRetriever(df, normalization="lemmatize", field_weighting="text_only")),
        ("tfidf_lemmatize_concat_2x",
         TFIDFRetriever(df, normalization="lemmatize", field_weighting="concat_2x")),
    ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_subset() -> list[dict]:
    """Load the 48-question T3 subset from evaluation/tier3_subset.json."""
    data = json.loads(SUBSET_FILE.read_text(encoding="utf-8"))
    return data["questions"]


def load_articles() -> pd.DataFrame:
    df = pd.read_parquet(
        PARQUET_PATH,
        columns=["article_id", "bsard_id", "law_code", "chapter_title",
                 "article_number", "article_text", "has_cross_references"],
    )
    return df[df["article_text"].notna()].reset_index(drop=True)


def retrieve_subset(
    retriever,
    questions: list[dict],
    top_k: int = 100,
) -> tuple[dict[int, list[int]], dict[int, float]]:
    """
    Run the retriever over the subset questions.

    Returns
    -------
    results   : {question_id: [article_id, ...]} ranked best-first
    latencies : {question_id: latency_ms}
    """
    results: dict[int, list[int]]   = {}
    latencies: dict[int, float]     = {}
    for q in questions:
        ranked, lat_ms = retriever.retrieve(q["question_text"], top_k=top_k)
        results[q["question_id"]]   = ranked
        latencies[q["question_id"]] = lat_ms
    return results, latencies


def merge_tier3_into_result(
    result_path: Path,
    raw_harness_metrics: dict[str, float],
) -> dict[str, float]:
    """
    Load existing result JSON, merge all harness output into
    subset_metrics.metrics, then save.

    Keys saved:
      - Legacy-mapped T0/T1/T2 supervised metrics (same format as the full
        test-set metrics, so analysis notebooks can compare directly)
      - T3/* autonomous metrics
      - T2-umbrela/* UMBRELA-bridged Tier-2 metrics

    Returns the dict of newly added/updated keys.
    """
    data = json.loads(result_path.read_text(encoding="utf-8"))

    if "subset_metrics" not in data:
        raise KeyError(f"No 'subset_metrics' block in {result_path.name}. "
                       "Run compute_subset_metrics.py first.")

    target = data["subset_metrics"].setdefault("metrics", {})

    # Supervised metrics (T0/T1/T2) mapped to legacy flat keys
    supervised = _trec_metrics_to_legacy(raw_harness_metrics)
    target.update(supervised)

    # Autonomous metrics (T3/*) and UMBRELA-bridged Tier-2 (T2-umbrela/*)
    autonomous = {k: v for k, v in raw_harness_metrics.items()
                  if k.startswith("T3/") or k.startswith("T2-umbrela/")}
    target.update(autonomous)

    new_keys = {**supervised, **autonomous}

    result_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return new_keys


def tier3_already_done(result_path: Path, sidecar_dir: Path) -> bool:
    """
    Return True iff:
      (a) T3/AQS is present in subset_metrics.metrics, AND
      (b) the per-query sidecar exists for this experiment.

    Both are required for downstream RQ3a/b analysis, so a system from a
    pre-patch run (T3/AQS present but no sidecar) is *not* "done".
    """
    try:
        data = json.loads(result_path.read_text(encoding="utf-8"))
        has_aqs = "T3/AQS" in data.get("subset_metrics", {}).get("metrics", {})
    except Exception:
        return False
    sidecar_path = sidecar_dir / result_path.name
    return has_aqs and sidecar_path.exists()


def write_per_query_sidecar(
    sidecar_dir: Path,
    exp_id: str,
    t3_result,
    contexts_with_ranks: dict,
) -> Path:
    """
    Persist the per-(q,d) and per-query Tier 3 data captured on the
    Tier3Result so RQ3a stratified analysis can run at full granularity.

    Writes to <sidecar_dir>/<exp_id>.json with this schema:

        {
          "ranks":    {qid: [doc_id_at_rank_1, doc_id_at_rank_2, ...]},
          "umbrela":  {qid: {doc_id: grade 0-3}, ...} | null,
          "erag":     {qid: {doc_id: 0|1},        ...} | null,
          "ragas_wa": {qid: float in [0, 1],      ...} | null,
          "hyde":     {qid: hyde_response_text,    ...} | null
        }

    `ranks` provides the canonical retrieval rank order so that downstream
    panel assembly does not have to rely on dict insertion order.  None
    entries on the evaluator fields indicate the component did not run.
    """
    ranks = {
        str(qid): [str(doc_id) for doc_id, _doc_text, _rank in ctx]
        for qid, ctx in contexts_with_ranks.items()
    }
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    sidecar_path = sidecar_dir / f"{exp_id}.json"
    payload = {
        "ranks":    ranks,
        "umbrela":  t3_result.umbrela_qrels,
        "erag":     t3_result.erag_per_doc,
        "ragas_wa": t3_result.ragas_wa_per_query,
        "hyde":     t3_result.hyde_responses,
    }
    sidecar_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return sidecar_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run Tier 3 autonomous eval on the 48-question T3 subset "
                    "for all 12 sparse retrieval experiments."
    )
    parser.add_argument("--dry-run",   action="store_true",
                        help="Build retrievers and check setup; skip LLM calls.")
    parser.add_argument("--skip-done", action="store_true",
                        help="Skip experiments that already have T3/AQS in their result JSON.")
    parser.add_argument("--exp",       type=str, default=None,
                        help="Run only this experiment (short name, e.g. bm25_anchor).")
    args = parser.parse_args()

    # ── Preflight checks ─────────────────────────────────────────────────────
    print("=" * 70)
    print("TIER 3 — SPARSE RETRIEVAL: AUTONOMOUS EVALUATION  (subset n=48)")
    print("=" * 70)

    if not os.environ.get("OPENAI_API_KEY"):
        print("\n[ERROR] OPENAI_API_KEY is not set.")
        print("  Windows PowerShell: $env:OPENAI_API_KEY='sk-...'")
        print("  Windows cmd:        set OPENAI_API_KEY=sk-...")
        if not args.dry_run:
            sys.exit(1)

    try:
        import ragas
        print(f"\n[OK] ragas {ragas.__version__}")
    except ImportError:
        print("\n[ERROR] ragas is not installed.")
        print("  pip install 'ragas>=0.2' datasets")
        if not args.dry_run:
            sys.exit(1)

    try:
        import datasets
        print(f"[OK] datasets {datasets.__version__}")
    except ImportError:
        print("[ERROR] datasets is not installed.")
        print("  pip install datasets")
        if not args.dry_run:
            sys.exit(1)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\n[1] Loading corpus …")
    df = load_articles()
    corpus = dict(zip(df["article_id"], df["article_text"].fillna("")))
    print(f"    Articles: {len(corpus):,}")

    print("\n[2] Loading T3 subset …")
    subset_questions = load_subset()
    print(f"    Questions: {len(subset_questions)}")
    print(f"    IDs: {[q['question_id'] for q in subset_questions[:5]]} …")

    # ── Build all retrievers ──────────────────────────────────────────────────
    print("\n[3] Building retrievers …")
    experiments = _experiments(df)
    if args.exp:
        experiments = [(n, r) for n, r in experiments if n == args.exp]
        if not experiments:
            print(f"[ERROR] Unknown experiment '{args.exp}'. "
                  f"Valid: {[n for n, _ in _experiments(df)]}")
            sys.exit(1)
    print(f"    {len(experiments)} retriever(s) ready.")

    if args.dry_run:
        print("\n[DRY RUN] Setup OK. Skipping LLM evaluation calls.")
        print("TierConfig summary:")
        print(TIER3_CFG.summary())
        return

    # ── Run Tier 3 for each experiment ────────────────────────────────────────
    print(f"\n[4] Running Tier 3 evaluation …")
    print(f"    Components: {TIER3_CFG.tier3_components}")
    print(f"    Model:      {TIER3_CFG.tier3_model}")
    print()

    harness = EvaluationHarness(TIER3_CFG)

    total = len(experiments)
    for idx, (short_name, retriever) in enumerate(experiments, 1):
        exp_id      = f"{short_name}_test"
        result_path = RESULTS_DIR / f"{exp_id}.json"

        print(f"[{idx}/{total}] {short_name}")

        if not result_path.exists():
            print(f"  [SKIP] Result file not found: {result_path.name}")
            continue

        if args.skip_done and tier3_already_done(result_path, SIDECAR_DIR):
            print(f"  [SKIP] T3/AQS + sidecar already present — use without --skip-done to overwrite.")
            continue

        # ── Retrieve on subset ───────────────────────────────────────────────
        print(f"  Retrieving on {len(subset_questions)} questions …", end=" ", flush=True)
        t0 = time.perf_counter()
        results, latencies = retrieve_subset(retriever, subset_questions, top_k=100)
        elapsed = time.perf_counter() - t0
        print(f"done ({elapsed:.1f}s)")

        # ── Build harness inputs ─────────────────────────────────────────────
        trec_run   = _to_trec_run(results)
        trec_qrels = _to_trec_qrels(
            {q["question_id"]: q["relevant_article_ids"] for q in subset_questions}
        )
        latency_dict = {str(q["question_id"]): latencies[q["question_id"]]
                        for q in subset_questions}
        queries_dict = {str(q["question_id"]): q["question_text"]
                        for q in subset_questions}
        contexts_with_ranks = build_contexts_with_ranks(results, corpus, k=10)

        # ── Evaluate ─────────────────────────────────────────────────────────
        print(f"  Running Tier 3 LLM evaluation …")
        t0 = time.perf_counter()
        metrics = harness.evaluate(
            qrels=trec_qrels,
            run=trec_run,
            latencies=latency_dict,
            queries=queries_dict,
            contexts_with_ranks=contexts_with_ranks,
            verbose=True,
        )
        elapsed = time.perf_counter() - t0

        # ── Extract and report key T3 metrics ────────────────────────────────
        aqs       = metrics.get("T3/AQS",             float("nan"))
        umbrela   = metrics.get("T3/umbrela/mean",    float("nan"))
        erag      = metrics.get("T3/erag/mean",        float("nan"))
        ragas_wa  = metrics.get("T3/ragas_wa/mean",   float("nan"))
        ragas_wb  = metrics.get("T3/ragas_wb/mean",   float("nan"))
        ndcg_umb  = metrics.get("T2-umbrela/P2/NDCG@10", float("nan"))

        print(f"  AQS={aqs:.4f}  umbrela={umbrela:.4f}  erag={erag:.4f}  "
              f"ragas_wa={ragas_wa:.4f}  ragas_wb={ragas_wb:.4f}  "
              f"NDCG@10(umb)={ndcg_umb:.4f}  ({elapsed:.0f}s)")

        # ── Merge into result JSON ────────────────────────────────────────────
        new_keys = merge_tier3_into_result(result_path, metrics)
        print(f"  Saved {len(new_keys)} new keys to {result_path.name}")

        # ── Persist per-query sidecar for RQ3a stratified analysis ───────────
        if harness.last_tier3_result is not None:
            sidecar_path = write_per_query_sidecar(
                SIDECAR_DIR, exp_id, harness.last_tier3_result, contexts_with_ranks
            )
            print(f"  Sidecar: {sidecar_path.relative_to(_ROOT)}")
        print()

    # ── Summary ──────────────────────────────────────────────────────────────
    print("=" * 70)
    print("SUMMARY  (sorted by T3/AQS)")
    print("=" * 70)
    rows = []
    for f in sorted(RESULTS_DIR.glob("*_test.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        sm = d.get("subset_metrics", {}).get("metrics", {})
        rows.append({
            "experiment": d["experiment_id"].replace("_test", ""),
            "AQS":       sm.get("T3/AQS",             float("nan")),
            "umbrela":   sm.get("T3/umbrela/mean",    float("nan")),
            "erag":      sm.get("T3/erag/mean",        float("nan")),
            "ragas_wa":  sm.get("T3/ragas_wa/mean",   float("nan")),
            "ragas_wb":  sm.get("T3/ragas_wb/mean",   float("nan")),
            "R@10_sub":  sm.get("Recall@10",          float("nan")),
            "R@100_sub": sm.get("Recall@100",         float("nan")),
        })
    rows.sort(key=lambda x: x["AQS"] if not (x["AQS"] != x["AQS"]) else -1,
              reverse=True)
    hdr = (f"{'Experiment':<38} {'AQS':>6} {'UMBL':>6} {'eRAG':>6} "
           f"{'RGS-A':>6} {'RGS-B':>6} {'R@10':>6} {'R@100':>6}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        def _f(v): return f"{v:.4f}" if v == v else "  —   "
        print(
            f"{r['experiment']:<38} "
            f"{_f(r['AQS']):>6} {_f(r['umbrela']):>6} {_f(r['erag']):>6} "
            f"{_f(r['ragas_wa']):>6} {_f(r['ragas_wb']):>6} "
            f"{_f(r['R@10_sub']):>6} {_f(r['R@100_sub']):>6}"
        )
    print(f"\nResults merged into: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
