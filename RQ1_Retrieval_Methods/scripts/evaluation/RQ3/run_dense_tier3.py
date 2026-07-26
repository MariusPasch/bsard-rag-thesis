"""
Run Tier 3 autonomous evaluation for the 7 dense bi-encoder experiments
(Qwen3 variants excluded — GPU constraint) on the fixed 48-question
T3 subset (evaluation/tier3_subset.json).

Mirrors scripts/evaluation/RQ3/run_sparse_tier3.py.  Re-uses cached
corpus embeddings under output/embeddings/ so retrieval on 48 queries
runs in seconds; the LLM evaluation is the dominant cost.

Per-(q,d) eRAG, per-query RAGAS-WA, and HyDE responses are persisted to
sidecar files under output/results/dense_retrieval/tier3_per_query/<exp>.json
to enable RQ3a stratified analysis.

Prerequisites
─────────────
  pip install "ragas>=0.2" datasets sentence-transformers faiss-cpu
  # And: subset_metrics block populated in each dense result JSON
  .venv/Scripts/python scripts/evaluation/compute_subset_metrics.py \\
      --results-dir output/results/dense_retrieval

  set OPENAI_API_KEY=sk-...           (Windows: $env:OPENAI_API_KEY="sk-...")

Usage (from project root)
─────────────────────────
  .venv/Scripts/python scripts/evaluation/RQ3/run_dense_tier3.py
  .venv/Scripts/python scripts/evaluation/RQ3/run_dense_tier3.py --dry-run
  .venv/Scripts/python scripts/evaluation/RQ3/run_dense_tier3.py --exp dense_me5_large
  .venv/Scripts/python scripts/evaluation/RQ3/run_dense_tier3.py --skip-done

Cost estimate: ~$0.66 per experiment × 7 ≈ ~$4.6 total
(gpt-4o-mini, k=10, components = umbrela + erag + ragas_wa).

RQ3 Tier 3 plan: RQ3_Autonomous_Evaluation/analysis/rq3_tier3/README.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

# Make the project root importable when running as a script
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from bsard_evaluation import EvaluationHarness
from bsard_evaluation.config import TierConfig
from evaluation.runner import (
    _to_trec_run, _to_trec_qrels, build_contexts_with_ranks,
    _trec_metrics_to_legacy,
)
from retrieval.dense import DenseRetriever

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_ROOT          = Path(__file__).parent.parent.parent.parent
PARQUET_PATH   = _ROOT / "output" / "bsard_articles_dedup.parquet"
SUBSET_FILE    = _ROOT / "evaluation" / "data" / "tier3_subset.json"
RESULTS_DIR    = _ROOT / "output" / "results" / "dense_retrieval"
SIDECAR_DIR    = RESULTS_DIR / "tier3_per_query"
EMBEDDINGS_DIR = _ROOT / "output" / "embeddings"

# ---------------------------------------------------------------------------
# Tier 3 TierConfig — RAGAS-WB excluded (failed run); ARES not in scope
# ---------------------------------------------------------------------------

_K_FULL = [1, 5, 10, 20, 50, 100, 200, 500]

TIER3_CFG = TierConfig(
    tiers=[0, 1, 2, 3],
    custom_k=_K_FULL,
    tier3_components=["umbrela", "erag", "ragas_wa"],
    tier3_sample_size=None,           # subset already pre-filtered to 48 q
    tier3_use_api=True,
    tier3_model="gpt-4o-mini",
    umbrela_judge_model="gpt-4o-mini",
)

# ---------------------------------------------------------------------------
# Experiment registry — 7 dense systems, Qwen3 excluded
# Each entry: (short_name, model_name, field_weighting, query_prefix,
#              passage_prefix, max_seq_length_override, batch_size)
# short_name + "_zeroshot_test" must match the dense result JSON filename.
# ---------------------------------------------------------------------------

_DENSE_EXPERIMENTS: list[tuple[str, str, str, str, str, int | None, int]] = [
    # (short_name, model_name, field_weighting, q_prefix, p_prefix, max_seq, batch)
    ("dense_camembert_base",
     "camembert-base",
     "text_only", "", "", None, 64),
    ("dense_camembert_lg",
     "dangvantuan/sentence-camembert-large",
     "text_only", "", "", 512, 64),
    ("dense_me5_base",
     "intfloat/multilingual-e5-base",
     "text_only", "query: ", "passage: ", None, 64),
    ("dense_me5_large",
     "intfloat/multilingual-e5-large",
     "text_only", "query: ", "passage: ", None, 64),
    ("dense_me5_large_concat2x",
     "intfloat/multilingual-e5-large",
     "concat_2x", "query: ", "passage: ", None, 64),
    ("dense_bge_m3",
     "BAAI/bge-m3",
     "text_only", "", "", 1024, 16),
    ("dense_mpnet_multi",
     "paraphrase-multilingual-mpnet-base-v2",
     "text_only", "", "", None, 64),
]


def _build_retriever(
    df: pd.DataFrame,
    model_name: str,
    field_weighting: str,
    query_prefix: str,
    passage_prefix: str,
    max_seq_length_override: int | None,
    batch_size: int,
    device: str,
) -> DenseRetriever:
    return DenseRetriever(
        df=df,
        model_name=model_name,
        field_weighting=field_weighting,
        passage_prefix=passage_prefix,
        query_prefix=query_prefix,
        batch_size=batch_size,
        device=device,
        embeddings_dir=EMBEDDINGS_DIR,
        max_seq_length_override=max_seq_length_override,
    )


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
    retriever: DenseRetriever,
    questions: list[dict],
    top_k: int = 100,
) -> tuple[dict[int, list[int]], dict[int, float]]:
    """Run the retriever over the subset; return ranked results and latencies."""
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
    Load existing dense result JSON, merge harness output into
    subset_metrics.metrics, then save.

    Same shape as the sparse merge helper in run_sparse_tier3.py.
    """
    data = json.loads(result_path.read_text(encoding="utf-8"))

    if "subset_metrics" not in data:
        raise KeyError(
            f"No 'subset_metrics' block in {result_path.name}. "
            f"Run compute_subset_metrics.py --results-dir output/results/dense_retrieval first."
        )

    target = data["subset_metrics"].setdefault("metrics", {})

    supervised = _trec_metrics_to_legacy(raw_harness_metrics)
    target.update(supervised)

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
    Return True iff T3/AQS is present in subset_metrics.metrics AND the
    per-query sidecar exists.  Both are required for downstream RQ3a/b
    analysis, so a system without its sidecar counts as not-done.
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
    Persist UMBRELA qrels, eRAG per-doc scores, RAGAS-WA per-query scores,
    HyDE responses, and the canonical retrieval rank order to
    <sidecar_dir>/<exp_id>.json.  Schema matches the sparse sidecar so
    downstream panel-assembly is family-agnostic.
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
                    "for the 7 non-Qwen3 dense retrieval experiments."
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Build retrievers and check setup; skip LLM calls.")
    parser.add_argument("--skip-done", action="store_true",
                        help="Skip experiments that already have T3/AQS.")
    parser.add_argument("--exp", type=str, default=None,
                        help="Run only this experiment (short name, e.g. dense_me5_large).")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda", "mps"],
                        help="Encoding device (default: cpu — CPU is sufficient for 48 queries).")
    args = parser.parse_args()

    # ── Preflight checks ─────────────────────────────────────────────────────
    print("=" * 70)
    print("TIER 3 — DENSE RETRIEVAL: AUTONOMOUS EVALUATION  (subset n=48)")
    print("Components: umbrela + erag + ragas_wa  (no WB, no ARES)")
    print("=" * 70)

    if not os.environ.get("OPENAI_API_KEY"):
        print("\n[ERROR] OPENAI_API_KEY is not set.")
        print("  Windows PowerShell: $env:OPENAI_API_KEY='sk-...'")
        if not args.dry_run:
            sys.exit(1)

    try:
        import ragas
        print(f"\n[OK] ragas {ragas.__version__}")
    except ImportError:
        print("\n[ERROR] ragas is not installed.  pip install 'ragas>=0.2' datasets")
        if not args.dry_run:
            sys.exit(1)

    try:
        import sentence_transformers
        print(f"[OK] sentence-transformers {sentence_transformers.__version__}")
    except ImportError:
        print("[ERROR] sentence-transformers is not installed.  pip install sentence-transformers")
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

    # ── Filter experiments ────────────────────────────────────────────────────
    experiments = list(_DENSE_EXPERIMENTS)
    if args.exp:
        experiments = [e for e in experiments if e[0] == args.exp]
        if not experiments:
            valid = [e[0] for e in _DENSE_EXPERIMENTS]
            print(f"[ERROR] Unknown experiment '{args.exp}'. Valid: {valid}")
            sys.exit(1)
    print(f"\n[3] {len(experiments)} dense experiment(s) selected.")

    if args.dry_run:
        print("\n[DRY RUN] Setup OK. Skipping retrieval and LLM evaluation.")
        print("TierConfig summary:")
        print(TIER3_CFG.summary())
        print("\nExperiments:")
        for stem, model, fw, qp, pp, ms, bs in experiments:
            print(f"  {stem:<30} model={model:<48} fw={fw} qp={qp!r} pp={pp!r} max_seq={ms} batch={bs}")
        return

    # ── Run Tier 3 for each experiment ────────────────────────────────────────
    print(f"\n[4] Running Tier 3 evaluation …")
    print(f"    Components: {TIER3_CFG.tier3_components}")
    print(f"    Model:      {TIER3_CFG.tier3_model}")
    print(f"    Device:     {args.device}")
    print()

    harness = EvaluationHarness(TIER3_CFG)

    total = len(experiments)
    for idx, (stem, model_name, fw, qp, pp, ms, bs) in enumerate(experiments, 1):
        exp_id      = f"{stem}_zeroshot_test"
        result_path = RESULTS_DIR / f"{exp_id}.json"

        print(f"[{idx}/{total}] {stem}")

        if not result_path.exists():
            print(f"  [SKIP] Result file not found: {result_path.name}")
            continue

        if args.skip_done and tier3_already_done(result_path, SIDECAR_DIR):
            print(f"  [SKIP] T3/AQS + sidecar already present — use without --skip-done to overwrite.")
            continue

        # ── Build retriever ──────────────────────────────────────────────────
        print(f"  Building retriever (model={model_name}, fw={fw}) …", end=" ", flush=True)
        t0 = time.perf_counter()
        retriever = _build_retriever(
            df=df, model_name=model_name, field_weighting=fw,
            query_prefix=qp, passage_prefix=pp,
            max_seq_length_override=ms, batch_size=bs, device=args.device,
        )
        print(f"done ({time.perf_counter() - t0:.1f}s)")

        # ── Retrieve on subset ───────────────────────────────────────────────
        print(f"  Retrieving on {len(subset_questions)} questions …", end=" ", flush=True)
        t0 = time.perf_counter()
        results, latencies = retrieve_subset(retriever, subset_questions, top_k=100)
        print(f"done ({time.perf_counter() - t0:.1f}s)")

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

        # ── Report key T3 metrics ────────────────────────────────────────────
        umbrela  = metrics.get("T3/umbrela/mean",        float("nan"))
        erag     = metrics.get("T3/erag/mean",            float("nan"))
        ragas_wa = metrics.get("T3/ragas_wa/mean",       float("nan"))
        ndcg_umb = metrics.get("T2-umbrela/P2/NDCG@10",  float("nan"))
        print(
            f"  umbrela={umbrela:.4f}  erag={erag:.4f}  ragas_wa={ragas_wa:.4f}  "
            f"NDCG@10(umb)={ndcg_umb:.4f}  ({elapsed:.0f}s)"
        )

        # ── Merge into result JSON ────────────────────────────────────────────
        new_keys = merge_tier3_into_result(result_path, metrics)
        print(f"  Saved {len(new_keys)} new keys to {result_path.name}")

        # ── Persist per-query sidecar ────────────────────────────────────────
        if harness.last_tier3_result is not None:
            sidecar_path = write_per_query_sidecar(
                SIDECAR_DIR, exp_id, harness.last_tier3_result, contexts_with_ranks
            )
            print(f"  Sidecar: {sidecar_path.relative_to(_ROOT)}")
        print()

    # ── Summary ──────────────────────────────────────────────────────────────
    print("=" * 70)
    print("SUMMARY  (sorted by T3/umbrela/mean)")
    print("=" * 70)
    rows = []
    for f in sorted(RESULTS_DIR.glob("*_zeroshot_test.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        sm = d.get("subset_metrics", {}).get("metrics", {})
        rows.append({
            "experiment": d["experiment_id"].replace("_zeroshot_test", ""),
            "umbrela":   sm.get("T3/umbrela/mean",    float("nan")),
            "erag":      sm.get("T3/erag/mean",        float("nan")),
            "ragas_wa":  sm.get("T3/ragas_wa/mean",   float("nan")),
            "R@10_sub":  sm.get("Recall@10",          float("nan")),
            "R@100_sub": sm.get("Recall@100",         float("nan")),
        })
    rows.sort(key=lambda x: x["umbrela"] if x["umbrela"] == x["umbrela"] else -1,
              reverse=True)
    hdr = (f"{'Experiment':<32} {'UMBL':>6} {'eRAG':>6} {'RGS-A':>6} "
           f"{'R@10':>6} {'R@100':>6}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        def _f(v): return f"{v:.4f}" if v == v else "  —   "
        print(
            f"{r['experiment']:<32} "
            f"{_f(r['umbrela']):>6} {_f(r['erag']):>6} {_f(r['ragas_wa']):>6} "
            f"{_f(r['R@10_sub']):>6} {_f(r['R@100_sub']):>6}"
        )
    print(f"\nResults merged into: {RESULTS_DIR}")
    print(f"Per-query sidecars : {SIDECAR_DIR}")


if __name__ == "__main__":
    main()
