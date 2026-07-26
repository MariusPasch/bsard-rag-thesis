"""
Run all Tier 1 sparse retrieval experiments.

Data split protocol (TIER1_SPARSE_RETRIEVAL_PLAN.md §3):
  - Benchmarking always runs on the TEST split (222 questions).
  - Hyperparameter grid (step 6) always runs on the VAL split (177 questions).
  - The test split is never used during hyperparameter selection.

Execution order:
  1.  Load corpus
  2.  Load TEST questions (all benchmarking below is on test)
  3.  Build / load lexical alignment strata
  4.  ANCHOR [Replication]: BM25 Okapi, none, text_only, k1=1.5, b=0.75
  5.  Normalization ablation [Proposal]: lemmatize, stem
  6.  Field weighting ablation [Proposal]: none+concat_2x, lemmatize+concat_2x
  7.  BM25 hyperparameter grid on VAL split [Proposal] → best k1/b applied to test
  8.  BM25+ and BM25L variants [Proposal]
  9.  FTS5 baseline [Proposal]
  10. TF-IDF ablation [Replication: tfidf_none_text_only; Proposal: others]
  11. Significance tests vs anchor
  12. Summary printed to stdout

Usage (from project root):
    .venv/Scripts/python scripts/evaluation/tier1/run_sparse_experiments.py
    .venv/Scripts/python scripts/evaluation/tier1/run_sparse_experiments.py --skip-tfidf
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# Make the project root importable when running as a script
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from evaluation.runner import add_significance, per_query_recall, run_experiment, save_result
from evaluation.split import load_questions, make_ground_truth
from evaluation.stratify import compute_strata, load_strata, save_strata
from retrieval.sparse import BM25Retriever, FTS5Retriever, TFIDFRetriever

DB_PATH      = Path("output/bsard_corpus.db")
PARQUET_PATH = Path("output/bsard_articles_dedup.parquet")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_articles() -> pd.DataFrame:
    df = pd.read_parquet(
        PARQUET_PATH,
        columns=["article_id", "bsard_id", "law_code", "chapter_title",
                 "article_number", "article_text", "has_cross_references"],
    )
    return df[df["article_text"].notna()].reset_index(drop=True)


def _preprocessing_block(normalization: str, field_weighting: str,
                          embedding_prefix: str = "none") -> dict:
    return {
        "normalization":    normalization,
        "stopword_list":    "spacy_fr_custom",
        "field_weighting":  field_weighting,
        "embedding_prefix": embedding_prefix,
    }


# Results live under the gitignored data root (default <repo>/output; override with BSARD_DATA_DIR)
_RESULTS_DIR = Path(__file__).parent.parent.parent.parent / "output" / "results" / "sparse_retrieval"


def _run(retriever, questions, experiment_id, hparams, preprocessing,
         strata, corpus, anchor_result=None, index_build_time_s: float = 0.0):
    print(f"\n  [test] {experiment_id} ...", end=" ", flush=True)
    result = run_experiment(
        retriever, questions, experiment_id, hparams, preprocessing, strata,
        corpus=corpus,
        index_build_time_s=index_build_time_s,
    )
    if anchor_result is not None:
        add_significance(result, anchor_result)
    path = save_result(result, results_dir=_RESULTS_DIR)
    r10  = result["metrics"].get("Recall@10", float("nan"))
    r100 = result["metrics"].get("Recall@100", float("nan"))
    print(f"R@10={r10:.4f}  R@100={r100:.4f}  -> {path.name}")
    return result


def _cross_ref_map(df: pd.DataFrame) -> dict[int, bool]:
    return dict(zip(df["article_id"], df["has_cross_references"].astype(bool)))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-tfidf", action="store_true",
                        help="Skip TF-IDF experiments (saves time)")
    args = parser.parse_args()

    # Benchmarking is always on the test split (plan §3 — data split rule).
    # Hyperparameter selection (step 7) uses the val split internally.
    print("=" * 70)
    print("TIER 1 — SPARSE RETRIEVAL EXPERIMENTS  (split=test)")
    print("=" * 70)

    # ── 1. Load corpus ───────────────────────────────────────────────────────
    print("\n[1] Loading corpus …")
    df = load_articles()
    print(f"    Articles: {len(df):,}")
    corpus = dict(zip(df["article_id"], df["article_text"].fillna("")))

    # ── 2. Load TEST questions ───────────────────────────────────────────────
    print("\n[2] Loading test questions …")
    questions = load_questions(DB_PATH, subset="test")
    print(f"    Questions: {len(questions)}")

    # ── 3. Build / load strata ───────────────────────────────────────────────
    print("\n[3] Building query strata …")
    strata = load_strata()
    if strata is None or not all(q["question_id"] in strata for q in questions):
        print("    Fitting anchor BM25 for stratification …")
        anchor_for_strata = BM25Retriever(
            df, normalization="none", field_weighting="text_only",
            variant="okapi", k1=1.5, b=0.75,
        )
        all_train_val = (
            load_questions(DB_PATH, "train") + load_questions(DB_PATH, "val")
        )
        cr_map = _cross_ref_map(df)
        strata = compute_strata(
            questions + all_train_val, anchor_for_strata, cr_map
        )
        save_strata(strata)
        print("    Strata saved to evaluation/data/query_strata.json")
    else:
        print("    Loaded from evaluation/data/query_strata.json")

    # ── 4. Anchor experiment [Replication] ───────────────────────────────────
    print("\n[4] Building ANCHOR retriever [Replication: BM25 Okapi] …")
    import time as _time
    t0 = _time.perf_counter()
    anchor_retriever = BM25Retriever(
        df, normalization="none", field_weighting="text_only",
        variant="okapi", k1=1.5, b=0.75,
    )
    anchor_build_s = _time.perf_counter() - t0
    anchor_result = _run(
        anchor_retriever, questions,
        "bm25_anchor_test",
        {"variant": "okapi", "k1": 1.5, "b": 0.75},
        _preprocessing_block("none", "text_only"),
        strata, corpus,
        anchor_result=None,
        index_build_time_s=anchor_build_s,
    )

    # ── 5. Normalization ablation [Proposal] ─────────────────────────────────
    print("\n[5] Normalization ablation …")
    for norm in ["lemmatize", "stem"]:
        t0 = _time.perf_counter()
        r = BM25Retriever(df, normalization=norm, field_weighting="text_only",
                          variant="okapi", k1=1.5, b=0.75)
        build_s = _time.perf_counter() - t0
        _run(r, questions,
             f"bm25_{norm}_text_only_test",
             {"variant": "okapi", "k1": 1.5, "b": 0.75},
             _preprocessing_block(norm, "text_only"),
             strata, corpus, anchor_result, index_build_time_s=build_s)

    # ── 6. Field weighting ablation [Proposal] ───────────────────────────────
    print("\n[6] Field weighting ablation …")
    for norm in ["none", "lemmatize"]:
        t0 = _time.perf_counter()
        r = BM25Retriever(df, normalization=norm, field_weighting="concat_2x",
                          variant="okapi", k1=1.5, b=0.75)
        build_s = _time.perf_counter() - t0
        _run(r, questions,
             f"bm25_{norm}_concat_2x_test",
             {"variant": "okapi", "k1": 1.5, "b": 0.75},
             _preprocessing_block(norm, "concat_2x"),
             strata, corpus, anchor_result, index_build_time_s=build_s)

    # ── 7. Hyperparameter grid on VAL split [Proposal] ───────────────────────
    #      Grid runs on val only; best params are evaluated on test.
    print("\n[7] Hyperparameter grid (k1 × b) on VAL split …")
    val_questions = load_questions(DB_PATH, "val")
    val_gt        = make_ground_truth(val_questions)

    best_k1, best_b, best_val_r10 = 1.5, 0.75, -1.0
    # Grid per TIER1_SPARSE_RETRIEVAL_PLAN.md §4
    grid = [(k1, b)
            for k1 in [0.5, 1.0, 1.5, 2.0]
            for b  in [0.25, 0.50, 0.75, 1.00]]

    for k1, b in tqdm(grid, desc="    k1/b grid"):
        r = BM25Retriever(df, normalization="lemmatize", field_weighting="text_only",
                          variant="okapi", k1=k1, b=b)
        results_val = {}
        for q in val_questions:
            ranked, _ = r.retrieve(q["question_text"])
            results_val[q["question_id"]] = ranked
        scores = per_query_recall(results_val, val_gt, k=10)
        r10_val = float(np.mean(scores))
        if r10_val > best_val_r10:
            best_val_r10, best_k1, best_b = r10_val, k1, b

    print(f"    Best on val: k1={best_k1}, b={best_b}, Recall@10={best_val_r10:.4f}")

    if (best_k1, best_b) != (1.5, 0.75):
        t0 = _time.perf_counter()
        r = BM25Retriever(df, normalization="lemmatize", field_weighting="text_only",
                          variant="okapi", k1=best_k1, b=best_b)
        build_s = _time.perf_counter() - t0
        _run(r, questions,
             f"bm25_tuned_k1{best_k1}_b{best_b}_test",
             {"variant": "okapi", "k1": best_k1, "b": best_b, "tuned_on": "val_split"},
             _preprocessing_block("lemmatize", "text_only"),
             strata, corpus, anchor_result, index_build_time_s=build_s)
    else:
        print("    Default k1/b already optimal — no separate tuned experiment.")

    # ── 8. BM25 variants [Proposal] ──────────────────────────────────────────
    print("\n[8] BM25+ and BM25L variants …")
    for variant in ["plus", "l"]:
        t0 = _time.perf_counter()
        r = BM25Retriever(df, normalization="lemmatize", field_weighting="text_only",
                          variant=variant)
        build_s = _time.perf_counter() - t0
        _run(r, questions,
             f"bm25_{variant}_lemmatize_test",
             {"variant": variant},
             _preprocessing_block("lemmatize", "text_only"),
             strata, corpus, anchor_result, index_build_time_s=build_s)

    # ── 9. FTS5 baseline [Proposal] ──────────────────────────────────────────
    print("\n[9] FTS5 baseline …")
    fts5 = FTS5Retriever(DB_PATH)
    _run(fts5, questions,
         "fts5_default_test",
         {"tokenizer": "unicode61"},
         _preprocessing_block("none", "text_only"),
         strata, corpus, anchor_result)

    # ── 10. TF-IDF ablation [Replication: tfidf_none_text_only; Proposal: rest]
    if not args.skip_tfidf:
        print("\n[10] TF-IDF ablation …")
        for norm, fw in [("none", "text_only"),
                         ("lemmatize", "text_only"),
                         ("lemmatize", "concat_2x")]:
            t0 = _time.perf_counter()
            r = TFIDFRetriever(df, normalization=norm, field_weighting=fw)
            build_s = _time.perf_counter() - t0
            _run(r, questions,
                 f"tfidf_{norm}_{fw}_test",
                 {},
                 _preprocessing_block(norm, fw),
                 strata, corpus, anchor_result, index_build_time_s=build_s)
    else:
        print("\n[10] TF-IDF skipped (--skip-tfidf).")

    # ── Summary (sorted by R@100 — primary metric per plan §3) ───────────────
    print("\n" + "=" * 70)
    print("SUMMARY  (sorted by R@100 — primary metric)")
    print("=" * 70)
    rows = []
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for f in sorted(_RESULTS_DIR.glob("*_test.json")):
        d = json.loads(f.read_text())
        sig = d["significance_vs_anchor"]
        rows.append({
            "experiment": d["experiment_id"].replace("_test", ""),
            "R@10":  d["metrics"].get("Recall@10", float("nan")),
            "R@100": d["metrics"].get("Recall@100", float("nan")),
            "MRR@100": d["metrics"].get("MRR@100", float("nan")),
            "NDCG@10": d["metrics"].get("NDCG@10", float("nan")),
            "MAP@100": d["metrics"].get("MAP@100", float("nan")),
            "lat_ms":  d["latency_ms_mean"],
            "p_val":   sig.get("p_value_recall10"),
            "sig":     sig.get("significant"),
        })

    rows.sort(key=lambda x: x["R@100"], reverse=True)
    header = (f"{'Experiment':<38} {'R@10':>6} {'R@100':>6} "
              f"{'MRR@100':>8} {'NDCG@10':>8} {'MAP@100':>8} "
              f"{'lat_ms':>7} {'p-val':>7} {'sig':>4}")
    print(header)
    print("-" * len(header))
    for row in rows:
        p = f"{row['p_val']:.3f}" if row["p_val"] is not None else "  —  "
        s = "Y" if row["sig"] else ("N" if row["sig"] is False else " ")
        print(
            f"{row['experiment']:<38} "
            f"{row['R@10']:>6.4f} "
            f"{row['R@100']:>6.4f} "
            f"{row['MRR@100']:>8.4f} "
            f"{row['NDCG@10']:>8.4f} "
            f"{row['MAP@100']:>8.4f} "
            f"{row['lat_ms']:>7.1f} "
            f"{p:>7} "
            f"{s:>4}"
        )
    print(f"\nResults saved to: {_RESULTS_DIR}")


if __name__ == "__main__":
    main()
