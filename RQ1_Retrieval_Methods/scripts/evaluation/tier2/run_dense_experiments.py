"""
Run all Tier 2 dense retrieval experiments.

Corpus source: HuggingFace `maastrichtlawtech/bsard` (canonical).
  Cached locally at output/bsard_hf_articles.parquet after first download.
  Columns after normalisation: article_id, article_text, law_code, article_title.

Split strategy:
  - All models are evaluated on the TEST split (222 questions) only.
  - The concat_2x field-weighting ablation winner (EXP-D7) is selected using
    a 100-question train sample (seed=42, persisted in evaluation/data/split_ids.json).
    This keeps the test split completely untouched during any selection step.

Execution order:
  1.  Load corpus (HuggingFace BSARD) and validate size.
  2.  Load test questions (222), train sample (100), and strata.
  3.  Token length audit for all candidate models.
  4.  Build BM25-tuned anchor on train sample for significance testing.
  5.  EXP-D1 : camembert-base              zero-shot  (paper benchmark anchor, ~4% R@100)
  6.  EXP-D2 : sentence-camembert-large    zero-shot  (French-specific)
  7.  EXP-D3 : multilingual-e5-base        zero-shot
  8.  EXP-D4 : multilingual-e5-large       zero-shot
  9.  EXP-D4c: BAAI/bge-m3                 zero-shot  (long-context, CLS pooling, max_seq=1024)
  10. EXP-D5 : paper checkpoint            pretrained (if --paper-checkpoint-path supplied)
  11. EXP-D6 : paraphrase-mpnet-base-v2    zero-shot  (128-token reference only)
  12. EXP-D10 : Qwen3-Embedding-0.6B       zero-shot  (decoder-only LLM, last-token pooling)
  13. EXP-D10i: Qwen3-Embedding-0.6B       zero-shot + instruction prefix (reuses D10 corpus emb)
  14. EXP-D7 : concat_2x ablation on winner selected from train sample; evaluated on test
  15. Summary table.

Usage (from project root):
    .venv/Scripts/python scripts/evaluation/tier2/run_dense_experiments.py
    .venv/Scripts/python scripts/evaluation/tier2/run_dense_experiments.py --device cuda
    .venv/Scripts/python scripts/evaluation/tier2/run_dense_experiments.py \\
        --paper-checkpoint-path output/models/bsard_checkpoint
    .venv/Scripts/python scripts/evaluation/tier2/run_dense_experiments.py --skip-audit
    .venv/Scripts/python scripts/evaluation/tier2/run_dense_experiments.py --skip-qwen3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from evaluation.runner   import add_significance, run_experiment, save_result
from evaluation.split    import load_questions, load_train_sample
from evaluation.stratify import load_strata
from retrieval.dense     import DenseRetriever, EmbeddingEncoder

DB_PATH          = Path("output/bsard_corpus.db")
DEDUP_PARQUET    = Path("output/bsard_articles_dedup.parquet")
HF_PARQUET_CACHE = Path("output/bsard_hf_articles.parquet")  # kept for reference; NOT used
RESULTS_DIR      = Path("output/results/dense_retrieval")

# ── Model registry ────────────────────────────────────────────────────────────
# (experiment_id_stem, hf_model_id, query_prefix, passage_prefix, note)
_ME5_INSTRUCT_PREFIX = (
    "Instruct: Given a legal question in French, retrieve the relevant "
    "Belgian statutory article\nQuery: "
)

MODELS = [
    ("dense_camembert_base",
     "camembert-base",
     "", "",
     "Paper benchmark anchor — expected R@100 ~4%; elevated to full paper metric replication"),
    ("dense_camembert_lg",
     "dangvantuan/sentence-camembert-large",
     "", "",
     "French-specific zero-shot candidate"),
    ("dense_me5_base",
     "intfloat/multilingual-e5-base",
     "query: ", "passage: ",
     "Multilingual E5 baseline"),
    ("dense_me5_large",
     "intfloat/multilingual-e5-large",
     "query: ", "passage: ",
     "Multilingual E5 large"),
    ("dense_bge_m3",
     "BAAI/bge-m3",
     "", "",
     "Long-context (max_seq=1024); CLS pooling; ~0% truncation on BSARD"),
    ("dense_mpnet_multi",
     "paraphrase-multilingual-mpnet-base-v2",
     "", "",
     "128-token reference only — NOT for model selection"),
    ("dense_qwen3_0_6b",
     "Qwen/Qwen3-Embedding-0.6B",
     "", "",
     "EXP-D10: decoder-only LLM embedding (0.6B params); last-token pooling; 32k ctx; truncate_dim=1024"),
    ("dense_qwen3_0_6b_instruct",
     "Qwen/Qwen3-Embedding-0.6B",
     _ME5_INSTRUCT_PREFIX, "",
     "EXP-D10i: Qwen3-0.6B with legal instruction prefix — reuses D10 corpus embeddings"),
]

# Model-specific constructor overrides
_MODEL_OVERRIDES: dict[str, dict] = {
    # CamemBERT-large: position IDs start at 2, so effective max is 512 not 514
    "dense_camembert_lg":         {"max_seq_length_override": 512},
    "dense_bge_m3":               {"max_seq_length_override": 1024, "batch_size": 16},
    # Qwen3: decoder-only LLM; last-token pooling handled by sentence-transformers;
    # truncate_dim=1024 for Matryoshka truncation; max_seq_length=512 matches other
    # models and avoids OOM from 32k padding on T4 (16 GB); batch_size=4 for GPU
    "dense_qwen3_0_6b":           {"batch_size": 4, "truncate_dim": 1024, "max_seq_length_override": 512},
    "dense_qwen3_0_6b_instruct":  {"batch_size": 4, "truncate_dim": 1024, "max_seq_length_override": 512},
}

# Excluded from Tier 3 model selection (reference-only)
# Excluded from EXP-D7 concat_2x winner selection.
# Qwen3 models are excluded because they use a decoder-only LLM backbone with
# last-token pooling — a fundamentally different architecture from the encoder-only
# candidates. The concat_2x ablation is scoped to encoder-only bi-encoders so that
# field-weighting effects are not confounded with architecture differences.
EXCLUDED_FROM_SELECTION = {
    "dense_camembert_base",       # paper anchor only; not for Tier 3
    "dense_mpnet_multi",          # 128-token reference only
    "dense_qwen3_0_6b",           # decoder-only LLM — different architecture
    "dense_qwen3_0_6b_instruct",  # decoder-only LLM — different architecture
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_articles() -> pd.DataFrame:
    """
    Load the BSARD corpus from the local deduplicated parquet.

    Source: output/bsard_articles_dedup.parquet (22,633 rows).
    Columns include: article_id, article_text, law_code, chapter_title,
    article_number, bsard_id, and many more.

    IMPORTANT: This file uses the local article_id scheme which matches the
    ground-truth IDs in the SQLite DB (evaluation/split.py).  The HuggingFace
    parquet (bsard_hf_articles.parquet) uses a DIFFERENT ID scheme (bsard_id)
    and must NOT be used directly — the IDs don't match the ground truth,
    causing near-zero recall.  See scripts/setup/build_hf_id_mapping.py for
    the mapping between the two ID schemes.
    """
    if not DEDUP_PARQUET.exists():
        print(f"    ERROR: {DEDUP_PARQUET} not found.")
        print(f"    Run scripts/setup/prepare_dedup_corpus.py first,")
        print(f"    or ensure the output/ junction is set up (see README.md).")
        sys.exit(1)

    df = pd.read_parquet(DEDUP_PARQUET)
    print(f"    Loaded {len(df):,} articles from {DEDUP_PARQUET}")
    return df[df["article_text"].notna()].reset_index(drop=True)


def check_corpus_size(df: pd.DataFrame) -> None:
    n = len(df)
    _EXPECTED = 22_633  # local dedup count; HF canonical corpus may differ slightly
    if n < 20_000 or n > 30_000:
        print(f"\n  !! UNEXPECTED CORPUS SIZE: {n:,} rows (expected ~{_EXPECTED:,}).")
        print(f"     Verify the HuggingFace BSARD corpus loaded correctly.\n")
        sys.exit(1)
    if n != _EXPECTED:
        print(f"    Corpus: {n:,} articles (HuggingFace BSARD). "
              f"Note: differs from local dedup ({_EXPECTED:,}).")
    else:
        print(f"    Corpus: {n:,} articles (HuggingFace BSARD). OK")


def _preprocessing_block(query_prefix: str, passage_prefix: str,
                          field_weighting: str) -> dict:
    prefix_str = f"{query_prefix}|{passage_prefix}" if (query_prefix or passage_prefix) else "none"
    return {
        "normalization":    "none",
        "stopword_list":    "none",
        "field_weighting":  field_weighting,
        "embedding_prefix": prefix_str,
    }


def _hparams(model_name: str, retriever: DenseRetriever,
             batch_size: int, device: str,
             truncate_dim: int | None = None) -> dict:
    h = {
        "model_name":           model_name,
        "max_seq_length":       retriever._encoder.max_seq_length,
        "embedding_dim":        retriever.embedding_dim,
        "normalize_embeddings": True,
        "batch_size":           batch_size,
        "device":               device,
    }
    if truncate_dim is not None:
        h["truncate_dim"] = truncate_dim
    return h


def _run(retriever, questions, experiment_id, hparams, preprocessing,
         strata, anchor_result=None, split_label="val",
         training_regime="zero_shot"):
    # Tier 0: record index build time from retriever.
    # "encode" = full corpus encoding (authoritative).
    # "disk"   = np.load + FAISS rebuild (fast path, still representative).
    # "cache"  = in-process cache hit (near-zero; record 0.0 so it's not misleading).
    build_time   = getattr(retriever, "_index_build_time_s", 0.0)
    build_source = getattr(retriever, "_index_build_source", "unknown")
    if build_source == "cache":
        build_time = 0.0  # suppress: only the first run's encoding time is meaningful

    print(f"\n  [{split_label}] {experiment_id} ...", end=" ", flush=True)
    result = run_experiment(
        retriever, questions, experiment_id, hparams, preprocessing,
        strata, top_k=500, index_build_time_s=build_time,
    )
    result["training_regime"] = training_regime
    result["index_build_source"] = build_source  # document provenance in JSON

    if anchor_result is not None:
        add_significance(result, anchor_result, k_values=[10, 100], primary_k=100)

    path = save_result(result, results_dir=RESULTS_DIR)
    r10  = result["metrics"]["Recall@10"]
    r100 = result["metrics"]["Recall@100"]
    mrr  = result["metrics"]["MRR@10"]
    build_note = f"  [build={build_time:.1f}s/{build_source}]" if build_time > 0.1 else ""
    print(f"R@10={r10:.4f}  R@100={r100:.4f}  MRR@10={mrr:.4f}  -> {path.name}{build_note}")
    return result


def load_anchor_result(df: pd.DataFrame, questions: list[dict],
                       strata: dict) -> dict | None:
    """
    Re-run the bm25_tuned anchor retriever to obtain _raw_results/_raw_gt for
    significance testing. (Saved JSONs strip those keys.)
    """
    try:
        from retrieval.sparse import BM25Retriever
        print("    Re-running bm25_tuned anchor for significance testing ...")
        anchor = BM25Retriever(
            df, normalization="lemmatize", field_weighting="text_only",
            variant="okapi", k1=1.5, b=0.5,
        )
        result = run_experiment(
            anchor, questions, "_anchor_for_dense_sig",
            {"variant": "okapi", "k1": 1.5, "b": 0.5},
            {"normalization": "lemmatize", "stopword_list": "spacy_fr_custom",
             "field_weighting": "text_only", "embedding_prefix": "none"},
            strata, top_k=500,
        )
        return result
    except Exception as e:
        print(f"    WARNING: could not build anchor — significance tests skipped. ({e})")
        return None


def print_token_audit_table(df: pd.DataFrame, device: str,
                            skip_qwen3: bool = False) -> None:
    texts = df["article_text"].fillna("").tolist()
    print(f"\n  {'Model':<55} {'max_seq':>7} {'mean':>6} {'p90':>5} {'trunc%':>7}")
    print("  " + "-" * 84)
    _QWEN3 = {"dense_qwen3_0_6b", "dense_qwen3_0_6b_instruct"}
    for stem, model_name, _qp, _pp, _note in MODELS:
        if skip_qwen3 and stem in _QWEN3:
            continue
        overrides = _MODEL_OVERRIDES.get(stem, {})
        try:
            enc   = EmbeddingEncoder(
                model_name, device=device,
                max_seq_length_override=overrides.get("max_seq_length_override"),
            )
            audit = enc.token_length_audit(texts)
            flag  = "  !! HIGH" if audit["fraction_truncated"] > 0.20 else ""
            print(
                f"  {model_name:<55} "
                f"{enc.max_seq_length:>7} "
                f"{audit['mean_tokens']:>6.0f} "
                f"{audit['p90_tokens']:>5.0f} "
                f"{audit['fraction_truncated']:>7.1%}"
                f"{flag}"
            )
        except Exception as e:
            print(f"  {model_name:<55}  ERROR: {e}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--paper-checkpoint-path", default=None,
                        help="Local path or HF ID for authors' released BSARD checkpoint")
    parser.add_argument("--skip-audit", action="store_true",
                        help="Skip token length audit (~5 min, loads all model tokenizers)")
    parser.add_argument("--skip-qwen3", action="store_true",
                        help="Skip Qwen3-Embedding-0.6B experiments (D10/D10i) — saves ~30-60 min on CPU")
    parser.add_argument("--only-qwen3", action="store_true",
                        help="Run only D10/D10i (Qwen3) — skips all other models and D7")
    parser.add_argument("--skip-done", action="store_true",
                        help="Skip any experiment whose result JSON already exists in RESULTS_DIR")
    parser.add_argument("--skip-d7", action="store_true",
                        help="Skip EXP-D7 concat_2x winner ablation (useful when selecting winner manually)")
    args = parser.parse_args()

    print("=" * 70)
    print("TIER 2 -- DENSE RETRIEVAL EXPERIMENTS  (test split, 222 questions)")
    print("=" * 70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Corpus ─────────────────────────────────────────────────────────────
    print("\n[1] Loading corpus ...")
    df = load_articles()
    check_corpus_size(df)

    # ── 2. Questions and strata ───────────────────────────────────────────────
    print("\n[2] Loading questions ...")
    test_questions = load_questions(DB_PATH, subset="test")
    train_sample   = load_train_sample(DB_PATH)   # 100-question seed-42 sample for D7 selection
    print(f"    Test questions  : {len(test_questions)} (evaluation)")
    print(f"    Train sample    : {len(train_sample)} (EXP-D7 winner selection only)")

    print("\n[3] Loading query strata ...")
    strata = load_strata()
    if strata is None:
        print("    ERROR: query_strata.json not found. Run Tier 1 first.")
        sys.exit(1)
    print(f"    Strata loaded for {len(strata):,} questions.")

    # ── 3. Token audit ────────────────────────────────────────────────────────
    if not args.skip_audit:
        print("\n[AUDIT] Token length audit for all candidate models:")
        print_token_audit_table(df, args.device, skip_qwen3=args.skip_qwen3)
    else:
        print("\n[AUDIT] Skipped (--skip-audit).")

    # ── 4. Anchor for significance testing (built on test questions) ──────────
    print("\n[4] Building significance anchor (bm25_tuned k1=1.5 b=0.5) ...")
    anchor_result = load_anchor_result(df, test_questions, strata)

    # ── 5. Zero-shot experiments on TEST split ────────────────────────────────
    all_results: dict[str, dict] = {}

    _QWEN3_STEMS = {"dense_qwen3_0_6b", "dense_qwen3_0_6b_instruct"}

    for stem, model_name, qp, pp, note in MODELS:
        if args.skip_qwen3 and stem in _QWEN3_STEMS:
            print(f"\n[EXP] {stem}_zeroshot_test — SKIPPED (--skip-qwen3)")
            continue
        if args.only_qwen3 and stem not in _QWEN3_STEMS:
            print(f"\n[EXP] {stem}_zeroshot_test — SKIPPED (--only-qwen3)")
            continue

        exp_id       = f"{stem}_zeroshot_test"
        if args.skip_done and (RESULTS_DIR / f"{exp_id}.json").exists():
            print(f"\n[EXP] {exp_id} — SKIPPED (result already exists)")
            continue
        overrides    = _MODEL_OVERRIDES.get(stem, {})
        batch_sz     = overrides.get("batch_size", args.batch_size)
        truncate_dim = overrides.get("truncate_dim")

        print(f"\n[EXP] {exp_id}")
        print(f"      Model : {model_name}")
        print(f"      Note  : {note}")

        retriever = DenseRetriever(
            df=df,
            model_name=model_name,
            field_weighting="text_only",
            passage_prefix=pp,
            query_prefix=qp,
            batch_size=batch_sz,
            device=args.device,
            max_seq_length_override=overrides.get("max_seq_length_override"),
            truncate_dim=truncate_dim,
        )
        result = _run(
            retriever, test_questions, exp_id,
            _hparams(model_name, retriever, batch_sz, args.device, truncate_dim),
            _preprocessing_block(qp, pp, "text_only"),
            strata, anchor_result, "test",
        )
        all_results[stem] = result

    # ── 6. Paper checkpoint (EXP-D5) ─────────────────────────────────────────
    if args.paper_checkpoint_path:
        cp = args.paper_checkpoint_path
        exp_id = "dense_paper_checkpoint_pretrained_test"
        print(f"\n[EXP] {exp_id}")
        print(f"      Checkpoint: {cp}")
        retriever = DenseRetriever(
            df=df, model_name=cp, field_weighting="text_only",
            batch_size=args.batch_size, device=args.device,
        )
        result = _run(
            retriever, test_questions, exp_id,
            _hparams(cp, retriever, args.batch_size, args.device),
            _preprocessing_block("", "", "text_only"),
            strata, anchor_result, "test",
            training_regime="pretrained_bsard",
        )
        all_results["dense_paper_checkpoint"] = result
    else:
        print("\n[EXP-D5] --paper-checkpoint-path not supplied. Skipping.")

    # ── 7. concat_2x ablation — winner selected on train sample ──────────────
    # Winner is determined by running all eligible models over the 100-question
    # train sample (never the test split). The winning model's concat_2x variant
    # is then evaluated on the full 222-question test split.
    eligible_stems = [
        stem for stem, *_ in MODELS
        if stem not in EXCLUDED_FROM_SELECTION
        and stem != "dense_paper_checkpoint"
    ]

    if args.skip_d7 or args.only_qwen3:
        print("\n[EXP-D7] Skipped (--skip-d7 / --only-qwen3).")
    elif eligible_stems:
        print("\n[EXP-D7] Selecting concat_2x winner on train sample "
              f"({len(train_sample)} questions) ...")
        # Build a temporary anchor on the train sample for the selection run
        train_anchor = load_anchor_result(df, train_sample, strata)
        selection_scores: dict[str, float] = {}
        for stem, model_name, qp, pp, _ in MODELS:
            if stem not in eligible_stems:
                continue
            overrides    = _MODEL_OVERRIDES.get(stem, {})
            batch_sz     = overrides.get("batch_size", args.batch_size)
            truncate_dim = overrides.get("truncate_dim")
            # Corpus embeddings already cached from the test-split run above
            retriever = DenseRetriever(
                df=df, model_name=model_name, field_weighting="text_only",
                passage_prefix=pp, query_prefix=qp,
                batch_size=batch_sz, device=args.device,
                max_seq_length_override=overrides.get("max_seq_length_override"),
                truncate_dim=truncate_dim,
            )
            sel_result = _run(
                retriever, train_sample, f"_sel_{stem}",
                _hparams(model_name, retriever, batch_sz, args.device, truncate_dim),
                _preprocessing_block(qp, pp, "text_only"),
                strata, train_anchor, "train_sample",
            )
            selection_scores[stem] = sel_result["metrics"]["Recall@100"]
            print(f"      {stem}: R@100={selection_scores[stem]:.4f} (train sample)")

        best_stem = max(selection_scores, key=selection_scores.__getitem__)
        print(f"\n      Winner: {best_stem} (R@100={selection_scores[best_stem]:.4f} on train sample)")
        print(f"      Running concat_2x on TEST split (222 questions) ...")

        _, best_model_name, best_qp, best_pp, _ = next(
            m for m in MODELS if m[0] == best_stem
        )
        overrides         = _MODEL_OVERRIDES.get(best_stem, {})
        batch_sz          = overrides.get("batch_size", args.batch_size)
        best_truncate_dim = overrides.get("truncate_dim")

        retriever = DenseRetriever(
            df=df, model_name=best_model_name, field_weighting="concat_2x",
            passage_prefix=best_pp, query_prefix=best_qp,
            batch_size=batch_sz, device=args.device,
            max_seq_length_override=overrides.get("max_seq_length_override"),
            truncate_dim=best_truncate_dim,
        )
        exp_id = f"{best_stem}_concat2x_zeroshot_test"
        _run(
            retriever, test_questions, exp_id,
            _hparams(best_model_name, retriever, batch_sz, args.device, best_truncate_dim),
            _preprocessing_block(best_qp, best_pp, "concat_2x"),
            strata, anchor_result, "test",
        )
    else:
        print("\n[EXP-D7] No eligible model for ablation.")

    # ── 8. Summary table (test results only) ──────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY  (test split, 222 questions)")
    print("=" * 70)

    result_files = sorted(RESULTS_DIR.glob("*_test.json"))
    rows = []
    for f in result_files:
        d   = json.loads(f.read_text())
        sig = d.get("significance_vs_anchor", {})
        aud = d.get("token_length_audit", {})
        rows.append({
            "experiment": d["experiment_id"].replace("_test", ""),
            "R@10":       d["metrics"].get("Recall@10",  0.0),
            "R@100":      d["metrics"].get("Recall@100", 0.0),
            "R@200":      d["metrics"].get("Recall@200", 0.0),
            "R@500":      d["metrics"].get("Recall@500", 0.0),
            "MRR@10":     d["metrics"].get("MRR@10",     0.0),
            "MAP@100":    d["metrics"].get("MAP@100",    d["metrics"].get("MAP", 0.0)),
            "lat_ms":     d.get("latency_ms_mean", 0.0),
            "trunc%":     aud.get("fraction_truncated", 0.0),
            "p_val":      sig.get("p_value_recall100", sig.get("p_value_recall10")),
            "sig":        sig.get("significant"),
        })

    rows.sort(key=lambda x: x["R@100"], reverse=True)

    hdr = (f"{'Experiment':<42} {'R@10':>6} {'R@100':>6} {'R@200':>6} {'R@500':>6} "
           f"{'MRR@10':>7} {'MAP@100':>8} {'lat_ms':>7} {'trunc%':>7} {'p-val':>7} {'sig':>4}")
    print(hdr)
    print("-" * len(hdr))
    for row in rows:
        p = f"{row['p_val']:.3f}" if row["p_val"] is not None else "  ---"
        s = "Y" if row["sig"] else ("N" if row["sig"] is False else " ")
        print(
            f"{row['experiment']:<42} "
            f"{row['R@10']:>6.4f} "
            f"{row['R@100']:>6.4f} "
            f"{row['R@200']:>6.4f} "
            f"{row['R@500']:>6.4f} "
            f"{row['MRR@10']:>7.4f} "
            f"{row['MAP@100']:>8.4f} "
            f"{row['lat_ms']:>7.1f} "
            f"{row['trunc%']:>7.1%} "
            f"{p:>7} "
            f"{s:>4}"
        )

    if rows:
        best = rows[0]["experiment"]
        print(f"\nBest model by R@100 (test): {best}")
        eligible_rows = [r for r in rows if not any(
            ex in r["experiment"] for ex in EXCLUDED_FROM_SELECTION
        )]
        if eligible_rows:
            best_eligible = eligible_rows[0]["experiment"]
            print(f"Best eligible for Tier 3  : {best_eligible}")

    print(f"\nResults saved to: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
