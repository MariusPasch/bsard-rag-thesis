"""
Select and lock few-shot examples for LLM-as-a-Judge prompts.

Output: evaluation/agentic/fewshot_examples.json

Protocol
--------
1. Load train split (80% of BSARD train, seed 42).
2. Retrieve top-5 docs per train question using the best Tier 3 first stage
   (hybrid RRF k=60: bm25_lemmatize_concat2x + me5_large_instruct_concat2x).
3. Identify candidates:
   - RELEVANT pair : GT article ranked in top-5 (use highest-ranked GT article text).
   - IRRELEVANT pair: no GT article in top-5 (use top-1 retrieved article text).
4. Stratified selection: 1 single-article + 1 multi-article question.
5. Write and commit fewshot_examples.json.

DO NOT regenerate after val experiments begin — data leakage risk.

Usage
-----
  python scripts/agentic/select_fewshot_examples.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from evaluation.split import load_questions
from evaluation.stratify import load_strata
from retrieval.dense import DenseRetriever
from retrieval.sparse import BM25Retriever
from retrieval.hybrid import HybridRetriever

# ---------------------------------------------------------------------------
# Constants — must match best T3 first-stage (hybrid_rrf_k60)
# ---------------------------------------------------------------------------

CORPUS_PATH    = Path("output/bsard_articles_dedup.parquet")
EMBEDDINGS_DIR = Path("output/embeddings")

DENSE_MODEL  = "intfloat/multilingual-e5-large-instruct"
QUERY_PREFIX = (
    "Instruct: Given a legal question in French, "
    "retrieve the relevant Belgian statutory article\nQuery: "
)
SPARSE_CONFIG = {
    "variant":         "okapi",
    "normalization":   "lemmatize",
    "field_weighting": "concat_2x",
    "k1": 1.5,
    "b":  0.75,
}
RRF_K        = 60
FIRST_STAGE_K = 100   # retrieve top-100; we only look at top-5 for selection

OUTPUT_PATH  = Path("evaluation/data/fewshot_examples.json")
SEED         = 42
MAX_ARTICLE_TOKENS = 1000   # word-level truncation (matches LLMJudgeReranker default)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _truncate(text: str, max_tokens: int) -> str:
    words = text.split()
    if len(words) <= max_tokens:
        return text
    return " ".join(words[:max_tokens])


def _build_retriever(corpus: pd.DataFrame) -> HybridRetriever:
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
    return HybridRetriever(
        sparse_retriever=sparse,
        dense_retriever=dense,
        fusion_method="rrf",
        rrf_k=RRF_K,
        first_stage_k=FIRST_STAGE_K,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if OUTPUT_PATH.exists():
        print(f"[select_fewshot] {OUTPUT_PATH} already exists — skipping regeneration.")
        print("  Delete the file manually if you need to regenerate (before val experiments).")
        return

    print("[select_fewshot] Loading corpus …")
    corpus = pd.read_parquet(CORPUS_PATH)
    article_text: dict[int, str] = dict(zip(corpus["article_id"], corpus["article_text"].fillna("")))

    print("[select_fewshot] Loading train split …")
    train_questions = load_questions(subset="train")
    print(f"  {len(train_questions)} train questions")

    print("[select_fewshot] Building hybrid RRF k=60 retriever …")
    retriever = _build_retriever(corpus)

    print("[select_fewshot] Retrieving top-5 per train question …")
    rng = random.Random(SEED)

    # Categorise questions
    relevant_single   = []  # (question, gt_article_id, gt_article_text)
    relevant_multi    = []
    irrelevant_single = []  # (question, top1_article_id, top1_article_text)
    irrelevant_multi  = []

    strata = load_strata()  # may be None

    for i, q in enumerate(train_questions):
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(train_questions)}] …")

        ranked, _ = retriever.retrieve(q["question_text"], top_k=5)
        gt_ids    = set(q["relevant_article_ids"])
        is_single = q["n_relevant_articles"] == 1

        # Check for RELEVANT: any GT article in top-5
        hit = next((aid for aid in ranked if aid in gt_ids), None)
        if hit is not None:
            text = _truncate(article_text.get(hit, ""), MAX_ARTICLE_TOKENS)
            if not text:
                continue
            entry = (q, hit, text)
            if is_single:
                relevant_single.append(entry)
            else:
                relevant_multi.append(entry)
        else:
            # IRRELEVANT: top-1 article is not a GT article
            if not ranked:
                continue
            top1 = ranked[0]
            text = _truncate(article_text.get(top1, ""), MAX_ARTICLE_TOKENS)
            if not text:
                continue
            entry = (q, top1, text)
            if is_single:
                irrelevant_single.append(entry)
            else:
                irrelevant_multi.append(entry)

    print(
        f"  relevant_single={len(relevant_single)}, relevant_multi={len(relevant_multi)}, "
        f"irrelevant_single={len(irrelevant_single)}, irrelevant_multi={len(irrelevant_multi)}"
    )

    # Stratified selection: pick 1 relevant + 1 irrelevant from different strata
    # Prefer: relevant=single_article, irrelevant=multi_article
    # Fallback: any available
    rel_pool  = relevant_single   if relevant_single   else relevant_multi
    irr_pool  = irrelevant_multi  if irrelevant_multi  else irrelevant_single

    if not rel_pool:
        raise RuntimeError("No RELEVANT candidates found in train split. Check retriever setup.")
    if not irr_pool:
        raise RuntimeError("No IRRELEVANT candidates found in train split. Check retriever setup.")

    rng.shuffle(rel_pool)
    rng.shuffle(irr_pool)

    rel_q,  rel_aid,  rel_text  = rel_pool[0]
    irr_q,  irr_aid,  irr_text  = irr_pool[0]

    def _strata_label(q: dict) -> str:
        return "single_article" if q["n_relevant_articles"] == 1 else "multi_article"

    examples = [
        {
            "role":                   "relevant",
            "question":               rel_q["question_text"],
            "article_text_truncated": rel_text,
            "article_id":             int(rel_aid),
            "strata":                 _strata_label(rel_q),
            "binary_label":           "Oui",
            "numeric_score":          9,
        },
        {
            "role":                   "irrelevant",
            "question":               irr_q["question_text"],
            "article_text_truncated": irr_text,
            "article_id":             int(irr_aid),
            "strata":                 _strata_label(irr_q),
            "binary_label":           "Non",
            "numeric_score":          1,
        },
    ]

    payload = {
        "generated_at":       "2026-03-27",
        "split":              "train",
        "seed":               SEED,
        "first_stage":        f"hybrid_rrf_k{RRF_K}",
        "max_article_tokens": MAX_ARTICLE_TOKENS,
        "examples":           examples,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[select_fewshot] Wrote {OUTPUT_PATH}")
    print("  RELEVANT  example:")
    print(f"    question_id={rel_q['question_id']}  article_id={rel_aid}  strata={_strata_label(rel_q)}")
    print(f"    question : {rel_q['question_text'][:100]}")
    print(f"    passage  : {rel_text[:120]} …")
    print("  IRRELEVANT example:")
    print(f"    question_id={irr_q['question_id']}  article_id={irr_aid}  strata={_strata_label(irr_q)}")
    print(f"    question : {irr_q['question_text'][:100]}")
    print(f"    passage  : {irr_text[:120]} …")
    print("\n  Next: commit this file to git immediately before running any val experiments.")


if __name__ == "__main__":
    main()
