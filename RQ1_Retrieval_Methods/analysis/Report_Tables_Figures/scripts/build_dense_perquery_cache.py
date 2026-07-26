"""
Regenerate the dense (E5-large + metadata) per-query top-100 ranking cache for
the 222-question BSARD test split.

Why this exists
---------------
The retrieval harness persisted dense *corpus* embeddings (output/embeddings/
*.npy) but not the dense *per-query* rankings for the full test set. The
difficulty-slopes figure (build_headline_difficulty_slopes.py) bins Recall@10 by
gold-article count, which needs per-query rankings. This script reconstructs
them by re-encoding the 222 test queries with the same model and scoring against
the cached corpus embeddings -- exactly what DenseRetriever does at eval time --
and writes:

    <data_root>/cache/dense_me5_large_concat2x_top100_test.json
        { question_id: [article_id, ... top 100 by cosine] }

The reconstruction reproduces the stored aggregates exactly (Recall@10 = 0.342,
single_article = 0.446, multi_article = 0.280); the script asserts this before
writing so a drifted encoder can never silently corrupt the cache.

Requirements: torch + sentence-transformers (declared in requirements.txt) and
the intfloat/multilingual-e5-large model (downloaded on first run, ~2.2 GB).

Usage:
  python analysis/Report_Tables_Figures/scripts/build_dense_perquery_cache.py
"""
from __future__ import annotations

import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPORT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPORT_DIR))

from load_results import RESULTS_DIR, load_records  # noqa: E402

DATA_ROOT = RESULTS_DIR.parent
EMB = DATA_ROOT / "embeddings" / "intfloat_multilingual_e5_large_concat_2x.npy"
EMB_IDS = DATA_ROOT / "embeddings" / "intfloat_multilingual_e5_large_concat_2x_ids.npy"
CORPUS_DB = DATA_ROOT / "bsard_corpus.db"
OUT = DATA_ROOT / "cache" / "dense_me5_large_concat2x_top100_test.json"

MODEL_NAME = "intfloat/multilingual-e5-large"
QUERY_PREFIX = "query: "          # E5 retrieval-query prefix (matches the harness)
DENSE_RESULT_ID = "dense_me5_large_concat2x_zeroshot_test"


def _gold_and_targets() -> tuple[dict[str, set[str]], dict[str, float]]:
    """Test qrels (per-query gold) + the stored aggregates to verify against."""
    recs = {rec.experiment_id: rec for _t, rec in load_records()}
    qrels = None
    for key in ("hybrid_rrf_k60_test", "crag_hybrid_rrf_k60_test_v2"):
        if key in recs and recs[key].payload.get("_trec_qrels"):
            qrels = {str(q): {str(d) for d in g}
                     for q, g in recs[key].payload["_trec_qrels"].items()}
            break
    if qrels is None:
        raise SystemExit("No _trec_qrels available to define gold / verify.")
    dr = recs[DENSE_RESULT_ID]
    targets = {
        "overall": dr.metrics["Recall@10"],
        "single": dr.stratified["single_article"]["Recall@10"],
        "multi": dr.stratified["multi_article"]["Recall@10"],
    }
    return qrels, targets


def _verify(rank100: dict[str, list[str]], qrels: dict[str, set[str]],
            targets: dict[str, float]) -> None:
    overall, single, multi = [], [], []
    for q, gold in qrels.items():
        ng = len(gold)
        if not ng:
            continue
        r = len(set(rank100[q][:10]) & gold) / ng
        overall.append(r)
        (single if ng == 1 else multi).append(r)
    got = {"overall": sum(overall) / len(overall),
           "single": sum(single) / len(single),
           "multi": sum(multi) / len(multi)}
    for k, v in got.items():
        if round(v, 3) != round(targets[k], 3):
            raise SystemExit(
                f"Dense reconstruction mismatch on {k}: got {v:.3f}, "
                f"expected {targets[k]:.3f}. Refusing to write the cache.")
    print(f"[verify] overall={got['overall']:.3f} single={got['single']:.3f} "
          f"multi={got['multi']:.3f} — matches stored aggregates.")


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    for p in (EMB, EMB_IDS, CORPUS_DB):
        if not p.exists():
            raise SystemExit(f"Required input missing: {p}")

    emb = np.load(EMB).astype(np.float32)            # (N, d), unit-normalised
    id_list = [str(int(x)) for x in np.load(EMB_IDS)]

    qrels, targets = _gold_and_targets()
    con = sqlite3.connect(str(CORPUS_DB))
    qtext = {str(qid): txt for qid, txt in
             con.execute("SELECT question_id, question_text FROM questions WHERE split='test'")}
    con.close()
    qids = [q for q in qrels if q in qtext]
    texts = [QUERY_PREFIX + qtext[q] for q in qids]
    print(f"Encoding {len(texts)} test queries with {MODEL_NAME} "
          "(first run downloads ~2.2 GB)…")

    from sentence_transformers import SentenceTransformer  # heavy import, deferred
    model = SentenceTransformer(MODEL_NAME)
    qvec = model.encode(texts, normalize_embeddings=True, batch_size=16,
                        show_progress_bar=False).astype(np.float32)

    scores = qvec @ emb.T                            # cosine (both unit-norm)
    top = np.argpartition(-scores, 100, axis=1)[:, :100]
    rank100: dict[str, list[str]] = {}
    for r, q in enumerate(qids):
        idx = top[r][np.argsort(-scores[r, top[r]])]
        rank100[q] = [id_list[i] for i in idx]

    _verify(rank100, qrels, targets)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({q: [int(d) for d in rank100[q]] for q in qids}),
                   encoding="utf-8")
    print(f"[done] Wrote {OUT}")


if __name__ == "__main__":
    main()
