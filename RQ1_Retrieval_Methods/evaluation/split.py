"""
Train / Validation / Test split management.

Decisions implemented:
  E-D2  Test  = BSARD official test split (222 questions) — canonical eval set
               All final metrics and RQ3 autonomous evaluation run on this split.
        Val   = 20% of BSARD train questions, seed 42 (legacy; no longer used
               for hyperparameter selection — kept for reference only)
        Train = remaining 80% (886 questions)

  E-D3  Hyperparameter selection uses a persisted random sample of train questions
        (load_train_sample, n=100, seed=42). Stored in split_ids.json under
        'train_sample_100' so every session uses the identical 100-question set.

The split is created once and persisted to evaluation/data/split_ids.json.
"""

from __future__ import annotations

import json
import random
import sqlite3
from pathlib import Path

from evaluation.paths import CORPUS_DB

SPLIT_FILE          = Path(__file__).parent / "data" / "split_ids.json"
DEFAULT_DB          = CORPUS_DB
SEED                = 42
VAL_FRAC            = 0.20
TRAIN_SAMPLE_N      = 100   # questions sampled from train for hyperparameter selection


def _load_questions(db_path: Path, split: str) -> list[dict]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT question_id, question_text, relevant_article_ids, "
        "n_relevant_articles "
        "FROM questions WHERE split = ?",
        (split,),
    ).fetchall()
    conn.close()

    questions = []
    for r in rows:
        q = dict(r)
        q["relevant_article_ids"] = json.loads(q["relevant_article_ids"])
        questions.append(q)
    return questions


def create_splits(db_path: Path = DEFAULT_DB) -> None:
    """
    Sample the validation split and train sample from BSARD train questions.
    Persists to split_ids.json. Idempotent — adds missing keys if file exists.
    """
    train_qs = _load_questions(db_path, "train")
    rng = random.Random(SEED)

    needs_write = False

    if SPLIT_FILE.exists():
        payload = json.loads(SPLIT_FILE.read_text())
    else:
        rng_copy = random.Random(SEED)
        shuffled = list(train_qs)
        rng_copy.shuffle(shuffled)
        n_val   = max(1, int(VAL_FRAC * len(shuffled)))
        val_ids = [q["question_id"] for q in shuffled[:n_val]]
        fit_ids = [q["question_id"] for q in shuffled[n_val:]]
        payload = {
            "seed":     SEED,
            "val_frac": VAL_FRAC,
            "n_val":    len(val_ids),
            "n_train":  len(fit_ids),
            "val":      val_ids,
            "train":    fit_ids,
        }
        needs_write = True
        print(f"[split] Created split — val: {len(val_ids)}, train: {len(fit_ids)}")

    # Add train_sample if missing (backward-compatible addition)
    sample_key = f"train_sample_{TRAIN_SAMPLE_N}"
    if sample_key not in payload:
        all_train_ids = [q["question_id"] for q in train_qs]
        rng.seed(SEED)
        sample_ids = rng.sample(all_train_ids, min(TRAIN_SAMPLE_N, len(all_train_ids)))
        payload[sample_key] = sample_ids
        needs_write = True
        print(f"[split] Added {sample_key}: {len(sample_ids)} questions (seed={SEED})")

    if needs_write:
        SPLIT_FILE.write_text(json.dumps(payload, indent=2))


def load_questions(
    db_path: Path = DEFAULT_DB,
    subset: str = "test",           # "test" | "val" | "train"
) -> list[dict]:
    """
    Load questions for the requested subset.

    subset="test"  → official BSARD test split (222 questions)
    subset="val"   → 20% of train, seed 42
    subset="train" → remaining 80% of train
    """
    create_splits(db_path)

    if subset == "test":
        return _load_questions(db_path, "test")

    split_data = json.loads(SPLIT_FILE.read_text())
    all_train  = {q["question_id"]: q for q in _load_questions(db_path, "train")}

    if subset == "val":
        ids = split_data["val"]
    elif subset == "train":
        ids = split_data["train"]
    else:
        raise ValueError(f"Unknown subset: {subset!r}. Use 'test', 'val', or 'train'.")

    return [all_train[qid] for qid in ids if qid in all_train]


def load_train_sample(
    db_path: Path = DEFAULT_DB,
    n: int = TRAIN_SAMPLE_N,
) -> list[dict]:
    """
    Return the persisted random sample of *n* train questions (seed=42).

    Used for hyperparameter selection (e.g., BM25 k1/b grid search, concat_2x
    ablation winner) so that the test split (222 questions) is never touched
    during tuning. The sample is stored in split_ids.json under
    'train_sample_{n}' and is identical across all sessions.

    Raises ValueError if n != TRAIN_SAMPLE_N and no matching key exists.
    """
    create_splits(db_path)
    payload   = json.loads(SPLIT_FILE.read_text())
    sample_key = f"train_sample_{n}"
    if sample_key not in payload:
        raise ValueError(
            f"No persisted sample for n={n} in split_ids.json. "
            f"Available: {[k for k in payload if k.startswith('train_sample_')]}"
        )
    all_train = {q["question_id"]: q for q in _load_questions(db_path, "train")}
    return [all_train[qid] for qid in payload[sample_key] if qid in all_train]


def make_ground_truth(questions: list[dict]) -> dict[int, list[int]]:
    """Return {question_id: [article_id, ...]} from a question list."""
    return {q["question_id"]: q["relevant_article_ids"] for q in questions}
