# BSARD Question Analysis by PDF Extraction Status

Categorisation of all 1,108 BSARD benchmark questions by the PDF-extraction status of their ground-truth (mapped) BSARD articles. Intended for downstream retrieval experiments that need to know, per question, whether the relevant evidence is actually present in the PDF corpus, was modified, or is missing entirely.

- **Notebook:** [analysis/question_extraction_analysis.ipynb](analysis/question_extraction_analysis.ipynb)
- **Output (Hugging Face dataset, not committed):** `output/question_analysis/questions_by_extraction_status.jsonl` and `output/question_analysis/summary.json`
- **Source DB:** `output/bsard_corpus.db` (read-only)

---

## Unit of analysis: `bsard_id`

**Every per-article record in this output is keyed by `bsard_id`, the unique BSARD article identifier.** This is the same `id` field used by the canonical HuggingFace BSARD dataset (`maastrichtlawtech/bsard`, corpus split). The local `articles.article_id` column is **not** referenced in the output — downstream projects should join on `bsard_id`.

Why dedup. The corpus database stores 33,741 BSARD article *rows* across 22,633 unique `bsard_id` values: the same logical BSARD article can be matched in several PDFs (multi-part codes such as Code Civil, Code Judiciaire, Code d'Instruction Criminelle have overlapping article-number ranges, so the linkage step legitimately produces multiple rows per `bsard_id`). For analysis at the level of "is this article present in the corpus?", duplication adds no information — the canonical text comes from the HuggingFace dataset and `verification_status` is consistent across all rows that share a `bsard_id` (verified: 0 inconsistent ids in the DB).

The output schema therefore reports each relevant article once per question (deduplicated by `bsard_id`) and lists *every* PDF in which that BSARD article appears.

---

## Categories

A question's category depends on the `verification_status` of its ground-truth BSARD articles in the `articles` table.

| Bucket | Definition |
| --- | --- |
| `exact` | All relevant BSARD articles have `verification_status = 'FOUND'` — present in the PDFs unchanged. |
| `partial` | All relevant BSARD articles have `verification_status = 'PARTIAL'` — present in the PDFs but text differs from the BSARD canonical version (typically post-BSARD amendments). |
| `not_present` | All relevant BSARD articles have `verification_status = 'NOT FOUND'`. This includes HuggingFace-only stubs, which always have `pdf_filename IS NULL` and `verification_status = 'NOT FOUND'`. |
| `mixed` | Relevant BSARD articles span more than one of the above buckets. |

## Multi-article rule

A question can map to multiple BSARD articles (median = 2, max = 109 unique `bsard_id`s). The rule:

1. Build the per-article bucket from `verification_status` (`FOUND` → `exact`, `PARTIAL` → `partial`, `NOT FOUND` → `not_present`).
2. Deduplicate the question's `relevant_bsard_ids` (one question in the source data lists a `bsard_id` twice — that duplicate is collapsed).
3. If all per-article buckets are equal, the question takes that bucket.
4. If the per-article buckets differ at all, the question is `mixed`.

**Why `mixed` over collapsing to a worst-case bucket.** Worst-case loses signal — a question with 9/10 articles `FOUND` would look identical to one with 0/10. `mixed` preserves the distinction. Any consumer that wants worst-case can derive it locally from the `bsard_articles` array in each output row without re-querying the DB.

**HuggingFace stubs.** Articles linked from the BSARD HF dataset that were never located in any PDF are stored as stubs (`pdf_filename IS NULL`) and always carry `verification_status = 'NOT FOUND'`. They are treated identically to other `NOT FOUND` articles for bucket assignment; the per-article breakdown still distinguishes them via an empty `pdf_filenames` list.

---

## Output file format

`output/question_analysis/questions_by_extraction_status.jsonl` — one JSON object per line, 1,108 lines.

| Field | Type | Description |
| --- | --- | --- |
| `question_id` | int | BSARD question ID (matches `id` in `maastrichtlawtech/bsard` questions split). |
| `split` | str | `train` or `test`. |
| `extraction_status` | str | One of `exact`, `partial`, `not_present`, `mixed`. |
| `n_relevant_bsard_articles` | int | Count of unique `bsard_id`s referenced by this question (after deduplicating `relevant_bsard_ids`). |
| `pdf_filenames` | list[str] | Sorted, deduplicated PDF filenames across all relevant articles. Empty list if all relevant articles are HF-only stubs. |
| `bsard_articles` | list[dict] | Per-article breakdown. Each entry: `bsard_id` (int), `verification_status` (`FOUND` / `PARTIAL` / `NOT FOUND`), `pdf_filenames` (list[str], possibly empty for HF-only stubs, possibly multiple for articles in multi-part codes). |

JSONL is used (rather than CSV) because each row carries a nested per-article list that doesn't flatten cleanly. It also matches the project's existing export format and is trivially streamable.

`output/question_analysis/summary.json` is a small companion file with the unit-of-analysis declaration and overall + per-split counts for traceability.

---

## How another project should consume this

### Load and filter with pandas

```python
import pandas as pd

df = pd.read_json(
    "output/question_analysis/questions_by_extraction_status.jsonl",
    lines=True,
)

# Test-split questions whose ground truth is fully present in the PDFs unchanged
strict = df[(df["split"] == "test") & (df["extraction_status"] == "exact")]

# Drop questions that have any missing or modified ground-truth articles
clean = df[df["extraction_status"] == "exact"]
question_ids = clean["question_id"].tolist()
```

### Join back to canonical BSARD article text via HuggingFace

```python
import pandas as pd
from datasets import load_dataset

df = pd.read_json(
    "output/question_analysis/questions_by_extraction_status.jsonl",
    lines=True,
)
corpus = load_dataset("maastrichtlawtech/bsard", "corpus", split="corpus").to_pandas()
# corpus columns: id, article, article_no, code  -- where 'id' == bsard_id

row = df.iloc[0]
for entry in row["bsard_articles"]:
    canonical = corpus.loc[corpus["id"] == entry["bsard_id"]].iloc[0]
    print(entry["bsard_id"], entry["verification_status"], canonical["article"][:200])
```

### Audit a single question without re-querying the DB

```python
import json

target = 42
with open("output/question_analysis/questions_by_extraction_status.jsonl", encoding="utf-8") as fh:
    for line in fh:
        row = json.loads(line)
        if row["question_id"] == target:
            print(row["extraction_status"], row["pdf_filenames"])
            for a in row["bsard_articles"]:
                print(a["bsard_id"], a["verification_status"], a["pdf_filenames"])
            break
```

### Re-derive a worst-case bucket

```python
ORDER = {"exact": 0, "partial": 1, "not_present": 2}
STATUS_TO_BUCKET = {"FOUND": "exact", "PARTIAL": "partial", "NOT FOUND": "not_present"}

def worst_case(row: dict) -> str:
    if not row["bsard_articles"]:
        return "not_present"
    return max(
        (STATUS_TO_BUCKET[a["verification_status"]] for a in row["bsard_articles"]),
        key=ORDER.get,
    )
```

---

## Summary

### Overall (n = 1,108 questions)

| Bucket | Count | Percentage |
| --- | ---: | ---: |
| `exact` | 491 | 44.31% |
| `partial` | 213 | 19.22% |
| `not_present` | 14 | 1.26% |
| `mixed` | 390 | 35.20% |

### By split

| Split | `exact` | `partial` | `not_present` | `mixed` | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| `train` | 385 (43.45%) | 167 (18.85%) | 10 (1.13%) | 324 (36.57%) | 886 |
| `test` | 106 (47.75%) | 46 (20.72%) | 4 (1.80%) | 66 (29.73%) | 222 |
| **all** | **491 (44.31%)** | **213 (19.22%)** | **14 (1.26%)** | **390 (35.20%)** | **1,108** |

Headline reading: only ~44% of questions have ground truth that is fully present in the PDFs unchanged; ~35% are mixed (some BSARD articles extracted exactly, others changed or missing); only ~1% are entirely absent. A retrieval evaluation that requires fully-present ground truth should restrict to `exact` (or to `exact` + per-article subsetting on `mixed` rows using the `bsard_articles` array).
