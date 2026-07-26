# Question subsetting metadata

Per-question categorisation of all 1,108 BSARD benchmark questions by the
PDF-extraction status of their ground-truth (mapped) BSARD articles. Used by
`evaluation.question_subsets` to assemble per-run ground-truth files.

## Files in this folder

| File | Origin |
| --- | --- |
| `questions_by_extraction_status.jsonl` | Vendored from Dataset_Creation; regenerable via `analysis/question_extraction_analysis.ipynb` |
| `summary.json` | Bucket counts (overall + per split) — regenerated alongside the JSONL |

The notebook reads `dataset_creation_output/bsard_corpus.db` (a junction at the
RQ2 repo root) and rewrites both files in place. Re-run it whenever the upstream
BSARD corpus DB changes.

## Unit of analysis: `bsard_id`

Every per-article record is keyed by `bsard_id` — the canonical BSARD article
identifier from `maastrichtlawtech/bsard`. This matches the keys used in T07's
ground-truth files (`bsard_train.json`, `bsard_test.json`).

The corpus DB stores 33,741 BSARD article *rows* across 22,633 unique
`bsard_id` values: the same logical BSARD article can be matched in several
PDFs (multi-part codes such as Code Civil, Code Judiciaire, Code d'Instruction
Criminelle have overlapping article-number ranges, so the linkage step
legitimately produces multiple rows per `bsard_id`).

## Buckets

A question's `extraction_status` is determined from the `verification_status`
of its ground-truth BSARD articles in the corpus DB.

| Bucket | Definition |
| --- | --- |
| `exact` | All relevant BSARD articles have `verification_status = 'FOUND'` — present in the PDFs unchanged. |
| `partial` | All relevant BSARD articles have `verification_status = 'PARTIAL'` — present in the PDFs but text differs from the BSARD canonical version (typically post-BSARD amendments). |
| `not_present` | All relevant BSARD articles have `verification_status = 'NOT FOUND'`. Includes HuggingFace-only stubs (`pdf_filename IS NULL`). |
| `mixed` | Relevant BSARD articles span more than one of the above buckets. |

## Output schema (`questions_by_extraction_status.jsonl`)

One JSON object per line, 1,108 lines total.

| Field | Type | Description |
| --- | --- | --- |
| `question_id` | int | BSARD question ID (matches `id` in `maastrichtlawtech/bsard` questions split). |
| `split` | str | `train` or `test`. |
| `extraction_status` | str | One of `exact`, `partial`, `not_present`, `mixed`. |
| `n_relevant_bsard_articles` | int | Count of unique `bsard_id`s referenced by this question (after deduplication). |
| `pdf_filenames` | list[str] | Sorted, deduplicated PDF filenames across all relevant articles. |
| `bsard_articles` | list[dict] | Per-article breakdown: `{bsard_id, verification_status, pdf_filenames, extraction_cosine}`. |

`bsard_articles[*].extraction_cosine` (added 2026-05-01) is a continuous companion
to `verification_status`: char-4-gram cosine between the canonical BSARD article
text (corpus DB) and the extracted span from this PDF (Dataset_Creation extraction
JSONL), produced by T03 (CN-T03-002) and merged in here at notebook-build time.
Aggregated by `max` across the article's PDFs ("most charitable" extraction).
`null` when no T03 doc has been regenerated under schema_version ≥ 3 for any of
the article's PDFs — informational only; downstream metrics gracefully degrade.

## Headline counts (n = 1,108 questions)

| Bucket | Count | Percentage |
| --- | ---: | ---: |
| `exact` | 491 | 44.31% |
| `partial` | 213 | 19.22% |
| `not_present` | 14 | 1.26% |
| `mixed` | 390 | 35.20% |

| Split | `exact` | `partial` | `not_present` | `mixed` | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| `train` | 385 | 167 | 10 | 324 | 886 |
| `test` | 106 | 46 | 4 | 66 | 222 |

Headline reading: only ~44% of questions have ground truth fully present in the
PDFs unchanged; ~35% are mixed (some BSARD articles exact, others changed or
missing); only ~1% are entirely absent.

## Consuming this data

The intended workflow for an evaluation run is:

```bash
python -m evaluation.question_subsets build \
    --status exact \
    --split test \
    --output ground_truth/runs/exact_test.json
```

Then point the orchestrator at `ground_truth/runs/exact_test.json` for a
single-subset run. See `src/evaluation/question_subsets.py` for the full CLI.
