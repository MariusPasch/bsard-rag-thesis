# Data bundle — BSARD-RQ1

The large artefacts for this project are **not** stored in git. They are
published as a companion **Hugging Face dataset** and downloaded on demand into
the local data root.

- Default HF repo: `mpaschalidis/bsard-rq1-data` (override with `BSARD_HF_REPO`)
- Download: `python scripts/download_data.py`
- Data root: `$BSARD_DATA_DIR`, or `<repo>/output` by default

> **License:** these artefacts are derivatives of BSARD and are distributed
> under **CC BY-NC-SA 4.0**. See [`DATA_LICENSE.md`](DATA_LICENSE.md). The raw
> BSARD corpus itself remains available from its authors at
> https://huggingface.co/datasets/maastrichtlawtech/bsard — this bundle only
> adds project-specific derived artefacts.

## Bundle contents

| Path | Size | Description |
|---|---|---|
| `bsard_corpus.db` | ~98 MB | SQLite corpus: ~22k deduplicated articles + 6,490 distractors, with a pre-built FTS5 index. Primary input for all tiers. |
| `bsard_articles_dedup.parquet` | ~9 MB | Deduplicated article corpus (22,633 rows) used by the dense/embedding pipeline. |
| `embeddings/*.npy` | ~580 MB | Cached dense corpus embeddings — one `(N, dim)` matrix + one `_ids.npy` per model (7 models). Large matrices are stored sharded (see below). |
| `cache/*.pkl` | ~109 MB | BM25 tokenization caches (pre-tokenized corpus per normalisation × field-weighting). Stored sharded. |
| `*_id_mapping.json` | <1 MB | Mapping between local `article_id` and the upstream HF BSARD ids. |
| `corpus_stats.json` | <1 MB | Corpus statistics (counts, overlap metrics). |
| `results/<tier>/*.json` | ~29 MB | All experiment result JSONs (sparse / dense / hybrid / agentic / RQ3). |

This is a **full** mirror (consistent with the RQ2 dataset): the cached
embeddings and BM25 stores are included so experiments run without recomputation.
All of it is regenerable from the corpus if preferred.

**FAISS indices are not shipped** — RQ1 builds its `IndexFlatIP` in memory from
the `.npy` embeddings at load time, so the embeddings *are* the index (same
information, just stored as vectors rather than a prebuilt index file).

### Sharded files

Some artefacts exceed the uploader's per-file limit, so they are split into
`<=9 MB` byte-parts named `<name>.partNNN` and listed in `sharded_files.json`
(with each original's size + SHA-256). `scripts/download_data.py` reassembles
them automatically after download — you don't interact with the parts directly.

Dev-only intermediates (raw `.jsonl`, `_clean`/`_only` corpus variants, source
PDFs, extraction logs) are intentionally **excluded** from the public bundle.

## Corpus facts

- ~22k unique articles (deduplicated; cite externally as "~22k articles").
- 1,108 questions (222 test / 886 train); mean 6.18 relevant articles/question.
- Ground truth is matched on `article_id`.

## Rebuilding from scratch

The embeddings can be regenerated from the corpus with the Tier-2 scripts; the
corpus DB itself is built from the upstream BSARD release (see
`scripts/setup/` and `RETRIEVAL_PROJECT.md`).
