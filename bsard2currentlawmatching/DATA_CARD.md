# Data bundle — bsard2currentlawmatching (corpus)

The large artefacts for this project are **not** stored in git. They live in the
companion **Hugging Face dataset** `Marios-Paschalidis-Thesis/bsard-rag-thesis-data` (subset
`corpus/`) and download on demand into a local gitignored data root.

- Data root: env `CORPUS_DATA_DIR`, default `<repo>/output` (gitignored)
- Download (from the mono-repo root): `python data_tooling/download_combined_hf.py --subset corpus`
- Or use the component's own script: `python scripts/download_from_hf.py`
  (pulls the `corpus/` subset; override the repo with `BSARD_HF_REPO`)

> **License:** these artefacts are derivatives of BSARD and are distributed
> under **CC BY-NC-SA 4.0**. See [`DATA_LICENSE.md`](DATA_LICENSE.md). The raw
> BSARD corpus itself remains available from its authors at
> https://huggingface.co/datasets/maastrichtlawtech/bsard — this bundle only
> adds project-specific derived artefacts.

> **Mono-repo note:** this corpus is the foundational dataset that the RQ1-RQ3
> retrieval projects consume. It is the `corpus/` subset of the combined Hugging
> Face dataset `Marios-Paschalidis-Thesis/bsard-rag-thesis-data`, published alongside the
> `rq1/` and `rq2/` subsets. See the mono-repo [DATA_CARD.md](../DATA_CARD.md)
> for the full layout.

## Bundle contents (~268 MB)

| Path | Size | Description |
|---|---|---|
| `bsard_corpus.db` | ~103 MB | Primary SQLite corpus: 40,231 articles + 1,108 questions + 27,712 citation edges, with a pre-built FTS5 index. The single source of truth — everything else derives from it. |
| `bsard_corpus_clean.db` | ~71 MB | Deduplicated, PDF-only companion DB (28,817 unique articles). See [CLEAN_DATASET.md](CLEAN_DATASET.md). |
| `bsard_articles_clean.parquet` | ~10 MB | Clean dataset in Parquet. |
| `bsard_articles_dedup.parquet` | ~10 MB | Earlier dedup variant (superseded by clean). |
| `bsard_hf_articles.parquet` | ~9 MB | HuggingFace BSARD corpus snapshot. |
| `bsard_full_verify.csv` | ~9 MB | Source URL + verification metadata (pipeline input). |
| `corpus_stats.json` | <1 MB | Chapter 3 corpus statistics. |
| `hf_to_local_id_mapping.json`, `local_to_hf_id_mapping.json` | <1 MB | BSARD-id ↔ local `article_id` maps. |
| `question_analysis/` | <1 MB | Per-question PDF-extraction status (2 files). |

This is the **authoritative** bundle: the two SQLite databases plus the small
support files and the pipeline's source CSV. Everything a downstream project
needs joins on `article_id`, which is stable across both databases.

### Regenerated locally, not published

Bulky derived artefacts regenerate cleanly from the published DBs and CSV, so
they are intentionally **excluded** from the bundle:

- `bsard_articles*.{parquet,jsonl}` full/subset exports — `python pipeline/export_corpus.py`
- `bsard_articles_clean.jsonl` — `python pipeline/build_clean_dataset.py`
- `pdfs/` (49 Justel source PDFs, ~61 MB) — `python pipeline/download_pdfs.py`
- `extracted/`, `linked/` Phase A/B/C intermediates — re-run the pipeline phases

## Corpus facts

- 40,231 articles (33,741 BSARD / 6,490 non-BSARD distractors); 22,633 unique BSARD ids.
- 1,108 questions (222 test / 886 train).
- 27,712 resolved citation edges; ground truth is matched on `article_id`.

## Rebuilding from scratch

The full corpus is rebuilt from the upstream BSARD release plus the 49 Justel
PDFs by running the five pipeline phases in order — see the **Pipeline** section
of [README.md](README.md) and the full spec in
[CORPUS_DATABASE_PROJECT.md](CORPUS_DATABASE_PROJECT.md).
