# Combined data bundle — `bsard-rag-thesis`

All large artefacts live in **one** companion Hugging Face dataset, organised
into per-component subsets. No data is stored in git.

- Default repo: `Marios-Paschalidis-Thesis/bsard-rag-thesis-data` (override with `BSARD_HF_COMBINED_REPO`)
- Upload:   `python data_tooling/upload_combined_hf.py`
- Download: `python data_tooling/download_combined_hf.py`

> **License:** every subset is a derivative of BSARD, distributed under
> **CC BY-NC-SA 4.0** (see [`DATA_LICENSE.md`](DATA_LICENSE.md)).

## Layout on the Hub

```
Marios-Paschalidis-Thesis/bsard-rag-thesis-data  (dataset)
├── corpus/   ← bsard2currentlawmatching: SQLite corpora + Parquet/JSONL exports,
│              verification CSV, id-mappings, corpus stats
├── rq1/      ← RQ1_Retrieval_Methods: corpus DB, cached dense embeddings, BM25 stores,
│              result JSONs (RQ3_Autonomous_Evaluation reuses this subset)
└── rq2/      ← RQ2_Structure_Aware_Retrieval: per-arm data, ground truth, result tables
```

Each subset's exact contents and sizes are documented in that component's own
`DATA_CARD.md` (`bsard2currentlawmatching/DATA_CARD.md`, `RQ1_Retrieval_Methods/DATA_CARD.md`,
`RQ2_Structure_Aware_Retrieval/DATA_CARD.md`). `RQ3_Autonomous_Evaluation` ships no data of its own — it
consumes the `rq1/` subset.

## Migrating from the old per-component datasets

The data was previously published as three separate datasets
(`bsard2currentlawmatching`, `bsard-rq1-data`, `bsard-rq2-data`). To consolidate:

1. **Make sure the local data is present.** Point each subset at the directory it
   already reads/writes (these hold the same artefacts you previously uploaded):

   | Subset | Resolved from | Default if unset |
   |---|---|---|
   | corpus | `$CORPUS_DATA_DIR` | `bsard2currentlawmatching/output` |
   | rq1 | `$BSARD_DATA_DIR` | `RQ1_Retrieval_Methods/output` |
   | rq2 | `$RQ2_DATA_DIR` | `RQ2_Structure_Aware_Retrieval/data` |

   e.g. `export BSARD_DATA_DIR="/path/to/bsard-rq1-data"` (or pass
   `--rq1-dir ...`). You can also re-pull each old dataset locally first with its
   component's `download_data.py`, then upload from there.

2. **Dry-run** to confirm the plan (files + sizes per subset):

   ```bash
   python data_tooling/upload_combined_hf.py
   ```

3. **Create the repo and upload** (first time):

   ```bash
   python data_tooling/upload_combined_hf.py --create --confirm
   ```

   or one subset at a time, e.g. `--only corpus --confirm`. The uploader is
   idempotent — re-running skips files already on the Hub and retries failures.

4. **Repoint the components.** Each component resolves its dataset id from an env
   var. Set them to the combined repo (and use the combined download helper):

   ```bash
   export BSARD_HF_REPO=Marios-Paschalidis-Thesis/bsard-rag-thesis-data   # corpus + rq1 + rq3
   export RQ2_HF_REPO=Marios-Paschalidis-Thesis/bsard-rag-thesis-data     # rq2
   python data_tooling/download_combined_hf.py               # pull all subsets to defaults
   ```

   > Note: the per-component `download_data.py` scripts fetch a whole repo root;
   > for the combined dataset use `data_tooling/download_combined_hf.py`, which pulls
   > only the relevant `<subset>/` and unpacks it into the component's data root.

5. (Optional) Once verified, archive/delete the three old datasets on the Hub.

## Authentication

`hf auth login` once, or set `HF_TOKEN` to a write-scoped token for the
target repo. Never commit the token — use `.env.local` (gitignored).
