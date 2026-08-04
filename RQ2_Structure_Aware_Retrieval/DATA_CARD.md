# Data bundle — BSARD-RQ2

The large artefacts for this project are **not** stored in git. They are
published as a companion **Hugging Face dataset** and downloaded on demand into
the local data root.

- Default HF repo: `mpaschalidis/bsard-rq2-data` (override with `RQ2_HF_REPO`)
- Download: `python scripts/download_data.py`
- Wire sub-projects: `python scripts/setup/link_data.py`
- Data root: `$RQ2_DATA_DIR`, or `<repo>/data` by default

> **License:** these artefacts are derivatives of BSARD and of Belgian statutory
> legislation, distributed under **CC BY-NC-SA 4.0**. See
> [`DATA_LICENSE.md`](DATA_LICENSE.md). The raw BSARD corpus remains available
> from its authors at https://huggingface.co/datasets/maastrichtlawtech/bsard.

## Layout

The bundle mirrors the project's shared data tree. After download + linking,
each sub-project finds its inputs under its own `data/` (a junction into the
root below):

```
<RQ2_DATA_DIR>/
    bsard_corpus.db                 # BSARD SQLite corpus + index
    shared_source/                  # T02 inputs: source PDFs + CSVs
    pdf_cache/                      # Arm 1 caches (shared by T03 + T07)
    AzureDI/                        # Azure Document Intelligence layout exports
    RQ2_T04_ARM2_METADATA/          # Arm 2A metadata indices
    RQ2_T05_ARM2_PAGEINDEX/         # Arm 2B PageIndex trees
```

## Bundle contents

| Path | Approx size | Description |
|---|---|---|
| `bsard_corpus.db` | ~0.10 GB | SQLite BSARD corpus + index; ground-truth source for every arm. |
| `shared_source/` | ~0.06 GB | The curated source PDFs of Belgian statutory codes + question/metadata CSVs (T02 loader inputs). |
| `pdf_cache/` | ~0.77 GB | Arm 1 (naive) per-PDF caches: extracted text, article spans, chunks, FAISS + BM25 indices, cached embeddings. Shared by T03 and T07. |
| `AzureDI/` | ~9 MB | Azure Document Intelligence layout export for the 5 selected docs (`VectorDB_Documents.json` + `MyDocuments.csv`). **Minimised for release** by `scripts/sanitise_azuredi.py`: fields not needed to reproduce the retrieval results are stripped, and non-selected documents removed. Reproduces Arm 2A / 2C identically (the loader re-embeds `page_content` at runtime). |
| `RQ2_T04_ARM2_METADATA/` | ~0.82 GB | Arm 2A metadata-aware retrieval indices + cached embeddings. |
| `RQ2_T05_ARM2_PAGEINDEX/` | ~0.43 GB | Arm 2B PageIndex LLM-built navigation trees + per-query results. |
| **Total** | **~1.9 GB** | Full mirror (AzureDI minimised; 11 oversized regenerable pools excluded — see below). |

Cached dense embeddings, FAISS indices and BM25 stores **are included** in this
bundle for turnkey reproduction. They can also be regenerated from the corpus
with each arm's `precompute_*` scripts if you prefer to rebuild from scratch.

## Corpus / scope facts

- Operational scope is a **curated 5-PDF set** of Belgian statutory codes (Code
  Civil, Code Pénal, Code Judiciaire ×2, Code du Logement); see
  `RQ2_T00_ORCHESTRATOR/data/selected_pdfs.json`.
- One BSARD document (2004A27101) was excluded for ground-truth drift; the
  three retrieval arms are evaluated per-document over the curated set.
- Ground truth lives in the code repo under
  `RQ2_T07_EVALUATION/ground_truth/`, matched on `bsard_id`.

## Excluded from the bundle

- **11 oversized retrieval-pool files** — the per-PDF
  `pdf_cache/<stem>/results/<hash>/retrieval_pool/{483beb56a47b,50744bfce1d0}.jsonl`
  candidate pools (each ~12–48 MB, ~300 MB total). These are **regenerable**:
  rebuild with `RQ2_T03_ARM1_NAIVE/scripts/precompute_retrieval.py`. (Every
  smaller pool/`.jsonl` and all FAISS/BM25 indices are included.)
- Source-code repos and the cross-arm `Report/` (in git / built locally).
- Cloud-run logs and dev intermediates.
- Anything else regenerable purely from the code + the artefacts above.
