# RQ2_Structure_Aware_Retrieval

Code and experiments for **RQ2** of a KU Leuven master's thesis: a per-document
comparison of retrieval strategies for Retrieval-Augmented Generation over the
**BSARD** Belgian statutory corpus.

The pipeline is split into nine sub-projects (`RQ2_T00`–`RQ2_T08`), each its own
Python package with its own `pyproject.toml` and `README.md`. This repository
ties them together as a monorepo.

## Research design — three retrieval arms

| Arm | Project | Approach |
|-----|---------|----------|
| **Arm 1 — Naive** | `RQ2_T03_ARM1_NAIVE` | Sliding-window chunking + hybrid retrieval (dense + BM25 sparse, fused with RRF) |
| **Arm 2A — Metadata** | `RQ2_T04_ARM2_METADATA` | Enriched article embeddings with LLM-extracted metadata for filtering/boosting |
| **Arm 2B — PageIndex** | `RQ2_T05_ARM2_PAGEINDEX` | Vectorless: ToC-tree construction + LLM-guided hierarchical navigation |
| **Arm 2C — Agentic** | `RQ2_T04_ARM2_METADATA/experiments/arm2c` | CRAG × ReAct corrective loop over a deep navigation tree (extension of Arm 2A) |

All arms are evaluated on the same ground truth and consolidated side-by-side.

## Quick start

> Part of the [**bsard-rag-thesis**](../README.md) mono-repo (RQ2). Its data is
> the `rq2/` subset of the combined Hugging Face dataset.

```bash
# 1. Clone the mono-repo and enter this component
git clone https://github.com/MariusPasch/bsard-rag-thesis.git
cd bsard-rag-thesis/RQ2_Structure_Aware_Retrieval

# 2. Download the rq2 data subset (~2.5 GB) into ./data (run from the mono-repo root)
pip install huggingface_hub
python ../data_tooling/download_combined_hf.py --subset rq2   # -> RQ2_Structure_Aware_Retrieval/data

# 3. Wire each sub-project's data/ directory to the downloaded bundle
python scripts/setup/link_data.py

# 4. Set up a sub-project and run it (each is self-contained)
cd RQ2_T03_ARM1_NAIVE
python -m venv .venv && .venv/Scripts/Activate.ps1   # PowerShell; use source .venv/bin/activate on POSIX
pip install -e .
```

Copy `.env.example` to `.env.local` and adjust paths/keys as needed. Nothing in
`.env.local` is committed.

## Data

The large artefacts (BSARD corpus DB, Azure Document Intelligence exports, the
per-arm FAISS / BM25 / PageIndex indices and cached embeddings, and the result
JSONs) are **not** in git. They live in the **`rq2/` subset** of the combined
Hugging Face dataset and download into a single data root:

- **Combined dataset:** [`Marios-Paschalidis-Thesis/bsard-rag-thesis-data`](https://huggingface.co/datasets/Marios-Paschalidis-Thesis/bsard-rag-thesis-data), subset `rq2/`
- **Download (preferred):** from the mono-repo root,
  `python data_tooling/download_combined_hf.py --subset rq2`
- **Data root:** `$RQ2_DATA_DIR`, or `<repo>/data` by default
- **Layout & sizes:** see [`DATA_CARD.md`](DATA_CARD.md) (mono-repo layout in [../DATA_CARD.md](../DATA_CARD.md))
- **Also:** the component's own `scripts/download_data.py` now pulls the `rq2/`
  subset of the same combined dataset (override the repo with `RQ2_HF_REPO`;
  `--allow <globs>` to fetch only part of it), then run `scripts/setup/link_data.py`

`scripts/setup/link_data.py` recreates the per-sub-project `data/` directories as
links into this root, so each project's code finds its inputs unchanged. The
links are portable: any machine works once the data root is downloaded.

## Pipeline order (T00 → T08)

| Project | Role |
|---------|------|
| `RQ2_T00_ORCHESTRATOR` | CLI entry point, config loading, cross-project coordination |
| `RQ2_T01_SHARED` | Shared components: embedding model, LLM wrapper, FAISS/BM25 stores |
| `RQ2_T02_DATA_LOADER` | Discovers PDF↔CSV pairs, extracts text, builds `Article` records |
| `RQ2_T03_ARM1_NAIVE` | **Arm 1** retrieval |
| `RQ2_T04_ARM2_METADATA` | **Arm 2A** retrieval (+ **Arm 2C** agentic extension) |
| `RQ2_T05_ARM2_PAGEINDEX` | **Arm 2B** retrieval |
| `RQ2_T06_ARM_RESULTS` | Cross-arm consolidation: tables, figures, error analyses (presentation only — no metric computation) |
| `RQ2_T07_EVALUATION` | Evaluation harness: binary + weighted IR metrics, autonomous LLM judge, cost tracking |
| `RQ2_T08_RAG_VISUALIZATION` | Local Streamlit viewer for chunks/articles/retrieval over the original PDF layout |

## How the arms are evaluated and reported

1. Each arm (T03/T04/T05) produces per-query retrieval results.
2. `RQ2_T07_EVALUATION` scores them against the BSARD ground truth
   (`RQ2_T07_EVALUATION/ground_truth/`) using a single shared metric path.
3. `RQ2_T06_ARM_RESULTS` consolidates the scored runs into cross-arm tables,
   figures, and error analyses. See `RQ2_T06_ARM_RESULTS/SOURCE_MAP.md` and
   `EVALUATION_METHODOLOGY.md` for provenance.

## Configuration (environment variables)

| Variable | Purpose | Default |
|---|---|---|
| `RQ2_DATA_DIR` | Data bundle root | `<repo>/data` |
| `RQ2_HF_REPO` | Companion HF dataset id | `Marios-Paschalidis-Thesis/bsard-rag-thesis-data` |
| `RQ2_BSARD_DB` | BSARD corpus DB path | `<RQ2_DATA_DIR>/bsard_corpus.db` |

See [`.env.example`](.env.example) for the full list, including the optional
LLM and Azure-notebook credentials.

## License & citation

- **Code:** MIT — see [`LICENSE`](LICENSE).
- **Data:** CC BY-NC-SA 4.0 (derivative of BSARD) — see [`DATA_LICENSE.md`](DATA_LICENSE.md).
- **Citation:** see [`CITATION.cff`](CITATION.cff). Please also cite the
  underlying [BSARD](https://huggingface.co/datasets/maastrichtlawtech/bsard) dataset.
