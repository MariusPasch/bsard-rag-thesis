# BSARD RAG Thesis — Retrieval-Augmented Generation over Belgian Statutory Law

**Author:** Marios Paschalidis · KU Leuven · Master's thesis

A mono-repo bundling the full pipeline behind the thesis: building a Belgian
statutory corpus from primary legal sources, then evaluating retrieval and
retrieval-augmented-generation (RAG) methods over it. All work is grounded in
the **Belgian Statutory Article Retrieval Dataset (BSARD)** and a corpus of 49
Justel consolidated Belgian law PDFs.

---

## Components

| Directory | Role | What it produces |
|---|---|---|
| [`bsard2currentlawmatching/`](bsard2currentlawmatching/) | **Corpus construction** — the foundational dataset every research question consumes. | A SQLite corpus database (40,231 articles, 1,108 questions, citation graph) with Parquet/JSONL exports, linking the 49 PDFs to BSARD. |
| [`RQ1_Retrieval_Methods/`](RQ1_Retrieval_Methods/) | **RQ1** — sparse, dense, hybrid and agentic retrieval. | Systematic comparison of BM25, dense bi-encoders, hybrid fusion, and LLM-judge / CRAG / ReAct pipelines. |
| [`RQ2_Structure_Aware_Retrieval/`](RQ2_Structure_Aware_Retrieval/) | **RQ2** — naive, metadata-aware and PageIndex retrieval arms. | Per-document comparison of three retrieval arms over Belgian statutory law. |
| [`RQ3_Autonomous_Evaluation/`](RQ3_Autonomous_Evaluation/) | **RQ3** — a reusable multi-tier evaluation harness. | A four-tier metric stack (efficiency, BSARD replication, supervised IR, autonomous evaluation). |

The dependency flow is linear: **`bsard2currentlawmatching`** builds the corpus,
and **RQ1–RQ3** each consume it. Start with the corpus project, then any RQ.

Each component is self-contained — see its own `README.md` for setup, the data
it needs, and how to run it.

---

## Data

No large data lives in git. All artefacts live in **one** companion Hugging Face
dataset, organised into per-component subsets, and download into local,
gitignored data roots.

- Dataset: `Marios-Paschalidis-Thesis/bsard-rag-thesis-data` (override with `BSARD_HF_COMBINED_REPO`)
- Layout: `corpus/`, `rq1/`, `rq2/` (RQ3 reuses the `rq1/` subset)
- Install the tooling deps once: `pip install -r requirements.txt`
- Pull everything: `python data_tooling/download_combined_hf.py`
- Full contents, sizes, and the upload/migration procedure: [DATA_CARD.md](DATA_CARD.md)

Point each component at the combined dataset via its env var
(`BSARD_HF_REPO` for corpus/RQ1/RQ3, `RQ2_HF_REPO` for RQ2) — the component code
reads the id from the environment, so no source changes are needed.

> The data was previously published as three separate datasets
> (`bsard2currentlawmatching`, `bsard-rq1-data`, `bsard-rq2-data`). The
> consolidation into the single dataset above is driven by
> [`data_tooling/upload_combined_hf.py`](data_tooling/upload_combined_hf.py) — see
> [DATA_CARD.md](DATA_CARD.md) for the step-by-step re-upload.

---

## Licensing

- **Source code** — MIT (see [LICENSE](LICENSE)).
- **Data artefacts** — derivatives of BSARD, distributed under **CC BY-NC-SA
  4.0** (see [DATA_LICENSE.md](DATA_LICENSE.md)). The raw BSARD corpus remains
  available from its authors at
  https://huggingface.co/datasets/maastrichtlawtech/bsard.

If you use this work, please cite it (see [CITATION.cff](CITATION.cff)) **and**
the underlying BSARD dataset (Louis & Spanakis, 2022).

---

## Repository layout

```
bsard-rag-thesis/
├── bsard2currentlawmatching/   ← corpus construction (run first)
├── RQ1_Retrieval_Methods/        ← RQ1: sparse / dense / hybrid / agentic retrieval
├── RQ2_Structure_Aware_Retrieval/ ← RQ2: naive / metadata / PageIndex arms
├── RQ3_Autonomous_Evaluation/    ← RQ3: multi-tier evaluation harness
├── data_tooling/               ← combined Hugging Face up/download tooling
│   ├── upload_combined_hf.py
│   └── download_combined_hf.py
├── README.md                   ← this file
├── requirements.txt            ← deps for data_tooling/ (huggingface_hub, requests)
├── DATA_CARD.md                ← combined dataset layout + re-upload procedure
├── LICENSE                     ← MIT (source code)
├── DATA_LICENSE.md             ← BSARD attribution (CC BY-NC-SA 4.0)
├── CITATION.cff                ← how to cite the thesis code
└── .env.example                ← shared Hugging Face data wiring
```

The only top-level `requirements.txt` covers the `data_tooling/` scripts. Each
component additionally keeps its own `requirements.txt` and (where relevant) a
separate virtual environment; there is no shared environment for the experiment
code.
