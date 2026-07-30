# BSARD Retrieval Experiments — RQ1

**Author:** Marios Paschalidis — KU Leuven, Master of Artificial Intelligence
**Thesis:** Enhancing Performance and Quality of Context Retrieval in RAG Systems
**Dataset:** Belgian Statutory Article Retrieval Dataset (BSARD)

A systematic comparison of **sparse, dense, hybrid, and agentic** retrieval
strategies over Belgian statutory law (French), evaluated with standard IR
metrics. This repository holds RQ1 of the thesis.

| License (code) | License (data) |
|---|---|
| [MIT](LICENSE) | [CC BY-NC-SA 4.0](DATA_LICENSE.md) (derived from BSARD) |

---

## Experimental tiers

| Tier | Method | Scope | Plan | Analysis |
|---|---|---|---|---|
| 1 | Sparse (TF-IDF, BM25, FTS5) | Lexical baseline | [plan](TIER1_SPARSE_RETRIEVAL_PLAN.md) | [analysis](analysis/sparse_retrieval/) |
| 2 | Dense (bi-encoder embeddings) | Semantic retrieval | [plan](TIER2_DENSE_RETRIEVAL_PLAN.md) | [analysis](analysis/dense_retrieval/) |
| 3 | Hybrid + cross-encoder rerank | Score fusion | [plan](TIER3_HYBRID_RETRIEVAL_PLAN.md) | [analysis](analysis/hybrid/) |
| 4.0 | LLM-as-a-Judge reranking | LLM reranking | [plan](TIER40_LLM_JUDGE_PLAN.md) | [analysis](analysis/agentic/tier40/) |
| 4.1 | Agentic CRAG | Corrective RAG | [plan](TIER41_AGENTIC_CRAG_PLAN.md) | [analysis](analysis/agentic/tier41/) |
| 4.2 | Agentic ReAct | Multi-step retrieval | [plan](TIER42_AGENTIC_REACT_PLAN.md) | [analysis](analysis/agentic/tier42/) |

All tiers are complete. Follow-up RQ3 evaluator analysis lives under
[analysis/RQ3/](analysis/RQ3/) (it consumes RQ1 result JSONs directly).

---

## Quickstart

> Part of the [**bsard-rag-thesis**](../README.md) mono-repo (RQ1). Needs the
> corpus built by [`bsard2currentlawmatching`](../bsard2currentlawmatching/),
> shipped here as the `rq1/` data subset.

```bash
# 1. Clone the mono-repo and enter this component
git clone https://github.com/MariusPasch/bsard-rag-thesis.git
cd bsard-rag-thesis/RQ1_Retrieval_Methods

# 2. Environment
python -m venv .venv
# Windows:  .\.venv\Scripts\Activate.ps1
# Linux/macOS:  source .venv/bin/activate
pip install -r requirements.txt

# 3. Download the rq1 data subset (run from the mono-repo root)
python ../data_tooling/download_combined_hf.py --subset rq1   # -> RQ1_Retrieval_Methods/output

# 4. Language models (sparse tier preprocessing)
python -m spacy download fr_core_news_lg
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

# 5. (optional) install the evaluation service — see below
```

Verify data access:

```python
import sqlite3
from evaluation.paths import CORPUS_DB
n = sqlite3.connect(str(CORPUS_DB)).execute(
    "SELECT COUNT(*) FROM articles WHERE is_bsard_article=1").fetchone()[0]
print(f"BSARD articles in DB: {n}")   # ~22,600
```

Run a tier (example — Tier 1 sparse):

```bash
python scripts/evaluation/tier1/run_sparse_experiments.py
```

---

## Data: where the big files live

Large artefacts (corpus DB, parquet, embeddings, result JSONs) are **not stored
in git**. They live in the **`rq1/` subset** of the combined Hugging Face
dataset [`Marios-Paschalidis-Thesis/bsard-rag-thesis-data`](https://huggingface.co/datasets/Marios-Paschalidis-Thesis/bsard-rag-thesis-data)
and are fetched on demand. See [DATA_CARD.md](DATA_CARD.md) for the bundle
contents and the mono-repo [DATA_CARD.md](../DATA_CARD.md) for the full layout.

- **Download (preferred):** from the mono-repo root,
  `python data_tooling/download_combined_hf.py --subset rq1`
- **Data root** is resolved by [evaluation/paths.py](evaluation/paths.py):
  1. the `BSARD_DATA_DIR` environment variable, if set; else
  2. `<repo>/output` (created on demand).
- **Also:** the component's own `scripts/download_data.py` now pulls the `rq1/`
  subset of the same combined dataset (override the repo with `BSARD_HF_REPO`;
  `--no-embeddings` to skip the cached embeddings).

So any machine works once the bundle is downloaded — no OS-specific symlinks
required. Copy [.env.example](.env.example) to `.env.local` to set
`BSARD_DATA_DIR`, `BSARD_HF_REPO`, `OPENAI_API_KEY`, or `HF_HOME`.

> The raw BSARD corpus belongs to its authors
> (https://huggingface.co/datasets/maastrichtlawtech/bsard). This project's
> bundle only redistributes **derived** artefacts under CC BY-NC-SA 4.0.

---

## Repository structure

```
RQ1_Retrieval_Methods/
├── retrieval/                  # Retrieval library (imported by scripts & notebooks)
│   ├── preprocessing.py        #   tokenization, normalisation, field builders
│   ├── sparse.py               #   Tier 1: TF-IDF, BM25, FTS5
│   ├── dense.py                #   Tier 2: bi-encoder + FAISS IndexFlatIP
│   ├── hybrid.py               #   Tier 3: RRF / linear fusion
│   ├── llm_reranker.py         #   Tier 3: cross-encoder reranking
│   └── agentic/                #   Tier 4: crag.py, react.py, tools.py, clients, prompts
│
├── evaluation/                 # Evaluation library + persisted runtime data
│   ├── metrics.py, runner.py, split.py, stratify.py
│   ├── paths.py                #   data-root resolver (BSARD_DATA_DIR)
│   └── data/                   #   split_ids.json, query_strata.json, tier3_subset.json, fewshot_examples.json
│
├── scripts/
│   ├── download_data.py        #   fetch the data bundle from Hugging Face
│   ├── evaluation/             #   experiment runners, by tier (tier1..tier4, RQ3, shared)
│   ├── hybrid/                 #   test-only hybrid orchestrator variant
│   └── setup/                  #   build_hf_id_mapping.py, prepare_dedup_corpus.py, download_tier4x_*.py, setup_new_device.ps1
│
├── azure_notebooks/            # Cloud GPU runners (Tier 2 Qwen3 + Tier 4) — see note below
├── analysis/                   # Result-analysis notebooks + generated figures (rendered)
├── tests/                      # Unit tests (pytest)
│
├── README.md
├── DATA_CARD.md                # data bundle contents & HF location
├── DATA_LICENSE.md             # data attribution & CC BY-NC-SA 4.0
├── LICENSE                     # MIT (code)
├── CITATION.cff
├── RETRIEVAL_PROJECT.md        # full corpus & benchmark reference
├── TIER*_*_PLAN.md             # per-tier design documents
├── requirements.txt
├── .env.example
└── .gitignore
```

> **`evaluation/` holds code and supporting data only** — never result outputs.
> All result JSONs go to `<data-root>/results/<tier>/`.

> **Azure notebooks** under `azure_notebooks/` are operational scaffolding used
> to run the GPU-heavy tiers on an Azure VM (clone repo → run → stage artefacts).
> They require your own `GITHUB_TOKEN` and `AZURE_CONTAINER_SAS_URL` (read from
> the environment — never hardcode them). They are **not** needed to reproduce
> results locally and are kept only as a record of the cloud setup.

---

## Evaluation protocol

The split applies uniformly to **all tiers**. The same 222 test questions and
177 val questions are used everywhere; no tier defines its own split.

| Split | Questions | Source | Permitted use |
|---|---|---|---|
| Train | 709 | BSARD train (80%, seed=42) | Corpus stats, preprocessing decisions |
| Val | 177 | BSARD train (20%, seed=42) | Hyperparameter selection |
| Test | 222 | BSARD official test set | Final benchmarking only |

Split persisted at `evaluation/data/split_ids.json`.

**Metrics** are computed via the `bsard_evaluation` package (the sibling
`RQ3_Autonomous_Evaluation` component — install it as an editable dependency):

```bash
pip install -e "../RQ3_Autonomous_Evaluation"
```

| Metric | k values | Role |
|---|---|---|
| **Recall@k** | 1, 5, 10, 20, 50, **100**, 200, 500 | **R@100 primary** (matches BSARD paper) |
| MRR@k | 10, 100 | Position of first relevant result |
| NDCG@k | 10, 100 | Rank quality |
| MAP@k | 10, 100 | Paper comparability |

---

## Corpus ID mapping (important)

BSARD uses **two article-ID schemes** that must not be mixed:

| Scheme | Source | ID column | Used by |
|---|---|---|---|
| **Local** | `bsard_corpus.db`, `bsard_articles_dedup.parquet` | `article_id` | Ground truth, all experiment scripts |
| **HuggingFace** | upstream BSARD parquet | `id` (= `bsard_id`) | Paper comparisons only |

The same article has different IDs in each scheme — using HF IDs against local
ground truth yields near-zero recall. Build the mapping with:

```bash
python scripts/setup/build_hf_id_mapping.py
```

This writes `hf_to_local_id_mapping.json` and `local_to_hf_id_mapping.json` into
the data root. **All experiment scripts use the local scheme.**

---

## Key data facts

- **Corpus:** ~22,600 unique BSARD articles (deduplicated) + 6,490 distractors.
- **Questions:** 709 train + 177 val + 222 test (val = 20% of BSARD train, seed=42).
- **Test set:** 222 questions, mean 6.18 relevant articles per question.
- **Lexical overlap:** median Jaccard (query ↔ article) = 0.045 — motivates dense retrieval.
- **Citation graph:** 27,712 directed edges.

---

## Citing

If you use this code, please cite it (see [CITATION.cff](CITATION.cff)) **and**
the underlying BSARD dataset (see [DATA_LICENSE.md](DATA_LICENSE.md)).
