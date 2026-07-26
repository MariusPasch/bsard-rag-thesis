# RQ2_T01_SHARED

Shared components: embedding model, LLM wrapper, FAISS/BM25 stores.

---

## Setup

### Step 1 — Create and activate the virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Step 2 — Install dependencies

```powershell
pip install -r requirements.txt
```

---

## Storage

This project is a **pure library** — it contains no large files and does not write to disk directly. No `data/` junction is required.

FAISS indices and BM25 pickles are written by the projects that call this library (T03–T05) and stored in their respective `data/` junctions.

### What lives in the Git repository

- All Python source files (`src/shared/`)
- `requirements.txt`, `.gitignore`, `README.md`, `PROJECT_CONTEXT.md`

---

## Public API

Install as a sibling dependency (required by T02–T08):

```powershell
pip install -e ..\RQ2_T01_SHARED
```

Key imports — all other projects use a subset of these:

```python
from shared.embeddings import EmbeddingModel             # encode texts → float32 vectors (mE5 / BGE-M3)
from shared.llm import LLMClient, LLMResponse            # call local Ollama LLaMA 3.1 8B, temperature=0
from shared.faiss_store import FAISSStore, SearchResult  # build / save / load FAISS IndexFlatIP
from shared.bm25_store import BM25Store                  # build / save / load BM25 index (pickle)
from shared.utils import setup_logger, timer, first_sentence, set_seeds  # logging, timing, text/seed utils
```

These are also re-exported from the package root (`from shared import LLMClient, ...`).
`SearchResult` (from `faiss_store`) is the dense-retrieval hit record; `LLMResponse`
(from `llm`) carries the generated text plus token counts and latency. There is no
`shared.models` module — the cross-arm `RetrievalResult` type that flows from every arm
into T07/T08 is defined in `data_loader.models` (T02), not here.

## Module

```
src/shared/
├── __init__.py
├── embeddings.py      # Embedding model + encoding (mE5-instruct / BGE-M3)
├── llm.py             # LLaMA 3.1 8B via Ollama wrapper
├── faiss_store.py     # FAISS IndexFlatIP management (save/load)
├── bm25_store.py      # BM25 index management (pickle save/load)
└── utils.py           # Logging, timing, text utilities
```

## Part of

RQ2 pipeline — Belgian Statutory Article Retrieval (BSARD) thesis project.
