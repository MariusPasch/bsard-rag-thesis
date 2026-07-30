# RQ2_T07_EVALUATION

Evaluation harness: binary + weighted IR metrics, autonomous LLM judge, cost tracking, reporting.

---

## Setup

### Step 1 — Download the data bundle and wire `data/` (shared with T03)

Large artefacts are not committed to git. They live in the companion Hugging Face dataset `Marios-Paschalidis-Thesis/bsard-rag-thesis-data` (subset `rq2`) and download into the data root (env `RQ2_DATA_DIR`, default `<repo>/data`). T07's `data/` is wired to the **same** `pdf_cache/` root used by `RQ2_T03_ARM1_NAIVE/data/`, so both sub-projects address the same per-PDF folders: T07 reads T03's chunks/raw_text/manifest and writes its `eval/` and `question_projections/` siblings without copies. Run once from the component root:

```powershell
python scripts/download_data.py
python scripts/setup/link_data.py
```

This links `RQ2_T07_EVALUATION/data/` and `RQ2_T03_ARM1_NAIVE/data/` to `<data root>/pdf_cache`.

### Step 2 — Create and activate the virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Step 3 — Install dependencies

```powershell
pip install -r requirements.txt
```

### Step 4 — Install sibling packages

```powershell
pip install -e ..\RQ2_T01_SHARED
pip install -e ..\RQ2_T02_DATA_LOADER
pip install -e ..\RQ2_T03_ARM1_NAIVE         # required: T07 reads T03's cache
```

The RQ3 evaluation service is also required if you call `evaluate(...)`:

```powershell
pip install -e "..\..\RQ3_Autonomous_Evaluation"
```

---

## Storage

### What lives in the Git repository

- All Python source files (`src/evaluation/`)
- `scripts/precompute_eval_weights.py`, `scripts/precompute_question_projection.py`
- `ground_truth/` — ground-truth bsard_ids per query (small, committed)
- `requirements.txt`, `.gitignore`, `README.md`, `PROJECT_CONTEXT.md`

### What lives in the data bundle only (via `data/`)

The `data/` link targets the shared `pdf_cache/` root; `RQ2_T03_ARM1_NAIVE/data/` points at the same target. T07 only writes inside two trees:

```
data/<doc_id>/
├── configs/<arm1_config_hash>/eval/
│   ├── manifest.json                       T07 — eval integrity anchor (§6.2)
│   ├── chunk_bsard_weights.csv             T07 — canonical weights
│   ├── chunk_bsard_weights.pkl             T07 — fast-load mirror
│   └── article_token_ranges.json           T07 — debug: bsard_id → (start, end) tokens
└── question_projections/
    └── <qset_hash>.json                    T07 — per (qset, doc) GT projection (§6.3)
```

`arm1_config_hash` matches T03's `config_hash`; T07's eval folder lives inside it so a chunking-param change moves to a new sibling and old eval data is preserved.

The PDF-level files (`pdf_meta.json`, `raw_text.txt`, `article_spans.json`) and the per-config chunks/indices are owned by T03 — see [RQ2_T03_ARM1_NAIVE/README.md](../RQ2_T03_ARM1_NAIVE/README.md).

### Cache invalidation

| Trigger | Effect |
|---|---|
| `tokenizer_name` change | T07 manifest mismatch → recompute weights |
| BSARD DB regeneration (`bsard_db_fingerprint` change) | T07 `eval/*` and `question_projections/*` recompute |
| T03 manifest moved / deleted (`t03_manifest_resolved_path` no longer resolves) | T07 manifest fails validation → recompute |
| Different question subset | New `qset_hash` → new file, old preserved |
| Unknown `schema_version` in any T07 manifest | Reader rejects loudly; no silent fallback |

A T03 chunking/model change produces a new `arm1_config_hash` directory; T07 simply doesn't have an `eval/` folder there yet. Re-run `precompute_eval_weights.py` with the new hash to populate it.

---

## Project context

See [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) for the full design specification. The shared
PDF-cache layout T07 reads/writes is implemented in `evaluation/cache.py` and T03's
`arm1_naive/cache.py` (and summarised under **Storage** above).

## Public API

Install as a sibling dependency (used by T00 orchestrator):

```powershell
pip install -e ..\RQ2_T07_EVALUATION
```

### Calling from the orchestrator

```python
from evaluation import evaluate

# Cache-aware path (recommended): wires evaluate(...) to read/build chunk-bsard
# weights via the PDF cache. All cache args are kw-only with defaults; pass
# them together to enable, otherwise the legacy chunks+articles path applies.
report = evaluate(
    all_results, ground_truth, config,
    use_cache=True,
    doc_id="1804_03_21_1804032150",
    arm1_config_hash="a3f7c2cd1e90",     # printed by T03's precompute_pdf.py
    t07_cache_root=Path("data"),
    t03_cache_root=Path("../RQ2_T03_ARM1_NAIVE/data"),
    db_conn=sqlite3.connect(db_path),
    pdf_filename="1804_03_21_1804032150.pdf",
)

# Legacy in-memory path (still supported):
report = evaluate(all_results, ground_truth, config, chunks=chunks, articles=articles)
```

### Cached weight precomputation (no eval, just the cache)

```python
from arm1_naive.cache import CacheRoot
from evaluation.cache import T07CacheRoot
from evaluation.weight_precomputer import precompute_weights_cached

weights = precompute_weights_cached(
    doc_id="1804_03_21_1804032150",
    arm1_config_hash="a3f7c2cd1e90",
    t07_cache_root=T07CacheRoot(Path("data")),
    t03_cache_root=CacheRoot(Path("../RQ2_T03_ARM1_NAIVE/data")),
    db_conn=conn,
    tokenizer_name="intfloat/multilingual-e5-large-instruct",
    pdf_filename="1804_03_21_1804032150.pdf",
)
```

### Question→PDF GT projections

```python
from arm1_naive.cache import PdfCache, CacheRoot
from evaluation.projection import build_projection
from evaluation.cache import PdfQuestionProjectionCache, compute_qset_hash, compute_bsard_db_fingerprint, T07CacheRoot

pdf_cache = PdfCache(CacheRoot("../RQ2_T03_ARM1_NAIVE/data"), "1804_03_21_1804032150")
rows = build_projection([192, 193, ...], "1804_03_21_1804032150", pdf_cache, conn)

cache = PdfQuestionProjectionCache(T07CacheRoot("data"), "1804_03_21_1804032150")
cache.save(compute_qset_hash([192, 193, ...]), rows,
           bsard_db_fingerprint=compute_bsard_db_fingerprint(conn))
```

### Per-question chunk scoring (strict / lenient)

```python
from evaluation.weighted_metrics import score_chunks_for_question
from evaluation.weight_precomputer import weights_to_lookup

scores = score_chunks_for_question(
    gt_bsard_ids={920, 1014, 1045},
    weights_lookup=weights_to_lookup(weights),
    chunk_ids=[c.chunk_id for c in chunks],
    mode="strict",          # or "lenient"
)
```

Pure derivation from the cached weight table — never materialised to disk.

### CLI scripts

```powershell
# Populate eval/ for one (PDF, arm1_config_hash):
python scripts/precompute_eval_weights.py --pdf 1804_03_21_1804032150.pdf --arm1-config-hash <hash>

# Project a question set onto one or more PDFs:
python scripts/precompute_question_projection.py `
    --qset ground_truth/bsard_test.json `
    --pdfs 1804_03_21_1804032150.pdf
```

Run either with `--help` for the full flag list.

### Ground truth

Ground-truth `bsard_id`s per query live in `ground_truth/` (committed in this repo, not in the data bundle). One file per BSARD split:

- `ground_truth/bsard_train.json` — 886 train queries, regenerated from `maastrichtlawtech/bsard` (HF).
- `ground_truth/bsard_test.json` — 222 test queries, ditto.
- `ground_truth/schema.json` — schema documentation.
- `ground_truth/question_extraction_status/` — per-question PDF-extraction-status metadata used by the question-subset CLI. See its README.
- `ground_truth/runs/` — gitignored landing pad for per-run curated GT files.

Other projects read GT from here via `config.yaml`:

```yaml
data:
  ground_truth_dir:  ../RQ2_T07_EVALUATION/ground_truth                            # full split GT
  ground_truth_file: ../RQ2_T07_EVALUATION/ground_truth/runs/<run_name>.json       # curated subset
```

Both keys are honoured by `evaluation.load_ground_truth(config)` /
`evaluation.ground_truth_exists(config)` ([ground_truth_loader.py](src/evaluation/ground_truth_loader.py)):

- **`ground_truth_file` wins** when both are set — the curated subset takes precedence over the dir scan.
- **Dir scan** loads every top-level `*.json` and union-merges by `query_id`; `schema.json` and any `_*.json` are skipped as metadata.
- **Shape** (file or merged dir): `{query_id_str: sorted unique [bsard_id_int, ...]}`.
- Missing source raises `FileNotFoundError`; `ground_truth_exists(config)` returns `False` only when neither key resolves.

### Curating a question subset for a run

Build a per-run GT file from the extraction-status metadata:

```powershell
PYTHONPATH=src python -m evaluation.question_subsets build `
    --status exact partial `
    --split test `
    --restrict-per-question-to FOUND `
    --output ground_truth/runs/2026-04-26_my_run.json
```

`inspect` prints bucket counts. Run with no args for the full CLI help.

### Strict / lenient PARTIAL handling

`evaluate_partial_views(...)` runs the evaluation twice on the same retrieval results:

- **strict** — GT keeps only `FOUND` BSARD articles. PARTIAL counts as a miss.
- **lenient** — GT keeps `FOUND` + `PARTIAL`. PARTIAL counts as a hit.

`NOT FOUND` articles are always dropped (not retrievable from the PDF corpus). Questions whose GT becomes empty under each regime are dropped from that view. Returns three EvalReports:

```python
from evaluation import evaluate_partial_views

reports = evaluate_partial_views(all_results, split="test")
# reports["strict"]   — pessimistic baseline
# reports["lenient"]  — optimistic upper bound
# reports["delta"]    — lenient − strict per method × metric (and per stratum)
```

For the test split (n=222): strict pool ≈ 172 questions, lenient pool ≈ 218. Cache args (`use_cache`, `doc_id`, `arm1_config_hash`, …) forward verbatim to both inner `evaluate(...)` calls.

### Bucket stratification

`evaluate(...)` auto-loads the per-question extraction-status bucket (`exact` / `partial` / `mixed` / `not_present`) from `ground_truth/question_extraction_status/questions_by_extraction_status.jsonl` and adds those as additional strata in `report.stratified`, alongside the existing `single_article` / `multi_article` axis. Pass `question_buckets={}` to disable.

## Module

```
src/evaluation/
├── __init__.py
├── adapter.py            # RetrievalResult → TREC qrels/run; bsard_id resolution
├── cache.py              # T07's view of the shared PDF cache (Phase 2)
├── projection.py         # Question→PDF GT projections (Phase 2)
├── metrics.py            # Binary Recall@k, MRR, NDCG
├── weighted_metrics.py   # Weighted Recall@k, MRR, NDCG, Precision@k, score_chunks_for_question
├── weight_precomputer.py # Chunk × BSARD article overlap weights (incl. cached wrapper)
├── comparator.py         # evaluate() / evaluate_partial_views() + significance + stratified analysis
├── autonomous_eval.py    # RAGAS / G-Eval (reference-free)
├── cost_tracker.py       # Token counting and cost aggregation
├── ground_truth_loader.py # load_ground_truth() / ground_truth_exists() (file-or-dir GT resolution)
├── question_subsets.py   # Build curated GT files from extraction-status metadata
├── eval_stamp.py         # Provenance stamp embedded in eval outputs
└── models.py             # EvalReport
```

## Part of

RQ2 pipeline — Belgian Statutory Article Retrieval (BSARD) thesis project.
