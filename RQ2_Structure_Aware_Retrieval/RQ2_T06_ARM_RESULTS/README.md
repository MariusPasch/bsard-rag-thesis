# RQ2_T06_ARM_RESULTS

Cross-arm results consolidation: tables, figures, error analyses, and slide assets aggregating T03 / T04 / T05 retrieval outputs and T07 metrics. Presentation layer only — no metric computation.

---

## Setup

### Step 1 — Download the data bundle and wire `data/`

Large artefacts are not committed to git. They live in the companion Hugging Face dataset `Marios-Paschalidis-Thesis/bsard-rag-thesis-data` (subset `rq2`) and download into the data root (env `RQ2_DATA_DIR`, default `<repo>/data`). Run once from the component root:

```powershell
python scripts/download_data.py
python scripts/setup/link_data.py
```

After this step, `data\tables\`, `data\figures\`, `data\per_query\`, and `data\error_analysis\` should be accessible (linked to `<data root>/RQ2_T06_ARM_RESULTS`).

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
pip install -e ..\RQ2_T07_EVALUATION
```

---

## Storage

### What lives in the Git repository

- All Python source files (`src/arm_results/`) and export scripts (`scripts/`)
- `requirements.txt`, `.gitignore`, `README.md`, `PROJECT_CONTEXT.md`, `SOURCE_MAP.md`, `EVALUATION_METHODOLOGY.md`
- Narrative reports (`reports/`)

### What lives in the data bundle only (via `data/`)

This project's `data/` is wired (via `scripts/setup/link_data.py`) to the data root:
```
<data root>/RQ2_T06_ARM_RESULTS/
```

| Path | Contents | Reason |
|---|---|---|
| `data/tables/` | Consolidated cross-arm metric tables (CSV + MD) summarising T07 outputs | Auto-generated |
| `data/figures/` | Plots and thesis figures (PNG / SVG) across arms | Auto-generated, referenced in thesis |
| `data/per_query/` | Per-question side-by-side comparisons across T03 / T04 / T05 | Auto-generated |
| `data/error_analysis/` | Categorised failure cases per arm with example chunks/articles | Auto-generated |

> Consumes outputs only — never writes back into T03 / T04 / T05 / T07 data directories.

---

## Cross-arm consolidation (usage)

Re-evaluates T03, T04 (every persisted variant) and T05 through **one** path —
T07's `evaluation.comparator.evaluate` (tiers 1+2) — against the **single**
per-PDF curated ground truth (`RQ2_T07_EVALUATION/ground_truth/runs/t04_<stem>.json`).
This is necessary because each arm originally produced its numbers through a
different evaluation path / GT definition (see [EVALUATION_METHODOLOGY.md](EVALUATION_METHODOLOGY.md)).
No retrieval is rerun and no metric is recomputed here — only loaded, scored via
T07, and reshaped.

```powershell
# after the venv + sibling installs above (T07 transitively pulls faiss/torch):
python -m arm_results.consolidate                       # all 5 curated PDFs
python -m arm_results.consolidate --stems 1804_03_21_1804032150
python -m arm_results.consolidate --include-ablations    # also load T04 boost-ablation runs
python -m arm_results.consolidate --effective --weighted # add drift-aware E/ + Arm-1 W/ metrics

# or, without `pip install -e` of the siblings (path-injecting runner):
python scripts/run_consolidate.py [--effective --weighted]
python scripts/run_errors.py                            # per-query side-by-side + failure summary
python scripts/run_significance.py                      # paired Wilcoxon/t-tests across arms
```

The narrative synthesis lives in [reports/cross_arm_analysis.md](reports/cross_arm_analysis.md).

`--effective` adds drift-aware `E/` metrics (recall denominator = GT locatable in the
PDF, via T07's `projection` + the BSARD db). `--weighted` adds Arm-1 `W/` partial-relevance
metrics for T03 from T07's cached chunk-overlap weights — **currently unusable** due to
incomplete weight coverage (see CN-T07-010 / [EVALUATION_METHODOLOGY.md](EVALUATION_METHODOLOGY.md)).
Both land in `cross_arm_extended.md` + `cross_arm_full.csv`.

Error analysis (`scripts/run_errors.py`) writes `data/per_query/<stem>.json` (each
question's GT + every arm's resolved ranking, hit@k, first-hit rank) and
`data/error_analysis/failure_summary.csv` (hit / near-miss / missed counts per stem × method).

Outputs (under `data/tables/`):

| File | Contents |
|---|---|
| `cross_arm_long.csv` | one row per stem × method × metric (raw T07 keys) |
| `cross_arm_headline.{csv,md}` | R@10 / R@100 / MRR@10 / nDCG@10 per method — macro (per-PDF mean) and micro (question-weighted) |
| `cross_arm_per_stem.md` | the same headline metrics, one block per PDF |
| `cross_arm_full.csv` | every raw T07 metric key, stem × method wide |
| `cross_arm_costs.json` | per-method cost aggregates (T05 LLM calls / tokens / latency) |

Figures: `arm_results.figures.plot_headline_bars(long_df)`.
Per-question side-by-side + failure buckets: `arm_results.errors.export_per_query(stem)` / `categorise_failures(stem, method)`.

**Sources:** [SOURCE_MAP.md](SOURCE_MAP.md) maps where every arm's results, parameters and prior analysis live.

## Project context

See [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) for full design specification.

## Scope (what this project owns vs. what it does NOT)

**Owns**
- Consolidated cross-arm comparison tables for the thesis (one row per (arm × stem × metric × k))
- Final thesis figures (per-arm curves, win/loss matrices, per-query scatter plots)
- Categorised error analyses across arms (false positives, missed articles, ranking inversions)
- Slide assets and reproducible export scripts

**Does NOT own**
- Metric computation — that is [RQ2_T07_EVALUATION](../RQ2_T07_EVALUATION/)
- Retrieval logic — owned by T03 / T04 / T05
- Per-arm internal comparisons (e.g. T04 variants `raw / enriched / full / terms`) — those live in T04's own `analysis/`

Consumed inputs come from:
- `RQ2_T07_EVALUATION/analysis/*.csv` and `RQ2_T07_EVALUATION/data/reports/`
- `RQ2_T03_ARM1_NAIVE/data/<doc_id>/configs/<run>/results.jsonl`
- `RQ2_T04_ARM2_METADATA/data/<doc_id>/results/<variant>/*.jsonl`
- `RQ2_T05_ARM2_PAGEINDEX/data/<doc_id>/results/<run>/*.jsonl`

## Module

```
src/arm_results/
├── __init__.py
├── paths.py          # stem labels, GT paths, T03 config-hash/weights resolution
├── loaders.py        # load_t03 / load_t04_variants / load_t05 / load_all_arms / load_ground_truth
├── consolidate.py    # evaluate_stem + build_long_dataframe (re-eval via T07); CLI: python -m arm_results.consolidate
├── tables.py         # emit_headline / emit_per_stem / emit_full / emit_extended / emit_all
├── figures.py        # plot_headline_bars
├── errors.py         # export_per_query / categorise_failures / export_all (CLI via run_errors.py)
└── significance.py   # build_recall_matrix / paired_tests / run (CLI via run_significance.py)
```

Driver scripts in `scripts/`: `run_consolidate.py`, `run_errors.py`, `run_significance.py`
(path-injecting runners that work without `pip install -e` of the siblings).

## Part of

RQ2 pipeline — Belgian Statutory Article Retrieval (BSARD) thesis project.
