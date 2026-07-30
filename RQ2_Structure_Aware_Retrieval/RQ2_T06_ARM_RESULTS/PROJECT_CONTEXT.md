# 06 — ARM RESULTS CONTEXT
## Cross-Arm Presentation Layer for the Thesis (Tables, Figures, Error Analyses, Slides)

---

## 1. PURPOSE

Consolidates the retrieval outputs from T03 (Arm 1 — naive chunking), T04 (Arm 2A — metadata enrichment), and T05 (Arm 2B — PageIndex hierarchical navigation), together with the metrics computed by T07 (Evaluation), into a single project that produces the thesis-grade artefacts:

- Cross-arm comparison tables (CSV + MD) intended for the thesis body and appendix
- Final plots and figures (PNG / SVG) that show per-arm performance curves, win/loss matrices, and per-question scatters
- Categorised error analyses describing where each arm fails relative to the others (false positives, missed articles, ranking inversions)
- Slide assets (figures, single-page summary tables) for thesis presentations

Arm 2C (GraphRAG) is out of scope for RQ2.

## 2. LANE vs T07_EVALUATION

| Concern | Owner |
|---|---|
| Metric formulas (binary + weighted IR), partial-relevance weights | T07 |
| Autonomous LLM-judge eval, cost tracking | T07 |
| Per-arm internal sweeps (e.g. T04 raw / enriched / full / terms / summary) | T04 |
| Cross-arm consolidated tables, figures, slide assets | **T06_ARM_RESULTS** |
| Per-question side-by-side comparisons for the thesis | **T06_ARM_RESULTS** |
| Categorised error analyses across arms | **T06_ARM_RESULTS** |

T07 is the metric authority. T06_ARM_RESULTS consumes T07's metric outputs and the upstream RetrievalResult JSONLs from T03 / T04 / T05; it never recomputes metrics and never writes back into any upstream project's `data/`.

## 3. DIRECTORY STRUCTURE

```
src/arm_results/
├── __init__.py
├── paths.py          # Stem labels, GT paths, T03 config-hash / weights resolution, output dirs
├── loaders.py        # Read T03/T04/T05 RetrievalResult JSONLs + GT (load_all_arms)
├── consolidate.py    # Re-eval each arm via T07 -> long dataframe (evaluate_stem, build_long_dataframe)
├── tables.py         # Render thesis tables (CSV + MD with AUTO-BEGIN sentinels)
├── figures.py        # Matplotlib figures (plot_headline_bars)
├── errors.py         # Per-query side-by-side + categorise failure cases per arm
└── significance.py   # Paired Wilcoxon/t-tests across arms (build_recall_matrix, paired_tests)
```

Sibling layout (in-repo):

```
RQ2_T06_ARM_RESULTS/
├── src/arm_results/
├── scripts/          # run_consolidate.py / run_errors.py / run_significance.py (path-injecting runners)
├── reports/          # Narrative MD reports referencing data/figures and data/tables
├── data/             # Wired to the data root (see §5)
├── requirements.txt
├── pyproject.toml
├── README.md
├── PROJECT_CONTEXT.md
├── SOURCE_MAP.md
└── EVALUATION_METHODOLOGY.md
```

## 4. DEPENDENCIES

- Uses: `shared.utils`, `evaluation` (T07 — for metric output schema), `data_loader` (T02 — for `Article`/`DocumentBundle`)
- Libraries: `pandas`, `numpy`, `scipy` (statistical tests — paired bootstrap, sign tests), `matplotlib`, `seaborn`
- Input:
  - `RQ2_T07_EVALUATION/analysis/*.csv` and `RQ2_T07_EVALUATION/data/reports/`
  - `RQ2_T03_ARM1_NAIVE/data/<doc_id>/configs/<run>/results.jsonl`
  - `RQ2_T04_ARM2_METADATA/data/<doc_id>/results/<variant>/*.jsonl`
  - `RQ2_T05_ARM2_PAGEINDEX/data/<doc_id>/results/<run>/*.jsonl`
- Output: tables, figures, error catalogues under `data/`

## 5. STORAGE

Large artefacts are not committed to git; they live in the companion Hugging Face
dataset `Marios-Paschalidis-Thesis/bsard-rag-thesis-data` (subset `rq2`) and download into the
data root (env `RQ2_DATA_DIR`, default `<repo>/data`). This project's `data/` is
wired to that root via `scripts/setup/link_data.py`:
```
<data root>/RQ2_T06_ARM_RESULTS/
```

| Path | Contents |
|---|---|
| `data/tables/<topic>_<date>.csv` and `.md` | Consolidated cross-arm tables (with `AUTO-BEGIN` sentinels so handwritten narrative can coexist with regenerated tables in the same file) |
| `data/figures/<topic>_<arm>_<date>.png` and `.svg` | Plots ready for thesis insertion |
| `data/per_query/<doc_id>/<qid>.json` | Per-question side-by-side comparison record (rank lists from each arm, ground-truth bsard_ids, hits/misses) |
| `data/error_analysis/<arm>/<category>/<doc_id>__<qid>.md` | One MD per failure case, with the query, the GT articles, the retrieved spans, and a short note on why it failed |

## 6. PUBLIC API

```python
from arm_results.loaders import load_all_arms, load_t03, load_t04_variants, load_t05, load_ground_truth
from arm_results.consolidate import evaluate_stem, build_long_dataframe   # re-eval via T07, build long df
from arm_results.tables import emit_headline, emit_per_stem, emit_full, emit_extended, emit_all
from arm_results.figures import plot_headline_bars
from arm_results.errors import export_per_query, categorise_failures, export_all
from arm_results.significance import build_recall_matrix, paired_tests, run
```

CLI entry points: `python -m arm_results.consolidate` (+ `--stems / --include-ablations /
--effective / --weighted`), and the `scripts/run_{consolidate,errors,significance}.py`
path-injecting runners. All entry points are pure functions: input artefacts in, output
artefacts on disk. No mutating state on T03 / T04 / T05 / T07.

## 7. NON-GOALS

- **No metric computation.** If a metric is missing, file a CN-T07-XXX request against [RQ2_T07_EVALUATION](../RQ2_T07_EVALUATION/), do not add it here.
- **No retrieval invocation.** This project never runs any arm.
- **No write-back.** Outputs land only under this project's `data/`.

## 8. ORCHESTRATOR INTEGRATION

T06_ARM_RESULTS is **not** wired into `RQ2_T00_ORCHESTRATOR/src/orchestrator/run_experiment.py` as a `--methods` arm. It is a **post-pipeline aggregation** project, invoked manually (or via a dedicated CLI such as `python -m arm_results.consolidate`) after the per-PDF pipeline (T02 → T03 / T04 / T05 → T07) has completed for the curated PDF set.

It is also **not** part of the orchestrator's per-PDF status registry — its artefacts are cross-stem, not per-PDF.
