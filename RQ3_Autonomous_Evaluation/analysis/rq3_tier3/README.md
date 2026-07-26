# RQ3 Tier 3 — Autonomous Evaluator Analysis

Workspace for the RQ3 thesis chapter: validation of LLM-based retrieval evaluators
on BSARD, and construction of a calibrated autonomous quality estimator.

The methodology has been reframed from a fixed-weight Autonomous Quality Score
into a two-step research design. No fixed weights are used anywhere in this
plan — any weights that appear are derived from data.

---

## 0. Research questions

- **RQ3a — Validation.** Do UMBRELA, eRAG, and RAGAS-WA recover BSARD ground
  truth retrieval quality on the 48-question stratified subset, and where do
  they fail?
- **RQ3b — Aggregation.** Given RQ3a's findings, can a calibrated estimator
  (predicted nDCG@10 from evaluator scores) outperform the best single
  evaluator and the unweighted-mean baseline, with the *same procedure*
  re-applicable per retriever family?

RAGAS-WB is excluded (failed run, ~0.045 mean across systems). ARES is not run
in the current scope.

**Retriever families in this batch:**

- **Sparse** (12 systems) — re-run with the patched harness to capture
  per-query data. T3 aggregates already exist on disk but per-query data was
  not persisted.
- **Dense** (7 systems, Qwen3 excluded due to GPU constraints) —
  `dense_bge_m3`, `dense_camembert_base`, `dense_camembert_lg`,
  `dense_me5_base`, `dense_me5_large`, `dense_me5_large_concat2x`,
  `dense_mpnet_multi`. None have Tier 3 metrics yet — first run.

Combined panel: **19 systems × 48 queries × 3 evaluators × ~10 docs ≈ 27k
(q,d,system,evaluator) judgments**. Hybrid and agentic families are added
later by re-running the same procedure.

---

## 1. Folder layout

```
analysis/rq3_tier3/
├── README.md                       — this plan; living document
├── data/                           — assembled inputs (created by scripts)
│   ├── sparse_panel.parquet
│   ├── dense_panel.parquet         — appears after dense T3 run
│   ├── hybrid_panel.parquet        — etc.
│   └── strata_summary.json
├── notebooks/                      — exploration, figure prototyping
├── scripts/                        — reproducible re-runs
├── figures/                        — committed final figures (PNG/PDF)
├── tables/                         — committed final tables (TSV/CSV)
└── report/                         — written analysis prose (chapter draft)
    ├── rq3a_validation.md
    ├── rq3b_aggregation.md
    └── threats_to_validity.md
```

Subdirectories are created on demand; this README is the only mandatory
artifact.

---

## 2. Data inventory and gaps

### What exists for the 12 sparse systems on the 48-query subset

For every `output/results/sparse_retrieval/<exp>_test.json` under RQ1, the
`subset_metrics.metrics` block already contains:

- **Aggregate evaluator means** (n=48):
  `T3/umbrela/mean_grade`, `T3/umbrela/mean`, `T3/umbrela/relevant_fraction`,
  `T3/erag/mean`, `T3/erag/grounded_fraction`, `T3/ragas_wa/*`, `T3/ragas_wb/*`
  (WB excluded from analysis).
- **UMBRELA-as-qrels supervised metrics** (`T2-umbrela/...`) — full TREC metric
  suite computed against UMBRELA's 0–3 grades. This *is* the per-system
  UMBRELA signal at any k; the per-(q,d) grade list lives behind it but is
  not currently persisted to disk.
- **Standard supervised metrics on the same 48 queries** under
  `subset_metrics.metrics` (Recall@k, NDCG@10, MAP, etc.) — anchors for RQ3a.
- **Strata labels** per question in `evaluation/data/tier3_subset.json` and
  `evaluation/data/query_strata.json` (under RQ1).

### What exists for the 7 dense systems on the 48-query subset

Nothing T3-related. The full-test-set retrieval result JSONs exist under
`output/results/dense_retrieval/`, but no `subset_metrics` block, no
`T3/*`, no `T2-umbrela/*`. Tier 3 is a first run for dense.

Dense retrieval *itself* has been run on the full 222-question test set;
re-running it on the 48-query subset is essentially free (corpus embeddings
already cached; encoding 48 queries on the seven encoder-only models takes
minutes on CPU).

### What is missing across both families — must be added before RQ3a can be fully run

- Per-query **eRAG** scores (computed in-memory but not saved).
- Per-query **RAGAS-WA** scores (computed in-memory but not saved).
- Persisted per-(q,d) **UMBRELA grades** as a sidecar file (in-memory only
  today; only their downstream T2-umbrela aggregates were saved).
- For dense: `subset_metrics.metrics` block (Recall@k, NDCG@10, MAP, etc.)
  — easy to backfill from the existing full-test-set TREC run via
  `compute_subset_metrics.py` (already a CPU-only script for sparse).

These gaps block (q,d)-level ROC/AUC/calibration analysis for eRAG, block
query-level Spearman analysis for eRAG and RAGAS-WA, and limit per-stratum
inference to system-level only.

### Prerequisite: harness patch

Patch `bsard_evaluation/tier3_autonomous.py`:

1. `run_umbrela` already returns a per-(q,d) qrels dict — persist it through
   the harness to a sidecar file.
2. `run_erag`: also return a per-(q,d) score list; persist alongside UMBRELA.
3. `run_ragas_workaround_a`: extract the per-query column from the RAGAS
   dataframe before aggregating, and persist.

Sidecar path convention:

```
output/results/<family>/tier3_per_query/<exp_id>.json
{
  "ranks":   { "<qid>": ["<doc_id_at_rank_1>", "<doc_id_at_rank_2>", ...] },
  "umbrela": { "<qid>": { "<doc_id>": <grade 0-3>, ... }, ... },
  "erag":    { "<qid>": { "<doc_id>": 0|1, ... }, ... },
  "ragas_wa":{ "<qid>": <float in [0,1]>, ... },
  "hyde":    { "<qid>": "<HyDE response text>", ... }
}
```

`ranks` is the canonical retrieval rank order (1-based via array index).
Panel assembly reads `ranks` to map each (qid, rank) cell to a doc_id, then
joins the evaluator dicts on (qid, doc_id).  Without it, downstream code
would have to rely on Python-dict insertion order — fragile across reruns
and tooling.

### Prerequisite: dense subset prep (CPU-only, no LLM cost)

1. Generalise `scripts/evaluation/compute_subset_metrics.py` to accept a
   `--results-dir` argument (currently hardcoded to sparse). Run it on
   `output/results/dense_retrieval/` to backfill the `subset_metrics` block
   for the 7 dense systems.
2. Add `scripts/evaluation/tier3/run_dense_tier3.py`, parallel to
   `run_sparse_tier3.py` but:
   - Iterates over the 7 non-Qwen3 dense experiments.
   - Builds `DenseRetriever` instances (encoders only — no Qwen3, no GPU).
   - Reuses cached corpus embeddings; encodes only the 48 subset queries.
   - Calls the patched harness with `tier3_components=["umbrela", "erag",
     "ragas_wa"]` (no WB).

### Combined re-run cost

| Family | Systems | Cost / system | Subtotal |
|---|---|---|---|
| Sparse (re-run with persistence) | 12 | ~$0.66 | ~$8.0 |
| Dense (first run) | 7 | ~$0.66 | ~$4.6 |
| **Total** | **19** | — | **~$13** |

gpt-4o-mini pricing, k=10, UMBRELA + eRAG + RAGAS-WA only.

---

## 3. Canonical analysis panel

A single long-format Parquet built by `scripts/assemble_panel.py`, one row per
`(family, system, question_id, doc_id)`:

| column                  | source                                  |
|---|---|
| `family`                | "sparse" / "dense" / "hybrid" / "agentic" |
| `system`                | experiment short name                    |
| `question_id`           | from `tier3_subset.json`                 |
| `doc_id`                | retrieved article id                     |
| `rank`                  | 1-based retrieval rank                   |
| `bsard_relevant`        | 0/1 from BSARD qrels                     |
| `umbrela_grade`         | 0–3 from UMBRELA sidecar                 |
| `erag_score`            | 0/1 from eRAG sidecar                    |
| `ragas_wa_query_score`  | per-query (broadcast over the k docs)    |
| `bm25_score`            | from `query_strata.json`                 |
| `article_count`         | stratum label                            |
| `lex_align`             | stratum label                            |
| `cross_ref`             | stratum label                            |
| `law_code`              | derived from gold articles               |

Plus a system-level summary table (`tables/system_summary.tsv`):

| `family`, `system`, `n_queries`, `R@10`, `R@100`, `nDCG@10`, `MAP`,
  `mean_umbrela`, `mean_erag`, `mean_ragas_wa`,
  `T2-umbrela/NDCG@10`, `T2-umbrela/MAP@100` |

These two artefacts feed every downstream analysis.

---

## 4. RQ3a — Validation analyses

### 4.1 Per-evaluator validity vs BSARD ground truth

Three resolutions, anchored to nDCG@10 (primary) with R@10 and MAP as
robustness anchors:

- **(q,d)-level** (UMBRELA, eRAG): ROC-AUC, average precision, calibration
  curve.
- **Query-level** (all three): Spearman of evaluator-query-aggregate vs
  query-level nDCG@10.
- **System-level** (all three): Kendall τ of system ranking under each
  evaluator vs system ranking under nDCG@10. UMBRELA additionally gets the
  T2-umbrela/NDCG@10 vs T2/NDCG@10 comparison directly.

### 4.2 Mixed-effects model — primary inferential layer

Outcome: per-(q,d) agreement with BSARD label for UMBRELA / eRAG; per-query
Spearman with nDCG@10 for RAGAS-WA.

Fixed effects: `article_count`, `lex_align`, `cross_ref` (main effects only —
no interactions at this n).

Random effects: query (n=48), system (n≈12 per family), evaluator.

Optional fixed effect: `law_code`.

This is where the inferential power lives; the (query × system × evaluator)
panel pools across thousands of judgments while the strata enter as
covariates.

### 4.3 Per-stratum descriptives — with shrinkage

Cell-level point estimates via hierarchical shrinkage (random-effect cell
draws toward axis-level mean). Cell `n` shown beside every estimate. Cells
with `n ≤ 2` flagged as descriptive-only, never inferential.

Bootstrap (cluster-by-query) CIs on axis-level slices — but flag the fragility
of `cross_ref` (39 with / 9 without).

### 4.4 Discriminativeness

For each evaluator, the slope of evaluator-mean-vs-nDCG@10 across the 12
sparse systems. Flat slope → the evaluator does not distinguish good
retrievers from bad ones, regardless of its mean correlation.

### 4.5 Cross-evaluator agreement

Pairwise Spearman / Kendall on system rankings (3 pairs). When evaluators
agree but disagree with BSARD, who is right? Manual case-study panel of
6–10 such queries.

### 4.6 HyDE qualitative inspection — substitute for the lost WA/WB control

Sample 12–15 HyDE responses across strata. Audit each on three axes:

1. Regurgitation — does the HyDE text echo gold-article wording or article
   numbers?
2. Hallucination — does it cite non-existent codes / articles?
3. Specificity — is it a concrete legal answer or generic legal-sounding
   prose?

Reported as a §6 subsection of the validation chapter, with example pairs.
This is the only defensible substitute for the lost WB control.

---

## 5. RQ3b — Calibrated estimator

### 5.1 Validity criteria (stated *before* fitting)

A calibrated estimator is acceptable iff:

- **(a)** Its leave-one-system-out predicted nDCG@10 beats the best single
  evaluator's system-level Kendall τ with nDCG@10.
- **(b)** Its leave-one-system-out predicted nDCG@10 also beats the
  parameter-free unweighted mean of rank-normalised evaluator scores.
- **(c)** Coefficient signs do not flip across retriever families.
- **(d)** Coefficient magnitudes are stable under leave-one-system-out
  (report the spread).

If neither (a) nor (b) holds, the honest RQ3b finding is "report the best
single evaluator; no aggregation is justified." That negative result is a
defensible chapter conclusion.

### 5.2 Construction

- **Inputs**: rank-normalised (within-family) system means of UMBRELA, eRAG,
  RAGAS-WA.
- **Target**: system-level nDCG@10 on the 48-query subset.
- **Form**: linear regression with non-negative weights summing to 1 (for
  interpretability). Unconstrained fit reported as a robustness check.
- **Validation**: leave-one-system-out cross-validation as the primary
  metric; in-sample fit reported only for completeness.
- **Refit per family** — the *procedure* is the cross-family invariant, not
  the weights. Per-family coefficients are reported and compared.

### 5.3 Outputs

- Per-family coefficient table.
- Predicted-vs-actual nDCG@10 scatter, with leave-one-system-out predictions.
- Cross-family weight-stability plot.
- Failure-mode panel: for which (system, family) does the estimator
  systematically under- or over-predict?

---

## 6. Cross-family workflow

Sparse and dense are run in the **same batch** — this is the headline change
from earlier drafts of the plan. Doing both families in one pass means RQ3b
cross-family validity (sign-stability, weight-stability under leave-one-
*family*-out) can be tested from day one rather than deferred until dense
arrives.

The procedure is family-agnostic:

1. Patched harness produces per-query sidecars for each system in the family.
2. `scripts/assemble_panel.py` ingests the family into a `<family>_panel.parquet`.
3. `scripts/fit_calibrated_estimator.py` re-runs RQ3a diagnostics + RQ3b
   calibration on the combined panel (sparse + dense initially; +hybrid,
   +agentic later).
4. Cross-family notebook re-renders the comparison figures.

No analysis logic is family-specific. This is enforced by the panel schema:
adding a new family (hybrid → agentic) is purely a data-pipeline operation.

**Why integrate dense now rather than sequentially:**

- The combined panel has 19 systems, not 12 — meaningful uplift for the
  mixed-effects model's system-level random effect.
- RQ3b's "weights stable across retriever families" criterion needs ≥2
  families to be testable. Sparse-only RQ3b is degenerate.
- Marginal cost is small (~$4.6 added) because dense uses the same harness
  and the same 48 queries.
- Cross-family sign-flips, if they occur, are *the* most interesting RQ3b
  finding — dense and sparse have categorically different failure modes
  (semantic-but-wrong vs. lexically-thin), and whether the evaluators handle
  this transition shapes the whole chapter.

---

## 7. Figures and tables to produce

Headline figures (committed to `figures/`):

- F1: Per-evaluator system-level scatter (4 panels — UMBRELA / eRAG / RAGAS-WA
  + composite — vs nDCG@10 across the 12 systems).
- F2: Mixed-effects coefficient forest plot (stratum effects per evaluator).
- F3: Shrunken cell heatmap (per-cell mean evaluator score, with n shown).
- F4: Cross-evaluator pairwise Spearman matrix.
- F5: Calibrated-estimator LOSO predicted-vs-actual scatter, per family.
- F6: Cross-family weight-stability plot.

Headline tables (committed to `tables/`):

- T1: System-level summary (sparse, then dense, then …).
- T2: ROC-AUC / AP / calibration ECE for UMBRELA and eRAG, per stratum axis.
- T3: Mixed-effects model summary.
- T4: Cross-evaluator Kendall τ matrix on system rankings.
- T5: Calibrated-estimator coefficients (per family, with LOSO CIs).
- T6: Performance gap of calibrated estimator vs best-single-evaluator and
  unweighted-mean baselines.

---

## 8. Threats to validity (chapter section)

- **n=48 is small.** Mitigated by mixed-effects panel inference, not by
  per-cell tests.
- **Same-LLM circularity.** UMBRELA, eRAG (when API), RAGAS-WA all use
  gpt-4o-mini as judge. Plan a sensitivity check with a different judge
  family on at least one evaluator, and ground every claim in a non-LLM
  anchor (BSARD qrels and derived nDCG/R/MAP).
- **HyDE confound in RAGAS-WA, with WB control lost.** Addressed by the
  qualitative HyDE inspection (§4.6) and by triangulation against the other
  two evaluators.
- **Cross_ref axis is imbalanced (39/9).** Flag results on this axis as
  fragile; do not headline them.
- **Calibrated-estimator overfitting risk** with three regressors and 12
  systems. Addressed by non-negative + sum-to-one constraints and LOSO.

---

## 9. Order of work

**One-shot batch (sparse + dense together):**

1. Patch the harness for per-query persistence (UMBRELA qrels, eRAG,
   RAGAS-WA, HyDE responses).
2. Generalise `compute_subset_metrics.py` to accept a `--results-dir` flag;
   run it on `output/results/dense_retrieval/` to backfill `subset_metrics`
   for the 7 dense systems.
3. Add `scripts/evaluation/tier3/run_dense_tier3.py` (parallel to sparse;
   skips Qwen3).
4. Re-run the 12 sparse Tier-3 experiments (≈ $8).
5. Run the 7 dense Tier-3 experiments (≈ $4.6).

**Analysis (combined panel, sparse + dense):**

6. Assemble `sparse_panel.parquet` and `dense_panel.parquet`; concatenate
   into `panel_combined.parquet`.
7. Run RQ3a notebooks 02–04 on the combined panel.
8. Run RQ3b notebook 05 on the combined panel (cross-family weight
   stability is testable now).
9. Draft `report/rq3a_validation.md` and `report/rq3b_aggregation.md`.

**Later families (hybrid, agentic):**

10. Add `<family>_panel.parquet` when those families' Tier 3 runs land.
    Re-run notebooks against the extended combined panel.
11. Update `report/rq3b_aggregation.md` with the cross-family stability
    section once ≥3 families are present.
12. Final pass on `report/threats_to_validity.md`.

---

*Living document. Update as the analysis progresses.*
