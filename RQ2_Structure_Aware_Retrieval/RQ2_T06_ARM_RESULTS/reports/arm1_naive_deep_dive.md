# Arm 1 (Naive chunking) — Retrieved-Article Deep Dive

A per-PDF and aggregate profile of the **retrieved articles** of Arm 1 (T03 naive
RRF over flat chunks), focused on the properties that affect downstream answer
generation, over the curated 5-PDF BSARD subset (725 questions). **Scope: Arm 1
only** — the cross-arm overlap comparison (D2) is out of scope here; the
chunk-vs-whole-unit coverage finding (Group B) is
the Arm-1-specific result that motivates the cross-arm story elsewhere.

All numbers are computed by reading the **persisted one-path artifacts** — no
retrieval was rerun (STYLEGUIDE §6, [EVALUATION_METHODOLOGY.md](../EVALUATION_METHODOLOGY.md)
§4). The article-aggregation path (chunks → bsard articles by max chunk score) is
verified to reproduce the published `T1/R@k` to 10 decimal places. Deterministic;
seed = 42 for the bootstrap CIs. Engine: `src/arm_results/arm1_deep_dive.py`,
runner `scripts/run_arm1_deep_dive.py` (T03 venv).

**Ground truth.** Binary fractional recall against the **full per-PDF GT**
(`runs/t04_<stem>.json`, via `loaders.load_ground_truth`) is **primary**; the
**effective drift-aware GT (`E/`)** is reported **alongside** at per-PDF and
aggregate granularity (read straight from `cross_arm_long.csv`). Per-question
metrics use the full GT (the per-query artifacts hold full-GT rankings); `E/` is
not defined per question here, so the strata are full-GT only.

**Two senses of *k* (labelled throughout).**
- **k = articles** — A1 recall, A2 hit, A5 rank-of-gold, D1 Jaccard: on the
  article-level ranking (chunks aggregated to articles by max score).
- **k = chunks** — A3 precision, A4 distinct, A6 gold/distractor mix, and all of
  Group B/C: on the top-k *chunks the generator actually receives*. This lens only
  exists for a chunk retriever — whole-unit arms (2A/2B) have no chunk budget,
  which is precisely why Groups B/C are Arm-1-specific.

Sources: tables in `data/tables/arm1_*` (CSV + MD), report-ready `.tex` in
`Report/tables/tab_rq2_arm1_*`, figures in `Report/figures/fig_rq2_arm1_*`.

---

## 1. Headline (aggregate over 725 questions)

| | micro (question-weighted) | macro (per-PDF mean) |
|---|---|---|
| Article Recall@5 | 0.357 | 0.357 |
| **Article Recall@10** | **0.472** [0.440, 0.504] | **0.461** [0.335, 0.643] |
| Article Recall@20 | 0.598 | 0.573 |
| Article Recall@100 | 0.860 | 0.855 |
| `E/`Recall@10 (drift-aware) | 0.483 | 0.471 |
| `E/`Recall@100 (drift-aware) | 0.869 | 0.858 |
| Hit@10 | 0.630 | 0.605 |
| Article Precision@10 (over top-10 chunks) | 0.113 | 0.107 |
| Distinct articles in top-10 chunks | 16.8 | 17.0 |
| Median rank of first gold article | 4 | 8.6 |
| Mean content coverage of *retrieved* gold (top-10 chunks) | 0.806 | 0.824 |
| % of retrieved-gold that are fragments (coverage < 0.5) | 18.0 | 16.3 |

(`data/tables/arm1_aggregate_summary.{csv,md}`; 95% bootstrap CI, seed 42, 2000
resamples, over the aggregation unit — questions for micro, the 5 PDFs for macro,
hence the wide macro band.)

**Reading.** Arm 1 puts at least one gold article in the top-10 for 63% of
questions (Hit@10) and recovers 47.2% of all gold articles by rank 10, rising to
86% by rank 100 — most of its competence lives in a deep pool, not the head. The
drift-aware `E/` recall (0.483) is marginally above binary (0.472): on the curated
set (drift doc 2 excluded, see [project_doc2_gt_drift]) almost all gold reaches the
PDF, so the two GT choices nearly coincide here. **Precision is low by
construction** — the top-10 chunks surface ~17 distinct articles, of which ~1–2 are
gold (P@10 ≈ 0.11), i.e. ~89% of the article-context the generator sees is
distractor material (§4, Group A6).

---

## 2. Per-PDF profile (Group A)

One row per PDF, ordered by question count. Recall/Hit on **k = articles**;
Precision/Distinct on **k = chunks**. Full table: `data/tables/arm1_per_pdf_summary.md`;
report-ready `Report/tables/tab_rq2_arm1_per_pdf.tex`.

| PDF (law type) | n | R@5 | R@10 | `E/`R@10 | Hit@10 | R@20 | R@100 | P@10 | distinct@10 | med. rank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Code Civil | 252 | 0.340 | 0.471 | 0.478 | 0.647 | 0.650 | 0.935 | 0.111 | 16.7 | 5 |
| Code Judiciaire (larger) | 204 | 0.245 | 0.343 | 0.349 | 0.554 | 0.472 | 0.733 | 0.126 | 18.3 | 5 |
| Code du Logement | 133 | 0.637 | 0.801 | 0.836 | 0.872 | 0.833 | 0.936 | 0.125 | 12.9 | 3 |
| Code Judiciaire (smaller) | 71 | 0.287 | 0.396 | 0.395 | 0.535 | 0.534 | 0.902 | 0.075 | 16.5 | 10 |
| Code Pénal | 65 | 0.275 | 0.296 | 0.296 | 0.415 | 0.376 | 0.767 | 0.096 | 20.7 | 20 |

**Reading.** A wide spread by law type. **Code du Logement** is the easy case
(R@10 0.801, median first gold at rank 3) — short, well-structured articles whose
text matches question vocabulary. **Code Pénal** is the hardest (R@10 0.296, median
first gold at rank 20, R@100 only 0.767 — a third of its gold never enters even the
200-pool), and the larger **Code Judiciaire** is similar (R@10 0.343). `E/`R@10
tracks binary R@10 within ±0.04 on every PDF, confirming drift is not the driver of
the per-PDF spread on this curated set. The binary→`E/` gap is largest on Code des
Code du Logement (0.801 → 0.836), the only PDF where a non-trivial slice of gold sits just
outside the extracted text.

---

## 3. Content coverage — the Arm-1-specific finding (Group B)

For each *retrieved* gold article, coverage@10 = the capped sum of chunk→article
token-overlap weights over the top-10 chunks = the **fraction of that article's
text the generator actually receives**. A whole-unit retriever (Arm 2A/2B) scores
≈1.0 on every hit by construction; for a chunk retriever it is a distribution.
(`data/tables/arm1_coverage_per_gold_article.{csv,md}`, 2874 gold-article rows;
per-PDF `Report/tables/tab_rq2_arm1_coverage.tex`. The coverage histogram
`fig_rq2_arm1_coverage_dist` is not included; the finding below is kept in
narrative form.)

- **Mean coverage of retrieved gold = 0.806** (micro). The distribution is
  **strongly bimodal**: a large spike at ≈1.0 (whole-article retrieval) and a thin,
  near-uniform tail of partial coverage.
- **64.0% of retrieved gold is near-complete** (coverage ≥ 0.9) — when Arm 1 finds
  an article it usually delivers most of it.
- **18.0% of retrieved gold is a fragment** (coverage < 0.5) — these are the
  partial-context cases that hurt generation: the article *counts* as a binary hit
  yet less than half its text is in context. This is **B2**: binary article recall
  systematically overstates the usable context Arm 1 supplies.

Per PDF (`tab_rq2_arm1_coverage.tex`), the fragment rate ranges 11–20% (lowest on
Code Pénal 11.1%, highest on Code Judiciaire-larger 19.6%); near-complete coverage
ranges 58–75%. Contrasting binary article recall@10 against mean
coverage-of-retrieved-gold per PDF (figure `fig_rq2_arm1_recall_vs_coverage`
not included), the two diverge most where binary recall is
already low (Pénal, Judiciaire), i.e. the hard PDFs lose context twice — fewer gold
articles retrieved *and* a larger fraction of those only partially.

---

## 4. Context efficiency, noise and rank depth (Groups A3/A4/A6, C)

- **A6 — gold/distractor mix (top-10 chunks).** The generator's article context is
  ~17 distinct articles, of which a mean 0.63 (single-article questions) / 2.57
  (multi-article questions) are gold; the remaining ~15–17 are distractors.
  Precision@10 ≈ 0.11 quantifies the pollution: ≈89% of the distinct articles in
  context are not gold.
- **C1 — chunks per distinct article@10 ≈ 0.70** (micro). This is **below 1**: the
  top-10 chunks surface ~17 distinct articles, so each chunk overlaps ~1.7 articles
  on average. The redundancy at k=10 is therefore **not duplicate chunks of one
  article** but the opposite — short statutory articles packed several-per-chunk.
  (Duplication of a single article across chunks does occur and is what drives the
  fragment tail in §3, but it does not dominate the k=10 budget.)
- **C2 — chunks consumed to first-touch the whole gold set** (median): 2 (Code du Logement)
  → 10 (Civil) → 17 (Judiciaire-larger) → 20 (Pénal) → 28 (Judiciaire-smaller). On
  the harder PDFs a generator would need to ingest 20–28 chunks to see every gold
  article at least once — well beyond a typical top-10 context window.
- **A5 — rank of first gold article.** Median 4 (micro), but a long tail: the
  first-gold-rank CDF (figure `fig_rq2_arm1_rank_of_gold_cdf` not included)
  plateaus at ≈0.94 by rank 100 — ~6% of questions never surface
  a gold article in the 200-pool at all. Half the questions hit a gold article by
  rank 4; reaching 0.8 of questions requires going to ~rank 35.
- **D1 — overlap with gold.** Mean Jaccard(top-10 articles, gold) = 0.105 (micro):
  the top-10 article set and the gold set overlap weakly, dominated by the large
  distinct-article count in the denominator.

---

## 5. Stratified analysis (full GT)

Single- vs multi-article questions (315 / 410, from GT cardinality) and extraction
status of the gold articles (FOUND / PARTIAL — there are **no NOT_FOUND or UNKNOWN**
questions on the curated set, since drift doc 2 is excluded and the GT is curated).
(`data/tables/arm1_strata_summary.md`, `Report/tables/tab_rq2_arm1_strata.tex`.
The Arm-1 strata figure is superseded by the cross-arm
`fig_rq2_recall_by_cardinality_byarm`.)

| stratum | level | n | R@5 | R@10 | R@20 | Hit@10 | med. rank | P@10 | coverage@10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cardinality | single | 315 | 0.432 | 0.546 | 0.660 | 0.546 | 8.5 | 0.046 | 0.548 |
| cardinality | multi | 410 | 0.299 | 0.416 | 0.550 | 0.695 | 4.0 | 0.164 | 0.418 |
| extraction | FOUND | 368 | 0.340 | 0.451 | 0.601 | 0.543 | 8.0 | 0.089 | 0.469 |
| extraction | PARTIAL | 357 | 0.375 | 0.494 | 0.594 | 0.720 | 3.0 | 0.137 | 0.480 |

**Reading.**
- **Multi-article questions have lower fractional recall** (0.416 vs 0.546) — they
  require *all* gold articles, and Arm 1 rarely retrieves the complete set — **but
  higher Hit@10** (0.695 vs 0.546), since any one of several gold articles can land
  in the head. Their P@10 is higher (0.164 vs 0.046) simply because more of the
  context can be gold. Per-question mean coverage is lower for multi (0.418 vs
  0.548): more gold articles means more opportunities for a fragment.
- **PARTIAL-extraction questions do *not* under-perform FOUND** — R@10 is actually
  slightly higher (0.494 vs 0.451) and coverage is comparable (0.480 vs 0.469). The
  ANALYSIS_PLAN hypothesis that "Arm-1 coverage gaps concentrate in PARTIAL/low-
  cosine extraction" **is not supported for Arm 1** on this set; PARTIAL questions
  are disproportionately multi-article, which lifts Hit@10 and precision and masks
  any extraction penalty. (This is an Arm-1 statement only — a node-index arm could
  still differ; that is a cross-arm question, out of scope here.)

---

## 6. Summary of findings

1. Arm 1 recovers 47% of gold articles by rank 10 and 86% by rank 100 — its
   strength is pool depth, not head precision (Hit@10 0.63, P@10 0.11).
2. Performance is law-type dependent: Code du Logement easy (R@10 0.80), Code
   Pénal hard (0.30); `E/` drift-aware GT tracks binary within ±0.04, so drift is
   not the per-PDF driver on the curated set.
3. **The chunk-specific liability: 18% of retrieved gold articles arrive as
   fragments (<half their text).** Binary article recall overstates usable context;
   coverage-of-retrieved-gold averages 0.806 with a bimodal near-complete/fragment
   shape. This is the structural reason whole-unit arms (2A/2B) differ — they cannot
   produce a fragment hit — and is the key Arm-1 input to the cross-arm story.
4. Multi-article questions trade fractional recall for hit rate; PARTIAL extraction
   does not depress Arm-1 recall on this set.

---

### Artifact index

| Artifact | Path |
|---|---|
| Per-question metrics (725 rows) | `data/tables/arm1_per_question.{csv,md}` |
| Per-gold-article coverage (2874 rows) | `data/tables/arm1_coverage_per_gold_article.{csv,md}` |
| Per-PDF summary | `data/tables/arm1_per_pdf_summary.{csv,md}` · `Report/tables/tab_rq2_arm1_per_pdf.tex` |
| Aggregate (micro+macro) | `data/tables/arm1_aggregate_summary.{csv,md}` |
| Coverage / efficiency per PDF | `Report/tables/tab_rq2_arm1_coverage.tex` |
| Strata | `data/tables/arm1_strata_summary.{csv,md}` · `Report/tables/tab_rq2_arm1_strata.tex` |
| ~~Coverage distribution~~ | *not included* |
| ~~Rank-of-gold CDF~~ | *not included* |
| Precision per PDF → cross-arm | `Report/figures/fig_rq2_precision_ceiling_per_pdf.{pdf,png}` |
| ~~Recall vs coverage per PDF~~ | *not included* |
| Cardinality strata → cross-arm | `Report/figures/fig_rq2_recall_by_cardinality_byarm.{pdf,png}` |
