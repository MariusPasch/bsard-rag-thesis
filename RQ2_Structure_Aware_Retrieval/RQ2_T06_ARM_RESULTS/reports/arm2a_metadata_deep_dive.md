# Arm 2A (Metadata enrichment) — Retrieved-Article Deep Dive

A per-PDF and aggregate profile of the **retrieved articles** of Arm 2A (T04
metadata-enriched index over AzureDI nodes / article units), focused on the
properties that affect downstream answer generation, over the curated 5-PDF BSARD
subset (725 questions). **Scope: Arm 2A only** — every metric is defined
arm-agnostically and computed by the *same* `deep_dive_common` primitives the
Arm-1 dive uses, so a later Arm-2A-vs-Arm-1 (or vs Arm-2B) synthesis is a simple
join; the cross-arm comparison itself (D2) is deferred. Latency is excluded
(Arm 2A query-time LLM calls = 0).

**Variants.** Three, with the boost-inert duplicates dropped (the inert query-time
boost makes enriched_node/full_node ≡ raw_node and full_article ≡ raw_article,
byte-identical — see `reports/cross_arm_analysis.md §2`, `analysis/boost_ablation_v4.md`):
- **summary_node** — the **canonical** Arm 2A (primary): AzureDI nodes embedded
  with their gpt-4o English summary.
- **raw_node** — identical node set, raw FR text embedded; isolates the
  *index-time enrichment lift*.
- **raw_article** — the whole-article unit; the **node-vs-article granularity**
  contrast. A unit is 1:1 with an article, so coverage ≡ 1.0 on a hit.

All numbers are read from the **persisted one-path artifacts** — no retrieval,
indexing or linking was rerun (STYLEGUIDE §6, [EVALUATION_METHODOLOGY.md](../EVALUATION_METHODOLOGY.md)
§4). The article-aggregation path (nodes → bsard articles by *max node score*) is
asserted to reproduce the published `T1/R@k`: **52 of 60 (variant, stem, k) cells
match to ≤1e-6**; the 8 exceptions are node-variant cells at k∈{5,10,20} that drift
≤7.8e-3 because the node FAISS/BM25 indexes were resynced *after* the consolidated
frame was built (a couple of articles crossing a rank-k boundary among many
near-tied node scores). The depth-stable anchors R@5/R@100 match to ≤2.7e-4, and
**raw_article reproduces every cell exactly**, confirming the aggregation logic is
identical — the drift is an artifact-vintage effect, immaterial to every finding.
Deterministic; seed = 42 for the bootstrap CIs. Engine
`src/arm_results/arm2a_deep_dive.py` + shared `src/arm_results/deep_dive_common.py`,
runner `scripts/run_arm2a_deep_dive.py` (T03 venv).

**Ground truth.** Binary fractional recall against the **full per-PDF GT**
(`runs/t04_<stem>.json`) is **primary**; the **effective drift-aware GT (`E/`)** is
reported **alongside** at per-PDF and aggregate granularity (read straight from
`cross_arm_long.csv`). Per-question strata use the full GT.

**Two senses of *k* (labelled throughout).**
- **k = articles** — A1 recall, A2 hit, A5 rank-of-gold, D1 Jaccard: on the
  article ranking (nodes aggregated to articles by max node score).
- **k = nodes** — A3 article-context precision, A4 distinct units, A6
  gold/distractor mix and all of Group B coverage: on the top-k *nodes the
  generator actually receives*. For raw_article a "node" already is an article,
  so these collapse onto the article level by design.

**Two coverage senses (labelled explicitly, per the Arm-1 lesson).**
*Coverage of retrieved gold* (conditional on the article being touched in the
top-10 nodes) and *coverage over all gold* (unconditional; missed = 0) are
reported separately and never conflated. The **primary** weight is the e5-token
analogue of Arm 1's char-overlap weight — `w_db_overlap(node, A) = |token-multiset
overlap of the node's FR text with the BSARD-DB article text| / |DB article
tokens|` — so coverage is directly comparable to Arm 1's 0.806 / 18% / 64%. Because
AzureDI nodes share no char-coordinate system with the DB text (unlike T03's
chunks), the mechanism is token-multiset overlap rather than char-span overlap;
the metric still means *"fraction of the DB article's text the node carries."* The
self-contained `w_node_share` (= node tokens / Σ tokens of the article's nodes,
which sums to 1.0 per article by construction) is reported alongside and tracks
the DB-overlap weight closely (0.708 vs 0.724 conditional, micro).

---

## 1. Headline (aggregate over 725 questions, summary_node — canonical)

| | micro (question-weighted) | macro (per-PDF mean) |
|---|---|---|
| Article Recall@5 | 0.390 | 0.370 |
| **Article Recall@10** | **0.482** [0.452, 0.512] | **0.457** [0.311, 0.655] |
| Article Recall@20 | 0.595 | 0.585 |
| Article Recall@100 | 0.807 | 0.805 |
| `E/`Recall@10 (drift-aware) | 0.472 | 0.443 |
| `E/`Recall@100 (drift-aware) | 0.763 | 0.744 |
| Hit@10 | 0.691 | 0.652 |
| Article precision@10 (gold ∕ distinct articles in top-10 nodes) | 0.147 | 0.137 |
| Distinct articles in top-10 nodes | 8.18 | 8.04 |
| Median rank of first gold article | 2 | 4.2 |
| **Coverage of *retrieved* gold** (conditional, top-10 nodes, DB-overlap) | **0.724** | 0.719 |
| Coverage over *all* gold (unconditional, top-10 nodes) | 0.204 | 0.225 |
| % of retrieved-gold that are fragments (coverage < 0.5) | 28.6 | 26.0 |
| % of retrieved-gold near-complete (coverage ≥ 0.9) | 54.6 | 57.6 |

(`data/tables/arm2a_aggregate_summary.{csv,md}`; 95% bootstrap CI, seed 42, 2000
resamples over the aggregation unit — questions for micro, the 5 PDFs for macro.)

**Reading.** Arm 2A's metadata node index puts at least one gold article in the
top-10 for **69%** of questions (Hit@10) and recovers **48%** of all gold by rank
10, with the **first gold article at median rank 2**. Its top-10 context is
**clean**: only ~8 distinct articles, of which ~1–2 are gold (P@10 ≈ 0.15). But
the node unit **fragments the gold it finds**: when a gold article is retrieved,
the top-10 nodes deliver on average **72% of its text**, and **28.6% of retrieved
gold arrives as a fragment** (<half its text). The conditional/unconditional gap is
large (0.724 vs 0.204): most gold articles are either missed or only partially in
the top-10 node budget.

---

## 2. Per-PDF profile (summary_node, Group A + coverage)

One row per PDF, ordered by question count. Recall/Hit on **k = articles**;
precision/distinct/coverage on **k = nodes**. Full table:
`data/tables/arm2a_per_pdf_summary.{csv,md}`; report-ready
`Report/tables/tab_rq2_arm2a_per_pdf.tex`.

| PDF (law type) | n | R@10 | `E/`R@10 | Hit@10 | R@100 | P@10 | distinct@10 | med. rank | coverage (touched) | % frag |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Code Civil | 252 | 0.514 | 0.508 | 0.710 | 0.877 | 0.139 | 8.79 | 3 | 0.764 | 26.8 |
| Code Judiciaire (larger) | 204 | 0.346 | 0.318 | 0.662 | 0.654 | 0.150 | 8.40 | 2 | 0.709 | 28.3 |
| Code du Logement | 133 | 0.805 | 0.831 | 0.895 | 0.937 | 0.213 | 6.77 | 2 | **0.641** | **36.7** |
| Code Judiciaire (smaller) | 71 | 0.383 | 0.368 | 0.577 | 0.850 | 0.090 | 7.48 | 7 | 0.728 | 25.0 |
| Code Pénal | 65 | 0.240 | 0.192 | 0.415 | 0.706 | 0.093 | 8.74 | 7 | **0.865** | **13.3** |

**Reading.** The recall ordering matches Arm 1 (Code du Logement easiest at R@10
0.805, Code Pénal hardest at 0.240), but **coverage runs in the *opposite*
direction**: the *easy* PDF (Code du Logement) has the **worst** coverage (0.641) and
**most** fragmentation (36.7%), while the *hard* PDF (Pénal) has the **best**
coverage (0.865) and least fragmentation (13.3%). Code du Logement' articles are
short and split into many small nodes (~4 nodes/article, 257 articles over 1054
nodes), so the index retrieves them readily (high recall) yet each retrieved node
carries only a slice — high recall, low coverage. This **inverts Arm 1's "hard PDFs
lose context twice"**: for the node index it is the *finely-fragmented* PDF that
loses context, regardless of difficulty.
(The per-PDF recall-vs-coverage figure `fig_rq2_arm2a_recall_vs_coverage` is not
included; the contrast is kept in narrative form.)

---

## 3. Content coverage — the node-fragmentation finding (Group B)

For each *retrieved* gold article, coverage@10 = the capped sum of node→article
token-overlap weights over the top-10 nodes = the fraction of that article's text
the generator receives. (`data/tables/arm2a_coverage_per_gold_article.{csv,md}`;
node→article weight table `data/tables/arm2a_node_bsard_weights.csv`, 4894 rows.
The coverage histogram `fig_rq2_arm2a_coverage_dist` is not included; the finding
below is kept in narrative form.)

- **Coverage of retrieved gold = 0.724** (micro, DB-overlap; 0.708 under the
  node-share weight — the two agree to ~0.02). The distribution is **bimodal** like
  Arm 1 (a spike near 1.0 + a partial tail), but **the fragment tail is heavier**.
- **54.6% of retrieved gold is near-complete** (coverage ≥ 0.9) vs Arm 1's 64.0%.
- **28.6% of retrieved gold is a fragment** (coverage < 0.5) vs Arm 1's 18.0% —
  **the node unit fragments more than Arm 1's chunks.** AzureDI nodes are
  *sub-article* structural units (mean ≈0.99 articles/node, but up to 11 nodes per
  article — e.g. 958 nodes ↔ 429 articles on Code Civil), finer-grained than a
  token-window chunk, so a single top-10 node more often delivers only part of a
  gold article. This is the structural cost of the granularity that lifts recall
  and rank (§6).
- **Weight-table sanity.** `w_node_share` sums to exactly 1.0 per article by
  construction. The DB-overlap weight sums to a per-article mean of ≈1.0 (median
  0.98; 82% of indexed articles ≥0.9), confirming the node set covers essentially
  all of each DB article's text — the fragmentation is a *retrieval-budget* effect
  (only some of an article's nodes reach the top-10), not an indexing gap.

---

## 4. Context efficiency, noise and rank depth (Groups A3/A4/A6)

- **A6 — gold/distractor mix (top-10 nodes).** The generator's article context is
  ~8 distinct articles (vs Arm 1's ~17), of which ~1–2 are gold; P@10 ≈ 0.15 (vs
  Arm 1's 0.11). The node index delivers a **smaller, cleaner** context than Arm 1's
  chunk pool — fewer distinct articles per unit because a node maps to ≤1 article,
  whereas a token-window chunk packs several short statutory articles.
- **A5 — rank of first gold.** Median 2 (micro) — the metadata index ranks gold
  **higher than Arm 1** (median 4). Per PDF the median first-gold rank is 2–3 on the
  easier codes, 7 on the two hardest (Pénal, Judiciaire-smaller).
- **A4 — distinct units.** 8.18 distinct articles in the top-10 nodes (summary_node)
  vs 10.0 for raw_article (1 article per unit by construction).
- **D1 — overlap with gold.** Mean Jaccard(top-10 articles, gold) = 0.099 (micro),
  dominated by the distinct-article denominator.

---

## 5. Stratified analysis (full GT)

Single- vs multi-article (315 / 410, from GT cardinality) and extraction status of
the gold (FOUND / PARTIAL — **no NOT_FOUND or UNKNOWN** on the curated set; same
bucketing rule as Arm 1: a question is PARTIAL if ≥1 gold article is PARTIAL).
`coverage@10` here is the per-question **unconditional** mean (missed = 0).
(`data/tables/arm2a_strata_summary.{csv,md}`, `Report/tables/tab_rq2_arm2a_strata`.
The Arm-2A strata figure is superseded by the cross-arm
`fig_rq2_arm2a_recall_curve_cardinality` — recall@k over pool depth, single vs multi.)

| variant | stratum | level | n | R@5 | R@10 | R@20 | Hit@10 | med. rank | P@10 | coverage@10 (uncond.) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| summary_node | cardinality | single | 315 | 0.486 | 0.575 | 0.683 | 0.575 | 3 | 0.076 | 0.394 |
| summary_node | cardinality | multi | 410 | 0.316 | 0.412 | 0.528 | 0.780 | 2 | 0.201 | 0.263 |
| summary_node | extraction | FOUND | 368 | 0.372 | 0.471 | 0.591 | 0.601 | 3 | 0.113 | 0.322 |
| summary_node | extraction | PARTIAL | 357 | 0.408 | 0.494 | 0.599 | 0.784 | 2 | 0.181 | 0.317 |

**Reading.**
- **Multi-article questions** have lower fractional recall (0.412 vs 0.575 — they
  require *all* gold) but higher Hit@10 (0.780 vs 0.575 — any one of several can land
  in the head), exactly as in Arm 1.
- **PARTIAL-extraction questions do *not* under-perform FOUND** — R@10 is slightly
  *higher* (0.494 vs 0.471) and unconditional coverage is essentially equal (0.317
  vs 0.322). The **ANALYSIS_PLAN-C2 hypothesis** — that "coverage gaps concentrate in
  PARTIAL / low-cosine extraction" — was framed *about this arm*, and **it is not
  supported even here**: PARTIAL questions are disproportionately multi-article,
  which lifts their Hit@10 and precision and masks any extraction penalty. The same
  pattern holds for raw_node and raw_article (`arm2a_strata_summary`).

---

## 6. Failure lens 1 — entire misses and where coverage gaps come from

For each gold article *not* retrieved within the persisted pool (top-100; note T04
persists **top-100** vs Arm 1's top-200, so "beyond pool" is assessed at a shallower
depth) we classify the miss as **ABSENT** (no node maps to that bsard_id — an
extraction/linking gap) or **BEYOND_POOL** (present in the index but ranked > 100).
(`data/tables/arm2a_missed_gold.{csv,md}`, summary `arm2a_miss_summary.{csv,md}`.
The figure `fig_rq2_arm2a_entire_miss_extraction` is not included — it would be
degenerate (the ABSENT category is ~0); the finding is kept below.)

| variant | % gold missed (top-100) | of which ABSENT | % entire-miss questions | missed FOUND | missed PARTIAL |
|---|---:|---:|---:|---:|---:|
| summary_node | 30.3 | **0.23%** (2 of 872) | 8.1 | 732 | 140 |
| raw_node | 38.8 | 0.18% (2 of 1114) | 9.8 | 937 | 177 |
| raw_article | 31.6 | 0.22% (2 of 908) | 7.0 | 765 | 143 |

**Reading — the decisive C2 result.** Across the entire 5-PDF set, **only 2 gold
articles are ABSENT from the index** (bsard 6504 in Code Pénal, bsard 5080 in Code
Judiciaire-larger) — and **both are FOUND-status**, i.e. a node-linking gap, not an
extraction failure. **99.8% of all misses are BEYOND_POOL**: the article *is*
indexed but ranked beyond the top-100. Arm 2A's misses are therefore a **ranking**
problem, not an indexing/extraction-coverage problem — the opposite of what
ANALYSIS_PLAN-C2 anticipated for this arm. Missed gold is overwhelmingly
FOUND-extraction (84% of summary_node misses), not PARTIAL, so the coverage gaps do
*not* concentrate in the PARTIAL/low-cosine stratum. The whole-article unit
(raw_article) yields the **fewest entire-miss questions** (7.0% vs the node unit's
8.1%), consistent with §7's granularity trade-off.

*(Failure lens 2 — retrieved-gold coverage — is the §3 fragment analysis.)*

---

## 7. Within-arm contrasts

`data/tables/arm2a_aggregate_summary` (micro), `Report/tables/tab_rq2_arm2a_within_arm.tex`.
(The granularity-contrast figure `fig_rq2_arm2a_node_vs_article` is not included;
the within-arm contrasts are carried by the table.)

**(a) Enrichment lift — summary_node vs raw_node (same node unit).**

| | R@10 | R@100 | Hit@10 | coverage (touched) | % frag |
|---|---:|---:|---:|---:|---:|
| summary_node | 0.482 | 0.807 | 0.691 | 0.724 | 28.6 |
| raw_node | 0.455 | 0.759 | 0.692 | 0.713 | 29.4 |

Index-time enrichment (the gpt-4o summary embedded in place of the raw FR text) is
a **pure recall win**: +0.027 R@10 and **+0.048 R@100** (it recovers more gold deep
in the ranking), at **identical Hit@10, coverage and fragmentation**. Enrichment
changes *what is embedded*, not the node granularity, so it cannot and does not
move the fragmentation liability — it only helps the right nodes rank higher.

**(b) Granularity — node (summary_node) vs article (raw_article) unit.**

| | R@10 | R@100 | Hit@10 | coverage (touched) | % frag | % entire-miss q |
|---|---:|---:|---:|---:|---:|---:|
| summary_node (node) | 0.482 | 0.807 | 0.691 | 0.724 | 28.6 | 8.1 |
| raw_article (article) | 0.450 | 0.810 | 0.674 | **1.000** | **0.0** | **7.0** |

The node unit trades **complete context for head recall**: it buys +0.032 R@10 and
+0.017 Hit@10 and a better median rank (2 vs 3), but the article unit delivers
**every retrieved gold article in full** (coverage ≡ 1.0, 0% fragments) and produces
**fewer entire-miss questions** (7.0% vs 8.1%) and equal deep recall (R@100 0.810 ≈
0.807). For a generator that is hurt by partial context, the article unit gives up
~3 recall points to eliminate the 28.6% fragment rate entirely — the central
node-vs-whole-unit tension of this arm.

---

## 8. Summary of findings

1. The canonical **summary_node** index recovers 48% of gold by rank 10 (Hit@10
   0.69, first gold at median rank 2) with a **clean, compact context** (~8 distinct
   articles in the top-10 nodes, P@10 0.15) — it ranks gold higher and with less
   distractor noise than Arm 1.
2. **The node unit's liability is fragmentation: 28.6% of retrieved gold arrives as
   a fragment** (coverage 0.724), *more* than Arm 1's 18% — AzureDI nodes are
   sub-article (up to 11 per article), so a top-10 node budget often delivers only
   part of a gold article. Coverage is *worst on the easy, finely-split PDF* (Code
   du Logement, 0.641 / 36.7% frag), inverting Arm 1's pattern.
3. **Misses are a ranking problem, not a coverage gap.** 99.8% of missed gold is
   present-but-ranked-beyond-100; only 2 gold articles in the whole set are absent
   from the index (both FOUND-status linking gaps). **ANALYSIS_PLAN-C2 is not
   supported for the arm it was about** — gaps do not concentrate in PARTIAL/low-cosine
   extraction; PARTIAL questions slightly out-recall FOUND.
4. **Within-arm:** index-time enrichment is a pure recall lift (+0.05 R@100) at
   unchanged coverage; the article unit trades ~3 recall points for full coverage
   (0% fragments) and fewer entire-misses — the granularity dial.

*(Built on the shared `deep_dive_common` engine so Group-A/D metrics are identical
to the Arm-1 dive; the Arm-1 reference values cited throughout (R@10 0.472, Hit@10
0.630, coverage 0.806, 18% fragment, distinct@10 16.8, median rank 4) are from
`reports/arm1_naive_deep_dive.md`. A formal cross-arm join is deferred.)*

---

### Artifact index

| Artifact | Path |
|---|---|
| Per-question metrics (2175 rows, 3 variants) | `data/tables/arm2a_per_question.{csv,md}` |
| Per-gold-article coverage (5748 rows, node variants) | `data/tables/arm2a_coverage_per_gold_article.{csv,md}` |
| Node→article token-weight table (4894 rows; both weights) | `data/tables/arm2a_node_bsard_weights.csv` |
| Missed-gold classification (2894 rows) | `data/tables/arm2a_missed_gold.{csv,md}` |
| Per-PDF summary | `data/tables/arm2a_per_pdf_summary.{csv,md}` · `Report/tables/tab_rq2_arm2a_per_pdf.tex` |
| Aggregate (micro+macro, 3 variants) | `data/tables/arm2a_aggregate_summary.{csv,md}` |
| Within-arm contrast | `Report/tables/tab_rq2_arm2a_within_arm.tex` |
| Strata | `data/tables/arm2a_strata_summary.{csv,md}` |
| Miss summary | `data/tables/arm2a_miss_summary.{csv,md}` · `Report/tables/tab_rq2_arm2a_miss.tex` |
| ~~Coverage distribution~~ | *not included* |
| ~~Recall vs coverage per PDF~~ | *not included* |
| ~~Node-vs-article granularity~~ | *not included* |
| ~~Entire-miss × extraction status~~ | *not included* |
| Cardinality strata → cross-arm | `Report/figures/fig_rq2_arm2a_recall_curve_cardinality.{pdf,png}` |
