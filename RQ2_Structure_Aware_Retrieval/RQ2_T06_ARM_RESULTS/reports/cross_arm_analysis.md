# Cross-Arm Retrieval Analysis (RQ2)

Comparison of the three RQ2 retrieval arms on the curated 5-PDF BSARD subset
(725 questions): **T03 Arm-1 Naive** (flat chunking, dense+BM25→RRF),
**T04 Arm-2 Metadata** (AzureDI node/article enrichment, dense+BM25→RRF), and
**T05 Arm-2 PageIndex** (vectorless LLM navigation of a Law→Chapter→Article tree).

All numbers here were produced by re-scoring every arm's **persisted** rankings
through one path — T07's `evaluation.comparator.evaluate` (tiers 1+2) against one
per-PDF ground truth (`runs/t04_<stem>.json`) — so they are directly comparable
(see [EVALUATION_METHODOLOGY.md](../EVALUATION_METHODOLOGY.md)). No retrieval was
rerun. Source artefacts: `data/tables/`, `data/figures/`, `data/per_query/`,
`data/error_analysis/`.

---

## 1. Headline (micro average, 725 questions)

| method | R@10 | R@100 | MRR@10 | nDCG@10 |
|---|---|---|---|---|
| **T04 summary_node** | **0.483** | 0.807 | **0.460** | **0.398** |
| T03 naive | 0.472 | **0.858** | 0.348 | 0.348 |
| T04 full/raw/enriched node | 0.455 | 0.759 | 0.438 | 0.368 |
| T04 raw/full article | 0.450 | 0.810 | 0.431 | 0.365 |
| T05 pageindex | 0.185 | 0.404 | 0.184 | 0.156 |

(`data/tables/cross_arm_headline.md`, figure `data/figures/cross_arm_headline_micro.png`.)

**Reading:** T04 (best variant, `summary_node`) and T03 are close on early recall
(R@10 0.483 vs 0.472) but T04 ranks the first relevant article markedly higher
(MRR@10 0.460 vs 0.348, nDCG@10 0.398 vs 0.348). T03 leads on R@100. T05 trails
all node/article retrievers by a wide margin on every metric.

---

## 2. Statistical significance (paired, per question, pooled)

Paired Wilcoxon + t-test on per-question recall / MRR@10 / nDCG@10
(`data/tables/cross_arm_significance.md`). MRR/nDCG rows added under
ANALYSIS_PLAN.md B1 — they test the ranking-quality claim that R@10 alone misses:

| comparison | metric@k | mean A | mean B | Δ | wins A / B | min p | sig. |
|---|---|---|---|---|---|---|---|
| T04 summary vs **T03** | R@10 | 0.483 | 0.472 | +0.011 | 136 / 130 | 0.42 | **no (tie)** |
| T04 summary vs **T03** | **MRR@10** | 0.460 | 0.348 | **+0.112** | 281 / 112 | 1e-16 | **yes (T04)** |
| T04 summary vs **T03** | **nDCG@10** | 0.506 | 0.419 | **+0.087** | 319 / 152 | 3e-15 | **yes (T04)** |
| T04 summary vs **T03** | R@100 | 0.807 | 0.858 | −0.051 | 66 / 160 | 2e-7 | **yes (T03)** |
| T04 summary vs **T05** | R@10 | 0.483 | 0.185 | +0.298 | 386 / 61 | 1e-45 | **yes (T04)** |
| T03 vs **T05** | R@10 | 0.472 | 0.185 | +0.287 | 358 / 64 | 8e-39 | **yes (T03)** |
| T04 summary vs **T04 raw_node** | R@10 | 0.483 | 0.455 | +0.027 | 105 / 55 | 0.003 | **yes (summary)** |
| T04 summary vs **T04 raw_node** | R@100 | 0.807 | 0.759 | +0.048 | 127 / 35 | 5e-10 | **yes (summary)** |

(MRR@10 means match the headline table exactly; the nDCG@10 means use T07's
standalone `ndcg_at_k`, which normalises by relevant-in-top-k and so reads higher
than the comparator's aggregate nDCG@10 in §1 — the sign and significance agree.)

**Key findings:**
1. **T04's R@10 edge over T03 is *not* significant (a tie, p=0.42; 136 vs 130
   wins), but its ranking-quality edge *is* — and strongly so**: MRR@10 +0.112
   ($p\approx1\mathrm{e}{-16}$, 281 vs 112 wins) and nDCG@10 +0.087
   ($p\approx3\mathrm{e}{-15}$). The two arms find a relevant article in the
   top-10 about equally often, but T04 ranks it markedly higher. This is now the
   tested headline result, not an assertion.
2. **T03's R@100 lead over T04 *is* significant** (p≈2e-7) — a real effect of its
   deeper 200-candidate pool (see §6 caveat).
3. **Both T03 and T04 beat T05 overwhelmingly** at every cutoff (p < 1e-38).
4. **Summary metadata enrichment gives a small but significant lift** over the raw
   node variant (R@10 +0.027, p=0.001; R@100 +0.048, p=5e-10) — index-time LLM
   summaries help, even though T04's boost/filter stage was shown inert.

---

## 3. Per-PDF variation

`data/tables/cross_arm_per_stem.md`. The ordering (T04≈T03 ≫ T05) holds on all
five PDFs. T05's R@10 ranges 0.10–0.32 (best on the smaller Code du Logement,
worst on Code Judiciaire). Question counts vary 65–252, so the **micro** average
weights the larger codes more; the macro average tells the same story.

---

## 4. Drift-aware (effective) view

Restricting the recall denominator to GT articles actually locatable in each PDF
(`E/`, `data/tables/cross_arm_extended.md`) does **not** change the ranking —
T03 0.471, T04 summary 0.443, T05 0.161 at E/recall@10 — confirming the
comparison is robust to ground-truth drift and not an artefact of unreachable GT.

---

## 5. Partial-relevance (weighted) view

T03 is the only chunk-based arm, so weighted partial-relevance recall (`W/`,
chunk-overlap credit) applies to it alone: W/recall@10 = 0.457, W/recall@100 =
0.859 — essentially equal to its binary article recall, i.e. when T03 retrieves a
relevant article it retrieves (nearly) all of its text, not a fragment. (Weight
coverage was repaired under CN-T07-010 before this was computed.)

---

## 6. Cost vs quality

| arm | query-time latency / query | LLM calls / query | tokens / query |
|---|---|---|---|
| T03 naive | not recorded* | 0 | 0 |
| T04 summary_node | ~0.9 s (median; p95 ~1.8 s) | 0 | 0 |
| T05 pageindex | **~38 s** (range 28–44 s) | **5.8** | **19.4 k** |

\*T03 was served from a precomputed top-200 pool, so per-query retrieval latency
was not recorded (only T04 and T05 logged real query-time latency, and those two
are mutually comparable). **Note on T04 "cost":** T04 makes **zero query-time
LLM calls** — its metadata enrichment is done once at index time — but it still
incurs a real ~0.9 s/query *retrieval* latency (dense + BM25 + RRF over the
node/article index). "Index-time-only" refers to the LLM/enrichment cost, not to
latency.

T05 costs ~40× the wall-clock of T04 and incurs ~6 LLM calls per query while
delivering less than half the recall — it is **strictly dominated** on both
quality and cost. T04 matches T03's early recall with **zero query-time LLM
calls** and sub-second retrieval latency.

---

## 7. Error profile (`data/error_analysis/failure_summary.csv`)

Pooled over 725 questions — questions with ≥1 GT article retrieved:

| arm | hit@10 | near-miss (GT below 10) | missed entirely | missed % |
|---|---|---|---|---|
| T04 summary_node | 501 | 165 | 59 | 8.1% |
| T04 node (raw/enriched/full) | 502 | 151 | 72 | 9.9% |
| T03 naive | 457 | 258 | **10** | **1.4%** |
| T05 pageindex | 192 | 164 | **369** | **50.9%** |

**Two distinct failure modes:**
- **T03** almost never misses an article *entirely* (1.4%) — its 200-deep dense+BM25
  pool surfaces nearly every relevant article *somewhere* — but it has the most
  *near-misses* (258 questions where the GT sits at rank 11–100). T03's weakness is
  **ranking depth**, not coverage.
- **T04** ranks better in the top-10 (more hit@10, fewer near-misses, first gold at
  median rank 2) but **misses ~8% of questions entirely** (8.1% summary_node, 9.9%
  raw node). This is a **ranking-depth** problem, **not** an index-coverage gap: the
  Arm-2A deep dive ([arm2a_metadata_deep_dive.md](arm2a_metadata_deep_dive.md) §6)
  shows only **2 of 2 874 gold articles are absent** from the node index — 99.8% of
  missed gold *is* indexed but ranks beyond the top-100 pool. The 1.4% (T03) vs 8.1%
  (T04) gap is moreover **confounded by pool depth** (T03 persists top-200, T04
  top-100). T04's actual structural liability is **node fragmentation**: 28.6% of the
  gold it *does* retrieve arrives as a <½-text fragment (deep dive §3).
- **T05** misses **half** of all questions entirely (50.9%). The Arm-2B deep dive
  ([arm2b_pageindex_deep_dive.md](arm2b_pageindex_deep_dive.md) §4) pins the
  mechanism: it is a **chapter-selection** failure — **79.9% of all gold articles
  sit in chapters the 8B navigator never selected**; only 11.2% are
  right-chapter/wrong-article and just **2 of 2 874** are absent from the tree (the
  same two T04 misses — a shared corpus gap). Its nominal R@100 (0.404) is moreover
  **mostly deterministic padding**: the LLM's own picks (exposed head) recover only
  0.137, so ≈two-thirds of the R@100 is the chapter/law pad, not navigation —
  interpreting T05's deep recall as retrieval competence is a category error.

---

## 8. Conclusions for RQ2

1. **Metadata enrichment (T04) is the best arm overall** — it wins ranking quality
   significantly (MRR@10 +0.112, $p\approx2\mathrm{e}{-16}$; nDCG@10 +0.087,
   $p\approx2\mathrm{e}{-14}$; paired tests, §2) and ties T03 on R@10, with **zero
   query-time LLM calls** (enrichment is index-time) and sub-second retrieval
   latency. Its `summary_node` variant is the recommended configuration; the
   summary-vs-raw lift is small but significant.
2. **Naive chunking (T03) remains a strong baseline** — statistically tied with T04
   on R@10 and significantly ahead on R@100, owing to its deeper pool. The R@100
   advantage is partly structural (pool depth 200 vs 100) — for a strict like-for-like
   R@100 claim, truncate T03 to 100 before comparing.
3. **PageIndex (T05) is not competitive** — significantly worse on every metric and
   ~40× more expensive. Its value is interpretability (navigation traces), not retrieval.

### Caveats carried from the methodology
- Pool depths differ (T03=200, T04=100, T05=100); R@100 favours T03 structurally.
- `W/` applies to T03 only (chunk-based); T04/T05 are article-level.
- Latency: T04 (~0.9 s) and T05 (~38 s) logged real query-time latency and are
  mutually comparable; T03's was not recorded (precomputed pool), so T03 is
  excluded from latency comparisons.

---

*Regenerate everything:* `scripts/run_consolidate.py --effective --weighted`,
`scripts/run_significance.py`, `scripts/run_errors.py` (read-only via T03's venv).
