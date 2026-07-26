# RQ2 T06 — Next-Analysis Plan

Companion to `reports/cross_arm_analysis.md` (what's done) and the `Report/`
deliverable layer (`RQ2_STORY.md`). Scopes the analysis still worth doing, what
data backs each, and which deliverable consumes it. Ordered by value/effort.

---

## A. What T06 already established (and where it feeds)

| Existing analysis | Module / output | Deliverable that consumes it |
|---|---|---|
| One-path cross-arm re-eval (the crown jewel — makes arms apples-to-apples) | `consolidate.py` → `cross_arm_long.csv` | every Report figure/table |
| Headline metrics, macro+micro | `tables.py` → `cross_arm_headline.*` | `tab_rq2_headline` (headline bars not included) |
| Per-PDF metrics | `tables.py` → `cross_arm_per_stem.md` | `fig_rq2_perpdf_bars_byarm` (was `_perpdf_bars`), per-PDF recall curve |
| E/ drift-aware + W/ partial | `tables.py` → `cross_arm_extended.md` | GT-sensitivity slope (planned) |
| Paired significance (R@10/R@100) | `significance.py` → `cross_arm_significance.csv` | headline `†`, dumbbell (dropped) |
| Failure buckets (hit/near-miss/missed) | `errors.py` → `failure_summary.csv` | error-profile table (planned) |
| Cost (LLM calls, tokens, **latency**) | `cross_arm_costs.json` | cost table + scatter (planned) |
| Per-query rankings + first-hit-rank | `per_query/<stem>.json` | **largely untapped — see C** |

**Verdict:** the comparability backbone is solid and defensible. The gaps are
(i) two rigor holes, and (ii) several high-value signals already on disk that no
analysis has touched yet.

---

## B. Rigor fixes (low effort, do first)

**B1 — MRR/nDCG significance.** The headline claim "Arm 2A wins *ranking
quality*" rests on MRR@10 (0.46 vs 0.35) and nDCG@10 (0.40 vs 0.35), but
`significance.py` only tests R@10/R@100. The R@10 tie (p=0.42) is tested; the
MRR/nDCG *win* is not. → Extend `significance.py` to compute per-question
MRR@10 and nDCG@10 vectors (T07 `evaluation.metrics.mrr_at_k`/`ndcg_at_k` on the
bsard-resolved rankings) and run the same paired tests. **Without this the
thesis's main positive finding for Arm 2A is unsupported.**

**B2 — Correct the cost claim.** `cross_arm_analysis.md` §6 marks T04 as
"(index-time only)*" — wrong. T04 records real per-query retrieval latency
(638–1043 ms; p95 up to 1.8 s). Only T03 latency is genuinely absent (precomputed
pool). → Fix the §6 paragraph and build the cost artifacts (B3).

**B3 — Cost–quality artifacts.** Data is ready in `cross_arm_costs.json`.
→ `tab_rq2_cost` (LLM calls/q, tokens/q, latency mean+p95 per arm). State
"T03 latency not recorded" in the caption. *(The companion scatter
`fig_rq2_arm2b_cost_vs_recall` is not included; the cost story is carried by the
table alone.)*

---

## C. Failure-mechanism deep dives (high value — sharpen the story)

**C1 — Rank-of-first-hit distribution.** `first_hit_rank` is in every
`per_query/<stem>.json` but only collapsed into 3 buckets. → CDF (or violin) of
first-hit rank per arm over the 725 questions. Sharpens the asserted story into
a measured one: report median / p90 first-hit rank per arm. Expected: Arm 1 long
tail but few ∞ (ranking-depth weakness); Arm 2A tight head but a spike at ∞
(coverage gaps); Arm 2B bimodal — a navigated head then a padded tail (e.g. doc
1804 q1053 first hits at rank 84). Deliverable: `fig_rq2_first_hit_cdf`.

**C2 — Stratified analysis (the RQ1-strata analog, now feasible).** Two strata
exist on disk that the styleguide wrongly assumed absent:
- **single- vs multi-article** (315 vs 410 questions) — derivable from GT
  cardinality.
- **extraction status** — `ground_truth/question_extraction_status/
  questions_by_extraction_status.jsonl` carries per-question status
  (exact/mixed/…) and per-article `verification_status` (FOUND/PARTIAL/
  NOT_FOUND) + `extraction_cosine`.
→ Recompute per-question recall split by each stratum (T07 comparator already
supports single/multi + extraction buckets). Tests two mechanistic hypotheses:
(a) all arms drop on multi-article questions (must retrieve *all* gold articles);
(b) **Arm 2A's coverage gaps concentrate in PARTIAL / low-cosine extraction
questions** — i.e. the node index fails exactly where AzureDI extracted the
article poorly. Deliverable: `fig_rq2_strata_smallmultiples` (2×N small
multiples, the RQ1 signature shape) + `tab_rq2_strata`.

**C3 — PageIndex navigation failure anatomy.** `loaders.load_t05` preserves the
per-query `trace` (the Law→Chapter→Article navigation path), never analysed. →
Re-export T05 with trace; categorise each miss as: (i) wrong chapter reached,
(ii) right chapter / wrong article, (iii) GT only in the pad-to-100 tail (not
navigated to at all). Cross with C1's bimodal hypothesis. Answers *why* 2B
misses 51%, turning "not competitive" into a mechanism. Deliverable:
`tab_rq2_arm2b_navigation` + 2–3 curated trace walkthroughs for the chapter.

---

## D. Complementarity (novel — answers "should the arms be combined?")

**D1 — Inter-arm agreement + oracle ceiling.** `ranked_top` (top-20 bsard_ids
per arm per query) supports, at k=10:
- **agreement** — mean Jaccard of top-10 between each arm pair (are they finding
  the *same* articles or different ones?);
- **oracle/union recall** — fraction of questions solved by ≥1 arm (the fusion
  ceiling) vs the best single arm;
- **unique contribution** — questions *only* arm X gets right.
If union recall ≫ best single arm, the arms are complementary and a meta-fusion
is the natural RQ2 follow-up; if union ≈ best, they're redundant and the winner
stands alone. Either result is a strong thesis statement. Deliverables:
`fig_rq2_arm_agreement` (heatmap, vlag) + `tab_rq2_oracle_ceiling`.

**D2 — Multi-article partial recall.** `per_query_recall.csv` recall is
*fractional* (|retrieved∩gold|/|gold|). For multi-article questions, do arms get
*all* gold articles or just one? → Distribution of per-question recall on the 410
multi-article questions per arm. Complements W/ (which is T03-only).
Deliverable: folds into C2's strata figure.

---

## E. Optional / lower priority

- **E1 — Per-PDF difficulty correlates.** Why is Code du Logement easy (0.84)
  and Code Pénal hard (0.30)? Correlate per-PDF recall with n_articles, mean
  extraction_cosine, and (for T05) tree depth. Explanatory, not core.
- **E2 — Score calibration.** T03/T04 keep fused scores; gap between hit/miss
  score could diagnose threshold behaviour. Low thesis value.

---

## Suggested execution order

1. **B1 + B2 + B3** — closes the rigor gaps; cost artifacts are nearly free.
2. **C1** — one figure, immediately sharpens the headline failure-mode story.
3. **C2** — the highest-value deep dive; gives RQ2 its strata chapter and a
   mechanistic explanation of Arm 2A's coverage gaps.
4. **D1** — the novel complementarity result; sets up any "combine the arms"
   discussion.
5. **C3** — qualitative depth for the PageIndex chapter.
6. **D2 / E** — as space allows.

Each new per-question analysis (C1, C2, D1, D2) reads from `per_query/*.json` +
`per_query_recall.csv` + the extraction-status jsonl — no retrieval rerun, same
one-path discipline as the rest of T06.
