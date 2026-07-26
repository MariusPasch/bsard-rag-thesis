# Arm 2B (PageIndex) — Retrieved-Article Deep Dive

A per-PDF and aggregate profile of the **retrieved articles** of Arm 2B (T05
vectorless LLM tree-navigation over the Law→Chapter→Article ToC), focused on the
properties that affect downstream answer generation, over the curated 5-PDF BSARD
subset (725 questions). **Scope: Arm 2B only** — every metric is defined
arm-agnostically and computed by the *same* `deep_dive_common` primitives the
Arm-1 and Arm-2A dives use, so a later 3-way synthesis is a simple join; the
cross-arm comparison itself is deferred. Latency is excluded (LLM-call, token and
navigation-step counts are reported).

**Single canonical config.** Arm 2B has no variant grid — the live `results/`
snapshot (post-fix, chapter-then-law padding) is the one shipped configuration.
All numbers are read from the **persisted one-path artifacts** (raw per-query
results + navigation trace, `tree.json`, per-PDF GT, extraction status,
`cross_arm_long.csv`) — **navigation is never rerun** (STYLEGUIDE §6,
[EVALUATION_METHODOLOGY.md](../EVALUATION_METHODOLOGY.md) §4). The article ranking
(one `bsard_id` per item, score-sorted with the score-0 padding last) is asserted
to reproduce the published `T1/R@k`: **all 20 (stem, k) cells match to ≤1e-9**
(max deviation 8.3e-17 — exact; unlike the Arm-2A node-variant drift, the T05
`results/` snapshot has not been resynced). Deterministic; seed = 42 for the
bootstrap CIs. Engine `src/arm_results/arm2b_deep_dive.py` + shared
`src/arm_results/deep_dive_common.py` (re-confirmed byte-identical for Arm-1/2A
after this dive), runner `scripts/run_arm2b_deep_dive.py` (T03 venv).

**No coverage — by construction.** Arm 2B retrieves **whole articles** (a tree
leaf *is* an article; one `bsard_id` per ranked item, `item.text` = the article
text). Coverage of a retrieved gold article is therefore ≡ 1.0 on every hit — the
Arm-1/Arm-2A token-coverage lens does not apply, and **no coverage distribution is
computed**. Its Arm-2B analogue is the navigated-vs-padded decomposition (§3).

**The three observable tiers of the result list (the Arm-2B lens).** The
persisted top-100 is the LLM's navigation output followed by deterministic padding
to `pad_to_k=100`. Reading the navigator, three tiers are observable per query and
are **labelled explicitly throughout** (the Arm-1/2A lesson — never conflate a
"head" recall with a padded recall):

- **Navigated set `N_nav`** — the trace's `summary.candidate_article_ids`: the
  deduped union of `selected_article_ids` over the article-selection steps (+ any
  `follow_refs` additions). This is the brief's "navigated set" — every article
  the LLM tree-walk actually reached and chose.
- **Exposed head `N_exp`** — the ranked items with `score > 0`. A navigated
  article keeps its LLM `evaluate` score (or the default `score_threshold = 1`
  when the model forgot to score it) and survives **iff** that score ≥ 1; an
  article the model explicitly scored 0 is dropped from the head and re-absorbed
  into the score-0 padded tail. `N_exp` is what sits **above** the padding — what
  a small-top-k generator actually receives as "the LLM's picks".
- **Padded tail** — ranked items with `score == 0`: the deterministic
  chapter-then-law fill (which also re-absorbs evaluate-rejected candidates).

The three recall senses: **`nav_recall`** (over `N_nav`), **`exposed_recall`**
(over `N_exp`), and the full-pool **`recall@k`** (padding-inflated at large *k*).

**Navigated/padded boundary — reconstruction + validation.** `N_nav` is read
directly from the persisted trace `summary` (the navigator recorded its own
deduped candidate set). `N_exp` is recomputed independently from the ranked-item
scores. Two cross-checks (the brief's "how cleanly the two methods agree"):
1. **`N_exp ⊆ N_nav` holds for all 725/725 queries** — the code invariant
   (exposed items can only come from the navigated candidates) is empirically
   confirmed.
2. **`N_exp == N_nav` in 57.0% of queries.** They diverge in the other 43% — not
   because the trace and the result disagree, but because the LLM `evaluate` step
   scored some navigated candidates 0, dropping them from the head into the tail.
   So the navigated set is recovered cleanly; the 43% gap *is itself the
   evaluate-gate effect*, quantified in §3.
   *(The RAW per-step `selected_article_ids` union is a noisy over-count — the
   trace records the model's pre-cap, pre-validity selections — and is recorded in
   `arm2b_per_question.csv` as `n_raw_selected` for transparency but never used.)*

**Ground truth.** Binary fractional recall against the **full per-PDF GT**
(`runs/t04_<stem>.json`) is **primary**; the **effective drift-aware GT (`E/`)** is
reported **alongside** at per-PDF and aggregate granularity (read straight from
`cross_arm_long.csv`). Per-question strata use the full GT, with the same
FOUND/PARTIAL rule as the prior dives (a question is **PARTIAL** if ≥1 gold article
is PARTIAL-extraction; **no NOT_FOUND or UNKNOWN** on the curated set).

---

## 1. Headline (aggregate over 725 questions)

| | micro (question-weighted) | macro (per-PDF mean) |
|---|---|---|
| Article Recall@5 | 0.145 | 0.141 |
| **Article Recall@10** | **0.185** [0.159, 0.211] | **0.174** [0.114, 0.254] |
| Article Recall@20 | 0.227 | 0.242 |
| Article Recall@100 (≡ padded recall) | 0.404 | 0.397 |
| `E/`Recall@10 (drift-aware) | 0.175 | 0.161 |
| `E/`Recall@100 (drift-aware) | 0.363 | 0.342 |
| Hit@10 | 0.265 | 0.243 |
| **Navigated recall** (over `N_nav`) | **0.145** | 0.147 |
| **Exposed-head recall** (over `N_exp`, score>0) | **0.137** | 0.140 |
| Exposed-head precision (gold ∕ exposed picks) | 0.091 | 0.088 |
| Article precision@10 (gold ∕ 10) | 0.050 | 0.045 |
| Median exposed-head size (articles) | 4 | 4.1 |
| Median rank of first gold article | 7 | 8.7 |
| LLM calls / query · tokens / query | 5.82 · 18 868 | 5.79 · 18 501 |

(`data/tables/arm2b_aggregate_summary.{csv,md}`; 95% bootstrap CI, seed 42, 2000
resamples over the aggregation unit — questions for micro, the 5 PDFs for macro,
hence the wide macro band.)

**Reading.** Arm 2B is the weakest of the three arms by a wide margin: it puts a
gold article in the top-10 for only **27%** of questions (Hit@10 0.265, vs Arm-1
0.63 / Arm-2A 0.69) and recovers **18.5%** of gold by rank 10 (vs ≈0.47–0.48). Its
defining property is **shallow navigation propped up by padding**: the LLM's own
selections recover only **13.7%** of gold (exposed-head recall), and navigation
*as a whole* reaches **14.5%** (navigated recall) — recall@100 reaches 0.404
**only because deterministic chapter/law padding fills the tail**. So ≈two-thirds
of the nominal R@100 is padding, not navigation. The navigated head is also **not
clean**: the LLM's ~4 hand-picked articles are <10% gold (exposed precision 0.091),
*lower* than Arm-2A's P@10 over 10 nodes (0.147) — contrary to the a-priori
expectation that an LLM's deliberate picks would be high-precision. The
navigated→exposed gap is small (0.145 → 0.137): the evaluate gate trims candidates
but rarely removes the *gold* among them — the loss happens earlier, at chapter
selection (§4).

---

## 2. Per-PDF profile

One row per PDF, ordered by question count. Recall/Hit on **k = articles**.
Full table `data/tables/arm2b_per_pdf_summary.{csv,md}`; report-ready
`Report/tables/tab_rq2_arm2b_per_pdf.tex`.

| PDF (law type) | n | R@10 | `E/`R@10 | Hit@10 | R@100 (padded) | R (exposed) | R (navigated) | P (exposed) | med. head | med. rank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Code Civil | 252 | 0.203 | 0.198 | 0.250 | 0.393 | 0.122 | 0.129 | 0.058 | 3 | 8 |
| Code Judiciaire (larger) | 204 | 0.122 | 0.105 | 0.265 | 0.406 | 0.066 | 0.078 | 0.088 | 5.5 | 13 |
| Code du Logement | 133 | 0.323 | 0.326 | 0.421 | 0.477 | 0.319 | 0.319 | 0.191 | 3 | 1 |
| Code Judiciaire (smaller) | 71 | 0.100 | 0.086 | 0.141 | 0.225 | 0.084 | 0.098 | 0.048 | 4 | 5.5 |
| Code Pénal | 65 | 0.123 | 0.092 | 0.138 | 0.485 | 0.110 | 0.110 | 0.055 | 5 | 16 |

**Reading.** The recall ordering matches the other arms — **Code du Logement** is
again the easy case (R@10 0.323, exposed precision 0.191, **median first gold at
rank 1**) and the two **Code Judiciaire** PDFs the hardest — but the *level* is far
lower everywhere. Two PDF-specific signatures stand out:
- **Code du Logement is the only PDF where navigation works:** exposed recall
  equals navigated recall (0.319 = 0.319), the evaluate gate removes nothing, and
  R@100 (0.477) sits only +0.16 above the exposed head — padding contributes least.
  Short, cleanly-chaptered articles let the 8B model pick the right subtree.
- **Code Pénal has the widest padding gap** (exposed 0.110 → R@100 0.485): almost
  *all* of its nominal recall is padding. Its median first-gold rank is 16 — the
  navigated head almost never contains the gold, and the only reason R@100 looks
  competitive is that the law-level pad fills the tail with a large fraction of a
  short code. [fig_rq2_arm2b_navigated_vs_padded](../../Report/figures/fig_rq2_arm2b_navigated_vs_padded.png)
  shows the exposed-vs-padded bars with the navigated-recall marker per PDF.

---

## 3. Navigated vs padded — the central Arm-2B lens (Group N)

This replaces Group B coverage. (`data/tables/arm2b_per_question.csv`,
`arm2b_gold_landing.csv`; figure `fig_rq2_arm2b_navigated_vs_padded` — now
overlaid with the random-padding counterfactual. The `fig_rq2_arm2b_rank_of_gold_cdf`
figure is not included.)

- **Padding-inflated recall.** Exposed-head recall **0.137** → padded recall@100
  **0.404** (micro). The +0.267 gap is entirely deterministic padding — gold that
  the LLM never navigated to but that happens to fall in a selected chapter or the
  fallback law. **Interpreting Arm-2B's R@100 as a retrieval competence is a
  category error**: it largely measures how much of a (often short) code the pad
  sweeps in, not what the navigation found.
- **What a small-top-k generator actually gets.** Of the questions that *hit*
  anywhere in the top-100, only **44.1%** have their first gold inside the exposed
  head; the other **55.9% reach gold only in the padded tail** (first-gold rank >
  exposed-head size). With a median exposed head of 4 articles and a median
  first-gold rank of 7, a generator reading the top ~5 "LLM picks" sees a gold
  article in well under half of even the nominal hits.
- **Where all gold lands (article-weighted, micro).** Of the 2 874 gold articles:
  **7.9% in the exposed head**, **36.1% in the padded tail**, **56.0% outside the
  top-100 entirely**. Padding is doing most of the visible recall work, and a
  majority of gold never enters the pool at all.
- **Navigated head precision is low, not high.** Exposed precision 0.091, navigated
  precision 0.075 (micro) — the LLM's selections are mostly distractors. The
  Arm-2B signature is therefore *not* "clean but shallow"; it is **shallow *and*
  noisy at the head, with a padded tail that supplies most of the recall**.
- **Navigated-set size.** Median 5 candidates / 4 exposed per query (per-PDF medians
  3–7 / 3–5.5); 14 questions navigated to **zero** candidates (`exit_reason =
  no_candidates`/`parse_fail`, §7) and rely entirely on padding.

---

## 4. Navigation-failure anatomy — *why* Arm 2B misses (ANALYSIS_PLAN C3)

For every gold article **not reached by navigation** (`gold − N_nav`), classified
via `tree.json` + the LLM's `selected_chapter_ids`. Percentages are of **all gold
articles** (article-weighted, micro). (`data/tables/arm2b_nav_anatomy.{csv,md}`
2621 rows, summary `arm2b_nav_anatomy_summary.{csv,md}`; report-ready
`Report/tables/tab_rq2_arm2b_navigation.tex`; figure `fig_rq2_arm2b_nav_anatomy`.)

| PDF (law type) | gold | % navigated | % wrong-chapter | % chapter-ok-article-missed | % absent-from-tree |
|---|---:|---:|---:|---:|---:|
| Code Civil | 973 | 7.1 | 78.7 | 14.2 | 0.0 |
| Code Judiciaire (larger) | 1155 | 8.1 | 84.2 | 7.6 | 0.1 |
| Code du Logement | 255 | 23.5 | 76.5 | 0.0 | 0.0 |
| Code Judiciaire (smaller) | 277 | 3.2 | 71.8 | 24.9 | 0.0 |
| Code Pénal | 214 | 10.3 | 76.2 | 13.1 | 0.5 |
| **ALL** | **2874** | **8.8** | **79.9** | **11.2** | **0.07** |

**The decisive C3 result — Arm 2B's misses are a chapter-selection problem.**
**79.9% of all gold articles sit in chapters the LLM never selected** (wrong
subtree) — uniformly 72–84% across every PDF. The article-selection step is rarely
the bottleneck: only **11.2%** are "right chapter, wrong article" (chapter selected
but the article not picked/kept), and that share is itself concentrated where the
LLM picks a chapter but caps or mis-selects its articles (Code Judiciaire-smaller
24.9%, Code du Logement 0% — when Code du Logement enters a chapter it keeps the article).
The tree is **essentially complete**: only **2 of 2 874 gold articles are absent**
(bsard 6504 in Code Pénal, 5080 in Code Judiciaire-larger) — **the very same two
articles the Arm-2A dive found absent from its node index**, i.e. a shared
corpus-level extraction/linking gap, not a PageIndex tree-build failure.

So the ~51% entire-miss rate is not an indexing gap and not (mostly) a within-chapter
selection error — it is the 8B navigator **reasoning its way into the wrong part of
the hierarchy**. This turns "Arm 2B is not competitive" into a concrete mechanism:
the chapter-selection LLM call, on a tree of 39–128 chapters per code, sends the
walk down a plausible-but-wrong branch for four of every five gold articles.

**Curated trace walkthrough (wrong-chapter, Code Civil q343).** *"Le père
biologique, mais non légal, doit-il payer une contribution alimentaire pour mon
enfant?"* The gold (art. 1150, **CHAPITRE 6 — De l'action en réclamation d'état**)
is a paternity-claim provision. The LLM selected four *plausible-adjacent*
filiation chapters — *Art. 203 (obligation alimentaire des père et mère)*,
*Généralités*, *De l'établissement de la filiation*, *Du moment de la conception* —
but **not** the "action en réclamation d'état" chapter where the gold lives. Result:
the gold never entered even the padded pool (`landing_tier = outside_pool`), because
padding only fills from *selected* chapters and the law-level fallback did not reach
it. A textbook wrong-chapter miss: topically near, hierarchically wrong.

*(Failure lens 1 = this anatomy; failure lens 2 = the entire-miss profile below.)*

---

## 5. Failure lens 2 — entire misses (50.9% of questions)

A question is an **entire miss** when *no* gold article lands in the top-100.
(`data/tables/arm2b_entire_miss_summary.{csv,md}`.)

| stratum | level | n | % entire-miss | % with any gold navigated | % with any gold exposed |
|---|---|---:|---:|---:|---:|
| overall | all | 725 | **50.9** | 22.5 | 21.7 |
| cardinality | single | 315 | 59.4 | 19.4 | 19.0 |
| cardinality | multi | 410 | 44.4 | 24.9 | 23.7 |
| extraction | FOUND | 368 | 57.3 | 13.0 | 12.2 |
| extraction | PARTIAL | 357 | 44.3 | 32.2 | 31.4 |

**Reading.** Arm 2B entirely misses **50.9%** of questions (matching the
`failure_summary` figure), and for **77.5%** of questions navigation reached *no*
gold article at all (only 22.5% have any gold navigated). Multi-article questions
miss less often (44.4% vs 59.4% single) — more gold articles means more chances for
*one* to fall in a selected chapter or the pad. The entire-miss rate is driven by
the same wrong-chapter mechanism as §4, not by GT cardinality.

---

## 6. Stratified analysis (full GT)

Single- vs multi-article (315 / 410) and extraction status (FOUND / PARTIAL).
(`data/tables/arm2b_strata_summary.{csv,md}`; figure
`fig_rq2_arm2b_strata_smallmultiples`.)

| stratum | level | n | R@5 | R@10 | R@20 | R@100 | Hit@10 | exposed R | med. rank | P@10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cardinality | single | 315 | 0.194 | 0.238 | 0.302 | 0.406 | 0.238 | 0.190 | 6 | 0.024 |
| cardinality | multi | 410 | 0.108 | 0.144 | 0.170 | 0.401 | 0.285 | 0.097 | 8.5 | 0.070 |
| extraction | FOUND | 368 | 0.114 | 0.158 | 0.226 | 0.371 | 0.196 | 0.090 | 15 | 0.033 |
| extraction | PARTIAL | 357 | 0.178 | 0.213 | 0.229 | 0.437 | 0.187 | 0.187 | 4 | 0.067 |

**Reading.**
- **Single-article questions out-recall multi** on the fractional metric (R@10
  0.238 vs 0.144 — multi requires *all* gold and the shallow head rarely supplies
  more than one), but **multi has higher Hit@10** (0.285 vs 0.238), the same
  trade-off seen in Arm-1/2A.
- **PARTIAL-extraction questions do *not* under-perform FOUND** — they out-recall
  it (R@10 0.213 vs 0.158, exposed recall 0.187 vs 0.090) with a far better median
  first-gold rank (4 vs 15). As in both prior dives, the **ANALYSIS_PLAN-C2
  hypothesis** (gaps concentrate in PARTIAL/low-cosine extraction) **is not
  supported** — and here the FOUND stratum is actually the *harder* one for
  navigation, because PARTIAL questions are disproportionately multi-article and
  topically concentrated where the chapter titles are more navigable.

---

## 7. Navigation cost (no latency)

Per-query LLM calls and tokens (`cost` dicts; `arm2b_per_question.csv`). Latency
is excluded per the brief.

- **5.82 LLM calls / query** (≈1 chapter-selection + ~4 article-selection + 1
  evaluate; `follow_refs` essentially never fires — **0 cross-reference follows
  across all 725 queries**, iterations ≈ 0).
- **≈18.9k tokens / query** (in+out), ranging 13.7k (Code du Logement) → 21.6k
  (Code Judiciaire-larger): cost scales with the chapter/article fan-out, not with
  success.
- **`exit_reason` ≈ 98% `sufficient`** — the navigator almost always terminates
  satisfied at iteration 0; the 17 non-`sufficient` queries are `no_candidates`
  (the chapter step picked nothing valid) or one `parse_fail`. The model's
  *confidence* (it stops, "sufficient") is decoupled from its *accuracy* (80% of
  gold is in an unselected chapter) — a notable calibration failure for an
  LLM-as-retriever.

---

## 8. What the failure modes reveal about LLM tree navigation

Only the failures that say something about *LLM tree navigation as a retrieval
method* belong in the story; ordinary implementation slips (a parser that dropped
`list[dict]` responses; hard-coded example IDs the 8B model echoed verbatim —
itself half a symptom of the truncation below) carry no methodological insight and
are excluded. What follows keeps only the three failures that *characterise the
approach*. Sources: `local_postprocess_experiments.csv`;
[EXECUTION_PARAMETERS.md](../../Report/EXECUTION_PARAMETERS.md) §4; the tree manifests.

**The foundation that never failed — the tree-build.** PageIndex is two stages:
*compile* each PDF into a deterministic Law→Chapter→Article ToC tree (the vectorless
"index"), then *navigate* it with the 8B LLM. The compile step is the quietly-hard
one, because chapter structure is **not** a clean field — AzureDI's `node_source` is
uniformly the document title and unusable (RQ2 §4.3), so `tree_builder.py`
*reconstructs* chapters by walking each article's `parent_id` chain to its header
nodes and cleaning the markdown-prefixed `page_content` of those headers — no LLM at
build time, cache-keyed on `pdf_sha256`. Despite that indirect derivation,
**version-1 succeeded on all five PDFs**: `chapter_derivable = True` and
**`bsard_id_coverage = 1.0`** everywhere (39–129 chapters, 257–767 articles). §4
confirms it from the retrieval side — only **2 of 2 874** gold articles are absent
from any tree. **The insight is the attribution it licenses:** because the index is
essentially complete, every downstream miss is a *navigation* failure, never an
indexing one. A sound tree-build is what makes the rest of this story interpretable.

**Failure 1 — the method is context-hungry, and it fails silently.** The
chapter-selection prompt must list every chapter of a code, so on real legislation
it runs **6–9k tokens**. The local navigator was called at Ollama's default
`num_ctx = 4096`, which **truncated those prompts from the top with no error** —
`tokens_in` pinned at exactly 4096 on 252/252 queries — so the model was choosing
chapters from a list it had never fully seen. *Why it caused failure, and why it is
an insight:* this is not a typo, it is a structural property of hierarchical LLM
navigation — prompt size scales with the **breadth of the hierarchy**, so the codes
with the most chapters are the ones that overrun a naive context window, and a local
LLM degrades *silently* (truncation, not an exception) rather than refusing. Raising
`num_ctx` to 16384 (with a `q8_0` KV cache to fit a T4) restored the full chapter
list — `tokens_in/step` 4096 → **18 790**. Context budget is a first-class design
parameter for this method, not an afterthought.

**Failure 2 — when the LLM is the selector, its conservatism *is* the recall
ceiling.** Even with the right chapter in hand and full context, the original
article-selection prompt led the 8B model to return an **empty `selected_article_ids`
65–84% of the time** (PDFs 1867/1967/2003). *Why it caused failure, and why it is an
insight:* in a vectorless design there is no score threshold to relax and no deeper
pool to fall back on — the model's output *is* the candidate set, so a precision-biased
"when unsure, return nothing" disposition translates one-to-one into zero recall. A
single prompt clause ("*Préférez l'inclusion … ne retournez une liste vide que si
aucun article n'a un lien thématique plausible*") lifted the selection rate and was
worth real recall. This is the same behaviour the dive measures downstream: a median
exposed head of **4 articles** and the **11.2%** "chapter-ok / article-missed" slice
(§3–§4) are this conservatism, now quantified rather than fatal.

**Failure 3 — the LLM picks chapters better than it picks articles, and that
asymmetry has a hard ceiling.** Once navigation ran, its few picks were padded to
100 for comparability with the ~100-candidate vector arms. A re-derivation over the
*same* traces (no extra LLM calls) swept the padding **scope**:

| padding scope | R@10 | R@100 |
|---|---:|---:|
| `selected_law` (original default) | 0.124 | 0.247 |
| `selected_chapter` first | 0.200 | 0.296 |
| **`selected_chapter` THEN `selected_law` (shipped)** | **0.203** | **0.393** |

*Why it worked, and why it could not do more.* Padding inside the *chosen chapters*
first (+63% rel R@10) works because the model reaches the right chapter more reliably
than it picks the right article inside it — chapter-scoped padding simply harvests the
gold it left behind there. But the *same* asymmetry bounds the gain: chapter padding
can only ever recover the **11.2%** chapter-ok slice (§4); it is structurally blind to
the **79.9%** of gold in chapters the LLM **never selected**. So the two-tier pad lifts
R@10 by ~63% relative *and still leaves the arm at 0.185* — the dominant failure is
upstream of anything padding can touch. (Tie-breakers and backfill were washes;
`RRF(T05, T04)` added nothing.) Shipped in `navigator.py`, replayed onto disk via
`rederive_padding.py`; this dive scores that `results/` snapshot.

**The through-line.** Strip the plumbing bugs and three method-level facts remain:
the tree-build (index) is **complete and not the problem**; the navigator is
**context-hungry and silently truncates**; and as a selector it is **precision-biased
and chapter-strong / article-weak**. All three point the same way — to a *navigation*
ceiling on top of a sound index — which is exactly the §4 finding that 79.9% of gold
sits in chapters an 8B model never reached. **PageIndex's index works; its reasoner is
the bottleneck.**

---

## 9. Summary of findings

1. **Arm 2B is the weakest arm and its recall is padding-propped.** R@10 0.185,
   Hit@10 0.265 (vs ≈0.47–0.48 / 0.63–0.69 for the other arms). Exposed-head recall
   is only 0.137; R@100 reaches 0.404 mostly through deterministic chapter/law
   padding — so its deep recall is not a navigation result, and only **44% of hits
   put the first gold in the LLM's exposed picks**.
2. **The head is shallow *and* noisy.** Median 4 exposed picks, exposed precision
   0.091 — lower than Arm-2A's 10-node P@10 — so the LLM's deliberate selections
   are mostly distractors, contrary to the "clean but shallow" expectation.
3. **The miss mechanism is wrong-chapter selection.** 79.9% of all gold sit in
   chapters the LLM never selected; only 11.2% are right-chapter/wrong-article and
   just 2 of 2 874 gold articles are absent from the tree (the same two missing
   from Arm-2A's index — a shared corpus gap). The ~51% entire-miss rate is the 8B
   navigator picking the wrong subtree, not an indexing or within-chapter failure.
4. **Confidence is decoupled from accuracy.** 98% `sufficient` exits at iteration 0,
   ~5.8 calls / ~19k tokens per query, zero cross-reference follows — the navigator
   reliably *commits* to a wrong branch. Multi-article and PARTIAL-extraction
   questions fare *better*, again refuting ANALYSIS_PLAN-C2 for this arm.

*(Built on the shared `deep_dive_common` engine — Group-A/D metrics are identical
code to the Arm-1/2A dives (re-confirmed byte-identical after this build). The
Arm-1 reference values (R@10 0.472, Hit@10 0.630, median rank 4) and Arm-2A
(R@10 0.482, Hit@10 0.691, median rank 2) cited for context are from
`reports/arm1_naive_deep_dive.md` / `reports/arm2a_metadata_deep_dive.md`; a formal
cross-arm join is deferred.)*

---

### Artifact index

| Artifact | Path |
|---|---|
| Per-question metrics (725 rows) | `data/tables/arm2b_per_question.{csv,md}` |
| Navigation-failure anatomy (2621 rows) | `data/tables/arm2b_nav_anatomy.{csv,md}` |
| Gold-landing table (2874 rows) | `data/tables/arm2b_gold_landing.{csv,md}` |
| Per-PDF summary | `data/tables/arm2b_per_pdf_summary.{csv,md}` · `Report/tables/tab_rq2_arm2b_per_pdf.tex` |
| Aggregate (micro+macro) | `data/tables/arm2b_aggregate_summary.{csv,md}` · `Report/tables/tab_rq2_arm2b_headline.tex` |
| Navigation anatomy summary | `data/tables/arm2b_nav_anatomy_summary.{csv,md}` · `Report/tables/tab_rq2_arm2b_navigation.tex` |
| Entire-miss summary | `data/tables/arm2b_entire_miss_summary.{csv,md}` |
| Strata | `data/tables/arm2b_strata_summary.{csv,md}` |
| Navigated-vs-padded recall (headline) + random-padding counterfactual | `Report/figures/fig_rq2_arm2b_navigated_vs_padded.{pdf,png}` |
| Navigation-failure anatomy (stacked) | `Report/figures/fig_rq2_arm2b_nav_anatomy.{pdf,png}` |
| Strata small-multiples (cardinality only) | `Report/figures/fig_rq2_arm2b_strata_smallmultiples.{pdf,png}` |
| ~~Rank-of-gold CDF~~ | *not included* |
| §8 recovery — four-bug investigation | `…/RQ2_T05_ARM2_PAGEINDEX/<stem>/experiments/REPORT.md` |
| §8 recovery — padding-scope sweep | `…/experiments/local_postprocess_experiments.csv` |
| §8 recovery — pre/post cross-arm (1804) | `…/experiments/cross_arm_comparison.csv` |
| §8 tree-build manifests (coverage = 1.0 ×5) | `…/RQ2_T05_ARM2_PAGEINDEX/<stem>/tree.json` (`manifest`) |
