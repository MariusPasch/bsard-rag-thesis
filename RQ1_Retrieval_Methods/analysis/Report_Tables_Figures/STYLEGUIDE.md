# Thesis Report Styleguide — RQ1 & RQ3 (BSARD Retrieval)

Project-specific specialization of the generic `STYLEGUIDE.md`. Anything not stated
here defers to that document. This file is the single authority for every figure
and table that appears in the report or the defence presentation.

**Scope.** Master thesis "Enhancing Performance and Quality of Context Retrieval in
RAG Systems" — RQ1 (lexical / dense / hybrid / agentic retrieval over BSARD) and
the RQ3 evaluator analysis kept in this repo (`analysis/RQ3/`). RQ2 lives in a
separate project and uses a parallel copy of this file.

---

## 1. The three layers, for this thesis

| Layer | Concrete location |
|---|---|
| **Data** (source of truth) | RQ1 result JSONs in `output/results/<tier>/`; RQ3 consolidated records in `output/results/RQ3/`; 48-q subset in `evaluation/data/tier3_subset.json`; strata in `evaluation/data/query_strata.json` |
| **Style** (one module) | `Report/thesis_style.py` (to be created) — exports `PALETTE`, `SYSTEMS`, `EVALUATORS`, `rcparams_report()`, `rcparams_slides()`, `save_figure()`, `save_table()` |
| **Export** | Outputs land in `Report/figures/` (`.pdf` + `.png` per fig) and `Report/tables/` (`.tex`, optional `.png`) |

Figures and tables are **never hand-edited** — regenerate from the canonical JSONs.

---

## 2. Locked palette — Okabe–Ito (colorblind-safe)

| Hex | Name | Reserved for |
|---|---|---|
| `#0072B2` | blue | Tier 1 — BM25 anchor |
| `#56B4E9` | sky blue | Tier 1 — tuned / variant BM25 |
| `#E69F00` | orange | Tier 2 — dense |
| `#009E73` | bluish green | Tier 3 — hybrid |
| `#D55E00` | vermillion | Tier 4.0 — LLM-Judge re-rank |
| `#CC79A7` | reddish purple | Tier 4.1 — CRAG (agentic) |
| `#117777` | dark teal | Tier 4.2 — ReAct (agentic) |
| `#000000` | black | RQ3 AQS composite only |
| `#F0E442` | yellow | **reserve** — strata highlights only (low contrast on white; never a primary series) |

**Sequential / heatmap:** `viridis`. **Diverging (correlation, deltas vs anchor):**
`vlag` (seaborn) or `RdBu_r`, always centred at 0.

---

## 3. Locked `system → (color, marker, linestyle)` — RQ1

The eight retrieval systems below are the ones that will *actually appear in plots
and tables*. Each is invariant across the entire thesis.

| Canonical key | Display name (report) | Display (slides) | Color | Marker | Line | Notes |
|---|---|---|---|---|---|---|
| `bm25_anchor` | BM25 anchor (Okapi, $k_1{=}1.5,\,b{=}0.75$) | BM25 (anchor) | `#0072B2` | `o` | `--` | Tier 1 anchor; paper-aligned |
| `bm25_tuned` | BM25 tuned ($b{=}0.25$, lemmatized) | BM25 (tuned) | `#56B4E9` | `D` | `-` | Sparse leg of T3-A; used in T4 first stages |
| `bm25_concat2x` | BM25 + concat-2x field weighting | BM25 (concat-2x) | `#56B4E9` | `s` | `:` | Best T1 R@100; **excluded from T4** (metadata leakage) |
| `dense_me5_large` | mE5-large, zero-shot, concat-2x | mE5-large | `#E69F00` | `D` | `-` | Tier 2 winner (EXP-D7); dense leg of T3-A |
| `hybrid_rrf_k60` | Hybrid RRF ($k{=}60$) | Hybrid (RRF) | `#009E73` | `D` | `-` | Tier 3 winner; first stage for all canonical T4 runs |
| `llm_rerank_top50` | LLM-Judge, binary, top-50 over Hybrid | LLM-Judge (top-50) | `#D55E00` | `D` | `-` | Canonical T4.0 |
| `llm_rerank_top20` | LLM-Judge, binary, top-20 over Hybrid | LLM-Judge (top-20) | `#D55E00` | `s` | `-.` | Matched-pool anchor for T4.1 / T4.2 |
| `crag_hybrid_v2` | CRAG (hybrid, v2) | CRAG | `#CC79A7` | `D` | `-` | Canonical T4.1 |
| `react_hybrid_v2` | ReAct (hybrid, v2) | ReAct | `#117777` | `D` | `-` | Canonical T4.2 |

**Ablation variants** (BM25-first-stage T4 runs, ReAct v1, linear-α / SGDR hybrid,
other dense models) share their family's color but use hollow markers (`mfc='white'`)
and `linestyle=':'`. They appear in tier-internal figures only; the headline
cross-tier figure shows the nine canonical rows above.

**Why these nine.** They are the entries cited in `RETRIEVAL_PROJECT.md` §5.1 as the
headline results. Every other run is a sensitivity check or ablation feeding one of
these.

---

## 4. Locked `evaluator → style` — RQ3

| Canonical key | Display name | Color | Marker |
|---|---|---|---|
| `umbrela` | UMBRELA (mean grade / 3) | `#E69F00` | `o` |
| `erag` | eRAG (mean) | `#009E73` | `s` |
| `ragas_wa` | RAGAS-WA (mean) | `#CC79A7` | `D` |
| `aqs` | AQS (composite, data-driven weights) | `#000000` | `*` |
| (T4.2 ReAct is recolored to `#117777` dark teal — see §3) | | | |
| `gold_ceiling` | Gold ceiling (oracle) | `#555555` | — — drawn as a horizontal dashed reference line, not a series |

When plotting RQ3 results *alongside* RQ1 systems on the same axes (e.g.
`R@10 vs AQS`), keep the RQ1 system styling above — the evaluator becomes an axis,
not a series. When plotting evaluators **as series** (e.g. evaluator agreement
heatmaps, density panels), use the table above.

---

## 5. Canonical display-name map

Defined once in `thesis_style.py` as `DISPLAY_NAMES: dict[str, str]`. Examples
already pinned above. Same English names are used in report and slides; slides may
substitute the shorter "Display (slides)" form when space is tight.

| Concept | Report term | Slide / caption shorthand |
|---|---|---|
| BSARD test split | "the 222-question BSARD test split" | "test (n=222)" |
| Stratified subset | "the 48-question stratified subset" | "subset (n=48)" |
| Recall@k metric | "Recall@$k$" | "R@$k$" |
| Corpus size (external) | "~22k unique articles" | "22k articles" |

Per the project memory rule, **never** quote the 33,741 raw count externally.

---

## 6. Metrics — decimal places, intervals, significance

| Metric | Decimals | CI / dispersion | Significance vs anchor |
|---|---|---|---|
| Recall@$k$, MRR@10, NDCG@10, MAP@10 | **3** (e.g. `0.402`) | 95 % paired-bootstrap over queries, ≥1000 resamples, seed=42 | paired two-sided t-test on per-query R@$k$, marker `†` for $p<0.05$ |
| Latency (ms) | 0 for $\geq 100$ ms; 1 otherwise | mean / std + p50 / p90 / p95 | not significance-tested |
| QPS | 1 | — | not significance-tested |
| Index build time | 1 (seconds) | — | not significance-tested |
| UMBRELA / eRAG / RAGAS-WA / AQS | **3** | same bootstrap protocol on the 48-q subset | Kendall-τ system-ranking agreement vs gold, not paired-t |
| Kendall-τ, Spearman-ρ, Cohen-κ | **3** | 95 % bootstrap | mark $p<0.05$ with `†` |
| $p$-values | **3** decimals if ≥ 0.001, else `<0.001` (never scientific notation in tables) | — | — |

**Anchors are tier-specific** (RETRIEVAL_PROJECT.md §4.2 — copied here so figure
captions don't have to chase them):

| Comparison | Anchor | Primary $k$ |
|---|---|---|
| Within Tier 1 | `bm25_anchor` | 10 |
| Within Tier 2 | `bm25 (k1=1.5, b=0.5, lemmatize, text_only)` (intentionally not the T1 winner) | 100 |
| Within Tier 3 | `dense_me5_large` (T2 winner) | 10 |
| Within Tier 4 (open) | `hybrid_rrf_k60` (T3-A) | 10 |
| Matched-pool agentic vs LLM-Judge | `llm_rerank_top20` | 10 |

**Matched-pool rule (non-negotiable).** When the displayed system is a reranker
with pool size $N$, **never** compare its R@$k$ to a first-stage R@$k$ for $k>N$.
Captions for any reranker figure must state the pool size explicitly. R@$k$ plateaus
at $k>N$ are a structural property of the pool, not a quality signal — call this
out in the caption.

---

## 7. Two profiles, same palette

| | **Report (LaTeX)** | **Slides (16:9 PPTX)** |
|---|---|---|
| Font family | `serif` (matches thesis body — Computer Modern or report-defined) | `sans-serif` (Inter / Source Sans / Arial — match slide master) |
| Body font size | 10 pt | 22 pt equivalent (≈ matplotlib 14 with high-res) |
| Figure width | one-column 3.3 in / full-width 6.7 in (set physical size, never rescale in LaTeX) | 6.5 in × 3.6 in (renders at 1280×720) |
| DPI for PNG | 300 | 300 |
| Saved formats | `.pdf` (vector) + `.png` (Word fallback) | `.png` only |
| Legend | inside axes, no frame | inside or below axes, no frame |
| Title | **none** — caption carries it | short title allowed |
| Gridlines | light horizontal only | light horizontal only |

`thesis_style.py` exposes `rcparams_report()` and `rcparams_slides()`; figure scripts
call one of them, never set rcParams ad-hoc.

---

## 8. Figure choices — what plot for which RQ1 / RQ3 result

These are the standard choices; deviations need a justification in the caption.

### RQ1

| Result type | Standard artifact |
|---|---|
| **Headline cross-tier comparison** (the nine canonical systems on R@10, R@100, latency) | One booktabs table (the "headline numbers" table, §5.1 of `RETRIEVAL_PROJECT.md`) + one **horizontal bar chart sorted by R@10** with $p<0.05$ markers |
| **Recall@$k$ curve over depth** ($k \in \{1, 5, 10, 20, 50, 100\}$) for the nine canonical systems | **Line plot, log-x**, markers at measured cutoffs, light shaded 95 % CI band. One figure for the headline; tier-internal repeats for each tier |
| **Tier 1 — BM25 k1×b grid** (val set) | **Heatmap** (`viridis`), annotate cell values to 3 dp, highlight winner |
| **Tier 2 — dense model comparison** | Sorted horizontal bars (R@10 and R@100 side-by-side as small multiples) + CI |
| **Tier 3 — RRF $k$-sweep, linear α-sweep, SGDR K-sweep** | One **line plot per parameter**, x = parameter, y = R@10 (primary) + R@100 (secondary axis or twin panel) |
| **Tier 4 — matched-pool comparison** | **Slope chart / dumbbell** of T3-A vs each T4 system at R@10, paired by query subset |
| **Recall vs latency Pareto** | **Scatter + Pareto frontier**, points labelled with system short names, log-x for latency, axes annotated with the matched-pool caveat |
| **Per-query distributions** (e.g. R@10 across queries for headline systems) | Box + strip overlay |
| **Stratified breakdowns** (single vs multi-article, lexical vs paraphrased, w/ vs w/o cross-refs) | **Small multiples**: 2 rows × 3 columns, one cell per stratum, same y-axis range, same systems |
| **Failure analysis** (e.g. lexical Jaccard distribution by question outcome) | Histogram / KDE overlaid |

### RQ3

| Result type | Standard artifact |
|---|---|
| **System-ranking agreement** (gold vs each evaluator, plus inter-evaluator) | **Correlation heatmap** (Kendall-τ + Spearman-ρ), `vlag` diverging, annotate cells |
| **Per-query rank scatter** (gold R@10 vs evaluator score) | Scatter, one panel per evaluator, identity line + LOESS, point alpha for density |
| **AQS weight space** | **Simplex / ternary plot** — already produced as `11a_aqs_weight_simplex.png`; regenerate via `thesis_style.py` for consistency |
| **AQS old vs new weights** | Dumbbell across systems — already produced as `11b_aqs_old_vs_new.png` |
| **Gold ceiling** | Sorted horizontal bars: evaluator score on retrieved top-10 vs evaluator score on gold; gap is the headline number |
| **Stratified evaluator behaviour** (R@10 vs evaluator by $n_{\text{gold}}$ bucket / lex-align stratum) | Dumbbell or grouped bars per stratum |
| **Score densities** (UMBRELA / eRAG / RAGAS-WA distributions) | KDE per evaluator, faceted by system family if relevant |

When in doubt, prefer **table over chart** for "many metrics × many systems".

---

## 9. Table conventions

1. **`booktabs` only** — `\toprule / \midrule / \bottomrule`, no vertical rules.
2. Decimal alignment via **siunitx `S` columns**. Right-align numbers, left-align
   labels.
3. **Bold = best in column** (subject to the matched-pool rule for rerankers).
   `†` = $p < 0.05$ against that tier's anchor.
4. Group related metrics under spanning headers (`\multicolumn{...}{c}{Recall@}`).
5. Standard headline-table column order:
   `System | R@1 | R@5 | R@10 | R@20 | R@50 | R@100 | MRR@10 | NDCG@10 | latency (ms) | first-stage / pool`.
6. Tables are self-contained — caption states dataset, $n$, what bold and `†`
   mean, and the matched-pool / pool-size caveat for any reranker row.
7. Define each styled DataFrame **once** (a function returning the styled `pd.io.formats.style.Styler`),
   then render to `.tex` (report) and PNG (slides) from that one spec.

---

## 10. File naming

Deterministic, no spaces, lowercase-underscore.

```
Report/
├── STYLEGUIDE.md              ← this file
├── thesis_style.py            ← palette, system map, save helpers
├── figures/
│   ├── fig_rq1_headline_bars_r10.{pdf,png}
│   ├── fig_rq1_recall_curve_logx.{pdf,png}
│   ├── fig_rq1_pareto_recall_latency.{pdf,png}
│   ├── fig_rq1_t1_bm25_grid.{pdf,png}
│   ├── fig_rq1_t2_dense_bars.{pdf,png}
│   ├── fig_rq1_t3_rrf_k_sweep.{pdf,png}
│   ├── fig_rq1_t3_linear_alpha_sweep.{pdf,png}
│   ├── fig_rq1_t4_matched_pool_dumbbell.{pdf,png}
│   ├── fig_rq1_strata_smallmultiples.{pdf,png}
│   ├── fig_rq3_eval_corr_heatmap.{pdf,png}
│   ├── fig_rq3_rank_scatter.{pdf,png}
│   ├── fig_rq3_aqs_weight_simplex.{pdf,png}
│   ├── fig_rq3_gold_ceiling.{pdf,png}
│   └── fig_rq3_eval_strata_dumbbell.{pdf,png}
└── tables/
    ├── tab_rq1_headline.tex
    ├── tab_rq1_t1_sparse.tex
    ├── tab_rq1_t2_dense.tex
    ├── tab_rq1_t3_hybrid.tex
    ├── tab_rq1_t4_agentic.tex
    ├── tab_rq1_matched_pool.tex
    ├── tab_rq1_strata.tex
    ├── tab_rq3_system_ranking_agreement.tex
    ├── tab_rq3_gold_ceiling.tex
    └── tab_rq3_aqs_weights.tex
```

`fig_*` / `tab_*` prefixes mirror chapter scope. This list is the **planned set**;
add new artifacts only when needed — keep the directory free of dead outputs.

---

## 11. Reproducibility checklist (per artifact)

Every figure / table script must:

1. Read from the canonical results path — never from a cached DataFrame in a
   notebook.
2. Apply `rcparams_report()` or `rcparams_slides()` at the top, before any plotting.
3. Pull system / evaluator styling from `SYSTEMS` / `EVALUATORS` dicts — no
   hard-coded colors.
4. Set the bootstrap seed (42) and the resample count (≥ 1000).
5. Call `save_figure(fig, "fig_<scope>_<name>")` or `save_table(df, "tab_<scope>_<name>")`.
6. Be idempotent — rerunning rebuilds the exact same bytes (modulo timestamps).

A "regenerate everything" entry-point script (`Report/build_all.py`) will be added
once more than two artifacts exist; until then, each script is run on demand.

---

## 12. What lives in `Report/` and what does not

**In `Report/`:** this styleguide, `thesis_style.py`, generated `.pdf` / `.png` /
`.tex` artifacts under `figures/` and `tables/`, and (later) chapter drafts of the
report and slide-deck source if hand-written content accumulates here.

**Not in `Report/`:** experiment code (lives in `retrieval/`, `evaluation/`,
`scripts/`), result JSONs (live in `output/results/`), exploratory notebooks (live
in `analysis/`), raw data (lives in `output/`). The Report folder is a *publishing*
target — it consumes the rest of the repo, it doesn't host it.
