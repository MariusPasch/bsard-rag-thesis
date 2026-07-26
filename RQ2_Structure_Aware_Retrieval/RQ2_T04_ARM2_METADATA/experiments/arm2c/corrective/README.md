# Arm-2C corrective loop (CRAG × ReAct) — tiered troubleshooting harness

A cheap, **additive** path to decide whether wrapping the single-pass Arm-2C
navigator in a CRAG-style corrective loop is worth a full GPU run. Nothing here
modifies the existing runs, results, figures, or `react_navigator.py` — it only
adds files under `corrective/` and outputs under `runs/_corrective_*`.

The "old" architecture = `react_navigator.navigate()` (one greedy DFS pass; prunes
the gold's branch at a visited node and never revisits it; exits `frontier_empty`
on ~11/40 budget). The "new" architecture = `corrective_navigator.navigate_corrective()`
(re-seed the frontier from *ranked* pruned sections to spend the idle budget on the
NOT_REACHED gold; final re-rank/commit on the reached pool for the selection gap).

## Tier (a) — zero-LLM counterfactual ceiling  *(run this first; no model)*

```
python corrective/corrective_ceiling.py
```
Reconstructs from the saved `enriched_rerank` traces + the deep tree which gold each
corrective mode *could* recover, and writes `runs/_corrective_ceiling/CEILING_SUMMARY.md`
+ `ceiling_<stem>.csv`. **Validation:** R@10_now reproduces the published headline R@10
exactly on all 5 PDFs, and `reached`/`selected` reproduce `load_arm2c.pure_comparison`
(Civil 0.470/0.081, Pénal & Housing selected 0.000).

These are **oracle ceilings** (perfect targeting + perfect inner descent). A real loop
recovers ≤ them. The `FINE (deep)` count and the `COARSE`/`ORPHAN` split are the
discounts.

### What it found (per-PDF, gold-article-weighted)

| PDF | R@10 now | FINE (deep) = re-nav upside | RESELECT = re-select upside | COARSE+ORPHAN = blind spot |
|---|--:|--|--|--|
| **Code Pénal** | 0.086 | **76% (88% deep)** | 11% | 8% / 0% |
| **Code Civil** | 0.327 | **54% (77% deep)** | 17% | 0% / 6% |
| **Code Judiciaire (larger)** | 0.271 | 24% | 30% | 8% / 13% |
| **Code Judiciaire (smaller)** | 0.097 | 12% | 26% | **56%** / 0% |
| **Housing (neg. control)** | 0.305 | 37% | 7% | 15% / 8% |

Oracle Recall@10 projections (question-weighted): re-navigate(FINE) lifts Pénal
0.086→**0.702**, Civil 0.327→**0.806**, Jud-larger 0.271→0.502; re-select lifts Civil
→0.470, Jud-larger →0.477.

### How to read it (the decision)

- **Re-navigate is the bigger lever on the depth/fine-prune codes** (Pénal, Civil,
  Jud-larger) — the agent enters the right top branch then prunes mid-tree, so the
  branch hangs off a committed node and a *ranked* re-seed can target it.
- **But FINE is mostly `deep`** (Pénal 88%, Civil 77%): the gold sits ≥3 edges below
  the committed node, so even after re-seeding the right section the 8B must make
  several more correct calls and will re-prune. The oracle ignores this, so the
  realized re-nav gain is **well below** the 0.70–0.81 ceiling, most severely on Pénal.
- **Re-select is the cheaper, safer lever** (no navigation risk; one extra call over
  the already-reached pool): Civil and Jud-larger each carry a ~0.47 re-select ceiling.
- **Code Judiciaire (smaller) is COARSE-dominated (56%)** — the agent never enters the
  gold's top branch. Backtracking re-seeds the same titles that already lost → the loop
  won't help; the real fix is a tree split of its 63-article section.
- **Housing is the negative control, with a twist:** the oracle re-nav ceiling looks
  high (0.637) but `selected = 0.000` — selection is collapsed, so reaching more gold
  *won't convert* unless re-select also works, which here it doesn't. The genuinely
  unrecoverable share is COARSE+ORPHAN = 23%. Fix selection before adding the loop.

## Tier (b) — control-flow test  *(no model; build-env acceptance gate)*

```
python corrective/test_corrective_mock.py
```
Drives `navigate_corrective()` with a scripted `MockLLM` over a synthetic tree and
asserts: memory carries across rounds; the re-seed pulls the round-1-pruned section;
round-1-alone misses the gold while the corrective round rescues it; the reached pool
grows monotonically; the loop terminates on the cap / when pruned sections run out; a
parse failure doesn't crash; `reseed_strategy="all"` takes the no-LLM path. All pass.

## Tier (c) — minimal real smoke  *(needs Ollama + llama3.1:8b → run on the Azure instance)*

Not runnable on the dev box (no Ollama). Runs OLD `navigate()` vs NEW
`navigate_corrective()` on the same queries and prints padding-free reach/selected
recall **and** padded R@10, old vs new. Two equivalent paths share one engine
(`smoke_corrective.run_smoke`), so they agree.

**Prerequisite (once):** the `corrective/` files must be on the instance. The instance
gets code by `git pull`, so commit + push `experiments/arm2c/corrective/` first. The
bundles (`deep_tree.json` + `queries.json`) are already committed.

### Path A — Azure terminal, one command

```bash
bash corrective/launch_smoke.sh --stem 1867_06_08_1867060850 --qids 1048,202,240
```
`launch_smoke.sh` resets + pins Ollama (`num_ctx 16384`, `keep_alive=-1` — avoids the
4096-truncation/idle-unload bugs), `git pull`s, ensures the `ollama` pkg, then runs the
smoke in the foreground. Other docs:

```bash
bash corrective/launch_smoke.sh --stem 1804_03_21_1804032150 --qids <R@10=0 qids> --max-rounds 2   # Civil
bash corrective/launch_smoke.sh --stem 2003_07_17_2013A31614 --qids <3 qids>                       # Housing (neg. control)
```

### Path B — notebook on the Azure kernel

Open `corrective/arm2c_corrective_smoke.ipynb` (regenerate with
`python corrective/_build_smoke_notebook.py`) from the cloned
`experiments/arm2c/corrective/` folder, attach the instance kernel, run top to bottom.
Edit cell 1 (`STEM`, `QIDS`, `MODE`, `MAX_ROUNDS`, `RESEED_STRATEGY`). The Ollama
warm/pin cell mirrors the existing Arm-2C notebooks.

### Notes
- Test order (plan): Pénal + Civil first, Housing as the negative control. Pick
  `--qids` that score `R@10=0` in `runs/arm2c_<stem>_enriched_rerank/per_query.csv` so
  there is headroom to recover.
- `--reseed-strategy all` swaps the ranked re-seed for the brute-force control (re-seed
  the first *m* pruned sections, no ranking call) — use it to check the ranking call is
  earning its keep.
- Output (additive): `runs/_corrective_smoke/<stem>_smoke.json`.
- Validated end-to-end on the real Pénal bundle with a mock LLM (no Ollama): paths,
  schema, loop termination all clean — only LLM quality differs on the instance.

## Re-select lever — embedding rerank over the reached pool (option 1)

The smoke (`SMOKE_FINDINGS.md`) showed the loop *reaches* the gold but the vectorless
8B rerank can't lift it into the top-10. This experiment swaps that reranker for
**Arm-2A's e5 encoder** and reranks the **same reached pool** by query–article cosine
— isolating the reranker as the only variable. It reconstructs each query's reached
pool from the *shipped* single-pass traces, so it needs **only the e5 model (no
Ollama)** and runs over the full query set. It answers: the shipped run reaches
R@100≈0.52 but ranks gold to only R@10≈0.33 — can the embedding ranker convert that
reached gold into the top-10?

```bash
# on the instance (first run downloads e5 ~2 GB; auto-uses GPU):
bash corrective/launch_embed_rerank.sh --stem 1804_03_21_1804032150              # Civil, raw article text
bash corrective/launch_embed_rerank.sh --stem 1804_03_21_1804032150 --text-field summary   # EN content-summary
bash corrective/launch_embed_rerank.sh --stem 1867_06_08_1867060850              # Penal
```

`E5Encoder` (`embed_rerank.py`) is byte-faithful to `RQ2_T01_SHARED/shared/embeddings`
(`intfloat/multilingual-e5-large-instruct`, `passage:`/`query:` prefixes, L2-norm,
cosine). Wiring is mock-testable: `--mock --limit N` uses a lexical encoder, no model
(validated: R@100 LLM/embed identical → pool reconstruction is faithful). Output:
`runs/_embed_rerank/<stem>_embed_<field>.{csv,md}`. Note: the reranker can only surface
gold that's *in the reached pool* (reach ceiling); padding-only gold is out of scope.

**E1 result (Code Civil, 252 q):** R@10 0.327 (8B) → **0.390 (e5)**, +0.063; conversion of
reachable gold into top-10 70% → 83%; R@100 0.519/0.519 (faithful pool). Embedding ranking
beats the 8B, but is capped by the single-pass reach (0.47) → still < Arm-1 0.471 / 2A 0.514.

## Full synthesis — E2: corrective navigation (reach) + e5 ranking

Closes the loop: ONE corrective navigation per query gives two pools (single-pass +
corrective), each ranked by both the 8B and e5 → **four R@10 off one navigation**. The
headline cell is **e5 over the corrective pool** (`e5/C`) — Arm-2C reach + Arm-2A ranking.

**Run it DECOUPLED (two phases).** Loading e5 *alongside* llama on the 16 GB T4 caused
VRAM contention that perturbed llama's greedy path and collapsed the corrective reach, so
run the navigator ALONE first, then the reranker alone:

```bash
# phase 1 — navigation only (Ollama), saves the pools; resumable, ~60-75 s/query
bash corrective/launch_e2_nav.sh --stem 1804_03_21_1804032150 \
     --qids 243,244,252,181,290,1057,158,159,1043,302 --max-rounds 2
# phase 2 — e5 rerank over the saved pools (e5 only, seconds); re-run for sweeps
bash corrective/launch_e2_rerank.sh --stem 1804_03_21_1804032150
bash corrective/launch_e2_rerank.sh --stem 1804_03_21_1804032150 --text-field summary
```
Phase-2 table: `8B/1 8B/C e5/1 e5/C` (ranker × {single, corrective} pool) + `reach1/reachC`.
Writes `runs/_corrective_e2/<stem>_{nav.json, e2_<field>.{csv,md}}`. Use ≥10 queries — the
corrective navigation is high-variance run-to-run, so small N is unreliable.

`smoke_e2.py` / `launch_e2.sh` do the same in ONE process (co-resident) — kept for
reference but **prefer the two-phase split** (it removes the contention confound).

## Files
- `corrective_ceiling.py` — tier (a), stdlib only.
- `corrective_navigator.py` — `navigate_corrective()` + helpers; reuses
  `react_navigator` machinery so round 1 == the old `navigate()`.
- `test_corrective_mock.py` — tier (b).
- `smoke_corrective.py` — tier (c) engine + CLI (`load_stem` / `run_smoke` shared).
- `launch_smoke.sh` — tier (c) Path A (one-command terminal launcher).
- `_build_smoke_notebook.py` → `arm2c_corrective_smoke.ipynb` — tier (c) Path B (Azure kernel).
- `embed_rerank.py` — re-select lever: `E5Encoder` (faithful to shared.embeddings) +
  reached-pool reconstruction + `embed_rerank_pool`.
- `eval_embed_rerank.py` — embed-rerank vs LLM-rerank over the shipped pool (e5 only,
  no Ollama, full sets; `--mock` for wiring).
- `launch_embed_rerank.sh` — one-command terminal launcher for the embed-rerank eval.
- `SMOKE_FINDINGS.md` — the locked tier-(c) corrective-loop finding (navigation solved,
  ranking is the bottleneck).
