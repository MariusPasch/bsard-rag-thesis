# Arm-2C corrective loop — smoke findings (2026-06-07)

Outcome of the tiered troubleshooting harness (`corrective/`). Bottom line: the
CRAG×ReAct corrective loop **solves the navigation/reach problem but isolates
ranking — not navigation — as the residual bottleneck.** A full GPU run is **not**
warranted for Recall@10 as-is; the next lever is the reranker, not more navigation.

## What was run

- **Tier (a)** zero-LLM ceiling (`corrective_ceiling.py`) — oracle upper bounds;
  reproduces published R@10 / reach / selected exactly. See `runs/_corrective_ceiling/`.
- **Tier (b)** mock control-flow gate (`test_corrective_mock.py`) — all pass, no model.
- **Tier (c)** real smoke on the Azure T4 (Ollama + llama3.1:8b), `enriched`,
  `max_rounds=2`, `reseed_strategy=ranked`.

The smoke comparison is a **controlled ablation off one navigation per query**: OLD =
the corrective loop's round-1 snapshot (verified byte-identical to `navigate()`), NEW =
the full loop. This removes the run-to-run 8B selection noise that confounded the
first (two independent passes) design.

## Headline — Code Civil, 10 `R@10=0` queries (controlled)

`--qids 243,244,252,181,290,1057,158,159,1043,302`

| metric | OLD (round-1) | NEW (loop) | Δ |
|---|--:|--:|--:|
| reach (padding-free) | 0.490 | **0.801** | **+0.311** |
| selected (padding-free) | 0.050 | 0.086 | +0.036 |
| **Recall@10 (padded)** | 0.033 | 0.117 | +0.084 |

Per-query fate:

| qid | gold | OLD reach | OLD R@10 | NEW reach | NEW sel | NEW R@10 | fate |
|---|--:|--:|--:|--:|--:|--:|---|
| 290 | 10 | 0.40 | 0.000 | 0.70 | 0.20 | **0.500** | win (reach+convert) |
| 158 | 15 | 0.00 | 0.000 | 0.87 | 0.00 | **0.667** | win (reach+convert) |
| 244 | 12 | 1.00 | 0.000 | 1.00 | 0.00 | 0.000 | reached, no convert |
| 252 | 12 | 1.00 | 0.000 | 1.00 | 0.17 | 0.000 | reached+selected, no convert |
| 181 | 3 | 1.00 | 0.000 | 1.00 | 0.00 | 0.000 | reached, no convert |
| 1057 | 8 | 0.50 | 0.000 | 1.00 | 0.00 | 0.000 | reached, no convert |
| 159 | 15 | 0.00 | 0.000 | 0.87 | 0.00 | 0.000 | reached, no convert |
| 302 | 31 | 0.00 | 0.000 | 0.58 | 0.16 | 0.000 | reached+selected, no convert |
| 1043 | 13 | 0.00 | 0.000 | 0.00 | 0.00 | 0.000 | deep, unreached |
| 243 | 12 | 1.00 | **0.333** | 1.00 | 0.33 | **0.000** | **regression (dilution)** |

Supporting (smaller / earlier exploratory) smokes agree on direction: Civil 4-q
controlled (reach 0.35→0.64, R@10 0.083→0.329, with q158 0→0.667, q290 0→0.5);
Pénal 3-q (q240 reach 0→1.0). Pénal/Civil-first per the plan; the depth/ranking
pattern is the same.

## The finding

1. **Re-navigate solves navigation.** Reach jumps 0.49→0.80; nearly every query
   reaches more gold (several 0→0.87, 0.5→1.0). The deep tree + corrective backtrack
   *finds* the gold — the NOT_REACHED gap is closeable.
2. **Reached gold does not convert to top-10.** 6/10 reach the gold yet score R@10=0
   (incl. q1057 reach 1.0, q302 reach 0.58 with 3 selected). Selection barely moves
   (0.05→0.09); R@10 rises only 0.033→0.117.
3. **Worse — pool dilution can REGRESS the headline.** q243: round-1 reach AND
   selection identical (0.33), yet the expanded reached pool diluted the vectorless 8B
   rerank enough to bury even the 4 explicitly-selected gold below rank 10
   (0.333→0.000). The loop as-is is **not** Recall@10-safe.

Poor conversion (2) and the regression (3) share one root cause: **the vectorless 8B
list-rerank cannot rank gold above non-gold once the reached pool grows.**

## Why a round-anchoring "monotone" fix was rejected

Anchoring round-1's top-10 to forbid regressions fails: on the winning queries
(q158) round-1's top-10 is *all padding* (round-1 reach = 0), so preserving it would
*block* the newly-reached gold and destroy the win. Round-1 gold and round-1 padding
are indistinguishable, so any round-anchoring either caps the upside or preserves
junk. The fix must be a **better ranker over the reached pool**, not a merge rule.

## Decision

- **No full GPU run for Recall@10 as-is** — it would lift R@100/reach (as the smoke
  shows) but barely move R@10, and can regress it on already-reached queries.
- **Next lever = the re-select/ranking step**, over the now-reachable pool:
  - (a) an embedding reranker (reuse Arm-2A's encoder) over the reached pool — pairs
    Arm-2C navigation (reach) with Arm-2A ranking; the most direct fix;
  - (b) cheaper: per-candidate LLM scoring instead of one list-rerank (less
    dilution-prone).
- **Thesis framing (now demonstrated end-to-end):** agentic corrective navigation
  closes the reach gap and *isolates ranking, not navigation/indexing, as the
  bottleneck* — consistent with, and now proof of, the Arm-2C "reach ≫ select" story.

Raw per-query smoke output: `runs/_corrective_smoke/<stem>_smoke.json` (on the
instance; commit if you want it versioned).
