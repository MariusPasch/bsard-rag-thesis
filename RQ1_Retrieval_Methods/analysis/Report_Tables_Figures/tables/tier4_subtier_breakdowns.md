### Table 5.4a — LLM-Judge: scoring paradigm × first stage × pool

| Scoring | First stage | Pool | R@10 | R@20 | R@50 | MRR@100 | NDCG@10 | Hit@10 |
|---|---|---|---|---|---|---|---|---|
| Binary | Hybrid | top-50 ★ | **0.445** | **0.518** | **0.580** | **0.406** | **0.358** | **0.653** |
| Binary | Hybrid | top-20 | 0.435 | 0.465 | 0.465 | 0.396 | 0.352 | 0.635 |
| Binary | Sparse | top-50 | 0.362 | 0.418 | 0.463 | 0.367 | 0.305 | 0.563 |
| 0–10 | Hybrid | top-50 | 0.272 | 0.379 | 0.463 | 0.265 | 0.222 | 0.473 |
| 0–10 | Hybrid | top-20 | 0.262 | 0.333 | 0.333 | 0.259 | 0.219 | 0.455 |

### Table 5.4b — CRAG: first-stage ablation

| First stage | R@10 | R@20 | R@50 | R@100 | MRR@100 | NDCG@10 | MAP@10 | Hit@10 |
|---|---|---|---|---|---|---|---|---|
| Hybrid RRF-k60 ★ | **0.426** | **0.465** | **0.582** | **0.654** | **0.401** | **0.346** | **0.268** | **0.626** |
| BM25 | 0.301 | 0.327 | 0.465 | 0.524 | 0.339 | 0.265 | 0.207 | 0.505 |

_Both at the final v2 config (LLaMA collective evaluator, eval_k 20, ≤6 iters); only the first stage differs._

### Table 5.4c — ReAct: v2 (final) vs v1

| Version | Key configuration | R@10 | R@20 | R@50 | MRR@100 | NDCG@10 | MAP@10 | Hit@10 |
|---|---|---|---|---|---|---|---|---|
| v2 (final) ★ | obs. window 3 · overlap guard 0.6 · function-calling · 8 steps · top-up + re-rank | **0.426** | **0.465** | **0.468** | **0.378** | **0.335** | **0.255** | **0.622** |
| v1 | obs. window 10 · no overlap guard · free-text actions · 5 steps | 0.277 | 0.292 | 0.292 | 0.269 | 0.228 | 0.171 | 0.432 |

_Same RRF-k60 first stage; the agent configuration differs._

★ headline configuration (carried into Table 5.4).
