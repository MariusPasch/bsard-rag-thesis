| Sub-tier | Mechanism | Pool | R@1 | R@5 | R@10 | R@20 | R@50 | R@100 | MRR@100 | NDCG@10 | MAP@10 | Hit@10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| — | Hybrid (RRF) first stage | full | 0.114 | 0.291 | 0.402 | 0.465 | 0.580 | 0.651 | 0.350 | 0.309 | 0.234 | 0.586 |
| T4.0 | LLM-Judge (binary) | top-50 | **0.136** | **0.366** | **0.445** | **0.518** | 0.580 | — | **0.406** | **0.358** | **0.275** | **0.653** |
| T4.0 | LLM-Judge (binary) | top-20 (matched) | **0.136** | 0.363 | 0.435 | 0.465 | 0.465 | — | 0.396 | 0.352 | 0.273 | 0.635 |
| T4.1 | CRAG | top-20 + loop | **0.136** | 0.352 | 0.426 | 0.465 | **0.582** | **0.654** | 0.401 | 0.346 | 0.268 | 0.626 |
| T4.2 | ReAct v2 | top-20 + loop | 0.127 | 0.324 | 0.426 | 0.465 | 0.468 | — | 0.378 | 0.335 | 0.255 | 0.622 |

_R@100 is omitted (—) for pool-limited re-rankers where it equals R@50; kept for CRAG (loop expands the pool) and the hybrid reference. R@50 is shown for all rows._
