| System | Size | Weighting | R@1 | R@5 | R@10 | R@50 | R@100 | MRR@100 | NDCG@10 | MAP@10 | Hit@10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Multilingual E5-large | 560M | text+meta | 0.098 | **0.245** | **0.342** | **0.563** | **0.622** | 0.322 | **0.270** | **0.202** | 0.523 |
| Qwen3-Embedding-0.6B (instruction-prefixed) | 600M | text | **0.108** | **0.245** | 0.339 | 0.512 | 0.597 | **0.328** | 0.268 | 0.197 | **0.559** |
| Multilingual E5-large | 560M | text | 0.083 | 0.211 | 0.296 | 0.486 | 0.594 | 0.300 | 0.230 | 0.163 | 0.505 |
| BGE-M3 | 568M | text | 0.082 | 0.222 | 0.314 | 0.509 | 0.592 | 0.303 | 0.242 | 0.174 | 0.514 |
| CamemBERT-large | 335M | text | 0.073 | 0.191 | 0.307 | 0.507 | 0.583 | 0.252 | 0.215 | 0.151 | 0.477 |
| Qwen3-Embedding-0.6B | 600M | text | 0.099 | 0.242 | 0.324 | 0.487 | 0.582 | 0.317 | 0.259 | 0.194 | 0.523 |
| Multilingual E5-base | 278M | text | 0.080 | 0.156 | 0.225 | 0.418 | 0.491 | 0.245 | 0.182 | 0.133 | 0.401 |
| Multilingual MPNet | 278M | text | 0.041 | 0.146 | 0.211 | 0.321 | 0.405 | 0.197 | 0.154 | 0.104 | 0.369 |
| CamemBERT-base | 110M | text | 0.006 | 0.007 | 0.008 | 0.016 | 0.017 | 0.022 | 0.010 | 0.007 | 0.027 |

_Size = approximate parameter count from the public model card (reference fact, not measured from the records)._
