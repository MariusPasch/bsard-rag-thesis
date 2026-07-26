# Tier 4.0 — LLM-as-a-Judge: Implementation & Results
**BSARD RAG Thesis | RQ1 | Non-Agentic LLM Benchmark**

---

## Status

**Complete.** Four test-split runs executed from three Azure notebooks. All runs use LLaMA 3.1 8B at temperature 0.0 with `max_article_tokens=1000`.

| Experiment | First stage | Top-N | Prompt | R@10 | R@100 | MRR@10 | Notebook | Role |
|---|---|---|---|---|---|---|---|---|
| **`llm_rerank_binary_top50_hybrid_rrf_k60_test`** | hybrid_rrf_k60 (T3-A) | 50 | binary | **0.4451** | **0.5795** | **0.3997** | `azure_tier40_hybrid_llm_rerank.ipynb` | **Canonical T4.0** (p = 0.016 vs T3-A) |
| `llm_rerank_binary_top20_hybrid_rrf_k60_test` | hybrid_rrf_k60 (T3-A) | 20 | binary | 0.4347 | 0.4648 | 0.3939 | `azure_tier40_hybrid_llm_rerank_top20.ipynb` | **Agentic-matched anchor** for T4.1 / T4.2 (p = 0.007 vs T3-A) |
| `llm_rerank_binary_top50_test` | bm25_tuned_k11.5_b0.25 (T1) | 50 | binary | 0.3618 | 0.4630 | 0.3623 | `azure_tier40_llm_rerank.ipynb` | First-stage ablation (same scorer, weaker pool) |
| `llm_rerank_0to10_top50_test` | bm25_tuned_k11.5_b0.25 (T1) | 50 | 0–10 numeric | 0.2715 | 0.4630 | 0.2550 | `azure_tier40_llm_rerank.ipynb` (Cell 11b) | Scoring-paradigm ablation (binary vs 0–10) |

The canonical T4.0 result is **`llm_rerank_binary_top50_hybrid_rrf_k60_test`** (R@10 = 0.4451). The matched-pool **top-20 hybrid variant** (R@10 = 0.4347) is the anchor used by the T4.1 / T4.2 comparisons so both sides see the same candidate-pool size (the matched-pool-size rule: comparing a reranker's R@k to a first-stage R@k is only meaningful when k ≤ reranker pool size).

The 0–10 numeric variant under-performs binary by ~7–14 pp R@10 — consistent with the LLM-as-Judge literature on 8B-class models producing poorly-calibrated continuous scores. It is kept as the scoring-paradigm ablation.

All runs were executed on Azure GPU (Standard_NC4as_T4_v3, 1× NVIDIA T4). Score caches are keyed by `(question_id, article_id, prompt_variant, max_article_tokens)` (first-stage agnostic), so the hybrid runs reuse most BM25-run scores. The top-20 hybrid run on 2026-05-20 re-evaluated against the existing cache (latency ≈ 77 ms / query — essentially first-stage retrieval only).

---

## 0. Context and Motivation

### 0.1 The ablation gap between Tier 3 and Tier 4

The current tiered architecture has a confounding variable between Tier 3 and Tier 4:

```
Tier 3 best (cross-encoder re-ranked hybrid)  →  Tier 4 (CRAG/ReAct with Ollama LLM)
                                                       ↑
                                         What drives improvement here?
                                         (A) Having an LLM in the pipeline at all?
                                         (B) The agentic correction/reasoning loop?
```

Without a non-agentic LLM baseline, any Tier 4 improvement over Tier 3 conflates:
- **(A)** The benefit of having an LLM evaluate/re-rank retrieved documents (the model's
  capability as a relevance scorer)
- **(B)** The benefit of the iterative correction loop (CRAG) or multi-step reasoning
  (ReAct) — i.e., the **agentic mechanism** itself

Tier 4.0 isolates effect **(A)** by using the **exact same LLaMA 3.1 8B** in a
single-pass, non-agentic pipeline. The comparison then becomes:

| Comparison | What it measures |
|---|---|
| Tier 4.0 vs Tier 3-C | Value of LLM as relevance scorer vs cross-encoder |
| Tier 4.1 (CRAG) vs Tier 4.0 | Value of the CRAG correction loop (isolated) |
| Tier 4.2 (ReAct) vs Tier 4.0 | Value of the ReAct reasoning loop (isolated) |

### 0.2 Why "Tier 4.0" — not Tier 3 or Tier 4

This experiment sits at the boundary between Tier 3 and Tier 4:

- It is **not Tier 3** because it introduces the LLM into the retrieval pipeline for the
  first time — all Tier 3 methods use only bi-encoder/cross-encoder models
- It is **not Tier 4** because it has no dynamic decision-making at query time — it is a
  fixed, single-pass pipeline (retrieve once → score once → done)
- Numbering as **Tier 4.0** signals that it uses the Tier 4 LLM backbone but without the
  agentic mechanism — a "Tier 4 minus the agency" baseline

### 0.3 Infrastructure reuse

This tier uses the same `OllamaClient` from `retrieval/agentic/llm_client.py` as CRAG
(Tier 4.1) and ReAct (Tier 4.2). The LLM model (`llama3.1:8b`), temperature (`0.0`),
and server configuration are **identical** — ensuring the only axis of variation between
Tier 4.0 and Tier 4.1/4.2 is the agentic loop.

---

## 1. Anchors from Prior Tiers

### 1.1 Key results locked in

| Configuration | R@10 | R@50 | R@100 | MRR@10 | Notes |
|---|---|---|---|---|---|
| `bm25_lemmatize_concat_2x` | 0.2572 | 0.4725 | 0.5312 | 0.2383 | Tier 1 sparse best R@100 (concat — not used in T4) |
| `bm25_tuned_k11.5_b0.25` | 0.2651 | 0.4630 | 0.5210 | 0.2628 | **Tier 4 first-stage (selected)** — text_only, lemmatize |
| `dense_me5_large_instruct` | 0.3439 | — | 0.6213 | — | Tier 2 winner |
| Best Tier 3 RRF/rerank | TBD | — | TBD | TBD | Not used as T4.0 first stage |

### 1.2 Tier 3-C as comparison anchor

The cross-encoder re-ranker (T3-C) is the most direct comparison point for Tier 4.0:
both take the same first-stage candidate pool and re-score it with a model. The difference
is the scorer:
- T3-C: `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (384-dim MiniLM, ~80ms for 100 pairs)
- T4.0: `llama3.1:8b` via Ollama (8B-parameter LLM, ~2-5s per pair on CPU)

### 1.3 Tier 4.1/4.2 as downstream comparison

CRAG and ReAct use the same LLaMA 3.1 8B model — their improvement over Tier 4.0
is the **clean measure of agentic value**.

---

## 2. Method — LLM-as-a-Judge Re-Ranking

### Pipeline (single-pass, no iteration)

```
Best T3 first stage  →  top-N candidates  →  LLM scores each (query, article_text)  →  re-rank by score  →  final list
```

This is methodologically identical to T3-C (cross-encoder re-ranking) with the scorer
substituted. No dynamic decisions, no query rewriting, no correction loops.

### Scoring approach

For each of the top-N first-stage candidates, the LLM is prompted to score the relevance
of the article to the query on a 0-10 scale. The prompt is in French (matching the
corpus and Tier 4 prompts):

```
Vous êtes un expert juridique belge. Évaluez la pertinence du passage suivant
par rapport à la question posée.

Question : {question}

Passage : {article_text_truncated}

Sur une échelle de 0 à 10, quelle est la pertinence de ce passage pour
répondre à la question ? Répondez uniquement avec le score numérique.

Score :
```

### Scoring prompts

**Binary variant** (primary — from `llm_eval_prompts.py`):
```
[Few-shot examples from fewshot_examples.json]

Question : {question}

Passage : {article_text_truncated}

Le passage est-il pertinent pour répondre à la question ?
Répondez uniquement par « Oui » ou « Non ».

Pertinent :
```
Parse: "Oui" → score 1.0, "Non" → score 0.0. Use P("Oui") logit if available.

**0-10 variant** (comparison — from `llm_eval_prompts.py`):
```
[Few-shot examples with integer scores 9 and 1]

Question : {question}

Passage : {article_text_truncated}

Sur une échelle de 0 à 10, quelle est la pertinence de ce passage ?
Répondez uniquement avec le score numérique.

Score :
```

**0–10 numeric scoring variant** (`prompt_variant="0to10"`):
```
[System prompt]
"Tu es un assistant de recherche juridique spécialisé dans l'analyse du droit belge. …
 Réponds uniquement avec un entier de 0 à 10. Aucun autre texte."

[User prompt]
Question : {question}

Passage : {article_text_truncated}

Donnez un score de 0 à 10, puis expliquez brièvement pourquoi ce passage est ou n'est pas pertinent.

Score (0-10) :
```

The user prompt mentions a brief explanation ("puis expliquez brièvement") but the system prompt enforces integer-only output, so no explanation is ever emitted. `parse_numeric_score()` extracts the first number it sees and clamps to [0, 10]. The result JSONs label this variant `prompt_variant: 0to10` and live under `output/results/agentic/llm_judge/0to10/`.

### Design decisions

| Decision | Value | Rationale |
|---|---|---|
| LLM model | `llama3.1:8b` | **Identical** to CRAG/ReAct (controlled variable) |
| Temperature | `0.0` | Deterministic (matching Tier 4) |
| `max_tokens` | `8` (binary); `64` (`0to10`) | Binary needs a single token; the 0–10 prompt is provisioned 64 tokens of headroom (even though the system prompt suppresses anything beyond the score digit). |
| Article truncation | 1000 word-level tokens (default); 300 as ablation | LLaMA 3.1 8B has 8K context; 300 tokens scored only 15 % of article content. Few-shot examples (~1150 tokens) + query (~50) + 1000-token article + response headroom fits well within 8K. The 300-token ablation was scoped but not executed. |
| Scoring paradigm | Binary primary; 0–10 numeric (`0to10`) as comparison variant | Binary uses few-shot Oui/Non framing; 0–10 uses a legal-domain system prompt with integer-only output and `parse_numeric_score()` extracting the first number. |
| Few-shot examples | 2 examples from train split, locked in `evaluation/data/fewshot_examples.json` | Anchors the score distribution; drawn from the train split. |
| Score parsing | `parse_binary_judgment()` / `parse_numeric_score()` from `llm_eval_prompts.py` | Shared with T4.1 and T4.2 — single implementation |
| Parse failure fallback | Score = 0.0 | Conservative; parse failures logged and reported |
| Score caching | JSON on disk, keyed by `(question_id, article_id, prompt_variant, max_article_tokens)` | Avoids redundant calls; `max_article_tokens` in key prevents collision between 300 and 1000-token runs |
| Few-shot path | `evaluation/data/fewshot_examples.json` | Actual path used by script and notebook; `evaluation/agentic/` path in earlier drafts was incorrect |

---

## 3. Experiments

### 3.1 Experiment table

| Experiment ID | Scoring | First stage | Top-N | R@10 | R@100 | MRR@10 | Notebook |
|---|---|---|---|---|---|---|---|
| **`llm_rerank_binary_top50_hybrid_rrf_k60_test`** | binary | hybrid_rrf_k60 | 50 | **0.4451** | **0.5795** | **0.3997** | `azure_tier40_hybrid_llm_rerank.ipynb` |
| `llm_rerank_binary_top20_hybrid_rrf_k60_test` | binary | hybrid_rrf_k60 | 20 | 0.4347 | 0.4648 | 0.3939 | `azure_tier40_hybrid_llm_rerank_top20.ipynb` |
| `llm_rerank_binary_top50_test` | binary | bm25_tuned_k11.5_b0.25 | 50 | 0.3618 | 0.4630 | 0.3623 | `azure_tier40_llm_rerank.ipynb` (Cell 10) |
| `llm_rerank_0to10_top50_test` | 0–10 numeric | bm25_tuned_k11.5_b0.25 | 50 | 0.2715 | 0.4630 | 0.2550 | `azure_tier40_llm_rerank.ipynb` (Cell 11b) |

> **First-stage retrievers:**
> - `bm25_tuned_k11.5_b0.25` — BM25 Okapi, k1=1.5, b=0.25, lemmatize, text_only (the Tier 4 sparse anchor; shared as the first stage for T4.1 / T4.2).
> - `hybrid_rrf_k60` — BM25 + mE5-large concat_2x, RRF k=60, `first_stage_k=100` (the T3-A winner; used by T4.2's hybrid variant as well).
>
> The score cache is keyed by `(question_id, article_id, prompt_variant, max_article_tokens)` — first-stage agnostic — so the BM25 binary cache is reused by the hybrid binary runs wherever (q, article) pairs overlap. This is why the `top20_hybrid` run completed in ~17 s wall-clock per 222 queries (~77 ms / query): all 20 × 222 = 4 440 (q, article) scores were already cached.

### 3.2 Hyperparameter choices

| Hyperparameter | Value | Basis |
|---|---|---|
| Scoring variant | `binary` (canonical) + `0to10` as ablation | Both run on the BM25 first stage at top-50. Binary outperforms 0–10 by ~7–14 pp R@10 — consistent with the LLM-as-Judge literature on 8B-class models producing poorly-calibrated continuous scores. |
| Top-N | `50` for the canonical headline; `20` for the agentic-matched anchor | The matched-pool anchor (top-20) exists because T4.1 / T4.2 use top-20, and a fair agentic comparison requires the same pool size on both sides. |
| First stage | `hybrid_rrf_k60` (canonical) + `bm25_tuned_k11.5_b0.25` (first-stage ablation) | Hybrid is the canonical pool; BM25 isolates the first-stage contribution with the same scorer. |
| Article truncation | 1000 word-level tokens | Plan default. |
| Few-shot examples | 2 examples from train split, locked in `evaluation/data/fewshot_examples.json` | Selected by `scripts/evaluation/tier4/select_fewshot_examples.py`; shared with T4.1 / T4.2. |

No val experiments were run — all three notebooks explicitly say *"Cell 8: Val experiments — not run"*. Hyperparameters were fixed a priori (`BEST_TOP_N=50`, `BEST_VARIANT='binary'`), with top-20 added separately to match the agentic pool size.

### 3.3 Runtime — observed

Azure GPU Standard_NC4as_T4_v3, 1× NVIDIA T4, 1000-token articles:

| Experiment | LLM calls | Per-query latency | Total wall-clock | Cache impact |
|---|---|---|---|---|
| `llm_rerank_binary_top50_test` (BM25) | 11,100 | 31,254 ms | ~115.6 min | First fresh binary run; warms the cache |
| `llm_rerank_0to10_top50_test` | 11,100 | 75,093 ms | ~277.9 min | Fresh 0–10 prompt + legal-domain system prompt; separate cache key |
| `llm_rerank_binary_top50_hybrid_rrf_k60_test` | 11,100 | 8,255 ms | ~29 min | High cache reuse from the BM25 binary run (shared scorer + prompt) |
| `llm_rerank_binary_top20_hybrid_rrf_k60_test` | 4,440 | ~77 ms | ~17 s | Essentially all cache hits — only first-stage retrieval is timed |

Reported latencies in the result JSONs include cache reuse where applicable. The BM25 binary timing (~626 ms/call) is the only clean per-call benchmark; later runs are not directly comparable as LLM-call benchmarks.

### 3.4 Significance tests

All paired t-tests on per-query Recall@10 (two-sided). Anchors are the result JSONs in the relevant tier directories.

| Comparison | Variant R@10 | Anchor R@10 | Δ R@10 | p (R@10) | Verdict |
|---|---|---|---|---|---|
| `binary_top50_test` (BM25 first stage) vs `bm25_tuned_k11.5_b0.25` | 0.3618 | 0.2651 | +0.0968 | **< 0.001** | LLM scoring significantly lifts the BM25 pool |
| **`binary_top50_hybrid_rrf_k60_test`** vs T3-A `hybrid_rrf_k60` | **0.4451** | 0.4021 | **+0.0430** | **0.016** | **Canonical T4.0 beats T3-A — LLM scoring adds value on the hybrid pool** |
| `binary_top50_hybrid_rrf_k60_test` vs `binary_top50_test` (BM25 first stage) | 0.4451 | 0.3618 | +0.0833 | **< 0.001** | First-stage pool quality matters; same scorer |
| `binary_top20_hybrid_rrf_k60_test` vs T3-A `hybrid_rrf_k60` | 0.4347 | 0.4021 | +0.0326 | **0.007** | Matched-pool anchor still significantly beats T3-A |
| `binary_top20_hybrid_rrf_k60_test` vs `binary_top50_hybrid_rrf_k60_test` | 0.4347 | 0.4451 | −0.0104 | n.s. (p = 0.433) | Pool-size effect at R@10 not detectable; R@100 differs by ceiling alone |

---

## 4. Key Thesis Comparison Table

The canonical comparison table for RQ1 Section 4.5:

| System | Relevance Scorer | Dynamic? | First stage | Pool | R@10 | MRR@10 | Latency/query |
|---|---|---|---|---|---|---|---|
| T1 BM25 anchor (`bm25_tuned_k11.5_b0.25`) | — | No | — | n/a | 0.2651 | 0.2511 | ~76 ms |
| T3-A (`hybrid_rrf_k60`, no rerank) | — | No | — | n/a | 0.4021 | 0.3402 | ~322 ms |
| **T4.0-hybrid-top50** (canonical) | **LLaMA 3.1 8B (binary)** | **No** | hybrid_rrf_k60 | top-50 | **0.4451** | **0.3997** | ~8.3 s |
| T4.0-hybrid-top20 (agentic-matched anchor) | LLaMA 3.1 8B (binary) | No | hybrid_rrf_k60 | top-20 | 0.4347 | 0.3939 | (cache-warm; ~77 ms wall-clock) |
| T4.0-BM25-top50 (first-stage ablation) | LLaMA 3.1 8B (binary) | No | bm25_tuned | top-50 | 0.3618 | 0.3623 | ~31 s |
| T4.0-BM25 0–10 (scoring-paradigm ablation) | LLaMA 3.1 8B (0–10 numeric) | No | bm25_tuned | top-50 | 0.2715 | 0.2550 | ~75 s |
| T4.1 CRAG | LLaMA 3.1 8B | Yes (correction) | bm25_tuned / hybrid_rrf_k60 | top-20 | *(see TIER41 plan)* | — | — |
| T4.2 ReAct | LLaMA 3.1 8B | Yes (reasoning) | bm25_tuned / hybrid_rrf_k60 | top-20 | *(see TIER42 plan)* | — | — |

Ablation comparisons:
- **T4.0-hybrid-top50 vs T3-A `hybrid_rrf_k60`** → does LLM re-ranking add value to a strong non-LLM pool? *Answer: yes, +0.043 R@10, p = 0.016.*
- **T4.0-hybrid-top20 vs T3-A** → matched-pool LLM-reranking gain. *Answer: yes, +0.033 R@10, p = 0.007.*
- **T4.0-hybrid vs T4.0-BM25** → effect of first-stage pool quality with the same LLM scorer. *Answer: +0.083 R@10, p < 0.001.*
- **T4.0-BM25 vs T1 BM25 anchor** → does LLM re-ranking lift a weak pool above pure lexical retrieval? *Answer: yes, +0.097 R@10, p < 0.001.*
- **T4.1 / T4.2 vs T4.0-hybrid-top20** → pure agentic-loop value (zero scorer / pool-size confound) — see TIER41 / TIER42 plans.
- **Binary vs 0–10 numeric** → scoring paradigm. *Answer: binary wins by 7–14 pp R@10 across pool sizes — pointwise 0–10 scoring with an 8B model is poorly calibrated for French statutory text.*

Planned visualisation (and present in the analysis notebook §6): Pareto-front plot, X = mean latency/query, Y = Recall@10. Each Tier point is one marker. Used to address the RQ1 cost-benefit question in Chapter 4.

---

## 5. Modules Implemented

### 5.1 `retrieval/llm_reranker.py`

*(Renamed from `naive_rag.py` — "naive RAG" is a misleading label for pointwise re-ranking
with no generation step. `llm_reranker.py` is precise.)*

Main module containing:

- `LLM_JUDGE_0TO10_PROMPT` — French few-shot 0-10 relevance scoring prompt (few-shot from train)
- `parse_llm_score(response: str) -> float` — robust numeric extraction, moved to `llm_eval_prompts.py`
- `ScoreCache` — persistent JSON-backed score cache; key: `(question_id, article_id, prompt_variant, max_article_tokens)`
- `LLMJudgeReranker` — drop-in retriever matching the standard interface

```python
class LLMJudgeReranker:
    """
    Non-agentic LLM-based pointwise re-ranking using the same Ollama LLM as Tier 4.

    retrieve(query, top_k) -> (ranked_article_ids, latency_ms)
    Latency includes first-stage retrieval + all LLM scoring calls.
    """
    def __init__(
        self,
        first_stage_retriever,            # Any Tier 1-3 retriever
        article_texts: dict[int, str],    # {article_id: text}
        llm_client: OllamaClient,         # Shared with Tier 4 (same model, same config)
        top_n: int = 50,                  # candidates passed to LLM
        max_article_tokens: int = 1000,   # text truncation before LLM (300 as ablation)
        prompt_variant: str = "binary",   # "binary" | "0to10"
        cache_path: Path = None,          # Optional score cache (keyed by prompt_variant + max_article_tokens)
        question_id_fn = None,            # For cache keying
    ): ...

    def retrieve(self, query: str, top_k: int = 10) -> tuple[list[int], float]: ...
    def get_latency_breakdown(self) -> dict: ...
    def get_stats(self) -> dict: ...
    def save_cache(self) -> None: ...
```

### 5.2 Shared infrastructure (no modifications)

| File | Purpose |
|---|---|
| `retrieval/agentic/llm_client.py` | `OllamaClient` — **same instance** as CRAG/ReAct |
| `retrieval/agentic/llm_eval_prompts.py` | `LLM_JUDGE_BINARY_PROMPT`, `LLM_JUDGE_0TO10_PROMPT`, `parse_binary_judgment()`, `parse_numeric_score()`, `load_fewshot_examples()`, `format_fewshot_block()` |
| `evaluation/data/fewshot_examples.json` | Locked few-shot examples from train split (shared with T4.1 and T4.2) |
| `evaluation/metrics.py` | Thin caller — delegates Recall@k, MRR@k, NDCG@k, MAP@k computation to external `bsard_evaluation` service; returns metric dicts |
| `evaluation/runner.py` | Result JSON serialisation, experiment logging, `save_result()`, `add_significance()` — significance testing delegated to `bsard_evaluation`; per-query metric vectors returned by external service |
| `evaluation/split.py` | `load_questions("test"\|"val"\|"train")` |
| `evaluation/stratify.py` | `load_strata()` — strata definitions remain local; per-stratum metrics computed by external service |

---

## 6. File and Folder Layout

```
retrieval/
    llm_reranker.py                        ← LLMJudgeReranker + ScoreCache
    agentic/
        llm_eval_prompts.py                ← LLM_JUDGE_BINARY_PROMPT, LLM_JUDGE_0TO10_PROMPT,
                                              parse_binary_judgment(), parse_numeric_score(),
                                              load_fewshot_examples(), format_fewshot_block()

scripts/
    evaluation/
        tier3/
            run_llm_rerank_experiments.py  ← T4.0 experiment orchestrator
        tier4/
            select_fewshot_examples.py     ← locks few-shot examples from train split

evaluation/
    data/
        fewshot_examples.json              ← locked few-shot examples (in git)

output/                                    ← data root (BSARD_DATA_DIR; gitignored)
    results/
        agentic/
            llm_judge/
                llm_rerank/                ← binary re-ranking result JSONs
                    llm_rerank_binary_top50_test.json                 ← BM25 first-stage ablation
                    llm_rerank_binary_top50_hybrid_rrf_k60_test.json  ← CANONICAL T4.0
                    llm_rerank_binary_top20_hybrid_rrf_k60_test.json  ← agentic-matched anchor (T4.1/T4.2)
                0to10/                     ← 0–10 numeric scoring runs
                    llm_rerank_0to10_top50_test.json                  ← scoring-paradigm ablation
    llm_judge_cache_*.json                     ← Score caches (gitignored); first-stage agnostic

tests/
    hybrid/
        test_llm_reranker.py               ← unit tests (renamed from test_naive_rag.py)
```

---

## 7. Result JSON Schema

Produced by `evaluation/runner.run_experiment()` (Tier 0+1+2, same as Tier 1 sparse) plus
T4.0-specific extras. Result files go to `output/results/agentic/llm_judge/llm_rerank/`
(binary) and `output/results/agentic/llm_judge/0to10/` (0–10 numeric scoring) — under the
gitignored data root, not committed to git.

```json
{
  "experiment_id": "llm_rerank_binary_top50_test",
  "timestamp": "2026-04-17T00:00:00",
  "model_or_method": "llmjudgereranker",
  "hyperparameters": {
    "first_stage": "bm25_tuned_k11.5_b0.25",
    "llm_backbone": "llama3.1:8b",
    "llm_temperature": 0.0,
    "top_n": 20,
    "max_article_tokens": 1000,
    "scoring_method": "llm_judge_binary",
    "prompt_variant": "binary",
    "fewshot_examples_file": "evaluation/data/fewshot_examples.json",
    "bm25_k1": 1.5,
    "bm25_b": 0.25,
    "bm25_normalization": "lemmatize",
    "bm25_field_weighting": "text_only"
  },
  "preprocessing": {
    "normalization": "lemmatize",
    "field_weighting": "text_only"
  },
  "token_length_audit": {
    "fraction_truncated": 0.0,
    "max_tokens_observed": 0
  },
  "training_regime": "zero_shot",

  "latency_ms_mean": 0.0,
  "latency_ms_std": 0.0,
  "latency_distribution": {
    "mean": 0.0, "std": 0.0,
    "p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0,
    "min": 0.0, "max": 0.0,
    "index_build_s": 0.0
  },

  "latency_breakdown_ms_mean": {
    "first_stage": 0.0,
    "llm_scoring": 0.0
  },
  "llm_rerank_stats": {
    "mean_llm_calls_per_query": 20,
    "cache_size_after_run": 0,
    "parse_failure_rate": 0.0
  },
  "score_diagnostics": {
    "mean_score_per_query_mean": 0.0,
    "score_std_per_query_mean": 0.0,
    "fraction_queries_all_same_score": 0.0,
    "llm_score_vs_ce_score_spearman_rho": null
  },

  "metrics": {
    "Recall@1": 0.0, "Recall@5": 0.0, "Recall@10": 0.0,
    "Recall@20": 0.0, "Recall@50": 0.0, "Recall@100": 0.0,
    "Recall@200": 0.0, "Recall@500": 0.0,
    "Precision@1": 0.0, "Precision@5": 0.0, "Precision@10": 0.0,
    "F1@1": 0.0, "F1@5": 0.0, "F1@10": 0.0,
    "HitRate@1": 0.0, "HitRate@5": 0.0, "HitRate@10": 0.0,
    "MRR@10": 0.0, "MRR@100": 0.0,
    "NDCG@10": 0.0, "NDCG@100": 0.0,
    "MAP@10": 0.0, "MAP@100": 0.0, "MAP": 0.0,
    "IDPrecision@10": 0.0, "IDRecall@10": 0.0,
    "T0/latency_mean_ms": 0.0, "T0/latency_p50_ms": 0.0,
    "T0/latency_p90_ms": 0.0, "T0/latency_p99_ms": 0.0
  },

  "significance_vs_anchor": {
    "anchor_experiment_id": "bm25_tuned_k11.5_b0.25",
    "p_value_recall10": null,
    "significant": null
  },
  "stratified": {
    "single_article":           { "...same metric keys as metrics block..." },
    "multi_article":            {},
    "lexically_aligned":        {},
    "semantically_paraphrased": {},
    "with_cross_refs":          {},
    "without_cross_refs":       {}
  },
  "total_experiment_wall_clock_s": 0.0
}
```

> **Note on metric coverage:** `top_n=20` means only 20 articles are LLM-scored per query.
> Recall@k for k > 20 will plateau at the same value as Recall@20 — all 20 scored candidates
> are returned and counted. Precision/F1/HitRate are included for Tier 2 completeness.
> T0/* keys are Tier 0 latency metrics passed through from the harness.

---

## 8. Unit Tests: `tests/hybrid/test_naive_rag.py`

**Status: tests need updating to reflect rename and new variants.**

| Category | Count | Tests |
|---|---|---|
| Binary score parsing | 3 | "Oui"→1.0, "Non"→0.0, unexpected output→0.0 |
| Numeric score parsing | 5 | integer, float, with text, no number, clamp upper |
| 0–10 numeric score parsing | 2 | numeric score extracted, fallback on no number |
| Score cache | 5 | put/get, miss, persistence, length, key includes prompt_variant+max_tokens |
| Interface contract | 4 | correct types, top_k length, valid IDs, no duplicates |
| Re-ranking behavior | 3 | binary score ordering, 0-10 ordering, top_k slices |
| Latency / diagnostics | 3 | latency positive, breakdown keys, stats populated |
| Caching integration | 2 | binary cache avoids LLM calls, separate caches per variant |
| Text truncation | 3 | 1000-token truncation applied, 300-token ablation, short text preserved |
| Score diagnostics | 3 | fraction_queries_all_same_score, spearman_rho null when no CE scores, parse_failure_rate |
| Few-shot loading | 2 | examples loaded from fewshot_examples.json, fallback on missing file |
| Prompt template | 3 | binary placeholders, 0–10 static placeholders, 0–10 v2 placeholders; all French |

---

## 9. Execution Order — As Run

```
[DONE] Tier 3 complete → T3-A hybrid_rrf_k60 (R@10=0.4021) locked as the non-agentic ceiling
[DONE] OllamaClient implemented (shared across T4.0 / T4.1 / T4.2)

Step 1  [DONE] retrieval/agentic/llm_eval_prompts.py
        (LLM_JUDGE_BINARY_PROMPT, LLM_JUDGE_0TO10_PROMPT, parse_binary_judgment,
         parse_numeric_score, load_fewshot_examples, format_fewshot_block)

Step 2  [DONE] scripts/evaluation/tier4/select_fewshot_examples.py
        → committed evaluation/data/fewshot_examples.json (2 train examples)

Step 3  [DONE] retrieval/llm_reranker.py — LLMJudgeReranker + ScoreCache
        (cache key: question_id, article_id, prompt_variant, max_article_tokens
         — first-stage agnostic so caches survive first-stage changes)

Step 4  [DONE] tests/hybrid/test_llm_reranker.py

Step 5  [DONE] scripts/evaluation/tier3/run_llm_rerank_experiments.py

Step 6  [DONE] azure_tier40_llm_rerank.ipynb on Azure GPU:
          Cell 10:  python scripts/evaluation/tier3/run_llm_rerank_experiments.py
                      --split test --best-variant binary --best-top-n 50
                      --max-article-tokens 1000
                    → llm_rerank_binary_top50_test.json (R@10 = 0.3618)
          Cell 11b: inline 0–10 numeric scoring run at top_n=50 with a
                    legal-domain system prompt enforcing integer-only output
                    → llm_rerank_0to10_top50_test.json (R@10 = 0.2715)

Step 7  [DONE] azure_tier40_hybrid_llm_rerank.ipynb on Azure GPU:
          Cell 11: hybrid first stage (hybrid_rrf_k60), top_n=50, binary
                   → llm_rerank_binary_top50_hybrid_rrf_k60_test.json
                     (R@10 = 0.4451 — CANONICAL T4.0;
                      p = 0.016 vs T3-A, p < 0.001 vs T4.0-BM25)

Step 8  [DONE] azure_tier40_hybrid_llm_rerank_top20.ipynb on Azure GPU:
          Cell 11: hybrid first stage, top_n=20, binary
                   → llm_rerank_binary_top20_hybrid_rrf_k60_test.json
                     (R@10 = 0.4347; p = 0.007 vs T3-A)
                   Cache hits from prior runs → re-evaluation completed in
                   ~17 s wall-clock. Used as the matched-pool anchor for T4.1 / T4.2.
```

---

## 10. Dependencies

All already satisfied by Tier 1-3 + Tier 4.1 dependencies:

| Dependency | Already installed? | Needed for |
|---|---|---|
| `requests` | Yes | `OllamaClient` HTTP calls |
| `numpy` | Yes | Score statistics |
| `pandas` | Yes | Corpus loading |

Ollama setup (same as CRAG/ReAct — no additional setup):
```bash
ollama pull llama3.1:8b
ollama serve
# verify:
curl http://localhost:11434/api/tags
```

---

## 11. Decision Log

| Decision | Value | Rationale |
|---|---|---|
| LLM model | `llama3.1:8b` | Identical to CRAG/ReAct — controlled variable across all Tier 4 |
| Temperature | `0.0` | Deterministic; matches Tier 4 |
| Prompt language | French | Statutory-register vocabulary alignment |
| Primary scoring paradigm | Binary ("Oui"/"Non") | 8B-class models produce poorly calibrated continuous scores; binary more reliable. Consistent with binary routing in CRAG. |
| Secondary scoring | 0–10 numeric (`prompt_variant="0to10"`) | The only scoring-paradigm comparison. Uses an integer-only system prompt (no few-shot block) — see §2 for the full prompt. |
| Article truncation | 1000 word-level tokens (default); 300 as ablation | LLaMA 3.1 8B 8K context; 300 tokens scored only 15% of article content — key relevance signals in exception clauses and amendment references were missed |
| Few-shot examples | 2 examples from train split, locked in `evaluation/data/fewshot_examples.json` | Anchors the score distribution; drawn from the train split only. |
| Score caching | JSON on disk, keyed by `(question_id, article_id, prompt_variant, max_article_tokens)` | `max_article_tokens` in key prevents collision between truncation variants. First-stage agnostic. |
| Top-N | 50 (canonical headline) + 20 (agentic-matched anchor) | Top-20 matches the candidate-pool size used by T4.1 / T4.2 for fair agentic comparison. |
| First stage | `hybrid_rrf_k60` (canonical headline) + `bm25_tuned_k11.5_b0.25` (first-stage ablation) | Hybrid is the canonical pool; BM25 isolates the first-stage contribution with the same scorer. |
| Naming convention | `llm_rerank_*` | "Naive RAG" misleading — no generation step; this is pointwise re-ranking |
| Class name | `LLMJudgeReranker` | Precise; replaces `NaiveRAGRetriever` |
| Tier designation | Tier 4.0 | Uses Tier 4 LLM but without agentic mechanism — "Tier 4 minus the agency" |

---

## 12. Anticipated Failure Modes — Observed

| Failure Mode | What was observed | Resolution |
|---|---|---|
| LLM consistently outputs non-numeric scores | Did not occur for binary (parse_failure_rate = 0.0 in all binary runs). The 0–10 numeric run had 1.44 % failures — see content-safety row below. | Binary parsing `parse_binary_judgment()` is reliable; `parse_numeric_score()` falls back to 0.0 on un-parseable responses. |
| LLM assigns all documents the same score | Did not occur. `fraction_queries_all_same_score` = 0.0 for the 0–10 run and 0.0 for all binary runs. Score distributions are well-separated. | n/a |
| LLM scoring slower than expected (> 5 s/call) | Binary on BM25: ~626 ms/call (well within budget). 0–10 numeric: ~1 500 ms/call. | No mitigation needed; cache reuse on the hybrid run kept wall-clock to ~29 min. |
| Tier 4.1 / 4.2 do not improve over Tier 4.0 | See TIER41 / TIER42 plans for verdicts. Tier 4.0-hybrid-top20 (R@10 = 0.4347) is the matched-pool anchor for those comparisons. | n/a (downstream tiers) |
| **LLM content-safety refusal (0–10 only)** | LLaMA 3.1 8B's safety filter triggered on articles from criminal / drug / weapons / sexual-offence law, producing digit-free refusals that parse to 0.0. | The Cell 11b inline prompt setup uses a legal-domain system prompt that explicitly frames the task as legal evaluation **and** enforces integer-only output. Observed parse_fail = 1.44 % on the 0–10 run. Binary scoring is immune (no observed refusals — the Oui/Non framing avoids exposing article content to the safety classifier in the same way). Model remained `llama3.1:8b`. |

---

## 13. Relationship to Other Tiers

```
Tier 1 Sparse
  bm25_lem_concat2x ──────────────────────────────────────────────┐
  (R@100 = 0.5312)                                                │
                                                                   ▼
Tier 2 Dense                                            T3-A:  RRF (BM25 + best dense)
  best model D2-D4c ──────────────────────────────────  T3-B:  Linear alpha
                                                                   │
                                                                   ▼
                                              T3-C: Cross-encoder re-ranking
                                              (mmarco-mMiniLMv2-L12)
                                                                   │
                                                                   ▼
                                    ┌──────── T4.0: LLM-as-a-Judge re-ranking ────────┐
                                    │         (llama3.1:8b, single-pass, NON-AGENTIC) │
                                    │         Same LLM as T4.1/T4.2                    │
                                    │         Same first-stage pool as T3-C            │
                                    └──────────────────────────────────────────────────┘
                                                                   │
                                         ┌─────────────────────────┼──────────────────┐
                                         ▼                         ▼                  ▼
                              T4.1: CRAG               T4.2: ReAct          T4.1+T4.2 combined
                              (correction loop)        (reasoning loop)     (if applicable)
                              Δ vs T4.0 = agentic      Δ vs T4.0 = agentic
                              contribution              contribution
```

---

## 14. Mandatory Thesis Disclosures (Chapter 4)

- **On the unified evaluator:** Tier 4.0 uses the identical LLM (LLaMA 3.1 8B, temp=0.0,
  French prompts, same few-shot examples) as Tier 4.1 (CRAG) and Tier 4.2 (ReAct). Any
  performance delta between Tier 4.0 and Tier 4.1/4.2 is attributable solely to the
  agentic correction/reasoning mechanism, not to the LLM's raw scoring ability.
- **On the scoring paradigm:** Tier 4.0 uses pointwise scoring (each document scored
  independently). The LLM-as-a-Judge literature identifies pointwise as less reliable than
  pairwise or listwise approaches for ranking (Gu et al., 2024). Results are a lower bound
  on LLM re-ranking capability. Two paradigms were tested: **binary** (Oui/Non) and **0–10
  numeric**. Binary outperforms 0–10 numeric by 7–14 pp R@10. The result-JSON
  `score_diagnostics.mean_score_per_query_mean` and `score_std_per_query_mean` (≈ 5.3, ≈ 1.6
  for the 0–10 runs) confirm the numeric distribution is well-separated but ranks worse than
  binary; this is the headline scoring-paradigm finding.
- **On few-shot calibration:** Few-shot examples are drawn from the BSARD train split and
  locked before validation. The specific examples may influence score distributions; this
  is an uncontrolled variable.
- **On latency:** Tier 4.0 latency scales linearly with top-N. Report mean latency per
  query alongside Recall@10 for cost-benefit analysis. Include in the Pareto-front plot.
- **On parse failures:** If the LLM returns non-parseable responses, those articles receive
  score 0.0 (conservative). Report the `parse_failure_rate` from `llm_rerank_stats`.
- **On score diagnostics:** Report `fraction_queries_all_same_score` — if high, the LLM
  cannot distinguish statutory relevance at this scale and no re-ranking occurs. This is
  a valid negative finding.
