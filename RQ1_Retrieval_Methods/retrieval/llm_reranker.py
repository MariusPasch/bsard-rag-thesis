"""
Tier 4.0 — LLM-as-a-Judge pointwise re-ranking.

Classes
-------
ScoreCache
    Persistent JSON-backed score cache.
    Key: (question_id, article_id, prompt_variant, max_article_tokens)

LLMJudgeReranker
    Non-agentic LLM-based pointwise re-ranking using the same OllamaClient
    as CRAG (Tier 4.1) and ReAct (Tier 4.2).

    Interface contract (identical to all Tier 1-3 retrievers):
        retrieve(query, top_k) -> (ranked_article_ids: list[int], latency_ms: float)
    Latency includes first-stage retrieval + all LLM scoring calls.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Callable

import numpy as np

from retrieval.agentic.llm_client import OllamaClient
from retrieval.agentic.llm_eval_prompts import (
    LLM_JUDGE_BINARY_PROMPT,
    LLM_JUDGE_0TO10_PROMPT,
    load_fewshot_examples,
    format_fewshot_block,
    parse_binary_judgment,
    parse_numeric_score,
)

logger = logging.getLogger(__name__)

_DEFAULT_FEWSHOT_PATH = Path("evaluation/data/fewshot_examples.json")
_DEFAULT_CACHE_PATH   = Path("output/llm_judge_score_cache.json")

# ---------------------------------------------------------------------------
# ScoreCache
# ---------------------------------------------------------------------------

class ScoreCache:
    """
    Persistent JSON-backed score cache.

    Key: (question_id: int, article_id: int, prompt_variant: str, max_article_tokens: int)
    Value: float score

    Scores from different prompt_variants and truncation lengths are stored in
    separate buckets so they never collide.
    """

    def __init__(self, path: Path | None = None):
        self._path: Path | None = path
        self._cache: dict[str, float] = {}
        if path is not None and path.exists():
            try:
                self._cache = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("ScoreCache: failed to load %s — starting empty. %s", path, exc)

    # ── Public interface ─────────────────────────────────────────────────────

    def get(
        self,
        question_id: int,
        article_id: int,
        prompt_variant: str,
        max_article_tokens: int,
    ) -> float | None:
        return self._cache.get(self._key(question_id, article_id, prompt_variant, max_article_tokens))

    def put(
        self,
        question_id: int,
        article_id: int,
        prompt_variant: str,
        max_article_tokens: int,
        score: float,
    ) -> None:
        self._cache[self._key(question_id, article_id, prompt_variant, max_article_tokens)] = score

    def __len__(self) -> int:
        return len(self._cache)

    def save(self) -> None:
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(self._cache, ensure_ascii=False),
            encoding="utf-8",
        )

    # ── Internal ─────────────────────────────────────────────────────────────

    @staticmethod
    def _key(
        question_id: int,
        article_id: int,
        prompt_variant: str,
        max_article_tokens: int,
    ) -> str:
        return f"{question_id}|{article_id}|{prompt_variant}|{max_article_tokens}"


# ---------------------------------------------------------------------------
# LLMJudgeReranker
# ---------------------------------------------------------------------------

class LLMJudgeReranker:
    """
    Non-agentic LLM-based pointwise re-ranking.

    Parameters
    ----------
    first_stage_retriever : any retriever with .retrieve(query, top_k) -> (list[int], float)
    article_texts         : {article_id: text}
    llm_client            : OllamaClient (shared with CRAG/ReAct)
    top_n                 : number of first-stage candidates passed to the LLM
    max_article_tokens    : word-level truncation before LLM call (default 1000)
    prompt_variant        : "binary" | "0to10"
    cache_path            : optional Path for persistent score cache
    question_id_fn        : callable(query_str) -> int for cache keying;
                            if None, a hash-based fallback is used
    fewshot_path          : path to fewshot_examples.json
    """

    def __init__(
        self,
        first_stage_retriever,
        article_texts: dict[int, str],
        llm_client: OllamaClient,
        top_n: int = 50,
        max_article_tokens: int = 1000,
        prompt_variant: str = "binary",
        cache_path: Path | None = None,
        question_id_fn: Callable[[str], int] | None = None,
        fewshot_path: Path = _DEFAULT_FEWSHOT_PATH,
    ):
        if prompt_variant not in ("binary", "0to10"):
            raise ValueError(
                f"prompt_variant must be 'binary' or '0to10'; got {prompt_variant!r}"
            )

        self._first_stage      = first_stage_retriever
        self._article_texts    = article_texts
        self._llm              = llm_client
        self.top_n             = top_n
        self.max_article_tokens = max_article_tokens
        self.prompt_variant    = prompt_variant
        self._question_id_fn   = question_id_fn

        # Load few-shot examples — fail fast if missing
        self._fewshot_examples = load_fewshot_examples(fewshot_path)
        self._fewshot_block    = format_fewshot_block(self._fewshot_examples, prompt_variant)

        # Score cache
        self._cache = ScoreCache(cache_path)

        # Diagnostics — reset per retrieve() call; aggregated across all calls
        self._parse_failures: int = 0
        self._total_llm_calls: int = 0
        self._latencies_first_stage: list[float] = []
        self._latencies_llm: list[float] = []

        # Per-query score lists (for score_diagnostics in get_stats())
        self._per_query_scores: list[list[float]] = []

    # ── Public interface ─────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 10) -> tuple[list[int], float]:
        """
        Retrieve top-k article IDs via LLM-based pointwise re-ranking.

        Returns
        -------
        ranked_article_ids : list[int]  (length = top_k, or fewer if fewer candidates)
        latency_ms         : wall-clock time for the entire call (first-stage + LLM scoring)
        """
        t0 = time.perf_counter()

        # 1. First-stage retrieval
        t_fs0 = time.perf_counter()
        candidates, _ = self._first_stage.retrieve(query, top_k=self.top_n)
        fs_ms = (time.perf_counter() - t_fs0) * 1000.0
        self._latencies_first_stage.append(fs_ms)

        # 2. LLM scoring
        question_id = self._get_question_id(query)
        t_llm0 = time.perf_counter()
        scores = self._score_candidates(query, question_id, candidates)
        llm_ms = (time.perf_counter() - t_llm0) * 1000.0
        self._latencies_llm.append(llm_ms)

        # 3. Re-rank
        ranked = sorted(candidates, key=lambda aid: scores.get(aid, 0.0), reverse=True)
        self._per_query_scores.append(list(scores.values()))

        total_ms = (time.perf_counter() - t0) * 1000.0
        return ranked[:top_k], total_ms

    def get_latency_breakdown(self) -> dict:
        """Mean latencies (ms) broken down by stage."""
        return {
            "first_stage_ms_mean": float(np.mean(self._latencies_first_stage)) if self._latencies_first_stage else 0.0,
            "llm_scoring_ms_mean": float(np.mean(self._latencies_llm)) if self._latencies_llm else 0.0,
        }

    def get_stats(self) -> dict:
        """Diagnostics and score statistics aggregated over all retrieve() calls."""
        n_queries = len(self._per_query_scores)

        # Score diagnostics
        if n_queries > 0:
            per_query_means = [np.mean(s) for s in self._per_query_scores if s]
            per_query_stds  = [np.std(s)  for s in self._per_query_scores if s]
            same_score_count = sum(
                1 for s in self._per_query_scores if len(set(s)) <= 1
            )
            mean_score_per_query_mean = float(np.mean(per_query_means)) if per_query_means else 0.0
            score_std_per_query_mean  = float(np.mean(per_query_stds))  if per_query_stds  else 0.0
            fraction_all_same         = same_score_count / n_queries
        else:
            mean_score_per_query_mean = 0.0
            score_std_per_query_mean  = 0.0
            fraction_all_same         = 0.0

        parse_failure_rate = (
            self._parse_failures / self._total_llm_calls
            if self._total_llm_calls > 0
            else 0.0
        )

        return {
            "n_queries":                    n_queries,
            "total_llm_calls":              self._total_llm_calls,
            "cache_size":                   len(self._cache),
            "parse_failure_rate":           round(parse_failure_rate, 4),
            "score_diagnostics": {
                "mean_score_per_query_mean":      mean_score_per_query_mean,
                "score_std_per_query_mean":       score_std_per_query_mean,
                "fraction_queries_all_same_score": fraction_all_same,
            },
        }

    def save_cache(self) -> None:
        self._cache.save()

    # ── Internal ─────────────────────────────────────────────────────────────

    def _score_candidates(
        self,
        query: str,
        question_id: int,
        candidates: list[int],
    ) -> dict[int, float]:
        scores: dict[int, float] = {}
        for article_id in candidates:
            # Check cache first
            cached = self._cache.get(question_id, article_id, self.prompt_variant, self.max_article_tokens)
            if cached is not None:
                scores[article_id] = cached
                continue

            # Build prompt
            text = self._truncate(self._article_texts.get(article_id, ""))
            prompt = self._build_prompt(query, text)

            # Call LLM — use generous timeout for long article prompts on CPU
            max_tokens = 8 if self.prompt_variant in ("binary", "0to10") else 64
            response, _ = self._llm.generate(
                prompt, temperature=0.0, max_tokens=max_tokens, timeout=600
            )
            self._total_llm_calls += 1

            # Parse
            score = self._parse_response(response)
            scores[article_id] = score
            self._cache.put(question_id, article_id, self.prompt_variant, self.max_article_tokens, score)

        return scores

    def _build_prompt(self, query: str, article_text: str) -> str:
        if self.prompt_variant == "binary":
            return LLM_JUDGE_BINARY_PROMPT.format(
                fewshot_block=self._fewshot_block,
                question=query,
                article_text_truncated=article_text,
            )
        # 0to10
        return LLM_JUDGE_0TO10_PROMPT.format(
            question=query,
            article_text_truncated=article_text,
        )

    def _parse_response(self, response: str) -> float:
        if self.prompt_variant == "binary":
            _, score = parse_binary_judgment(response)
            # Track parse failures (anything that isn't Oui/Non)
            first_word = response.strip().split()[0].lower().rstrip(".,;:!?»\"'") if response.strip() else ""
            if first_word not in ("oui", "non"):
                self._parse_failures += 1
            return score
        else:
            score = parse_numeric_score(response)
            if score == 0.0 and not any(c.isdigit() for c in response):
                self._parse_failures += 1
            return score

    def _truncate(self, text: str) -> str:
        words = text.split()
        if len(words) <= self.max_article_tokens:
            return text
        return " ".join(words[:self.max_article_tokens])

    def _get_question_id(self, query: str) -> int:
        if self._question_id_fn is not None:
            return self._question_id_fn(query)
        # Fallback: stable hash (consistent within a Python session)
        return hash(query) & 0x7FFFFFFF
