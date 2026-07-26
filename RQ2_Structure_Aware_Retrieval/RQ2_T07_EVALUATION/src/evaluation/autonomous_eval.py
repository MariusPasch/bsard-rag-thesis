"""Autonomous (reference-free) evaluation using RAGAS, RAGChecker, and G-Eval.

Used when no ground-truth annotations exist, or as a complement to binary metrics.
Heavy dependencies (ragas, deepeval) are imported lazily so the rest of the package
remains importable without them installed.
"""

from __future__ import annotations

import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


def run_autonomous_evaluation(
    all_results: dict[str, list],  # dict[str, list[RetrievalResult]]
    llm_client,                    # shared.llm.LLMClient
    config: dict,
) -> dict[str, dict[str, float]]:
    """Score all retrieval systems using reference-free metrics.

    For each query, generates an answer from the retrieved context, then scores
    the (query, context, answer) triple with RAGAS and/or G-Eval.
    """
    scores: dict[str, list[dict]] = defaultdict(list)

    for method_name, method_results in all_results.items():
        logger.info("Running autonomous eval for method=%s  n=%d", method_name, len(method_results))
        for result in method_results:
            context = "\n\n".join(
                item["text"] for item in result.ranked_items[:10]
            )
            answer = llm_client.generate(
                prompt=(
                    f"Based on the following legal articles:\n{context}\n\n"
                    f"Answer this question: {result.query_text}"
                )
            )

            score: dict[str, float] = {}

            if config.get("evaluation", {}).get("use_ragas", True):
                try:
                    score.update(_compute_ragas_scores(result.query_text, answer.text, context))
                except Exception as exc:
                    logger.warning("RAGAS failed for query %s: %s", result.query_id, exc)

            if config.get("evaluation", {}).get("use_geval", False):
                try:
                    score.update(_compute_geval_scores(result.query_text, answer.text, context, llm_client))
                except Exception as exc:
                    logger.warning("G-Eval failed for query %s: %s", result.query_id, exc)

            scores[method_name].append(score)

    return {method: _aggregate_scores(method_scores) for method, method_scores in scores.items()}


# ---------------------------------------------------------------------------
# RAGAS (reference-free dimensions)
# ---------------------------------------------------------------------------

def _compute_ragas_scores(query: str, answer: str, context: str) -> dict[str, float]:
    """Compute RAGAS faithfulness and answer relevance (no reference needed)."""
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import faithfulness, answer_relevancy
    from datasets import Dataset

    data = Dataset.from_dict({
        "question": [query],
        "answer": [answer],
        "contexts": [[context]],
    })
    result = ragas_evaluate(data, metrics=[faithfulness, answer_relevancy])
    return {
        "ragas_faithfulness": float(result["faithfulness"]),
        "ragas_answer_relevancy": float(result["answer_relevancy"]),
    }


# ---------------------------------------------------------------------------
# G-Eval (LLM-as-judge with custom rubrics, 1–5 scale)
# ---------------------------------------------------------------------------

_GEVAL_SYSTEM_PROMPT = (
    "You are an expert evaluator of legal information retrieval systems. "
    "You will be given a question, retrieved context, and an answer. "
    "Score strictly on the criterion described. Respond with a single integer 1-5."
)

_GEVAL_RUBRICS = {
    "completeness": (
        "Does the answer address all dimensions of the question? "
        "1 = major aspects missing, 5 = fully complete."
    ),
    "factual_accuracy": (
        "Does the answer avoid hallucinated article numbers, dates, or provision references? "
        "1 = multiple hallucinations, 5 = fully accurate."
    ),
    "coherence": (
        "Is the answer internally consistent with no conflation of distinct provisions? "
        "1 = contradictory or confused, 5 = perfectly coherent."
    ),
}


def _compute_geval_scores(
    query: str,
    answer: str,
    context: str,
    llm_client,
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for criterion, rubric in _GEVAL_RUBRICS.items():
        prompt = (
            f"Question: {query}\n\n"
            f"Retrieved context (first 2000 chars):\n{context[:2000]}\n\n"
            f"Answer: {answer}\n\n"
            f"Criterion: {rubric}"
        )
        response = llm_client.generate(prompt=prompt, system_prompt=_GEVAL_SYSTEM_PROMPT)
        try:
            score = float(response.text.strip().split()[0])
            scores[f"geval_{criterion}"] = max(1.0, min(5.0, score))
        except (ValueError, IndexError):
            logger.warning("G-Eval parse failed for criterion=%s  response=%r", criterion, response.text)
    return scores


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _aggregate_scores(per_query: list[dict[str, float]]) -> dict[str, float]:
    if not per_query:
        return {}
    import numpy as np
    keys = {k for q in per_query for k in q}
    return {
        k: float(np.mean([q[k] for q in per_query if k in q]))
        for k in keys
    }
