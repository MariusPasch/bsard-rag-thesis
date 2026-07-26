"""
Shared LLM-as-a-Judge prompt templates and parsing utilities.

Shared across:
  Tier 4.0  — LLMJudgeReranker (retrieval/llm_reranker.py)
  Tier 4.1  — CRAG evaluate node (retrieval/agentic/crag.py)
  Tier 4.2  — ReAct D1 re-ranking

Any prompt change here propagates to all three tiers automatically.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

# Binary relevance prompt (primary).
# Placeholders: {fewshot_block}, {question}, {article_text_truncated}
LLM_JUDGE_BINARY_PROMPT = """\
{fewshot_block}\
Question : {question}

Passage : {article_text_truncated}

Le passage est-il pertinent pour répondre à la question ?
Répondez uniquement par « Oui » ou « Non ».

Pertinent :"""


# 0–10 numeric scoring prompt (comparison variant to binary).
# Placeholders: {question}, {article_text_truncated}
# Used together with a legal-domain system prompt that enforces integer-only
# output ("Réponds uniquement avec un entier de 0 à 10. Aucun autre texte.")
# — see azure_notebooks/azure_tier40_llm_rerank.ipynb Cell 11b. The system
# prompt makes parse_numeric_score's "extract the first number" behaviour the
# end of the response, so any text after the score is suppressed.
LLM_JUDGE_0TO10_PROMPT = """\
Question : {question}

Passage : {article_text_truncated}

Donnez un score de 0 à 10, puis expliquez brièvement pourquoi ce passage est ou n'est pas pertinent.

Score :
Explication :"""


# Collective relevance prompt — strict variant (used at iteration 0).
# Evaluates a numbered list of passages in a single LLM call.
# Placeholders: {question}, {articles_block}
# Expected response: "Oui" if at least one passage is relevant, else "Non".
LLM_JUDGE_COLLECTIVE_STRICT_PROMPT = """\
Question : {question}

Passages :
{articles_block}

Au moins un des passages ci-dessus est-il pertinent pour répondre à la question ?
Répondez uniquement par « Oui » ou « Non ».

Pertinent :"""


# Collective relevance prompt — lenient variant (used at iterations 1+).
# Same structure but with a more permissive instruction.
# Placeholders: {question}, {articles_block}
LLM_JUDGE_COLLECTIVE_PROMPT = """\
Question : {question}

Passages :
{articles_block}

Parmi les passages ci-dessus, y en a-t-il au moins un qui contient des informations utiles pour répondre à la question ?
Répondez uniquement par « Oui » ou « Non ».

Pertinent :"""

# ---------------------------------------------------------------------------
# Article block formatter (for collective prompts)
# ---------------------------------------------------------------------------

def format_articles_block(
    article_texts: list[str],
    max_tokens_per_article: int = 300,
) -> str:
    """
    Format a list of article texts into a numbered block for injection into
    the {articles_block} placeholder of collective prompts.

    Each article is truncated to max_tokens_per_article whitespace-delimited
    tokens and prefixed with its 1-based index.
    """
    lines = []
    for i, text in enumerate(article_texts, start=1):
        words = text.split()
        if len(words) > max_tokens_per_article:
            text = " ".join(words[:max_tokens_per_article]) + " …"
        lines.append(f"[{i}] {text}")
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Few-shot example loading
# ---------------------------------------------------------------------------

_DEFAULT_FEWSHOT_PATH = Path("evaluation/data/fewshot_examples.json")


def load_fewshot_examples(
    path: str | Path = _DEFAULT_FEWSHOT_PATH,
) -> dict:
    """
    Load the locked few-shot examples JSON.

    Raises FileNotFoundError if the file does not exist — fail fast so that
    experiments never silently run without calibrated examples.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Few-shot examples file not found: {p}\n"
            "Run scripts/agentic/select_fewshot_examples.py first, then commit the result."
        )
    return json.loads(p.read_text(encoding="utf-8"))


def format_fewshot_block(examples: dict, prompt_variant: str) -> str:
    """
    Format the few-shot examples as an inline text block for injection into
    the {fewshot_block} placeholder of the binary prompt.

    Parameters
    ----------
    examples       : dict returned by load_fewshot_examples()
    prompt_variant : "binary" | "0to10"

    Returns
    -------
    Formatted string (ends with a blank line separator), or "" if no examples.
    The 0–10 prompt does not use a {fewshot_block} placeholder, so this returns
    "" for that variant.
    """
    items = examples.get("examples", [])
    if not items:
        return ""

    if prompt_variant == "binary":
        return _format_binary_fewshot(items)
    # 0to10 and any unknown variant: no few-shot block
    return ""


_FEWSHOT_ARTICLE_MAX_TOKENS = 200   # keep few-shot articles short to limit prompt length


def _truncate_fewshot(text: str, max_tokens: int = _FEWSHOT_ARTICLE_MAX_TOKENS) -> str:
    words = text.split()
    if len(words) <= max_tokens:
        return text
    return " ".join(words[:max_tokens]) + " …"


def _format_binary_fewshot(items: list[dict]) -> str:
    lines = []
    for ex in items:
        lines.append(f"Question : {ex['question']}")
        lines.append("")
        lines.append(f"Passage : {_truncate_fewshot(ex['article_text_truncated'])}")
        lines.append("")
        lines.append("Le passage est-il pertinent pour répondre à la question ?")
        lines.append("Répondez uniquement par « Oui » ou « Non ».")
        lines.append("")
        lines.append(f"Pertinent : {ex['binary_label']}")
        lines.append("")
        lines.append("---")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Score parsers
# ---------------------------------------------------------------------------

def parse_binary_judgment(response: str) -> tuple[bool, float]:
    """
    Parse a binary Oui/Non judgment from the LLM response.

    Rules
    -----
    - Strip whitespace; inspect the first word (case-insensitive).
    - "Oui" → (True, 1.0)
    - "Non" → (False, 0.0)
    - Anything else → (False, 0.0) + warning logged as parse failure

    Returns
    -------
    (is_relevant: bool, score: float)
    """
    first_word = response.strip().split()[0].lower() if response.strip() else ""

    # Normalize: strip punctuation from the end of the first word
    first_word = first_word.rstrip(".,;:!?»\"'")

    if first_word == "oui":
        return True, 1.0
    elif first_word == "non":
        return False, 0.0
    else:
        logger.warning(
            "parse_binary_judgment: unexpected response %r (first_word=%r) — "
            "scoring as Non (0.0)",
            response[:80],
            first_word,
        )
        return False, 0.0


def parse_numeric_score(response: str) -> float:
    """
    Extract the first number from *response* and clamp to [0.0, 10.0].

    Returns 0.0 on failure (no number found).
    """
    match = re.search(r"-?\d+(?:[.,]\d+)?", response)
    if match is None:
        logger.warning(
            "parse_numeric_score: no number found in response %r — returning 0.0",
            response[:80],
        )
        return 0.0

    # Replace comma decimal separator (French locale) with period
    raw = match.group(0).replace(",", ".")
    try:
        value = float(raw)
    except ValueError:
        return 0.0

    return float(min(max(value, 0.0), 10.0))
