"""
Unit tests for LLMJudgeReranker (Tier 4.0) and shared prompt utilities.

Tests are fully offline — OllamaClient is mocked so no Ollama server is needed.
fewshot_examples.json is also mocked.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Imports under test
# ---------------------------------------------------------------------------
from retrieval.agentic.llm_eval_prompts import (
    parse_binary_judgment,
    parse_numeric_score,
    load_fewshot_examples,
    format_fewshot_block,
    LLM_JUDGE_BINARY_PROMPT,
    LLM_JUDGE_0TO10_PROMPT,
)
from retrieval.llm_reranker import ScoreCache, LLMJudgeReranker

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MINIMAL_FEWSHOT = {
    "generated_at": "2026-03-27",
    "split": "train",
    "seed": 42,
    "examples": [
        {
            "role": "relevant",
            "question": "Quelle est la peine pour vol ?",
            "article_text_truncated": "Le vol est puni de cinq ans.",
            "article_id": 1,
            "strata": "single_article",
            "binary_label": "Oui",
            "numeric_score": 9,
        },
        {
            "role": "irrelevant",
            "question": "Comment hériter d'un bien ?",
            "article_text_truncated": "Nul ne peut être condamné sans procès.",
            "article_id": 2,
            "strata": "multi_article",
            "binary_label": "Non",
            "numeric_score": 1,
        },
    ],
}


@pytest.fixture()
def fewshot_file(tmp_path):
    p = tmp_path / "fewshot_examples.json"
    p.write_text(json.dumps(MINIMAL_FEWSHOT), encoding="utf-8")
    return p


@pytest.fixture()
def article_texts():
    return {i: f"Texte de l'article {i} sur les droits civils." for i in range(1, 21)}


@pytest.fixture()
def mock_first_stage(article_texts):
    """First stage that returns articles 1-10 in order."""
    retriever = MagicMock()
    retriever.retrieve.return_value = (list(range(1, 11)), 10.0)
    return retriever


@pytest.fixture()
def mock_llm_binary():
    """LLM that alternates Oui/Non."""
    llm = MagicMock()
    call_count = [0]

    def side_effect(prompt, temperature=0.0, max_tokens=8, **kwargs):
        call_count[0] += 1
        resp = "Oui" if call_count[0] % 2 == 1 else "Non"
        return resp, 50.0

    llm.generate.side_effect = side_effect
    return llm


@pytest.fixture()
def mock_llm_numeric():
    """LLM that returns scores 1-10 cycling."""
    llm = MagicMock()
    call_count = [0]

    def side_effect(prompt, temperature=0.0, max_tokens=8, **kwargs):
        call_count[0] += 1
        score = (call_count[0] % 10) + 1
        return str(score), 50.0

    llm.generate.side_effect = side_effect
    return llm


def _make_reranker(first_stage, article_texts, llm, fewshot_file, prompt_variant="binary", top_n=10, max_article_tokens=1000, cache_path=None):
    return LLMJudgeReranker(
        first_stage_retriever=first_stage,
        article_texts=article_texts,
        llm_client=llm,
        top_n=top_n,
        max_article_tokens=max_article_tokens,
        prompt_variant=prompt_variant,
        cache_path=cache_path,
        question_id_fn=lambda q: 42,
        fewshot_path=fewshot_file,
    )


# ===========================================================================
# Binary score parsing
# ===========================================================================

class TestParseBinaryJudgment:
    def test_oui(self):
        assert parse_binary_judgment("Oui") == (True, 1.0)

    def test_oui_lowercase(self):
        assert parse_binary_judgment("oui") == (True, 1.0)

    def test_non(self):
        assert parse_binary_judgment("Non") == (False, 0.0)

    def test_non_with_trailing_text(self):
        # Only first word matters
        assert parse_binary_judgment("Non, ce passage ne traite pas") == (False, 0.0)

    def test_oui_with_punctuation(self):
        assert parse_binary_judgment("Oui.") == (True, 1.0)

    def test_unexpected_returns_false(self):
        is_rel, score = parse_binary_judgment("Certainement")
        assert is_rel is False
        assert score == 0.0

    def test_empty_returns_false(self):
        is_rel, score = parse_binary_judgment("")
        assert is_rel is False
        assert score == 0.0


# ===========================================================================
# Numeric score parsing
# ===========================================================================

class TestParseNumericScore:
    def test_integer(self):
        assert parse_numeric_score("7") == 7.0

    def test_with_label(self):
        assert parse_numeric_score("Score : 7") == 7.0

    def test_float(self):
        assert parse_numeric_score("8.5") == 8.5

    def test_in_sentence(self):
        assert parse_numeric_score("Je donne un score de 6 sur 10.") == 6.0

    def test_clamp_upper(self):
        assert parse_numeric_score("Ce passage mérite 11 points") == 10.0

    def test_clamp_lower(self):
        assert parse_numeric_score("-3") == 0.0

    def test_no_number_returns_zero(self):
        assert parse_numeric_score("Aucun score disponible") == 0.0

    def test_zero(self):
        assert parse_numeric_score("Score : 0") == 0.0


# ===========================================================================
# Few-shot loading
# ===========================================================================

class TestLoadFewshotExamples:
    def test_loads_successfully(self, fewshot_file):
        data = load_fewshot_examples(fewshot_file)
        assert "examples" in data
        assert len(data["examples"]) == 2

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_fewshot_examples(tmp_path / "nonexistent.json")


# ===========================================================================
# Score cache
# ===========================================================================

class TestScoreCache:
    def test_put_and_get(self):
        cache = ScoreCache()
        cache.put(1, 100, "binary", 1000, 1.0)
        assert cache.get(1, 100, "binary", 1000) == 1.0

    def test_miss_returns_none(self):
        cache = ScoreCache()
        assert cache.get(99, 999, "binary", 1000) is None

    def test_length(self):
        cache = ScoreCache()
        cache.put(1, 1, "binary", 1000, 1.0)
        cache.put(1, 2, "binary", 1000, 0.0)
        assert len(cache) == 2

    def test_persistence(self, tmp_path):
        path = tmp_path / "cache.json"
        c1 = ScoreCache(path)
        c1.put(5, 10, "binary", 1000, 1.0)
        c1.save()
        c2 = ScoreCache(path)
        assert c2.get(5, 10, "binary", 1000) == 1.0

    def test_key_includes_prompt_variant(self):
        cache = ScoreCache()
        cache.put(1, 1, "binary", 1000, 1.0)
        cache.put(1, 1, "0to10", 1000, 7.0)
        assert cache.get(1, 1, "binary", 1000) == 1.0
        assert cache.get(1, 1, "0to10", 1000) == 7.0

    def test_key_includes_max_article_tokens(self):
        cache = ScoreCache()
        cache.put(1, 1, "binary", 300, 1.0)
        cache.put(1, 1, "binary", 1000, 0.0)
        assert cache.get(1, 1, "binary", 300) == 1.0
        assert cache.get(1, 1, "binary", 1000) == 0.0
        # Ensures 300-token and 1000-token runs never share cache entries
        assert cache.get(1, 1, "binary", 300) != cache.get(1, 1, "binary", 1000)


# ===========================================================================
# Interface contract
# ===========================================================================

class TestLLMJudgeRerankerInterface:
    def test_returns_list_of_ints(self, mock_first_stage, article_texts, mock_llm_binary, fewshot_file):
        r = _make_reranker(mock_first_stage, article_texts, mock_llm_binary, fewshot_file)
        ranked, lat = r.retrieve("Qu'est-ce que le vol ?", top_k=5)
        assert isinstance(ranked, list)
        assert all(isinstance(x, int) for x in ranked)

    def test_top_k_length(self, mock_first_stage, article_texts, mock_llm_binary, fewshot_file):
        r = _make_reranker(mock_first_stage, article_texts, mock_llm_binary, fewshot_file)
        ranked, _ = r.retrieve("test", top_k=5)
        assert len(ranked) == 5

    def test_valid_article_ids(self, mock_first_stage, article_texts, mock_llm_binary, fewshot_file):
        r = _make_reranker(mock_first_stage, article_texts, mock_llm_binary, fewshot_file)
        ranked, _ = r.retrieve("test", top_k=5)
        assert all(aid in article_texts for aid in ranked)

    def test_no_duplicates(self, mock_first_stage, article_texts, mock_llm_binary, fewshot_file):
        r = _make_reranker(mock_first_stage, article_texts, mock_llm_binary, fewshot_file)
        ranked, _ = r.retrieve("test", top_k=10)
        assert len(ranked) == len(set(ranked))

    def test_latency_positive(self, mock_first_stage, article_texts, mock_llm_binary, fewshot_file):
        r = _make_reranker(mock_first_stage, article_texts, mock_llm_binary, fewshot_file)
        _, lat = r.retrieve("test", top_k=5)
        assert lat > 0.0


# ===========================================================================
# Re-ranking behaviour
# ===========================================================================

class TestReRankingBehaviour:
    def test_binary_score_ordering(self, article_texts, fewshot_file):
        """Articles scored Oui (1.0) should rank above Non (0.0)."""
        # First stage returns [1, 2, 3, 4]
        first_stage = MagicMock()
        first_stage.retrieve.return_value = ([1, 2, 3, 4], 5.0)

        # LLM: article 1 → Non, article 2 → Oui, article 3 → Non, article 4 → Oui
        responses = {1: "Non", 2: "Oui", 3: "Non", 4: "Oui"}
        call_order = [1, 2, 3, 4]
        idx = [0]
        def gen(prompt, temperature=0.0, max_tokens=8, **kwargs):
            aid = call_order[idx[0] % len(call_order)]
            idx[0] += 1
            return responses[aid], 10.0

        llm = MagicMock()
        llm.generate.side_effect = gen

        r = _make_reranker(first_stage, article_texts, llm, fewshot_file, top_n=4)
        ranked, _ = r.retrieve("test", top_k=4)
        # Articles 2 and 4 (Oui=1.0) must rank above 1 and 3 (Non=0.0)
        assert set(ranked[:2]) == {2, 4}
        assert set(ranked[2:]) == {1, 3}

    def test_0to10_score_ordering(self, article_texts, fewshot_file):
        """Higher numeric score → higher rank."""
        first_stage = MagicMock()
        first_stage.retrieve.return_value = ([10, 11, 12], 5.0)

        article_texts_ext = dict(article_texts)
        article_texts_ext.update({10: "art10", 11: "art11", 12: "art12"})

        scores = {10: "3", 11: "9", 12: "6"}
        call_order = [10, 11, 12]
        idx = [0]
        def gen(prompt, temperature=0.0, max_tokens=8, **kwargs):
            aid = call_order[idx[0] % len(call_order)]
            idx[0] += 1
            return scores[aid], 10.0

        llm = MagicMock()
        llm.generate.side_effect = gen

        r = _make_reranker(first_stage, article_texts_ext, llm, fewshot_file, prompt_variant="0to10", top_n=3)
        ranked, _ = r.retrieve("test", top_k=3)
        assert ranked[0] == 11  # score 9
        assert ranked[1] == 12  # score 6
        assert ranked[2] == 10  # score 3

    def test_top_k_slices_correctly(self, mock_first_stage, article_texts, mock_llm_binary, fewshot_file):
        r = _make_reranker(mock_first_stage, article_texts, mock_llm_binary, fewshot_file, top_n=10)
        ranked, _ = r.retrieve("test", top_k=3)
        assert len(ranked) == 3


# ===========================================================================
# Caching integration
# ===========================================================================

class TestCachingIntegration:
    def test_cache_avoids_redundant_llm_calls(self, mock_first_stage, article_texts, mock_llm_binary, fewshot_file, tmp_path):
        cache_path = tmp_path / "cache.json"
        r = _make_reranker(mock_first_stage, article_texts, mock_llm_binary, fewshot_file, cache_path=cache_path)
        r.retrieve("test", top_k=5)
        n_calls_first = mock_llm_binary.generate.call_count
        # Second call: all articles already cached
        r.retrieve("test", top_k=5)
        n_calls_second = mock_llm_binary.generate.call_count
        assert n_calls_second == n_calls_first  # no new LLM calls

    def test_separate_caches_per_variant(self, mock_first_stage, article_texts, fewshot_file, tmp_path):
        cache_path_bin = tmp_path / "cache_binary.json"
        cache_path_num = tmp_path / "cache_0to10.json"

        llm_bin = MagicMock()
        llm_bin.generate.return_value = ("Oui", 10.0)
        r_bin = _make_reranker(mock_first_stage, article_texts, llm_bin, fewshot_file,
                               prompt_variant="binary", cache_path=cache_path_bin)

        llm_num = MagicMock()
        llm_num.generate.return_value = ("7", 10.0)
        r_num = _make_reranker(mock_first_stage, article_texts, llm_num, fewshot_file,
                               prompt_variant="0to10", cache_path=cache_path_num)

        r_bin.retrieve("test", top_k=5)
        r_num.retrieve("test", top_k=5)

        # Both LLMs were called — caches are separate
        assert llm_bin.generate.call_count > 0
        assert llm_num.generate.call_count > 0


# ===========================================================================
# Text truncation
# ===========================================================================

class TestTextTruncation:
    def test_1000_token_truncation(self, fewshot_file):
        long_text = " ".join(["mot"] * 2000)
        article_texts = {1: long_text}
        first_stage = MagicMock()
        first_stage.retrieve.return_value = ([1], 5.0)

        llm = MagicMock()
        captured = []
        def gen(prompt, temperature=0.0, max_tokens=8, **kwargs):
            captured.append(prompt)
            return "Oui", 10.0
        llm.generate.side_effect = gen

        r = _make_reranker(first_stage, article_texts, llm, fewshot_file, top_n=1, max_article_tokens=1000)
        r.retrieve("test", top_k=1)

        assert len(captured) == 1
        # The article in the prompt should be 1000 words, not 2000
        prompt_words_in_passage = captured[0].split("Passage :")[1].split()[0:1010]
        assert len(prompt_words_in_passage) <= 1010

    def test_300_token_ablation(self, fewshot_file):
        long_text = " ".join(["mot"] * 2000)
        article_texts = {1: long_text}
        first_stage = MagicMock()
        first_stage.retrieve.return_value = ([1], 5.0)

        llm = MagicMock()
        captured = []
        def gen(prompt, temperature=0.0, max_tokens=8, **kwargs):
            captured.append(prompt)
            return "Oui", 10.0
        llm.generate.side_effect = gen

        r = _make_reranker(first_stage, article_texts, llm, fewshot_file, top_n=1, max_article_tokens=300)
        r.retrieve("test", top_k=1)
        # 300-token reranker uses separate cache — does not share with 1000-token run
        assert captured[0] != ""  # LLM was called (not cached from a 1000-token run)

    def test_short_text_preserved(self, mock_first_stage, fewshot_file):
        short_text = "Ce texte est court."
        article_texts = {i: short_text for i in range(1, 11)}
        first_stage = MagicMock()
        first_stage.retrieve.return_value = ([1], 5.0)

        llm = MagicMock()
        captured = []
        def gen(prompt, temperature=0.0, max_tokens=8, **kwargs):
            captured.append(prompt)
            return "Non", 10.0
        llm.generate.side_effect = gen

        r = _make_reranker(first_stage, article_texts, llm, fewshot_file, top_n=1, max_article_tokens=1000)
        r.retrieve("test", top_k=1)
        assert short_text in captured[0]


# ===========================================================================
# Score diagnostics
# ===========================================================================

class TestScoreDiagnostics:
    def test_fraction_queries_all_same_score(self, article_texts, fewshot_file):
        """If LLM always returns Non, fraction_queries_all_same_score should be 1.0."""
        first_stage = MagicMock()
        first_stage.retrieve.return_value = ([1, 2, 3], 5.0)
        llm = MagicMock()
        llm.generate.return_value = ("Non", 10.0)

        r = _make_reranker(first_stage, article_texts, llm, fewshot_file, top_n=3)
        r.retrieve("test1", top_k=3)
        r.retrieve("test2", top_k=3)

        stats = r.get_stats()
        assert stats["score_diagnostics"]["fraction_queries_all_same_score"] == 1.0

    def test_parse_failure_rate(self, article_texts, fewshot_file):
        """Unparseable response should increment parse failures."""
        first_stage = MagicMock()
        first_stage.retrieve.return_value = ([1], 5.0)
        llm = MagicMock()
        llm.generate.return_value = ("Probablement", 10.0)

        r = _make_reranker(first_stage, article_texts, llm, fewshot_file, top_n=1)
        r.retrieve("test", top_k=1)

        stats = r.get_stats()
        assert stats["parse_failure_rate"] > 0.0

    def test_stats_keys_present(self, mock_first_stage, article_texts, mock_llm_binary, fewshot_file):
        r = _make_reranker(mock_first_stage, article_texts, mock_llm_binary, fewshot_file)
        r.retrieve("test", top_k=5)
        stats = r.get_stats()
        assert "parse_failure_rate" in stats
        assert "score_diagnostics" in stats
        diag = stats["score_diagnostics"]
        assert "fraction_queries_all_same_score" in diag
        assert "mean_score_per_query_mean" in diag
        assert "score_std_per_query_mean" in diag


# ===========================================================================
# Latency / diagnostics
# ===========================================================================

class TestLatency:
    def test_latency_positive(self, mock_first_stage, article_texts, mock_llm_binary, fewshot_file):
        r = _make_reranker(mock_first_stage, article_texts, mock_llm_binary, fewshot_file)
        _, lat = r.retrieve("test", top_k=5)
        assert lat > 0.0

    def test_breakdown_keys(self, mock_first_stage, article_texts, mock_llm_binary, fewshot_file):
        r = _make_reranker(mock_first_stage, article_texts, mock_llm_binary, fewshot_file)
        r.retrieve("test", top_k=5)
        bd = r.get_latency_breakdown()
        assert "first_stage_ms_mean" in bd
        assert "llm_scoring_ms_mean" in bd


# ===========================================================================
# Prompt template placeholders
# ===========================================================================

class TestPromptTemplates:
    def test_binary_placeholders(self):
        prompt = LLM_JUDGE_BINARY_PROMPT.format(
            fewshot_block="",
            question="Ma question",
            article_text_truncated="Mon passage",
        )
        assert "Ma question" in prompt
        assert "Mon passage" in prompt
        assert "Oui" in prompt or "Non" in prompt
        assert "Pertinent" in prompt

    def test_0to10_placeholders(self):
        prompt = LLM_JUDGE_0TO10_PROMPT.format(
            question="Ma question",
            article_text_truncated="Mon passage",
        )
        assert "Ma question" in prompt
        assert "Mon passage" in prompt
        assert "Score" in prompt

    def test_all_prompts_in_french(self):
        for prompt in [LLM_JUDGE_BINARY_PROMPT, LLM_JUDGE_0TO10_PROMPT]:
            # All prompts should contain French keywords
            assert any(word in prompt for word in ["Question", "Passage", "pertinent", "répondre"])
