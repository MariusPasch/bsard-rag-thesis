"""
Unit tests for the preprocessing pipeline.

T1-D3  Preprocessing symmetry — verifies that the same tokenizer is used
       for documents and queries, and that OOV rate stays below threshold.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running as a script from the project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from retrieval.preprocessing import (
    LEGAL_KEEP,
    NORMALIZATION_VARIANTS,
    TOKENIZERS,
    get_stopwords,
    tokenize_lemmatize,
    tokenize_none,
    tokenize_stem,
)


# ---------------------------------------------------------------------------
# Stopword list sanity
# ---------------------------------------------------------------------------

def test_legal_tokens_not_in_stopwords():
    """All mandatory legal function words must be absent from the stopword list."""
    stops = get_stopwords()
    for token in LEGAL_KEEP:
        assert token not in stops, (
            f"Legal token '{token}' was incorrectly added to the stopword list."
        )


def test_stopwords_nonempty():
    stops = get_stopwords()
    assert len(stops) > 100, "Expected at least 100 French stopwords."


# ---------------------------------------------------------------------------
# Tokenizer output sanity
# ---------------------------------------------------------------------------

SAMPLE_TEXT = (
    "Les conditions de licenciement d'un travailleur ne peuvent pas être "
    "ignorées sauf si l'employeur obtient une autorisation sous réserve "
    "des dispositions légales applicables."
)

SAMPLE_QUERY = "Quelles sont les conditions de licenciement d'un travailleur?"


def test_tokenize_none_returns_list():
    tokens = tokenize_none(SAMPLE_TEXT)
    assert isinstance(tokens, list)
    assert len(tokens) > 5


def test_tokenize_lemmatize_returns_list():
    tokens = tokenize_lemmatize(SAMPLE_TEXT)
    assert isinstance(tokens, list)
    assert len(tokens) > 5


def test_tokenize_stem_returns_list():
    tokens = tokenize_stem(SAMPLE_TEXT)
    assert isinstance(tokens, list)
    assert len(tokens) > 5


def test_legal_tokens_preserved_in_output():
    """
    Tokens like 'pas', 'ne', 'sauf', 'sans' must appear in lemmatized output
    when they are present in the input text.
    """
    tokens = set(tokenize_lemmatize(SAMPLE_TEXT, remove_stopwords=True))
    # 'pas', 'ne', 'sauf', 'sous réserve' carry legal force
    for legal_tok in ["pas", "ne", "sauf"]:
        assert legal_tok in tokens, (
            f"Legal token '{legal_tok}' was removed from tokenizer output — "
            "check LEGAL_KEEP overrides."
        )


def test_all_variants_registered():
    for v in NORMALIZATION_VARIANTS:
        assert v in TOKENIZERS, f"Variant '{v}' missing from TOKENIZERS registry."


# ---------------------------------------------------------------------------
# Preprocessing symmetry (T1-D3)
# ---------------------------------------------------------------------------

def test_preprocessing_symmetry_low_oov():
    """
    When the same tokenizer is applied to a sample corpus and a sample query,
    the fraction of query tokens absent from the corpus vocabulary should be < 50 %.
    If this fails, the document and query pipelines are mismatched.
    """
    # Build a small pseudo-corpus
    corpus = [
        "Les conditions de licenciement sont définies par la loi.",
        "L'employeur doit respecter les délais de préavis légaux.",
        "Le travailleur a droit à une indemnité en cas de licenciement abusif.",
        "Le contrat de travail peut être résilié sous réserve des dispositions.",
        "Sauf stipulation contraire, les obligations contractuelles s'appliquent.",
    ]

    for norm in NORMALIZATION_VARIANTS:
        tokenize = TOKENIZERS[norm]
        vocab = set()
        for doc in corpus:
            vocab.update(tokenize(doc))

        query_tokens = tokenize(SAMPLE_QUERY)
        if not query_tokens:
            continue
        oov = [t for t in query_tokens if t not in vocab]
        oov_rate = len(oov) / len(query_tokens)
        assert oov_rate < 0.5, (
            f"Normalization='{norm}': {oov_rate:.0%} OOV tokens in query — "
            f"possible pipeline mismatch. OOV tokens: {oov}"
        )


# ---------------------------------------------------------------------------
# Doc builders
# ---------------------------------------------------------------------------

def test_doc_builders():
    from retrieval.preprocessing import DOC_BUILDERS

    row = {
        "law_code": "Code Civil",
        "chapter_title": "Des obligations",
        "article_number": "Art. 1234",
        "article_text": "Tout fait quelconque de l'homme qui cause à autrui un dommage.",
    }

    text_only = DOC_BUILDERS["text_only"](row)
    assert text_only == row["article_text"]

    concat = DOC_BUILDERS["concat_2x"](row)
    # law_code should appear twice
    assert concat.count("Code Civil") == 2
    # article_text should be present
    assert row["article_text"] in concat


# ---------------------------------------------------------------------------
# Entry point for direct execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_legal_tokens_not_in_stopwords,
        test_stopwords_nonempty,
        test_tokenize_none_returns_list,
        test_tokenize_lemmatize_returns_list,
        test_tokenize_stem_returns_list,
        test_legal_tokens_preserved_in_output,
        test_all_variants_registered,
        test_preprocessing_symmetry_low_oov,
        test_doc_builders,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
        except Exception as e:
            print(f"  ERROR {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} tests passed.")
