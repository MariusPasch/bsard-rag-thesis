"""
Text preprocessing pipeline for Tier 1 sparse retrieval.

Decisions implemented:
  T1-D1  Three-way normalization ablation: none / lemmatize / stem
  T1-D2  Legal stopword overrides — mandatory legal function words retained
  T1-D3  Preprocessing symmetry — same pipeline for docs and queries
  T1-D4  Field weighting via law_code repetition (×2 boost)

Performance design:
  - 'none' and 'stem' use a fast regex tokenizer (no spaCy needed).
  - 'lemmatize' uses spaCy fr_core_news_lg with nlp.pipe() for batch processing.
  - batch_tokenize() is the corpus-indexing path; single tokenize_*() functions
    are the query-time path. Both paths use identical logic (T1-D3 symmetry).
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Callable

# ---------------------------------------------------------------------------
# French character regex — covers base ASCII + all accented French letters
# ---------------------------------------------------------------------------
# Matches sequences of word characters including French accented chars.
_FR_TOKEN_RE = re.compile(
    r"[a-zA-Z0-9àâäãæçéèêëîïíìôöóòœùûüúñýÿÀÂÄÃÆÇÉÈÊËÎÏÍÌÔÖÓÒŒÙÛÜÚÑÝŸ]+"
)


def _raw_tokens(text: str) -> list[str]:
    """Fast regex-based tokenization (lower-cased, no punctuation)."""
    return _FR_TOKEN_RE.findall(text.lower())


# ---------------------------------------------------------------------------
# Lazy singletons
# ---------------------------------------------------------------------------

_spacy_nlp = None
_stemmer = None


def _get_nlp():
    global _spacy_nlp
    if _spacy_nlp is None:
        import spacy
        # Disable components not needed for lemmatisation
        _spacy_nlp = spacy.load("fr_core_news_lg", disable=["parser", "ner", "senter"])
    return _spacy_nlp


def _get_stemmer():
    global _stemmer
    if _stemmer is None:
        from nltk.stem.snowball import FrenchStemmer
        _stemmer = FrenchStemmer()
    return _stemmer


# ---------------------------------------------------------------------------
# Legal stopword list (T1-D2)
# ---------------------------------------------------------------------------

# Tokens that MUST be retained — they carry legal force in statutory text.
LEGAL_KEEP: frozenset[str] = frozenset({
    "ne", "pas", "non", "sauf", "sans", "ni",
    "jusqu'à", "dès lors que", "à moins que", "pourvu que",
    "à condition que", "sous réserve",
})


@lru_cache(maxsize=1)
def get_stopwords() -> frozenset[str]:
    """spaCy French built-in stopwords minus mandatory legal function words."""
    nlp = _get_nlp()
    return frozenset(nlp.Defaults.stop_words) - LEGAL_KEEP


# ---------------------------------------------------------------------------
# Single-text tokenizers — query-time path (T1-D3 symmetry)
# ---------------------------------------------------------------------------

def tokenize_none(text: str) -> list[str]:
    """
    Raw tokenization — regex only, no normalization, no stopword removal.
    Baseline variant for the normalization ablation.
    """
    return _raw_tokens(text)


def tokenize_lemmatize(text: str, remove_stopwords: bool = True) -> list[str]:
    """
    Lemmatization via spaCy fr_core_news_lg.
    Default normalization strategy (T1-D1).
    """
    nlp = _get_nlp()
    doc = nlp(text)
    tokens = [t.lemma_.lower() for t in doc if not t.is_space and not t.is_punct]
    if remove_stopwords:
        stops = get_stopwords()
        tokens = [t for t in tokens if t not in stops]
    return tokens


def tokenize_stem(text: str, remove_stopwords: bool = True) -> list[str]:
    """
    Snowball French stemming — regex tokenization + NLTK FrenchStemmer.
    Ablation variant — not the default (T1-D1).
    """
    stemmer = _get_stemmer()
    tokens = [stemmer.stem(t) for t in _raw_tokens(text)]
    if remove_stopwords:
        stops = get_stopwords()
        stemmed_stops = frozenset(stemmer.stem(s) for s in stops)
        tokens = [t for t in tokens if t not in stemmed_stops]
    return tokens


#: Registry of all normalization variants — same function used for docs and queries
TOKENIZERS: dict[str, Callable[[str], list[str]]] = {
    "none":      tokenize_none,
    "lemmatize": tokenize_lemmatize,
    "stem":      tokenize_stem,
}

NORMALIZATION_VARIANTS: list[str] = ["none", "lemmatize", "stem"]


# ---------------------------------------------------------------------------
# Batch tokenization — fast corpus-indexing path
# ---------------------------------------------------------------------------

def batch_tokenize(
    texts: list[str],
    normalization: str,
    batch_size: int = 256,
) -> list[list[str]]:
    """
    Tokenize a list of texts efficiently.

    - 'none' : pure regex — effectively instant for any corpus size.
    - 'stem' : regex + NLTK Snowball — no spaCy overhead.
    - 'lemmatize' : spaCy nlp.pipe() with parser/ner/senter disabled.

    Produces identical token sequences as the single-text tokenize_*() functions
    (T1-D3 preprocessing symmetry).
    """
    if normalization == "none":
        return [_raw_tokens(t) for t in texts]

    if normalization == "stem":
        stemmer = _get_stemmer()
        stops   = get_stopwords()
        stemmed_stops = frozenset(stemmer.stem(s) for s in stops)
        results = []
        for text in texts:
            toks = [stemmer.stem(t) for t in _raw_tokens(text)]
            toks = [t for t in toks if t not in stemmed_stops]
            results.append(toks)
        return results

    # lemmatize — use spaCy pipe
    from tqdm import tqdm
    nlp   = _get_nlp()
    stops = get_stopwords()
    results: list[list[str]] = []
    for doc in tqdm(nlp.pipe(texts, batch_size=batch_size),
                    total=len(texts), desc="        lemmatize", unit="doc"):
        toks = [t.lemma_.lower() for t in doc if not t.is_space and not t.is_punct]
        toks = [t for t in toks if t not in stops]
        results.append(toks)
    return results


# ---------------------------------------------------------------------------
# Document builders — field weighting (T1-D4)
# ---------------------------------------------------------------------------

def build_text_only(row) -> str:
    """Baseline: article_text only."""
    return str(row.get("article_text") or "")


def build_concat_2x(row) -> str:
    """
    Field-weighted concatenation.
    law_code repeated twice = effective 2× BM25/embedding boost without modifying the scorer.
    Format: law_code law_code <title> article_number article_text

    Title field falls back: chapter_title (local dedup parquet) → article_title (HF corpus).
    """
    title = str(row.get("chapter_title") or row.get("article_title") or "")
    parts = [
        str(row.get("law_code") or ""),       # ×1
        str(row.get("law_code") or ""),       # ×2 (repetition boost)
        title,
        str(row.get("article_number") or ""),
        str(row.get("article_text") or ""),
    ]
    return " ".join(p for p in parts if p)


#: Registry of all field-weighting variants
DOC_BUILDERS: dict[str, Callable] = {
    "text_only":    build_text_only,
    "concat_2x":    build_concat_2x,
}

FIELD_WEIGHTING_VARIANTS: list[str] = ["text_only", "concat_2x"]
