"""
Cross-encoder evaluator for the CRAG evaluate node (Tier 4.1).

Classes
-------
CrossEncoderEvaluator
    Wraps Alibaba-NLP/gte-multilingual-reranker-base via HuggingFace
    AutoTokenizer + AutoModelForSequenceClassification.

    Replaces nreimers/mmarco-mMiniLMv2-L6-H384-v1 (L6) which failed the
    BSARD val gate at 44.4% accuracy — consistent with the Tier 3-C
    reranker failure (same model family, EN→FR legal domain transfer gap).

    mGTE advantages for BSARD:
      - Trained on multilingual data including French
      - Native 8192-token context (eliminates BSARD article truncation pressure)
      - trust_remote_code=True required (custom pooling layer)
      - Outputs raw logits → sigmoid maps to [0, 1]
"""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# CrossEncoderEvaluator
# ---------------------------------------------------------------------------

class CrossEncoderEvaluator:
    """
    Thin wrapper around Alibaba-NLP/gte-multilingual-reranker-base for the
    CRAG evaluate node.

    Parameters
    ----------
    model_name          : HuggingFace model ID
    max_length          : tokenizer max length (1024 covers BSARD p90 article length;
                          mGTE natively supports up to 8192)
    max_article_tokens  : word-level pre-truncation before tokenisation (keeps
                          batch assembly fast; 300 words << 1024 subword tokens)
    device              : torch device string; defaults to 'cuda' if available
    """

    DEFAULT_MODEL = "Alibaba-NLP/gte-multilingual-reranker-base"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        max_length: int = 1024,
        max_article_tokens: int = 300,
        device: str | None = None,
    ):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.model_name         = model_name
        self.max_length         = max_length
        self.max_article_tokens = max_article_tokens
        self.device             = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        self._model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        self._model.eval()
        self._model.to(self.device)

    # ── Public interface ─────────────────────────────────────────────────────

    def predict(
        self,
        pairs: list[tuple[str, str]],
    ) -> tuple[np.ndarray, float]:
        """
        Score a batch of (query, doc_text) pairs.

        Parameters
        ----------
        pairs : list of (query, doc_text) tuples.
                doc_text is word-level pre-truncated to max_article_tokens
                before tokenisation.

        Returns
        -------
        scores     : np.ndarray of shape (len(pairs),) — sigmoid scores in [0, 1].
        latency_ms : wall-clock time for this batch only (time.perf_counter()).
        """
        truncated_pairs = [
            (query, self._truncate(doc_text))
            for query, doc_text in pairs
        ]

        queries   = [p[0] for p in truncated_pairs]
        doc_texts = [p[1] for p in truncated_pairs]

        t0 = time.perf_counter()

        inputs = self._tokenizer(
            queries,
            doc_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self._model(**inputs).logits.squeeze(-1)

        scores = F.sigmoid(logits).cpu().numpy().astype(np.float32)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        return scores, latency_ms

    def score_documents(
        self,
        query: str,
        doc_texts: list[str],
    ) -> tuple[np.ndarray, float]:
        """
        Convenience wrapper: score a single query against multiple doc texts.
        Returns (scores, latency_ms) — same contract as predict().
        """
        pairs = [(query, doc) for doc in doc_texts]
        return self.predict(pairs)

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _truncate(self, text: str) -> str:
        """
        Pre-truncate text to max_article_tokens using whitespace splitting.
        Word-level (not subword) to keep truncation fast and deterministic.
        """
        words = text.split()
        if len(words) <= self.max_article_tokens:
            return text
        return " ".join(words[: self.max_article_tokens])
