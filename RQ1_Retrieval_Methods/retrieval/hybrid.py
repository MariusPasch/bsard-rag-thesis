"""
Hybrid retrieval for Tier 3.

Classes
-------
HybridRetriever
    Fuses BM25Retriever + DenseRetriever via Reciprocal Rank Fusion (RRF)
    or linear score interpolation.

CrossEncoderReranker
    Two-stage pipeline: first_stage_retriever -> top-N candidates ->
    cross-encoder re-score and re-rank.

Standalone functions
--------------------
hyde_retrieve       -- Hypothetical Document Embeddings (HyDE)
ragfusion_retrieve  -- RAG-Fusion (multi-paraphrase RRF)

All classes expose the same interface contract as BM25Retriever / DenseRetriever:
    retrieve(query: str, top_k: int) -> tuple[list[int], float]
    Returns (ranked_article_ids, latency_ms).
    Latency covers only the retrieval call, not index/model load time.
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np


# ---------------------------------------------------------------------------
# HybridRetriever — RRF and linear interpolation
# ---------------------------------------------------------------------------

class HybridRetriever:
    """
    Fuses a BM25Retriever and a DenseRetriever via RRF or linear interpolation.

    Parameters
    ----------
    sparse_retriever  : BM25Retriever (pre-initialised)
    dense_retriever   : DenseRetriever (pre-initialised, embeddings loaded)
    fusion_method     : "rrf" | "linear"
    alpha             : T3-B only. Weight for dense score in [0, 1].
                        combined = alpha * dense + (1-alpha) * sparse_norm
    rrf_k             : T3-A only. RRF dampening constant (default 60).
    first_stage_k     : number of candidates retrieved per modality before fusion.
    """

    def __init__(
        self,
        sparse_retriever,
        dense_retriever,
        fusion_method: str = "rrf",
        alpha: float = 0.5,
        rrf_k: int = 60,
        first_stage_k: int = 100,
    ):
        if fusion_method not in ("rrf", "linear"):
            raise ValueError(f"fusion_method must be 'rrf' or 'linear', got {fusion_method!r}")
        self._sparse        = sparse_retriever
        self._dense         = dense_retriever
        self.fusion_method  = fusion_method
        self.alpha          = alpha
        self.rrf_k          = rrf_k
        self.first_stage_k  = first_stage_k

    # ── Public interface ─────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 10) -> tuple[list[int], float]:
        """
        Retrieve top-k article IDs via hybrid fusion.

        Latency covers both first-stage calls + fusion computation.
        """
        t0 = time.perf_counter()

        sparse_ids, _ = self._sparse.retrieve(query, top_k=self.first_stage_k)
        dense_ids,  _ = self._dense.retrieve(query,  top_k=self.first_stage_k)

        if self.fusion_method == "rrf":
            ranked = self._rrf(sparse_ids, dense_ids)
        else:
            sparse_scores = self._get_sparse_scores(query, sparse_ids)
            dense_scores  = self._get_dense_scores(query, dense_ids)
            ranked = self._linear(sparse_ids, dense_ids, sparse_scores, dense_scores)

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return ranked[:top_k], latency_ms

    # ── RRF ─────────────────────────────────────────────────────────────────

    def _rrf(
        self,
        sparse_ids: list[int],
        dense_ids: list[int],
    ) -> list[int]:
        """
        Reciprocal Rank Fusion.

        score(d) = 1/(k + rank_sparse(d)) + 1/(k + rank_dense(d))

        Ranks are 1-based. Articles absent from a list get
        rank = first_stage_k + 1 (contributes negligibly).
        """
        k   = self.rrf_k
        fsk = self.first_stage_k

        # Build rank maps (1-based)
        sparse_rank = {aid: i + 1 for i, aid in enumerate(sparse_ids)}
        dense_rank  = {aid: i + 1 for i, aid in enumerate(dense_ids)}

        all_ids = set(sparse_ids) | set(dense_ids)
        scores: dict[int, float] = {}
        for aid in all_ids:
            rs = sparse_rank.get(aid, fsk + 1)
            rd = dense_rank.get(aid,  fsk + 1)
            scores[aid] = 1.0 / (k + rs) + 1.0 / (k + rd)

        return sorted(all_ids, key=lambda a: scores[a], reverse=True)

    # ── Linear interpolation ─────────────────────────────────────────────────

    def _get_sparse_scores(self, query: str, sparse_ids: list[int]) -> dict[int, float]:
        """
        Return raw BM25 scores for the sparse candidate pool.
        Uses the BM25 internal scoring without a second retrieve() call.
        """
        tokens = self._sparse._tokenize(query)
        raw    = self._sparse.bm25.get_scores(tokens)
        id_to_idx = {aid: i for i, aid in enumerate(self._sparse._article_ids)}
        return {aid: float(raw[id_to_idx[aid]]) for aid in sparse_ids if aid in id_to_idx}

    def _get_dense_scores(self, query: str, dense_ids: list[int]) -> dict[int, float]:
        """
        Return cosine similarity scores (inner product on L2-normalised vectors)
        for the dense candidate pool without a second full FAISS search.

        We reuse the FAISS index: encode the query, then compute dot products
        only for the candidate article indices.
        """
        import faiss

        q_text = self._dense._query_prefix + query
        q_vec  = self._dense._encoder.encode_query(q_text).reshape(1, -1)
        faiss.normalize_L2(q_vec)

        # Map article_ids back to FAISS positions
        idx_map = {int(aid): i for i, aid in enumerate(self._dense._article_ids)}

        scores: dict[int, float] = {}
        for aid in dense_ids:
            pos = idx_map.get(aid)
            if pos is None:
                scores[aid] = 0.0
                continue
            # Reconstruct vector from FAISS index and compute dot product
            vec = np.zeros((1, self._dense.embedding_dim), dtype=np.float32)
            self._dense._index.reconstruct(pos, vec[0])
            scores[aid] = float(np.dot(q_vec[0], vec[0]))

        return scores

    def _linear(
        self,
        sparse_ids: list[int],
        dense_ids: list[int],
        sparse_scores: dict[int, float],
        dense_scores: dict[int, float],
    ) -> list[int]:
        """
        Linear score interpolation.

        combined(d) = alpha * dense_norm(d) + (1 - alpha) * sparse_norm(d)

        Dense scores are cosine similarities, already in [0, 1] for L2-normalised
        vectors — no further normalisation needed.

        Sparse (BM25) scores are normalised over the candidate pool:
            sparse_norm(d) = max(0, BM25(d) / (max_BM25 + eps))
        Articles only in the dense pool receive sparse_norm = 0.
        """
        eps = 1e-8
        alpha = self.alpha

        # Sparse normalisation over the sparse candidate pool
        max_bm25 = max(sparse_scores.values()) if sparse_scores else 0.0
        sparse_norm: dict[int, float] = {
            aid: max(0.0, s / (max_bm25 + eps))
            for aid, s in sparse_scores.items()
        }

        all_ids = set(sparse_ids) | set(dense_ids)
        combined: dict[int, float] = {}
        for aid in all_ids:
            d = dense_scores.get(aid, 0.0)
            s = sparse_norm.get(aid, 0.0)
            combined[aid] = alpha * d + (1.0 - alpha) * s

        return sorted(all_ids, key=lambda a: combined[a], reverse=True)

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def audit(self) -> dict:
        """Delegate to dense retriever's audit (used by runner.py)."""
        return self._dense.audit


# ---------------------------------------------------------------------------
# CrossEncoderReranker
# ---------------------------------------------------------------------------

class CrossEncoderReranker:
    """
    Two-stage pipeline: first_stage_retriever -> top-N candidates ->
    cross-encoder re-score -> re-rank.

    Parameters
    ----------
    first_stage_retriever   : any object with .retrieve(query, top_k) interface
    model_name              : HuggingFace CE model ID
    top_n                   : number of first-stage candidates passed to CE
    article_texts           : {article_id: article_text} lookup for CE input
    max_article_tokens      : truncate article text to this many tokens before CE
    """

    def __init__(
        self,
        first_stage_retriever,
        model_name: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        top_n: int = 100,
        article_texts: dict[int, str] | None = None,
        max_article_tokens: int = 400,
    ):
        from sentence_transformers import CrossEncoder

        self._first_stage       = first_stage_retriever
        self._ce                = CrossEncoder(model_name)
        self.top_n              = top_n
        self._article_texts     = article_texts or {}
        self.max_article_tokens = max_article_tokens
        self.model_name         = model_name

        # Latency breakdown (updated on each retrieve() call)
        self.last_first_stage_ms: float = 0.0
        self.last_rerank_ms:      float = 0.0
        self._last_fraction_truncated: float = 0.0

    def _truncate(self, text: str) -> str:
        """
        Truncate *text* to at most max_article_tokens tokens using the CE
        tokenizer, then decode back to a string.
        Falls back to a character-based approximation if tokenizer fails.
        """
        try:
            tok     = self._ce.tokenizer
            enc     = tok(text, add_special_tokens=False)
            ids     = enc["input_ids"][: self.max_article_tokens]
            return tok.decode(ids, skip_special_tokens=True)
        except Exception:
            # ~4 chars per token fallback
            limit = self.max_article_tokens * 4
            return text[:limit]

    def retrieve(self, query: str, top_k: int = 10) -> tuple[list[int], float]:
        """
        1. Retrieve top-N candidates from first stage.
        2. Re-score each (query, article) pair with the cross-encoder.
        3. Return top_k re-ranked article IDs.
        """
        # ── Stage 1: first-stage retrieval ───────────────────────────────────
        t0 = time.perf_counter()
        first_stage_ids, fs_lat = self._first_stage.retrieve(query, top_k=self.top_n)
        self.last_first_stage_ms = fs_lat

        # ── Stage 2: cross-encoder re-ranking ────────────────────────────────
        t1 = time.perf_counter()
        pairs: list[tuple[str, str]] = []
        n_truncated = 0
        for aid in first_stage_ids:
            art_text = self._article_texts.get(aid, "")
            original_len = len(art_text)
            trunc = self._truncate(art_text)
            if len(trunc) < original_len:
                n_truncated += 1
            pairs.append((query, trunc))

        ce_scores = self._ce.predict(pairs, show_progress_bar=False)
        self._last_fraction_truncated = n_truncated / len(first_stage_ids) if first_stage_ids else 0.0

        # Re-rank by CE score descending
        ranked = [aid for aid, _ in sorted(
            zip(first_stage_ids, ce_scores),
            key=lambda x: x[1],
            reverse=True,
        )]

        self.last_rerank_ms = (time.perf_counter() - t1) * 1000.0
        total_ms = (time.perf_counter() - t0) * 1000.0
        return ranked[:top_k], total_ms

    @property
    def fraction_truncated(self) -> float:
        """Fraction of last batch's articles that were truncated."""
        return self._last_fraction_truncated

    @property
    def audit(self) -> dict:
        """Delegate to first stage (for runner.py compatibility)."""
        return getattr(self._first_stage, "audit", {"fraction_truncated": 0.0, "max_tokens_observed": 0})


# ---------------------------------------------------------------------------
# HyDE — Hypothetical Document Embeddings
# ---------------------------------------------------------------------------

_HYDE_FEW_SHOT = """\
Exemples :

Question : Quelles sont les conditions pour résilier un contrat de bail ?
Article : Le preneur peut résilier le contrat de bail à tout moment en respectant un préavis \
de trois mois, sauf convention contraire entre les parties.

Question : Qui est responsable en cas d'accident causé par un animal ?
Article : Le propriétaire d'un animal est responsable du dommage que l'animal a causé, \
soit que l'animal fût sous sa garde, soit qu'il fût égaré ou échappé.

---

"""

_HYDE_PROMPT_TEMPLATE = (
    _HYDE_FEW_SHOT
    + 'Question : "{question}"\n'
    + "Rédigez un court article de loi belge (2-3 phrases) qui répondrait directement "
    + "à cette question.\nArticle :"
)


def hyde_retrieve(
    query: str,
    llm_fn: Callable[[str], str],
    dense_retriever,
    top_k: int = 10,
    hyde_cache: dict[str, str] | None = None,
) -> tuple[list[int], float]:
    """
    Hypothetical Document Embeddings retrieval (k=1, no averaging).

    Parameters
    ----------
    query           : natural language question
    llm_fn          : callable(prompt) -> generated_text (one call per query)
    dense_retriever : DenseRetriever — embeddings already loaded
    top_k           : number of articles to return
    hyde_cache      : optional {query: hypothetical_article} dict for caching
                      (mutated in-place if provided)

    Returns
    -------
    (ranked_article_ids, latency_ms)
    Latency covers LLM call + query encoding + FAISS search.
    """
    t0 = time.perf_counter()

    # Check cache first
    if hyde_cache is not None and query in hyde_cache:
        hyp_text = hyde_cache[query]
    else:
        prompt   = _HYDE_PROMPT_TEMPLATE.format(question=query)
        hyp_text = llm_fn(prompt).strip()
        if hyde_cache is not None:
            hyde_cache[query] = hyp_text

    # Retrieve using the hypothetical article as the "query"
    # (embed hyp_text WITHOUT query prefix — it is document-register text)
    ranked, _ = dense_retriever.retrieve(hyp_text, top_k=top_k)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    return ranked[:top_k], latency_ms


# ---------------------------------------------------------------------------
# RAG-Fusion — multi-paraphrase RRF
# ---------------------------------------------------------------------------

_RAGFUSION_PROMPT_TEMPLATE = (
    "Générez 4 reformulations de la question suivante en français, "
    "en variant la structure syntaxique et le vocabulaire, "
    "mais en conservant le sens juridique exact :\n"
    '"{question}"\n'
    "Reformulations :\n1."
)


def _parse_paraphrases(raw: str, n: int = 4) -> list[str]:
    """
    Parse numbered paraphrases from LLM output.
    Expects lines starting with "1.", "2.", ... or returns raw lines as fallback.
    """
    import re
    lines = raw.strip().splitlines()
    results: list[str] = []
    for line in lines:
        stripped = re.sub(r"^\s*\d+[\.\)]\s*", "", line).strip()
        if stripped:
            results.append(stripped)
        if len(results) >= n:
            break
    # Fallback: use first n non-empty lines
    if not results:
        results = [l.strip() for l in lines if l.strip()][:n]
    return results[:n]


def ragfusion_retrieve(
    query: str,
    paraphrase_fn: Callable[[str], list[str]],
    retriever,
    n_paraphrases: int = 4,
    rrf_k: int = 60,
    top_k: int = 10,
    include_original: bool = True,
    paraphrase_cache: dict[str, list[str]] | None = None,
) -> tuple[list[int], float]:
    """
    RAG-Fusion: generate N paraphrases, retrieve for each, merge via RRF.

    Parameters
    ----------
    query             : original question
    paraphrase_fn     : callable(query) -> list[str] of paraphrases (length n_paraphrases)
    retriever         : DenseRetriever or HybridRetriever
    n_paraphrases     : number of paraphrase variants to generate
    rrf_k             : RRF dampening constant
    top_k             : final ranked list length
    include_original  : always True — original query merged as additional list
    paraphrase_cache  : optional {query: [paraphrase, ...]} (mutated in-place)

    Returns
    -------
    (ranked_article_ids, latency_ms)
    """
    t0 = time.perf_counter()

    # Paraphrase generation
    if paraphrase_cache is not None and query in paraphrase_cache:
        paraphrases = paraphrase_cache[query]
    else:
        paraphrases = paraphrase_fn(query)[:n_paraphrases]
        if paraphrase_cache is not None:
            paraphrase_cache[query] = paraphrases

    # Build query list: always include original
    queries = ([query] if include_original else []) + paraphrases

    # Retrieve for each query
    all_ranked: list[list[int]] = []
    for q in queries:
        ids, _ = retriever.retrieve(q, top_k=100)
        all_ranked.append(ids)

    # RRF over all lists
    all_ids: set[int] = set()
    for ranked in all_ranked:
        all_ids.update(ranked)

    scores: dict[int, float] = {aid: 0.0 for aid in all_ids}
    for ranked in all_ranked:
        rank_map = {aid: i + 1 for i, aid in enumerate(ranked)}
        n_total  = len(ranked)
        for aid in all_ids:
            r = rank_map.get(aid, n_total + 1)
            scores[aid] += 1.0 / (rrf_k + r)

    merged = sorted(all_ids, key=lambda a: scores[a], reverse=True)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    return merged[:top_k], latency_ms
