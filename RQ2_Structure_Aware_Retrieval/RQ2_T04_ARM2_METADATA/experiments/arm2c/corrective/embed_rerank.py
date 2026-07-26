"""Embedding reranker over the reached pool — the re-SELECT lever (option 1).

The corrective-loop smoke showed Arm-2C *reaches* the gold (reach 0.49->0.80) but
the vectorless 8B list-rerank can't lift it into the top-10 (R@10 stuck ~0.12). This
swaps that reranker for Arm-2A's encoder: rank the reached pool by query-article
embedding cosine instead of by an LLM list call. Same pool, different ranker — so
any R@10 lift is purely the ranker.

`E5Encoder` faithfully replicates `RQ2_T01_SHARED/src/shared/embeddings.py` (the
encoder Arm-1 and Arm-2A share) — `intfloat/multilingual-e5-large-instruct`, the
`passage: ` / `query: ` prefixes (no instruct template), L2-normalised, cosine via
inner product — without importing the `shared` package (whose __init__ pulls
rank_bm25, absent on the light instance), exactly as LlamaClient mirrors shared.llm.
The encode is deterministic, so this needs NO Ollama and can rerank the SHIPPED
single-pass pool reconstructed from the saved traces (see eval_embed_rerank.py).

`sentence-transformers` + the e5 model are the only extra deps (vs the stdlib-only
navigator runtime). The encoder is injectable, so the wiring is mock-testable.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Keep transformers/sentence-transformers from importing TensorFlow/Flax: on the
# azureml conda env that chain (transformers -> tf -> keras) drags in NumPy-1.x-built
# binaries that crash under NumPy 2. We only need the torch path. Must be set BEFORE
# sentence_transformers is imported (done lazily in E5Encoder).
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))           # experiments/arm2c
from navigator_tools import DeepTree                    # noqa: E402
from react_navigator import _to_bsard_and_pad           # noqa: E402  (pool->bsard, pad to k)

ME5_MODEL = "intfloat/multilingual-e5-large-instruct"


class E5Encoder:
    """mE5-instruct encoder, byte-faithful to shared.embeddings.EmbeddingModel."""

    def __init__(self, model_name: str = ME5_MODEL, device: str | None = None,
                 batch_size: int = 64) -> None:
        from sentence_transformers import SentenceTransformer   # lazy: heavy dep
        self._model = SentenceTransformer(model_name, device=device)
        self._bs = batch_size

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        out = self._model.encode([f"passage: {t}" for t in texts],
                                 batch_size=self._bs, show_progress_bar=False,
                                 normalize_embeddings=True, convert_to_numpy=True)
        return out.astype(np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        out = self._model.encode([f"query: {query}"], show_progress_bar=False,
                                 normalize_embeddings=True, convert_to_numpy=True)
        return out[0].astype(np.float32)

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        out = self._model.encode([f"query: {q}" for q in queries], batch_size=self._bs,
                                 show_progress_bar=False, normalize_embeddings=True,
                                 convert_to_numpy=True)
        return out.astype(np.float32)


# ── reached-pool reconstruction (from a saved q*.json trace) ──────────────────

def reached_pool_from_trace(tree: DeepTree, trace: dict) -> list[str]:
    """Rebuild the reached pool (ordered unique leaf node_ids) exactly as
    react_navigator.navigate() does: the agent's selected leaves first (= union of
    each step's `articles`, in visit order), then leaf children of every visited
    node. This is the candidate set the LLM rerank operated on."""
    steps = trace.get("steps", [])
    pool, seen = [], set()
    for s in steps:                                # selected leaves, visit order
        for nid in s.get("articles", []):
            if tree.is_leaf(nid) and nid not in seen:
                seen.add(nid); pool.append(nid)
    for s in steps:                                # leaf children of visited nodes
        node = tree.node(s.get("node_id"))
        if node:
            for c in node.sub_nodes:
                if c.is_leaf and c.node_id not in seen:
                    seen.add(c.node_id); pool.append(c.node_id)
    return pool


def _leaf_text(node, text_field: str) -> str:
    if text_field == "summary":                    # native-EN content summary (Arm-2A enriched)
        meta = node.metadata or {}
        return meta.get("content_summary_en") or node.text or node.title or ""
    return node.text or node.title or ""           # raw French article text (Arm-2A article-unit)


def embed_rerank_pool(query: str, pool_node_ids: list[str], tree: DeepTree, encoder,
                      *, k: int = 100, text_field: str = "text") -> list[int]:
    """Rank the reached pool by query-article embedding cosine, then pad to k via
    the same document-order breadth fill navigate() uses. Returns bsard_ids."""
    cands = [nid for nid in pool_node_ids if tree.is_leaf(nid)]
    if not cands:
        return _to_bsard_and_pad([], tree, k)
    texts = [_leaf_text(tree.node(nid), text_field) for nid in cands]
    qv = encoder.encode_query(query)               # (d,)
    pv = encoder.encode_passages(texts)            # (N, d)
    sims = pv @ qv                                 # cosine (both L2-normed)
    order = np.argsort(-sims)                       # descending
    ranked_nodes = [cands[int(i)] for i in order]
    return _to_bsard_and_pad(ranked_nodes, tree, k)


# ── cached path: encode each unique article ONCE, then look up (no GPU needed) ──

def encode_nodes(node_ids, tree: DeepTree, encoder, *, text_field: str = "text") -> dict:
    """Encode the UNIQUE leaf node_ids' texts once → {node_id: vector}. Reranking
    many queries re-embeds the same articles repeatedly; dedup makes a CPU run
    feasible (minutes vs an hour). Deterministic, so the result is identical."""
    uniq = [n for n in dict.fromkeys(node_ids) if tree.is_leaf(n)]
    if not uniq:
        return {}
    texts = [_leaf_text(tree.node(n), text_field) for n in uniq]
    vecs = encoder.encode_passages(texts)
    return {n: vecs[i] for i, n in enumerate(uniq)}


def embed_rerank_pool_cached(query_vec, pool_node_ids, tree: DeepTree,
                             vec_by_node: dict, *, k: int = 100) -> list[int]:
    """Rank a pool by cosine using precomputed vectors (`encode_nodes` + a
    precomputed query vector). Same output as embed_rerank_pool, no encoding."""
    cands = [nid for nid in pool_node_ids if nid in vec_by_node]
    if not cands:
        return _to_bsard_and_pad([], tree, k)
    mat = np.vstack([vec_by_node[nid] for nid in cands])
    order = np.argsort(-(mat @ query_vec))
    return _to_bsard_and_pad([cands[int(i)] for i in order], tree, k)
