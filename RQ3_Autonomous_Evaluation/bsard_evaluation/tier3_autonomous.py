"""
evaluation/tier3_autonomous.py
────────────────────────────────
Tier 3 — Autonomous reference-free retrieval evaluation.

Retrieval-only scope: all components evaluate retrieved contexts without
requiring a pre-generated answer.  The UMBRELA qrels produced here bridge
back into Tier 2 to create an autonomous analogue of supervised NDCG/MAP.

Components
──────────
  umbrela   : UMBRELA zero-shot DNA prompt (LLM judge, 0–3 grade per doc).
               Produces TREC qrels that feed directly into Tier 2 NDCG/MAP.
               Cross-model discipline: judge_model must differ from any LLM
               used in Tier 4 agentic systems (self-preference bias confirmed
               in TREC 2024 RAG Track).
  erag      : eRAG — per-document answer-generation utility score.
               LLaMA 3.1 8B via Ollama (or any OpenAI-compatible endpoint).
               Unique signal: measures whether a retrieved article actually
               enables correct reasoning, not just whether it is "relevant".
  ragas_wa  : RAGAS LLMContextPrecisionWithoutReference + HyDE synthetic
               response.  HyDE call is made independently of retrieval, so
               the synthetic response acts as a proxy reference without
               circular dependency.
  ragas_wb  : RAGAS LLMContextPrecisionWithoutReference + query-as-response.
               Zero extra LLM calls.  Diagnostic baseline vs. Workaround A;
               NOT included in AQS.
  ares      : ARES fine-tuned T5-large + PPI confidence bounds.
               context_relevance only when answers=None (retrieval-only scope).

AQS (Autonomous Quality Score)
───────────────────────────────
Weighted average over {umbrela, erag, ares, ragas_wa} mean scores.
ragas_wb is intentionally excluded — it is a diagnostic comparison baseline.

Input format
────────────
queries             : dict[query_id, str]
contexts_with_ranks : dict[query_id, list[tuple[doc_id, doc_text, rank]]]
                      doc_id matches BSARD article id (str or int).
                      rank is 1-based position in retrieval output.

Cost guidance (222 test queries, k=10 docs, gpt-4o-mini pricing)
──────────────────────────────────────────────────────────────────
  umbrela  : ~$0.17 total  | ~4 min parallel  | API only, no GPU
  erag     : ~$0.25 total  | ~17-55 min GPU   | GPU or API
  ragas_wa : ~$0.24 total  | ~4 min parallel  | API only, no GPU
  ragas_wb : ~$0.20 total  | ~4 min parallel  | API only, no GPU
  ares     : $0 inference  | ~18s GPU / 3-7 min CPU | GPU for fine-tune (one-time)

Import failures are handled gracefully — missing libraries skip their
component with a warning rather than crashing.
"""

from __future__ import annotations

import random
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Quota-error helper
# ---------------------------------------------------------------------------

def _is_quota_error(exc: Exception) -> bool:
    """Return True if exc is an OpenAI insufficient_quota (429) error."""
    msg = str(exc).lower()
    return "insufficient_quota" in msg or "exceeded your current quota" in msg


def _wait_on_quota(exc: Exception, context: str = "") -> bool:
    """
    If exc is a quota error, print a clear message and block until the user
    presses Enter.  Returns True so the caller can retry.  Returns False for
    any other exception so it falls through to the normal warning path.
    """
    if not _is_quota_error(exc):
        return False
    label = f" [{context}]" if context else ""
    print(
        f"\n{'='*70}\n"
        f"QUOTA EXCEEDED{label}\n"
        f"{exc}\n\n"
        f"Fix your OpenAI billing / quota, then press Enter to retry ...\n"
        f"{'='*70}",
        flush=True,
    )
    input()
    print("Retrying ...", flush=True)
    return True

# ── AQS component weights (sum to 1.0 over included components) ───────────────
# Weights reflect external validation strength and downstream-task correlation:
#   umbrela  — highest: confirmed human-assessor agreement (TREC 2024 RAG Track)
#   erag     — second: highest empirical correlation with downstream RAG perf
#   ares     — moderate: adds statistical confidence bounds via PPI
#   ragas_wa — supplementary: HyDE-based context precision signal
#   ragas_wb — excluded: diagnostic baseline, not an independent signal
AQS_WEIGHTS: Dict[str, float] = {
    "umbrela":  0.35,
    "erag":     0.30,
    "ares":     0.20,
    "ragas_wa": 0.15,
}


# ── Return type carrying both metrics and UMBRELA qrels ───────────────────────

@dataclass
class Tier3Result:
    """
    Return type for evaluate_tier3().

    metrics            : flat dict of all "T3/<comp>/<dim>" scores plus "T3/AQS".
    umbrela_qrels      : TREC qrels dict {query_id: {doc_id: grade}} produced by
                         UMBRELA; None if UMBRELA was not in the component list.
                         Consumed by the harness to bridge into a second Tier 2
                         pass keyed "T2-umbrela/", enabling the core RQ3
                         comparison: do UMBRELA-based rankings agree with
                         BSARD-supervised rankings?
    erag_per_doc       : {query_id: {doc_id: 0|1}} per-(q,d) eRAG score
                         (1 = grounded answer, 0 = abstention); None if eRAG
                         did not run.  Required for (q,d)-level ROC/AUC and
                         calibration analysis in RQ3a.
    ragas_wa_per_query : {query_id: context_precision} per-query RAGAS-WA
                         score; None if RAGAS-WA did not run.  Required for
                         query-level Spearman/Kendall analysis.
    hyde_responses     : {query_id: hyde_response_text} synthetic responses
                         generated during RAGAS-WA's HyDE step; None if
                         RAGAS-WA did not run.  Used in the qualitative HyDE
                         inspection that substitutes for the lost RAGAS-WB
                         control.
    """
    metrics: Dict[str, float] = field(default_factory=dict)
    umbrela_qrels: Optional[Dict[str, Dict[str, int]]] = None
    erag_per_doc: Optional[Dict[str, Dict[str, int]]] = None
    ragas_wa_per_query: Optional[Dict[str, float]] = None
    hyde_responses: Optional[Dict[str, str]] = None


# ════════════════════════════════════════════════════════════════════════
# Component: UMBRELA
# ════════════════════════════════════════════════════════════════════════

_UMBRELA_SYSTEM = (
    "You are a relevance assessor for a Belgian statutory law retrieval system. "
    "Rate how relevant the given passage is for answering the query."
)

_UMBRELA_PROMPT = """\
Query: {query}

Passage: {passage}

Rate the relevance of the passage for answering the query using this scale:
[0] Not relevant — the passage has nothing to do with the query.
[1] Related — the passage is topically related but does not answer the query.
[2] Relevant — the passage partially answers the query; the answer may be implicit.
[3] Highly relevant — the passage directly and completely addresses the query.

Respond with only the single digit (0, 1, 2, or 3)."""


def run_umbrela(
    queries: Dict[str, str],
    contexts_with_ranks: Dict[str, List[Tuple[str, str, int]]],
    judge_model: str = "gpt-4o-mini",
    sample_ids: Optional[List[str]] = None,
) -> Tuple[Dict[str, float], Dict[str, Dict[str, int]]]:
    """
    UMBRELA — Zero-shot DNA relevance grading via LLM judge.

    Makes one LLM call per (query, document) pair using the UMBRELA DNA
    prompt and assigns a grade on the 0–3 TREC scale.  Returns both
    aggregate scalar metrics and a full TREC qrels dict so grades can feed
    directly into tier2_supervised.py NDCG/MAP computations without
    transformation.

    The judge_model MUST differ from the LLM used in Tier 4 agentic systems
    to prevent self-preference bias (confirmed in TREC 2024 RAG Track study).

    Parameters
    ----------
    queries             : {query_id: query_text}
    contexts_with_ranks : {query_id: [(doc_id, doc_text, rank), ...]}
    judge_model         : OpenAI-compatible model name; default gpt-4o-mini
    sample_ids          : restrict evaluation to these query IDs

    Returns
    -------
    (metrics_dict, qrels_dict)
      metrics_dict : T3/umbrela/* scalar scores
      qrels_dict   : {query_id: {doc_id: int}} — TREC format, grades 0–3

    Cost (gpt-4o-mini, 222 queries × k=10)
    ─────────────────────────────────────
    ~$0.000075 per (query, doc) pair → ~$0.17 total
    ~0.5–1.5 s per pair (API) → ~4 min with 10× parallelism

    References
    ----------
    Upadhyay et al. (2024). UMBRELA: UMbrela is the Recreational Evaluation
    for LLMS Assessing RAG. TREC 2024 RAG Track.
    """
    try:
        from openai import OpenAI
    except ImportError:
        warnings.warn(
            "openai not installed.  Run: pip install openai  "
            "Skipping T3/umbrela.",
            stacklevel=2,
        )
        return {}, {}

    client = OpenAI()
    qids = sample_ids or list(queries.keys())

    qrels: Dict[str, Dict[str, int]] = {}
    all_grades: List[int] = []

    for qid in qids:
        query_text = queries[qid]
        docs = contexts_with_ranks.get(qid, [])
        qrels[qid] = {}

        for doc_id, doc_text, _rank in docs:
            prompt = _UMBRELA_PROMPT.format(query=query_text, passage=doc_text)
            while True:
                try:
                    response = client.chat.completions.create(
                        model=judge_model,
                        messages=[
                            {"role": "system", "content": _UMBRELA_SYSTEM},
                            {"role": "user",   "content": prompt},
                        ],
                        max_tokens=5,
                        temperature=0.0,
                    )
                    raw = response.choices[0].message.content.strip()
                    grade = int(raw[0]) if raw and raw[0].isdigit() else 0
                    grade = max(0, min(3, grade))
                    break
                except Exception as exc:
                    if _wait_on_quota(exc, f"UMBRELA qid={qid} doc={doc_id}"):
                        continue
                    warnings.warn(
                        f"UMBRELA call failed for qid={qid}, doc={doc_id}: {exc}",
                        stacklevel=2,
                    )
                    grade = 0
                    break

            qrels[qid][str(doc_id)] = grade
            all_grades.append(grade)

    out: Dict[str, float] = {}
    if all_grades:
        out["T3/umbrela/mean_grade"]       = float(np.mean(all_grades))
        # Normalise to [0, 1] for AQS (grades are 0–3)
        out["T3/umbrela/mean"]             = float(np.mean(all_grades)) / 3.0
        out["T3/umbrela/relevant_fraction"] = float(
            sum(1 for g in all_grades if g >= 2) / len(all_grades)
        )

    return out, qrels


# ════════════════════════════════════════════════════════════════════════
# Component: eRAG
# ════════════════════════════════════════════════════════════════════════

_ERAG_SYSTEM = (
    "You are an assistant that answers legal questions using only the provided document. "
    "If the document does not contain enough information to answer the question, "
    "respond with exactly: "
    "'I cannot answer this question based on the provided document.'"
)

_ERAG_PROMPT = """\
Document:
{document}

Question: {query}

Using only the information in the document above, answer the question. \
If the document lacks sufficient information, respond with: \
"I cannot answer this question based on the provided document.\""""

_ERAG_ABSTENTION = "i cannot answer this question based on the provided document"


def run_erag(
    queries: Dict[str, str],
    contexts_with_ranks: Dict[str, List[Tuple[str, str, int]]],
    llm_model: str = "gpt-4o-mini",
    sample_ids: Optional[List[str]] = None,
) -> Tuple[Dict[str, float], Dict[str, Dict[str, int]]]:
    """
    eRAG — per-document answer-generation utility score.

    For each (query, document) pair, prompts the LLM to answer the query
    using only that document as context.  Scores each document by whether
    the LLM produces a grounded answer (score 1.0) or abstains (score 0.0).

    This is the only Tier 3 metric that measures whether a retrieved article
    actually *enables* correct reasoning — a signal orthogonal to relevance
    judgement.  Its high empirical correlation with downstream RAG performance
    makes it the second-highest-weighted AQS component.

    For local GPU inference, prefix the model name with "ollama/":
        llm_model="ollama/llama3.1:8b"
    This routes calls to the Ollama OpenAI-compatible endpoint at
    http://localhost:11434/v1.

    Parameters
    ----------
    queries             : {query_id: query_text}
    contexts_with_ranks : {query_id: [(doc_id, doc_text, rank), ...]}
    llm_model           : model name; prefix "ollama/" for local GPU routing
    sample_ids          : restrict evaluation to these query IDs

    Returns
    -------
    (metrics_dict, per_doc_dict)
      metrics_dict : T3/erag/* aggregate scores
      per_doc_dict : {query_id: {doc_id: 0|1}} — 1 if grounded answer,
                     0 if abstention; required for (q,d)-level analysis.

    Cost (LLaMA 3.1 8B via Together.ai, 222 queries × k=10)
    ─────────────────────────────────────────────────────────
    ~$0.000112 per (query, doc) pair → ~$0.25 total (API)
    ~1–2.5 s per pair on GPU → ~17–55 min (RTX 3090 / A100)
    GPU VRAM: ~16 GB (bf16) or ~4.5 GB (4-bit quantised)

    References
    ----------
    Salemi & Zamani (2024). Evaluating Retrieval Quality for RAG.
    SIGIR 2024.
    """
    try:
        from openai import OpenAI
    except ImportError:
        warnings.warn(
            "openai not installed (required for both API and Ollama endpoint).  "
            "Run: pip install openai  Skipping T3/erag.",
            stacklevel=2,
        )
        return {}, {}

    base_url: Optional[str] = None
    model = llm_model
    api_key = None
    if llm_model.startswith("ollama/"):
        base_url = "http://localhost:11434/v1"
        model    = llm_model[len("ollama/"):]
        api_key  = "ollama"

    client = OpenAI(base_url=base_url, api_key=api_key)
    qids = sample_ids or list(queries.keys())
    per_query_scores: List[float] = []
    per_doc: Dict[str, Dict[str, int]] = {}

    for qid in qids:
        query_text = queries[qid]
        docs = contexts_with_ranks.get(qid, [])
        doc_scores: List[float] = []
        per_doc[qid] = {}

        for doc_id, doc_text, _rank in docs:
            prompt = _ERAG_PROMPT.format(document=doc_text, query=query_text)
            while True:
                try:
                    response = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": _ERAG_SYSTEM},
                            {"role": "user",   "content": prompt},
                        ],
                        max_tokens=200,
                        temperature=0.0,
                    )
                    answer = response.choices[0].message.content.strip().lower()
                    score  = 0.0 if _ERAG_ABSTENTION in answer else 1.0
                    break
                except Exception as exc:
                    if _wait_on_quota(exc, f"eRAG qid={qid} doc={doc_id}"):
                        continue
                    warnings.warn(
                        f"eRAG call failed for qid={qid}, doc={doc_id}: {exc}",
                        stacklevel=2,
                    )
                    score = 0.0
                    break
            doc_scores.append(score)
            per_doc[qid][str(doc_id)] = int(score)

        if doc_scores:
            per_query_scores.append(float(np.mean(doc_scores)))

    out: Dict[str, float] = {}
    if per_query_scores:
        out["T3/erag/mean"]              = float(np.mean(per_query_scores))
        out["T3/erag/grounded_fraction"] = float(np.mean(per_query_scores))
    return out, per_doc


# ════════════════════════════════════════════════════════════════════════
# Component: RAGAS Workaround A (HyDE synthetic response)
# ════════════════════════════════════════════════════════════════════════

_HYDE_SYSTEM = (
    "You are a Belgian statutory law expert. "
    "Write a concise hypothetical statutory answer to the question. "
    "Do not reference any retrieved documents — rely only on your knowledge."
)

_HYDE_PROMPT = "Question: {query}\n\nHypothetical statutory answer:"


def run_ragas_workaround_a(
    queries: Dict[str, str],
    contexts_with_ranks: Dict[str, List[Tuple[str, str, int]]],
    llm_model: str = "gpt-4o-mini",
    sample_ids: Optional[List[str]] = None,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, str]]:
    """
    RAGAS Workaround A — LLMContextPrecisionWithoutReference + HyDE response.

    Step 1: Generate one hypothetical statutory answer per query (HyDE-style),
            without any retrieved context.  Because this synthetic response is
            generated independently of the retrieval output, it acts as a
            proxy reference without introducing circular dependency.

    Step 2: Pass (query, synthetic_response, retrieved_contexts) to RAGAS
            LLMContextPrecisionWithoutReference.  RAGAS then scores each
            retrieved chunk for how well it supports the synthetic response.

    Expected to produce higher-quality scores than Workaround B because the
    HyDE response encodes domain priors about what a correct answer looks like.

    Parameters
    ----------
    queries             : {query_id: query_text}
    contexts_with_ranks : {query_id: [(doc_id, doc_text, rank), ...]}
    llm_model           : OpenAI-compatible model for both HyDE and RAGAS calls
    sample_ids          : restrict evaluation to these query IDs

    Returns
    -------
    (metrics_dict, per_query_dict, hyde_dict)
      metrics_dict   : T3/ragas_wa/* aggregate scores
      per_query_dict : {query_id: float context_precision in [0, 1]}
      hyde_dict      : {query_id: str synthetic response}; needed for the
                       qualitative HyDE inspection that substitutes for the
                       lost RAGAS-WB control.

    Cost (gpt-4o-mini, 222 queries × k=10)
    ────────────────────────────────────────
    HyDE: ~$0.000021 per query (amortised per pair: ~$0.0000021)
    RAGAS scoring: ~$0.000096 per (query, doc) pair
    Total per pair: ~$0.000107 → ~$0.24 total
    ~1 s per pair → ~4 min with parallelism; no GPU required

    References
    ----------
    Es et al. (2023). Ragas: Automated Evaluation of Retrieval Augmented
    Generation. arXiv:2309.15217.
    Lewis et al. (2021). Retrieval-Augmented Generation for Knowledge-
    Intensive NLP Tasks. NeurIPS 2020. (HyDE concept)
    """
    try:
        from openai import OpenAI
    except ImportError:
        warnings.warn(
            "openai not installed.  Run: pip install openai  "
            "Skipping T3/ragas_wa.",
            stacklevel=2,
        )
        return {}, {}, {}

    try:
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import LLMContextPrecisionWithoutReference
        from datasets import Dataset
    except ImportError:
        warnings.warn(
            "ragas or datasets not installed.  "
            "Run: pip install 'ragas>=0.2' datasets  Skipping T3/ragas_wa.",
            stacklevel=2,
        )
        return {}, {}, {}

    client = OpenAI()
    qids   = sample_ids or list(queries.keys())

    # Step 1 — HyDE: generate one synthetic response per query
    synthetic: Dict[str, str] = {}
    for qid in qids:
        while True:
            try:
                resp = client.chat.completions.create(
                    model=llm_model,
                    messages=[
                        {"role": "system", "content": _HYDE_SYSTEM},
                        {"role": "user",   "content": _HYDE_PROMPT.format(query=queries[qid])},
                    ],
                    max_tokens=200,
                    temperature=0.3,
                )
                synthetic[qid] = resp.choices[0].message.content.strip()
                break
            except Exception as exc:
                if _wait_on_quota(exc, f"HyDE qid={qid}"):
                    continue
                warnings.warn(
                    f"HyDE generation failed for qid={qid}: {exc}.  "
                    "Falling back to query text.",
                    stacklevel=2,
                )
                synthetic[qid] = queries[qid]
                break

    # Step 2 — RAGAS LLMContextPrecisionWithoutReference
    data = {
        "user_input":          [queries[qid] for qid in qids],
        "response":            [synthetic.get(qid, queries[qid]) for qid in qids],
        "retrieved_contexts":  [
            [doc_text for _, doc_text, _ in contexts_with_ranks.get(qid, [])]
            for qid in qids
        ],
    }
    dataset = Dataset.from_dict(data)

    while True:
        try:
            result = ragas_evaluate(dataset, metrics=[LLMContextPrecisionWithoutReference()])
            df     = result.to_pandas()
            col    = "llm_context_precision_without_reference"
            if col in df.columns:
                score = float(df[col].mean())
                per_query = {
                    qid: float(v) for qid, v in zip(qids, df[col].tolist())
                }
                return (
                    {
                        "T3/ragas_wa/context_precision": score,
                        "T3/ragas_wa/mean":              score,
                    },
                    per_query,
                    synthetic,
                )
            break
        except Exception as exc:
            if _wait_on_quota(exc, "RAGAS-WA evaluate"):
                continue
            warnings.warn(
                f"RAGAS Workaround A evaluation failed: {exc}  Skipping.",
                stacklevel=2,
            )
            break
    return {}, {}, synthetic


# ════════════════════════════════════════════════════════════════════════
# Component: RAGAS Workaround B (query-as-response, zero extra LLM calls)
# ════════════════════════════════════════════════════════════════════════

def run_ragas_workaround_b(
    queries: Dict[str, str],
    contexts_with_ranks: Dict[str, List[Tuple[str, str, int]]],
    sample_ids: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    RAGAS Workaround B — LLMContextPrecisionWithoutReference + query as response.

    Zero-cost baseline: populates the RAGAS 'response' field with the query
    text itself.  No extra LLM calls beyond the k chunk-scoring judgements that
    RAGAS makes internally per query.

    Expected to produce lower scores than Workaround A because a query is a
    weaker proxy for the answer than a HyDE response.  However, for BSARD
    statutory text, where correct articles frequently share exact legal
    terminology with the query, the gap may be small — making this a useful
    cost-free diagnostic.

    Diagnostic use only — NOT included in AQS.  Results appear under
    "T3/ragas_wb/" and should be compared against "T3/ragas_wa/" to
    quantify how much the HyDE step adds.

    Parameters
    ----------
    queries             : {query_id: query_text}
    contexts_with_ranks : {query_id: [(doc_id, doc_text, rank), ...]}
    sample_ids          : restrict evaluation to these query IDs

    Returns keys prefixed "T3/ragas_wb/".

    Cost (gpt-4o-mini, 222 queries × k=10)
    ────────────────────────────────────────
    ~$0.0000885 per (query, doc) pair → ~$0.20 total; no GPU required.
    """
    try:
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import LLMContextPrecisionWithoutReference
        from datasets import Dataset
    except ImportError:
        warnings.warn(
            "ragas or datasets not installed.  "
            "Run: pip install 'ragas>=0.2' datasets  Skipping T3/ragas_wb.",
            stacklevel=2,
        )
        return {}

    qids = sample_ids or list(queries.keys())
    data = {
        "user_input":          [queries[qid] for qid in qids],
        "response":            [queries[qid] for qid in qids],   # query = proxy response
        "retrieved_contexts":  [
            [doc_text for _, doc_text, _ in contexts_with_ranks.get(qid, [])]
            for qid in qids
        ],
    }
    dataset = Dataset.from_dict(data)

    while True:
        try:
            result = ragas_evaluate(dataset, metrics=[LLMContextPrecisionWithoutReference()])
            df     = result.to_pandas()
            col    = "llm_context_precision_without_reference"
            if col in df.columns:
                score = float(df[col].mean())
                return {
                    "T3/ragas_wb/context_precision": score,
                    "T3/ragas_wb/mean":              score,
                }
            break
        except Exception as exc:
            if _wait_on_quota(exc, "RAGAS-WB evaluate"):
                continue
            warnings.warn(
                f"RAGAS Workaround B evaluation failed: {exc}  Skipping.",
                stacklevel=2,
            )
            break
    return {}


# ════════════════════════════════════════════════════════════════════════
# Component: ARES (context_relevance only in retrieval-only scope)
# ════════════════════════════════════════════════════════════════════════

def run_ares(
    queries: Dict[str, str],
    contexts_with_ranks: Dict[str, List[Tuple[str, str, int]]],
    answers: Optional[Dict[str, str]] = None,
    in_domain_passages_path: str = "data/bsard_passages.tsv",
    human_pref_path: str = "data/ares_human_pref.tsv",
    few_shot_path: str = "data/ares_few_shot.tsv",
    sample_ids: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    ARES — Automated RAG Evaluation System.

    Generates synthetic training data from the BSARD corpus, fine-tunes
    lightweight T5-large LM judges, and uses Prediction-Powered Inference
    (PPI) to produce statistically calibrated system rankings with confidence
    bounds — the only Tier 3 component to offer this.

    Retrieval-only scope (answers=None or empty): evaluates context_relevance
    only, skipping answer_faithfulness and answer_relevance.  When answers
    are provided, all three dimensions are evaluated.

    Parameters
    ----------
    queries             : {query_id: query_text}
    contexts_with_ranks : {query_id: [(doc_id, doc_text, rank), ...]}
    answers             : optional {query_id: generated_answer}; if None,
                          only context_relevance is scored
    in_domain_passages_path : BSARD passage TSV for synthetic data generation
    human_pref_path         : ~150 human-annotated TSV for PPI calibration
    few_shot_path           : few-shot prompt examples TSV

    Returns keys prefixed "T3/ares/".

    Cost
    ────
    Fine-tuning (one-time): ~6–8 h on T4 GPU (~$12–18 on Colab Pro)
    Inference: $0 — local T5-large forward pass
      GPU (4 GB+ VRAM): ~5–15 ms per (query, doc) pair
      CPU: ~80–150 ms per (query, doc) pair

    References
    ----------
    Saad-Falcon et al. (2024). ARES: An Automated Evaluation Framework
    for RAG Systems. NAACL 2024.
    """
    try:
        from ares import ARES as ARESEvaluator
    except ImportError:
        warnings.warn(
            "ARES not installed.  Run: pip install ares-ai  "
            "Skipping T3/ares.",
            stacklevel=2,
        )
        return {}

    qids = sample_ids or list(queries.keys())
    retrieval_only = answers is None or all(not v for v in answers.values())
    labels = ["context_relevance"]
    if not retrieval_only:
        labels += ["answer_faithfulness", "answer_relevance"]

    unlabeled_data = [
        {
            "query":   queries[qid],
            "context": " ".join(
                doc_text for _, doc_text, _ in contexts_with_ranks.get(qid, [])
            ),
            "answer":  (answers or {}).get(qid, ""),
        }
        for qid in qids
    ]

    ares   = ARESEvaluator()
    config = {
        "in_domain_prompts_dataset": few_shot_path,
        "unlabeled_evaluation_set":  unlabeled_data,
        "gold_label_path":           human_pref_path,
        "documents":                 in_domain_passages_path,
        "labels":                    labels,
        "GPT_scoring":               False,
        "swap_human_labels_for_gpt": False,
    }

    try:
        results = ares.evaluate_RAG(config)
    except Exception as exc:
        warnings.warn(f"ARES evaluation failed: {exc}  Skipping T3/ares.", stacklevel=2)
        return {}

    out: Dict[str, float] = {}
    for dim in labels:
        val = results.get(dim)
        if val is not None:
            out[f"T3/ares/{dim}"] = float(val)

    if out:
        out["T3/ares/mean"] = float(np.mean(list(out.values())))
    return out


# ════════════════════════════════════════════════════════════════════════
# AQS — Autonomous Quality Score
# ════════════════════════════════════════════════════════════════════════

def compute_aqs(component_results: Dict[str, float]) -> Optional[float]:
    """
    Compute the Autonomous Quality Score (AQS) as a weighted average
    of available component mean scores.

    Only components that actually ran (have a "T3/<comp>/mean" key)
    contribute.  Weights are re-normalised to sum to 1 over available
    components so that a missing component does not deflate the score.

    ragas_wb is intentionally absent from AQS_WEIGHTS — it is a diagnostic
    comparison baseline, not an independent quality signal.

    Used as the aggregate signal for the RQ3 rank-correlation analysis
    against supervised Recall@100 and Tier 2 NDCG rankings.
    """
    available: Dict[str, float] = {}
    for comp, weight in AQS_WEIGHTS.items():
        if f"T3/{comp}/mean" in component_results:
            available[comp] = weight

    if not available:
        return None

    total_weight = sum(available.values())
    aqs = sum(
        component_results[f"T3/{comp}/mean"] * w / total_weight
        for comp, w in available.items()
    )
    return float(aqs)


# ════════════════════════════════════════════════════════════════════════
# Main Tier 3 entry point
# ════════════════════════════════════════════════════════════════════════

def evaluate_tier3(
    queries: Dict[str, str],
    contexts_with_ranks: Dict[str, List[Tuple[str, str, int]]],
    components: Optional[List[str]] = None,
    sample_size: Optional[int] = 150,
    llm_model: str = "gpt-4o-mini",
    judge_model_tier4: Optional[str] = None,
    ares_paths: Optional[Dict[str, str]] = None,
    answers: Optional[Dict[str, str]] = None,
    seed: int = 42,
) -> Tier3Result:
    """
    Run all requested Tier 3 autonomous evaluation components.

    Parameters
    ----------
    queries             : {query_id: query_text}
    contexts_with_ranks : {query_id: [(doc_id, doc_text, rank), ...]}
                          Ranked retrieval output with BSARD article IDs and
                          texts.  rank is 1-based.
    components          : subset of ["umbrela", "erag", "ares", "ragas_wa",
                          "ragas_wb"].  Defaults to AQS_WEIGHTS keys
                          (umbrela, erag, ares, ragas_wa).
    sample_size         : number of queries evaluated by LLM-based components
                          for cost control.  None = evaluate all queries.
    llm_model           : model for eRAG and RAGAS workaround LLM calls.
                          Prefix "ollama/" for local GPU routing (eRAG only).
    judge_model_tier4   : if provided, used as UMBRELA judge model instead of
                          llm_model — enforces cross-model discipline when
                          evaluating Tier 4 agentic systems.  Should be a
                          different model family from the Tier 4 LLM.
    ares_paths          : dict with keys "passages", "human_pref", "few_shot"
                          pointing to ARES input files under data_storage/
    answers             : optional {query_id: str}; if provided, passed to
                          ARES so it can also evaluate answer_faithfulness
                          and answer_relevance beyond context_relevance
    seed                : random seed for reproducible subsampling

    Returns
    -------
    Tier3Result
      .metrics       — flat dict: "T3/<comp>/<dim>" scores + "T3/AQS"
      .umbrela_qrels — {query_id: {doc_id: grade}} if umbrela ran, else None
    """
    if components is None:
        components = list(AQS_WEIGHTS.keys())

    all_qids = list(queries.keys())
    if sample_size and sample_size < len(all_qids):
        random.seed(seed)
        sample_ids = random.sample(all_qids, sample_size)
    else:
        sample_ids = all_qids

    metrics: Dict[str, float] = {}
    umbrela_qrels: Optional[Dict[str, Dict[str, int]]] = None
    erag_per_doc: Optional[Dict[str, Dict[str, int]]] = None
    ragas_wa_per_query: Optional[Dict[str, float]] = None
    hyde_responses: Optional[Dict[str, str]] = None

    if "umbrela" in components:
        umbrela_judge = judge_model_tier4 or llm_model
        u_metrics, umbrela_qrels = run_umbrela(
            queries, contexts_with_ranks, umbrela_judge, sample_ids
        )
        metrics.update(u_metrics)

    if "erag" in components:
        e_metrics, erag_per_doc = run_erag(
            queries, contexts_with_ranks, llm_model, sample_ids
        )
        metrics.update(e_metrics)

    if "ares" in components:
        paths = ares_paths or {}
        metrics.update(
            run_ares(
                queries,
                contexts_with_ranks,
                answers=answers,
                in_domain_passages_path=paths.get("passages", "data/bsard_passages.tsv"),
                human_pref_path=paths.get("human_pref", "data/ares_human_pref.tsv"),
                few_shot_path=paths.get("few_shot",     "data/ares_few_shot.tsv"),
                sample_ids=sample_ids,
            )
        )

    if "ragas_wa" in components:
        ra_metrics, ragas_wa_per_query, hyde_responses = run_ragas_workaround_a(
            queries, contexts_with_ranks, llm_model, sample_ids
        )
        metrics.update(ra_metrics)

    if "ragas_wb" in components:
        metrics.update(
            run_ragas_workaround_b(queries, contexts_with_ranks, sample_ids)
        )

    aqs = compute_aqs(metrics)
    if aqs is not None:
        metrics["T3/AQS"] = aqs

    return Tier3Result(
        metrics=metrics,
        umbrela_qrels=umbrela_qrels,
        erag_per_doc=erag_per_doc,
        ragas_wa_per_query=ragas_wa_per_query,
        hyde_responses=hyde_responses,
    )
