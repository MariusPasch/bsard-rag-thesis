"""
evaluation/config.py
────────────────────
Central configuration for the RAG evaluation harness.

k-value presets and TierConfig control what gets computed at runtime.
Use the factory functions at the bottom to get pre-built configs for
common scenarios.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional

# ── k-value presets ────────────────────────────────────────────────────────────

K_PRESETS: dict[str, list[int]] = {
    # Exactly the values reported in the BSARD paper  ← required for Tier 1
    "bsard_paper": [1, 5, 10, 100],

    # Minimum viable — covers the two most decision-critical cutoffs
    "minimal": [10, 100],

    # Covers what the LLM generator actually sees as context
    "generation_window": [5, 10, 20],

    # Standard academic reporting (good default for Tier 2)
    "standard": [5, 10, 20, 100],

    # Full ablation — expensive but reveals k-sensitivity across the board
    "full": [1, 3, 5, 10, 20, 50, 100, 200, 500],
}


# ── Tier 3 component registry ──────────────────────────────────────────────────

TIER3_COMPONENTS = ["umbrela", "erag", "ares", "ragas_wa", "ragas_wb"]


# ── Main config dataclass ──────────────────────────────────────────────────────

@dataclass
class TierConfig:
    """
    Controls which evaluation tiers run and at which k values.

    Parameters
    ----------
    tiers : list of ints from {0, 1, 2, 3}
        Which tiers to execute.
          0 — Efficiency metrics (latency, throughput, timing breakdown)
          1 — BSARD paper metrics (Recall@k, MRR@100)
          2 — Full supervised IR metrics (3 panels)
          3 — Autonomous LLM-based metrics (cost warning!)
    k_preset : str
        Named k-value preset from K_PRESETS.  Ignored if custom_k is set.
    custom_k : list[int] | None
        Override the preset with an explicit list of k values.
    tier2_graded : bool
        If True, run Panel 3 metrics (RA-nWG, N-Recall4+) which require
        1–5 graded qrels.  If qrels are binary, Panel 3 is skipped even
        if this flag is True.
    include_id_based : bool
        If True, run Panel 3 ID-based metrics (IDPrecision@k, IDRecall@k)
        using binary BSARD qrels — no graded labels required.  Set False
        when article IDs are not tracked in the run dict (e.g. chunk-level
        retrieval where IDs differ from BSARD article IDs).
    tier3_sample_size : int
        Number of queries passed to LLM-based metrics (cost control).
        Use None to run on all queries.
    tier3_use_api : bool
        True → OpenAI API for LLM judge calls.
        False → local GPU via Ollama (prefix model name with "ollama/").
    tier3_model : str
        Model for eRAG and RAGAS workaround LLM calls.
        Prefix "ollama/" for local GPU routing (eRAG only).
    umbrela_judge_model : str
        Model used exclusively for UMBRELA relevance grading.  Must be a
        different model from any LLM used in Tier 4 agentic systems to
        prevent self-preference bias.  Defaults to "gpt-4o-mini".
    tier3_components : list[str]
        Subset of ["umbrela", "erag", "ares", "ragas_wa", "ragas_wb"].
        "ragas_wb" is diagnostic only — excluded from AQS by design.
    """

    tiers: List[int] = field(default_factory=lambda: [0, 1, 2])
    k_preset: Optional[str] = "standard"
    custom_k: Optional[List[int]] = None

    # Panel 3 switches
    tier2_graded: bool = False
    include_id_based: bool = True

    # Tier 3 options
    tier3_sample_size: Optional[int] = 150
    tier3_use_api: bool = True
    tier3_model: str = "gpt-4o-mini"
    umbrela_judge_model: str = "gpt-4o-mini"
    tier3_components: List[str] = field(
        default_factory=lambda: list(TIER3_COMPONENTS)
    )

    def __post_init__(self) -> None:
        invalid_tiers = set(self.tiers) - {0, 1, 2, 3}
        if invalid_tiers:
            raise ValueError(
                f"Invalid tiers: {invalid_tiers}. Must be a subset of {{0, 1, 2, 3}}."
            )
        if self.k_preset and self.k_preset not in K_PRESETS:
            raise ValueError(
                f"Unknown k_preset '{self.k_preset}'. "
                f"Valid options: {list(K_PRESETS.keys())}"
            )
        invalid_t3 = set(self.tier3_components) - set(TIER3_COMPONENTS)
        if invalid_t3:
            raise ValueError(
                f"Unknown Tier 3 components: {invalid_t3}. "
                f"Valid: {TIER3_COMPONENTS}"
            )

    @property
    def k_values(self) -> List[int]:
        """Resolved, sorted, deduplicated list of k values to evaluate at."""
        if self.custom_k:
            return sorted(set(self.custom_k))
        return K_PRESETS.get(self.k_preset, K_PRESETS["standard"])

    @property
    def run_tier0(self) -> bool:
        return 0 in self.tiers

    @property
    def run_tier1(self) -> bool:
        return 1 in self.tiers

    @property
    def run_tier2(self) -> bool:
        return 2 in self.tiers

    @property
    def run_tier3(self) -> bool:
        return 3 in self.tiers

    def summary(self) -> str:
        lines = [
            f"Tiers:      {self.tiers}",
            f"k values:   {self.k_values}",
        ]
        if self.run_tier0:
            lines.append("Tier 0:     Efficiency metrics (latency, throughput)")
        if self.run_tier2:
            lines.append(f"Graded P3:  {self.tier2_graded}")
            lines.append(f"ID-based P3:{self.include_id_based}")
        if self.run_tier3:
            lines.append(
                f"Tier 3:     {self.tier3_components} "
                f"(n={self.tier3_sample_size}, "
                f"{'API' if self.tier3_use_api else 'local GPU'}, "
                f"model={self.tier3_model}, "
                f"umbrela_judge={self.umbrela_judge_model})"
            )
        return "\n".join(lines)


# ── Factory functions ──────────────────────────────────────────────────────────

def bsard_benchmark() -> TierConfig:
    """
    Exact BSARD paper replication.

    Tier 0 + Tier 1 only, fixed k ∈ {1, 5, 10, 100} plus MRR@100.
    Use this when comparing against the CamemBERT bi-encoder baseline.
    """
    return TierConfig(tiers=[0, 1], k_preset="bsard_paper", include_id_based=True)


def supervised_standard() -> TierConfig:
    """
    Tiers 0 + 1 + 2, standard k values.  No LLM calls.
    Good default for day-to-day retrieval experiments.
    """
    return TierConfig(tiers=[0, 1, 2], k_preset="standard", include_id_based=True)


def supervised_full() -> TierConfig:
    """
    Tiers 0 + 1 + 2 with full k sweep, graded Panel 3 metrics, and
    ID-based Panel 3 metrics.  Graded Panel 3 requires 1–5 qrels.
    """
    return TierConfig(
        tiers=[0, 1, 2],
        k_preset="full",
        tier2_graded=True,
        include_id_based=True,
    )


def fast_debug() -> TierConfig:
    """Minimal k, Tier 0+1+2 only — for rapid iteration during development."""
    return TierConfig(tiers=[0, 1, 2], k_preset="minimal")


def efficiency_only() -> TierConfig:
    """Tier 0 only — just latency and throughput analysis, no quality metrics."""
    return TierConfig(tiers=[0], k_preset="minimal")


def full_evaluation() -> TierConfig:
    """
    All four tiers, full k sweep.  Requires GPU or OpenAI API for Tier 3.

    Tier 3 runs umbrela, erag, ares, ragas_wa (production set).
    ragas_wb is excluded — add it manually for diagnostic runs.
    UMBRELA judge defaults to gpt-4o-mini; override umbrela_judge_model
    when evaluating Tier 4 systems to enforce cross-model discipline.
    """
    return TierConfig(
        tiers=[0, 1, 2, 3],
        k_preset="full",
        tier2_graded=True,
        include_id_based=True,
        tier3_components=["umbrela", "erag", "ares", "ragas_wa"],
    )


def generation_window_eval() -> TierConfig:
    """
    Focus on what the LLM generator actually sees: k ∈ {5, 10, 20}.
    Tier 0 + 1 + 2, no LLM calls.
    """
    return TierConfig(tiers=[0, 1, 2], k_preset="generation_window")
