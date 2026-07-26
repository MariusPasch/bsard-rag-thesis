"""
Single source of truth for thesis-report figure and table styling.

What this module exposes
------------------------
- PALETTE                : the Okabe-Ito colors locked in STYLEGUIDE.md section 2
- SYSTEMS                : per-system (display_name, color, marker, linestyle, family)
                          mapping for every retrieval system that may appear in a
                          figure or table; the nine canonical systems carry full
                          styling, tier-internal ablations inherit family color
                          with hollow markers + dotted lines.
- EVALUATORS             : RQ3 evaluator -> (display_name, color, marker) map.
                          Kept here so the RQ3 chapter consumes the same module.
- DISPLAY_NAMES          : flat dict of canonical key -> report display name
- METRIC_FORMATS         : decimal places per metric
- rcparams_report()      : applies report-profile matplotlib rcParams
- rcparams_slides()      : applies slide-profile matplotlib rcParams
- save_figure(fig, name) : writes Report/figures/<name>.{pdf,png}
- save_table(df_or_styler, name, **kwargs) : writes Report/tables/<name>.tex

The module is the *only* place these things are defined. Figure / table scripts
import from it and never set rcParams or hard-code a color.

Reference: Report/STYLEGUIDE.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPORT_DIR  = Path(__file__).resolve().parent
FIGURES_DIR = REPORT_DIR / "figures"
TABLES_DIR  = REPORT_DIR / "tables"
DATA_DIR    = REPORT_DIR / "data"

# ---------------------------------------------------------------------------
# Palette (Okabe-Ito) -- locked in STYLEGUIDE.md section 2
# ---------------------------------------------------------------------------

PALETTE = {
    "blue":          "#0072B2",  # Tier 1 anchor
    "sky":           "#56B4E9",  # Tier 1 variants
    "orange":        "#E69F00",  # Tier 2 (and RQ3 UMBRELA)
    "green":         "#009E73",  # Tier 3 (and RQ3 eRAG)
    "vermillion":    "#D55E00",  # Tier 4.0 LLM-Judge
    "purple":        "#CC79A7",  # Tier 4.1 CRAG (and RQ3 RAGAS-WA)
    "teal":          "#117777",  # Tier 4.2 ReAct
    "black":         "#000000",  # RQ3 AQS (composite)
    "yellow":        "#F0E442",  # reserve -- strata highlights only
    "grey_dark":     "#555555",  # ablations / gold-ceiling reference
    "grey_mid":      "#888888",  # ablations
    "grey_light":    "#BBBBBB",  # backgrounds / non-focal series
}

# ---------------------------------------------------------------------------
# System styling -- locked in STYLEGUIDE.md sections 3 and 5
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SystemStyle:
    key: str
    display: str
    display_short: str
    color: str
    marker: str
    linestyle: str
    family: Literal["sparse", "dense", "hybrid", "llm_judge", "agentic_crag",
                    "agentic_react"]
    canonical: bool = False           # one of the nine cross-tier-headline systems
    result_id: str | None = None      # canonical experiment_id in result JSONs


# Order matters for sorting / legend ordering: top-down by tier.
_SYSTEMS: list[SystemStyle] = [
    # ---------- Tier 1 -- sparse (blue family) ----------------------------
    SystemStyle(
        key="bm25_anchor",
        display=r"BM25 anchor (Okapi, $k_1{=}1.5,\,b{=}0.75$)",
        display_short="BM25 (anchor)",
        color=PALETTE["blue"], marker="o", linestyle="--",
        family="sparse", canonical=True,
        result_id="bm25_anchor_test",
    ),
    SystemStyle(
        key="bm25_tuned",
        display=r"BM25 tuned ($b{=}0.25$, lemmatized)",
        display_short="BM25 (tuned)",
        color=PALETTE["sky"], marker="D", linestyle="-",
        family="sparse", canonical=True,
        result_id="bm25_tuned_k11.5_b0.25_test",
    ),
    SystemStyle(
        key="bm25_concat2x",
        display=r"BM25 + concat-2x field weighting (lemmatized)",
        display_short="BM25 (concat-2x)",
        color=PALETTE["sky"], marker="s", linestyle=":",
        family="sparse", canonical=True,
        result_id="bm25_lemmatize_concat_2x_test",
    ),
    SystemStyle(
        key="bm25_lemmatize_text_only",
        display=r"BM25 (Okapi, lemmatized, text only)",
        display_short="BM25 (lemmatize)",
        color=PALETTE["sky"], marker="^", linestyle=":",
        family="sparse",
        result_id="bm25_lemmatize_text_only_test",
    ),
    SystemStyle(
        key="bm25_stem_text_only",
        display=r"BM25 (Okapi, stemmed, text only)",
        display_short="BM25 (stem)",
        color=PALETTE["sky"], marker="v", linestyle=":",
        family="sparse",
        result_id="bm25_stem_text_only_test",
    ),
    SystemStyle(
        key="bm25_none_concat2x",
        display=r"BM25 (Okapi, none, concat-2x)",
        display_short="BM25 (none, concat-2x)",
        color=PALETTE["sky"], marker="P", linestyle=":",
        family="sparse",
        result_id="bm25_none_concat_2x_test",
    ),
    SystemStyle(
        key="bm25_plus_lemmatize",
        display=r"BM25+ (lemmatized)",
        display_short="BM25+",
        color=PALETTE["sky"], marker="X", linestyle=":",
        family="sparse",
        result_id="bm25_plus_lemmatize_test",
    ),
    SystemStyle(
        key="bm25_l_lemmatize",
        display=r"BM25L (lemmatized)",
        display_short="BM25L",
        color=PALETTE["sky"], marker="*", linestyle=":",
        family="sparse",
        result_id="bm25_l_lemmatize_test",
    ),
    SystemStyle(
        key="fts5_default",
        display=r"SQLite FTS5 (default)",
        display_short="FTS5",
        color=PALETTE["grey_mid"], marker="h", linestyle=":",
        family="sparse",
        result_id="fts5_default_test",
    ),
    SystemStyle(
        key="tfidf_lemmatize_text_only",
        display=r"TF-IDF (lemmatized, text only)",
        display_short="TF-IDF (lemmatize)",
        color=PALETTE["grey_dark"], marker="o", linestyle=":",
        family="sparse",
        result_id="tfidf_lemmatize_text_only_test",
    ),
    SystemStyle(
        key="tfidf_none_text_only",
        display=r"TF-IDF (none, text only)",
        display_short="TF-IDF (none)",
        color=PALETTE["grey_dark"], marker="s", linestyle=":",
        family="sparse",
        result_id="tfidf_none_text_only_test",
    ),
    SystemStyle(
        key="tfidf_lemmatize_concat2x",
        display=r"TF-IDF (lemmatized, concat-2x)",
        display_short="TF-IDF (concat-2x)",
        color=PALETTE["grey_dark"], marker="D", linestyle=":",
        family="sparse",
        result_id="tfidf_lemmatize_concat_2x_test",
    ),

    # ---------- Tier 2 -- dense (orange family) ---------------------------
    # The mE5 family shares marker D and varies linestyle (canonical / dashed /
    # dotted). Qwen3 shares marker P. Single-family models use unique markers.
    # The paper-replication anchor (CamemBERT-base) is rendered in grey to
    # signal "baseline expected to collapse" rather than "competing dense
    # encoder" -- it reaches R@10 = 0.008 on this corpus.
    SystemStyle(
        key="dense_me5_large",
        display=r"mE5-large, zero-shot, concat-2x",
        display_short="mE5-large (concat-2x)",
        color=PALETTE["orange"], marker="D", linestyle="-",
        family="dense", canonical=True,
        result_id="dense_me5_large_concat2x_zeroshot_test",
    ),
    SystemStyle(
        key="dense_me5_large_plain",
        display=r"mE5-large, zero-shot",
        display_short="mE5-large",
        color=PALETTE["orange"], marker="D", linestyle="--",
        family="dense",
        result_id="dense_me5_large_zeroshot_test",
    ),
    SystemStyle(
        key="dense_me5_base",
        display=r"mE5-base, zero-shot",
        display_short="mE5-base",
        color=PALETTE["orange"], marker="D", linestyle=":",
        family="dense",
        result_id="dense_me5_base_zeroshot_test",
    ),
    SystemStyle(
        key="dense_qwen3_instruct",
        display=r"Qwen3-Embedding 0.6B (instruction-prefixed)",
        display_short="Qwen3 (instr.)",
        color=PALETTE["orange"], marker="P", linestyle="--",
        family="dense",
        result_id="dense_qwen3_0_6b_instruct_zeroshot_test",
    ),
    SystemStyle(
        key="dense_qwen3_plain",
        display=r"Qwen3-Embedding 0.6B",
        display_short="Qwen3",
        color=PALETTE["orange"], marker="P", linestyle=":",
        family="dense",
        result_id="dense_qwen3_0_6b_zeroshot_test",
    ),
    SystemStyle(
        key="dense_bge_m3",
        display=r"BGE-M3 (long context, CLS pooling)",
        display_short="BGE-M3",
        color=PALETTE["orange"], marker="h", linestyle=":",
        family="dense",
        result_id="dense_bge_m3_zeroshot_test",
    ),
    SystemStyle(
        key="dense_camembert_lg",
        display=r"sentence-camembert-large (French zero-shot)",
        display_short="CamemBERT-lg",
        color=PALETTE["orange"], marker="^", linestyle=":",
        family="dense",
        result_id="dense_camembert_lg_zeroshot_test",
    ),
    SystemStyle(
        key="dense_mpnet_multi",
        display=r"paraphrase-multilingual-mpnet-v2 (128-tok reference)",
        display_short="mpnet-multi",
        color=PALETTE["orange"], marker="X", linestyle=":",
        family="dense",
        result_id="dense_mpnet_multi_zeroshot_test",
    ),
    SystemStyle(
        key="dense_camembert_base",
        display=r"CamemBERT-base (paper replication anchor)",
        display_short="CamemBERT-base",
        color=PALETTE["grey_dark"], marker="o", linestyle="--",
        family="dense",
        result_id="dense_camembert_base_zeroshot_test",
    ),
    # ----------------------------------------------------------------------

    # ---------- Tier 3 -- hybrid (green family) ---------------------------
    # Three sub-families share one base hue (Okabe-Ito green) and vary marker
    # to separate visually. Within a sub-family, parameter variants share a
    # marker and vary linestyle (canonical = solid, others = dashed / dotted).
    # Rows are declared in parameter order (not canonical-first) so tables and
    # legends read monotonically; the canonical winner of each family is
    # marked instead via SystemStyle.canonical=True (RRF k60 only is also the
    # cross-tier canonical).
    # RRF sub-family -- 3 entries, k = 30 / 60 / 120. k=60 is canonical.
    SystemStyle(
        key="hybrid_rrf_k30",
        display=r"Hybrid RRF ($k{=}30$)",
        display_short=r"RRF ($k{=}30$)",
        color=PALETTE["green"], marker="D", linestyle="--",
        family="hybrid",
        result_id="hybrid_rrf_k30_test",
    ),
    SystemStyle(
        key="hybrid_rrf_k60",
        display=r"Hybrid RRF ($k{=}60$)",
        display_short=r"RRF ($k{=}60$)",
        color=PALETTE["green"], marker="D", linestyle="-",
        family="hybrid", canonical=True,
        result_id="hybrid_rrf_k60_test",
    ),
    SystemStyle(
        key="hybrid_rrf_k120",
        display=r"Hybrid RRF ($k{=}120$)",
        display_short=r"RRF ($k{=}120$)",
        color=PALETTE["green"], marker="D", linestyle=":",
        family="hybrid",
        result_id="hybrid_rrf_k120_test",
    ),
    # Linear-alpha sweep -- 9 entries, alpha = 0.1 ... 0.9. alpha=0.9 is the
    # family canonical (highest R@10). All share marker o; canonical = solid.
    SystemStyle(
        key="hybrid_linear_alpha_0.1",
        display=r"Hybrid linear ($\alpha{=}0.1$)",
        display_short=r"Linear ($\alpha{=}0.1$)",
        color=PALETTE["green"], marker="o", linestyle=":",
        family="hybrid",
        result_id="hybrid_linear_alpha_0.1_test",
    ),
    SystemStyle(
        key="hybrid_linear_alpha_0.2",
        display=r"Hybrid linear ($\alpha{=}0.2$)",
        display_short=r"Linear ($\alpha{=}0.2$)",
        color=PALETTE["green"], marker="o", linestyle=":",
        family="hybrid",
        result_id="hybrid_linear_alpha_0.2_test",
    ),
    SystemStyle(
        key="hybrid_linear_alpha_0.3",
        display=r"Hybrid linear ($\alpha{=}0.3$)",
        display_short=r"Linear ($\alpha{=}0.3$)",
        color=PALETTE["green"], marker="o", linestyle=":",
        family="hybrid",
        result_id="hybrid_linear_alpha_0.3_test",
    ),
    SystemStyle(
        key="hybrid_linear_alpha_0.4",
        display=r"Hybrid linear ($\alpha{=}0.4$)",
        display_short=r"Linear ($\alpha{=}0.4$)",
        color=PALETTE["green"], marker="o", linestyle=":",
        family="hybrid",
        result_id="hybrid_linear_alpha_0.4_test",
    ),
    SystemStyle(
        key="hybrid_linear_alpha_0.5",
        display=r"Hybrid linear ($\alpha{=}0.5$)",
        display_short=r"Linear ($\alpha{=}0.5$)",
        color=PALETTE["green"], marker="o", linestyle=":",
        family="hybrid",
        result_id="hybrid_linear_alpha_0.5_test",
    ),
    SystemStyle(
        key="hybrid_linear_alpha_0.6",
        display=r"Hybrid linear ($\alpha{=}0.6$)",
        display_short=r"Linear ($\alpha{=}0.6$)",
        color=PALETTE["green"], marker="o", linestyle=":",
        family="hybrid",
        result_id="hybrid_linear_alpha_0.6_test",
    ),
    SystemStyle(
        key="hybrid_linear_alpha_0.7",
        display=r"Hybrid linear ($\alpha{=}0.7$)",
        display_short=r"Linear ($\alpha{=}0.7$)",
        color=PALETTE["green"], marker="o", linestyle=":",
        family="hybrid",
        result_id="hybrid_linear_alpha_0.7_test",
    ),
    SystemStyle(
        key="hybrid_linear_alpha_0.8",
        display=r"Hybrid linear ($\alpha{=}0.8$)",
        display_short=r"Linear ($\alpha{=}0.8$)",
        color=PALETTE["green"], marker="o", linestyle=":",
        family="hybrid",
        result_id="hybrid_linear_alpha_0.8_test",
    ),
    SystemStyle(
        key="hybrid_linear_alpha_0.9",
        display=r"Hybrid linear ($\alpha{=}0.9$)",
        display_short=r"Linear ($\alpha{=}0.9$)",
        color=PALETTE["green"], marker="o", linestyle="-",
        family="hybrid",
        result_id="hybrid_linear_alpha_0.9_test",
    ),
    # SGDR sub-family -- 3 entries, K = 1000 / 2000 / 5000. K=1000 is canonical.
    SystemStyle(
        key="hybrid_sgdr_k1000",
        display=r"Hybrid SGDR ($K{=}1000$)",
        display_short=r"SGDR ($K{=}1000$)",
        color=PALETTE["green"], marker="s", linestyle="-",
        family="hybrid",
        result_id="hybrid_sgdr_k1000_test",
    ),
    SystemStyle(
        key="hybrid_sgdr_k2000",
        display=r"Hybrid SGDR ($K{=}2000$)",
        display_short=r"SGDR ($K{=}2000$)",
        color=PALETTE["green"], marker="s", linestyle="--",
        family="hybrid",
        result_id="hybrid_sgdr_k2000_test",
    ),
    SystemStyle(
        key="hybrid_sgdr_k5000",
        display=r"Hybrid SGDR ($K{=}5000$)",
        display_short=r"SGDR ($K{=}5000$)",
        color=PALETTE["green"], marker="s", linestyle=":",
        family="hybrid",
        result_id="hybrid_sgdr_k5000_test",
    ),

    # ---------- Tier 4.0 -- LLM-Judge (vermillion family) -----------------
    # Four canonical T4.0 runs span three orthogonal ablation axes:
    # first-stage (hybrid vs BM25), pool size (top-50 vs top-20), and scoring
    # paradigm (binary vs 0-10 numeric). The hybrid-first-stage variants are
    # the cross-tier headline anchors; the BM25 variants are first-stage and
    # scoring-paradigm ablations.
    SystemStyle(
        key="llm_rerank_top50",
        display=r"LLM-Judge (hybrid, binary, top-50)",
        display_short=r"LLM-Judge top-50 (hyb.)",
        color=PALETTE["vermillion"], marker="D", linestyle="-",
        family="llm_judge", canonical=True,
        result_id="llm_rerank_binary_top50_hybrid_rrf_k60_test",
    ),
    SystemStyle(
        key="llm_rerank_top20",
        display=r"LLM-Judge (hybrid, binary, top-20)",
        display_short=r"LLM-Judge top-20 (hyb.)",
        color=PALETTE["vermillion"], marker="s", linestyle="-.",
        family="llm_judge", canonical=True,
        result_id="llm_rerank_binary_top20_hybrid_rrf_k60_test",
    ),
    SystemStyle(
        key="llm_rerank_top50_bm25",
        display=r"LLM-Judge (BM25, binary, top-50)",
        display_short=r"LLM-Judge top-50 (BM25)",
        color=PALETTE["vermillion"], marker="D", linestyle=":",
        family="llm_judge",
        result_id="llm_rerank_binary_top50_test",
    ),
    SystemStyle(
        key="llm_rerank_0to10_top50_bm25",
        display=r"LLM-Judge (BM25, numeric 0--10, top-50)",
        display_short=r"LLM-Judge 0--10 (BM25)",
        color=PALETTE["vermillion"], marker="^", linestyle=":",
        family="llm_judge",
        result_id="llm_rerank_0to10_top50_test",
    ),

    # ---------- Tier 4.1 -- CRAG (purple family) --------------------------
    # Two canonical T4.1 runs (per RETRIEVAL_PROJECT.md sec 5.3), both at the
    # v2 design point (eval_k=20, backbone_top_k=100). The hybrid first stage
    # is the cross-tier canonical; the BM25 first stage is a first-stage
    # ablation. CRAG's output pool is 100 via first-stage fallback, so no
    # recall-curve truncation is needed (unlike T4.0 rerankers).
    SystemStyle(
        key="crag_hybrid_v2",
        display=r"CRAG (hybrid, v2)",
        display_short=r"CRAG (hyb.)",
        color=PALETTE["purple"], marker="D", linestyle="-",
        family="agentic_crag", canonical=True,
        result_id="crag_hybrid_rrf_k60_test_v2",
    ),
    SystemStyle(
        key="crag_bm25_v2",
        display=r"CRAG (BM25, v2)",
        display_short=r"CRAG (BM25)",
        color=PALETTE["purple"], marker="D", linestyle=":",
        family="agentic_crag",
        result_id="crag_bm25_test_v2",
    ),

    # ---------- Tier 4.2 -- ReAct (teal) ----------------------------------
    # Three canonical T4.2 runs (per RETRIEVAL_PROJECT.md sec 5.3). The v1
    # design (max_steps=5, top_k_shown=10, no gap prompt) under-performed and
    # is retained as a design-evolution reference; v2 (max_steps=8, gap
    # prompt, function-calling, fewshot trajectories) lifts R@10 from 0.277
    # to 0.426. All three plateau by R@50 (effective output pool ~20-50);
    # the per-tier figures treat pool = 20 to match the T4.0 / T4.1 anchor.
    SystemStyle(
        key="react_hybrid_v2",
        display=r"ReAct (hybrid, v2)",
        display_short=r"ReAct (hyb.)",
        color=PALETTE["teal"], marker="D", linestyle="-",
        family="agentic_react", canonical=True,
        result_id="react_hybrid_rrf_k60_test_v2",
    ),
    SystemStyle(
        key="react_hybrid_v1",
        display=r"ReAct (hybrid, v1)",
        display_short=r"ReAct v1 (hyb.)",
        color=PALETTE["teal"], marker="D", linestyle="--",
        family="agentic_react",
        result_id="react_hybrid_rrf_k60_test",
    ),
    SystemStyle(
        key="react_bm25_v1",
        display=r"ReAct (BM25, v1)",
        display_short=r"ReAct v1 (BM25)",
        color=PALETTE["teal"], marker="D", linestyle=":",
        family="agentic_react",
        result_id="react_bm25_test",
    ),
]

SYSTEMS: dict[str, SystemStyle] = {s.key: s for s in _SYSTEMS}
SYSTEMS_BY_RESULT_ID: dict[str, SystemStyle] = {
    s.result_id: s for s in _SYSTEMS if s.result_id is not None
}
CANONICAL_KEYS: list[str] = [s.key for s in _SYSTEMS if s.canonical]
DISPLAY_NAMES: dict[str, str] = {s.key: s.display for s in _SYSTEMS}


def get_system(key_or_result_id: str) -> SystemStyle | None:
    """Return the SystemStyle for either a canonical key or a result_id."""
    if key_or_result_id in SYSTEMS:
        return SYSTEMS[key_or_result_id]
    return SYSTEMS_BY_RESULT_ID.get(key_or_result_id)


# ---------------------------------------------------------------------------
# RQ3 evaluator styling (Chapter 14 only) ------------------------------------
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EvaluatorStyle:
    key: str
    display: str
    color: str
    marker: str


EVALUATORS: dict[str, EvaluatorStyle] = {
    "umbrela": EvaluatorStyle("umbrela", r"UMBRELA (mean grade / 3)",
                              PALETTE["orange"], "o"),
    "erag":    EvaluatorStyle("erag",    r"eRAG (mean)",
                              PALETTE["green"],  "s"),
    "ragas_wa":EvaluatorStyle("ragas_wa", r"RAGAS-WA (mean)",
                              PALETTE["purple"], "D"),
    "aqs":     EvaluatorStyle("aqs",     r"AQS (composite)",
                              PALETTE["black"],  "*"),
    "gold_ceiling": EvaluatorStyle("gold_ceiling", r"Gold ceiling (oracle)",
                                   PALETTE["grey_dark"], ""),
}

# ---------------------------------------------------------------------------
# Metric formatting ----------------------------------------------------------
# ---------------------------------------------------------------------------

# decimal places per metric (STYLEGUIDE section 6)
METRIC_FORMATS: dict[str, int] = {
    "Recall@1":    3,
    "Recall@5":    3,
    "Recall@10":   3,
    "Recall@20":   3,
    "Recall@50":   3,
    "Recall@100":  3,
    "Hit@10":      3,
    "MRR@10":      3,
    "MRR@100":     3,
    "NDCG@10":     3,
    "MAP@10":      3,
    "UMBRELA":     3,
    "eRAG":        3,
    "RAGAS-WA":    3,
    "AQS":         3,
    "kendall_tau": 3,
    "spearman_rho":3,
}


def fmt_metric(metric: str, value: float) -> str:
    """Format a metric value to the spec-defined decimal places."""
    if value is None or (isinstance(value, float) and value != value):  # NaN
        return "--"
    dp = METRIC_FORMATS.get(metric, 3)
    return f"{value:.{dp}f}"


def fmt_pvalue(p: float | None) -> str:
    """3-decimal p, or '<0.001'. Never scientific in tables."""
    if p is None or (isinstance(p, float) and p != p):
        return "--"
    if p < 0.001:
        return r"$<0.001$"
    return f"{p:.3f}"


def fmt_count(n: float | int | None, total: int | None = None) -> str:
    """Integer count, optionally "n / total"."""
    if n is None or (isinstance(n, float) and n != n):
        return "--"
    n = int(round(n))
    return f"{n} / {total}" if total is not None else str(n)


# ---------------------------------------------------------------------------
# rcParams profiles ----------------------------------------------------------
# ---------------------------------------------------------------------------

_COMMON_RCPARAMS = {
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "axes.grid":           True,
    "axes.grid.axis":      "y",
    "grid.color":          "#DDDDDD",
    "grid.linewidth":      0.6,
    "legend.frameon":      False,
    "axes.titlesize":      "medium",
    "axes.titleweight":    "regular",
    "axes.labelpad":       4.0,
    "figure.dpi":          120,
    "savefig.dpi":         300,
    "savefig.bbox":        "tight",
    "pdf.fonttype":        42,    # TrueType-embedded, selectable in LaTeX viewers
    "ps.fonttype":         42,
    "errorbar.capsize":    2.0,
}


def rcparams_report() -> None:
    """Report profile: serif 10pt, single-column physical sizing."""
    mpl.rcParams.update(_COMMON_RCPARAMS)
    mpl.rcParams.update({
        "font.family":      "serif",
        "font.serif":       ["Computer Modern Roman", "DejaVu Serif", "Times New Roman"],
        "mathtext.fontset": "cm",
        "font.size":        10,
        "axes.titlesize":   10,
        "axes.labelsize":   10,
        "xtick.labelsize":  9,
        "ytick.labelsize":  9,
        "legend.fontsize":  9,
        "lines.linewidth":  1.5,
        "lines.markersize": 5,
    })


def rcparams_slides() -> None:
    """Slide profile: sans-serif equivalent of ~22pt body, 16:9 sizing."""
    mpl.rcParams.update(_COMMON_RCPARAMS)
    mpl.rcParams.update({
        "font.family":      "sans-serif",
        "font.sans-serif":  ["Inter", "Source Sans Pro", "Arial", "DejaVu Sans"],
        "font.size":        14,
        "axes.titlesize":   14,
        "axes.labelsize":   14,
        "xtick.labelsize":  12,
        "ytick.labelsize":  12,
        "legend.fontsize":  12,
        "lines.linewidth":  2.2,
        "lines.markersize": 7,
    })


# Convenience: physical figure sizes (inches) per profile.
FIGSIZE = {
    "report_onecol":  (3.3, 2.4),
    "report_wide":    (6.7, 3.6),
    "report_tall":    (3.3, 4.0),
    "report_square":  (4.0, 4.0),
    "slides_wide":    (6.5, 3.6),     # 16:9 at 1280x720 -> 6.5 x 3.6 in @ ~200dpi
    "slides_square":  (4.5, 4.5),
}


# ---------------------------------------------------------------------------
# save helpers ---------------------------------------------------------------
# ---------------------------------------------------------------------------

def save_figure(fig, name: str, *, formats: Iterable[str] = ("pdf", "png"),
                subdir: str | None = None) -> list[Path]:
    """
    Write `fig` to Report/figures/[subdir/]<name>.<fmt> for each requested fmt.

    The name should NOT include an extension; the helper adds it. The PDF is
    vector (for LaTeX includegraphics); the PNG is rasterised at the dpi set
    in rcParams (300 by default).

    Returns the list of paths written, in declaration order.
    """
    out_dir = FIGURES_DIR / subdir if subdir else FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for fmt in formats:
        p = out_dir / f"{name}.{fmt}"
        fig.savefig(p, format=fmt)
        paths.append(p)
    return paths


def save_table(content: str | pd.io.formats.style.Styler | pd.DataFrame,
               name: str, *, subdir: str | None = None) -> Path:
    """
    Persist a LaTeX table source.

    `content` may be:
      - a finished LaTeX string  -> written verbatim
      - a pandas DataFrame       -> rendered via .to_latex(...) with sensible
                                    booktabs defaults (the caller should
                                    really pre-render to a Styler instead)
      - a pandas Styler          -> rendered via Styler.to_latex(...)

    Returns the .tex path written.
    """
    out_dir = TABLES_DIR / subdir if subdir else TABLES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.tex"

    if isinstance(content, str):
        tex = content
    elif isinstance(content, pd.io.formats.style.Styler):
        tex = content.to_latex(hrules=True, multicol_align="c")
    elif isinstance(content, pd.DataFrame):
        tex = content.to_latex(index=False, escape=False, na_rep="--",
                                column_format=None)
    else:
        raise TypeError(f"Unsupported content type for save_table: {type(content)}")

    out_path.write_text(tex, encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# Convenience -- legend and significance markers -----------------------------
# ---------------------------------------------------------------------------

SIGNIFICANCE_MARKER = r"$\dagger$"   # appended to a metric value where p<0.05


def order_systems(keys: Iterable[str]) -> list[str]:
    """
    Return the given system keys sorted by their canonical declaration order.

    Keys not in SYSTEMS go to the end in original order. Use this any time you
    iterate over a set of systems for a figure or table -- it guarantees the
    legend / row order is the same across every artifact in the thesis.
    """
    order_map = {key: i for i, key in enumerate(SYSTEMS)}
    known = [k for k in keys if k in order_map]
    unknown = [k for k in keys if k not in order_map]
    return sorted(known, key=lambda k: order_map[k]) + unknown


__all__ = [
    "PALETTE",
    "SystemStyle", "SYSTEMS", "SYSTEMS_BY_RESULT_ID", "CANONICAL_KEYS",
    "DISPLAY_NAMES", "get_system", "order_systems",
    "EvaluatorStyle", "EVALUATORS",
    "METRIC_FORMATS", "fmt_metric", "fmt_pvalue", "fmt_count",
    "rcparams_report", "rcparams_slides", "FIGSIZE",
    "save_figure", "save_table",
    "SIGNIFICANCE_MARKER",
    "REPORT_DIR", "FIGURES_DIR", "TABLES_DIR", "DATA_DIR",
]
