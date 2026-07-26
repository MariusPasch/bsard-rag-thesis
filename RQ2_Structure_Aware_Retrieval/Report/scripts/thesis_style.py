"""Single source of truth for RQ2 figure and table styling.

Sibling of ``RQ1_Retrieval_Methods/Report/thesis_style.py``. The two modules share
the same Okabe-Ito palette, the same metric decimals, the same rcParams
profiles, and the same ``save_figure`` / ``save_table`` helpers, so that RQ1 and
RQ2 artifacts are visually consistent when placed side-by-side (STYLEGUIDE.md
§13, the cross-RQ contract).

What this module exposes
------------------------
- PALETTE                : the Okabe-Ito colors locked in STYLEGUIDE.md §2.
                          Imported from the RQ1 module if reachable, else the
                          identical hex codes are inlined (§1).
- ARMS                   : the three retrieval-arm *families* (Naive / Metadata
                          / PageIndex), each owning a color (§3).
- VARIANTS               : per-variant (display, color, marker, linestyle, arm)
                          styling for every run that may appear in a figure or
                          table. Keyed by a canonical key; each carries the
                          ``result_method`` name as it appears in the T06
                          consolidated long frame (``T03_arm1_naive`` etc.).
- DISPLAY_NAMES          : flat canonical-key -> report display name.
- METRIC_SPECS           : clean metric name -> ordered T07 metric keys
                          (first present in the long frame wins) + decimals.
- POOL_DEPTHS            : per-arm first-stage pool depth (the §5 caveat).
- rcparams_report()      : report-profile matplotlib rcParams (serif 10pt).
- rcparams_slides()      : slide-profile matplotlib rcParams (sans 22pt-eq).
- save_figure(fig, name) : writes Report/figures/<name>.{pdf,png}.
- save_table(content,name): writes Report/tables/<name>.tex.
- load_long_dataframe()  : reads the canonical T06 ``cross_arm_long.csv``
                          (the persisted output of
                          ``arm_results.consolidate.build_long_dataframe``).

The module is the *only* place these things are defined. Figure / table scripts
import from it and never set rcParams or hard-code a color.

Reference: Report/STYLEGUIDE.md.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import matplotlib as mpl
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPORT_DIR  = Path(__file__).resolve().parent.parent  # this module now lives in Report/scripts/
FIGURES_DIR = REPORT_DIR / "figures"
TABLES_DIR  = REPORT_DIR / "tables"

# RQ2 reads from the single re-evaluation path (STYLEGUIDE §6/§11): the
# consolidated long frame T06 persists. Overridable for the other device.
REPO_ROOT = REPORT_DIR.parent
CROSS_ARM_LONG_CSV = Path(
    os.environ.get(
        "RQ2_CROSS_ARM_LONG_CSV",
        REPO_ROOT / "RQ2_T06_ARM_RESULTS" / "data" / "tables" / "cross_arm_long.csv",
    )
)
# Companion T06 artefacts consumed by specific figures/tables.
T06_DATA_DIR = CROSS_ARM_LONG_CSV.parent.parent          # .../RQ2_T06_ARM_RESULTS/data
SIGNIFICANCE_CSV = T06_DATA_DIR / "tables" / "cross_arm_significance.csv"
PER_QUERY_RECALL_CSV = T06_DATA_DIR / "tables" / "per_query_recall.csv"
FAILURE_SUMMARY_CSV = T06_DATA_DIR / "error_analysis" / "failure_summary.csv"
COSTS_JSON = T06_DATA_DIR / "tables" / "cross_arm_costs.json"

# ---------------------------------------------------------------------------
# Palette (Okabe-Ito) -- locked in STYLEGUIDE.md §2, shared verbatim with RQ1
# ---------------------------------------------------------------------------

# Single-source rule: prefer the RQ1 module's PALETTE so a colour can only ever
# be edited in one place. Falls back to the identical inline hex codes when the
# RQ1 tree is not reachable (other device / detached checkout) -- STYLEGUIDE §1.
_INLINE_PALETTE = {
    "blue":       "#0072B2",
    "sky":        "#56B4E9",
    "orange":     "#E69F00",
    "green":      "#009E73",
    "vermillion": "#D55E00",
    "purple":     "#CC79A7",
    "teal":       "#117777",
    "black":      "#000000",
    "yellow":     "#F0E442",
    "grey_dark":  "#555555",
    "grey_mid":   "#888888",
    "grey_light": "#BBBBBB",
}


def _load_rq1_palette() -> dict[str, str] | None:
    """Import PALETTE from the RQ1 thesis_style module if it is reachable."""
    candidates = [
        Path(os.environ["RQ1_REPORT_DIR"]) if "RQ1_REPORT_DIR" in os.environ else None,
        REPO_ROOT.parent / "RQ1_Retrieval_Methods" / "Report",
    ]
    for cand in candidates:
        if cand and (cand / "thesis_style.py").exists():
            import importlib.util

            spec = importlib.util.spec_from_file_location(
                "_rq1_thesis_style", cand / "thesis_style.py"
            )
            try:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)  # type: ignore[union-attr]
                pal = getattr(mod, "PALETTE", None)
                if isinstance(pal, dict) and pal:
                    return pal
            except Exception:  # noqa: BLE001 -- never let a styling import break a build
                return None
    return None


PALETTE: dict[str, str] = _load_rq1_palette() or dict(_INLINE_PALETTE)
# Guarantee the keys RQ2 relies on exist even if a future RQ1 edit drops one.
for _k, _v in _INLINE_PALETTE.items():
    PALETTE.setdefault(_k, _v)

# ---------------------------------------------------------------------------
# Arms (families) -- locked in STYLEGUIDE.md §3
# ---------------------------------------------------------------------------

ArmKey = Literal["arm1", "arm2a", "arm2b", "arm2c"]


@dataclass(frozen=True)
class ArmStyle:
    key: ArmKey
    display: str            # report display name
    display_short: str      # slide / caption shorthand
    color: str
    rq1_link: str           # the RQ1 family this colour is shared with (§3)


ARMS: dict[str, ArmStyle] = {
    "arm1": ArmStyle(
        "arm1", "Arm 1 (Naive)", "Arm 1",
        PALETTE["green"], "RQ1 Hybrid-RRF (structure-agnostic chunking)",
    ),
    "arm2a": ArmStyle(
        "arm2a", "Arm 2A (Metadata)", "Arm 2A",
        PALETTE["orange"], "RQ1 Tier-2 dense (structure signal in the embedding)",
    ),
    "arm2b": ArmStyle(
        "arm2b", "Arm 2B (PageIndex)", "Arm 2B",
        PALETTE["vermillion"], "RQ1 Tier-4.0 LLM-Judge (LLM-driven retrieval)",
    ),
    # Arm 2C -- the follow-up to 2B: a rebuilt deep tree navigated by a ReAct
    # agent (vectorless LLaMA 3.1 8B). Teal = the RQ1 agentic/ReAct family, and
    # distinct from 2B's vermillion so the two navigators read apart in a figure.
    # NOT in the T06 long frame -- its numbers come from load_arm2c (the arm2c
    # experiment runs), so it carries no result_method here.
    "arm2c": ArmStyle(
        "arm2c", "Arm 2C (Agentic deep-tree nav)", "Arm 2C",
        PALETTE["teal"], "RQ1 Tier-4.2 ReAct (agentic LLM navigation)",
    ),
}

# ---------------------------------------------------------------------------
# Variants -- the runs that actually shipped (STYLEGUIDE.md §3.1 reconciled to
# the data). The variant grid that landed is *unit x enrichment*, NOT the
# planned raw/enriched/filtered/full/terms list:
#
#   unit       in {article, node}
#   enrichment in {raw, enriched, full, summary(node only)}
#
# The query-time boost/filter stage proved INERT on this corpus, so several
# variants are byte-identical to their unfiltered base (see ``inert_dup_of``):
#   T04_enriched_node == T04_full_node == T04_raw_node
#   T04_full_article  == T04_raw_article
# The only enrichment that *moves* a number is the index-time LLM summary
# (``summary_node``), which is therefore the canonical Arm-2A headline variant
# -- NOT ``full``, as the styleguide's planning table had assumed.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VariantStyle:
    key: str                       # canonical RQ2 key
    display: str                   # report display name
    display_short: str             # slide / caption shorthand
    arm: ArmKey
    color: str
    marker: str
    linestyle: str
    result_method: str             # method name in the T06 long frame
    canonical: bool = False        # one of the 3 cross-arm headline rows
    inert_dup_of: str | None = None  # boost-inert duplicate of this canonical key


_VARIANTS: list[VariantStyle] = [
    # ---------- Arm 1 -- Naive (green) -----------------------------------
    VariantStyle(
        "arm1_naive", "Arm 1 (Naive, RRF over chunks)", "Arm 1",
        "arm1", PALETTE["green"], "D", "-",
        result_method="T03_arm1_naive", canonical=True,
    ),

    # ---------- Arm 2A -- Metadata (orange) ------------------------------
    # summary-node: the canonical Arm-2A; the only variant that beats raw.
    VariantStyle(
        "arm2a_summary_node", "Arm 2A: summary-node (canonical)", "2A: summary",
        "arm2a", PALETTE["orange"], "*", "-",
        result_method="T04_summary_node_hybrid", canonical=True,
    ),
    # raw node / raw article: the within-arm anchors (no enrichment).
    VariantStyle(
        "arm2a_raw_node", "Arm 2A: raw node", "2A: raw node",
        "arm2a", PALETTE["orange"], "s", ":",
        result_method="T04_raw_node_hybrid",
    ),
    VariantStyle(
        "arm2a_raw_article", "Arm 2A: raw article", "2A: raw art.",
        "arm2a", PALETTE["orange"], "v", ":",
        result_method="T04_raw_article_hybrid",
    ),
    # enriched / full: boost-inert duplicates kept for completeness + ablation
    # transparency (their byte-equality IS a finding). Hollow marker + dashed,
    # mirroring the RQ1 convention for inert ablations.
    VariantStyle(
        "arm2a_enriched_node", "Arm 2A: enriched node", "2A: enriched",
        "arm2a", PALETTE["orange"], "D", "--",
        result_method="T04_enriched_node_hybrid", inert_dup_of="arm2a_raw_node",
    ),
    VariantStyle(
        "arm2a_full_node", "Arm 2A: full node (enriched+boost)", "2A: full node",
        "arm2a", PALETTE["orange"], "^", "--",
        result_method="T04_full_node_hybrid", inert_dup_of="arm2a_raw_node",
    ),
    VariantStyle(
        "arm2a_full_article", "Arm 2A: full article", "2A: full art.",
        "arm2a", PALETTE["orange"], "o", "--",
        result_method="T04_full_article_hybrid", inert_dup_of="arm2a_raw_article",
    ),

    # ---------- Arm 2B -- PageIndex (vermillion) -------------------------
    VariantStyle(
        "arm2b_pageindex", "Arm 2B (PageIndex, LLM nav)", "Arm 2B",
        "arm2b", PALETTE["vermillion"], "D", "-",
        result_method="T05_pageindex", canonical=True,
    ),

    # ---------- Arm 2C -- Agentic deep-tree navigation (teal) ------------
    # NOT a T06 method: result_method is the synthetic id emitted by
    # load_arm2c.arm2c_long_rows() (= load_arm2c.ARM2C_METHOD). canonical=False
    # so CANONICAL_METHODS stays the three re-eval arms; cross-arm builders add
    # this method explicitly via load_results(..., include_arm2c=True).
    VariantStyle(
        "arm2c_agentic", "Arm 2C (Agentic deep-tree nav)", "Arm 2C",
        "arm2c", PALETTE["teal"], "o", "-",
        result_method="ARM2C_agentic",
    ),
]

VARIANTS: dict[str, VariantStyle] = {v.key: v for v in _VARIANTS}
VARIANTS_BY_METHOD: dict[str, VariantStyle] = {v.result_method: v for v in _VARIANTS}

# Method-name aliases across T06 artefacts: the headline long frame names Arm 1
# ``T03_arm1_naive``, but the per-query and significance CSVs name it
# ``T03_naive``. Register the alias so every script resolves a single variant.
METHOD_ALIASES: dict[str, str] = {"T03_naive": "T03_arm1_naive"}
for _alias, _canon in METHOD_ALIASES.items():
    if _canon in VARIANTS_BY_METHOD:
        VARIANTS_BY_METHOD[_alias] = VARIANTS_BY_METHOD[_canon]
CANONICAL_KEYS: list[str] = [v.key for v in _VARIANTS if v.canonical]
CANONICAL_METHODS: list[str] = [v.result_method for v in _VARIANTS if v.canonical]
DISPLAY_NAMES: dict[str, str] = {v.key: v.display for v in _VARIANTS}


def get_variant(key_or_method: str) -> VariantStyle | None:
    """Return the VariantStyle for either a canonical key or a result_method."""
    if key_or_method in VARIANTS:
        return VARIANTS[key_or_method]
    return VARIANTS_BY_METHOD.get(key_or_method)


def order_variants(keys: Iterable[str]) -> list[str]:
    """Sort variant keys / methods by their canonical declaration order.

    Accepts either canonical keys or result_method names; unknown entries go to
    the end in original order. Use this everywhere a set of variants is
    iterated so legend / row order is identical across every artifact.
    """
    order_map: dict[str, int] = {}
    for i, v in enumerate(_VARIANTS):
        order_map[v.key] = i
        order_map[v.result_method] = i
    for alias, canon in METHOD_ALIASES.items():
        if canon in VARIANTS_BY_METHOD:
            order_map[alias] = order_map[VARIANTS_BY_METHOD[canon].result_method]
    known = [k for k in keys if k in order_map]
    unknown = [k for k in keys if k not in order_map]
    return sorted(known, key=lambda k: order_map[k]) + unknown


# ---------------------------------------------------------------------------
# Pool depths -- the non-negotiable §5 caveat
# ---------------------------------------------------------------------------

POOL_DEPTHS: dict[str, int] = {"arm1": 200, "arm2a": 100, "arm2b": 100,
                               "arm2c": 100}

# The recall-curve cutoffs actually measured by the T07 "standard" k-preset.
# (The styleguide's illustrative {1,5,10,20,50,100} is wider than what shipped.)
RECALL_K_CURVE: list[int] = [5, 10, 20, 100]

# ---------------------------------------------------------------------------
# Metric specs -- clean name -> ordered candidate T07 keys + decimals
# ---------------------------------------------------------------------------
# "First present in the long frame wins", exactly as arm_results.tables resolves
# its HEADLINE_METRICS. ``E/`` = effective / drift-aware; ``W/`` = token-weighted
# partial-relevance (Arm-1 only); both are RQ2-specific families (STYLEGUIDE §5).


@dataclass(frozen=True)
class MetricSpec:
    name: str               # clean display name
    keys: tuple[str, ...]   # candidate T07 long-frame metric keys, in priority order
    decimals: int = 3
    family: Literal["binary", "effective", "weighted", "cost"] = "binary"


_METRICS: list[MetricSpec] = [
    MetricSpec("R@5",      ("T1/R@5", "T2/P1/Recall@5")),
    MetricSpec("R@10",     ("T1/R@10", "T2/P1/Recall@10")),
    MetricSpec("R@20",     ("T1/R@20", "T2/P1/Recall@20")),
    MetricSpec("R@100",    ("T1/R@100", "T2/P1/Recall@100")),
    MetricSpec("Hit@10",   ("T2/P1/HitRate@10",)),
    MetricSpec("MRR@10",   ("T2/P2/MRR@10",)),
    MetricSpec("MRR@100",  ("T2/P2/MRR@100", "T1/MRR@100")),
    MetricSpec("nDCG@10",  ("T2/P2/NDCG@10",)),
    MetricSpec("nDCG@100", ("T2/P2/NDCG@100",)),
    MetricSpec("MAP@10",   ("T2/P2/MAP@10",)),
    MetricSpec("P@10",     ("T2/P1/Precision@10",)),
    # Effective / drift-aware (all arms)
    MetricSpec("E/R@10",    ("E/recall@10",),  family="effective"),
    MetricSpec("E/R@100",   ("E/recall@100",), family="effective"),
    MetricSpec("E/MRR@10",  ("E/mrr@10",),     family="effective"),
    MetricSpec("E/nDCG@10", ("E/ndcg@10",),    family="effective"),
    MetricSpec("E/n_eval",  ("E/n_evaluable",), decimals=0, family="effective"),
    # Token-weighted partial relevance (Arm-1 only)
    MetricSpec("W/R@10",    ("W/recall@10",),  family="weighted"),
    MetricSpec("W/R@100",   ("W/recall@100",), family="weighted"),
    # Cost (deterministic counts; see COSTS_JSON)
    MetricSpec("LLM calls/q", ("llm_calls_per_query",), decimals=0, family="cost"),
]

METRIC_SPECS: dict[str, MetricSpec] = {m.name: m for m in _METRICS}
# Default headline metric for cross-arm comparison (STYLEGUIDE §5).
HEADLINE_METRIC = "E/R@10"
# The order the cross-arm headline table presents columns (STYLEGUIDE §9.4).
HEADLINE_COLUMN_ORDER = ["E/R@10", "R@10", "Hit@10", "E/R@100", "R@100",
                         "MRR@10", "nDCG@10"]


def resolve_metric(available: Iterable[str], name: str) -> str | None:
    """Return the first T07 key for clean ``name`` present in ``available``."""
    spec = METRIC_SPECS.get(name)
    if spec is None:
        return name if name in set(available) else None
    avail = set(available)
    return next((k for k in spec.keys if k in avail), None)


def metric_decimals(name: str) -> int:
    spec = METRIC_SPECS.get(name)
    return spec.decimals if spec else 3


# ---------------------------------------------------------------------------
# Formatting helpers (identical contract to RQ1)
# ---------------------------------------------------------------------------

def fmt_metric(name: str, value: float | None) -> str:
    """Format a metric value to its spec-defined decimal places."""
    if value is None or (isinstance(value, float) and value != value):  # NaN
        return "--"
    return f"{value:.{metric_decimals(name)}f}"


def fmt_pvalue(p: float | None) -> str:
    """3-decimal p, or '<0.001'. Never scientific in tables."""
    if p is None or (isinstance(p, float) and p != p):
        return "--"
    if p < 0.001:
        return r"$<0.001$"
    return f"{p:.3f}"


def fmt_count(n: float | int | None, total: int | None = None) -> str:
    """Integer count, optionally 'n / total'."""
    if n is None or (isinstance(n, float) and n != n):
        return "--"
    n = int(round(n))
    return f"{n} / {total}" if total is not None else str(n)


# ---------------------------------------------------------------------------
# rcParams profiles -- inherited verbatim from RQ1 (STYLEGUIDE §7)
# ---------------------------------------------------------------------------

_COMMON_RCPARAMS = {
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "axes.grid.axis":    "y",
    "grid.color":        "#DDDDDD",
    "grid.linewidth":    0.6,
    "legend.frameon":    False,
    "axes.titlesize":    "medium",
    "axes.titleweight":  "regular",
    "axes.labelpad":     4.0,
    "figure.dpi":        120,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "pdf.fonttype":      42,
    "ps.fonttype":       42,
    "errorbar.capsize":  2.0,
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
        "font.family":     "sans-serif",
        "font.sans-serif": ["Inter", "Source Sans Pro", "Arial", "DejaVu Sans"],
        "font.size":       14,
        "axes.titlesize":  14,
        "axes.labelsize":  14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "lines.linewidth": 2.2,
        "lines.markersize": 7,
    })


# Physical figure sizes (inches) per profile -- same as RQ1.
FIGSIZE = {
    "report_onecol": (3.3, 2.4),
    "report_wide":   (6.7, 3.6),
    "report_tall":   (3.3, 4.0),
    "report_square": (4.0, 4.0),
    "slides_wide":   (6.5, 3.6),
    "slides_square": (4.5, 4.5),
}

SIGNIFICANCE_MARKER = r"$\dagger$"   # appended where p < 0.05 vs the anchor


# ---------------------------------------------------------------------------
# Data IO -- read the canonical re-evaluated long frame (STYLEGUIDE §6/§11)
# ---------------------------------------------------------------------------

def load_long_dataframe(path: Path | None = None) -> pd.DataFrame:
    """Load the consolidated cross-arm long frame.

    This is the persisted output of
    ``arm_results.consolidate.build_long_dataframe`` -- the single
    re-evaluation path that makes the three arms apples-to-apples
    (STYLEGUIDE §6). Columns: stem, stem_label, method, n_method_q, n_gt_q,
    metric, value. Figure / table scripts read from here, never from a per-arm
    native CSV.
    """
    path = path or CROSS_ARM_LONG_CSV
    if not path.exists():
        raise FileNotFoundError(
            f"Consolidated long frame not found at {path}. Regenerate it with "
            "RQ2_T06_ARM_RESULTS: `python -m arm_results.consolidate --effective "
            "--weighted` (run through T03's venv)."
        )
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# save helpers -- identical contract to RQ1
# ---------------------------------------------------------------------------

def save_figure(fig, name: str, *, formats: Iterable[str] = ("png",),
                subdir: str | None = None) -> list[Path]:
    """Write ``fig`` to Report/figures/[subdir/]<name>.<fmt> for each fmt."""
    out_dir = FIGURES_DIR / subdir if subdir else FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for fmt in formats:
        p = out_dir / f"{name}.{fmt}"
        fig.savefig(p, format=fmt)
        paths.append(p)
    return paths


def _styler_cls():
    """Return the pandas Styler class across pandas 1.x-3.x, or None."""
    try:
        from pandas.io.formats.style import Styler

        return Styler
    except Exception:  # noqa: BLE001
        return None


def save_table(content, name: str, *, subdir: str | None = None) -> Path:
    """Persist a LaTeX table source (string / DataFrame / Styler)."""
    out_dir = TABLES_DIR / subdir if subdir else TABLES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.tex"

    styler_cls = _styler_cls()
    if isinstance(content, str):
        tex = content
    elif styler_cls is not None and isinstance(content, styler_cls):
        tex = content.to_latex(hrules=True, multicol_align="c")
    elif isinstance(content, pd.DataFrame):
        tex = content.to_latex(index=False, escape=False, na_rep="--")
    else:
        raise TypeError(f"Unsupported content type for save_table: {type(content)}")

    out_path.write_text(tex, encoding="utf-8")
    return out_path


__all__ = [
    "PALETTE",
    "ArmStyle", "ARMS",
    "VariantStyle", "VARIANTS", "VARIANTS_BY_METHOD",
    "CANONICAL_KEYS", "CANONICAL_METHODS", "DISPLAY_NAMES",
    "get_variant", "order_variants",
    "POOL_DEPTHS", "RECALL_K_CURVE",
    "MetricSpec", "METRIC_SPECS", "HEADLINE_METRIC", "HEADLINE_COLUMN_ORDER",
    "resolve_metric", "metric_decimals",
    "fmt_metric", "fmt_pvalue", "fmt_count",
    "rcparams_report", "rcparams_slides", "FIGSIZE", "SIGNIFICANCE_MARKER",
    "load_long_dataframe", "save_figure", "save_table",
    "REPORT_DIR", "FIGURES_DIR", "TABLES_DIR",
    "CROSS_ARM_LONG_CSV", "T06_DATA_DIR", "SIGNIFICANCE_CSV",
    "PER_QUERY_RECALL_CSV", "FAILURE_SUMMARY_CSV", "COSTS_JSON",
]
