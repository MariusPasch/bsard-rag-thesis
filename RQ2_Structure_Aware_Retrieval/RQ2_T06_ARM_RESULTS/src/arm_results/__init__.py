"""Cross-arm results consolidation: tables, figures, error analyses, and slide
assets aggregating T03 / T04 / T05 retrieval outputs and T07 metrics.

Presentation layer only — metric computation is delegated to T07's
``evaluation`` package; this project loads persisted rankings, invokes T07
through one consistent path, and reshapes the results. No retrieval is run.
"""

# Loaders are dependency-light (stdlib only) and always available.
from .loaders import (
    load_ground_truth,
    load_t03,
    load_t04_variants,
    load_t05,
    load_all_arms,
)

__all__ = [
    "load_ground_truth",
    "load_t03",
    "load_t04_variants",
    "load_t05",
    "load_all_arms",
]

# The rest need pandas / T07 / matplotlib; expose them when those are installed
# so a bare `import arm_results` still works in a minimal environment.
try:  # noqa: SIM105
    from .consolidate import evaluate_stem, build_long_dataframe
    from .tables import emit_all, emit_headline, emit_per_stem, emit_full, emit_extended
    from .errors import export_per_query, categorise_failures, export_all
    from .significance import build_recall_matrix, paired_tests, run as run_significance

    __all__ += [
        "evaluate_stem",
        "build_long_dataframe",
        "emit_all",
        "emit_headline",
        "emit_per_stem",
        "emit_full",
        "emit_extended",
        "export_per_query",
        "categorise_failures",
        "export_all",
        "build_recall_matrix",
        "paired_tests",
        "run_significance",
    ]
except ImportError:
    pass
