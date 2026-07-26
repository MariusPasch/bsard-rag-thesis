from __future__ import annotations
import numpy as np
from scipy import stats

def _paired_test(scores_a, scores_b, method: str = "ttest") -> float:
    """Two-sided paired significance test. Returns p-value."""
    a = np.array(scores_a, dtype=float)
    b = np.array(scores_b, dtype=float)
    if len(a) != len(b) or len(a) < 2:
        return 1.0
    if np.allclose(a, b):
        return 1.0
    if method == "ttest":
        _, p = stats.ttest_rel(a, b)
    return float(p)
