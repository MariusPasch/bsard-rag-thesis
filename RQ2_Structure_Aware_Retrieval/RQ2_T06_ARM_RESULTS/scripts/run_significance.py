"""Convenience runner for cross-arm paired significance tests.

Same sys.path injection as run_consolidate.py (needs T07's adapter + metrics,
which pull in the T03 retrieval stack transitively).

Usage (from this project root):
    python scripts/run_significance.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_T06 = Path(__file__).resolve().parents[1]
_RQ2 = _T06.parent
_RQ3 = _RQ2.parent / "RQ3_Autonomous_Evaluation"

for _src in sorted(_RQ2.glob("RQ2_T0*/src")):
    sys.path.insert(0, str(_src))
if _RQ3.exists():
    sys.path.insert(0, str(_RQ3))
sys.path.insert(0, str(_T06 / "src"))

from arm_results.significance import main  # noqa: E402

if __name__ == "__main__":
    main()
