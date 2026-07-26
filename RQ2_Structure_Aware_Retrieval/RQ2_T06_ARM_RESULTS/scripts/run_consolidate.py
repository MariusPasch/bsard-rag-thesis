"""Convenience runner for the cross-arm consolidation.

Injects the sibling `RQ2_T0*/src` directories and the RQ3 `bsard_evaluation`
package onto sys.path, so the consolidation runs without `pip install -e` of the
siblings (mirrors the path-injection trick in T04/T05 scripts). You still need
the heavy runtime deps that T07 pulls in transitively — `faiss`, `torch`,
`pandas`, and the RQ3 IR-eval deps — installed in the active environment.

Usage (from this project root, venv active):
    python scripts/run_consolidate.py
    python scripts/run_consolidate.py --stems 1804_03_21_1804032150
    python scripts/run_consolidate.py --include-ablations
"""

from __future__ import annotations

import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
_T06 = _THIS.parents[1]                      # RQ2_T06_ARM_RESULTS
_RQ2 = _T06.parent                           # RQ2_Structure_Aware_Retrieval
_THESIS = _RQ2.parent                        # mono-repo root
_RQ3 = _THESIS / "RQ3_Autonomous_Evaluation"

for _src in sorted(_RQ2.glob("RQ2_T0*/src")):
    sys.path.insert(0, str(_src))
if _RQ3.exists():
    sys.path.insert(0, str(_RQ3))            # provides the bsard_evaluation package
sys.path.insert(0, str(_T06 / "src"))        # arm_results

from arm_results.consolidate import main  # noqa: E402

if __name__ == "__main__":
    main()
