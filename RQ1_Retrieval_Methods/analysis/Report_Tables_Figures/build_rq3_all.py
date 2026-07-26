"""
Regenerate every RQ3 (Chapter 14) figure and table from the canonical JSONs.

Runs each build script in a fresh subprocess (isolated matplotlib / rcParams
state), in story order. Rerunning rebuilds the exact same artifacts.

    .venv/Scripts/python Report/build_rq3_all.py

Per STYLEGUIDE.md sec 6 / sec 11: scoped to RQ3 for now; an RQ1 counterpart and a
combined entry-point can be added the same way.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPORT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = REPORT_DIR / "scripts"

# Story order: Act 1, Act 2, synthesis.
SCRIPTS = [
    "build_rq3_agreement.py",       # fig_rq3_eval_corr_heatmap + table
    "build_rq3_strata_heatmap.py",  # fig_rq3_eval_strata_heatmap
    "build_rq3_eval_strata.py",     # fig_rq3_eval_trust_strata + _strata_dumbbell
    "build_rq3_gold_ceiling.py",    # fig_rq3_gold_ceiling + table
    "build_rq3_aqs.py",             # fig_rq3_aqs_weight_simplex + _old_vs_new + table
]


def main() -> int:
    failures: list[str] = []
    for name in SCRIPTS:
        path = SCRIPTS_DIR / name
        print(f"\n=== {name} ===")
        result = subprocess.run([sys.executable, str(path)], cwd=str(REPORT_DIR.parent))
        if result.returncode != 0:
            failures.append(name)
    print("\n" + "=" * 48)
    if failures:
        print(f"FAILED: {', '.join(failures)}")
        return 1
    print(f"All {len(SCRIPTS)} RQ3 build scripts completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
