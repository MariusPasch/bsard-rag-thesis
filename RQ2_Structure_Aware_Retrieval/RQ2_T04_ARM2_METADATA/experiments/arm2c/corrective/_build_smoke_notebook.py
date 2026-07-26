"""Generate ``arm2c_corrective_smoke.ipynb`` — the Azure-kernel variant of the
corrective-loop smoke (OLD navigate() vs NEW navigate_corrective()). Re-run after
edits:  python corrective/_build_smoke_notebook.py

Mirrors _build_jupyterlab_notebook.py's conventions (locate cloned repo, git pull,
pip install ollama, warm+pin Ollama at num_ctx 16384). The run cell calls the SAME
engine the CLI uses (smoke_corrective.run_smoke), so terminal and notebook agree.
Short run (a few queries), foreground — no detach needed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

CELLS: list[tuple[str, str]] = []


def md(s: str) -> None:
    CELLS.append(("markdown", s))


def code(s: str) -> None:
    CELLS.append(("code", s))


md("""\
# Arm-2C corrective-loop smoke (Azure kernel)

OLD `navigate()` (single greedy pass) vs NEW `navigate_corrective()` (CRAG×ReAct
corrective loop) on a few queries, side by side. Reports padding-free reach /
selected recall **and** padded R@10 — so you see whether the loop actually reaches /
selects more gold, not just whether padding moved.

Open this notebook from the cloned repo's `experiments/arm2c/corrective/` folder and
attach the instance's Python kernel (the one with `ollama`).

**Order:** config → locate+deps → Ollama warm+pin → load → run smoke → (save).

Test order (plan): Pénal + Civil first (backtrack/select-gap), Housing as the
negative control.""")

# 1 — config
code('''\
# ── SMOKE CONFIG ────────────────────────────────────────────────────────────
STEM        = "1867_06_08_1867060850"   # Penal | 1804_..2150 Civil | 2003_..1614 Housing(neg)
QIDS        = ["1048", "202", "240"]    # query ids from bundles/<STEM>/queries.json
MODE        = "enriched"                # "enriched" (FR labels + EN summaries) | "bare"
MAX_ROUNDS  = 2                         # corrective rounds (round 1 == old navigate())
RESEED_STRATEGY = "ranked"             # "ranked" (LLM-ranked pruned re-seed) | "all" (brute force)
MAX_NODES   = 40
RESEED_M    = 5
OLLAMA_MODEL = "llama3.1:8b"
print("STEM", STEM, "| QIDS", QIDS, "| MODE", MODE, "| MAX_ROUNDS", MAX_ROUNDS,
      "| RESEED", RESEED_STRATEGY)
''')

# 2 — locate + deps
md("## 1. Locate the experiment dir + dependencies")
code('''\
import sys, subprocess
from pathlib import Path

# This notebook lives in experiments/arm2c/corrective/. Find arm2c (has react_navigator.py).
_cands = [Path.cwd(), Path.cwd().parent,
          Path("/home/azureuser/repos/RQ2_T04_ARM2_METADATA/experiments/arm2c")]
ARM2C = next((p for p in _cands if (p / "react_navigator.py").exists()), None)
assert ARM2C is not None, "Open this notebook from the cloned experiments/arm2c/corrective/ folder."
CORR = ARM2C / "corrective"
subprocess.run(["bash", "-c", f"cd '{ARM2C}' && git pull -q || true"])
for p in (str(ARM2C), str(CORR)):
    if p not in sys.path:
        sys.path.insert(0, p)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ollama"])
print("ARM2C =", ARM2C)
''')

# 3 — GPU + Ollama warm/pin
md("""\
## 2. GPU + Ollama

Warms the model at `num_ctx=16384` and pins it (`keep_alive=-1`) so prompts aren't
truncated (4096 = the old T05 bug) and it can't idle-unload/wedge.""")
code('''\
import subprocess, urllib.request, json

print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout.split("Processes")[0])

def _up():
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3) as r:
            return r.status == 200
    except Exception:
        return False
assert _up(), "Ollama not serving on :11434 — run `sudo systemctl start ollama` in a terminal."

req = json.dumps({"model": OLLAMA_MODEL, "options": {"num_ctx": 16384},
                  "keep_alive": -1, "prompt": "ok", "stream": False}).encode()
urllib.request.urlopen(urllib.request.Request(
    "http://localhost:11434/api/generate", data=req,
    headers={"Content-Type": "application/json"}), timeout=180).read()
print(subprocess.run(["ollama", "ps"], capture_output=True, text=True).stdout)
print(">> want: CONTEXT 16384,  UNTIL Forever")
''')

# 4 — load
md("## 3. Load deep tree + queries for the stem")
code('''\
from smoke_corrective import load_stem
tree, b2l, queries = load_stem(STEM, arm2c_dir=ARM2C)
print(f"tree nodes {len(tree.by_id)} | queries {len(queries)}")
missing = [q for q in QIDS if str(q) not in queries]
print("WARNING missing qids:", missing) if missing else print("all QIDS present")
''')

# 5 — run smoke
md("""\
## 4. Run the smoke — OLD vs NEW

One `LlamaClient`, both navigators, same queries. The table shows padding-free
**reach** / **selected** recall and padded **R@10**, old vs new, plus the corrective
loop's rounds / nodes / calls / seconds. `RESEED_STRATEGY="all"` swaps the ranked
re-seed for the brute-force control.""")
code('''\
from react_navigator import LlamaClient
from smoke_corrective import run_smoke

llm = LlamaClient(model=OLLAMA_MODEL, temperature=0.0, num_ctx=16384)
rows = run_smoke(tree, b2l, queries, [str(q) for q in QIDS], llm,
                 mode=MODE, max_nodes=MAX_NODES, max_rounds=MAX_ROUNDS,
                 reseed_strategy=RESEED_STRATEGY, reseed_m=RESEED_M)
''')

# 6 — save
md("## 5. Save (optional, additive)")
code('''\
from smoke_corrective import write_smoke
p = write_smoke(STEM, MODE, MAX_ROUNDS, RESEED_STRATEGY, rows, arm2c_dir=ARM2C)
print("wrote", p)
''')


def _cell(kind: str, source: str) -> dict:
    base = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        base["execution_count"] = None
        base["outputs"] = []
    return base


def build() -> dict:
    return {
        "cells": [_cell(k, s) for k, s in CELLS],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }


def main() -> int:
    out = Path(__file__).resolve().parent / "arm2c_corrective_smoke.ipynb"
    out.write_text(json.dumps(build(), ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Wrote {out} ({len(CELLS)} cells)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
