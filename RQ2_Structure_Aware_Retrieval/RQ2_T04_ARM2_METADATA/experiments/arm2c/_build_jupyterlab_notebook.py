"""Generate ``arm2c_jupyterlab_run.ipynb`` — the JupyterLab-on-the-compute-instance
variant. Re-run after edits:  python experiments/arm2c/_build_jupyterlab_notebook.py

Differences from the Azure-clone notebook:
  * Assumes it runs ON the instance, inside the cloned repo (no GitHub clone / blob).
  * The long run is LAUNCHED DETACHED (run_all.py via setsid nohup) so it survives
    kernel restarts, browser disconnects, laptop sleep — the kernel only orchestrates.
  * Monitor + analysis cells are re-runnable anytime (work on partial or complete).
  * Ollama: the client pins keep_alive=-1 on every call, so no idle-unload wedge and
    no sudo/service config needed; this cell just verifies it's up at num_ctx 16384.
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
# Arm-2C — JupyterLab run (on the compute instance)

Runs the agentic tree-navigation experiment with the long loop **detached** on the
VM, so kernel restarts / browser drops don't kill it. Open this notebook from the
cloned repo's `experiments/arm2c/` folder.

**Order:** config → locate+deps → Ollama check → load → smoke+pilot gate →
**launch detached** → monitor (re-run) → analysis (re-run).""")

# 1 — config
code('''\
# ── PER-RUN CONFIG ──────────────────────────────────────────────────────────
DOC_ID     = "1804_03_21_1804032150"   # Code Civil (252 q)
MODE       = "enriched"                # "enriched" (FR labels + EN summaries) | "bare"
RERANK     = True                      # +1 call/q: re-rank the navigated pool
MAX_NODES  = 40
MAX_BRANCH = 5

RESULTS_ROOT = "/home/azureuser/results"
OLLAMA_MODEL = "llama3.1:8b"
RUN_NAME = f"arm2c_{DOC_ID}_{MODE}" + ("_rerank" if RERANK else "")
print("RUN_NAME", RUN_NAME, "| MODE", MODE, "| RERANK", RERANK)
''')

# 2 — locate + deps
md("## 1. Locate the experiment dir + dependencies")
code('''\
import sys, subprocess
from pathlib import Path

# This notebook lives in experiments/arm2c/. Find that dir (cwd, or the standard clone path).
_cands = [Path.cwd(), Path("/home/azureuser/repos/RQ2_T04_ARM2_METADATA/experiments/arm2c")]
EXP = next((p for p in _cands if (p / "run_all.py").exists()), None)
assert EXP is not None, "Open this notebook from the cloned experiments/arm2c/ folder."
subprocess.run(["bash", "-c", f"cd '{EXP}' && git pull -q || true"])
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ollama"])
print("EXP =", EXP)
''')

# 3 — GPU + Ollama
md("""\
## 2. GPU + Ollama

The client pins `keep_alive=-1` on every call, so the model can't idle-unload/wedge
— no sudo needed. This cell just confirms Ollama is serving and warms the model at
`num_ctx=16384` (4096 would truncate the prompts).""")
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

# load/pin the model at the context the run uses
req = json.dumps({"model": OLLAMA_MODEL, "options": {"num_ctx": 16384},
                  "keep_alive": -1, "prompt": "ok", "stream": False}).encode()
urllib.request.urlopen(urllib.request.Request(
    "http://localhost:11434/api/generate", data=req,
    headers={"Content-Type": "application/json"}), timeout=180).read()
print(subprocess.run(["ollama", "ps"], capture_output=True, text=True).stdout)
print(">> want: CONTEXT 16384,  UNTIL Forever")
''')

# 4 — load
md("## 3. Load deep tree, queries, baselines")
code('''\
import json
from navigator_tools import DeepTree

bundle = EXP / "bundles" / DOC_ID
tree = DeepTree.load(bundle / "deep_tree.json")
queries = json.loads((bundle / "queries.json").read_text(encoding="utf-8"))
baselines = json.loads((bundle / "baselines.json").read_text(encoding="utf-8"))
print(f"tree nodes {len(tree.by_id)} (depth {tree.manifest.get('stat_max_depth')}) | "
      f"queries {len(queries)}")
print("baselines recall@10:", baselines["baselines_recall@10"])
''')

# 5 — smoke + pilot gate
md("""\
## 4. Smoke + pilot gate

**Do not launch the full run unless the pilot shows `parse-fail = 0%`** — a non-zero
rate means Ollama is unstable and the run would be corrupted.""")
code('''\
import time, statistics
from react_navigator import LlamaClient, navigate

llm = LlamaClient(model=OLLAMA_MODEL, num_ctx=16384)
# single-query smoke
q = queries[0]
t0 = time.perf_counter()
r = navigate(q["query_text"], tree, llm, mode=MODE, max_nodes=MAX_NODES,
             max_branch=MAX_BRANCH, rerank=RERANK)
print(f"[smoke] qid={q['query_id']} {time.perf_counter()-t0:.1f}s "
      f"visited={r.nodes_visited} calls={r.llm_calls} selected={r.selected_bsard_ids[:8]}")

# 5-query pilot
calls, lat, pf, tot = [], [], 0, 0
for q in queries[:5]:
    t0 = time.perf_counter()
    rr = navigate(q["query_text"], tree, llm, mode=MODE, max_nodes=MAX_NODES,
                  max_branch=MAX_BRANCH, rerank=RERANK)
    lat.append(time.perf_counter() - t0); calls.append(rr.llm_calls)
    pf += sum(1 for s in rr.steps if not s.parse_ok); tot += rr.llm_calls
print(f"[pilot] calls/q {statistics.mean(calls):.1f} | s/q {statistics.mean(lat):.1f} | "
      f"parse-fail {pf}/{tot} ({100*pf/max(tot,1):.1f}%)")
print(f"[pilot] projected full run: ~{statistics.mean(lat)*len(queries)/60:.0f} min")
print(">> GATE: parse-fail must be 0.0% before launching the full run")
''')

# 6 — launch detached
md("""\
## 5. Launch the full run — DETACHED

`run_all.py` runs as a background VM process (`setsid nohup`). It is **independent of
this kernel**: restart the kernel, close the browser, sleep the laptop — it keeps
going. It resumes any `q*.json` already on disk. (Only an instance shutdown stops it,
so make sure auto-shutdown won't fire during the run.)""")
code('''\
import subprocess
flag = "--rerank" if RERANK else ""
cmd = (f"cd '{EXP}' && setsid nohup {sys.executable} run_all.py "
       f"--doc-id {DOC_ID} --mode {MODE} {flag} "
       f"--max-nodes {MAX_NODES} --max-branch {MAX_BRANCH} "
       f"> run.log 2>&1 < /dev/null &")
subprocess.Popen(["bash", "-c", cmd])
print("launched detached:\\n ", cmd, "\\n\\nRe-run the monitor cell to watch.")
''')

# 7 — monitor
md("## 6. Monitor (re-run anytime — also after a kernel restart)")
code('''\
import subprocess
print(subprocess.run(["bash", "-c", f"tail -20 '{EXP}/run.log'"],
                     capture_output=True, text=True).stdout)
n = subprocess.run(["bash", "-c", f"ls {RESULTS_ROOT}/{RUN_NAME}/q*.json 2>/dev/null | wc -l"],
                   capture_output=True, text=True).stdout.strip()
print(f"--- {n}/{len(queries)} done ---")
''')

# 8 — analysis
md("""\
## 7. Analysis (re-run anytime)

Works on partial or complete results. When done, shows `[1] HEADLINE`, the
`[1b] RERANK EFFECT` (before/after), the miss decomposition, and the recommendation.
Also written to `<results>/analysis_report.md` + `per_query.csv`.""")
code('''\
import importlib, analyze_results
from pathlib import Path
importlib.reload(analyze_results)
_ = analyze_results.analyze(Path(RESULTS_ROOT) / RUN_NAME, bundle, bundle / "deep_tree.json")
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
    out = Path(__file__).resolve().parent / "arm2c_jupyterlab_run.ipynb"
    out.write_text(json.dumps(build(), ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Wrote {out} ({len(CELLS)} cells)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
