"""Generate ``azure_arm2c_run.ipynb`` from in-source cell strings (T05 pattern).

Re-run whenever a cell changes:  python experiments/arm2c/_build_azure_notebook.py

The Arm-2C runtime is stdlib-only (vendored Node; self-contained LlamaClient),
so the notebook clones ONE repo (this T04 repo, which carries the experiment code
+ pre-built deep_tree.json + the query bundle) and pip-installs only `ollama`.
No T05/shared/rank_bm25/blob machinery.

PREREQUISITE: push experiments/arm2c/ (code + data/<DOC_ID>/deep_tree.json +
bundles/<DOC_ID>/{queries,baselines}.json) to GitHub so the clone picks them up.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

CELLS: list[tuple[str, str]] = []


def md(s: str) -> None:
    CELLS.append(("markdown", f"*— cell {len(CELLS)} —*\n\n{s}"))


def code(s: str) -> None:
    CELLS.append(("code", f"# === cell {len(CELLS)} ===\n{s}"))


# 0 — Title
md("""\
# Arm-2C — Agentic tree navigation (Azure GPU run)

A ReAct agent (LLaMA 3.1 8B, vectorless) descends a **deep** AzureDI tree
(LIVRE › TITRE › CHAPITRE › Section › Article) rebuilt from the full header
stack — testing whether a better-built tree fixes Arm-2B/PageIndex's
wrong-chapter miss.

**One run = one PDF × one mode.** Sequence: GPU → Ollama → clone repo →
pip install ollama → load tree+queries → warmup → single-query smoke →
5-query pilot + ETA gate → full run → **success/failure check**.

Resume-friendly: any `q<qid>.json` already on disk is skipped.
""")

# 1 — Config
code('''\
# ── PER-RUN CONFIG (edit these) ─────────────────────────────────────────────
DOC_ID    = "1804_03_21_1804032150"   # best-chance PDF: Code Civil — most gold (252 q),
                                       # richest deep tree (depth 7), biggest flat->deep contrast
MODE      = "enriched"                 # "enriched" = FR labels + native-EN level_summary (branches)
                                       # + content_summary/keywords (articles); "bare" = labels only.
                                       # bare baseline already run (R@10 0.166); enriched is the test.
MAX_NODES = 40                         # frontier-descent budget: nodes visited per query (= LLM calls)
MAX_BRANCH = 5                         # max sub-sections descended per node (caps frontier/cost; T05 used 5)
RERANK    = True                       # +1 call/query: re-rank the navigated pool by relevance.
                                       # enriched (no rerank) gave R@10 0.214 with R@100 0.515 -> rerank
                                       # converts that reachable gold into the top-10.

# ── REPO + AUTH ─────────────────────────────────────────────────────────────
GITHUB_TOKEN  = ""                     # PAT with read scope (leave blank if public)
GITHUB_OWNER  = "MariusPasch"
GITHUB_REPO   = "RQ2_T04_ARM2_METADATA"
GITHUB_BRANCH = "master"

# ── VM PATHS ────────────────────────────────────────────────────────────────
REPOS_DIR    = "/home/azureuser/repos"
RESULTS_ROOT = "/home/azureuser/results"
OLLAMA_MODEL = "llama3.1:8b"

RUN_NAME = f"arm2c_{DOC_ID}_{MODE}" + ("_rerank" if RERANK else "")
print("RUN_NAME   ", RUN_NAME)
print(f"DOC_ID {DOC_ID} | MODE {MODE} | MAX_NODES {MAX_NODES} | RERANK {RERANK}")
''')

# 2 — GPU
md("## 1. GPU sanity check")
code("!nvidia-smi\n")

# 3 — Ollama (proven cell, copied from T05)
md("""\
## 2. Ollama install / start / pull

Idempotent: skips install if present, skips start if 11434 serves, skips
pull if cached. KV-cache q8_0 keeps the 16k context inside a T4's VRAM.""")
code('''\
import os, subprocess, time, urllib.request

if subprocess.run(["which", "ollama"], capture_output=True, text=True).returncode != 0:
    print("Installing Ollama...")
    subprocess.run("curl -fsSL https://ollama.com/install.sh | sh", shell=True, check=True)

os.environ["OLLAMA_NUM_GPU"] = "99"
os.environ["OLLAMA_FLASH_ATTENTION"] = "1"
os.environ["OLLAMA_KV_CACHE_TYPE"] = "q8_0"
os.environ["OLLAMA_HOST"] = "0.0.0.0:11434"

def _up(url="http://localhost:11434/api/tags"):
    try:
        with urllib.request.urlopen(url, timeout=2) as r:
            return r.status == 200
    except Exception:
        return False

if not _up():
    print("Starting ollama serve...")
    subprocess.Popen(["bash", "-c", "ollama serve > /tmp/ollama.log 2>&1 &"])
    for _ in range(30):
        if _up():
            break
        time.sleep(1)
    else:
        raise RuntimeError("ollama serve not reachable on 11434 in 30s. See /tmp/ollama.log.")

print("Pulling model (no-op if cached)...")
subprocess.run(["ollama", "pull", OLLAMA_MODEL], check=True)
print("Ollama ready.")
''')

# 4 — Clone repo
md("## 3. Clone the T04 repo (carries Arm-2C code + tree + query bundle)")
code('''\
import subprocess
from pathlib import Path

parent = Path(REPOS_DIR); parent.mkdir(parents=True, exist_ok=True)
target = parent / GITHUB_REPO
expected = f"https://github.com/{GITHUB_OWNER}/{GITHUB_REPO}.git"
auth = f"https://{GITHUB_TOKEN}@github.com/{GITHUB_OWNER}/{GITHUB_REPO}.git" if GITHUB_TOKEN else expected

if target.exists():
    subprocess.run(["git", "-C", str(target), "remote", "set-url", "origin", auth], check=True)
    subprocess.run(["git", "-C", str(target), "fetch", "--quiet"], check=True)
    subprocess.run(["git", "-C", str(target), "checkout", GITHUB_BRANCH], check=True)
    subprocess.run(["git", "-C", str(target), "pull", "--quiet"], check=True)
else:
    subprocess.run(["git", "clone", "--quiet", "--branch", GITHUB_BRANCH, auth, str(target)], check=True)

head = subprocess.run(["git", "-C", str(target), "log", "-1", "--oneline"],
                      capture_output=True, text=True).stdout.strip()
print("repo:", target, "|", head)
''')

# 5 — pip + path
md("## 4. Install `ollama` + add Arm-2C to the path")
code('''\
import subprocess, sys
from pathlib import Path

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ollama"], check=True)
EXP_DIR = str(Path(REPOS_DIR) / GITHUB_REPO / "experiments" / "arm2c")
if EXP_DIR not in sys.path:
    sys.path.insert(0, EXP_DIR)

import navigator_tools, react_navigator, check_results   # stdlib + ollama only
print("Arm-2C modules import OK from", EXP_DIR)
''')

# 6 — Load tree + queries
md("## 5. Load deep tree, queries, baselines")
code('''\
import json
from pathlib import Path
from navigator_tools import DeepTree

EXP = Path(REPOS_DIR) / GITHUB_REPO / "experiments" / "arm2c"
bundle = EXP / "bundles" / DOC_ID
tree = DeepTree.load(bundle / "deep_tree.json")   # tree shipped inside the bundle
queries = json.loads((bundle / "queries.json").read_text(encoding="utf-8"))
baselines = json.loads((bundle / "baselines.json").read_text(encoding="utf-8"))

print(f"tree nodes : {len(tree.by_id)}  (depth {tree.manifest.get('stat_max_depth')}, "
      f"orphan {tree.manifest.get('stat_orphan_pct')}%)")
print(f"queries    : {len(queries)}")
print(f"baselines recall@10 : {baselines['baselines_recall@10']}")
print(f"  -> bar to clear (Arm-2B): {baselines['baselines_recall@10'].get('arm2b')}")
''')

# 7 — Warmup
md("## 6. Pre-flight smokes\\n\\n### 6a. LLM warmup (3 cold calls)")
code('''\
import time
from react_navigator import LlamaClient

llm = LlamaClient(model=OLLAMA_MODEL, num_ctx=16384)
for i in range(3):
    t0 = time.perf_counter()
    txt = llm.generate('Reponds uniquement en JSON: {"ok": true}')
    print(f"call {i+1}: {(time.perf_counter()-t0)*1000:7.0f} ms   text={txt[:60]!r}")
''')

# 8 — Single-query smoke
md("### 6b. Single-query end-to-end (confirms FR-JSON descent + trace)")
code('''\
import time
from react_navigator import navigate

q = queries[0]
print(f"qid={q['query_id']}  {q['query_text'][:90]!r}\\n  gold={q['gold_bsard_ids'][:12]}\\n")
t0 = time.perf_counter()
res = navigate(q["query_text"], tree, llm, mode=MODE, max_nodes=MAX_NODES, max_branch=MAX_BRANCH, rerank=RERANK)
print(f"{time.perf_counter()-t0:.1f}s  visited={res.nodes_visited}  calls={res.llm_calls}  "
      f"exit={res.exit_reason}")
print(f"selected bsard_ids: {res.selected_bsard_ids[:12]}")
print("steps:")
for s in res.steps:
    flag = "" if s.parse_ok else "  PARSE_FAIL"
    print(f"  [{s.title[:38]:38}] sec={len(s.sections)} art={len(s.articles)}  "
          f"{s.latency_ms:6.0f}ms{flag}")
''')

# 9 — Pilot + ETA gate
md("""\
### 6c. 5-query pilot + time-budget gate

**Continue past this cell only if the projected total and parse-fail rate
look acceptable.** The next cell starts the full run.""")
code('''\
import statistics, time
from react_navigator import navigate

pilot = queries[: min(5, len(queries))]
calls, lat, pf, tot = [], [], 0, 0
for q in pilot:
    t0 = time.perf_counter()
    r = navigate(q["query_text"], tree, llm, mode=MODE, max_nodes=MAX_NODES, max_branch=MAX_BRANCH, rerank=RERANK)
    lat.append(time.perf_counter() - t0)
    calls.append(r.llm_calls)
    pf += sum(1 for s in r.steps if not s.parse_ok)
    tot += r.llm_calls

mean_lat = statistics.mean(lat)
print(f"pilot ({len(pilot)} q): calls/q mean={statistics.mean(calls):.1f}  "
      f"s/q mean={mean_lat:.1f}  parse-fail={pf}/{tot} ({100*pf/max(tot,1):.1f}%)")
print(f"PROJECTED full run on {len(queries)} q: ~{mean_lat*len(queries)/60:.0f} min")
print("Inspect, then run the full-run cell.")
''')

# 9b — Fresh-run guard
md("""\
### Fresh-run guard

The full run is resume-friendly (skips any `q<qid>.json` already on disk) — great
for resuming an interrupted run, but it means a **code change silently reuses old
results**. Set `FRESH_RUN = True` to wipe THIS `RUN_NAME`'s results for a clean
from-scratch run; `False` to resume.""")
code('''\
import shutil
from pathlib import Path

FRESH_RUN = True   # True = wipe and start clean; False = resume an interrupted run

RESULTS = Path(RESULTS_ROOT) / RUN_NAME
if FRESH_RUN and RESULTS.exists():
    shutil.rmtree(RESULTS)
    print("wiped", RESULTS)
RESULTS.mkdir(parents=True, exist_ok=True)
print(f"FRESH_RUN={FRESH_RUN}  results dir: {RESULTS}")
''')

# 10 — Full run
md("## 7. Full run (resume-friendly — skips any q<qid>.json already on disk)")
code('''\
import json, time
from dataclasses import asdict
from pathlib import Path
from react_navigator import navigate

RESULTS = Path(RESULTS_ROOT) / RUN_NAME
RESULTS.mkdir(parents=True, exist_ok=True)
done = {p.stem for p in RESULTS.glob("q*.json")}
print(f"resuming: {len(done)}/{len(queries)} already done")

t0 = time.perf_counter()
for i, q in enumerate(queries, 1):
    qid = str(q["query_id"])
    if f"q{qid}" in done:
        continue
    r = navigate(q["query_text"], tree, llm, mode=MODE, max_nodes=MAX_NODES, max_branch=MAX_BRANCH, rerank=RERANK)
    rec = {
        "query_id": qid, "query_text": q["query_text"], "gold_bsard_ids": q["gold_bsard_ids"],
        "selected_bsard_ids": r.selected_bsard_ids, "ranked_bsard_ids": r.ranked_bsard_ids,
        "ranked_bsard_ids_prererank": r.ranked_bsard_ids_prererank,
        "nodes_visited": r.nodes_visited, "llm_calls": r.llm_calls,
        "exit_reason": r.exit_reason, "mode": MODE,
        "steps": [asdict(s) for s in r.steps],
    }
    (RESULTS / f"q{qid}.json").write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
    if i % 10 == 0:
        print(f"  {i}/{len(queries)}  ({(time.perf_counter()-t0)/max(i-len(done),1):.1f}s/q)")

print(f"\\nDONE — {len(list(RESULTS.glob('q*.json')))} results in {RESULTS}")
''')

# 11 — Analysis & decision
md("""\
## 8. Analysis & decision

Not just pass/fail — a full report to decide **scale to the other PDFs, or change
the approach (and what)**: headline vs baselines, navigation behaviour, and the key
**miss decomposition** — every gold article classified as HIT / SEEN_NOT_SELECTED
(seen in a menu, not picked → selection problem) / NOT_REACHED (branch pruned → tree
problem, the Arm-2B failure mode) / ORPHAN_UNREACHED (Unfiled). Plus per-cardinality
recall, worst/best queries, and a rule-based recommendation. Writes
`analysis_report.md` + `per_query.csv` next to the results.""")
code('''\
import importlib, analyze_results
importlib.reload(analyze_results)
summary = analyze_results.analyze(RESULTS, bundle, bundle / "deep_tree.json")
''')

# 12 — Download hint + cleanup
md("""\
## 9. Copy results down + cleanup

Per-query `q<qid>.json` + `analysis_report.md` + `per_query.csv` are under
`RESULTS_ROOT/RUN_NAME/`. Pull them into the **committable** `runs/` dir (NOT
`results/`, which is gitignored) so they persist for future analysis /
bare-vs-enriched compare:

```
scp -r azureuser@<vm-ip>:/home/azureuser/results/<RUN_NAME> \\
       "<repo>/RQ2_T04_ARM2_METADATA/experiments/arm2c/runs/"
```""")
code('''\
import subprocess
subprocess.run(["pkill", "-f", "ollama"], check=False)
print("Ollama stopped. Results in:", RESULTS)
''')


# ── Builder ──────────────────────────────────────────────────────────────────


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
    out = Path(__file__).resolve().parent / "azure_arm2c_run.ipynb"
    out.write_text(json.dumps(build(), ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Wrote {out} ({len(CELLS)} cells)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
