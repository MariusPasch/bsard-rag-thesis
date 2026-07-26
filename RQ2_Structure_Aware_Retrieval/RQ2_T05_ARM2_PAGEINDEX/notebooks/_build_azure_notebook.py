"""Generate ``azure_t05_pageindex_run.ipynb`` from in-source cell strings.

Re-run this script whenever a cell's content changes. Hand-editing the
.ipynb directly is fine for ad-hoc tweaks but the generator stays the
source of truth.

    python notebooks/_build_azure_notebook.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


# ── Cell sources ─────────────────────────────────────────────────────────────


CELLS: list[tuple[str, str]] = []


def md(s: str) -> None:
    idx = len(CELLS)
    CELLS.append(("markdown", f"*— cell {idx} —*\n\n{s}"))


def code(s: str) -> None:
    idx = len(CELLS)
    CELLS.append(("code", f"# === cell {idx} ===\n{s}"))


# 0 — Title
md("""\
# T05 Arm 2B (PageIndex) — Azure GPU run

Vectorless, reasoning-based retrieval. LLaMA 3.1 8B navigates a deterministic
ToC tree built locally from the AzureDI dump.

**Sequence:** GPU check → install Ollama → install pip deps → download
bundle → pre-flight smokes → time-budget gate → full run → upload
results → cleanup.

`run_subset` is **idempotent on `out_dir`** — restarting the kernel
mid-run resumes from any per-query files already written, locally or
from the blob.
""")


# 1 — Config
code("""\
# Edit only the two lines under "PER-RUN CONFIG" between launches.
# Everything else is derived.

# ── PER-RUN CONFIG ──────────────────────────────────────────────────────────
# One run = one PDF. Pick the BSARD stem from RQ2_T00_ORCHESTRATOR/data/selected_pdfs.json.
# RUN_NAME conventionally is "t04_<stem>" — matches the GT filename and the
# bundle name produced by scripts/prepare_azure_bundle.py.
RUN_NAME = "t04_1804_03_21_1804032150"   # sanity-check PDF for the num_ctx/prompt fixes
#RUN_NAME = "t04_1867_06_08_1867060850"
#RUN_NAME = "t04_1967_10_10_1967101055"
#RUN_NAME = "t04_1967_10_10_1967101056"
#RUN_NAME = "t04_2003_07_17_2013A31614"

# Container-level SAS with read+write+list+create. Rotate per session.
AZURE_CONTAINER_SAS_URL = ""

# ── REPO + AUTH ─────────────────────────────────────────────────────────────
GITHUB_TOKEN = ""                                # PAT with read scope
GITHUB_OWNER = "MariusPasch"
GITHUB_BRANCH = "master"

# T01 (LLMClient), T02 (RetrievalResult + Article), T05 (this arm). T03+T04 are
# local-only deps used by build_tree.py — the tree is shipped pre-built in the
# bundle so we don't need them here.
SIBLING_REPOS = {
    "T01_SHARED":      "RQ2_T01_SHARED",
    "T02_DATA_LOADER": "RQ2_T02_DATA_LOADER",
    "T05_PAGEINDEX":   "RQ2_T05_ARM2_PAGEINDEX",
}

# ── BLOB LAYOUT (derived) ───────────────────────────────────────────────────
# Convention shared with notebooks/local_t05_eval_and_compare.ipynb:
#   t05_azure_bundles/<RUN_NAME>.zip   — uploaded by scripts/upload_to_blob.py
#   t05_results/<RUN_NAME>/q*.json     — written by section 9 below
BUNDLE_BLOB_NAME = f"t05_azure_bundles/{RUN_NAME}.zip"
RESULTS_BLOB_PREFIX = f"t05_results/{RUN_NAME}"

# ── LOCAL PATHS ON THE VM ───────────────────────────────────────────────────
REPOS_DIR = "/home/azureuser/repos"
BUNDLE_DIR = "/home/azureuser/bundle"
RESULTS_DIR = "/home/azureuser/results"
OLLAMA_MODEL = "llama3.1:8b"

print(f"RUN_NAME            {RUN_NAME}")
print(f"BUNDLE_BLOB_NAME    {BUNDLE_BLOB_NAME}")
print(f"RESULTS_BLOB_PREFIX {RESULTS_BLOB_PREFIX}")
""")


# 2 — GPU check
md("## 1. GPU sanity check\\n\\nBail early if there is no CUDA device — the run is hours of CPU.")
code("""\
!nvidia-smi
""")


# 3 — Ollama install + serve + pull
md("""\
## 2. Ollama install / start / pull

Idempotent: skips install if already present, skips start if 11434 is
serving, skips pull if the model layer is cached.""")
code("""\
import os
import subprocess
import time
import urllib.request

# Install Ollama if missing.
which = subprocess.run(["which", "ollama"], capture_output=True, text=True)
if which.returncode != 0:
    print("Installing Ollama...")
    subprocess.run(
        "curl -fsSL https://ollama.com/install.sh | sh",
        shell=True, check=True,
    )

# Push the GPU env vars into ollama serve's environment.
os.environ["OLLAMA_NUM_GPU"] = "99"
os.environ["OLLAMA_FLASH_ATTENTION"] = "1"
# KV-cache in 8-bit halves VRAM use at 16k context (Llama 3.1 8B's KV
# cache otherwise grows past the T4's 16 GB at num_ctx=16384). Has a
# negligible quality impact on this navigation task.
os.environ["OLLAMA_KV_CACHE_TYPE"] = "q8_0"
os.environ["OLLAMA_HOST"] = "0.0.0.0:11434"


def _ollama_up(url: str = "http://localhost:11434/api/tags") -> bool:
    try:
        with urllib.request.urlopen(url, timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


if not _ollama_up():
    print("Starting ollama serve...")
    subprocess.Popen(
        ["bash", "-c", "ollama serve > /tmp/ollama.log 2>&1 &"],
    )
    for _ in range(30):
        if _ollama_up():
            break
        time.sleep(1)
    else:
        raise RuntimeError(
            "ollama serve did not become reachable on 11434 in 30s. "
            "Check /tmp/ollama.log."
        )

print("Pulling model (no-op if cached)...")
subprocess.run(["ollama", "pull", OLLAMA_MODEL], check=True)
print("Ollama ready.")
""")


# 4 — Clone sibling repos
md("""\
## 3. Clone sibling repos

Clones the three required GitHub repos into separate directories under
`REPOS_DIR`. Validates that any pre-existing directory's `origin` matches
the configured URL - mismatches abort loudly rather than silently
pulling from the wrong remote (a previous bug).""")
code("""\
import subprocess
from pathlib import Path


def _origin_url(target: Path) -> str | None:
    r = subprocess.run(
        ["git", "-C", str(target), "remote", "get-url", "origin"],
        capture_output=True, text=True,
    )
    return r.stdout.strip() if r.returncode == 0 else None


repo_paths: dict[str, Path] = {}
parent = Path(REPOS_DIR)
parent.mkdir(parents=True, exist_ok=True)

for label, repo_name in SIBLING_REPOS.items():
    target = parent / repo_name
    expected_url = f"https://github.com/{GITHUB_OWNER}/{repo_name}.git"
    auth_url = (
        f"https://{GITHUB_TOKEN}@github.com/{GITHUB_OWNER}/{repo_name}.git"
        if GITHUB_TOKEN else expected_url
    )

    if target.exists():
        existing = _origin_url(target) or ""
        # Strip embedded token before comparing so a token rotation doesn't
        # trigger a false mismatch.
        normalized = existing.split("@github.com", 1)[-1]
        expected_path = expected_url.split("github.com", 1)[-1]
        if not normalized.endswith(expected_path):
            raise RuntimeError(
                f"{target} exists but origin = {existing!r} (expected "
                f"{expected_url!r}). Either rm -rf {target} or fix "
                f"SIBLING_REPOS['{label}']."
            )
        # Refresh the embedded token in case it rotated since last clone.
        subprocess.run(
            ["git", "-C", str(target), "remote", "set-url", "origin", auth_url],
            check=True,
        )
        subprocess.run(["git", "-C", str(target), "fetch", "--quiet"], check=True)
        subprocess.run(["git", "-C", str(target), "checkout", GITHUB_BRANCH], check=True)
        subprocess.run(["git", "-C", str(target), "pull", "--quiet"], check=True)
    else:
        subprocess.run(
            ["git", "clone", "--quiet", "--branch", GITHUB_BRANCH, auth_url, str(target)],
            check=True,
        )

    repo_paths[label] = target
    head = subprocess.run(
        ["git", "-C", str(target), "log", "-1", "--oneline"],
        capture_output=True, text=True,
    ).stdout.strip()
    print(f"  {label:<18}  {target}  ({head})")

print(f"\\nCloned/updated {len(repo_paths)} repos in {REPOS_DIR}")
""")


# 5 — pip install
md("""\
## 4. Install Python packages

Editable installs of T01, T02, T05. T03/T04 are deliberately excluded -
they are only needed locally by `build_tree.py`, not at Azure runtime,
since the tree is shipped pre-built in the bundle.""")
code("""\
import subprocess
import sys

PIP = [sys.executable, "-m", "pip", "install", "-q"]

base_pkgs = [
    "numpy>=1.26", "pandas>=2.0", "tqdm>=4.66", "ollama>=0.2", "PyMuPDF",
    "azure-storage-blob",
]
subprocess.run(PIP + base_pkgs, check=True)

for label, target in repo_paths.items():
    subprocess.run(PIP + ["-e", str(target)], check=True)
    print(f"  installed {label} from {target}")

# Defensive: PEP 660 editable installs occasionally fail to register a .pth
# in this Anaconda env without raising (`pip show` succeeds, but the
# package is missing from sys.path). Inject src/ paths so the package
# imports either way.
for label, target in repo_paths.items():
    src_path = str(target / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

print("\\nAll packages installed (+ src/ paths added as fallback).")
""")


# 5b — Fresh-run guard
md("""\
## 5b. Fresh-run guard

`run_subset` is idempotent on `out_dir` — it reads any per-query JSON
already on disk and skips that LLM call. That's the right default for
resuming an interrupted run, but it silently makes a CODE-CHANGE re-run
reuse the OLD results.

Set `FRESH_RUN = True` to wipe both the VM-local bundle dir
(`/home/azureuser/bundle/<RUN_NAME>/`) and results dir
(`/home/azureuser/results/<RUN_NAME>/`) for THIS `RUN_NAME` only.
Other PDFs' caches on the same VM are untouched. Default `False` so
ordinary resumes still work.""")
code("""\
import shutil
from pathlib import Path

from azure.storage.blob import ContainerClient

FRESH_RUN = True   # FLIP THIS WHEN RE-RUNNING WITH PATCHED CODE.

if FRESH_RUN:
    # 1. Local VM dirs.
    for base in (BUNDLE_DIR, RESULTS_DIR):
        p = Path(base) / RUN_NAME
        if p.exists():
            print(f"Wiping {p}")
            shutil.rmtree(p)
        else:
            print(f"Already absent: {p}")
    # 2. Azure blob prefix. Cell 21's resume step pulls anything under
    #    t05_results/<RUN_NAME>/ back into the local results dir, which
    #    would silently re-cache stale per-query JSONs and make
    #    run_subset skip every query. Cleaning the blob here is the only
    #    way to guarantee a truly fresh run.
    _container = ContainerClient.from_container_url(AZURE_CONTAINER_SAS_URL)
    _prefix = f"{RESULTS_BLOB_PREFIX}/"
    _stale = [b.name for b in _container.list_blobs(name_starts_with=_prefix)]
    # Defensive: every name we delete must live under our prefix.
    assert all(n.startswith(_prefix) for n in _stale), \\
        f"refusing — found blob outside {_prefix}"
    for name in _stale:
        _container.delete_blob(name)
    print(f"Cleared {len(_stale)} stale blobs from {_prefix}")
    print("\\nFresh-run wipe complete.")
else:
    print("FRESH_RUN=False — keeping any existing local + blob results for"
          f" {RUN_NAME}. run_subset will resume from cached per-query JSONs.")
""")


# 6 — Download bundle
md("""\
## 5. Download bundle from Azure Blob

The bundle was built locally with `scripts/prepare_azure_bundle.py` and
uploaded to the container. It contains the trees, queries (with
question texts joined from `bsard_corpus.db`), and the original GT
file.""")
code("""\
import zipfile
from pathlib import Path

from azure.storage.blob import ContainerClient

# One bundle dir per RUN_NAME — without this, reusing a VM that has run a
# different PDF will MIX trees/manifest from that prior PDF with the
# newly-downloaded bundle, and cell 13's `trees/doc_*.json` glob will
# pick up whichever trees Linux happens to list first.
bundle_dir = Path(BUNDLE_DIR) / RUN_NAME
bundle_dir.mkdir(parents=True, exist_ok=True)

container = ContainerClient.from_container_url(AZURE_CONTAINER_SAS_URL)
local_zip = bundle_dir / "bundle.zip"
if not local_zip.exists():
    print(f"Downloading {BUNDLE_BLOB_NAME} ...")
    with local_zip.open("wb") as f:
        f.write(
            container.get_blob_client(BUNDLE_BLOB_NAME)
            .download_blob().readall()
        )
print(f"Bundle zip: {local_zip} ({local_zip.stat().st_size/1024:.1f} KB)")

# Extract (idempotent — overwrites in place).
with zipfile.ZipFile(local_zip) as zf:
    zf.extractall(bundle_dir)

print("\\nBundle contents:")
for p in sorted(bundle_dir.rglob("*")):
    if p.is_file() and p != local_zip:
        rel = p.relative_to(bundle_dir)
        print(f"  {str(rel):<40}  {p.stat().st_size:>9,} B")
""")


# 7 — Load tree + queries
md("## 6. Load tree, queries, manifest")
code("""\
import json
from pathlib import Path

bundle_dir = Path(BUNDLE_DIR) / RUN_NAME

# Cell 9's defensive sys.path hook makes this import safe even when the
# editable install didn't register cleanly.
from arm2_pageindex.tree_builder import compose_corpus_tree, load_law_tree

queries = json.loads((bundle_dir / "queries.json").read_text(encoding="utf-8"))
gt = json.loads((bundle_dir / "gt.json").read_text(encoding="utf-8"))
manifest_in = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))

# Hard fail if the bundle's manifest doesn't match RUN_NAME — this is the
# canary for a stale bundle / wrong-PDF mix-up.
if manifest_in.get("run_name") not in (RUN_NAME, None):
    raise RuntimeError(
        f"Bundle manifest run_name={manifest_in.get('run_name')!r} "
        f"does not match RUN_NAME={RUN_NAME!r}. "
        f"Wipe {bundle_dir} and re-run cell 11."
    )

law_subtrees = []
for tp in sorted((bundle_dir / "trees").glob("doc_*.json")):
    node, m = load_law_tree(tp)
    law_subtrees.append(node)
    print(f"  {tp.name}  doc_id={m.get('document_id')}  "
          f"arts={m.get('n_articles')}  chaps={m.get('n_chapters')}  "
          f"derivable={m.get('chapter_derivable')}")

# When more than one law tree is loaded, wrap them under a corpus root so
# the navigator's law-selection step is meaningful.
tree = compose_corpus_tree(law_subtrees) if len(law_subtrees) > 1 else law_subtrees[0]

print(
    f"\\nQueries: {len(queries)}  Trees: {len(law_subtrees)}  "
    f"GT entries: {len(gt)}  Run name: {manifest_in.get('run_name')!r}"
)
""")


# 8 — Pre-flight smokes header
md("""\
## 7. Pre-flight smokes

Three gates before the long run.

1. **LLM warmup** — 3 cold calls; drop call 1, average 2-3 as the
   "warm" baseline tok/s.
2. **Single-query end-to-end** — full navigation on the first query;
   confirms French JSON-strict parsing.
3. **5-question pilot** — small batch with parse-fail rate +
   LLM-calls-per-query distribution + extrapolated ETA.""")


# 9 — Smoke 1: LLM warmup
code("""\
import time

from shared.llm import LLMClient

# num_ctx=16384 overrides Ollama's 4k default. T05 chapter-selection
# prompts are 6-9k tokens on the curated 5 PDFs and were silently
# truncated at 4k pre-fix, erasing the chapter list (see CHANGE_NOTES).
llm = LLMClient(model=OLLAMA_MODEL, num_ctx=16384)

warmup_prompt = (
    "Réponds par un seul mot en JSON : {\\\"reply\\\": \\\"oui\\\"}\\n"
    "Question : Le ciel est-il bleu ?"
)

latencies = []
for i in range(3):
    t0 = time.perf_counter()
    r = llm.generate(warmup_prompt)
    dt = (time.perf_counter() - t0) * 1000
    latencies.append(dt)
    tps = r.output_tokens / max(r.latency_ms / 1000, 0.001)
    print(
        f"call {i+1}: {dt:>7.0f} ms  "
        f"in={r.input_tokens}  out={r.output_tokens}  ~{tps:.1f} tok/s  "
        f"text={r.text!r}"
    )

warm_mean_ms = sum(latencies[1:]) / max(len(latencies) - 1, 1)
print(f"\\nWarm baseline (calls 2-3 mean): {warm_mean_ms:.0f} ms / call")
""")


# 10 — Smoke 2: single-query e2e
code("""\
import time

from arm2_pageindex.navigator import NavigatorConfig
from arm2_pageindex.retriever import run_arm2b

cfg = NavigatorConfig(max_iterations=2)
smoke_q = queries[0]
print(f"Smoke query (qid={smoke_q['query_id']}): {smoke_q['query_text']!r}\\n")

t0 = time.perf_counter()
result = run_arm2b(
    query=smoke_q["query_text"],
    query_id=smoke_q["query_id"],
    tree=tree,
    llm=llm,
    cfg=cfg,
)
elapsed = time.perf_counter() - t0

print(f"Wall time:       {elapsed:.1f} s")
print(f"LLM calls:       {result.cost.get('llm_calls')}")
print(f"Tokens in/out:   {result.cost.get('tokens_in')} / {result.cost.get('tokens_out')}")
print(f"Parse failures:  {result.cost.get('parse_failures')}")
print(f"Exit reason:     {result.cost.get('exit_reason')}")
print(f"Ranked:          {len(result.ranked_items)}")
for item in result.ranked_items[:5]:
    md_ = item["metadata"]
    print(
        f"  score={item['score']:>4.1f}  "
        f"bsard_id={md_.get('bsard_id')}  "
        f"art_no={md_.get('article_number')!r}  {item['id']}"
    )

print("\\nSteps:")
for s in (result.trace or {}).get("steps", []):
    name = s.get("step", "?")
    if s.get("skipped"):
        print(f"  [SKIP]   {name}  reason={s.get('reason')}")
    else:
        ok = "ok" if s.get("parse_ok") else "FAIL"
        print(
            f"  parse={ok:<4}  {name:<38}  "
            f"in={s.get('tokens_in', 0):>5}  out={s.get('tokens_out', 0):>5}  "
            f"{s.get('latency_ms', 0):>7.1f} ms"
        )
""")


# 11 — Smoke 3: 5-question pilot
code("""\
import statistics

from arm2_pageindex.pipeline import run_subset

pilot_queries = queries[: min(5, len(queries))]

pilot_results = run_subset(
    tree=tree,
    queries=pilot_queries,
    llm=llm,
    cfg=cfg,
    out_dir=None,            # in-memory only — do not pollute results dir yet
    progress=False,
    log_every=1,
)

call_counts = [r.cost.get("llm_calls", 0) for r in pilot_results]
latencies = [r.cost.get("latency_ms", 0) for r in pilot_results]
parse_fails = sum(r.cost.get("parse_failures", 0) for r in pilot_results)
total_calls = sum(call_counts)
exit_reasons = {}
for r in pilot_results:
    er = r.cost.get("exit_reason", "?")
    exit_reasons[er] = exit_reasons.get(er, 0) + 1

print(f"Pilot ({len(pilot_results)} queries):")
print(
    f"  LLM calls/q (mean / median / max): "
    f"{statistics.mean(call_counts):.1f} / "
    f"{int(statistics.median(call_counts))} / {max(call_counts)}"
)
print(
    f"  Latency ms/q (mean / max):         "
    f"{statistics.mean(latencies):.0f} / {max(latencies):.0f}"
)
print(
    f"  Parse failures total:              "
    f"{parse_fails} / {total_calls} calls "
    f"({100 * parse_fails / max(total_calls, 1):.1f}%)"
)
print(f"  Exit reasons:                      {exit_reasons}")
""")


# 12 — Time-budget gate
md("""\
### Time-budget gate

Inspect the projected total below. **Continue past this cell only when
the numbers look acceptable.** Cell 14 starts the full run.""")
code("""\
import statistics

mean_latency_s = statistics.mean(latencies) / 1000.0
mean_calls = statistics.mean(call_counts)
total_q = len(queries)

projected_s = mean_latency_s * total_q
projected_calls = mean_calls * total_q
projected_fails = (parse_fails / max(total_calls, 1)) * projected_calls

print(f"Projected full run on {total_q} queries:")
print(f"  Wall time:   ~{projected_s/60:.1f} min  ({projected_s:.0f} s)")
print(f"  LLM calls:   ~{projected_calls:.0f}")
print(f"  Parse fails: ~{projected_fails:.0f}")
print()
print("Inspect — then run the next cell to start the long run.")
""")


# 13 — Full run header
md("""\
## 8. Full run

Resume-friendly: any per-query JSON already in `RESULTS_DIR/<run>/`
short-circuits in `run_subset`. The cell below also pre-pulls any
prior results from the blob so a fresh kernel restart picks up where
the previous one left off.""")


# 14 — Full run
code("""\
from pathlib import Path

from arm2_pageindex.pipeline import run_subset

run_name = manifest_in.get("run_name", "default")
results_dir = Path(RESULTS_DIR) / run_name
results_dir.mkdir(parents=True, exist_ok=True)

# Resume: pull any prior per-query JSONs from blob.
prefix = f"{RESULTS_BLOB_PREFIX}/"
n_resumed = 0
for blob in container.list_blobs(name_starts_with=prefix):
    name = blob.name[len(prefix):]
    target = results_dir / name
    if target.exists():
        continue
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as f:
        f.write(container.get_blob_client(blob).download_blob().readall())
    n_resumed += 1
if n_resumed:
    print(f"Resumed: pulled {n_resumed} prior per-query JSONs from blob.")

results = run_subset(
    tree=tree,
    queries=queries,
    llm=llm,
    cfg=cfg,
    out_dir=results_dir,
    progress=True,
    log_every=10,
)
print(f"\\nDone — {len(results)} results in {results_dir}")
""")


# 15 — Upload results
md("## 9. Upload results to Blob")
code("""\
n_uploaded = 0
for path in sorted(results_dir.iterdir()):
    if not path.is_file():
        continue
    blob_name = f"{RESULTS_BLOB_PREFIX}/{path.name}"
    with path.open("rb") as f:
        container.get_blob_client(blob_name).upload_blob(f, overwrite=True)
    n_uploaded += 1

print(f"Uploaded {n_uploaded} files to {RESULTS_BLOB_PREFIX}/")
""")


# 15b — Stage results for local copy
md("""\
### Stage results for copy back to the local repo

Downloads per-query `q<qid>.json` files from blob into a directory on
the VM that mirrors the target layout in the local repo
(`RQ2_T05_ARM2_PAGEINDEX/data/<pdf-stem>/results/`). Independent of the
run — works on a fresh kernel that just has `container` and
`RESULTS_BLOB_PREFIX` defined. Set `PDF_STEM` per run.""")
code("""\
# Mirror layout: <pdf-stem>/results/q<qid>.json so a single scp -r
# drops the files straight into RQ2_T05_ARM2_PAGEINDEX/data/.
import re
from pathlib import Path

PDF_STEM = "2004_05_27_2004A27101"           # edit per run
LOCAL_TARGET_REL = f"RQ2_T05_ARM2_PAGEINDEX/data/{PDF_STEM}/results"
EXPORT_ROOT = Path("/home/azureuser/local_export")
EXPORT_DIR = EXPORT_ROOT / PDF_STEM / "results"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

per_query_re = re.compile(r"^q\\d+\\.json$")
prefix = f"{RESULTS_BLOB_PREFIX}/"

downloaded = []
for blob in container.list_blobs(name_starts_with=prefix):
    name = blob.name[len(prefix):]
    if not per_query_re.match(name):
        continue
    target = EXPORT_DIR / name
    with target.open("wb") as f:
        f.write(container.get_blob_client(blob).download_blob().readall())
    downloaded.append(target)

print(f"Staged {len(downloaded)} per-query JSONs at {EXPORT_DIR}:")
for p in sorted(downloaded):
    print(f"  {p}  ({p.stat().st_size:,} B)")

print()
print(f"Local target in repo: {LOCAL_TARGET_REL}/")
print()
print("To copy them down (replace <vm-ip>):")
print(f"  scp -r azureuser@<vm-ip>:{EXPORT_ROOT}/{PDF_STEM} \\\\")
print(f'         "<repo-root>/RQ2_T05_ARM2_PAGEINDEX/data/"')
print()
print("Or browse via the Azure portal file explorer.")
""")


# 16 — Cleanup
md("## 10. Cleanup\\n\\nKill the Ollama daemon to release GPU memory.")
code("""\
import subprocess
subprocess.run(["pkill", "-f", "ollama"], check=False)
print("Ollama stopped.")
""")


# ── Builder ───────────────────────────────────────────────────────────────────


def _to_source(s: str) -> list[str]:
    """Split a multi-line string into Jupyter's list-of-lines format.

    Each line keeps its trailing newline except the last (matches
    JupyterLab's save format)."""
    lines = s.splitlines(keepends=True)
    return lines


def _cell(kind: str, source: str) -> dict:
    base = {"cell_type": kind, "metadata": {}, "source": _to_source(source)}
    if kind == "code":
        base["execution_count"] = None
        base["outputs"] = []
    return base


def build() -> dict:
    return {
        "cells": [_cell(kind, src) for kind, src in CELLS],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> int:
    out = Path(__file__).resolve().parent / "azure_t05_pageindex_run.ipynb"
    out.write_text(
        json.dumps(build(), ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(f"Wrote {out} ({len(CELLS)} cells)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
