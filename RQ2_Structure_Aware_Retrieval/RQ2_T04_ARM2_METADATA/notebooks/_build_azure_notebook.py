"""Generate ``azure_t04_precompute_run.ipynb`` from in-source cell strings.

Re-run this script whenever a cell's content changes. Hand-editing the
.ipynb directly is fine for ad-hoc tweaks but the generator stays the
source of truth.

    python notebooks/_build_azure_notebook.py

Mirrors the structure of RQ2_T05_ARM2_PAGEINDEX/notebooks/_build_azure_notebook.py.

Design properties:
- **Per-stem cell-level idempotency** — each stem cell checks for a complete
  set of 6 SMOKE_PLAN configs under linker v4 in the cache_root before
  invoking the script; fully-built stems are skipped silently.
- **Per-variant timing** — the smoke cell builds the smallest stem
  (doc 5) end-to-end and persists per-variant wall times to
  ``<cache_root>/.smoke_timings.json``. The time-gate cell uses those
  per-variant per-unit rates to project the other 3 stems.
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
# T04 Arm 2A (metadata) — Azure GPU precompute, remaining 4 stems

Re-precompute the dense + sparse indices for **doc 5, 6, 7, 9** under
`LINKER_VERSION = 4` on an Azure ML compute instance with a CUDA GPU.
Doc 8 is already complete under v4.

**Properties:**
- Per-stem cells let you start/stop/monitor each PDF independently.
- Cell-level idempotency: each stem cell skips entirely if its 6
  SMOKE_PLAN configs are already on disk under linker v4 (no script
  invocation, no model load). Safe to re-run cells after kernel
  restarts.
- The script (`precompute_t04_indices.py`) is also idempotent at the
  (unit, variant) granularity — interrupting a stem mid-build and
  re-running its cell picks up at the next un-built variant.

**Sequence:** GPU sanity → clone repos → pip deps → download bundle →
preflight → **smoke (= full doc 5 build, measures per-variant times)**
→ time gate (extrapolates the other 3 stems) → per-stem cells (4) →
verify → upload → stage for `scp` back to the local data root → cleanup.
""")


# 1 — Config
code("""\
# Edit these for the run.
GITHUB_TOKEN = ""                                # PAT with read scope on the 5 sibling repos
GITHUB_OWNER = "MariusPasch"
GITHUB_BRANCH = "master"

# Five sibling repos required at runtime (T01–T04 + T07).
SIBLING_REPOS = {
    "T01_SHARED":      "RQ2_T01_SHARED",
    "T02_DATA_LOADER": "RQ2_T02_DATA_LOADER",
    "T03_ARM1_NAIVE":  "RQ2_T03_ARM1_NAIVE",
    "T04_ARM2_METADATA": "RQ2_T04_ARM2_METADATA",
    "T07_EVALUATION":  "RQ2_T07_EVALUATION",
}

# Container-level SAS URL with read+write+list. Bundle must already exist.
AZURE_CONTAINER_SAS_URL = ""
BUNDLE_BLOB_NAME = "t04_azure_bundles/v4_remaining4.zip"
RESULTS_BLOB_PREFIX = "t04_results/v4_remaining4"

# Paths on the Azure compute instance.
REPOS_DIR = "/home/azureuser/repos"
BUNDLE_DIR = "/home/azureuser/bundle"
RESULTS_DIR = "/home/azureuser/results"

# Stems to process. Doc 5 is smallest → used as the smoke target.
# After the smoke cell doc 5 is fully built, and the doc 5 per-stem cell
# below will skip via the idempotency check.
STEMS = [
    "1967_10_10_1967101056",   # doc 5 — Code Judiciaire (smaller)  ← smoke target
    "1867_06_08_1867060850",   # doc 6 — Code Pénal
    "1804_03_21_1804032150",   # doc 9 — Code Civil
    "1967_10_10_1967101055",   # doc 7 — Code Judiciaire (larger)
]
SMOKE_STEM = STEMS[0]

# SMOKE_PLAN — must match arm2_metadata SMOKE_PLAN exactly (used for
# both the smoke and the completion check). Two units × variants split:
#   4 node variants + 2 article variants = 6 configs per stem.
SMOKE_PLAN = [
    ("node",    "raw"),
    ("node",    "enriched"),
    ("node",    "summary"),
    ("node",    "full"),
    ("article", "raw"),
    ("article", "full"),
]

# HF model id (must match your local machine — part of compute_config_hash).
EMBEDDING_MODEL = "intfloat/multilingual-e5-large-instruct"

# Workload per stem (indexable nodes, indexable articles). Used by the
# time-gate cell to scale per-variant per-unit times. Source: AzureDI
# dump post-2026-05-20 corpus (see CHANGE_NOTES.md, 2026-05-25/27 entries).
WORKLOAD = {
    "1967_10_10_1967101056": ( 619,  294),   # doc 5
    "1867_06_08_1867060850": ( 680,  380),   # doc 6
    "1804_03_21_1804032150": ( 958,  422),   # doc 9
    "1967_10_10_1967101055": (1616,  762),   # doc 7
}

# Where smoke per-variant timings are persisted (so the time-gate cell
# survives kernel restarts and so a re-run of the smoke cell on cache-hit
# data falls back to the original measurements).
SMOKE_TIMINGS_FILE = ".smoke_timings.json"
""")


# 2 — GPU check
md("""\
## 1. GPU sanity check

Bail early if there is no CUDA device — the whole point of moving to Azure
is to use the GPU. The script would still run on CPU but at your local
machine's rate (~67 h for the 4 stems).""")
code("""\
import subprocess
import sys

r = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
print(r.stdout if r.returncode == 0 else r.stderr)
if r.returncode != 0:
    raise RuntimeError("nvidia-smi failed — no GPU on this compute instance.")

import importlib
try:
    torch = importlib.import_module("torch")
    print(f"torch:                {torch.__version__}")
    print(f"torch.cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"torch.cuda.device(0): {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING — torch sees no CUDA. Re-install with the CUDA wheel "
              "via cell 5 (pip install). Re-run this cell after install.")
except ImportError:
    print("torch not installed yet — that's fine, cell 5 will install it.")
""")


# 3 — Clone sibling repos
md("""\
## 2. Clone the 5 sibling repos

Pulls T01–T04 + T07 from GitHub. Validates that any pre-existing dir's
`origin` matches the configured URL, refreshes the embedded PAT, and
hard-asserts `LINKER_VERSION = 4` is present in T04 before continuing.""")
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
        normalized = existing.split("@github.com", 1)[-1]
        expected_path = expected_url.split("github.com", 1)[-1]
        if not normalized.endswith(expected_path):
            raise RuntimeError(
                f"{target} exists but origin = {existing!r} (expected "
                f"{expected_url!r}). Either rm -rf {target} or fix "
                f"SIBLING_REPOS['{label}']."
            )
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

ret_path = repo_paths["T04_ARM2_METADATA"] / "src" / "arm2_metadata" / "retriever.py"
content = ret_path.read_text(encoding="utf-8")
if "LINKER_VERSION = 4" not in content:
    raise RuntimeError(
        f"{ret_path} does NOT contain LINKER_VERSION = 4. Stop and fix "
        "before continuing — running under v3 would build useless caches."
    )
print("\\nLINKER_VERSION = 4 verified in T04.")
""")


# 4 — pip install
md("""\
## 3. Install Python packages

CUDA-enabled PyTorch via the cu121 extra-index-url, then the standard
T04 dependencies, then editable installs of the 5 sibling packages.""")
code("""\
import subprocess
import sys

PIP = [sys.executable, "-m", "pip", "install", "-q"]

import importlib
need_cuda_torch = True
try:
    torch = importlib.import_module("torch")
    if torch.cuda.is_available():
        need_cuda_torch = False
        print(f"  CUDA torch already installed ({torch.__version__})")
except ImportError:
    pass

if need_cuda_torch:
    print("  installing CUDA torch from cu121 wheel index...")
    subprocess.run(
        PIP + ["--extra-index-url", "https://download.pytorch.org/whl/cu121",
               "torch>=2.0"],
        check=True,
    )

deps = [
    "numpy>=1.26", "pandas>=2.0", "tqdm>=4.66",
    "sentence-transformers>=2.7", "transformers>=4.40",
    "faiss-cpu>=1.8", "rank-bm25>=0.2",
    "PyMuPDF",
    "azure-storage-blob",
]
subprocess.run(PIP + deps, check=True)

INSTALL_ORDER = ["T01_SHARED", "T02_DATA_LOADER", "T03_ARM1_NAIVE",
                 "T04_ARM2_METADATA", "T07_EVALUATION"]
for label in INSTALL_ORDER:
    target = repo_paths[label]
    subprocess.run(PIP + ["-e", str(target)], check=True)
    print(f"  installed {label} from {target}")

for label, target in repo_paths.items():
    src_path = str(target / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

from arm2_metadata.retriever import LINKER_VERSION
assert LINKER_VERSION == 4, f"runtime LINKER_VERSION = {LINKER_VERSION}, expected 4"
print(f"\\nAll packages installed; arm2_metadata.LINKER_VERSION = {LINKER_VERSION}")
""")


# 5 — Download bundle
md("""\
## 4. Download the input bundle from Azure Blob

Built locally with `scripts/prepare_azure_bundle.py` and uploaded once.
Idempotent — skipped if `bundle.zip` is already on disk.

**Bundle contents (~430 MB):** `azuredi/` (308 MB JSON + small CSVs),
`dataset_creation_output/bsard_corpus.db`, `pdf_document_map.csv`, and
the 4 source PDFs (needed for the pdf_sha256 fingerprint).""")
code("""\
import zipfile
from pathlib import Path

from azure.storage.blob import ContainerClient

bundle_dir = Path(BUNDLE_DIR)
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
print(f"Bundle zip: {local_zip} ({local_zip.stat().st_size/1024/1024:.1f} MB)")

with zipfile.ZipFile(local_zip) as zf:
    zf.extractall(bundle_dir)

print("\\nBundle contents:")
n_files = 0
total = 0
for p in sorted(bundle_dir.rglob("*")):
    if p.is_file() and p != local_zip:
        rel = p.relative_to(bundle_dir)
        size = p.stat().st_size
        total += size
        n_files += 1
        if n_files <= 20 or rel.suffix in (".csv", ".db"):
            print(f"  {str(rel):<55}  {size:>12,} B")
print(f"\\n  ({n_files} files, {total/1024/1024:.1f} MB total)")
""")


# 6 — Wire bundle paths
md("""\
## 5. Wire bundle paths into the T04 layout

`precompute_t04_indices.py` resolves paths via CLI flags. We construct a
`PATHS` dict here that's used by every downstream cell — so all inputs
come from the bundle, not the repo's own (empty on Azure) directories.""")
code("""\
from pathlib import Path

bundle = Path(BUNDLE_DIR)
PATHS = {
    "azuredi_dir":  bundle / "azuredi",
    "pdf_doc_map":  bundle / "pdf_document_map.csv",
    "bsard_db":     bundle / "dataset_creation_output" / "bsard_corpus.db",
    "pdf_dir":      bundle / "pdfs",
    "cache_root":   Path(RESULTS_DIR),
}
PATHS["cache_root"].mkdir(parents=True, exist_ok=True)

required = ["azuredi_dir", "pdf_doc_map", "bsard_db", "pdf_dir"]
missing = [k for k in required if not PATHS[k].exists()]
if missing:
    raise FileNotFoundError(f"Missing bundle inputs: {missing}")

for stem in STEMS:
    pdf = PATHS["pdf_dir"] / f"{stem}.pdf"
    if not pdf.exists():
        raise FileNotFoundError(f"Bundle missing PDF: {pdf}")
    print(f"  {stem}.pdf  {pdf.stat().st_size/1024:.1f} KB")

print(f"\\nResults will be written to: {PATHS['cache_root']}")
""")


# 7 — Idempotency helper (defined here so smoke + per-stem cells can use it)
md("""\
## 6. Idempotency helper — `is_stem_complete()`

Returns True when every (unit, variant) in `SMOKE_PLAN` for `stem` has
a config directory under `cache_root/<stem>/configs/` whose manifest
declares `linker_version == 4` AND whose four required files are
non-zero. Used by the smoke cell to short-circuit if doc 5 is already
built, and by each per-stem cell to skip the script entirely when the
stem is done.""")
code("""\
import json
from pathlib import Path

REQUIRED_FILES = ["bm25.pkl", "faiss.index", "faiss_meta.json", "manifest.json"]


def is_stem_complete(stem: str, cache_root: Path, smoke_plan=SMOKE_PLAN) -> bool:
    \"\"\"True iff cache_root/<stem>/configs/ holds a valid config dir
    (linker_version=4, all 4 required files non-zero) for every (unit,
    variant) in smoke_plan.\"\"\"
    cfg_dir = Path(cache_root) / stem / "configs"
    if not cfg_dir.exists():
        return False
    found_pairs = set()
    for sub in cfg_dir.iterdir():
        if not sub.is_dir():
            continue
        manifest = sub / "manifest.json"
        if not manifest.exists():
            continue
        try:
            m = json.loads(manifest.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        params = m.get("chunking_params") or {}
        if params.get("linker_version") != 4 or params.get("arm") != "2a":
            continue
        # All required files present and non-zero?
        ok = all(
            (sub / f).exists() and (sub / f).stat().st_size > 0
            for f in REQUIRED_FILES
        )
        if not ok:
            continue
        found_pairs.add((params.get("unit"), params.get("variant")))
    needed = {tuple(p) for p in smoke_plan}
    return needed.issubset(found_pairs)


def stem_completion_status(stem: str, cache_root: Path, smoke_plan=SMOKE_PLAN) -> dict:
    \"\"\"Diagnostic: which (unit, variant) pairs are built? Useful for
    spotting partial state without running the precompute.\"\"\"
    cfg_dir = Path(cache_root) / stem / "configs"
    needed = {tuple(p) for p in smoke_plan}
    built = set()
    if cfg_dir.exists():
        for sub in cfg_dir.iterdir():
            if not sub.is_dir():
                continue
            manifest = sub / "manifest.json"
            if not manifest.exists():
                continue
            try:
                m = json.loads(manifest.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            params = m.get("chunking_params") or {}
            if params.get("linker_version") == 4 and params.get("arm") == "2a":
                ok = all((sub / f).exists() and (sub / f).stat().st_size > 0
                         for f in REQUIRED_FILES)
                if ok:
                    built.add((params.get("unit"), params.get("variant")))
    return {
        "built": sorted(built),
        "missing": sorted(needed - built),
        "n_built": len(built & needed),
        "n_needed": len(needed),
        "complete": needed.issubset(built),
    }


# Show current status for every target stem (mostly empty on a fresh VM).
print("Current per-stem completion status:")
for s in STEMS:
    status = stem_completion_status(s, PATHS["cache_root"])
    print(f"  {s}: {status['n_built']}/{status['n_needed']} variants built"
          f"{' (COMPLETE)' if status['complete'] else ''}")
""")


# 8 — Preflight
md("""\
## 7. Pre-flight (check_precompute_requirements.py)

Loads the AzureDI corpus, runs the BSARD linker, prints per-doc
coverage. Under linker v4 expect higher distinct-bsard_id counts than
v3 (doc 5: 308; doc 6: 410; doc 7: 772; doc 9: 434).""")
code("""\
import subprocess
import sys

T04_REPO = repo_paths["T04_ARM2_METADATA"]
script = T04_REPO / "scripts" / "check_precompute_requirements.py"
cmd = [
    sys.executable, str(script),
    "--azuredi-dir", str(PATHS["azuredi_dir"]),
    "--pdf-doc-map", str(PATHS["pdf_doc_map"]),
    "--bsard-db",    str(PATHS["bsard_db"]),
    "--pdf-dir",     str(PATHS["pdf_dir"]),
    "--cache-root",  str(PATHS["cache_root"]),
]
print(" ".join(cmd))
r = subprocess.run(cmd)
if r.returncode != 0:
    raise RuntimeError("Preflight failed — see output above.")
""")


# 9 — Smoke = full doc 5 build + per-variant timings
md("""\
## 8. Smoke — full SMOKE_PLAN build on doc 5

Builds **all 6 SMOKE_PLAN variants on the smallest stem (doc 5)** and
times each one independently via a single `precompute_t04_indices.py`
call. The per-variant wall times are extracted from the script's log
and persisted to `<cache_root>/.smoke_timings.json` so the time-gate
cell below survives kernel restarts.

If doc 5 is already complete (this notebook was re-run after a kernel
restart), the smoke cell loads timings from
`<cache_root>/.smoke_timings.json` instead and only re-runs the script
when the timings file is missing.

Expected wall on first run (fresh build): ~6 min on T4, ~3 min on A100.
Cache-hit re-run: ~10 seconds.""")
code("""\
import json
import re
import subprocess
import sys
import time
from pathlib import Path

timings_path = PATHS["cache_root"] / SMOKE_TIMINGS_FILE
smoke_complete = is_stem_complete(SMOKE_STEM, PATHS["cache_root"])
timings: dict[str, float] = {}

if smoke_complete and timings_path.exists():
    timings = json.loads(timings_path.read_text(encoding="utf-8"))
    print(f"[skip] {SMOKE_STEM} already complete; loaded timings from {timings_path}")
else:
    if smoke_complete and not timings_path.exists():
        print(f"[note] {SMOKE_STEM} already complete but no timings file — "
              "re-running smoke for measurements (variants will [cache hit] fast).")
    cmd = [
        sys.executable, str(T04_REPO / "scripts" / "precompute_t04_indices.py"),
        "--doc-id",          SMOKE_STEM,
        "--pdf-dir",         str(PATHS["pdf_dir"]),
        "--azuredi-dir",     str(PATHS["azuredi_dir"]),
        "--pdf-doc-map",     str(PATHS["pdf_doc_map"]),
        "--bsard-db",        str(PATHS["bsard_db"]),
        "--cache-root",      str(PATHS["cache_root"]),
        "--embedding-model", EMBEDDING_MODEL,
        "-v",
    ]
    print(" ".join(cmd))
    t0 = time.perf_counter()
    r = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.perf_counter() - t0
    print(r.stdout[-4000:] if r.stdout else "")
    if r.returncode != 0:
        print(r.stderr[-2000:])
        raise RuntimeError("Smoke build failed.")

    # Parse per-variant timings from the script's log.
    # Match: "[built]    node/raw -> 261f1a1cd37b   n_units=619  truncated=12 (1.9%)  in 53.2s"
    pat = re.compile(
        r"\\[built\\]\\s+(?P<unit>node|article)/(?P<variant>\\w+)\\s+->\\s+\\S+.*?in\\s+(?P<sec>[\\d.]+)s",
        re.IGNORECASE,
    )
    log = (r.stdout or "") + "\\n" + (r.stderr or "")
    for m in pat.finditer(log):
        key = f"{m.group('unit')}/{m.group('variant')}"
        timings[key] = float(m.group("sec"))

    # Anything we didn't find in the log probably cache-hit (already on disk
    # before this cell ran). Record 0.0 for those so the time-gate can warn.
    for unit, variant in SMOKE_PLAN:
        key = f"{unit}/{variant}"
        timings.setdefault(key, 0.0)

    timings_path.parent.mkdir(parents=True, exist_ok=True)
    timings_path.write_text(json.dumps(timings, indent=2), encoding="utf-8")
    print(f"\\nSmoke wall: {wall:.1f}s (sum-of-variants {sum(timings.values()):.1f}s)")
    print(f"Persisted per-variant timings -> {timings_path}")

print("\\nPer-variant times on", SMOKE_STEM, ":")
for unit, variant in SMOKE_PLAN:
    key = f"{unit}/{variant}"
    s = timings.get(key, 0.0)
    marker = " (cache-hit — not representative)" if s < 1.0 else ""
    print(f"  {key:<20s}  {s:>7.1f} s{marker}")
""")


# 10 — Time-budget gate
md("""\
## 9. Time-budget gate

Uses the per-variant wall times measured on doc 5 to project the other
3 stems. Per-variant per-unit rate × per-stem unit count = projected
wall. Inspect the projection below; **only run the per-stem cells
once the numbers look acceptable.**""")
code("""\
n_nodes_doc5, n_articles_doc5 = WORKLOAD[SMOKE_STEM]

per_unit_s: dict[str, float] = {}
for unit, variant in SMOKE_PLAN:
    key = f"{unit}/{variant}"
    units_doc5 = n_nodes_doc5 if unit == "node" else n_articles_doc5
    per_unit_s[key] = timings.get(key, 0.0) / max(units_doc5, 1)

print("Per-variant per-unit rates (s/unit, from doc 5 smoke):")
for k, v in per_unit_s.items():
    print(f"  {k:<20s}  {v*1000:>7.1f} ms/unit")

print()
print("Projected per-stem wall (sum across 6 variants):")
total_remaining = 0.0
for s in STEMS:
    n_n, n_a = WORKLOAD[s]
    stem_total = 0.0
    for unit, variant in SMOKE_PLAN:
        key = f"{unit}/{variant}"
        units = n_n if unit == "node" else n_a
        stem_total += per_unit_s[key] * units
    if s == SMOKE_STEM:
        marker = "  [done in smoke]"
    else:
        total_remaining += stem_total
        marker = ""
    print(f"  {s}:  ~{stem_total/60:>6.1f} min  ({stem_total:>6.0f}s){marker}")

print(f"\\nProjected REMAINING wall (doc 6 + doc 9 + doc 7): ~{total_remaining/60:.1f} min")
print(f"                                                       ({total_remaining/3600:.2f} h)")
print()
print("Inspect — then run the per-stem cells below.")
""")


# 11 — Per-stem cells header
md("""\
## 10. Per-stem precompute (4 cells, idempotent)

Each cell runs `precompute_t04_indices.py` for one stem. **Each cell
short-circuits with `[skip]` if the stem is already fully built under
linker v4** (no script invocation, no model load) — so re-running the
notebook after a kernel restart safely skips finished stems.

The doc 5 cell will skip immediately on a normal run because the smoke
cell above already built it. It's kept for symmetry / explicit "doc 5
is done" reporting.""")


def _stem_cell(stem: str, label: str, est_minutes_t4: int, est_minutes_a100: int) -> None:
    """Generate an idempotent per-stem precompute cell."""
    code(f"""\
# {label}  — est ~{est_minutes_t4} min on T4, ~{est_minutes_a100} min on A100
import subprocess
import sys
import time

STEM = "{stem}"
status = stem_completion_status(STEM, PATHS["cache_root"])

if status["complete"]:
    print(f"[skip] {{STEM}}: all {{status['n_needed']}}/{{status['n_needed']}} "
          "SMOKE_PLAN configs already built under linker v4 "
          f"(at {{PATHS['cache_root'] / STEM / 'configs'}})")
else:
    if status["n_built"] > 0:
        print(f"[partial] {{STEM}}: {{status['n_built']}}/{{status['n_needed']}} "
              f"variants already built; will rebuild only missing {{status['missing']}}")
    cmd = [
        sys.executable, str(repo_paths["T04_ARM2_METADATA"] / "scripts" / "precompute_t04_indices.py"),
        "--doc-id",          STEM,
        "--pdf-dir",         str(PATHS["pdf_dir"]),
        "--azuredi-dir",     str(PATHS["azuredi_dir"]),
        "--pdf-doc-map",     str(PATHS["pdf_doc_map"]),
        "--bsard-db",        str(PATHS["bsard_db"]),
        "--cache-root",      str(PATHS["cache_root"]),
        "--embedding-model", EMBEDDING_MODEL,
        "-v",
    ]
    print(" ".join(cmd))
    t0 = time.perf_counter()
    r = subprocess.run(cmd)
    dt = time.perf_counter() - t0
    print(f"\\n=== {{STEM}} done: wall {{dt/60:.1f}} min  exit={{r.returncode}} ===")
    if r.returncode != 0:
        raise RuntimeError(f"precompute failed on {{STEM}}.")
    # Post-run sanity: stem must now report complete.
    if not is_stem_complete(STEM, PATHS["cache_root"]):
        post = stem_completion_status(STEM, PATHS["cache_root"])
        raise RuntimeError(
            f"{{STEM}} ran but is_stem_complete=False; missing={{post['missing']}}"
        )
""")


_stem_cell("1967_10_10_1967101056", "Doc 5 — Code Judiciaire (smaller)",  6, 4)
_stem_cell("1867_06_08_1867060850", "Doc 6 — Code Pénal",                 7, 5)
_stem_cell("1804_03_21_1804032150", "Doc 9 — Code Civil",                10, 6)
_stem_cell("1967_10_10_1967101055", "Doc 7 — Code Judiciaire (larger)",  17, 11)


# 12 — Verify
md("""\
## 11. Verify all 4 stems are now complete

Cross-checks every stem via `is_stem_complete()`. Fails loudly if any
stem is short of the 6 SMOKE_PLAN configs.""")
code("""\
all_good = True
for stem in STEMS:
    status = stem_completion_status(stem, PATHS["cache_root"])
    if status["complete"]:
        cfg_dir = PATHS["cache_root"] / stem / "configs"
        dirs = [d for d in cfg_dir.iterdir() if d.is_dir()]
        print(f"  [{stem}]  OK  ({len(dirs)} dirs)")
    else:
        print(f"  [{stem}]  INCOMPLETE  built={status['n_built']}/{status['n_needed']}  missing={status['missing']}")
        all_good = False

if not all_good:
    raise RuntimeError("One or more stems are incomplete — investigate before upload.")
print("\\nAll 4 stems verified complete under linker v4.")
""")


# 13 — Upload to Blob
md("""\
## 12. Upload results to Azure Blob

Mirrors the layout `<stem>/configs/<v4-hash>/<file>` under
`RESULTS_BLOB_PREFIX` so a single `azcopy sync` into the local data
root picks up the whole tree.""")
code("""\
n_uploaded = 0
total_bytes = 0
for stem in STEMS:
    cfg_dir = PATHS["cache_root"] / stem / "configs"
    for sub in cfg_dir.iterdir():
        if not sub.is_dir():
            continue
        for path in sub.iterdir():
            if not path.is_file():
                continue
            blob_name = f"{RESULTS_BLOB_PREFIX}/{stem}/configs/{sub.name}/{path.name}"
            with path.open("rb") as f:
                container.get_blob_client(blob_name).upload_blob(f, overwrite=True)
            n_uploaded += 1
            total_bytes += path.stat().st_size

print(f"Uploaded {n_uploaded} files ({total_bytes/1024/1024:.1f} MB) under {RESULTS_BLOB_PREFIX}/")
""")


# 14 — Stage for scp back
md("""\
## 13. Stage results for `scp` back to the local data root

Builds a directory tree on the VM that mirrors the local
`RQ2_T04_ARM2_METADATA/data/<stem>/configs/<v4-hash>/` layout, so a
single `scp -r` drops the new configs straight into the local data
root.""")
code("""\
import shutil
from pathlib import Path

EXPORT_ROOT = Path("/home/azureuser/local_export")
if EXPORT_ROOT.exists():
    shutil.rmtree(EXPORT_ROOT)
EXPORT_ROOT.mkdir(parents=True)

n_files = 0
for stem in STEMS:
    src = PATHS["cache_root"] / stem / "configs"
    dst = EXPORT_ROOT / stem / "configs"
    dst.mkdir(parents=True)
    for sub in src.iterdir():
        if not sub.is_dir():
            continue
        target = dst / sub.name
        shutil.copytree(sub, target)
        n_files += sum(1 for _ in target.rglob("*") if _.is_file())

print(f"Staged {n_files} files at {EXPORT_ROOT}")
print()
print("To copy them down (replace <vm-ip> and <repo-root>):")
print(f"  scp -r azureuser@<vm-ip>:{EXPORT_ROOT}/. \\\\")
print(f'         "<repo-root>/RQ2_T04_ARM2_METADATA/data/"')
""")


# 15 — Cleanup
md("""\
## 14. Cleanup

Free CUDA memory and shut down. No background daemon to kill (T04
doesn't use Ollama, unlike T05).""")
code("""\
import gc
import importlib

gc.collect()
try:
    torch = importlib.import_module("torch")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("CUDA cache cleared.")
except ImportError:
    pass
print("Done.")
""")


# ── Builder ──────────────────────────────────────────────────────────────────


def _to_source(s: str) -> list[str]:
    return s.splitlines(keepends=True)


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
    out = Path(__file__).resolve().parent / "azure_t04_precompute_run.ipynb"
    out.write_text(
        json.dumps(build(), ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(f"Wrote {out} ({len(CELLS)} cells)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
