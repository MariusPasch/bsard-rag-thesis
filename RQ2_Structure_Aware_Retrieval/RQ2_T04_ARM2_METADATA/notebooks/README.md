# T04 Azure notebooks

Two paired artefacts, mirroring the
[T05 PageIndex notebooks structure](../../RQ2_T05_ARM2_PAGEINDEX/notebooks/):

- [`_build_azure_notebook.py`](_build_azure_notebook.py) — Python source of
  truth for the notebook. Re-run this to regenerate the `.ipynb`. Cells
  are defined as in-source strings; hand-editing the JSON is fine for
  one-off tweaks but the generator is canonical.
- [`azure_t04_precompute_run.ipynb`](azure_t04_precompute_run.ipynb) — the
  notebook itself. Open in Azure ML Studio (or Jupyter against a remote
  kernel) and execute cells manually.

## What the notebook does

Re-precomputes T04's dense + sparse indices for the 4 curated stems still
on `LINKER_VERSION = 3` (**doc 5, 6, 7, 9**; doc 8 already complete).
End-to-end on Azure GPU compute: GPU sanity check → clone
sibling repos → install CUDA torch + sentence-transformers → download an
input bundle from Azure Blob → idempotency helper → preflight → **smoke
build (= full SMOKE_PLAN on doc 5, measures per-variant wall times)** →
time-budget gate (extrapolates to other 3 stems via per-variant per-unit
rates) → per-stem precompute cells (idempotent — skip if stem already
complete under linker v4) → verify outputs → upload to Blob → stage
results for `scp` back to the local data root.

### Cell-level idempotency

Every per-stem cell checks `is_stem_complete(stem, cache_root)` before
invoking the script. If the stem has all 6 SMOKE_PLAN configs on disk
under linker v4 (with non-zero `bm25.pkl + faiss.index + faiss_meta.json
+ manifest.json` each), the cell prints `[skip]` and exits — no model
load, no script invocation. Partial state is also handled: if 5 of 6
variants are present, the script call rebuilds only the missing one
(via its existing `(unit, variant)`-level idempotency).

The smoke cell uses the same logic: if doc 5 is already complete AND
`<cache_root>/.smoke_timings.json` exists, it loads timings from that
file instead of re-running. This survives kernel restarts.

### Time estimate (cell 9)

The smoke cell parses per-variant wall times out of
`precompute_t04_indices.py`'s log lines (`[built] node/raw -> HASH ...
in 53.2s`) and writes them to `.smoke_timings.json`. The time-gate cell
divides each variant's wall by doc 5's unit count to get a per-unit
rate, then multiplies by each other stem's (n_nodes / n_articles) to
project the remaining wall.

More accurate than a single linear extrapolation because the 6
variants have very different per-unit costs (node/summary truncates
~44 % of entries, article/full has the biggest aggregated text, etc.).

Estimated wall time (just the precompute, excluding setup):

| GPU SKU | Per stem | 4-stem total |
|---|---:|---:|
| T4 (`NC4as_T4_v3`) | ~30-40 min | ~2-2.5 h |
| V100 (`NC6s_v3`) | ~20 min | ~1.5 h |
| A100 (`NC24ads_A100_v4`) | ~16 min | ~1 h |

Add ~1 h on the first run for env setup + bundle upload/download.

## Workflow (local → Azure GPU → local)

### 1. Build the input bundle on your local machine

```powershell
cd "<repo>\RQ2_Structure_Aware_Retrieval"
python RQ2_T04_ARM2_METADATA\scripts\prepare_azure_bundle.py
# wrote RQ2_T04_ARM2_METADATA\bundle.zip (~430 MB)
```

The bundle contains everything the precompute needs:
`azuredi/` (the 308 MB VectorDB + small CSVs), the BSARD corpus DB,
`pdf_document_map.csv`, and the 4 PDFs. The PDFs matter because the
script uses their SHA-256 to fingerprint the cache hash — without them
the Azure-built caches would have a different hash than your local
machine expects.

### 2. Upload the bundle to Azure Blob (one-time)

```powershell
azcopy copy "RQ2_T04_ARM2_METADATA\bundle.zip" `
       "<container-sas-url>/t04_azure_bundles/v4_remaining4.zip"
```

Or use Azure Storage Explorer / the portal UI.

### 3. Provision an Azure ML compute instance with a GPU

Recommended SKU for cost-effectiveness: **`Standard_NC4as_T4_v3`**
(1× T4, 16 GB VRAM, ~$0.60/h). For fastest turnaround: A100. Anything
smaller than a T4 will probably be slower than just running on your
local machine.

### 4. Open the notebook on the compute instance and run cells

- Upload `azure_t04_precompute_run.ipynb` to the instance (or `git
  clone` the T04 repo there — the notebook is a tracked file).
- Open it in JupyterLab via the Azure ML Studio web UI.
- **Edit cell 1**: set `GITHUB_TOKEN` (read-scope PAT), `GITHUB_OWNER`,
  `AZURE_CONTAINER_SAS_URL`.
- Execute cells top-to-bottom. The per-stem precompute cells (15-18 in
  the generated notebook) can be re-run individually if a particular
  stem fails or you want to monitor progress.

### 5. Pull results back to the local data root

Cell 16 stages the new `<stem>/configs/<v4-hash>/` directories under
`/home/azureuser/local_export/` on the VM. Download via:

```powershell
# From your local machine (replace <vm-ip>):
scp -r azureuser@<vm-ip>:/home/azureuser/local_export/* `
       "<RQ2_DATA_DIR>\RQ2_T04_ARM2_METADATA\data\"
```

Or use the Azure ML Studio file explorer. Cell 15 also uploads to Blob
under `RESULTS_BLOB_PREFIX`, so `azcopy sync` from the blob works too.

### 6. On your local machine: verify + run the unified 5-stem comparison

```powershell
cd "<repo>\RQ2_Structure_Aware_Retrieval"
# Cache-hit verify the 4 stems
$stems = @(
    "1967_10_10_1967101056",
    "1867_06_08_1867060850",
    "1804_03_21_1804032150",
    "1967_10_10_1967101055"
)
foreach ($s in $stems) {
    python RQ2_T04_ARM2_METADATA\scripts\precompute_t04_indices.py --doc-id $s -v 2>&1 |
        Select-String "cache hit|building|FAIL"
}
# Run the full 5-stem comparison sweep
$all_stems = $stems + @("2003_07_17_2013A31614")
foreach ($s in $all_stems) {
    python RQ2_T04_ARM2_METADATA\scripts\compare_t03_vs_t04.py --doc-id $s --smoke -v
}
```

## Why the notebook design (vs a single shell script)

- **Granular start/stop**: each per-stem cell is independent. You can
  start doc 5, monitor progress in the cell output, and only proceed
  to doc 6 when satisfied.
- **Resumable**: `precompute_t04_indices.py` is idempotent per
  (unit, variant) — interrupting a stem and re-running its cell picks
  up at the next un-built variant.
- **Smoke gate**: the single-variant smoke (cell 13) + time-budget
  cell (cell 14) let you catch "no GPU detected" or "per-unit time
  unexpectedly high" before committing to the full run.
- **Visibility**: log lines stream into cell output as the script runs,
  so you can watch the embedding pass progress in real time.
