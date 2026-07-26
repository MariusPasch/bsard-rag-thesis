# T05 notebooks

Two notebook pairs, each with a Python builder as the canonical source
of truth and a generated `.ipynb` as the executable artefact. Both
artefacts are tracked in git; the `.gitattributes` policy at the repo
root resolves merge conflicts on `.ipynb` files via last-write-wins
(`merge=theirs`).

| Builder (source of truth) | Generated notebook | Where it runs |
|---|---|---|
| [`_build_azure_notebook.py`](_build_azure_notebook.py) | [`azure_t05_pageindex_run.ipynb`](azure_t05_pageindex_run.ipynb) | Azure ML compute (GPU) |
| [`_build_local_eval_notebook.py`](_build_local_eval_notebook.py) | [`local_t05_eval_and_compare.ipynb`](local_t05_eval_and_compare.ipynb) | Local machine (CPU) |

Cells are defined as in-source strings inside the builder. Hand-editing
the `.ipynb` JSON is fine for one-off tweaks but the generator is
canonical — re-running the builder regenerates the notebook from the
in-source strings. To regenerate either notebook:

```powershell
python notebooks\_build_azure_notebook.py
python notebooks\_build_local_eval_notebook.py
```

## Workflow (local build → Azure GPU run → local eval)

T05 runs **one PDF per launch** of the Azure notebook (5 launches total
for the curated 5-PDF set; see
[RQ2_T00_ORCHESTRATOR/data/selected_pdfs.json](../../RQ2_T00_ORCHESTRATOR/data/selected_pdfs.json)).
The local eval notebook is run **once at the end** over all 5.

### 1. Build the input bundles locally (one per PDF)

```powershell
cd "<repo>\RQ2_Structure_Aware_Retrieval\RQ2_T05_ARM2_PAGEINDEX"
$env:AZURE_CONTAINER_SAS_URL = "<container-sas-url>"

$stems = @(
    "1804_03_21_1804032150",
    "1867_06_08_1867060850",
    "1967_10_10_1967101055",
    "1967_10_10_1967101056",
    "2003_07_17_2013A31614"
)
foreach ($stem in $stems) {
    $run = "t04_$stem"
    python scripts\build_tree.py --doc-id $stem
    python scripts\prepare_azure_bundle.py `
        --gt "..\RQ2_T07_EVALUATION\ground_truth\runs\$run.json" `
        --doc-id $stem `
        --out "data\azure_bundles\$run" `
        --run-name $run
    python scripts\upload_to_blob.py upload `
        --file "data\azure_bundles\$run.zip" `
        --blob-name "t05_azure_bundles\$run.zip"
}
```

Each bundle is small (~150 KB – 1 MB) because the per-PDF tree is
deterministic JSON, not embeddings.

### 2. Provision an Azure ML compute instance with a GPU

Recommended SKU: **`Standard_NC4as_T4_v3`** (1× T4, 16 GB VRAM,
~$0.60/h). T05 runs LLaMA 3.1 8B in 4-bit on a T4 fine.

### 3. Open the Azure notebook on the compute instance

- `git clone` the T05 repo on the instance (the notebook is now tracked).
- Open `azure_t05_pageindex_run.ipynb` in JupyterLab via the Azure ML
  Studio web UI.
- **Edit cell 1**: set `RUN_NAME` (single source of truth for which PDF
  to run), `GITHUB_TOKEN`, `AZURE_CONTAINER_SAS_URL`.
- Execute cells top-to-bottom. Stop at the time-budget gate (cell 19),
  inspect the projection, then run cells 21 → 23 → 9b → 25 to complete
  the run.
- Repeat for the next PDF — only `RUN_NAME` changes between launches.

### 4. Pull results back to the local machine

Cell 23 uploads per-query JSONs to
`t05_results/<RUN_NAME>/q*.json` in the blob, and cell 9b stages them
under `/home/azureuser/local_export/<stem>/results/` for `scp`:

```powershell
scp -r azureuser@<vm-ip>:/home/azureuser/local_export/* `
       "<data root>\RQ2_T05_ARM2_PAGEINDEX\"
```

Or skip the `scp` — the local eval notebook can pull straight from blob.

### 5. Run the local eval notebook

After all 5 Azure runs complete, locally:

- Open `local_t05_eval_and_compare.ipynb`.
- In cell 2, set `AZURE_CONTAINER_SAS_URL` and toggle `DOWNLOAD_T05 = True`.
- **Run All**. Produces a combined headline table over all 5 PDFs, a
  per-PDF R@10 breakdown, cost summary, stratified analysis (by
  extraction_status, n_relevant_bsard_articles, and source PDF), pairwise
  significance, and a cost-vs-recall plot.

## Why the notebook design (vs a single shell script)

- **Granular start/stop**: each Azure launch is one PDF; if a particular
  PDF crashes or the SKU is reclaimed, you re-run only that PDF.
- **Resumable**: `run_subset` is idempotent per query — interrupting the
  full-run cell and re-running it picks up at the next un-saved query
  (the resume helper at the top of cell 21 also pulls any prior
  per-query JSONs back from blob, so a fresh kernel restart works).
- **Smoke gate**: the warmup + single-query + 5-question pilot (cells
  15–17) plus the time-budget cell (19) let you catch "no GPU" or
  "per-query time unexpectedly high" before committing the full ~2 h.
- **Visibility**: log lines stream into cell output as `run_subset`
  iterates, so you watch progress in real time.
