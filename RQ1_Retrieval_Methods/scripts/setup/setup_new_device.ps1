<#
.SYNOPSIS
    One-time setup for the BSARD RQ1 project after cloning.

.DESCRIPTION
    Run this script once after cloning the repository. It will:
      1. Create a Python virtual environment
      2. Install all dependencies from requirements.txt
      3. Install the bsard_evaluation package (sibling RQ3_Autonomous_Evaluation component)
      4. Download the spaCy French model
      5. Download required NLTK data
      6. Download the data bundle into the local data root

    Large data artefacts (corpus DB, parquet exports, embeddings, FAISS indices,
    result JSONs) are not in git. They live in the companion Hugging Face dataset
    `mpaschalidis/bsard-rag-thesis-data` and download into a local gitignored data
    root. The RQ1 data root is the `BSARD_DATA_DIR` environment variable, defaulting
    to `<repo>/output`.

    Prerequisites:
      - Python 3.10+ on PATH
      - The sibling RQ3_Autonomous_Evaluation component present (for bsard_evaluation)

.PARAMETER DataDir
    Directory to use as the data root. The download step populates it, and
    experiment scripts read/write it (set BSARD_DATA_DIR to point here at run time).
    Default: "<repo>/output".

.PARAMETER RQ3ProjectPath
    Path to the sibling RQ3_Autonomous_Evaluation component.
    Required for installing the bsard_evaluation package.
    Default: sibling directory "RQ3_Autonomous_Evaluation" relative to this project.

.EXAMPLE
    # Run from the project root with defaults:
    .\scripts\setup\setup_new_device.ps1

.EXAMPLE
    # Use a custom data root and a non-standard RQ3 location:
    .\scripts\setup\setup_new_device.ps1 `
        -DataDir "D:\bsard_data" `
        -RQ3ProjectPath "C:\Users\you\Projects\RQ3_Autonomous_Evaluation"
#>

param(
    [string]$DataDir,
    [string]$RQ3ProjectPath = (Join-Path (Split-Path $PSScriptRoot -Parent | Split-Path -Parent) "RQ3_Autonomous_Evaluation")
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot ".." "..")
Set-Location $ProjectRoot

if (-not $DataDir) {
    $DataDir = Join-Path $ProjectRoot "output"
}

Write-Host ""
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host " BSARD RQ1 — Setup" -ForegroundColor Cyan
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host " Project root : $ProjectRoot"
Write-Host " Data root    : $DataDir"
Write-Host " RQ3 project  : $RQ3ProjectPath"
Write-Host ""

# ── Step 1: Create virtual environment ────────────────────────────────────────
Write-Host "[1/6] Setting up Python virtual environment..." -ForegroundColor Yellow
$VenvPath = Join-Path $ProjectRoot ".venv"

if (Test-Path $VenvPath) {
    Write-Host "    .venv already exists — skipping creation." -ForegroundColor Green
} else {
    python -m venv $VenvPath
    Write-Host "    Created .venv" -ForegroundColor Green
}

$Pip    = Join-Path $VenvPath "Scripts\pip.exe"
$Python = Join-Path $VenvPath "Scripts\python.exe"

# ── Step 2: Install dependencies ──────────────────────────────────────────────
Write-Host "[2/6] Installing dependencies from requirements.txt..." -ForegroundColor Yellow
& $Pip install --upgrade pip --quiet
& $Pip install -r (Join-Path $ProjectRoot "requirements.txt")
Write-Host "    requirements.txt installed." -ForegroundColor Green

# ── Step 3: Install bsard_evaluation (sibling RQ3 component) ───────────────────
Write-Host "[3/6] Installing bsard_evaluation package (RQ3)..." -ForegroundColor Yellow
if (-not (Test-Path $RQ3ProjectPath)) {
    Write-Warning @"
RQ3 component not found at: $RQ3ProjectPath
Skipping bsard_evaluation install. Re-run with -RQ3ProjectPath once available:
  pip install -e "../RQ3_Autonomous_Evaluation"
"@
} else {
    & $Pip install -e $RQ3ProjectPath --quiet
    Write-Host "    bsard_evaluation installed from $RQ3ProjectPath" -ForegroundColor Green
}

# ── Step 4: Download language model data ──────────────────────────────────────
Write-Host "[4/6] Downloading language models..." -ForegroundColor Yellow

Write-Host "    spaCy fr_core_news_lg..."
& $Python -m spacy download fr_core_news_lg --quiet

Write-Host "    NLTK punkt tokenizer..."
& $Python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"

Write-Host "    Done." -ForegroundColor Green

# ── Step 5: Download the data bundle into the data root ───────────────────────
Write-Host "[5/6] Downloading data bundle from Hugging Face..." -ForegroundColor Yellow
$env:BSARD_DATA_DIR = $DataDir
& $Python (Join-Path $ProjectRoot "scripts\download_data.py")
Write-Host "    Data bundle downloaded to $DataDir" -ForegroundColor Green

# ── Step 6: Ensure results subdirectories exist ───────────────────────────────
Write-Host "[6/6] Ensuring results subdirectories exist..." -ForegroundColor Yellow
$ResultsRoot = Join-Path $DataDir "results"
foreach ($sub in @("sparse_retrieval", "dense_retrieval", "hybrid", "agentic")) {
    $dir = Join-Path $ResultsRoot $sub
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
}
Write-Host "    Results subdirectories ready under $DataDir/results/" -ForegroundColor Green

# ── Summary ───────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host " Setup complete." -ForegroundColor Green
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Activate the environment:"
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host ""
Write-Host "Point experiment scripts at the data root (if not the default output/):"
Write-Host "  `$env:BSARD_DATA_DIR = `"$DataDir`""
Write-Host ""
Write-Host "Run Tier 1 sparse experiments:"
Write-Host "  python scripts\evaluation\tier1\run_sparse_experiments.py"
Write-Host ""
Write-Host "Results are written under the data root:"
Write-Host "  $DataDir\results\sparse_retrieval\"
Write-Host ""
