$ErrorActionPreference = "Continue"
$proj = $PSScriptRoot
$PYTHON = Join-Path $proj ".venv\Scripts\python.exe"
$log = Join-Path $proj "_run_precompute.log"
Set-Location $proj

# Force Python stdout/stderr to UTF-8 so Unicode prints (e.g. box-drawing chars
# in precompute_retrieval.py line 280) don't crash on Windows cp1252.
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

$utf8 = [System.Text.UTF8Encoding]::new($false)
[System.IO.File]::WriteAllText($log, "", $utf8)

function Write-Both {
    param([Parameter(ValueFromPipeline=$true)]$msg)
    process {
        $line = if ($null -eq $msg) { "" }
                elseif ($msg -is [System.Management.Automation.ErrorRecord]) { $msg.ToString() }
                else { "$msg" }
        [System.IO.File]::AppendAllText($log, "$line`r`n", $utf8)
        [Console]::Out.WriteLine($line)
    }
}

$pdfs  = @(
    "1804_03_21_1804032150",
    "1804_03_21_1804032153",
    "1867_06_08_1867060850",
    "1967_10_10_1967101055",
    "1967_10_10_1967101056",
    "2003_07_17_2013A31614"
)
# --force applies only to the PDF that was originally re-run-with-force per the
# handoff prompt. Phase 1 is already complete for all 6 (verified on disk
# 2026-05-05); this script resumes from Phase 2 only.
$rerun = @("1804_03_21_1804032150")
$qsets = @(
    "..\RQ2_T07_EVALUATION\ground_truth\bsard_test.json",
    "..\RQ2_T07_EVALUATION\ground_truth\bsard_train.json"
)

function Stamp { (Get-Date -Format "HH:mm:ss") }

Write-Both "[$(Stamp)] === Phase 2: precompute_retrieval (resume after Phase 1) ==="
foreach ($p in $pdfs) {
    $useForce = $rerun -contains $p
    Write-Both ""
    Write-Both "[$(Stamp)] --- Phase2 PDF=$p force=$useForce ---"
    if ($useForce) {
        & $PYTHON scripts/precompute_retrieval.py --pdf "$p.pdf" --qset $qsets --top-k 200 --force -v 2>&1 | Write-Both
    } else {
        & $PYTHON scripts/precompute_retrieval.py --pdf "$p.pdf" --qset $qsets --top-k 200 -v 2>&1 | Write-Both
    }
    if ($LASTEXITCODE -ne 0) {
        Write-Both "[$(Stamp)] FAILED Phase2 PDF=$p exit=$LASTEXITCODE"
        exit 1
    }
}

Write-Both ""
Write-Both "[$(Stamp)] === ALL DONE ==="
