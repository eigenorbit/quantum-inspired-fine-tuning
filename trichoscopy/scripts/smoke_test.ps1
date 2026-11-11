<#
.SYNOPSIS
    Verify the environment and the pipeline in a couple of minutes.

.DESCRIPTION
    Checks CUDA, checks that the data is where the code expects it and that the
    splits have the documented sizes, then trains two truncated models -- one
    per step, so both the 5-class and the 3-class paths are exercised -- and
    builds a report. Nothing is written to `models/` or `results/`: the run
    goes to a temporary folder that is deleted afterwards.

    Run this before `train_all.ps1`.

.EXAMPLE
    .\scripts\smoke_test.ps1
#>
[CmdletBinding()]
param(
    # default: `python` on PATH (activate the venv first)
    [string] $Python   = "python",
    # default: DERMA_DATA_ROOT from the environment or paths.toml
    [string] $DataRoot = ""
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = Join-Path $projectRoot "src"
if ($DataRoot) { $env:DERMA_DATA_ROOT = $DataRoot }

$scratch = Join-Path ([System.IO.Path]::GetTempPath()) "repro_smoke_$(Get-Random)"
$env:DERMA_REPRO_OUT = $scratch

try {
    Write-Host "=== environment ===" -ForegroundColor Green
    & $Python -m repro.check
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    Write-Host "`n=== step 2, 1 epoch on a 960-image subset ===" -ForegroundColor Green
    & $Python -m repro.train --step 2 --arch efficientnet_b0 --epochs 1 `
        --limit-train 960 --log-every 0
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    Write-Host "`n=== step 1, 1 epoch on a 960-image subset ===" -ForegroundColor Green
    & $Python -m repro.train --step 1 --arch efficientnet_b0 --epochs 1 `
        --limit-train 960 --log-every 0
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    Write-Host "`n=== report ===" -ForegroundColor Green
    & $Python -m repro.report
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    Write-Host "`nSmoke test passed. The numbers above mean nothing (1 epoch on" -ForegroundColor Cyan
    Write-Host "a fraction of the data) -- only that the pipeline runs." -ForegroundColor Cyan
    Write-Host "To also prove runs are bit-for-bit reproducible (~5 min):" -ForegroundColor Cyan
    Write-Host "    python -m repro.selftest" -ForegroundColor Cyan
}
finally {
    if (Test-Path $scratch) { Remove-Item -Recurse -Force $scratch }
}
