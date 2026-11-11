<#
.SYNOPSIS
    Train all six models of the reproduction, then build the comparison report.

.DESCRIPTION
    Runs the cheap models first, so a problem shows up in minutes rather than
    hours, and Step-2 before Step-1, because Step-2 is the half that compares
    directly with the published numbers.

    Every run is resumable: re-running this script picks each model up from its
    last completed epoch (`last_full.pth`). A run that already finished all 30
    epochs is skipped in seconds.

    Measured on an RTX 4070 Laptop (8 GB), batch 32, full precision:

        step 2  efficientnet_b0    ~30 min
        step 2  densenet169        ~70 min
        step 2  resnet152         ~110 min
        step 1  efficientnet_b0    ~60 min
        step 1  densenet169       ~140 min
        step 1  resnet152         ~230 min
        ------------------------------------
        total                     ~10.5 h

    -Amp roughly halves that (and halves GPU memory) at the cost of running the
    forward/backward pass in float16. The paper does not mention mixed
    precision, so it is off by default.

.PARAMETER Unique
    Train on unique images only: drop the byte-identical duplicates the
    augmentation left behind (19.6% of step-1 train, 28.1% of step-2). Writes to
    <arch>_unique/, so baseline runs are kept, and the run is correspondingly
    faster because there is less data.

.PARAMETER Amp
    Enable mixed precision. ~2x faster; a small numerical deviation from the
    paper's recipe.

.PARAMETER Python
    Interpreter to use. Defaults to the activated conda environment when there
    is one, else trichoscopy\.venv, else `python` on PATH (an activated venv).

.PARAMETER DataRoot
    Folder holding model_data_step1_anon and model_data_step2_anon. Default:
    DERMA_DATA_ROOT from the environment or paths.toml (see paths.example.toml);
    outputs go to DERMA_REPRO_OUT (default <DataRoot>\repro_out).

.EXAMPLE
    .\scripts\train_all.ps1
.EXAMPLE
    .\scripts\train_all.ps1 -Amp
.EXAMPLE
    .\scripts\train_all.ps1 -Unique
#>
[CmdletBinding()]
param(
    [switch] $Unique,
    [switch] $Amp,
    [string] $Python   = "",
    [string] $DataRoot = ""
)

$ErrorActionPreference = "Stop"

function Resolve-Python {
    <#
        Which interpreter to run. In order of preference:
          1. -Python, if given explicitly;
          2. the activated conda environment ($env:CONDA_PREFIX), so running
             this from an Anaconda PowerShell Prompt after `conda activate`
             just works;
          3. a .venv inside the project;
          4. whatever `python` resolves to on PATH (e.g. an activated venv).
    #>
    param([string] $Explicit, [string] $ProjectRoot)

    if ($Explicit) { return $Explicit }
    if ($env:CONDA_PREFIX) {
        $candidate = Join-Path $env:CONDA_PREFIX "python.exe"
        if (Test-Path $candidate) { return $candidate }
    }
    $candidate = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
    if (Test-Path $candidate) { return $candidate }
    return "python"
}

function Get-RepoLocation([string] $Name) {
    # One rule for every location, shared with the Python code: environment,
    # else paths.toml in the repository root (repro.settings).
    $value = & $Python -m repro.where $Name
    if ($LASTEXITCODE -ne 0 -or -not $value) {
        throw "$Name is not set: set it in the environment or in paths.toml (see paths.example.toml)"
    }
    return $value
}

$projectRoot = Split-Path -Parent $PSScriptRoot
$Python = Resolve-Python -Explicit $Python -ProjectRoot $projectRoot
$env:PYTHONPATH = Join-Path $projectRoot "src"
if ($DataRoot) { $env:DERMA_DATA_ROOT = $DataRoot }
$DataRoot = Get-RepoLocation "DERMA_DATA_ROOT"

$outputRoot = Get-RepoLocation "DERMA_REPRO_OUT"
$logDir = Join-Path $outputRoot "results\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

# Cheapest first; Step-2 before Step-1.
$runs = @(
    @{ Step = 2; Arch = "efficientnet_b0" },
    @{ Step = 2; Arch = "densenet169"     },
    @{ Step = 2; Arch = "resnet152"       },
    @{ Step = 1; Arch = "efficientnet_b0" },
    @{ Step = 1; Arch = "densenet169"     },
    @{ Step = 1; Arch = "resnet152"       }
)

$started = Get-Date
Write-Host "Training $($runs.Count) models. Data root: $DataRoot" -ForegroundColor Cyan
if ($Amp)    { Write-Host "Mixed precision: ON" -ForegroundColor Yellow }
if ($Unique) { Write-Host "Deduplicated training set: ON" -ForegroundColor Yellow }

foreach ($run in $runs) {
    $label = "step$($run.Step)_$($run.Arch)" + $(if ($Unique) { "_unique" } else { "" })
    $log   = Join-Path $logDir "$label.log"
    Write-Host "`n=== $label ===" -ForegroundColor Green

    $arguments = @(
        "-m", "repro.train",
        "--step", $run.Step,
        "--arch", $run.Arch,
        "--resume"
    )
    if ($Amp)    { $arguments += "--amp" }
    if ($Unique) { $arguments += "--unique" }

    & $Python @arguments 2>&1 | Tee-Object -FilePath $log
    if ($LASTEXITCODE -ne 0) {
        Write-Host "$label FAILED (exit $LASTEXITCODE); see $log" -ForegroundColor Red
        Write-Host "Re-run this script to resume from the last completed epoch." -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

Write-Host "`n=== report ===" -ForegroundColor Green
if ($Unique) {
    & $Python -m repro.report --variant unique
    & $Python -m repro.report --variant unique --which best --no-figures
} else {
    & $Python -m repro.report
    & $Python -m repro.report --which best --no-figures
}

$elapsed = (Get-Date) - $started
Write-Host ("`nAll runs finished in {0:hh\:mm\:ss}." -f $elapsed) -ForegroundColor Cyan
Write-Host "Results: $(Join-Path $outputRoot 'results')" -ForegroundColor Cyan
