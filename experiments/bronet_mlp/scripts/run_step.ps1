<#
.SYNOPSIS
    Whole pipeline for one step, resumable at every stage.

.DESCRIPTION
    L_pre (once) -> feature cache -> cache check -> lr tuning on the validation
    split -> modes x seeds with the tuned settings (best and stable) -> report.
    Locations: DERMA_DATA_ROOT / DERMA_REPRO_OUT / BRONET_ROOT / BRONET_CKPT from
    the environment or paths.toml in the repository root (see paths.example.toml).
    Each stage logs to its own file under <DERMA_REPRO_OUT>\bronet_mlp\; the
    console shows progress and timings.

.EXAMPLE
    .\experiments\bronet_mlp\scripts\run_step.ps1 -Step 2 -CacheBroWeights
.EXAMPLE
    .\experiments\bronet_mlp\scripts\run_step.ps1 -Step 2 -Seeds 0,1,2 -TrainBatchSize 32
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][ValidateSet(1, 2)][int] $Step,
    [int[]] $Seeds = @(0),
    [string[]] $Modes = @("head-only", "butterfly", "lora", "full"),
    [string] $Python = "",
    [int] $BatchSize = 64,          # feature extraction
    [int] $TrainBatchSize = 256,    # training (tune, run_grid); 256 = the default protocol
    [switch] $CacheBroWeights,
    [switch] $Tests
)
$ErrorActionPreference = "Stop"

$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
if (-not $Python) {
    # the activated conda environment, else `python` on PATH (e.g. an activated venv)
    $Python = if ($env:CONDA_PREFIX -and (Test-Path (Join-Path $env:CONDA_PREFIX "python.exe"))) {
        Join-Path $env:CONDA_PREFIX "python.exe"
    } else { "python" }
}
$cache = if ($CacheBroWeights) { @("--cache-bro-weights") } else { @() }

function Invoke-Stage([string[]] $Arguments) {
    Write-Host "`n=== $($Arguments -join ' ') ===" -ForegroundColor Green
    & $Python -u @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "stage failed (exit $LASTEXITCODE): $($Arguments -join ' '). Re-run to resume."
    }
}

Push-Location $repo
try {
    if ($Tests) { Invoke-Stage @("-m", "pytest", "experiments/bronet_mlp/tests", "-m", "not slow", "-q") }
    Invoke-Stage (@("-m", "experiments.bronet_mlp.bronet", "lpre") + $cache)
    Invoke-Stage (@("-m", "experiments.bronet_mlp.features", "extract", "--step", "$Step",
                    "--batch-size", "$BatchSize") + $cache)
    Invoke-Stage (@("-m", "experiments.bronet_mlp.features", "verify", "--step", "$Step") + $cache)
    Invoke-Stage (@("-m", "experiments.bronet_mlp.tune", "--step", "$Step",
                    "--batch-size", "$TrainBatchSize", "--modes") + $Modes)
    Invoke-Stage (@("-m", "experiments.bronet_mlp.run_grid", "--step", "$Step", "--from-tuning",
                    "--batch-size", "$TrainBatchSize", "--modes") + $Modes +
                  @("--seeds") + ($Seeds | ForEach-Object { "$_" }))
} finally {
    Pop-Location
}
