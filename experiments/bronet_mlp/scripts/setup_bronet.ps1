<#
.SYNOPSIS
    Fetch BRONet (code + ImageNet checkpoint) and install the Python dependencies.

.DESCRIPTION
    Uses the same locations as the code (environment, else paths.toml in the
    repository root; defaults under DERMA_DATA_ROOT):
      BRONET_ROOT   clone of ntuaislab/BRONet           (<DATA_ROOT>\pretrained\BRONet)
      BRONET_CKPT   imagenet-14x588_399.pth + config    (<DATA_ROOT>\pretrained\bronet_imagenet)
    Safe to re-run: an existing clone / verified checkpoint is kept. The checkpoint is
    1.62 GB; its sha256 is checked and a mismatch is an error.

.EXAMPLE
    .\experiments\bronet_mlp\scripts\setup_bronet.ps1
.EXAMPLE
    .\experiments\bronet_mlp\scripts\setup_bronet.ps1 -Python C:\path\to\python.exe -SkipPip
#>
[CmdletBinding()]
param(
    [string] $Python = "",
    [switch] $SkipPip
)
$ErrorActionPreference = "Stop"

$pkg = Split-Path -Parent $PSScriptRoot
$repo = (Resolve-Path (Join-Path $pkg "..\..")).Path
if (-not $Python) { $Python = "python" }   # e.g. an activated venv

# One rule for every location, shared with the Python code (repro.settings,
# standard library only, so this works before the packages are installed).
$env:PYTHONPATH = Join-Path $repo "trichoscopy\src"
function Get-RepoLocation([string] $Name) {
    $value = & $Python -m repro.where $Name
    if ($LASTEXITCODE -ne 0 -or -not $value) {
        throw "$Name is not set: set it in the environment or in paths.toml (see paths.example.toml)"
    }
    return $value
}
$dataRoot = Get-RepoLocation "DERMA_DATA_ROOT"
$bronetRoot = Get-RepoLocation "BRONET_ROOT"
$bronetCkpt = Get-RepoLocation "BRONET_CKPT"
$commit = "9d59c9b"
$hf = "https://huggingface.co/pinhank121/BRONet_ImageNet/resolve/main"
$ckpt = "imagenet-14x588_399.pth"
$sha256 = "e8521d477bdeb80b0a623b09a7ce3025b874f21984e50686e869ab61694ca95f"

Write-Host "=== BRONet source -> $bronetRoot ===" -ForegroundColor Green
if (-not (Test-Path (Join-Path $bronetRoot ".git"))) {
    New-Item -ItemType Directory -Force (Split-Path -Parent $bronetRoot) | Out-Null
    git clone https://github.com/ntuaislab/BRONet.git $bronetRoot
    if ($LASTEXITCODE -ne 0) { throw "git clone failed" }
}
git -C $bronetRoot checkout --quiet $commit
if ($LASTEXITCODE -ne 0) { throw "git checkout $commit failed" }
Write-Host "at $(git -C $bronetRoot rev-parse --short HEAD)"

Write-Host "=== checkpoint -> $bronetCkpt ===" -ForegroundColor Green
New-Item -ItemType Directory -Force $bronetCkpt | Out-Null
$ProgressPreference = "SilentlyContinue"   # Invoke-WebRequest is very slow with the progress bar
$cfg = Join-Path $bronetCkpt "config.yaml"
if (-not (Test-Path $cfg)) { Invoke-WebRequest "$hf/config.yaml" -OutFile $cfg }
$target = Join-Path $bronetCkpt $ckpt
if (-not (Test-Path $target)) {
    Invoke-WebRequest "$hf/$ckpt" -OutFile "$target.part"
    Move-Item "$target.part" $target
}
$got = (Get-FileHash -Algorithm SHA256 $target).Hash.ToLower()
if ($got -ne $sha256) { throw "sha256 mismatch for ${ckpt}: got $got, expected $sha256" }
Write-Host "sha256 ok"

if (-not $SkipPip) {
    Write-Host "=== python packages ($Python) ===" -ForegroundColor Green
    & $Python -c "import torch"
    if ($LASTEXITCODE -ne 0) {
        throw ("torch is not installed. Install the CUDA build first, e.g. " +
               "pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 " +
               "--index-url https://download.pytorch.org/whl/cu121")
    }
    & $Python -m pip install --quiet -r (Join-Path $pkg "requirements.txt")
    & $Python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
}

Write-Host "`nDone. Locations used by the runs (python -m repro.where):" -ForegroundColor Cyan
& $Python -m repro.where
