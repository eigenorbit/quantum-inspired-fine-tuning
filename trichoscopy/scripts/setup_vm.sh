#!/usr/bin/env bash
# Prepare a fresh Linux GPU box for training. Written against the AWS
# "Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)", which ships
# the NVIDIA driver and CUDA but no Python frameworks.
#
#   ./scripts/setup_vm.sh
#
# Creates trichoscopy/.venv, installs the CUDA build of torch and the rest of
# requirements.txt, and verifies the GPU is actually visible from torch. Safe
# to re-run: an existing venv is reused.
#
# Ubuntu 22.04's system Python is 3.10, which torch 2.4.1 supports, so no conda
# is needed. Pass --python python3.12 if you have another interpreter you would
# rather use (torch 2.4.1 has no wheels above 3.12).

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_PYTHON="python3"
TORCH_INDEX="https://download.pytorch.org/whl/cu121"
TORCH="torch==2.4.1+cu121"
VISION="torchvision==0.19.1+cu121"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --python)      BASE_PYTHON="$2"; shift 2 ;;
        --torch-index) TORCH_INDEX="$2"; shift 2 ;;
        -h|--help)     sed -n '2,15p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

echo "=== GPU ==="
if ! command -v nvidia-smi >/dev/null; then
    echo "nvidia-smi not found: this box has no NVIDIA driver." >&2
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

echo
echo "=== system packages ==="
sudo apt-get update -qq
# python3-venv is not always present on the Base AMI; tmux is what keeps a
# training run alive across a dropped SSH session.
sudo apt-get install -y -qq python3-venv python3-pip tmux

echo
echo "=== virtualenv ==="
if [[ ! -d "$PROJECT_ROOT/.venv" ]]; then
    "$BASE_PYTHON" -m venv "$PROJECT_ROOT/.venv"
fi
VENV_PY="$PROJECT_ROOT/.venv/bin/python"
"$VENV_PY" -m pip install --quiet --upgrade pip
echo "python: $("$VENV_PY" --version)"

echo
echo "=== torch (CUDA build) ==="
# The +cu121 suffix is load-bearing: plain torch==2.4.1 resolves to the CPU
# build, and pip then counts that as satisfying every later torch requirement,
# so the mistake is silent. --force-reinstall makes re-running this script
# repair a box that already has the CPU build.
"$VENV_PY" -m pip install --force-reinstall --no-deps "$TORCH" "$VISION" \
    --index-url "$TORCH_INDEX"

echo
echo "=== remaining packages ==="
"$VENV_PY" -m pip install --quiet -r "$PROJECT_ROOT/requirements.txt"

echo
echo "=== verify ==="
"$VENV_PY" - <<'PY'
import torch, torchvision
print("torch      ", torch.__version__)
print("torchvision", torchvision.__version__)
print("cuda       ", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not visible from torch - training would run on the CPU")
props = torch.cuda.get_device_properties(0)
print("gpu        ", props.name, f"{props.total_memory / 2**30:.1f} GiB")
PY

echo
echo "Done. Next:"
echo "  cp $(dirname "$PROJECT_ROOT")/paths.example.toml $(dirname "$PROJECT_ROOT")/paths.toml"
echo "  # set DERMA_DATA_ROOT in paths.toml (or: export DERMA_DATA_ROOT=/path/to/data)"
echo "  export PYTHONPATH=$PROJECT_ROOT/src"
echo "  $VENV_PY -m repro.check"
