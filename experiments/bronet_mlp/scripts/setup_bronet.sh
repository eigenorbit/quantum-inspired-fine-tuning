#!/usr/bin/env bash
# Fetch BRONet (code + ImageNet checkpoint) and install the Python dependencies.
#
#   experiments/bronet_mlp/scripts/setup_bronet.sh [--python PY] [--skip-pip]
#
# Uses the same locations as the code (environment, else paths.toml in the
# repository root; defaults under DERMA_DATA_ROOT):
#   BRONET_ROOT   clone of ntuaislab/BRONet           (<DATA_ROOT>/pretrained/BRONet)
#   BRONET_CKPT   imagenet-14x588_399.pth + config    (<DATA_ROOT>/pretrained/bronet_imagenet)
# Safe to re-run: an existing clone / verified checkpoint is kept.
# The checkpoint is 1.62 GB; its sha256 is checked and a mismatch is an error.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG="$(dirname "$HERE")"
PYTHON="python3"
SKIP_PIP=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --python)   PYTHON="$2"; shift 2 ;;
        --skip-pip) SKIP_PIP=1; shift ;;
        -h|--help)  sed -n '2,13p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

# One rule for every location, shared with the Python code (repro.settings,
# standard library only, so this works before the packages are installed).
export PYTHONPATH="$(cd "$PKG/../.." && pwd)/trichoscopy/src"
BRONET_ROOT="$("$PYTHON" -m repro.where BRONET_ROOT)"
BRONET_CKPT="$("$PYTHON" -m repro.where BRONET_CKPT)"
COMMIT="9d59c9b"
HF="https://huggingface.co/pinhank121/BRONet_ImageNet/resolve/main"
CKPT="imagenet-14x588_399.pth"
SHA256="e8521d477bdeb80b0a623b09a7ce3025b874f21984e50686e869ab61694ca95f"

echo "=== BRONet source -> $BRONET_ROOT ==="
if [[ ! -d "$BRONET_ROOT/.git" ]]; then
    mkdir -p "$(dirname "$BRONET_ROOT")"
    git clone https://github.com/ntuaislab/BRONet.git "$BRONET_ROOT"
fi
git -C "$BRONET_ROOT" checkout --quiet "$COMMIT"
echo "at $(git -C "$BRONET_ROOT" rev-parse --short HEAD)"

echo "=== checkpoint -> $BRONET_CKPT ==="
mkdir -p "$BRONET_CKPT"
if [[ ! -f "$BRONET_CKPT/config.yaml" ]]; then
    curl -fL --retry 3 -o "$BRONET_CKPT/config.yaml" "$HF/config.yaml"
fi
if [[ ! -f "$BRONET_CKPT/$CKPT" ]]; then
    curl -fL --retry 3 -C - -o "$BRONET_CKPT/$CKPT.part" "$HF/$CKPT"
    mv "$BRONET_CKPT/$CKPT.part" "$BRONET_CKPT/$CKPT"
fi
got="$(sha256sum "$BRONET_CKPT/$CKPT" | cut -d' ' -f1)"
if [[ "$got" != "$SHA256" ]]; then
    echo "sha256 mismatch for $CKPT: got $got, expected $SHA256" >&2
    exit 1
fi
echo "sha256 ok"

if [[ $SKIP_PIP -eq 0 ]]; then
    echo "=== python packages ==="
    "$PYTHON" -c "import torch" 2>/dev/null || {
        echo "torch is not installed for $PYTHON. Install the CUDA build first, e.g." >&2
        echo "  $PYTHON -m pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121" >&2
        echo "(trichoscopy/scripts/setup_vm.sh does this on an AWS DLAMI)" >&2
        exit 1
    }
    "$PYTHON" -m pip install --quiet -r "$PKG/requirements.txt"
    "$PYTHON" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
fi

echo
echo "Done. Locations used by the runs (python -m repro.where):"
"$PYTHON" -m repro.where
