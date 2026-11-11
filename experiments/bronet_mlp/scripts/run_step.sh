#!/usr/bin/env bash
# Whole pipeline for one step, resumable at every stage:
#   L_pre (once) -> feature cache -> cache check -> lr tuning on the validation
#   split -> modes x seeds with the tuned settings (best and stable) -> report
#
#   experiments/bronet_mlp/scripts/run_step.sh --step 2 [--seeds "0 1 2"] [--modes "..."]
#       [--python PY] [--batch-size 64] [--train-batch-size 256] [--cache-bro-weights] [--tests]
#
# Run from anywhere. Locations (DERMA_DATA_ROOT / DERMA_REPRO_OUT / BRONET_ROOT /
# BRONET_CKPT) come from the environment or paths.toml in the repository root.
# Each stage logs to its own file under <DERMA_REPRO_OUT>/bronet_mlp/; stdout
# shows progress and timings. --batch-size is the feature-extraction batch,
# --train-batch-size the training batch of tune and run_grid.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
PYTHON="${PYTHON:-python3}"
STEP=""
SEEDS="0"
MODES="head-only butterfly lora full"
BATCH=64
TRAIN_BATCH=256
CACHE_BRO=""
TESTS=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --step)              STEP="$2"; shift 2 ;;
        --seeds)             SEEDS="$2"; shift 2 ;;
        --modes)             MODES="$2"; shift 2 ;;
        --python)            PYTHON="$2"; shift 2 ;;
        --batch-size)        BATCH="$2"; shift 2 ;;
        --train-batch-size)  TRAIN_BATCH="$2"; shift 2 ;;
        --cache-bro-weights) CACHE_BRO="--cache-bro-weights"; shift ;;
        --tests)             TESTS=1; shift ;;
        -h|--help)           sed -n '2,13p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done
[[ -n "$STEP" ]] || { echo "--step is required" >&2; exit 2; }

cd "$REPO"
run() { echo; echo "=== $* ==="; "$PYTHON" -u "$@"; }

if [[ $TESTS -eq 1 ]]; then
    "$PYTHON" -m pytest experiments/bronet_mlp/tests -m "not slow" -q
fi
run -m experiments.bronet_mlp.bronet lpre $CACHE_BRO
run -m experiments.bronet_mlp.features extract --step "$STEP" --batch-size "$BATCH" $CACHE_BRO
run -m experiments.bronet_mlp.features verify --step "$STEP" $CACHE_BRO
# shellcheck disable=SC2086
run -m experiments.bronet_mlp.tune --step "$STEP" --batch-size "$TRAIN_BATCH" --modes $MODES
# shellcheck disable=SC2086
run -m experiments.bronet_mlp.run_grid --step "$STEP" --from-tuning --batch-size "$TRAIN_BATCH" \
    --modes $MODES --seeds $SEEDS
