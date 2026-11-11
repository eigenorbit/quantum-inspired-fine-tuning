#!/usr/bin/env bash
# Train all six models, then build the comparison report. Linux counterpart of
# train_all.ps1.
#
# Runs the cheap models first, so a problem shows up in minutes rather than
# hours, and Step-2 before Step-1, because Step-2 is the half that compares
# directly with the published numbers.
#
# Every run is resumable: re-running this script picks each model up from its
# last completed epoch (last_full.pth). A run that already finished all 30
# epochs is skipped in seconds.
#
# Usage:
#   ./scripts/train_all.sh                    # the paper's recipe
#   ./scripts/train_all.sh --unique           # deduplicated training set
#   ./scripts/train_all.sh --amp              # mixed precision, ~2x faster
#   ./scripts/train_all.sh --python /path/to/python --data-root /data/derma
#
# Meant to be started inside tmux, so it survives a dropped SSH session.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
DATA_ROOT=""   # default: DERMA_DATA_ROOT from the environment or paths.toml
UNIQUE=0
AMP=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --unique)    UNIQUE=1; shift ;;
        --amp)       AMP=1; shift ;;
        --python)    PYTHON="$2"; shift 2 ;;
        --data-root) DATA_ROOT="$2"; shift 2 ;;
        -h|--help)   sed -n '2,19p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

export PYTHONPATH="$PROJECT_ROOT/src"
if [[ -n "$DATA_ROOT" ]]; then export DERMA_DATA_ROOT="$DATA_ROOT"; fi
# One rule for every location, shared with the Python code (repro.settings):
# environment, else paths.toml in the repository root.
DATA_ROOT="$("$PYTHON" -m repro.where DERMA_DATA_ROOT)"

OUTPUT_ROOT="$("$PYTHON" -m repro.where DERMA_REPRO_OUT)"
LOG_DIR="$OUTPUT_ROOT/results/logs"
mkdir -p "$LOG_DIR"

# Cheapest first; Step-2 before Step-1.
RUNS=(
    "2 efficientnet_b0"
    "2 densenet169"
    "2 resnet152"
    "1 efficientnet_b0"
    "1 densenet169"
    "1 resnet152"
)

echo "Training ${#RUNS[@]} models."
echo "  python    : $PYTHON"
echo "  data root : $DATA_ROOT"
if [[ $UNIQUE -eq 1 ]];  then echo "  deduplicated training set: ON"; fi
if [[ $AMP -eq 1 ]];     then echo "  mixed precision: ON"; fi

"$PYTHON" -m repro.check

STARTED=$SECONDS
for run in "${RUNS[@]}"; do
    read -r step arch <<< "$run"
    # Plain `if`, not `cond && assign`: under `set -e` an assignment whose
    # command substitution exits non-zero aborts the script.
    variant=""
    if [[ $UNIQUE -eq 1 ]];  then variant="${variant}_unique"; fi
    label="step${step}_${arch}${variant}"
    log="$LOG_DIR/$label.log"
    echo
    echo "=== $label ==="

    args=(-m repro.train --step "$step" --arch "$arch" --resume)
    if [[ $UNIQUE -eq 1 ]];  then args+=(--unique); fi
    if [[ $AMP -eq 1 ]];     then args+=(--amp); fi

    # `set -e` would abort before the message below, so handle failure here.
    if ! "$PYTHON" "${args[@]}" 2>&1 | tee "$log"; then
        echo "$label FAILED; see $log" >&2
        echo "Re-run this script to resume from the last completed epoch." >&2
        exit 1
    fi
done

echo
echo "=== report ==="
report_variant="${variant#_}"
if [[ -z "$report_variant" ]]; then report_variant="full"; fi
"$PYTHON" -m repro.report --variant "$report_variant"
"$PYTHON" -m repro.report --variant "$report_variant" --which best --no-figures

elapsed=$(( SECONDS - STARTED ))
printf '\nAll runs finished in %02d:%02d:%02d.\n' \
    $(( elapsed / 3600 )) $(( elapsed % 3600 / 60 )) $(( elapsed % 60 ))
echo "Results: $OUTPUT_ROOT/results"
