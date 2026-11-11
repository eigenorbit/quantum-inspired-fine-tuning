"""Run from the repository root: python -m pytest experiments/common/tests"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import experiments.bronet_mlp  # noqa: E402,F401  (locations, repro on sys.path)
