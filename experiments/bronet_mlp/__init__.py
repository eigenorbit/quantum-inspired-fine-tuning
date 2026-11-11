"""Certified adaptation of BRONet to trichoscopy through its dense MLP.

How to run: ``README.md`` in this folder.

Importing the package does these things, in this order, before anything else
can import torch:

1. puts ``trichoscopy/src`` (package ``repro``) on ``sys.path`` and imports
   ``repro``, which sets ``CUBLAS_WORKSPACE_CONFIG`` for deterministic cuBLAS
   (read when the CUDA context is created) and fills in the locations of the
   data, the outputs and BRONet (`repro.settings`; ``env.apply_defaults``) --
   in particular ``DERMA_REPRO_OUT``, so that ``repro`` never writes its hash
   cache into the repository tree;
2. loads ``experiments/robustness/core.py`` as module ``core`` and
   ``experiments/robustness/init_draws.py`` as module ``init_draws`` -- imported,
   never copied;
3. turns TF32 off (below).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
_REPRO_SRC = REPO_ROOT / "trichoscopy" / "src"
if str(_REPRO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPRO_SRC))

import repro  # noqa: E402,F401  (side effects: CUBLAS_WORKSPACE_CONFIG, locations)

from . import env  # noqa: E402

env.apply_defaults()


def _load_robustness_module(name: str) -> None:
    """Register experiments/robustness/<name>.py as module ``name``.

    Loaded by file path rather than by putting experiments/robustness on
    sys.path: that folder also holds ``experiments.py``, which would shadow this
    very package (``experiments.bronet_mlp``) in spawned DataLoader workers.
    """
    if name in sys.modules:
        return
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        name, REPO_ROOT / "experiments" / "robustness" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)


_load_robustness_module("core")
_load_robustness_module("init_draws")

# Full float32 everywhere. On Ampere+ GPUs cuDNN convolutions default to TF32,
# which moves phi(x) by up to 5e-2 (|phi| ~ 200) against the CPU forward --
# larger than every other numerical effect here. Measured: 8e-5 with TF32 off.
import torch  # noqa: E402

torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
