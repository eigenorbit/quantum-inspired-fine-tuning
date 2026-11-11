"""Reproduction of Caposiena Caro et al., JDDG 2026;24:44-55 (DOI 10.1111/ddg.15847).

Trains the paper's two-step trichoscopy framework -- three CNN architectures
per step -- on the study's preprocessed data and compares the
result with the published Tables 5 and 6.

    python -m repro.check
    python -m repro.train  --step 2 --arch densenet169
    python -m repro.report --step 2
"""

import os

# cuBLAS reuses workspace buffers across streams, which makes some matmuls
# nondeterministic. Reserving explicit workspaces fixes that, but the variable
# is read when the CUDA context is created -- so it has to be set before
# anything imports torch. Importing `repro` always happens first, hence here.
# Harmless when the run is not deterministic, and an explicit setting from the
# environment wins.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

# Locations (data, outputs, pretrained weights): environment, else paths.toml,
# else defaults below the data root -- see `repro.settings`. Done here, before
# any module reads them.
from . import settings as _settings  # noqa: E402

_settings.apply()

__all__ = [
    "check",
    "config",
    "console",
    "data",
    "dedup",
    "engine",
    "metrics",
    "models",
    "paths",
    "published",
    "report",
    "selftest",
    "settings",
    "where",
    "seeding",
    "train",
]
