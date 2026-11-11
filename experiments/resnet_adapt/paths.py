"""Output locations: everything below <DERMA_REPRO_OUT>/resnet_adapt/, never in the repo.

The names and suffix rules are those of every experiment package
(`experiments.common.layout`): tuning, runs and reports of a batch size other
than the default 256 live in their own folders (`step_<n>_bs<B>`,
`summary_step<n>_bs<B>.md`), so protocols with different batch sizes coexist and
never overwrite each other. The feature cache does not depend on the batch size
and is shared.
"""

from __future__ import annotations

from pathlib import Path

from experiments.bronet_mlp import env  # noqa: F401  (environment set-up)
from experiments.common import layout

DEFAULT_BATCH_SIZE = layout.DEFAULT_BATCH_SIZE
tag = layout.batch_tag
run_name = layout.run_name
_TREE = layout.RESNET


def out_root() -> Path:
    return _TREE.root()


def features_dir(step: int) -> Path:
    return _TREE.features_dir(step)


def runs_dir(step: int, batch_size: int = DEFAULT_BATCH_SIZE) -> Path:
    return _TREE.runs_dir(step, batch_size)


def run_dir(step: int, mode: str, seed: int, variant: str = "best",
            batch_size: int = DEFAULT_BATCH_SIZE) -> Path:
    return _TREE.run_dir(step, mode, seed, variant, batch_size)


def tuning_dir(step: int, batch_size: int = DEFAULT_BATCH_SIZE) -> Path:
    return _TREE.tuning_dir(step, batch_size)


def results_dir() -> Path:
    return _TREE.results_dir()
