"""Filesystem layout of the reproduction.

The image data is large and lives outside the repository, at ``DERMA_DATA_ROOT``
(environment or ``paths.toml``, see `repro.settings`)::

    <DATA_ROOT>/model_data_step1_anon/{train,test}/{0,1,2,3,4}/*.jpg
    <DATA_ROOT>/model_data_step2_anon/{train,test}/{0,1,2}/*.jpg

Everything the reproduction *writes* — checkpoints, metric tables, figures,
logs, hash caches — goes below ``DERMA_REPRO_OUT`` (default
``<DATA_ROOT>/repro_out``), never into the repository.
"""

from __future__ import annotations

from pathlib import Path

from . import settings

# .../trichoscopy/src/repro/paths.py -> .../trichoscopy/
PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[1]

DATA_ROOT = Path(settings.require("DERMA_DATA_ROOT"))

# Augmented, class-balanced folders delivered by the authors. `train/` is what
# the models are fitted on; the published metrics come from the *original*
# (non-augmented) images inside `test/` — see `data.list_images`.
STEP_DIRS = {
    1: DATA_ROOT / "model_data_step1_anon",
    2: DATA_ROOT / "model_data_step2_anon",
}

OUTPUT_ROOT = Path(settings.require("DERMA_REPRO_OUT"))
MODELS_DIR = OUTPUT_ROOT / "models"
RESULTS_DIR = OUTPUT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"


def split_dir(step: int, split: str) -> Path:
    """Directory holding one split of one step, e.g. ``.../test/``."""
    if step not in STEP_DIRS:
        raise ValueError(f"step must be 1 or 2, got {step!r}")
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")
    return STEP_DIRS[step] / split


# Dataset variants that can be trained on. "full" is the delivered data;
# "unique" drops byte-identical duplicates (`repro.dedup`). Each writes to its
# own folder, so one never overwrites another.
VARIANTS = ("full", "unique")


def is_variant(variant: str) -> bool:
    return variant in VARIANTS


def run_dir(step: int, architecture: str, variant: str = "full") -> Path:
    """Where one run writes its outputs."""
    if not is_variant(variant):
        raise ValueError(f"unknown variant {variant!r}")
    suffix = "" if variant == "full" else f"_{variant}"
    return MODELS_DIR / f"step_{step}" / f"{architecture}{suffix}"


def check_data_root() -> None:
    """Fail early, with a useful message, when the data is not where we look."""
    missing = [d for d in STEP_DIRS.values() if not d.is_dir()]
    if missing:
        listing = "\n".join(f"  {d}" for d in missing)
        raise FileNotFoundError(
            f"dataset folders not found:\n{listing}\n"
            f"DATA_ROOT is {DATA_ROOT}; set DERMA_DATA_ROOT to the folder that "
            "contains model_data_step1_anon and model_data_step2_anon."
        )
