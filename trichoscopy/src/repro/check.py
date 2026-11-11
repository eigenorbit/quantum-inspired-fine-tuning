"""Verify the environment and the dataset before committing hours to a run.

    python -m repro.check

Reports the torch/CUDA situation and compares every split against the sizes
the paper states, so a wrong ``DERMA_DATA_ROOT`` or a partially copied dataset
fails here instead of three hours into training. Exit code 1 if anything is
wrong.
"""

from __future__ import annotations

import os
import sys

from . import console, paths
from .config import STEPS

# Sizes measured on the delivered dataset. The augmented totals are exactly the
# ones the paper reports (Results, p. 48): 39,368 + 9,851 = 49,219 for Step-1
# and 19,076 + 4,809 = 23,885 for Step-2.
EXPECTED = {
    1: {
        "train_all": 39_368,
        "test_all": 9_851,
        "test_originals": {"AA": 241, "FFA-LPP": 121, "DF": 60, "AGA": 14, "CHP": 281},
    },
    2: {
        "train_all": 19_076,
        "test_all": 4_809,
        "test_originals": {"AA active": 78, "AA inactive": 48, "AA in regrowth": 114},
    },
}


def check_torch() -> list[str]:
    """Report the torch install; return a list of problems."""
    problems = []
    try:
        import torch
        import torchvision
    except ImportError as error:
        return [f"torch is not installed: {error}"]

    print(f"  python       {sys.version.split()[0]}")
    print(f"  torch        {torch.__version__}")
    print(f"  torchvision  {torchvision.__version__}")

    workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if workspace in (":4096:8", ":16:8"):
        print(f"  cublas       {workspace} (deterministic)")
    else:
        problems.append(
            f"CUBLAS_WORKSPACE_CONFIG is {workspace!r}; deterministic runs need "
            "':4096:8'. `repro/__init__.py` sets it, so seeing this means torch "
            "was imported before the package."
        )

    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        print(f"  cuda         yes - {properties.name}, "
              f"{properties.total_memory / 2**30:.1f} GiB")
        if properties.total_memory / 2**30 < 7.0:
            problems.append(
                "less than 7 GiB of GPU memory: ResNet152 at batch 32 peaks at "
                "about 6.0 GiB in full precision. Use --amp (about 3.5 GiB) or "
                "a smaller batch."
            )
    else:
        print("  cuda         NO")
        problems.append(
            "CUDA is not available, so training would run on the CPU and take "
            "weeks. Install the CUDA build:\n"
            "      pip install torch==2.4.1+cu121 torchvision==0.19.1+cu121 "
            "--index-url https://download.pytorch.org/whl/cu121"
        )
    return problems


def check_data() -> list[str]:
    """Compare every split with the expected sizes; return a list of problems."""
    from .data import class_counts, list_images

    problems = []
    print(f"  data root    {paths.DATA_ROOT}")
    try:
        paths.check_data_root()
    except FileNotFoundError as error:
        return [str(error)]

    for step in STEPS:
        expected = EXPECTED[step]
        train_all = list_images(step, "train", include_augmented=True)
        test_all = list_images(step, "test", include_augmented=True)
        test_originals = list_images(step, "test", include_augmented=False)

        def compare(label, actual, reference):
            mark = "ok" if actual == reference else "MISMATCH"
            print(f"  step {step} {label:<22} {actual:>7,}  (expected {reference:,})  {mark}")
            if actual != reference:
                problems.append(
                    f"step {step} {label}: found {actual:,}, expected {reference:,}"
                )

        compare("train (augmented)", len(train_all), expected["train_all"])
        compare("test (augmented)", len(test_all), expected["test_all"])
        compare(
            "test (originals)",
            len(test_originals),
            sum(expected["test_originals"].values()),
        )

        counts = class_counts(test_originals, step)
        if counts != expected["test_originals"]:
            problems.append(
                f"step {step} test class counts: {counts} != {expected['test_originals']}"
            )
        else:
            print(f"  step {step} test class counts     {counts}")
    return problems


def main(argv=None) -> int:
    console.init()
    print("=== environment ===")
    problems = check_torch()
    print("\n=== dataset ===")
    problems += check_data()

    print()
    if problems:
        print(f"{len(problems)} problem(s):")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("All checks passed.")
    print(
        "\nRuns are deterministic by default: the same command with the same seed\n"
        "gives byte-identical weights, including across a resume. Prove it on this\n"
        "machine with `python -m repro.selftest` (about five minutes)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
