"""Prove that runs are reproducible, on this machine, right now.

    python -m repro.selftest

Trains the same small configuration four ways and checks that all four produce
byte-identical weights and identical metrics:

    straight          the reference run
    straight again    a second run of the same command
    resumed           one epoch at a time, stopped and restarted between each
    fewer workers     same command with a different --num-workers

Each is a separate process, so this tests reproducibility the way it is
actually used, not just repeatability inside one interpreter. Roughly five
minutes on an RTX 4070; nothing is written outside a temporary folder.

Exit code 0 if every check passes, 1 otherwise.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from . import console

# Small but not trivial: two epochs is enough for a divergence in epoch 1 to
# propagate, and for the resume path to be exercised at a real boundary.
ARCHITECTURE = "efficientnet_b0"
STEP = 2
EPOCHS = 2
LIMIT_TRAIN = 320


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _train(output_root: Path, *, epochs: int, resume: bool, workers: int,
           extra: list[str] | None = None) -> None:
    command = [
        sys.executable, "-m", "repro.train",
        "--step", str(STEP),
        "--arch", ARCHITECTURE,
        "--epochs", str(epochs),
        "--limit-train", str(LIMIT_TRAIN),
        "--num-workers", str(workers),
        "--log-every", "0",
    ]
    if resume:
        command.append("--resume")
    command += extra or []

    environment = {**_environ(), "DERMA_REPRO_OUT": str(output_root)}
    result = subprocess.run(command, env=environment, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"training failed (exit {result.returncode}):\n{result.stdout}\n{result.stderr}"
        )


def _environ() -> dict:
    import os

    environment = dict(os.environ)
    # The child process must import this package; when repro is not installed,
    # it is found through PYTHONPATH.
    source_root = str(Path(__file__).resolve().parents[1])
    existing = environment.get("PYTHONPATH", "")
    if source_root not in existing.split(os.pathsep):
        environment["PYTHONPATH"] = (
            f"{source_root}{os.pathsep}{existing}" if existing else source_root
        )
    return environment


def _fingerprints(output_root: Path) -> dict:
    """Everything that must match: checkpoint bytes and recorded metrics."""
    run_dir = output_root / "models" / f"step_{STEP}" / ARCHITECTURE
    result = {}
    for name in ("final.pth", "best.pth"):
        result[name] = _digest(run_dir / name)
    for name in ("metrics_final.json", "metrics_best.json"):
        stored = json.loads((run_dir / name).read_text(encoding="utf-8"))
        result[name] = json.dumps(stored["report"], sort_keys=True)
        result[f"{name}:fingerprint"] = stored["fingerprint"]
    for name in ("predictions_final.csv", "predictions_best.csv"):
        result[name] = _digest(run_dir / name)
    return result


def _compare(label: str, reference: dict, candidate: dict) -> bool:
    differing = [key for key in reference if reference[key] != candidate.get(key)]
    if differing:
        print(f"  FAIL  {label}")
        for key in differing:
            print(f"          {key} differs")
        return False
    print(f"  ok    {label}")
    return True


def main(argv=None) -> int:
    console.init()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--keep", action="store_true",
                        help="do not delete the temporary run folder")
    args = parser.parse_args(argv)

    workspace = Path(tempfile.mkdtemp(prefix="repro_selftest_"))
    print(
        f"Step-{STEP} / {ARCHITECTURE}, {EPOCHS} epochs on {LIMIT_TRAIN} images, "
        f"four ways.\nWorkspace: {workspace}\n",
        flush=True,
    )
    try:
        print("training reference run...", flush=True)
        _train(workspace / "reference", epochs=EPOCHS, resume=False, workers=4)
        reference = _fingerprints(workspace / "reference")

        print("training repeat run...", flush=True)
        _train(workspace / "repeat", epochs=EPOCHS, resume=False, workers=4)

        print("training resumed run, one epoch per process...", flush=True)
        for epoch in range(1, EPOCHS + 1):
            _train(workspace / "resumed", epochs=epoch, resume=True, workers=4)

        print("training run with a different worker count...", flush=True)
        _train(workspace / "workers", epochs=EPOCHS, resume=False, workers=1)

        print("\nresults:", flush=True)
        passed = all([
            _compare("a second identical run matches", reference,
                     _fingerprints(workspace / "repeat")),
            _compare("an interrupted-and-resumed run matches", reference,
                     _fingerprints(workspace / "resumed")),
            _compare("a different --num-workers matches", reference,
                     _fingerprints(workspace / "workers")),
        ])

        print()
        if passed:
            print("Reproducible: identical weights, metrics and predictions in every case.")
            print(f"  final.pth sha256 {reference['final.pth'][:32]}...")
            return 0
        print("NOT reproducible. Something in the run is still unpinned.")
        return 1
    finally:
        if args.keep:
            print(f"\nkept {workspace}")
        else:
            shutil.rmtree(workspace, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
