"""Paths, device selection and small I/O helpers.

Every location is an environment variable, resolved by `repro.settings`
(environment, else ``paths.toml`` in the repository root, else a default below
the data root); nothing machine-specific lives in the code:

    DERMA_DATA_ROOT       model_data_step{1,2}_anon/{train,test}/<label>/*.jpg
    DERMA_REPRO_OUT       root of every output; this package writes only under
                          <DERMA_REPRO_OUT>/bronet_mlp/.  default <DATA_ROOT>/repro_out
    BRONET_ROOT           clone of ntuaislab/BRONet at commit 9d59c9b
                          default <DATA_ROOT>/pretrained/BRONet
    BRONET_CKPT           folder with imagenet-14x588_399.pth and config.yaml
                          default <DATA_ROOT>/pretrained/bronet_imagenet
    DERMA_BASELINE_ROOTS  where `repro` runs live (McNemar baselines)
                          default <DERMA_REPRO_OUT>
    DERMA_COLOR_REFERENCE the dataset's colour-normalisation reference photograph
                          (`lossy --renorm authors|ref`); no default

Nothing in this module imports torch at import time: `__init__` calls
`apply_defaults` before torch is imported.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

from experiments.common import layout
from experiments.common.device import _cuda_index, device_info, resolve_device  # noqa: F401
from experiments.common.io import (atomic_torch_save, atomic_write_bytes, atomic_write_text,
                                   read_json, sha256_bytes, sha256_file, sha256_json, Timer,
                                   write_csv, write_json, _json_default)  # noqa: F401  (re-exported)
from experiments.common.io import setup_logging as _setup_logging
from experiments.common.locations import bronet_ckpt_dir, bronet_root, data_root, repro_out  # noqa: F401

BRONET_COMMIT = "9d59c9b"
CKPT_NAME = "imagenet-14x588_399.pth"
CKPT_SHA256 = "e8521d477bdeb80b0a623b09a7ce3025b874f21984e50686e869ab61694ca95f"

# BRO frequency matrices of all 8 blocks: 2784 complex64 588x588 = ~7.7 GB.
# Above this much free GPU memory they are kept on the GPU in one piece.
GPU_WEIGHTS_MIN_FREE = 12 * 2**30

BRO_STRATEGIES = ("gpu", "stream", "cpu")


def apply_defaults() -> None:
    """Fill in unset path variables (`repro.settings.apply`); explicit settings
    always win. Needs trichoscopy/src on sys.path (see `__init__`)."""
    from repro import settings

    settings.apply()


# Output folders of this package: experiments.common.layout (names and suffix
# rules shared with resnet_adapt). Runs, tuning and reports of a batch size other
# than 256 live in their own `_bs<B>` folders and files; the feature cache, L_pre
# and the validation split are shared.
DEFAULT_BATCH_SIZE = layout.DEFAULT_BATCH_SIZE
batch_tag = layout.batch_tag
run_name = layout.run_name
_TREE = layout.BRONET


def out_root() -> Path:
    """Everything this package writes goes below here, never into the repo."""
    return _TREE.root()


def backbone_dir() -> Path:
    return _TREE.backbone_dir()


def features_dir(step: int) -> Path:
    return _TREE.features_dir(step)


def runs_dir(step: int, batch_size: int = DEFAULT_BATCH_SIZE) -> Path:
    return _TREE.runs_dir(step, batch_size)


def run_dir(step: int, mode: str, seed: int, variant: str = "best",
            batch_size: int = DEFAULT_BATCH_SIZE) -> Path:
    return _TREE.run_dir(step, mode, seed, variant, batch_size)


def tuning_dir(step: int, batch_size: int = DEFAULT_BATCH_SIZE, tag: str = "") -> Path:
    return _TREE.tuning_dir(step, batch_size, tag)


def results_dir() -> Path:
    return _TREE.results_dir()


def baseline_roots() -> list[Path]:
    """Where `repro` runs of the paper's models live (their predictions_final.csv
    feed the McNemar tests). DERMA_BASELINE_ROOTS, os.pathsep-separated; default
    <DERMA_REPRO_OUT>, read now so that it follows a changed DERMA_REPRO_OUT."""
    spec = os.environ.get("DERMA_BASELINE_ROOTS")
    if spec:
        return [Path(p) for p in spec.split(os.pathsep) if p]
    return [repro_out()]


def env_snapshot() -> dict:
    return {k: os.environ.get(k) for k in
            ("DERMA_DATA_ROOT", "DERMA_REPRO_OUT", "BRONET_ROOT", "BRONET_CKPT",
             "CUBLAS_WORKSPACE_CONFIG")}


# ---------------------------------------------------------------------------
# BRO weight placement
# ---------------------------------------------------------------------------


def resolve_bro_strategy(device, spec: str = "auto") -> str:
    """Where the ~7.7 GB of BRO eval_weights live during a forward pass.

    gpu     all on the GPU (needs >= 12 GB free): fastest
    stream  on the CPU, one block at a time copied to the GPU while it runs
    cpu     everything on the CPU (no GPU at all; slow, for smoke tests)
    """
    import torch

    if spec != "auto":
        if spec not in BRO_STRATEGIES:
            raise ValueError(f"--bro-weights must be auto or one of {BRO_STRATEGIES}")
        if spec != "cpu" and device.type != "cuda":
            raise ValueError(f"--bro-weights {spec} needs a CUDA device, got {device}")
        if spec == "cpu" and device.type == "cuda":
            raise ValueError("--bro-weights cpu needs --device cpu")
        return spec
    if device.type != "cuda":
        return "cpu"
    free, _ = torch.cuda.mem_get_info(_cuda_index(device))
    return "gpu" if free >= GPU_WEIGHTS_MIN_FREE else "stream"


def checkpoint_sha256(verify: bool = True) -> str:
    """sha256 of the BRONet checkpoint, cached by (size, mtime).

    Hashing 1.6 GB takes a few seconds; the cache sits under backbone/.
    """
    ckpt = bronet_ckpt_dir() / CKPT_NAME
    if not ckpt.is_file():
        raise FileNotFoundError(
            f"BRONet checkpoint not found: {ckpt}\n"
            "Set BRONET_CKPT or run scripts/setup_bronet.(ps1|sh).")
    stat = ckpt.stat()
    key = {"path": str(ckpt), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    cache = backbone_dir() / "ckpt_sha256.json"
    digest = None
    if cache.exists():
        try:
            cached = read_json(cache)
            if cached.get("key") == key:
                digest = cached["sha256"]
        except (OSError, ValueError, KeyError):
            pass
    if digest is None:
        digest = sha256_file(ckpt)
        write_json(cache, {"key": key, "sha256": digest})
    if verify and digest != CKPT_SHA256:
        raise RuntimeError(f"{ckpt} has sha256 {digest}, expected {CKPT_SHA256}")
    return digest


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------

def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=Path(__file__).resolve().parent,
                             capture_output=True, check=True)
        return out.stdout.decode("utf-8", errors="replace")
    except (OSError, subprocess.CalledProcessError):
        return None


def git_info() -> dict:
    """Branch, HEAD, and a hash of everything uncommitted.

    The uncommitted part includes untracked files: this whole package starts
    life untracked, so `git diff` alone would not see it.
    """
    head = _git("rev-parse", "HEAD")
    if head is None:
        return {"available": False}
    branch = (_git("rev-parse", "--abbrev-ref", "HEAD") or "").strip()
    diff = _git("diff", "HEAD") or ""
    untracked = [p for p in (_git("ls-files", "--others", "--exclude-standard",
                                  "--", ".") or "").splitlines() if p]
    digest = hashlib.sha256(diff.encode())
    here = Path(__file__).resolve().parent
    for rel in sorted(untracked):
        path = here / rel
        if path.is_file():
            digest.update(rel.encode())
            digest.update(path.read_bytes())
    status = (_git("status", "--porcelain") or "").strip()
    return {"available": True, "branch": branch, "head": head.strip(),
            "dirty": bool(status), "uncommitted_sha256": digest.hexdigest(),
            "untracked_files_in_package": len(untracked)}


def code_sha256() -> str:
    """Hash of the .py files of this package and of experiments/common, which it
    uses (independent of git; tests excluded)."""
    here = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for root, prefix in ((here, ""), (here.parent / "common", "common/")):
        for path in sorted(root.rglob("*.py")):
            if "tests" in path.relative_to(root).parts:
                continue
            digest.update((prefix + path.relative_to(root).as_posix()).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def setup_logging(log_file: Path | None = None, name: str = "bronet_mlp"):
    """`experiments.common.io.setup_logging` with this package's logger name."""
    return _setup_logging(log_file, name)
