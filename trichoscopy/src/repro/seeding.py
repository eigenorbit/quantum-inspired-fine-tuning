"""Determinism: seeding, algorithm selection, and RNG state across a resume.

Running the same command twice must produce the same weights and the same
metrics, bit for bit. Four separate things have to be pinned for that to hold.

**1. The seeds.** Python's `random`, numpy and torch (CPU and every CUDA
device) each have their own generator. `set_seed` covers all of them.

**2. The algorithms.** cuDNN picks convolution algorithms by benchmarking them
at runtime, and the winner depends on machine load, so two runs can take
different code paths and accumulate floating-point error differently.
`configure_determinism` turns benchmarking off, asks cuDNN for deterministic
kernels, and switches on torch's global deterministic mode, which makes any
remaining nondeterministic op raise instead of silently varying. All three
architectures used here run clean under it.

cuBLAS also needs `CUBLAS_WORKSPACE_CONFIG` to be set *before* the CUDA context
is created, which is too early for this module -- `repro/__init__.py` sets it at
import time, before anything can import torch.

**3. The data order.** Handled in `data.build_loaders`: `EpochSeededSampler`
draws each epoch's order from a generator seeded by (seed, epoch), and no random
transform runs in the workers. Batch composition is therefore independent of
`--num-workers` and of a resume.

**4. The RNG state across a resume.** Seeding at startup is not enough: a run
resumed at epoch 20 would restart its generators from the initial seed, while
an uninterrupted run would be 19 epochs deep into the stream. Dropout masks
would differ from that point on. `capture`/`restore` move the full generator
state through the checkpoint so a resumed run continues the same stream.

That fixes dropout, but not the shuffling order: the loader's sampler has its
own `torch.Generator`, which `torch.set_rng_state` does not touch, so a resumed
run would re-draw epoch 1's permutation for epoch 20. Rather than carry that
generator through the checkpoint, the training loader uses
`data.EpochSeededSampler`, whose permutation is derived from `(seed, epoch)`
alone; `seed_epoch` tells it which epoch is starting. The order of any given
epoch then depends only on those two numbers -- never on how the run got there.

The cost of determinism on this workload is about 5-10% throughput. Pass
`--no-deterministic` to trade reproducibility for that.
"""

from __future__ import annotations

import random

import numpy as np
import torch


def configure_determinism(enabled: bool = True) -> None:
    """Pin (or release) algorithm selection.

    With `enabled`, any op lacking a deterministic implementation raises a
    `RuntimeError` naming itself, rather than quietly producing run-to-run
    variation.
    """
    torch.backends.cudnn.deterministic = enabled
    torch.backends.cudnn.benchmark = not enabled
    torch.use_deterministic_algorithms(enabled)


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch, on CPU and on every CUDA device."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_epoch(loader, epoch: int) -> None:
    """Tell the training sampler which epoch it is about to serve.

    See `data.EpochSeededSampler`: the permutation is derived from the seed and
    the epoch number, so a resumed epoch sees the batches an uninterrupted one
    would.
    """
    sampler = getattr(loader, "sampler", None)
    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)


def capture() -> dict:
    """Snapshot every generator, in a form `torch.load(weights_only=True)` accepts.

    numpy and `random` hand back tuples containing numpy arrays, which the
    restricted unpickler rejects; both are converted to tensors here and back
    in `restore`.
    """
    python_version, python_keys, python_gauss = random.getstate()
    numpy_kind, numpy_keys, numpy_pos, numpy_has_gauss, numpy_gauss = np.random.get_state()

    state = {
        "python_version": python_version,
        "python_keys": torch.tensor(list(python_keys), dtype=torch.int64),
        "python_gauss": python_gauss,
        "numpy_kind": numpy_kind,
        "numpy_keys": torch.tensor(np.asarray(numpy_keys, dtype=np.int64)),
        "numpy_pos": int(numpy_pos),
        "numpy_has_gauss": int(numpy_has_gauss),
        "numpy_gauss": float(numpy_gauss),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore(state: dict | None) -> bool:
    """Put the generators back where `capture` found them.

    Returns False when there is nothing to restore -- a checkpoint written
    before RNG state was tracked -- so the caller can warn that the resumed run
    will not match an uninterrupted one.

    Every tensor is pulled back to the CPU first: the checkpoint is loaded with
    `map_location=device`, which moves these state tensors onto the GPU along
    with the weights, and generator state has to be set from host memory.
    """
    if not state:
        return False

    random.setstate((
        state["python_version"],
        tuple(int(value) for value in state["python_keys"].cpu().tolist()),
        state["python_gauss"],
    ))
    np.random.set_state((
        state["numpy_kind"],
        state["numpy_keys"].cpu().numpy().astype(np.uint32),
        state["numpy_pos"],
        state["numpy_has_gauss"],
        state["numpy_gauss"],
    ))
    torch.set_rng_state(state["torch"].cpu().to(torch.uint8))
    cuda_state = state.get("cuda")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([tensor.cpu().to(torch.uint8) for tensor in cuda_state])
    return True


def fingerprint(model: torch.nn.Module) -> str:
    """A short hash of every parameter, to compare two runs cheaply.

    Written into `metrics.json`; two runs of the same command with the same
    seed must produce the same fingerprint.
    """
    import hashlib

    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()[:16]
