"""Orthogonal channel rotations inside the last BRO blocks of BRONet (option 1b).

    python -m experiments.bronet_mlp.conv_adapt extract --step 2 --blocks 1 --cache-bro-weights
    python -m experiments.bronet_mlp.conv_adapt tune    --step 2 --blocks 1 --mlp-modes butterfly lora --cache-bro-weights
    python -m experiments.bronet_mlp.conv_adapt train   --step 2 --blocks 1 --mlp-modes butterfly lora --from-tuning --seeds 0 1 2 --cache-bro-weights
    python -m experiments.bronet_mlp.conv_adapt summary --step 2 --blocks 1
    python -m experiments.bronet_mlp.conv_adapt export  --step 2 --blocks 1   (numbers only -> zip)

The protocol runs adapt only the MLP after the neck; everything before it is frozen
and phi is cached. Here the last `--blocks` BRO blocks get an adapter as well: in
block i the output of the orthogonal convolution is rotated over its 588 channels
before the bias and the MinMax activation,

    h <- MinMax( R_i(theta_i) Conv_i(h) + b_i ),   R_i = PartitionedButterfly(588)

the convolutional analogue of the MLP butterfly R(theta) W. R_i is exactly
orthogonal (588 = 512 + 64 + 8 + 4, two stages of block-diagonal butterflies mixed
by a fixed permutation, see partitioned.py) and the identity at theta = 0, so
||R_i Conv_i h|| = ||Conv_i h||: L_pre, and with it the certificate's L, is unchanged.
The MLP is adapted as in the protocol (`--mlp-modes`, default butterfly) and the
LLN head is new; both start exactly as the protocol run of the same seed.

Everything else is the protocol (`run_grid --from-tuning`): the same fit /
validation split, cert_ce loss with eps 36/255 and tau 4, Adam + StepLR(2, 0.7),
batch 256, 30 epochs, best-validation and last checkpoints, both evaluated on test
with the same certificate code (`train.evaluate`). The rotations and the MLP have
their own learning rates (an angle and a LoRA weight want rates ~100x apart).

tune     per MLP mode and protocol variant (best,
         stable where it differs), on the protocol's fit / validation split, seed 0:
         stage A, the MLP lr is the protocol's and the rotation lr runs over
         CONV_LR_GRID; stage B, at the chosen rotation lr the MLP lr x 1/3 and x 3, taken
         only if it is MLP_SWITCH_MARGIN points better on validation. Selection as the
         protocol's: best (run, epoch) on validation; `stable` only among runs whose last
         epoch is not collapsed and within 2 points of their best. 6 runs per (mode,
         variant); the test set is never used. -> conv_adapt/tuning/step_<n>/last<k>/
         selection.json and tuning_step<n>_last<k>.md.
         `train --from-tuning` trains the selected rates (folders <mode>[_stable]);
         without it, `--lr` / `--lr-conv` by hand (folders <mode>_manual).

extract  Caches the input of the first adapted block for every train and test image
         (float16 by default: 0.9-1.1 MB per image; step-2 ~13-15 GB, step-1
         ~29-34 GB) as .npy memory maps; resumable. Checks the cache by running the
         frozen rest of BRONet on it and comparing with the phi cache.
train    One run per (MLP mode, seed). A batch of 256 is computed in pieces of
         `--micro-batch` (gradient accumulation; the same update as one batch of 256
         up to float rounding): 64 fits 8 GB for one or two blocks. Resumable.
summary  Mean ± SD over seeds, next to the protocol runs of the same MLP mode.

Only new files and new output folders: nothing of the protocol is read for writing.
Outputs, below <DERMA_REPRO_OUT>/bronet_mlp/: conv_features/step_<n>/last<k>_<dtype>/,
conv_adapt/step_<n>/last<k>/<mlp mode>/seed<s>/, results/conv_adapt/.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from . import env

import numpy as np
import torch
import torch.nn as nn

from experiments.common import layout

from . import adapters, certify, losses, train
from . import data as bdata
from .partitioned import PartitionedButterfly

TREE = layout.BRONET
CACHE_FORMAT = 1
CHECKPOINT_FORMAT = 1
DTYPES = {"float16": (np.float16, torch.float16), "float32": (np.float32, torch.float32)}
SPLITS = ("train", "test")
TUNE_RESULT = "tune_result.json"
# lr tuning: the MLP lr starts at the protocol's
# (variant best / stable); stage A tunes the rotations' lr on CONV_LR_GRID, stage B
# tries the MLP lr x MLP_LR_FACTORS at the chosen rotation lr and switches only if
# it beats stage A by MLP_SWITCH_MARGIN points of validation accuracy (~1 standard
# error on the 1.4k validation images of step-2).
CONV_LR_GRID = (1e-3, 3e-3, 1e-2, 3e-2)
MLP_LR_FACTORS = (1 / 3, 3.0)
MLP_SWITCH_MARGIN = 1.0
TUNE_SEED = 0
VARIANTS = ("best", "stable")


# ---------------------------------------------------------------------------
# cache of the block inputs
# ---------------------------------------------------------------------------

def _rels(paths) -> list[str]:
    return [bdata.relative_path(p) for p in paths]


def cache_meta(step: int, blocks: int, dtype: str, rels: dict) -> dict:
    from . import bronet

    return {"format": CACHE_FORMAT, "step": step, "blocks": blocks, "dtype": dtype,
            "checkpoint_sha256": env.checkpoint_sha256(), "bronet_commit": bronet.bronet_commit(),
            "transform_version": bdata.TRANSFORM_VERSION,
            "records_sha256": env.sha256_json(rels),
            "definition": "input of BRO block depth-blocks: conv blocks 0..depth-blocks-1 of "
                          "conv(conv_1(stem(x - 0.5))), same arithmetic as bronet.phi_layer_major"}


def _index_path(d: Path) -> Path:
    return d / "index.json"


def read_cache(step: int, blocks: int, dtype: str, mmap: bool = True) -> dict:
    """{"train": array (N, C, s, s), "test": array, "index": dict}; errors if the cache
    is missing, incomplete or made from other data."""
    d = TREE.conv_features_dir(step, blocks, dtype)
    if not _index_path(d).exists():
        raise FileNotFoundError(f"no cache in {d}; run  python -m experiments.bronet_mlp."
                                f"conv_adapt extract --step {step} --blocks {blocks}")
    index = env.read_json(_index_path(d))
    train_recs, test_recs, _ = bdata.records(step)
    want = cache_meta(step, blocks, dtype, {"train": _rels(r.path for r in train_recs),
                                            "test": _rels(r.path for r in test_recs)})
    if index["meta"] != want:
        diff = {k: (index["meta"].get(k), v) for k, v in want.items() if index["meta"].get(k) != v}
        raise RuntimeError(f"{d} was made with other settings or data: {diff}")
    out = {"index": index}
    for split in SPLITS:
        n = index["shapes"][split][0]
        if index["done"][split] != n:
            raise RuntimeError(f"{d}: {split} is incomplete ({index['done'][split]}/{n}); "
                               "rerun extract")
        out[split] = np.load(d / f"{split}.npy", mmap_mode="r" if mmap else None)
    return out


@torch.no_grad()
def block_inputs(model, x: torch.Tensor, device, first: int, minibatch: int = 64) -> torch.Tensor:
    """Input of BRO block `first` for a chunk of images, one block at a time (the
    arithmetic of `bronet.phi_layer_major`, stopped before block `first`)."""
    hs = [model.conv_1(model.stem(x[i:i + minibatch].to(device, non_blocking=True).sub(0.5)))
          for i in range(0, len(x), minibatch)]
    c = model.conv
    for i in range(first):
        weight = c.eval_weights[i]
        if weight.device != device:
            weight = weight.to(device, non_blocking=True)
        bias = (c.bias[i] - c.running_mean[i]).view(-1, 1, 1)
        for k in range(len(hs)):
            hs[k] = c.act(c.frequency_convolution(hs[k], weight, i) + bias)
        del weight
    return torch.cat([h.float().cpu() for h in hs])


def extract(step: int, blocks: int, dtype: str, *, device, strategy: str,
            cache_bro_weights: bool, batch_size: int = 64, chunk_size: int = 1024,
            num_workers: int = 4, force: bool = False, logger=None) -> dict:
    """Write (or complete) the cache; returns its index."""
    from torch.utils.data import DataLoader

    from . import bronet, features

    log = logger.info if logger else print
    np_dtype, _ = DTYPES[dtype]
    d = TREE.conv_features_dir(step, blocks, dtype)
    train_recs, test_recs, _ = bdata.records(step, logger)
    paths = {"train": [r.path for r in train_recs], "test": [r.path for r in test_recs]}
    meta = cache_meta(step, blocks, dtype, {k: _rels(v) for k, v in paths.items()})
    index = env.read_json(_index_path(d)) if _index_path(d).exists() else None
    if index is not None and index["meta"] != meta:
        if not force:
            raise RuntimeError(f"{d} holds a cache made with other settings or data; "
                               "--force deletes it")
        for split in SPLITS:
            (d / f"{split}.npy").unlink(missing_ok=True)
        index = None
    if index is not None and all(index["done"][s] == len(paths[s]) for s in SPLITS) \
            and "check" in index:
        log(f"cache {d} complete")
        return index

    model, _ = bronet.load_bronet(device, strategy, logger, cache_bro_weights=cache_bro_weights)
    c = model.conv
    first = c.depth - blocks
    if not 0 <= first < c.depth:
        raise ValueError(f"--blocks must be 1..{c.depth}")
    side = c.padding_input_sizes[first] - 2 * (c.kernel_size // 2) if c.zero_padding \
        else c.padding_input_sizes[first]
    shape = (c.width, side, side)
    if index is None:
        index = {"meta": meta, "first_block": first, "item_shape": list(shape),
                 "shapes": {s: [len(paths[s]), *shape] for s in SPLITS},
                 "done": {s: 0 for s in SPLITS}, "passes": []}
        d.mkdir(parents=True, exist_ok=True)
        env.write_json(_index_path(d), index)
    mb = shape[0] * shape[1] * shape[2] * np.dtype(np_dtype).itemsize / 1e6
    log(f"conv cache step-{step}, last {blocks} block(s) (input of block {first}): "
        f"{shape} {dtype}, {mb:.2f} MB/image, "
        f"{sum(len(p) for p in paths.values()) * mb / 1e3:.1f} GB in total -> {d}")

    t0 = time.perf_counter()
    for split in SPLITS:
        n = len(paths[split])
        file = d / f"{split}.npy"
        arr = np.lib.format.open_memmap(file, mode="r+" if file.exists() else "w+",
                                        dtype=np_dtype, shape=(n, *shape))
        start = index["done"][split]
        if start >= n:
            continue
        loader = DataLoader(bdata.ImageDataset(paths[split][start:]), batch_size=batch_size,
                            shuffle=False, num_workers=num_workers,
                            pin_memory=device.type == "cuda")
        xs, done, t1 = [], start, time.perf_counter()

        def flush(xs):
            nonlocal done
            h = block_inputs(model, torch.cat(xs), device, first, batch_size)
            arr[done:done + len(h)] = h.numpy().astype(np_dtype)
            arr.flush()
            done += len(h)
            index["done"][split] = done
            env.write_json(_index_path(d), index)
            rate = (done - start) / max(time.perf_counter() - t1, 1e-9)
            log(f"  {split}: {done:,}/{n:,}  {rate:.1f} img/s  ETA {(n - done) / max(rate, 1e-9) / 60:.1f} min")

        try:
            for x, _ in loader:
                xs.append(x)
                if sum(len(t) for t in xs) >= chunk_size:
                    flush(xs)
                    xs = []
            if xs:
                flush(xs)
        except torch.cuda.OutOfMemoryError as exc:
            raise RuntimeError(f"CUDA out of memory; lower --chunk-size ({chunk_size}) or "
                               f"--batch-size ({batch_size}). Progress is saved.") from exc
        del arr

    # check: the frozen rest of BRONet on the cached input must give the phi cache
    k = min(64, len(paths["test"]))
    h = torch.from_numpy(np.load(d / "test.npy", mmap_mode="r")[:k].astype(np.float32))
    tail = BroTail(model, first, device)
    with torch.no_grad():
        phi_c = torch.cat([tail.phi(h[i:i + 16].to(device)).cpu() for i in range(0, k, 16)])
    ref = features.load(step, paths["test"][:k])
    diff = float((phi_c - ref).abs().max())
    scale = float(ref.abs().max())
    index["check"] = {"images": k, "phi_max_abs_diff": diff, "phi_max_abs": scale}
    index["passes"].append({"seconds": time.perf_counter() - t0, "strategy": strategy,
                            **env.device_info(device)})
    env.write_json(_index_path(d), index)
    log(f"check: phi from the cache vs the phi cache, max |diff| {diff:.2e} "
        f"(max |phi| {scale:.2f}, {dtype})")
    return index


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------

class BroTail:
    """The frozen last BRO blocks and the neck of a loaded BRONet. Not an nn.Module:
    nothing of BRONet is trained or saved. The blocks' frequency matrices stay on
    `device` (~1 GB per block)."""

    def __init__(self, model, first: int, device):
        self.c, self.neck = model.conv, model.neck
        for p in model.parameters():
            p.requires_grad_(False)
        self.blocks = list(range(first, self.c.depth))
        self.width = self.c.width
        self.weights = [self.c.eval_weights[i].to(device) for i in self.blocks]
        self._bias = [(self.c.bias[i] - self.c.running_mean[i]).view(-1, 1, 1).to(device)
                      for i in self.blocks]

    def conv(self, h, j: int):
        return self.c.frequency_convolution(h, self.weights[j], self.blocks[j])

    def post(self, z, j: int):
        return self.c.act(z + self._bias[j])

    def phi(self, h):
        """The frozen function (no rotation): neck(blocks(h))."""
        for j in range(len(self.blocks)):
            h = self.post(self.conv(h, j), j)
        return self.neck(h)


class ConvAdapted(nn.Module):
    """Rotations R_j inside the tail's blocks, then the neck, then an
    `adapters.AdaptedClassifier` (MLP in its mode + LLN head)."""

    def __init__(self, tail, clf: adapters.AdaptedClassifier, stages: int = 2,
                 perm_seed: int = 0):
        super().__init__()
        self.tail = tail
        self.rotations = nn.ModuleList(PartitionedButterfly(tail.width, stages, perm_seed + j)
                                       for j in range(len(tail.blocks)))
        self.clf = clf

    @property
    def mlp(self):
        return self.clf.mlp

    @property
    def head(self):
        return self.clf.head

    def features(self, h: torch.Tensor) -> torch.Tensor:
        for j, rot in enumerate(self.rotations):
            h = self.tail.post(rot(self.tail.conv(h, j)), j)
        return self.tail.neck(h)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.clf(self.features(h))

    def trainable_state(self) -> dict[str, torch.Tensor]:
        return {n: p.detach().cpu().clone() for n, p in self.named_parameters() if p.requires_grad}

    def load_trainable_state(self, state: dict[str, torch.Tensor]) -> None:
        own = dict(self.named_parameters())
        want = {n for n, p in own.items() if p.requires_grad}
        if set(state) != want:
            raise KeyError(f"trainable state mismatch: missing {sorted(want - set(state))}, "
                           f"unexpected {sorted(set(state) - want)}")
        with torch.no_grad():
            for n, t in state.items():
                own[n].copy_(t.to(own[n].device, own[n].dtype))

    def num_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def orthogonality_error(self) -> float:
        return max(r.orthogonality_error() for r in self.rotations)


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

@dataclass
class ConvConfig:
    step: int
    blocks: int = 1
    mlp_mode: str = "butterfly"
    seed: int = 0
    epochs: int = 30
    batch_size: int = 256
    micro_batch: int = 64
    lr: float = 1e-3
    lr_conv: float = 1e-3
    lr_source: str = "given"          # "given" | "tuning/step_<n>/selection.json (best)"
    weight_decay: float = 0.0
    lr_step_size: int = 2
    lr_gamma: float = 0.7
    loss: str = "cert_ce"
    tau: float = 4.0
    eps_train: float = 36 / 255
    lora_rank: int = 4
    lip_power_iters: int = 100
    stages: int = 2
    perm_seed: int = 0
    cache_dtype: str = "float16"
    deterministic: bool = True
    device: str = "auto"              # runtime only

    RUNTIME = ("device",)

    def __post_init__(self):
        if self.mlp_mode not in adapters.MODES:
            raise ValueError(f"mlp_mode must be one of {adapters.MODES}")
        if self.cache_dtype not in DTYPES:
            raise ValueError(f"cache_dtype must be one of {tuple(DTYPES)}")
        if self.batch_size % self.micro_batch:
            raise ValueError("batch_size must be a multiple of micro_batch")

    def compat(self) -> dict:
        return {k: v for k, v in dataclasses.asdict(self).items() if k not in self.RUNTIME}

    def run_config(self) -> train.RunConfig:
        """The protocol's RunConfig of the MLP part (for evaluation and outputs)."""
        return train.RunConfig(step=self.step, mode=self.mlp_mode, seed=self.seed,
                               epochs=self.epochs, batch_size=self.batch_size, lr=self.lr,
                               weight_decay=self.weight_decay, lr_step_size=self.lr_step_size,
                               lr_gamma=self.lr_gamma, loss=self.loss, tau=self.tau,
                               eps_train=self.eps_train, lora_rank=self.lora_rank,
                               lip_power_iters=self.lip_power_iters,
                               deterministic=self.deterministic, early_stop="val",
                               device=self.device)


def protocol_lr(step: int, mlp_mode: str) -> float | None:
    """The learning rate tuned for `mlp_mode` (variant best), if the selection exists."""
    from . import tune

    if not tune.selection_path(step).exists():
        return None
    return tune.tuned_params(step, mlp_mode, train.RunConfig(step=step, mode=mlp_mode))["lr"]


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

@dataclass
class ConvData:
    h_train: np.ndarray            # (N_train, C, s, s), memory map
    h_test: torch.Tensor           # (N_test, C, s, s), in RAM, cache dtype
    y_train: torch.Tensor
    fit_idx: np.ndarray
    val_idx: np.ndarray
    inputs: train.RunInputs        # protocol inputs (labels, test rows, L, base, provenance)
    cache_sha256: str


def load_data(step: int, blocks: int, dtype: str, logger=None) -> ConvData:
    from . import tune

    cache = read_cache(step, blocks, dtype)
    base = train.prepare(step, logger)
    split = tune.make_split(base, step, save=False)
    rows = [row[0] for row in base.manifest["train"]]
    if len(rows) != cache["train"].shape[0]:
        raise RuntimeError("cache and protocol train sets differ in size")
    inputs = tune.final_inputs(base, split)
    return ConvData(h_train=cache["train"], h_test=torch.from_numpy(np.asarray(cache["test"])),
                    y_train=base.train_y, fit_idx=np.asarray(split["fit_idx"]),
                    val_idx=np.asarray(split["val_idx"]), inputs=inputs,
                    cache_sha256=env.sha256_json(cache["index"]["meta"]))


def _rows(h, idx: np.ndarray) -> torch.Tensor:
    """Rows `idx` (sorted) of a memory map or tensor, as float32."""
    if isinstance(h, torch.Tensor):
        return h[torch.as_tensor(idx)].float()
    return torch.from_numpy(np.ascontiguousarray(h[idx])).float()


@torch.no_grad()
def predict(model: ConvAdapted, h, idx: np.ndarray, device, micro: int) -> torch.Tensor:
    model.eval()
    out = []
    for i in range(0, len(idx), micro):
        out.append(model(_rows(h, idx[i:i + micro]).to(device)).cpu())
    return torch.cat(out)


@torch.no_grad()
def features_of(model: ConvAdapted, h, device, micro: int) -> torch.Tensor:
    model.eval()
    idx = np.arange(len(h))
    return torch.cat([model.features(_rows(h, idx[i:i + micro]).to(device)).cpu()
                      for i in range(0, len(idx), micro)])


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------

def is_complete(run_dir: Path) -> bool:
    return train.is_complete(run_dir)


def build_model(cfg: ConvConfig, tail, inputs: train.RunInputs, num_classes: int,
                device) -> ConvAdapted:
    clf = adapters.build(inputs.base_weight, inputs.base_bias, cfg.mlp_mode, num_classes,
                         cfg.lora_rank, cfg.seed)
    return ConvAdapted(tail, clf, cfg.stages, cfg.perm_seed).to(device)


def _checkpoint(model, opt, sched, epoch, global_step, cfg, data, best) -> dict:
    from repro import seeding

    state = model.trainable_state()
    return {"format": CHECKPOINT_FORMAT, "trainable_state": state,
            "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
            "epoch": epoch, "global_step": global_step, "rng": seeding.capture(),
            "config": cfg.compat(), "cache_sha256": data.cache_sha256,
            "data_manifest_sha256": data.inputs.provenance["data_manifest_sha256"],
            "mlp_base_sha256": data.inputs.provenance.get("mlp_base_sha256", ""),
            "fingerprint": adapters.fingerprint(state), "best": best}


def _check_resume(ckpt: dict, cfg: ConvConfig, data: ConvData, path: Path) -> None:
    problems = []
    if ckpt["config"] != cfg.compat():
        problems.append(f"config: {ckpt['config']} vs {cfg.compat()}")
    for key, now in (("cache_sha256", data.cache_sha256),
                     ("data_manifest_sha256", data.inputs.provenance["data_manifest_sha256"]),
                     ("mlp_base_sha256", data.inputs.provenance.get("mlp_base_sha256", ""))):
        if ckpt.get(key) != now:
            problems.append(f"{key} differs")
    if problems:
        raise RuntimeError(f"{path} is not compatible with this run: {problems}; "
                           "--restart trains from scratch")


def fit(cfg: ConvConfig, data: ConvData, tail, run_dir: Path, *, logger=None,
        restart: bool = False, stop_after: int | None = None,
        evaluate_test: bool = True) -> dict | None:
    """Train one run. With `evaluate_test=False` (lr tuning) the test set is never
    touched: the run ends with tune_result.json (validation only)."""
    from repro import metrics as rmetrics
    from repro import seeding

    log = logger.info if logger else (lambda *_: None)
    run_dir = Path(run_dir)
    ckdir = run_dir / layout.CHECKPOINTS_DIR
    device = env.resolve_device(cfg.device)
    if restart and run_dir.exists():
        train._clear_outputs(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    seeding.configure_determinism(cfg.deterministic)
    seeding.set_seed(cfg.seed)
    inputs = data.inputs
    if inputs.L_mlp0 is None:
        inputs.L_mlp0 = certify.lmlp_exact(inputs.base_weight)
    num_classes = int(max(inputs.train_y.max(), inputs.test_y.max())) + 1
    model = build_model(cfg, tail, inputs, num_classes, device)
    conv_params = [p for p in model.rotations.parameters() if p.requires_grad]
    clf_params = [p for p in model.clf.parameters() if p.requires_grad]
    opt = torch.optim.Adam([{"params": conv_params, "lr": cfg.lr_conv},
                            {"params": clf_params, "lr": cfg.lr}],
                           weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=cfg.lr_step_size, gamma=cfg.lr_gamma)

    # at theta = 0 the model is the protocol model of this seed: phi must match its cache
    k = min(64, len(data.h_test))
    init_diff = float((features_of(model, data.h_test[:k], device, cfg.micro_batch)
                       - inputs.test_x[:k]).abs().max())
    record = {"config": dataclasses.asdict(cfg), "run_config": cfg.run_config().to_dict(),
              "provenance": inputs.provenance, "cache_sha256": data.cache_sha256,
              "trainable_params": model.num_trainable(),
              "conv_angles": sum(r.num_angles for r in model.rotations),
              "adapted_blocks": tail.blocks, "init_phi_max_abs_diff": init_diff,
              "n_fit": len(data.fit_idx), "n_val": len(data.val_idx), "n_test": len(data.h_test),
              "env": env.env_snapshot(), "device": env.device_info(device),
              "git": env.git_info(), "command": sys.argv}
    env.write_json(run_dir / "config.json", record)

    history, start_epoch, global_step, best = [], 0, 0, None
    last = ckdir / layout.checkpoint_file("last")
    if last.exists():
        ckpt = torch.load(last, map_location="cpu", weights_only=True)
        _check_resume(ckpt, cfg, data, last)
        model.load_trainable_state(ckpt["trainable_state"])
        opt.load_state_dict(ckpt["optimizer"])
        sched.load_state_dict(ckpt["scheduler"])
        seeding.restore(ckpt["rng"])
        start_epoch, global_step, best = ckpt["epoch"] + 1, ckpt["global_step"], ckpt["best"]
        history = [h for h in env.read_json(run_dir / "history.json") if h["epoch"] < start_epoch]
        log(f"resumed from {last} at epoch {start_epoch}")

    fit_idx, val_idx = data.fit_idx, data.val_idx
    y_val = data.y_train[torch.as_tensor(val_idx)]
    counts = torch.bincount(y_val, minlength=num_classes).double()
    val_majority = 100 * float(counts.max() / counts.sum())
    fixed_L = cfg.mlp_mode in ("head-only", "butterfly")
    log(f"step-{cfg.step} last {len(tail.blocks)} BRO block(s) + MLP {cfg.mlp_mode}, seed "
        f"{cfg.seed}: {model.num_trainable():,} trainable, fit {len(fit_idx):,} / val "
        f"{len(val_idx):,} / test {len(data.h_test):,}, lr {cfg.lr:g} (conv {cfg.lr_conv:g}), "
        f"batch {cfg.batch_size} in pieces of {cfg.micro_batch}; phi at init vs cache "
        f"{init_diff:.2e}")

    for epoch in range(start_epoch, cfg.epochs):
        t0 = time.perf_counter()
        model.train()
        L_mlp = (inputs.L_mlp0["L_mlp"] if fixed_L
                 else certify.lmlp_power(model.mlp.effective_matrices(), cfg.lip_power_iters))
        L_train = inputs.L_pre_official * L_mlp
        g = torch.Generator().manual_seed(cfg.seed * 1_000_003 + epoch)
        perm = fit_idx[torch.randperm(len(fit_idx), generator=g).numpy()]
        tot_loss, tot_correct = 0.0, 0
        for b in range(0, len(perm), cfg.batch_size):
            batch = perm[b:b + cfg.batch_size]
            opt.zero_grad(set_to_none=True)
            for m in range(0, len(batch), cfg.micro_batch):
                idx = np.sort(batch[m:m + cfg.micro_batch])
                h = _rows(data.h_train, idx).to(device)
                y = data.y_train[torch.as_tensor(idx)].to(device)
                logits = model(h)
                loss = losses.compute(cfg.loss, logits, y, model.head.get_weight(), L_train,
                                      cfg.eps_train, cfg.tau)
                (loss * (len(idx) / len(batch))).backward()
                tot_loss += float(loss) * len(idx)
                tot_correct += int((logits.argmax(1) == y).sum())
            opt.step()
            global_step += 1
        sched.step()
        train_seconds = time.perf_counter() - t0

        vl = predict(model, data.h_train, np.sort(val_idx), device, cfg.micro_batch)
        y_sorted = data.y_train[torch.as_tensor(np.sort(val_idx))]
        vq = rmetrics.overall(rmetrics.Predictions(
            y_true=y_sorted.numpy(), y_pred=vl.argmax(1).numpy(),
            probabilities=np.zeros((len(y_sorted), num_classes))))
        collapsed = bool(vq["accuracy"] <= val_majority + 1e-6)
        key = [int(collapsed), -float(vq["accuracy"]), -float(vq["f1_weighted"])]
        is_best = best is None or key < best["key"]
        if is_best:
            best = {"epoch": epoch, "key": key, "val_accuracy": float(vq["accuracy"]),
                    "val_f1_weighted": float(vq["f1_weighted"]), "val_collapsed": collapsed}
            env.atomic_torch_save(_checkpoint(model, opt, sched, epoch, global_step, cfg, data,
                                              best), ckdir / layout.checkpoint_file("best"))
        history.append({"epoch": epoch, "lr": opt.param_groups[1]["lr"],
                        "lr_conv": opt.param_groups[0]["lr"], "loss": tot_loss / len(perm),
                        "train_accuracy_running": 100 * tot_correct / len(perm),
                        "val_accuracy": float(vq["accuracy"]),
                        "val_f1_weighted": float(vq["f1_weighted"]), "val_collapsed": collapsed,
                        "is_best_so_far": is_best, "L_train": L_train,
                        "conv_orthogonality_error": model.orthogonality_error(),
                        "seconds_train": train_seconds,
                        "seconds_epoch": time.perf_counter() - t0})
        env.atomic_torch_save(_checkpoint(model, opt, sched, epoch, global_step, cfg, data, best),
                              last)
        env.write_json(run_dir / "history.json", history)
        hh = history[-1]
        log(f"  epoch {epoch + 1:2d}/{cfg.epochs}  loss {hh['loss']:.4f}  train "
            f"{hh['train_accuracy_running']:.2f}  val {hh['val_accuracy']:.2f}"
            f"{'*' if is_best else ' '}  |R^T R - I| {hh['conv_orthogonality_error']:.1e}  "
            f"({hh['seconds_epoch']:.0f}s)")
        if stop_after is not None and epoch + 1 >= stop_after and epoch + 1 < cfg.epochs:
            log(f"stopping after epoch {epoch + 1} (--stop-after)")
            return None

    final = _checkpoint(model, opt, sched, cfg.epochs - 1, global_step, cfg, data, best)
    env.atomic_torch_save(final, ckdir / layout.checkpoint_file("final"))
    if not evaluate_test:
        last_h = history[-1]
        result = {"config": cfg.compat(), "best": best,
                  "last_epoch": {"epoch": cfg.epochs - 1, "val_accuracy": last_h["val_accuracy"],
                                 "val_f1_weighted": last_h["val_f1_weighted"],
                                 "val_collapsed": last_h["val_collapsed"]},
                  "val_majority_rate": val_majority,
                  "conv_orthogonality_error": model.orthogonality_error()}
        env.write_json(run_dir / TUNE_RESULT, result)
        log(f"tuning run done: best val {best['val_accuracy']:.2f} (epoch {best['epoch'] + 1}), "
            f"last {last_h['val_accuracy']:.2f}")
        return result
    run_cfg = cfg.run_config()
    common = {"config": run_cfg.to_dict(), "conv_config": dataclasses.asdict(cfg),
              "provenance": inputs.provenance, "trainable_params": model.num_trainable(),
              "epochs_done": cfg.epochs, "device": env.device_info(device),
              "history_seconds_train_sum": sum(h["seconds_epoch"] for h in history),
              "early_stopping": {**best, "best_epoch_1based": best["epoch"] + 1,
                                 "n_val": len(val_idx), "val_majority_rate": val_majority}}
    by_ckpt = {}
    for which, ck in (("final", final),
                      ("best", torch.load(ckdir / layout.checkpoint_file("best"),
                                          map_location="cpu", weights_only=True))):
        model.load_trainable_state(ck["trainable_state"])
        by_ckpt[which] = evaluate(model, run_cfg, data, device, run_dir, which,
                                  {**common, "fingerprint": ck["fingerprint"], "epoch": ck["epoch"]})
    for which, m in by_ckpt.items():
        log(f"{which} (epoch {m['epoch'] + 1}): test acc {m['accuracy']:.2f}  "
            f"median r {m['certificate_official']['median_radius']:.4f}  cert@36/255 "
            f"{100 * m['certificate_official']['certified_accuracy_named']['36/255']:.1f}%  "
            f"|R^T R - I| {m['conv_rotations']['orthogonality_error_max']:.1e}")
    return train.write_headline(run_dir, run_cfg, by_ckpt)


def evaluate(model: ConvAdapted, run_cfg: train.RunConfig, data: ConvData, device,
             run_dir: Path, which: str, extra: dict) -> dict:
    """The protocol's evaluation (`train.evaluate`) on phi of the adapted blocks. The
    rotations are orthogonal, so L_pre is the protocol's; their orthogonality error
    is recorded next to it."""
    phi_test = features_of(model, data.h_test, device, run_cfg.batch_size // 4 or 1)
    inputs = dataclasses.replace(data.inputs, test_x=phi_test)
    ev = train.evaluate(model.clf, run_cfg, inputs, device)
    rot = {"orthogonality_error_max": model.orthogonality_error(),
           "angles": sum(r.num_angles for r in model.rotations),
           "adapted_blocks": model.tail.blocks,
           "note": "R orthogonal: L_pre unchanged, the certificate uses the protocol's L"}
    return train.write_outputs(run_dir, ev, run_cfg, inputs, {**extra, "conv_rotations": rot},
                               which)


# ---------------------------------------------------------------------------
# learning-rate tuning
# ---------------------------------------------------------------------------

CRITERION = ("max validation accuracy over (run, epoch); ties: weighted F1, then the earlier "
             "epoch, then the smaller lr; collapsed epochs only if all collapsed")
IGNORED_FOR_REUSE = ("micro_batch", "lr_source")   # do not change what a finished run means


def _lr(v: float) -> float:
    return float(f"{v:.6g}")


def same_settings(saved: dict, cfg: ConvConfig) -> bool:
    now = cfg.compat()
    return all(saved.get(k) == v for k, v in now.items() if k not in IGNORED_FOR_REUSE)


def protocol_variant_lr(step: int, mlp_mode: str, variant: str) -> float | None:
    """The protocol's MLP lr of `variant`, None if that variant does not exist."""
    from . import tune

    if not tune.selection_path(step).exists():
        raise FileNotFoundError(f"{tune.selection_path(step)} not found: the protocol tuning "
                                "is needed (run `tune` first, or copy its tuning/step_<n>/)")
    v = env.read_json(tune.selection_path(step))["modes"][mlp_mode]["variants"][variant]
    return None if v is None else float(v["lr"])


def variants_of(step: int, mlp_mode: str) -> list[str]:
    """The protocol's variants of a mode: best, plus stable when it differs."""
    from . import tune

    if tune.stable_equals_best(step, mlp_mode) or protocol_variant_lr(step, mlp_mode, "stable") is None:
        return ["best"]
    return ["best", "stable"]


def tune_complete(run_dir: Path) -> bool:
    return (run_dir / layout.CHECKPOINTS_DIR / layout.checkpoint_file("final")).exists() \
        and (run_dir / TUNE_RESULT).exists()


def _key(c: dict):
    return (-c["accuracy"], -c["f1_weighted"], c["epochs"], c["lr_conv"], c["lr_mlp"])


def _select(cands: list[dict]) -> dict:
    pool = [c for c in cands if not c["collapsed"]] or cands
    return sorted(pool, key=_key)[0]


def run_candidates(run_dir: Path, lr_conv: float, lr_mlp: float) -> tuple[list[dict], dict]:
    """(one candidate per epoch, summary of the run) from a finished tuning run."""
    from . import tune

    hist = env.read_json(run_dir / "history.json")
    cands = [{"lr_conv": lr_conv, "lr_mlp": lr_mlp, "epochs": h["epoch"] + 1,
              "accuracy": h["val_accuracy"], "f1_weighted": h["val_f1_weighted"],
              "collapsed": h["val_collapsed"]} for h in hist]
    best, last = _select(cands), cands[-1]
    stable = (not last["collapsed"]) and last["accuracy"] >= best["accuracy"] - tune.STABLE_DROP
    return cands, {**best, "final_epoch_accuracy": last["accuracy"],
                   "final_epoch_collapsed": last["collapsed"], "stable_run": stable,
                   "run_dir": str(run_dir)}


def choose(runs: list[tuple[list[dict], dict]], variant: str) -> dict | None:
    """The criterion over every epoch of the runs (`best`) or of the stable runs only
    (`stable`, the protocol's STABLE_RULE); None when no run qualifies."""
    pool = [c for cands, s in runs for c in cands if variant == "best" or s["stable_run"]]
    return _select(pool) if pool else None


def tune_mode(step: int, blocks: int, mlp_mode: str, variant: str, data: ConvData, tail, *,
              base: ConvConfig, conv_grid=CONV_LR_GRID, logger=None, restart: bool = False,
              lr_mlp0: float | None = None) -> dict:
    """Stage A (rotation lr) and stage B (MLP lr x 1/3, x 3) for one (mode, variant)."""
    log = logger.info if logger else (lambda *_: None)
    if lr_mlp0 is None:
        lr_mlp0 = protocol_variant_lr(step, mlp_mode, variant)
    if lr_mlp0 is None:
        raise ValueError(f"{mlp_mode} has no protocol variant {variant!r}")

    def run(lr_conv: float, lr_mlp: float):
        cfg = dataclasses.replace(base, step=step, blocks=blocks, mlp_mode=mlp_mode,
                                  seed=TUNE_SEED, lr=lr_mlp, lr_conv=lr_conv,
                                  lr_source=f"conv_adapt tuning, {variant}")
        rd = TREE.conv_tune_run_dir(step, blocks, mlp_mode, variant, lr_conv, lr_mlp, TUNE_SEED)
        if tune_complete(rd) and not restart:
            if not same_settings(env.read_json(rd / TUNE_RESULT)["config"], cfg):
                raise RuntimeError(f"{rd} is complete but was trained with other settings; "
                                   "--restart retrains the tuning")
            log(f"skip {rd.parent.name}: complete")
        else:
            log(f"tune {layout.run_name(mlp_mode, variant)}: rotations lr {lr_conv:g}, "
                f"MLP lr {lr_mlp:g}")
            fit(cfg, data, tail, rd, restart=restart, evaluate_test=False,
                logger=env.setup_logging(rd / "train.log",
                                         name=f"bronet_mlp.conv_adapt.tune.{rd.parent.name}"))
        return run_candidates(rd, lr_conv, lr_mlp)

    stage_a = [run(_lr(lc), lr_mlp0) for lc in conv_grid]
    chosen_a = choose(stage_a, variant)
    out = {"protocol_lr_mlp": lr_mlp0, "stage_a": [s for _, s in stage_a]}
    if chosen_a is None:
        log(f"{mlp_mode} {variant}: no stable run in stage A")
        return {**out, "lr_mlp": None, "lr_conv": None, "stage_b": []}
    stage_b = [run(chosen_a["lr_conv"], _lr(lr_mlp0 * f)) for f in MLP_LR_FACTORS]
    a_run = [r for r in stage_a if r[1]["lr_conv"] == chosen_a["lr_conv"]]
    chosen_b = choose(a_run + stage_b, variant)
    switch = (chosen_b is not None and chosen_b["lr_mlp"] != lr_mlp0
              and chosen_b["accuracy"] >= chosen_a["accuracy"] + MLP_SWITCH_MARGIN)
    final = chosen_b if switch else chosen_a
    log(f"{layout.run_name(mlp_mode, variant)}: rotations lr {final['lr_conv']:g}, MLP lr "
        f"{final['lr_mlp']:g} (val {final['accuracy']:.2f}, epoch {final['epochs']}"
        f"{'; MLP lr changed' if switch else ''})")
    return {**out, "lr_mlp": final["lr_mlp"], "lr_conv": final["lr_conv"],
            "epochs": final["epochs"], "val_accuracy": final["accuracy"],
            "val_f1_weighted": final["f1_weighted"], "stage_b": [s for _, s in stage_b],
            "stage_a_choice": chosen_a, "stage_b_choice": chosen_b, "mlp_lr_changed": switch,
            "lr_conv_at_grid_edge": final["lr_conv"] in (_lr(min(conv_grid)), _lr(max(conv_grid)))}


def selection_path(step: int, blocks: int) -> Path:
    return TREE.conv_tuning_dir(step, blocks) / layout.SELECTION_FILE


def tuning_fixed(cfg: ConvConfig) -> dict:
    return {k: v for k, v in cfg.compat().items()
            if k not in ("mlp_mode", "seed", "lr", "lr_conv", "lr_source", "micro_batch")}


def write_selection(step: int, blocks: int, results: dict, fixed: dict, conv_grid) -> dict:
    """selection.json and its table; modes / variants tuned before with the same fixed
    settings are kept."""
    path = selection_path(step, blocks)
    modes = {}
    if path.exists():
        old = env.read_json(path)
        if old.get("fixed") == fixed:
            modes = old["modes"]
    for mode, per_variant in results.items():
        modes.setdefault(mode, {}).update(per_variant)
    sel = {"step": step, "blocks": blocks, "criterion": CRITERION,
           "conv_lr_grid": [_lr(v) for v in conv_grid], "mlp_lr_factors": list(MLP_LR_FACTORS),
           "mlp_switch_margin": MLP_SWITCH_MARGIN, "seed": TUNE_SEED, "fixed": fixed,
           "modes": modes}
    env.write_json(path, sel)
    lines = [f"# Learning rates for rotations in the last {blocks} BRO block(s), step-{step}", "",
             f"Validation part of train, seed {TUNE_SEED}, {fixed['epochs']} epochs. Criterion: "
             f"{CRITERION}. Stage A: rotation lr on {sel['conv_lr_grid']} with the protocol's MLP "
             f"lr; stage B: MLP lr x {MLP_LR_FACTORS[0]:.3g} and x {MLP_LR_FACTORS[1]:g} at the "
             f"chosen rotation lr, adopted only if at least {MLP_SWITCH_MARGIN:g} points better. "
             "`stable`: the same over runs whose last epoch is not collapsed and within 2 points "
             "of their own best.", "",
             "| mode | stage | rotation lr | MLP lr | best epoch | val acc, % | val acc last epoch, % "
             "| stable run | chosen |", "|---|---|---|---|---|---|---|---|---|"]
    for mode, per_variant in modes.items():
        for variant, r in per_variant.items():
            name = layout.run_name(mode, variant)
            for stage in ("stage_a", "stage_b"):
                for s in r[stage]:
                    mark = "**chosen**" if (r["lr_conv"], r["lr_mlp"]) == (s["lr_conv"], s["lr_mlp"]) \
                        else ""
                    lines.append(f"| {name} | {stage[-1].upper()} | {s['lr_conv']:g} | {s['lr_mlp']:g} | "
                                 f"{s['epochs']} | {s['accuracy']:.2f} | {s['final_epoch_accuracy']:.2f} | "
                                 f"{'yes' if s['stable_run'] else 'no'} | {mark} |")
    edge = [layout.run_name(m, v) for m, pv in modes.items() for v, r in pv.items()
            if r.get("lr_conv_at_grid_edge")]
    if edge:
        lines += ["", f"Warning: rotation lr at the edge of the grid for {', '.join(edge)} -- the "
                  "optimum may lie outside it (--conv-grid)."]
    env.atomic_write_text(path.with_name(f"tuning_step{step}_last{blocks}.md"), "\n".join(lines) + "\n")
    return sel


def tuned_lrs(step: int, blocks: int, mlp_mode: str, variant: str, cfg: ConvConfig) -> dict | None:
    """{"lr_mlp", "lr_conv"} chosen for (mode, variant), None if not tuned / no stable run;
    errors if the tuning was made with other fixed settings."""
    path = selection_path(step, blocks)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run  python -m experiments.bronet_mlp."
                                f"conv_adapt tune --step {step} --blocks {blocks}")
    sel = env.read_json(path)
    if sel["fixed"] != tuning_fixed(cfg):
        diff = {k: (sel["fixed"].get(k), v) for k, v in tuning_fixed(cfg).items()
                if sel["fixed"].get(k) != v}
        raise RuntimeError(f"{path} was made with other settings: {diff}")
    r = sel["modes"].get(mlp_mode, {}).get(variant)
    if r is None or r["lr_mlp"] is None:
        return None
    return {"lr_mlp": float(r["lr_mlp"]), "lr_conv": float(r["lr_conv"])}


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------

def _mean_sd(v) -> str:
    v = np.asarray(v, dtype=float)
    return f"{v.mean():.1f}" if len(v) < 2 else f"{v.mean():.1f} ± {v.std(ddof=1):.1f}"


def summary(step: int, blocks: int, which: str, modes=adapters.MODES, seeds=(0, 1, 2)) -> Path:
    names = ("36/255", "72/255", "108/255")
    lines = [f"# Rotations in the last {blocks} BRO block(s), step-{step} "
             f"({'best validation' if which == 'best' else 'last'} epoch)", "",
             "Protocol: MLP adapted in the given mode, LLN head; *conv*: plus PartitionedButterfly "
             "rotations (588 = 512+64+8+4) inside the last BRO blocks, learning rates from the "
             "conv tuning (*conv, manual lr*: given by hand). Mean ± SD over seeds.", "",
             "| model | run | seeds | clean, % | median r | L/L₀ | "
             + " | ".join(f"cert. {n}, %" for n in names) + " |",
             "|---|---|---|---|---|---|" + "---|" * len(names)]
    for mode in modes:
        for variant in ("best", "stable", "manual"):
            rows = [(f"conv, last {blocks}" + (", manual lr" if variant == "manual" else ""),
                     [TREE.conv_run_dir(step, blocks, mode, s, variant) for s in seeds])]
            if variant != "manual":
                rows.insert(0, ("protocol", [env.run_dir(step, mode, s, variant) for s in seeds]))
            for label, dirs in rows:
                ms = [env.read_json(d / layout.metrics_file(which)) for d in dirs
                      if (d / layout.metrics_file(which)).exists()]
                if not ms:
                    continue
                cert = [m["certificate_official"] for m in ms]
                lines.append(
                    f"| {label} | {layout.run_name(mode, variant)} | {len(ms)} | "
                    f"{_mean_sd([m['accuracy'] for m in ms])} | "
                    f"{np.mean([c['median_radius'] for c in cert]):.3f} | "
                    f"{np.exp(np.mean(np.log([m['lipschitz']['L_over_L0'] for m in ms]))):.4g} | "
                    + " | ".join(_mean_sd([100 * c['certified_accuracy_named'][n] for c in cert])
                                 for n in names) + " |")
    path = TREE.conv_summary_file(step, blocks, which)
    env.atomic_write_text(path, "\n".join(lines) + "\n")
    return path


# ---------------------------------------------------------------------------
# export: the numbers only, for reports made elsewhere
# ---------------------------------------------------------------------------

EXPORT_SUFFIXES = (".json", ".jsonl", ".csv", ".md", ".npz", ".log", ".png", ".txt")
EXPORT_NEVER = (".pt", ".npy", ".tmp")        # weights, the block-input cache


def export_files(step: int, blocks: int) -> list[Path]:
    """Every result file of (step, blocks): runs, tuning, attacks, summaries, the
    cache's index -- not the checkpoints and not the cache itself."""
    root = TREE.root()
    trees = [root / "conv_adapt" / f"step_{step}" / f"last{blocks}",
             TREE.conv_tuning_dir(step, blocks), TREE.results_dir() / "conv_adapt"]
    files = []
    for t in trees:
        if t.exists():
            files += [f for f in t.rglob("*") if f.is_file() and f.suffix in EXPORT_SUFFIXES
                      and f.suffix not in EXPORT_NEVER]
    files = [f for f in files if t_matches(f, step, blocks)]
    for dtype in DTYPES:
        d = TREE.conv_features_dir(step, blocks, dtype)
        files += [f for f in (d / "index.json", d / "extract.log") if f.exists()]
    return sorted(set(files))


def t_matches(f: Path, step: int, blocks: int) -> bool:
    """results/conv_adapt holds every (step, blocks): keep only this one's files."""
    parts = f.parts
    if "results" not in parts:
        return True
    name = f.relative_to(TREE.results_dir() / "conv_adapt").parts[0]
    return f"step{step}_last{blocks}" in name


def export(step: int, blocks: int, logger=None) -> Path:
    import zipfile

    log = logger.info if logger else print
    files = export_files(step, blocks)
    if not files:
        raise FileNotFoundError(f"nothing to export for step {step}, last {blocks} block(s)")
    out = TREE.conv_export_file(step, blocks)
    tmp = out.with_name(out.name + ".tmp")
    root = TREE.root()
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, f.relative_to(root).as_posix())
    tmp.replace(out)
    log(f"{len(files)} files, {out.stat().st_size / 1e6:.1f} MB -> {out} (unzip into "
        f"<DERMA_REPRO_OUT>/bronet_mlp/ where the report is made)")
    return out


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------

def _base_config(a) -> ConvConfig:
    """The settings shared by every run of this command (lr filled in later)."""
    return ConvConfig(step=a.step, blocks=a.blocks, epochs=a.epochs,
                      batch_size=a.train_batch_size, micro_batch=a.micro_batch,
                      stages=a.stages, perm_seed=a.perm_seed, cache_dtype=a.dtype,
                      device=a.device)


def main(argv=None) -> int:
    from repro import console

    console.init()
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("command", choices=("extract", "tune", "train", "summary", "export"))
    p.add_argument("--step", type=int, choices=(1, 2), required=True)
    p.add_argument("--blocks", type=int, default=1, help="adapt the last k BRO blocks (of 8)")
    p.add_argument("--dtype", choices=tuple(DTYPES), default="float16", help="cache precision")
    p.add_argument("--device", default="auto")
    p.add_argument("--bro-weights", default="auto")
    p.add_argument("--cache-bro-weights", action="store_true")
    # extract
    p.add_argument("--batch-size", type=int, default=64, help="extract: images per pass")
    p.add_argument("--chunk-size", type=int, default=1024,
                   help="extract: images per chunk (every BRO block is read once per chunk)")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--force", action="store_true", help="extract: replace a mismatching cache")
    # tune / train
    p.add_argument("--mlp-modes", nargs="+", choices=adapters.MODES, default=["butterfly"])
    p.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS),
                   help="tune / train --from-tuning: the protocol's MLP-lr variants (stable "
                        "only where it differs from best)")
    p.add_argument("--conv-grid", nargs="+", type=float, default=list(CONV_LR_GRID),
                   help="tune: rotation learning rates of stage A")
    p.add_argument("--from-tuning", action="store_true",
                   help="train: learning rates from `tune` (folders <mode>[_stable]); without "
                        "it --lr / --lr-conv are used (folders <mode>_manual)")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--train-batch-size", type=int, default=256)
    p.add_argument("--micro-batch", type=int, default=64,
                   help="tune / train: piece of the batch per forward/backward (memory)")
    p.add_argument("--lr", type=float, default=None,
                   help="train without --from-tuning: MLP lr (default: the protocol's, best)")
    p.add_argument("--lr-conv", type=float, default=None,
                   help="train without --from-tuning: rotation lr (default: --lr)")
    p.add_argument("--stages", type=int, default=2)
    p.add_argument("--perm-seed", type=int, default=0)
    p.add_argument("--restart", action="store_true",
                   help="tune / train: discard finished or partial runs")
    p.add_argument("--stop-after", type=int, default=None)
    # summary
    p.add_argument("--checkpoint", choices=("best", "final"), default="final")
    a = p.parse_args(argv)

    if a.command == "summary":
        log_path = TREE.conv_summary_file(a.step, a.blocks, a.checkpoint).with_suffix(".log")
        logger = env.setup_logging(log_path, name="bronet_mlp.conv_adapt.summary")
        for which in ("best", "final"):
            logger.info(f"-> {summary(a.step, a.blocks, which)}")
        return 0
    if a.command == "export":
        export(a.step, a.blocks)
        return 0

    device = env.resolve_device(a.device)
    strategy = env.resolve_bro_strategy(device, a.bro_weights)
    if a.command == "extract":
        d = TREE.conv_features_dir(a.step, a.blocks, a.dtype)
        logger = env.setup_logging(d / "extract.log", name="bronet_mlp.conv_adapt.extract")
        extract(a.step, a.blocks, a.dtype, device=device, strategy=strategy,
                cache_bro_weights=a.cache_bro_weights, batch_size=a.batch_size,
                chunk_size=a.chunk_size, num_workers=a.num_workers, force=a.force,
                logger=logger)
        return 0

    from . import bronet

    where = (TREE.conv_tuning_dir(a.step, a.blocks) if a.command == "tune"
             else TREE.root() / "conv_adapt" / f"step_{a.step}" / f"last{a.blocks}")
    logger = env.setup_logging(where / f"{a.command}.log", name=f"bronet_mlp.conv_adapt.{a.command}")
    base = _base_config(a)
    data = load_data(a.step, a.blocks, a.dtype, logger)
    # the frozen tail needs only its own blocks on the GPU: never load all 8 there
    model, _ = bronet.load_bronet(device, "stream" if strategy == "gpu" else strategy, logger,
                                  cache_bro_weights=a.cache_bro_weights)
    tail = BroTail(model, model.conv.depth - a.blocks, device)

    if a.command == "tune":
        results = {}
        for mode in a.mlp_modes:
            for variant in [v for v in variants_of(a.step, mode) if v in a.variants]:
                results.setdefault(mode, {})[variant] = tune_mode(
                    a.step, a.blocks, mode, variant, data, tail, base=base,
                    conv_grid=a.conv_grid, logger=logger, restart=a.restart)
        sel = write_selection(a.step, a.blocks, results, tuning_fixed(base), a.conv_grid)
        logger.info(f"-> {selection_path(a.step, a.blocks)}")
        for mode, pv in sel["modes"].items():
            for variant, r in pv.items():
                logger.info(f"{layout.run_name(mode, variant)}: rotations lr {r['lr_conv']}, "
                            f"MLP lr {r['lr_mlp']} (protocol {r['protocol_lr_mlp']})")
        return 0

    for mode in a.mlp_modes:
        if a.from_tuning:
            plans = []
            for variant in [v for v in variants_of(a.step, mode) if v in a.variants]:
                lrs = tuned_lrs(a.step, a.blocks, mode, variant, base)
                if lrs is None:
                    logger.warning(f"{layout.run_name(mode, variant)}: not tuned or no stable "
                                   "run in the tuning -- skipped")
                    continue
                plans.append((variant, lrs["lr_mlp"], lrs["lr_conv"],
                              f"conv_adapt tuning, {variant}"))
        else:
            lr = a.lr if a.lr is not None else protocol_lr(a.step, mode)
            if lr is None:
                p.error(f"no protocol tuning for step {a.step}: pass --lr")
            plans = [("manual", lr, a.lr_conv if a.lr_conv is not None else lr,
                      "given" if a.lr is not None else "protocol lr (best), manual run")]
        for variant, lr, lr_conv, source in plans:
            for seed in a.seeds:
                cfg = dataclasses.replace(base, mlp_mode=mode, seed=seed, lr=lr, lr_conv=lr_conv,
                                          lr_source=source)
                rd = TREE.conv_run_dir(a.step, a.blocks, mode, seed, variant)
                name = f"{layout.run_name(mode, variant)} seed {seed}"
                if is_complete(rd) and not a.restart:
                    saved = env.read_json(rd / "metrics.json").get("conv_config", {})
                    if not same_settings(saved, cfg):
                        raise RuntimeError(f"{rd} is complete but was trained with other "
                                           "settings; --restart to retrain")
                    logger.info(f"skip {name}: complete")
                    continue
                logger.info(f"start {name}: MLP lr {lr:g}, rotations lr {lr_conv:g}")
                fit(cfg, data, tail, rd, restart=a.restart, stop_after=a.stop_after,
                    logger=env.setup_logging(rd / "train.log",
                                             name=f"bronet_mlp.conv_adapt.{mode}.{variant}.{seed}"))
    # every mode, not only this command's: a later `train --mlp-modes lora` must not
    # drop the butterfly rows of the same summary file
    summary(a.step, a.blocks, "final")
    summary(a.step, a.blocks, "best")
    return 0


if __name__ == "__main__":
    sys.exit(main())
