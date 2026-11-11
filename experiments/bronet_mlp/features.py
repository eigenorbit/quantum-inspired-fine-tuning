"""Cache of phi(x): BRONet up to and including the neck, one 2048-d vector per image.

    python -m experiments.bronet_mlp.features extract --step 2
    python -m experiments.bronet_mlp.features verify  --step 2
    python -m experiments.bronet_mlp.features status  --step 2

Layout, under <DERMA_REPRO_OUT>/bronet_mlp/features/step_<n>/:

    shard_XXXX.pt   {"paths": [...], "features": float32 (rows, 2048)}
    index.json      {"meta": ..., "files": {rel path: {shard, row, size, mtime_ns}}}
    DONE            written when an extraction pass has covered every request

Entries are keyed by the path relative to DERMA_DATA_ROOT, so a different
record selection only computes what is missing. Extraction writes one shard at
a time (atomically) and updates the index after each, so an interrupted run
resumes where it stopped.

Validation is strict for everything that changes the numbers (checkpoint
sha256, BRONet commit, transform version, file size and mtime): a mismatch is
an error, never a silent recompute. torch / CUDA versions and the device are
recorded but not enforced -- float32 results agree across them to ~1e-5, and
enforcing them would forbid reading a cache made on another machine.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from . import env

import torch
from torch.utils.data import DataLoader

from . import data as bdata

PHI_DEFINITION = ("neck(conv(conv_1(stem(x - 0.5)))); eval weights computed on CPU; "
                  "float32 with TF32 disabled")
FEATURE_DIM = 2048
HARD_META = ("checkpoint_sha256", "bronet_commit", "transform_version", "phi_definition",
             "feature_dim")


def cache_dir(step: int) -> Path:
    return env.features_dir(step)


def _index_path(step: int) -> Path:
    return cache_dir(step) / "index.json"


def _done_path(step: int) -> Path:
    return cache_dir(step) / "DONE"


def expected_meta() -> dict:
    from .bronet import bronet_commit

    return {"checkpoint_sha256": env.checkpoint_sha256(), "bronet_commit": bronet_commit(),
            "transform_version": bdata.TRANSFORM_VERSION, "phi_definition": PHI_DEFINITION,
            "feature_dim": FEATURE_DIM}


def read_index(step: int) -> dict | None:
    path = _index_path(step)
    return env.read_json(path) if path.exists() else None


def check_meta(index: dict, meta: dict) -> None:
    bad = {k: (index["meta"].get(k), meta[k]) for k in HARD_META
           if index["meta"].get(k) != meta[k]}
    if bad:
        lines = "\n".join(f"  {k}: cache {c!r} vs now {n!r}" for k, (c, n) in bad.items())
        raise RuntimeError(f"feature cache {cache_dir(index['meta']['step'])} was built "
                           f"under different settings:\n{lines}\n"
                           "Move the folder away to rebuild it deliberately.")


def _stat(path: Path) -> dict:
    st = path.stat()
    return {"size": st.st_size, "mtime_ns": st.st_mtime_ns}


def check_files(index: dict, paths: list[Path], ignore_mtime: bool = False) -> None:
    files = index["files"]
    bad = []
    for path in paths:
        rel = bdata.relative_path(path)
        entry = files.get(rel)
        if entry is None:
            continue
        st = _stat(path)
        if st["size"] != entry["size"] or (not ignore_mtime and st["mtime_ns"] != entry["mtime_ns"]):
            bad.append(rel)
    if bad:
        raise RuntimeError(f"{len(bad)} image(s) changed on disk since their features were "
                           f"cached, e.g. {bad[:3]}. (--ignore-mtime accepts a pure mtime "
                           "change, e.g. after copying the data to another machine.)")


def missing_paths(step: int, paths: list[Path]) -> list[Path]:
    index = read_index(step)
    have = set(index["files"]) if index else set()
    return [p for p in paths if bdata.relative_path(p) not in have]


def extract(step: int, paths: list[Path], *, device, strategy: str, batch_size: int = 64,
            num_workers: int = 4, shard_size: int = 4096, ignore_mtime: bool = False,
            logger=None, model=None, cache_bro_weights: bool = False,
            chunk_size: int = 1024) -> dict:
    """Compute and cache phi for every path not yet cached. Returns timing info."""
    log = logger.info if logger else print
    meta = expected_meta()
    index = read_index(step)
    if index is not None:
        check_meta(index, meta)
        check_files(index, paths, ignore_mtime)
    else:
        index = {"meta": {**meta, "step": step}, "files": {}, "shards": [], "passes": []}
    todo = [p for p in paths if bdata.relative_path(p) not in index["files"]]
    done = _done_path(step)
    if not todo:
        if not done.exists():
            env.atomic_write_text(done, f"{len(index['files'])}\n")
        log(f"features step-{step}: all {len(paths):,} cached")
        return {"computed": 0, "seconds": 0.0}
    if done.exists():
        done.unlink()

    t0 = time.perf_counter()
    if model is None:
        from .bronet import load_bronet

        model, _ = load_bronet(device, strategy, logger, cache_bro_weights=cache_bro_weights)
    t_load = time.perf_counter() - t0
    loader = DataLoader(bdata.ImageDataset(todo), batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=device.type == "cuda")
    log(f"features step-{step}: {len(todo):,} to compute ({len(paths) - len(todo):,} cached), "
        f"batch {batch_size}, chunk {chunk_size}, workers {num_workers}, BRO weights {strategy}")

    buf_feats, buf_paths, computed = [], [], 0
    t1 = time.perf_counter()

    def flush():
        nonlocal buf_feats, buf_paths
        if not buf_paths:
            return
        k = len(index["shards"])
        name = f"shard_{k:04d}.pt"
        feats = torch.cat(buf_feats)
        env.atomic_torch_save({"paths": buf_paths, "features": feats}, cache_dir(step) / name)
        for row, rel in enumerate(buf_paths):
            src = env.data_root() / rel
            index["files"][rel] = {"shard": name, "row": row, **_stat(src)}
        index["shards"].append(name)
        env.write_json(_index_path(step), index)
        buf_feats, buf_paths = [], []

    from .bronet import phi_layer_major

    def process(xs, ids):
        nonlocal computed
        # Layer-major: every BRO block is copied to the device once per chunk.
        f = phi_layer_major(model, torch.cat(xs), device, batch_size)
        buf_feats.append(f)
        buf_paths.extend(bdata.relative_path(todo[i]) for i in torch.cat(ids).tolist())
        computed += len(f)
        if sum(len(b) for b in buf_feats) >= shard_size:
            flush()
        rate = computed / max(time.perf_counter() - t1, 1e-9)
        eta = (len(todo) - computed) / max(rate, 1e-9)
        log(f"  {computed:,}/{len(todo):,}  {rate:.1f} img/s  ETA {eta / 60:.1f} min")

    try:
        xs, ids = [], []
        for x, idx in loader:
            xs.append(x)
            ids.append(idx)
            if sum(len(i) for i in ids) >= chunk_size:
                process(xs, ids)
                xs, ids = [], []
        if ids:
            process(xs, ids)
    except torch.cuda.OutOfMemoryError as exc:
        flush()
        raise RuntimeError(
            f"CUDA out of memory (batch {batch_size}, chunk {chunk_size}). Lower --chunk-size "
            "(activations of a whole chunk stay on the GPU) or --batch-size, or use "
            "--bro-weights stream (or cpu). Progress so far is saved.") from exc
    flush()
    seconds = time.perf_counter() - t1
    index["passes"].append({"computed": computed, "seconds_extract": seconds,
                            "seconds_load_model": t_load, "batch_size": batch_size,
                            "chunk_size": chunk_size,
                            "num_workers": num_workers, "strategy": strategy,
                            **env.device_info(device)})
    env.write_json(_index_path(step), index)
    env.atomic_write_text(done, f"{len(index['files'])}\n")
    log(f"features step-{step}: {computed:,} computed in {seconds:.0f}s "
        f"({computed / max(seconds, 1e-9):.1f} img/s), model load {t_load:.0f}s")
    return {"computed": computed, "seconds": seconds, "seconds_load_model": t_load}


def bronet_phi(model, x):
    from .bronet import phi

    return phi(model, x)


def load(step: int, paths: list[Path], ignore_mtime: bool = False) -> torch.Tensor:
    """Cached features for `paths`, in that order. Errors if any is missing."""
    index = read_index(step)
    if index is None or not _done_path(step).exists():
        raise FileNotFoundError(
            f"feature cache {cache_dir(step)} is missing or incomplete; run\n"
            f"  python -m experiments.bronet_mlp.features extract --step {step}")
    check_meta(index, expected_meta())
    check_files(index, paths, ignore_mtime)
    rels = [bdata.relative_path(p) for p in paths]
    missing = [r for r in rels if r not in index["files"]]
    if missing:
        raise KeyError(f"{len(missing)} image(s) not in the feature cache, e.g. {missing[:3]}; "
                       "run features extract")
    out = torch.empty(len(rels), index["meta"]["feature_dim"], dtype=torch.float32)
    by_shard: dict[str, list[tuple[int, int]]] = {}
    for i, rel in enumerate(rels):
        e = index["files"][rel]
        by_shard.setdefault(e["shard"], []).append((i, e["row"]))
    for shard, pairs in by_shard.items():
        blob = torch.load(cache_dir(step) / shard, map_location="cpu", weights_only=True)
        dst = torch.tensor([p[0] for p in pairs])
        src = torch.tensor([p[1] for p in pairs])
        out[dst] = blob["features"][src]
    return out


def features_sha256(paths_rel: list[str], feats: torch.Tensor) -> str:
    """Identity of the exact features a run trains and tests on."""
    import hashlib

    digest = hashlib.sha256()
    digest.update("\n".join(paths_rel).encode())
    digest.update(feats.contiguous().numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def verify(step: int, *, device, strategy: str, n: int = 32, seed: int = 0, logger=None,
           model=None, paths: list[Path] | None = None, cache_bro_weights: bool = False) -> dict:
    """Cached phi vs a fresh computation, and head(linear(phi_cached)) vs BRONet's own
    end-to-end forward `model(x)` (ImageNet head), on `n` random cached images."""
    log = logger.info if logger else print
    index = read_index(step)
    if index is None:
        raise FileNotFoundError(f"no feature cache for step {step}")
    rels = sorted(index["files"]) if paths is None else [bdata.relative_path(p) for p in paths]
    g = torch.Generator().manual_seed(seed)
    pick = [rels[i] for i in torch.randperm(len(rels), generator=g)[:n].tolist()]
    files = [env.data_root() / r for r in pick]
    cached = load(step, files)
    if model is None:
        from .bronet import load_bronet

        model, _ = load_bronet(device, strategy, logger, cache_bro_weights=cache_bro_weights)
    x = bdata.load_images(files)
    fresh = torch.cat([bronet_phi(model, x[i:i + 8].to(device)).cpu()
                       for i in range(0, len(x), 8)])
    # BRONet's own forward, on the device holding its BRO matrices.
    if strategy == "stream":
        model.cpu()
        ref = model(x).cpu()
        model.to(device)
    else:
        ref = model(x.to(device)).cpu()
    ours = model.head(model.linear(cached.to(device))).cpu()
    res = {"n": len(files), "phi_max_abs_diff": float((fresh - cached).abs().max()),
           "logits_max_abs_diff": float((ours - ref).abs().max()),
           "logits_max_abs": float(ref.abs().max()),
           "top1_agree": float((ours.argmax(1) == ref.argmax(1)).float().mean())}
    # float32 through ~20 layers: tolerance 1e-4 relative to the largest magnitude
    # (|phi| reaches ~200, |logits| ~25); measured ~4e-7 relative with TF32 off.
    res["phi_max_abs"] = float(cached.abs().max())
    res["tolerance_rel"] = 1e-4
    res["ok"] = (res["phi_max_abs_diff"] <= 1e-4 * max(1.0, res["phi_max_abs"])
                 and res["logits_max_abs_diff"] <= 1e-4 * max(1.0, res["logits_max_abs"])
                 and res["top1_agree"] == 1.0)
    log(f"verify step-{step}: {res}")
    return res


def main(argv=None) -> int:
    from repro import console

    console.init()
    p = argparse.ArgumentParser(description="phi(x) feature cache")
    p.add_argument("command", choices=("extract", "verify", "status"))
    p.add_argument("--step", type=int, choices=(1, 2), required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--bro-weights", default="auto", help="auto|gpu|stream|cpu")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--chunk-size", type=int, default=1024,
                   help="images pushed through one BRO block before the next is loaded")
    p.add_argument("--num-workers", type=int, default=min(4, os.cpu_count() or 1))
    p.add_argument("--shard-size", type=int, default=4096)
    p.add_argument("--ignore-mtime", action="store_true")
    p.add_argument("--n", type=int, default=32, help="verify: number of images")
    p.add_argument("--cache-bro-weights", action="store_true",
                   help="memory-map BRO eval weights from backbone/eval_weights/ (low RAM)")
    a = p.parse_args(argv)

    logger = env.setup_logging(cache_dir(a.step).parent / f"step_{a.step}_{a.command}.log")
    train, test, _ = bdata.records(a.step, logger)
    paths = [r.path for r in train] + [r.path for r in test]
    if a.command == "status":
        miss = missing_paths(a.step, paths)
        logger.info(f"step-{a.step}: {len(paths) - len(miss):,}/{len(paths):,} cached, "
                    f"DONE={'yes' if _done_path(a.step).exists() else 'no'}")
        return 0
    device = env.resolve_device(a.device)
    strategy = env.resolve_bro_strategy(device, a.bro_weights)
    logger.info(f"device {env.device_info(device)}  BRO weights: {strategy}")
    if a.command == "extract":
        extract(a.step, paths, device=device, strategy=strategy, batch_size=a.batch_size,
                num_workers=a.num_workers, shard_size=a.shard_size,
                ignore_mtime=a.ignore_mtime, logger=logger,
                cache_bro_weights=a.cache_bro_weights, chunk_size=a.chunk_size)
        total = sum(f.stat().st_size for f in cache_dir(a.step).glob("shard_*.pt"))
        logger.info(f"cache size {total / 2**20:.1f} MiB in {cache_dir(a.step)}")
    else:
        res = verify(a.step, device=device, strategy=strategy, n=a.n, logger=logger,
                     paths=paths, cache_bro_weights=a.cache_bro_weights)
        env.write_json(cache_dir(a.step) / "verify.json", res)
        return 0 if res["ok"] else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
