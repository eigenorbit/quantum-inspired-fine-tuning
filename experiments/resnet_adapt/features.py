"""Cache of h(x) = Front(x): the input of ``layer4.2``, 2048 x 7 x 7 per image.

    python -m experiments.resnet_adapt.features extract --step 2 [--dtype float16]
    python -m experiments.resnet_adapt.features verify  --step 2
    python -m experiments.resnet_adapt.features status  --step 2

Layout, under <DERMA_REPRO_OUT>/resnet_adapt/features/step_<n>/<dtype>/:

    features.npy   (rows, 2048, 7, 7), one preallocated .npy, read as a memory map
    index.json     meta + the ordered row list (train records, then test) + per-file
                   size / mtime + `done` (rows written and flushed) + extraction passes
    DONE           written when every row is filled

Sizes: float16 200 KB per image (step-1 6.5 GB, step-2 2.8 GB); float32 twice
that. A memory map is used because step-1 does not fit into 8 GB of GPU memory
and float32 step-1 not into 16 GB of RAM; batches are gathered from it.

Rows are written in order and `done` is advanced only after a flush, so an
interrupted extraction resumes at the first unfinished row. Validation is
strict for everything that changes the numbers (weights, front definition,
transform, dtype, the row list, file size and mtime): a mismatch is an error,
never a silent recompute.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from . import paths

import numpy as np
import torch
from torch.utils.data import DataLoader

from experiments.common import data as dataset
from experiments.common.device import resolve_device, device_info
from experiments.common.io import read_json, write_json, atomic_write_text, sha256_json, setup_logging
from experiments.common.locations import data_root

from . import backbone

DTYPES = {"float16": np.float16, "float32": np.float32}
HARD_META = ("weights", "weights_sha256", "front_definition", "transform_version",
             "feature_shape", "dtype", "rows_sha256")
# verify(): max |cached - fresh| relative to max |fresh|; float16 keeps 11 bits.
TOLERANCE = {"float16": 2e-3, "float32": 1e-4}


def cache_dir(step: int, dtype: str) -> Path:
    return paths.features_dir(step) / dtype


def _index_path(step, dtype) -> Path:
    return cache_dir(step, dtype) / "index.json"


def _data_path(step, dtype) -> Path:
    return cache_dir(step, dtype) / "features.npy"


def _done_path(step, dtype) -> Path:
    return cache_dir(step, dtype) / "DONE"


def expected_meta(dtype: str, rels: list[str], weights_sha256: str) -> dict:
    if dtype not in DTYPES:
        raise ValueError(f"dtype must be one of {tuple(DTYPES)}")
    return {"weights": backbone.WEIGHTS, "weights_sha256": weights_sha256,
            "front_definition": backbone.FRONT_DEFINITION,
            "transform_version": dataset.TRANSFORM_VERSION,
            "feature_shape": list(backbone.FEATURE_SHAPE), "dtype": dtype,
            "rows_sha256": sha256_json(rels)}


def read_index(step: int, dtype: str) -> dict | None:
    p = _index_path(step, dtype)
    return read_json(p) if p.exists() else None


def check_meta(index: dict, meta: dict, keys=HARD_META) -> None:
    bad = {k: (index["meta"].get(k), meta[k]) for k in keys if index["meta"].get(k) != meta[k]}
    if bad:
        lines = "\n".join(f"  {k}: cache {str(c)[:40]!r} vs now {str(n)[:40]!r}"
                          for k, (c, n) in bad.items())
        raise RuntimeError(f"feature cache {cache_dir(index['meta']['step'], meta['dtype'])} "
                           f"was built under different settings:\n{lines}\n"
                           "Move the folder away to rebuild it deliberately.")


def _stat(path: Path) -> list[int]:
    st = Path(path).stat()
    return [st.st_size, st.st_mtime_ns]


def check_files(index: dict, ignore_mtime: bool = False) -> None:
    bad = []
    for rel, (size, mtime) in zip(index["rows"], index["stat"]):
        s, m = _stat(data_root() / rel)
        if s != size or (not ignore_mtime and m != mtime):
            bad.append(rel)
    if bad:
        raise RuntimeError(f"{len(bad)} image(s) changed on disk since their features were "
                           f"cached, e.g. {bad[:3]}. (--ignore-mtime accepts a pure mtime "
                           "change, e.g. after copying the data to another machine.)")


def extract(step: int, image_paths: list[Path], *, device, net=None, dtype: str = "float16",
            batch_size: int = 64, num_workers: int = 4, flush_rows: int = 1024,
            ignore_mtime: bool = False, logger=None, limit: int | None = None) -> dict:
    """Fill the cache for `image_paths` (in this order). `limit`: stop after that
    many new rows (tests of the resumption)."""
    log = logger.info if logger else print
    net = backbone.load_resnet152() if net is None else net
    rels = [dataset.relative_path(p) for p in image_paths]
    meta = expected_meta(dtype, rels, backbone.state_sha256(net))
    index = read_index(step, dtype)
    shape = (len(rels),) + tuple(backbone.FEATURE_SHAPE)
    if index is not None:
        check_meta(index, meta)
        check_files(index, ignore_mtime)
    else:
        cache_dir(step, dtype).mkdir(parents=True, exist_ok=True)
        np.lib.format.open_memmap(_data_path(step, dtype), mode="w+", dtype=DTYPES[dtype],
                                  shape=shape).flush()
        index = {"meta": {**meta, "step": step, "n_rows": len(rels)}, "rows": rels,
                 "stat": [_stat(p) for p in image_paths], "done": 0, "passes": []}
        write_json(_index_path(step, dtype), index)
    start = index["done"]
    stop = len(rels) if limit is None else min(len(rels), start + limit)
    done_marker = _done_path(step, dtype)
    if start >= len(rels):
        if not done_marker.exists():
            atomic_write_text(done_marker, f"{len(rels)}\n")
        log(f"features step-{step} ({dtype}): all {len(rels):,} cached")
        return {"computed": 0, "seconds": 0.0}

    front = backbone.Front(net).to(device)
    loader = DataLoader(dataset.ImageDataset(list(image_paths[start:stop])),
                        batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        pin_memory=device.type == "cuda")
    mm = np.load(_data_path(step, dtype), mmap_mode="r+")
    log(f"features step-{step} ({dtype}): rows {start:,}..{stop:,} of {len(rels):,}, "
        f"batch {batch_size}, workers {num_workers}, device {device}")
    row, flushed, t0 = start, start, time.perf_counter()

    def flush():
        nonlocal flushed
        mm.flush()
        index["done"] = row
        write_json(_index_path(step, dtype), index)
        flushed = row

    try:
        with torch.no_grad():
            for x, _ in loader:
                h = front(x.to(device, non_blocking=True)).cpu().numpy()
                with np.errstate(over="ignore"):
                    h = h.astype(DTYPES[dtype])
                if not np.isfinite(h).all():
                    raise FloatingPointError(
                        f"non-finite features at rows {row}..{row + len(h)} after the cast to "
                        f"{dtype} (max |h| {float(np.abs(h[np.isfinite(h)]).max(initial=0)):.3g}"
                        "); use --dtype float32")
                mm[row:row + len(h)] = h
                row += len(h)
                if row - flushed >= flush_rows:
                    flush()
                    rate = (row - start) / max(time.perf_counter() - t0, 1e-9)
                    log(f"  {row:,}/{len(rels):,}  {rate:.1f} img/s  "
                        f"ETA {(len(rels) - row) / max(rate, 1e-9) / 60:.1f} min")
    except torch.cuda.OutOfMemoryError as exc:
        flush()
        raise RuntimeError(f"CUDA out of memory at batch {batch_size}; lower --batch-size "
                           "(progress is saved)") from exc
    flush()
    del mm
    seconds = time.perf_counter() - t0
    index["passes"].append({"rows": [start, row], "seconds": seconds, "batch_size": batch_size,
                            "num_workers": num_workers, **device_info(device)})
    write_json(_index_path(step, dtype), index)
    if row >= len(rels):
        atomic_write_text(done_marker, f"{len(rels)}\n")
    log(f"features step-{step} ({dtype}): {row - start:,} computed in {seconds:.0f}s "
        f"({(row - start) / max(seconds, 1e-9):.1f} img/s)")
    return {"computed": row - start, "seconds": seconds}


@dataclass
class FeatureStore:
    """The cached array (a read-only memory map) and its row lookup."""

    array: np.ndarray
    index: dict
    feature_id: str

    def rows_of(self, rels: list[str]) -> np.ndarray:
        where = {r: i for i, r in enumerate(self.index["rows"])}
        missing = [r for r in rels if r not in where]
        if missing:
            raise KeyError(f"{len(missing)} image(s) not in the feature cache, e.g. "
                           f"{missing[:3]}")
        return np.array([where[r] for r in rels], dtype=np.int64)


def open_store(step: int, dtype: str, weights_sha256: str, ignore_mtime: bool = False,
               check_stat: bool = True) -> FeatureStore:
    index = read_index(step, dtype)
    if index is None or not _done_path(step, dtype).exists():
        raise FileNotFoundError(
            f"feature cache {cache_dir(step, dtype)} is missing or incomplete; run\n"
            f"  python -m experiments.resnet_adapt.features extract --step {step} "
            f"--dtype {dtype}")
    now = expected_meta(dtype, index["rows"], weights_sha256)
    check_meta(index, now)
    if check_stat:
        check_files(index, ignore_mtime)
    arr = np.load(_data_path(step, dtype), mmap_mode="r")
    if arr.shape != (len(index["rows"]),) + tuple(backbone.FEATURE_SHAPE) \
            or arr.dtype != DTYPES[dtype]:
        raise RuntimeError(f"{_data_path(step, dtype)}: shape {arr.shape} / {arr.dtype} "
                           "does not match its index")
    fid = sha256_json({"meta": index["meta"], "done": index["done"],
                           "bytes": _data_path(step, dtype).stat().st_size})
    return FeatureStore(array=arr, index=index, feature_id=fid)


def gather(array: np.ndarray, rows: np.ndarray, device) -> torch.Tensor:
    """float32 batch of the given rows (any order), on `device`."""
    x = torch.from_numpy(np.ascontiguousarray(array[rows]))
    return x.to(device, non_blocking=True).float()


@torch.no_grad()
def verify(step: int, dtype: str, *, device, net=None, n: int = 16, seed: int = 0,
           logger=None) -> dict:
    """Cached h vs a fresh Front(x), and torchvision's own end-to-end logits vs
    fc(pool(layer4.2(h_cached))), on `n` random cached images."""
    log = logger.info if logger else print
    net = backbone.load_resnet152() if net is None else net
    store = open_store(step, dtype, backbone.state_sha256(net))
    g = torch.Generator().manual_seed(seed)
    pick = sorted(torch.randperm(len(store.index["rows"]), generator=g)[:n].tolist())
    files = [data_root() / store.index["rows"][i] for i in pick]
    x = dataset.load_images(files).to(device)
    net = net.to(device).eval()
    front = backbone.Front(net).to(device)
    fresh = front(x).cpu()
    cached = torch.from_numpy(np.asarray(store.array[pick])).float()
    ref = net(front.normalize(x)).cpu()
    ours = net.fc(net.layer4[2](cached.to(device)).mean(dim=(2, 3))).cpu()
    tol = TOLERANCE[dtype]
    res = {"n": len(pick), "dtype": dtype,
           "h_max_abs_diff": float((fresh - cached).abs().max()),
           "h_max_abs": float(fresh.abs().max()),
           "logits_max_abs_diff": float((ours - ref).abs().max()),
           "logits_max_abs": float(ref.abs().max()),
           "top1_agree": float((ours.argmax(1) == ref.argmax(1)).float().mean()),
           "tolerance_rel": tol}
    res["ok"] = (res["h_max_abs_diff"] <= tol * max(1.0, res["h_max_abs"])
                 and res["logits_max_abs_diff"] <= tol * max(1.0, res["logits_max_abs"])
                 and res["top1_agree"] == 1.0)
    log(f"verify step-{step} ({dtype}): {res}")
    return res


def record_paths(step: int, logger=None) -> list[Path]:
    """Rows of the cache: the deduplicated train records, then the test originals."""
    train, test, _ = dataset.records(step, logger)
    return [r.path for r in train] + [r.path for r in test]


def main(argv=None) -> int:
    from repro import console

    console.init()
    p = argparse.ArgumentParser(description="layer4.2 input feature cache")
    p.add_argument("command", choices=("extract", "verify", "status"))
    p.add_argument("--step", type=int, choices=(1, 2), required=True)
    p.add_argument("--dtype", choices=tuple(DTYPES), default="float16")
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=min(4, os.cpu_count() or 1))
    p.add_argument("--ignore-mtime", action="store_true")
    p.add_argument("--n", type=int, default=16, help="verify: number of images")
    a = p.parse_args(argv)

    logger = setup_logging(paths.features_dir(a.step) / f"{a.dtype}_{a.command}.log",
                               name="resnet_adapt.features")
    device = resolve_device(a.device)
    if a.command == "status":
        index = read_index(a.step, a.dtype)
        n = index["meta"]["n_rows"] if index else 0
        logger.info(f"step-{a.step} ({a.dtype}): {index['done'] if index else 0:,}/{n:,} rows, "
                    f"DONE={'yes' if _done_path(a.step, a.dtype).exists() else 'no'}")
        return 0
    if a.command == "extract":
        image_paths = record_paths(a.step, logger)
        extract(a.step, image_paths, device=device, dtype=a.dtype, batch_size=a.batch_size,
                num_workers=a.num_workers, ignore_mtime=a.ignore_mtime, logger=logger)
        size = _data_path(a.step, a.dtype).stat().st_size
        logger.info(f"cache size {size / 2**30:.2f} GiB in {cache_dir(a.step, a.dtype)}")
        return 0
    res = verify(a.step, a.dtype, device=device, n=a.n, logger=logger)
    write_json(cache_dir(a.step, a.dtype) / "verify.json", res)
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
