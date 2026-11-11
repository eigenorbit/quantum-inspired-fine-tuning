"""Does the certificate cover lossy image handling, and can training make it cover it?

    python -m experiments.bronet_mlp.lossy measure --step 2 --cache-bro-weights
    python -m experiments.bronet_mlp.lossy measure --step 2 --ops jpeg90 jpeg80 --at input
    python -m experiments.bronet_mlp.lossy sweep   --step 2 --eps 72 108 144 195
    python -m experiments.bronet_mlp.lossy report  --step 2 --eps 36 72 108 144 195
    python -m experiments.bronet_mlp.lossy measure --step 2 --ops jpeg100 jpeg95 jpeg90 jpeg80 --renorm authors

The question of the robustness notebook (`experiments/robustness`, §8.5), for BRONet. Things
nobody calls perturbations happen to a clinical image -- a PACS re-encodes it to
JPEG, a viewer resizes it, a format stores fewer bits, a sensor adds read noise. If
the change T makes is smaller than the certified radius, ||T(x) - x||_2 < r(x), the
certificate proves the diagnosis unchanged without any test.

Operations (`--ops`, default all; the set of `pipeline_ops` in the notebook):
    jpeg100 jpeg95 jpeg90 jpeg80     JPEG re-encode at that quality (PIL defaults)
    bits7 bits6                      requantise to 7 / 6 bits per channel
    resize86 resize71 resize50       bicubic down to 86 / 71 / 50 % of the side and back
                                     (the notebook's 28 -> 24 / 20 / 14 -> 28)
    noise0.002 noise0.005            Gaussian sensor noise of that sigma (pixels in [0, 1])
Where (`--at`):
    native   (default) on the delivered 600x600 file, which then goes through the network
             preprocessing (bilinear resize to 224x224, [0, 1]) -- what happens to a file in
             a hospital; the result is stored as 8-bit like any image file
    input    on the 224x224 network input itself, in floating point, as the notebook does on
             its 28x28 images
d(x) = ||x_T - x||_2 is always measured in the 224x224 input, the space of r(x).
r(x) is the certificate of every BRONet report, min_j (z_p - z_j) / (L ||w_p - w_j||);
the notebook uses (z_y - max_j z_j) / (sqrt(2) L) for its own head.

measure  phi(x_T) of every test image is computed once with the exact float32 BRONet and
         cached (model independent). Then, for every finished run of the protocol:
         covered = correct and d(x) < r(x) (in % of the correctly classified images, as
         the notebook's `certified_unchanged`), the factor d / r by which r would have to grow,
         and whether the prediction really changes (in % of all images, as the
         notebook's `changed_prediction`). The summary answers "how many times larger must r be to
         cover JPEG q = 90" and lists the eps_train values that would ask for it.
sweep    Retrains with a larger eps_train. The certificate-aware loss raises the logits
         of the competing classes by L * eps * ||w_y - w_j||, so eps is the radius the
         training asks for. Everything else is the final protocol of
         `run_grid --from-tuning`: the learning rate and epoch ceiling tuned at
         eps = 36/255 (not re-tuned), fit / validation split, early stopping. eps = 36
         is the protocol itself: its runs are the existing ones under runs/step_<n>/,
         nothing is retrained. Too large an eps degenerates the model, so sweep it.
         `--eps-warmup F` ramps the loss eps linearly from
         `--eps-start` to the target over the first F of the epochs; such runs (eps = 36
         included) go to eps_sweep/step_<n>/eps<k>_wu<F>/ and the best-validation
         epoch is chosen after the ramp only.
report   Per mode and eps: clean accuracy, median radius, certified accuracy, and the
         coverage of every cached operation (no BRONet needed).

Renormalisation after the operation (`--renorm`, native only): the
certified radius r(x) is that of the delivered image x, as always. The changed file
T(x) is then colour-normalised again, N(T(x)), and the total change
d = ||N(T(x)) − x||₂ is compared with r(x). N is one of
    authors  the dataset's own normalisation, with its fitted
             constants: the channel means of the dataset's reference photograph (setting
             DERMA_COLOR_REFERENCE; cropped 1080×1080), damping 0.8 per channel plus an
             undamped global-mean term, integer arithmetic on the 8-bit image
    ref      every channel mean shifted exactly to the reference's (idempotent)
    oracle   every channel mean shifted back to that of x itself -- the most any
             mean shift can undo; needs x, so an upper bound, not a usable method
With a renormalisation the operation `none` (N alone, no change) is measured too, and
the change without N, ||T(x) − x||₂, is kept as `dist_attack`. Outputs go to the
place `native_renorm-<mode>` (cache and results folders).

Only the training batch 256 protocol. Outputs, below <DERMA_REPRO_OUT>/bronet_mlp/:
lossy_features/step_<n>/<at>/<op>.pt; eps_sweep/step_<n>/eps<k>[_wu<F>]/<run>/seed<s>/;
results/lossy/step_<n>_<checkpoint>_<at>/: coverage.md/.csv and coverage_images.csv
(measure), eps_sweep[_wu<F>].md/.csv (report).
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import io
import math
import sys
from dataclasses import dataclass
from pathlib import Path

from . import env

import numpy as np
import torch
import torch.nn.functional as F

from experiments.common import layout

from . import adapters, train
from . import data as bdata

DEFAULT_EPS_255 = 36                   # eps_train of the protocol, in units of 1/255
COVER_SHARES = (0.5, 0.75, 0.9)        # "r must cover this share of the correct images"
EPS_NAMES = ("36/255", "72/255", "108/255")
PLACES = ("native", "input")
NOISE_SEED = 20251006
TREE = layout.BRONET
# renormalisation after the operation; see the module docstring
RENORMS = ("none", "authors", "ref", "oracle")
# the reference photograph: setting DERMA_COLOR_REFERENCE (`locations.color_reference`)
REFERENCE_CROP = 420                                          # px off the left and right
COLOR_PENALTY = 0.8


# ---------------------------------------------------------------------------
# the operations
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Op:
    key: str
    kind: str          # jpeg | bits | resize | noise
    value: float       # quality | bits | fraction of the side | sigma

    def label(self, side: int) -> str:
        if self.kind == "none":
            return "no change"
        if self.kind == "jpeg":
            return f"jpeg q={self.value:g}"
        if self.kind == "bits":
            return f"{self.value:g}-bit depth"
        if self.kind == "resize":
            return f"resize {side}->{resize_side(side, self.value)}->{side}"
        return f"sensor noise {self.value:g}"


OPS = {o.key: o for o in (
    Op("jpeg100", "jpeg", 100), Op("jpeg95", "jpeg", 95), Op("jpeg90", "jpeg", 90),
    Op("jpeg80", "jpeg", 80), Op("bits7", "bits", 7), Op("bits6", "bits", 6),
    Op("resize86", "resize", 24 / 28), Op("resize71", "resize", 20 / 28),
    Op("resize50", "resize", 14 / 28), Op("noise0.002", "noise", 0.002),
    Op("noise0.005", "noise", 0.005))}
# no operation: with a renormalisation, the change N alone makes
IDENTITY = Op("none", "none", 0)


def get_op(key: str) -> Op:
    return IDENTITY if key == IDENTITY.key else OPS[key]


def with_identity(ops: list[str], renorm: str) -> list[str]:
    """The operations measured: `none` first when a renormalisation is applied."""
    return ([IDENTITY.key] if renorm != "none" else []) + [k for k in ops if k != IDENTITY.key]


def place(at: str, renorm: str = "none") -> str:
    """Folder tag of where the operation is applied and what follows it."""
    return at if renorm == "none" else f"{at}_renorm-{renorm}"


def resize_side(side: int, fraction: float) -> int:
    return int(round(side * fraction))


def _jpeg_pil(image, quality: int):
    from PIL import Image

    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    with Image.open(buf) as out:
        return out.convert("RGB")


def apply_native(image, op: Op, gen: torch.Generator):
    """`op` on a decoded RGB file (PIL, uint8); the result is an 8-bit image again."""
    from PIL import Image

    if op.kind == "none":
        return image
    if op.kind == "jpeg":
        return _jpeg_pil(image, op.value)
    if op.kind == "resize":
        w, h = image.size
        small = image.resize((resize_side(w, op.value), resize_side(h, op.value)), Image.BICUBIC)
        return small.resize((w, h), Image.BICUBIC)
    a = np.asarray(image, dtype=np.float64) / 255.0
    if op.kind == "bits":
        levels = 2 ** int(op.value) - 1
        a = np.round(a * levels) / levels
    else:                                                    # noise
        noise = torch.randn(a.shape, generator=gen, dtype=torch.float64).numpy()
        a = np.clip(a + op.value * noise, 0.0, 1.0)
    return Image.fromarray(np.round(a * 255.0).astype(np.uint8))


def apply_input(x: torch.Tensor, op: Op, gen: torch.Generator) -> torch.Tensor:
    """`op` on network inputs (B, 3, H, W) in [0, 1], in floating point, as the
    notebook's `pipeline_ops` (its _jpeg_roundtrip, _requantise, _resize_roundtrip, noise)."""
    from PIL import Image

    if op.kind == "jpeg":
        out = []
        for img in x:
            a = (img.permute(1, 2, 0).numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
            b = np.asarray(_jpeg_pil(Image.fromarray(a), op.value)).astype(np.float32) / 255.0
            out.append(torch.from_numpy(b).permute(2, 0, 1))
        return torch.stack(out)
    if op.kind == "bits":
        levels = 2 ** int(op.value) - 1
        return (x * levels).round() / levels
    if op.kind == "resize":
        side = x.shape[-1]
        s = resize_side(side, op.value)
        h = F.interpolate(x, size=(s, s), mode="bicubic", align_corners=False)
        return F.interpolate(h, size=x.shape[-2:], mode="bicubic", align_corners=False).clamp(0, 1)
    return (x + op.value * torch.randn(x.shape, generator=gen)).clamp(0, 1)


_REFERENCE: dict | None = None


def reference_means() -> dict:
    """The fitted constants of the dataset's colour normalisation: channel means (RGB)
    of the reference image after the crop, as integers the way the dataset's
    normalisation truncates them, and as floats. The photograph is the setting DERMA_COLOR_REFERENCE."""
    global _REFERENCE
    if _REFERENCE is None:
        from PIL import Image

        from experiments.common.locations import color_reference

        path = color_reference()
        with Image.open(path) as image:
            a = np.asarray(image.convert("RGB"))
        a = a[:, REFERENCE_CROP:a.shape[1] - REFERENCE_CROP]
        means = np.mean(a, axis=(0, 1))
        _REFERENCE = {"sha256": env.sha256_file(path), "crop": REFERENCE_CROP,
                      "rgb_int": [int(v) for v in means.astype(np.int32)],
                      "global_int": int(np.mean(a).astype(np.int32)),
                      "rgb": [float(v) for v in means]}
    return _REFERENCE


def renormalise(image, mode: str, clean=None):
    """Colour-normalise a decoded 8-bit RGB image (PIL) again; `clean` is the image
    before the operation (needed by `oracle`). See the module docstring for the modes."""
    from PIL import Image

    if mode == "none":
        return image
    a = np.asarray(image)
    if mode == "authors":
        # the dataset's colour normalisation, channels in RGB order
        ref = reference_means()
        mean_rgb = np.mean(a, axis=(0, 1)).astype(np.int32)
        gl_mean = np.mean(a, axis=(0, 1, 2)).astype(np.int32)
        delta = (np.subtract(ref["rgb_int"], mean_rgb) * COLOR_PENALTY).astype(np.int32)
        shifted = a.astype(np.int32) + delta + (ref["global_int"] - gl_mean)
        return Image.fromarray(np.clip(shifted, 0, 255).astype(np.uint8))
    if mode == "ref":
        target = np.array(reference_means()["rgb"])
    elif mode == "oracle":
        if clean is None:
            raise ValueError("renormalisation 'oracle' needs the image before the operation")
        target = np.mean(np.asarray(clean), axis=(0, 1))
    else:
        raise ValueError(f"unknown renormalisation {mode!r}, expected one of {RENORMS}")
    shifted = a.astype(np.float64) + (target - np.mean(a, axis=(0, 1)))
    return Image.fromarray(np.clip(np.round(shifted), 0, 255).astype(np.uint8))


def load_inputs(paths: list[Path], op: Op | None = None, at: str = "native",
                renorm: str = "none") -> torch.Tensor:
    """Network inputs of the files (as `data.load_images`), after `op` applied `at` the
    native file or at the network input, and then (native only) colour-normalised
    again by `renorm`. Noise is seeded, the same for every call."""
    from PIL import Image

    if renorm != "none" and at != "native":
        raise ValueError("a renormalisation needs at='native' (it works on the 8-bit file)")
    transform = bdata.bronet_transform()
    gen = torch.Generator().manual_seed(NOISE_SEED)
    out = []
    for p in paths:
        with Image.open(p) as image:
            image = image.convert("RGB")
        if op is not None and at == "native":
            changed = apply_native(image, op, gen)
            image = renormalise(changed, renorm, clean=image)
        out.append(transform(image))
    x = torch.stack(out)
    if op is not None and at == "input":
        x = apply_input(x, op, gen)
    return x


# ---------------------------------------------------------------------------
# features of the transformed test images (model independent, cached)
# ---------------------------------------------------------------------------

def _cache_meta(step: int, paths: list[Path], op: Op, at: str, renorm: str = "none") -> dict:
    import PIL

    meta = {"step": step, "op": op.key, "kind": op.kind, "value": op.value, "at": at,
            "noise_seed": NOISE_SEED,
            "images_sha256": env.sha256_json([bdata.relative_path(p) for p in paths]),
            "transform": bdata.TRANSFORM_VERSION, "pillow": PIL.__version__,
            "bronet_ckpt_sha256": env.checkpoint_sha256()}
    if renorm != "none":       # absent otherwise: the caches made before stay valid
        meta.update(renorm=renorm, penalty=COLOR_PENALTY, reference=reference_means())
    return meta


def _read_cache(path: Path, meta: dict) -> dict | None:
    if not path.exists():
        return None
    blob = torch.load(path, map_location="cpu", weights_only=False)
    return blob if blob.get("meta") == meta else None


def _test_paths(step: int) -> list[Path]:
    _, test, _ = bdata.records(step)
    return [r.path for r in test]


def cached_features(step: int, ops: list[str], at: str, renorm: str = "none") -> dict:
    """{op key: blob} for the operations whose cache is valid (no BRONet needed)."""
    paths = _test_paths(step)
    where = place(at, renorm)
    out = {k: _read_cache(TREE.lossy_features_file(step, where, k),
                          _cache_meta(step, paths, get_op(k), at, renorm))
           for k in ops}
    return {k: b for k, b in out.items() if b is not None}


def transformed_features(step: int, ops: list[str], at: str, *, device, strategy: str,
                         cache_bro_weights: bool, eval_chunk: int = 256, minibatch: int = 32,
                         logger=None, renorm: str = "none") -> dict:
    """{op key: {"images", "dist", "features", ...}} for the test images of `step`; the
    real BRONet computes what the cache misses. With a renormalisation, "dist" is
    ||N(T(x)) − x|| and "dist_attack" ||T(x) − x||."""
    from . import bronet, features

    log = logger.info if logger else print
    paths = _test_paths(step)
    where = place(at, renorm)
    out = cached_features(step, ops, at, renorm)
    todo = [k for k in ops if k not in out]
    if not todo:
        return {k: out[k] for k in ops}
    net, _ = bronet.load_bronet(device, strategy, logger, cache_bro_weights=cache_bro_weights)

    def phi(x):
        return torch.cat([bronet.phi_layer_major(net, x[i:i + eval_chunk], device, minibatch)
                          for i in range(0, len(x), eval_chunk)])

    x = load_inputs(paths)
    clean_diff = float((phi(x) - features.load(step, paths)).abs().max())
    log(f"clean phi end to end vs the feature cache: max |diff| {clean_diff:.2e}")
    for k in todo:
        op = get_op(k)
        xt = load_inputs(paths, op, at, renorm)
        dist = (xt - x).flatten(1).double().norm(dim=1)
        label = op.label(600 if at == "native" else 224)
        blob = {"meta": _cache_meta(step, paths, op, at, renorm),
                "label": label if renorm == "none" else f"{label} + renorm {renorm}",
                "images": [p.name for p in paths], "dist": dist, "features": phi(xt),
                "clean_phi_max_abs_diff": clean_diff}
        if renorm != "none":
            blob["dist_attack"] = (load_inputs(paths, op, at) - x).flatten(1).double().norm(dim=1)
        env.atomic_torch_save(blob, TREE.lossy_features_file(step, where, k))
        out[k] = blob
        log(f"{blob['label']} ({at}): median ||x_T - x|| {float(dist.median()):.4f} "
            f"({float(dist.median()) * 255:.1f}/255)"
            + (f", without renorm {float(blob['dist_attack'].median()):.4f}"
               if "dist_attack" in blob else ""))
    return {k: out[k] for k in ops}


# ---------------------------------------------------------------------------
# coverage of one run
# ---------------------------------------------------------------------------

def coverage(dist, r_off, r_rig, y, pred, pred_t) -> dict:
    """Per-image arrays: covered by the certificate (official / rigorous L), factor
    by which r would have to grow (correct images; inf where r = 0), changed prediction."""
    dist, r_off, r_rig = (np.asarray(a, dtype=float) for a in (dist, r_off, r_rig))
    correct = np.asarray(y) == np.asarray(pred)
    with np.errstate(divide="ignore"):
        factor = np.where(correct, dist / r_off, np.nan)
    return {"correct": correct,
            "covered_official": correct & (dist < r_off),
            "covered_rigorous": correct & (dist < r_rig),
            "factor": factor,
            "changed": np.asarray(pred_t) != np.asarray(pred),
            "correct_after": np.asarray(pred_t) == np.asarray(y)}


def run_coverage(run_dir: Path, which: str, transformed: dict, device) -> list[dict]:
    """Per (image, operation) rows of one finished run."""
    run_dir = Path(run_dir)
    arrays = np.load(run_dir / layout.arrays_file("certificates", which))
    with open(run_dir / layout.predictions_file(which), newline="", encoding="utf-8") as handle:
        names = [r["image"] for r in csv.DictReader(handle)]
    model = train.load_run(run_dir, device=device, which=which)
    rows = []
    for key, blob in transformed.items():
        if blob["images"] != names:
            raise RuntimeError(f"{run_dir}: test images differ from the cache of {key}")
        with torch.no_grad():
            pred_t = model(blob["features"].to(device)).argmax(1).cpu().numpy()
        c = coverage(blob["dist"].numpy(), arrays["radius_official"], arrays["radius_rigorous"],
                     arrays["y"], arrays["pred"], pred_t)
        for i, name in enumerate(names):
            row = {"image": name, "operation": key, "dist": float(blob["dist"][i]),
                   "radius_official": float(arrays["radius_official"][i]),
                   "radius_rigorous": float(arrays["radius_rigorous"][i]),
                   "correct": bool(c["correct"][i]),
                   "covered_official": bool(c["covered_official"][i]),
                   "covered_rigorous": bool(c["covered_rigorous"][i]),
                   "factor": float(c["factor"][i]), "changed": bool(c["changed"][i]),
                   "correct_after": bool(c["correct_after"][i])}
            if "dist_attack" in blob:
                da = float(blob["dist_attack"][i])
                row["dist_attack"] = da
                row["covered_attack_official"] = bool(c["correct"][i]) and da < row["radius_official"]
            rows.append(row)
    return rows


def summarise(rows: list[dict]) -> dict:
    """One run, one operation: covered in % of the correct images, changed in % of all
    images, and the factor r needs to cover COVER_SHARES of the correct ones."""
    correct = [r for r in rows if r["correct"]]
    factors = np.array([r["factor"] for r in correct])
    out = {"n": len(rows), "n_correct": len(correct),
           "accuracy": 100 * len(correct) / len(rows),
           "accuracy_after": 100 * np.mean([r["correct_after"] for r in rows]),
           "median_dist": float(np.median([r["dist"] for r in rows])),
           "max_dist": float(np.max([r["dist"] for r in rows])),
           "median_radius": float(np.median([r["radius_official"] for r in correct])),
           "covered_official": 100 * np.mean([r["covered_official"] for r in correct]),
           "covered_rigorous": 100 * np.mean([r["covered_rigorous"] for r in correct]),
           "changed": 100 * np.mean([r["changed"] for r in rows])}
    for share in COVER_SHARES:
        out[f"factor_{share:g}"] = float(np.quantile(factors, share, method="inverted_cdf"))
    if rows and "dist_attack" in rows[0]:          # with a renormalisation
        out["median_dist_attack"] = float(np.median([r["dist_attack"] for r in rows]))
        out["covered_attack_official"] = 100 * np.mean([r["covered_attack_official"]
                                                        for r in correct])
    return out


# ---------------------------------------------------------------------------
# which runs
# ---------------------------------------------------------------------------

def protocol_runs(step: int, modes, variants, seeds) -> list[tuple[str, int, Path]]:
    """(run name, seed, dir) of the finished runs of the batch-256 protocol."""
    found = []
    for mode in modes:
        for variant in variants:
            for seed in seeds:
                rd = env.run_dir(step, mode, seed, variant)
                if train.is_complete(rd):
                    found.append((layout.run_name(mode, variant), seed, rd))
    return found


def sweep_run_dir(step: int, eps_255: float, mode: str, seed: int, variant: str = "best",
                  warmup: float = 0.0, start_255: float = 0.0) -> Path:
    """eps = 36/255 without warm-up is the protocol itself: its run is the existing one."""
    if math.isclose(eps_255, DEFAULT_EPS_255) and not warmup:
        return env.run_dir(step, mode, seed, variant)
    return TREE.eps_sweep_run_dir(step, eps_255, mode, seed, variant,
                                  layout.warmup_tag(warmup, start_255 / 255))


def sweep_config(step: int, mode: str, seed: int, eps_255: float, variant: str = "best",
                 device: str = "auto", warmup: float = 0.0,
                 start_255: float = 0.0) -> train.RunConfig:
    """The final protocol of `run_grid --from-tuning` with eps_train = eps_255 / 255
    (and an eps warm-up over the fraction `warmup` of the epochs, from start_255/255);
    learning rate and epoch ceiling from the tuning at the protocol's eps."""
    from . import tune

    cfg = train.RunConfig(step=step, mode=mode, seed=seed, device=device)
    tuned = tune.tuned_params(step, mode, cfg, variant)
    cfg.lr, cfg.epochs, cfg.early_stop = tuned["lr"], tuned["ceiling"], "val"
    if not math.isclose(eps_255, DEFAULT_EPS_255):
        cfg.eps_train = eps_255 / 255
    if warmup:      # replace() re-runs the checks of RunConfig
        cfg = dataclasses.replace(cfg, eps_warmup=warmup, eps_start=start_255 / 255)
    return cfg


def _variant_runs(step: int, mode: str, variant: str, log) -> bool:
    """Whether `variant` has runs of its own (as run_grid decides)."""
    if variant == "best":
        return True
    from . import tune

    if tune.stable_equals_best(step, mode):
        return False
    sel = env.read_json(tune.selection_path(step))["modes"][mode]
    if sel["variants"]["stable"] is None:
        log(f"{mode}: no stable grid point, no stable runs")
        return False
    return True


# ---------------------------------------------------------------------------
# reports
# ---------------------------------------------------------------------------

def _mean_sd(values) -> str:
    v = np.asarray(values, dtype=float)
    return f"{v.mean():.1f}" if len(v) < 2 else f"{v.mean():.1f} ± {v.std(ddof=1):.1f}"


def _groups(rows: list[dict], key) -> dict:
    out = {}
    for r in rows:
        out.setdefault(tuple(r[k] for k in key), []).append(r)
    return out


def _run_order() -> dict:
    return {n: i for i, n in enumerate(layout.run_name(m, v) for m in adapters.MODES
                                       for v in ("best", "stable"))}


RENORM_TEXT = {
    "authors": "the dataset's own colour normalisation (its reference photograph, damping "
               "0.8 per channel plus an undamped global-mean term, integer arithmetic)",
    "ref": "every channel mean shifted exactly to the reference image's",
    "oracle": "every channel mean shifted back to that of the image before the operation "
              "(an upper bound: it needs the original)"}


def _where(at: str, renorm: str = "none") -> str:
    where = ("applied to the delivered 600×600 file, then the network preprocessing"
             if at == "native" else "applied to the 224×224 network input (as in the robustness notebook)")
    if renorm == "none":
        return where
    return (f"applied to the delivered 600×600 file, which is then colour-normalised again "
            f"({RENORM_TEXT[renorm]}) and goes through the network preprocessing")


def write_measure(od: Path, step: int, which: str, at: str, transformed: dict,
                  per_seed: list[dict], renorm: str = "none") -> None:
    env.write_csv(od / "coverage.csv", per_seed)
    labels = {k: b["label"] for k, b in transformed.items()}
    rn = renorm != "none"
    lines = [f"# Lossy image handling against the certified radius, step-{step} "
             f"({which} checkpoint)", "",
             f"Every operation is {_where(at, renorm)}; d = ||x_T − x||₂ in the 224×224 input "
             "(pixels in [0, 1]), the space of the certified radius r"
             + (", where x_T is the changed and renormalised image and x the delivered one; "
                "r is the radius of x" if rn else "")
             + ". *Covered*: correct and "
             "d < r (official L) — the certificate proves the prediction unchanged. *Changed*: "
             "the prediction actually changed. Covered in % of correctly classified images, "
             "changed in % of all test images; means over seeds.", "",
             "## How large the change is (model independent)", "",
             "| operation | median d | d at 50 / 75 / 90 % of images | in units of 1/255 |"
             + (" median d without renormalisation |" if rn else ""),
             "|---|---|---|---|" + ("---|" if rn else "")]
    for k, b in transformed.items():
        d = b["dist"].numpy()
        qs = [float(np.quantile(d, s, method="inverted_cdf")) for s in COVER_SHARES]
        lines.append(f"| {labels[k]} | {np.median(d):.3f} | {' / '.join(f'{v:.3f}' for v in qs)} | "
                     f"{' / '.join(f'{v * 255:.0f}' for v in qs)} |"
                     + (f" {float(b['dist_attack'].median()):.3f} |" if rn else ""))
    lines += ["", "The last column is the eps_train (in units of 1/255) at which the training "
              "loss asks for a radius as large as the change of that share of the images.",
              "", "## Coverage per mode", "",
              "| mode | seeds | operation | median r | median d | covered, % | covered (rigorous L), % "
              "| changed, % | r must grow ×, to cover 50 / 75 / 90 % |"
              + (" covered without renormalisation, % |" if rn else ""),
              "|---|---|---|---|---|---|---|---|---|" + ("---|" if rn else "")]
    order, op_order = _run_order(), {k: i for i, k in enumerate(transformed)}
    groups = _groups(per_seed, ("run", "operation"))
    for run, k in sorted(groups, key=lambda g: (order.get(g[0], 99), op_order[g[1]])):
        g = groups[(run, k)]
        f = " / ".join(f"{np.mean([r[f'factor_{s:g}'] for r in g]):.1f}" for s in COVER_SHARES)
        lines.append(f"| {run} | {len(g)} | {labels[k]} | {np.mean([r['median_radius'] for r in g]):.3f} | "
                     f"{np.mean([r['median_dist'] for r in g]):.3f} | "
                     f"{_mean_sd([r['covered_official'] for r in g])} | "
                     f"{_mean_sd([r['covered_rigorous'] for r in g])} | "
                     f"{_mean_sd([r['changed'] for r in g])} | {f} |"
                     + (f" {_mean_sd([r['covered_attack_official'] for r in g])} |" if rn else ""))
    lines += ["", "*r must grow ×*: the factor d / r over the correctly classified images, at the "
              "given share — the certified radius must be this many times larger for the "
              "certificate to cover the operation for that share of the images (mean over seeds)."]
    if rn:
        lines += ["", "*no change*: the renormalisation alone, applied to the delivered file. "
                  "*Covered without renormalisation*: the same images and radii, with "
                  "d = ||T(x) − x|| — the measurement without `--renorm`."]
    env.atomic_write_text(od / "coverage.md", "\n".join(lines) + "\n")


def _warmup_text(warmup: float, start_255: float) -> str:
    if not warmup:
        return ""
    return (f" The loss eps ramps linearly from {start_255:g}/255 to eps_train over the first "
            f"{warmup:g} of the epochs; the best-validation epoch is chosen after the ramp.")


def write_sweep(od: Path, step: int, which: str, at: str, rows: list[dict], transformed: dict,
                warmup: float = 0.0, start_255: float = 0.0, renorm: str = "none") -> None:
    wtag = layout.warmup_tag(warmup, start_255 / 255)
    env.write_csv(od / f"eps_sweep{wtag}.csv", rows)
    lines = [f"# eps_train sweep{' with eps warm-up' if warmup else ''}, step-{step} "
             f"({which} checkpoint)", "",
             "Final protocol at batch 256 with the learning rate tuned at eps = 36/255; only "
             "eps_train in the certificate-aware loss changes." + _warmup_text(warmup, start_255)
             + " Mean ± SD over seeds.", "",
             "| mode | eps_train | seeds | clean, % | collapsed | median r | L/L₀ | "
             + " | ".join(f"cert. {n}, %" for n in EPS_NAMES) + " |",
             "|---|---|---|---|---|---|---|" + "---|" * len(EPS_NAMES)]
    order = _run_order()
    groups = _groups(rows, ("run", "eps_255"))
    keys = sorted(groups, key=lambda g: (order.get(g[0], 99), g[1]))
    for run, eps in keys:
        g = groups[(run, eps)]
        lines.append(f"| {run} | {eps:g}/255 | {len(g)} | {_mean_sd([r['accuracy'] for r in g])} | "
                     f"{sum(r['collapsed'] for r in g)} | "
                     f"{np.mean([r['median_radius'] for r in g]):.3f} | "
                     f"{np.exp(np.mean(np.log([r['L_over_L0'] for r in g]))):.4g} | "
                     + " | ".join(_mean_sd([r[f'cert_{n}'] for r in g]) for n in EPS_NAMES) + " |")
    lines += ["", "L/L₀: geometric mean over seeds. Collapsed: seeds whose model predicts one class."]
    if transformed:
        lines += ["", f"## Coverage of lossy handling ({_where(at, renorm)})", "",
                  "Covered: % of correctly classified images whose change is below their "
                  "certified radius (official L); changed: % of all images whose prediction moved.",
                  "", "| mode | eps_train | " + " | ".join(b["label"] for b in transformed.values())
                  + " |", "|---|---|" + "---|" * len(transformed)]
        for run, eps in keys:
            g = groups[(run, eps)]
            cells = [f"{_mean_sd([r[f'covered_{k}'] for r in g])} / {_mean_sd([r[f'changed_{k}'] for r in g])}"
                     for k in transformed]
            lines.append(f"| {run} | {eps:g}/255 | " + " | ".join(cells) + " |")
        lines += ["", "Each cell: covered, % / changed, %."]
    env.atomic_write_text(od / f"eps_sweep{wtag}.md", "\n".join(lines) + "\n")


def sweep_rows(step: int, which: str, eps_list, modes, variants, seeds, transformed: dict,
               device, warmup: float = 0.0, start_255: float = 0.0) -> list[dict]:
    rows = []
    for eps in eps_list:
        for mode in modes:
            for variant in variants:
                for seed in seeds:
                    rd = sweep_run_dir(step, eps, mode, seed, variant, warmup, start_255)
                    if not train.is_complete(rd):
                        continue
                    m = env.read_json(rd / layout.metrics_file(which))
                    cert = m["certificate_official"]
                    row = {"run": layout.run_name(mode, variant), "eps_255": float(eps),
                           "seed": seed, "accuracy": m["accuracy"],
                           "collapsed": bool(m["collapsed"]),
                           "median_radius": cert["median_radius"],
                           "L_over_L0": m["lipschitz"]["L_over_L0"]}
                    for n in EPS_NAMES:
                        row[f"cert_{n}"] = 100 * cert["certified_accuracy_named"][n]
                    if transformed:
                        cov = run_coverage(rd, which, transformed, device)
                        for k in transformed:
                            s = summarise([r for r in cov if r["operation"] == k])
                            row[f"covered_{k}"] = s["covered_official"]
                            row[f"changed_{k}"] = s["changed"]
                    rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------

def cmd_measure(a, logger) -> int:
    device = env.resolve_device(a.device)
    strategy = env.resolve_bro_strategy(device, a.bro_weights)
    ops = with_identity(a.ops, a.renorm)
    transformed = transformed_features(a.step, ops, a.at, device=device, strategy=strategy,
                                       cache_bro_weights=a.cache_bro_weights,
                                       eval_chunk=a.eval_chunk, minibatch=a.minibatch,
                                       logger=logger, renorm=a.renorm)
    od = TREE.lossy_dir(a.step, a.checkpoint, place(a.at, a.renorm))
    per_seed, per_image = [], []
    for run, seed, rd in protocol_runs(a.step, a.modes, a.variants, a.seeds):
        rows = run_coverage(rd, a.checkpoint, transformed, device)
        per_image += [{"run": run, "seed": seed, **r} for r in rows]
        for k in transformed:
            per_seed.append({"run": run, "seed": seed, "operation": k,
                             **summarise([r for r in rows if r["operation"] == k])})
        logger.info(f"{run} seed {seed}: done")
    if not per_seed:
        raise FileNotFoundError(f"no finished runs under {env.runs_dir(a.step)}")
    env.write_csv(od / "coverage_images.csv", per_image)
    write_measure(od, a.step, a.checkpoint, a.at, transformed, per_seed, a.renorm)
    logger.info(f"-> {od}")
    return 0


def cmd_sweep(a, logger) -> int:
    inputs = None
    for eps in a.eps:
        for mode in a.modes:
            for variant in a.variants:
                if not _variant_runs(a.step, mode, variant, logger.info):
                    continue
                for seed in a.seeds:
                    rd = sweep_run_dir(a.step, eps, mode, seed, variant, a.eps_warmup,
                                       a.eps_start)
                    cfg = sweep_config(a.step, mode, seed, eps, variant, a.device,
                                       a.eps_warmup, a.eps_start)
                    name = (f"eps {eps:g}/255{layout.warmup_tag(cfg.eps_warmup, cfg.eps_start)} "
                            f"{layout.run_name(mode, variant)} seed {seed}")
                    if train.is_complete(rd) and not a.force:
                        diff = train.completed_config_mismatch(rd, cfg)
                        if diff:
                            raise RuntimeError(f"{rd} is complete but was trained with other "
                                               f"settings {diff}; pass --force to retrain it")
                        logger.info(f"skip {name}: complete")
                        continue
                    if math.isclose(eps, DEFAULT_EPS_255) and not a.eps_warmup:
                        raise RuntimeError(f"{rd} is missing: eps 36/255 is the protocol's own "
                                           "run, train it with run_grid --from-tuning")
                    if inputs is None:
                        inputs = train.final_inputs(a.step, logger)
                    logger.info(f"start {name}: lr {cfg.lr:g}, up to {cfg.epochs} epochs")
                    run_logger = env.setup_logging(rd / "train.log",
                                                   name=f"bronet_mlp.sweep.{eps:g}.{mode}.{seed}")
                    train.fit(cfg, inputs, rd, logger=run_logger, restart=a.force)
    a.eps = sorted(set(a.eps) | {DEFAULT_EPS_255})     # the protocol's runs as the reference
    return cmd_report(a, logger)


def cmd_report(a, logger) -> int:
    device = env.resolve_device(a.device)
    ops = with_identity(a.ops, a.renorm)
    transformed = cached_features(a.step, ops, a.at, a.renorm)
    missing = [k for k in ops if k not in transformed]
    if missing:
        logger.warning(f"no cache for {missing} ({place(a.at, a.renorm)}): run `lossy measure` "
                       "first; they are left out of the coverage table")
    rows = sweep_rows(a.step, a.checkpoint, a.eps, a.modes, a.variants, a.seeds, transformed,
                      device, a.eps_warmup, a.eps_start)
    if not rows:
        raise FileNotFoundError("no finished sweep runs")
    od = TREE.lossy_dir(a.step, a.checkpoint, place(a.at, a.renorm))
    write_sweep(od, a.step, a.checkpoint, a.at, rows, transformed, a.eps_warmup, a.eps_start,
                a.renorm)
    logger.info(f"-> {od / ('eps_sweep' + layout.warmup_tag(a.eps_warmup, a.eps_start / 255))}.md")
    return 0


def main(argv=None) -> int:
    from repro import console

    console.init()
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("command", choices=("measure", "sweep", "report"))
    p.add_argument("--step", type=int, choices=(1, 2), required=True)
    p.add_argument("--ops", nargs="+", choices=list(OPS), default=list(OPS),
                   help="operations to apply (default: all): " + " ".join(OPS))
    p.add_argument("--at", choices=PLACES, default="native",
                   help="native: on the 600x600 file (default); input: on the 224x224 "
                        "network input, as in the robustness notebook")
    p.add_argument("--renorm", choices=RENORMS, default="none",
                   help="measure / report: colour-normalise the changed file again before the "
                        "network (native only): authors | ref | oracle; see the module docstring")
    p.add_argument("--checkpoint", choices=("best", "final"), default="best",
                   help="checkpoint of the runs (best validation epoch by default)")
    p.add_argument("--modes", nargs="+", choices=adapters.MODES, default=list(adapters.MODES))
    p.add_argument("--variants", nargs="+", choices=("best", "stable"), default=["best", "stable"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--eps", nargs="+", type=float, default=[DEFAULT_EPS_255],
                   help="sweep / report: eps_train values in units of 1/255 (36 = the protocol)")
    p.add_argument("--eps-warmup", type=float, default=0.0,
                   help="sweep / report: ramp the loss eps over this fraction of the epochs "
                        "(0 = off, the original sweep; e.g. 0.33)")
    p.add_argument("--eps-start", type=float, default=0.0,
                   help="sweep / report: eps at the start of the warm-up, in units of 1/255")
    p.add_argument("--force", action="store_true", help="sweep: retrain finished runs")
    p.add_argument("--device", default="auto")
    p.add_argument("--bro-weights", default="auto")
    p.add_argument("--cache-bro-weights", action="store_true")
    p.add_argument("--eval-chunk", type=int, default=256,
                   help="measure: images per exact BRONet pass; lower on OOM")
    p.add_argument("--minibatch", type=int, default=32)
    a = p.parse_args(argv)
    a.ops = [k for k in OPS if k in a.ops]               # canonical order
    if a.renorm != "none" and a.at != "native":
        p.error("--renorm needs --at native")
    od = TREE.lossy_dir(a.step, a.checkpoint, place(a.at, a.renorm))
    logger = env.setup_logging(od / f"{a.command}.log", name=f"bronet_mlp.lossy.{a.command}")
    return {"measure": cmd_measure, "sweep": cmd_sweep, "report": cmd_report}[a.command](a, logger)


if __name__ == "__main__":
    sys.exit(main())
