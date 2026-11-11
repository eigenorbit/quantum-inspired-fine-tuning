"""Does the certificate of the conv_adapt runs cover lossy image handling (JPEG ...)?

    python -m experiments.bronet_mlp.conv_lossy --step 2 --blocks 1 --cache-bro-weights
    python -m experiments.bronet_mlp.conv_lossy --step 2 --blocks 1 --ops jpeg90 --renorm oracle --cache-bro-weights

The measurement of `lossy.py measure` (same operations, same `--at` / `--renorm`, same
coverage arithmetic and tables) for the runs with rotations inside the last BRO
blocks. Every transformed test image goes once through the frozen part of BRONet up
to the first adapted block (shared by all runs, ~1 min per operation on step-2);
each run then applies only its own adapted blocks, the neck, its MLP and head.

d(x) = ||T(x) - x||_2 in the 224x224 input (model independent) is compared with the
run's certified radius r(x) from its certificates_<checkpoint>.npz -- the numbers
of its metrics. *Changed* compares the prediction on T(x) with the saved one; the
saved predictions come from the float16 block-input cache, the transformed ones are
float32, so the number of clean images whose float32 prediction differs from the
saved one is reported too (`clean_pred_mismatch`, expected 0).

Outputs (numbers only): results/conv_adapt/lossy_step<n>_last<k>_<checkpoint>_<place>/
coverage.md / coverage.csv (per run and seed), coverage_images.csv (per image),
config.json.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import env

import numpy as np
import torch

from experiments.common import layout

from . import adapters, conv_adapt, conv_attacks, lossy

TREE = layout.BRONET


# ---------------------------------------------------------------------------
# model-independent part: d(x) and the input of the first adapted block
# ---------------------------------------------------------------------------

class Inputs:
    """The test images of `step`, clean or transformed by one operation at a time, as
    the input of the first adapted block (float32, CPU). One operation is held at a
    time: step-1 needs ~1.5 GB per operation."""

    def __init__(self, step: int, blocks: int, at: str, renorm: str, *, net, device,
                 chunk: int = 128, minibatch: int = 32):
        self.paths = lossy._test_paths(step)
        self.images = [p.name for p in self.paths]
        self.at, self.renorm, self.net, self.device = at, renorm, net, device
        self.first = net.conv.depth - blocks
        self.chunk, self.minibatch = chunk, minibatch
        self.x = lossy.load_inputs(self.paths)

    @torch.no_grad()
    def h_of(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([conv_adapt.block_inputs(self.net, x[i:i + self.chunk], self.device,
                                                  self.first, self.minibatch)
                          for i in range(0, len(x), self.chunk)])

    def clean(self) -> torch.Tensor:
        return self.h_of(self.x)

    def transformed(self, key: str) -> dict:
        """{"h", "dist", "label"[, "dist_attack"]} for operation `key`."""
        op = lossy.get_op(key)
        xt = lossy.load_inputs(self.paths, op, self.at, self.renorm)
        label = op.label(600 if self.at == "native" else 224)
        blob = {"h": self.h_of(xt), "dist": (xt - self.x).flatten(1).double().norm(dim=1),
                "label": label if self.renorm == "none" else f"{label} + renorm {self.renorm}"}
        if self.renorm != "none":
            blob["dist_attack"] = (lossy.load_inputs(self.paths, op, self.at) - self.x
                                   ).flatten(1).double().norm(dim=1)
        return blob


# ---------------------------------------------------------------------------
# one run, one operation
# ---------------------------------------------------------------------------

@torch.no_grad()
def predictions(model, h: torch.Tensor, device, micro: int = 32) -> np.ndarray:
    model.eval()
    return torch.cat([model(h[i:i + micro].to(device)).argmax(1).cpu()
                      for i in range(0, len(h), micro)]).numpy()


def op_rows(key: str, blob: dict, arrays, pred_t: np.ndarray, images: list[str]) -> list[dict]:
    """Per-image rows of one run and one operation, as lossy.run_coverage."""
    c = lossy.coverage(blob["dist"].numpy(), arrays["radius_official"], arrays["radius_rigorous"],
                       arrays["y"], arrays["pred"], pred_t)
    rows = []
    for i, name in enumerate(images):
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


# ---------------------------------------------------------------------------
# command
# ---------------------------------------------------------------------------

def evaluate(inputs, models: list[tuple[str, int, object, dict]], keys: list[str], device,
             log=print) -> tuple[list[dict], list[dict], dict, dict]:
    """(per-seed summaries, per-image rows, clean mismatches, blobs without h) for the
    models [(run name, seed, model, certificate arrays)] and the operations `keys`."""
    h_clean = inputs.clean()
    mismatches = {f"{n}/seed{s}": int((predictions(m, h_clean, device) != arr["pred"]).sum())
                  for n, s, m, arr in models}
    del h_clean
    rows = {(n, s): [] for n, s, _, _ in models}
    blobs = {}
    for key in keys:
        blob = inputs.transformed(key)
        for n, s, m, arr in models:
            rows[(n, s)] += op_rows(key, blob, arr, predictions(m, blob["h"], device), inputs.images)
        blobs[key] = {k: v for k, v in blob.items() if k != "h"}
        log(f"{blob['label']}: median d {float(blob['dist'].median()):.4f}, {len(models)} runs done")
        del blob
    per_seed, per_image = [], []
    for (n, s), rs in rows.items():
        per_image += [{"run": n, "seed": s, **r} for r in rs]
        for key in keys:
            per_seed.append({"run": n, "seed": s, "operation": key,
                             **lossy.summarise([r for r in rs if r["operation"] == key])})
    return per_seed, per_image, mismatches, blobs


def measure(step: int, blocks: int, ops: list[str], at: str, renorm: str, checkpoint: str, *,
            modes, variants, seeds, device, strategy: str, cache_bro_weights: bool,
            logger) -> Path:
    from . import bronet

    log = logger.info
    runs = conv_attacks.discover(step, blocks, modes, variants, seeds)
    net, _ = bronet.load_bronet(device, strategy, logger, cache_bro_weights=cache_bro_weights)
    for p in net.parameters():
        p.requires_grad_(False)
    tail = conv_adapt.BroTail(net, net.conv.depth - blocks, device)
    base = bronet.load_mlp_base()
    models = []
    for name, seed, d in runs:
        model, _ = conv_attacks.load_conv_run(d, checkpoint, tail, base["weight"], base["bias"],
                                              device)
        models.append((name, seed, model,
                       dict(np.load(d / layout.arrays_file("certificates", checkpoint)))))
    keys = lossy.with_identity(ops, renorm)
    inputs = Inputs(step, blocks, at, renorm, net=net, device=device)
    per_seed, per_image, mismatches, blobs = evaluate(inputs, models, keys, device, log)
    od = TREE.conv_lossy_dir(step, blocks, checkpoint, lossy.place(at, renorm))
    env.write_csv(od / "coverage_images.csv", per_image)
    lossy.write_measure(od, step, checkpoint, at, blobs, per_seed, renorm)
    env.write_json(od / "config.json", {"step": step, "blocks": blocks, "ops": keys, "at": at,
                                        "renorm": renorm, "checkpoint": checkpoint,
                                        "runs": [f"{n}/seed{s}" for n, s, _ in runs],
                                        "adapted_blocks": tail.blocks,
                                        "clean_pred_mismatch": mismatches,
                                        "device": env.device_info(device)})
    log(f"clean float32 vs saved predictions differ: {mismatches}")
    log(f"-> {od}")
    return od


def main(argv=None) -> int:
    from repro import console

    console.init()
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--step", type=int, choices=(1, 2), required=True)
    p.add_argument("--blocks", type=int, default=1)
    p.add_argument("--ops", nargs="+", choices=list(lossy.OPS),
                   default=["jpeg100", "jpeg95", "jpeg90", "jpeg80"])
    p.add_argument("--at", choices=lossy.PLACES, default="native")
    p.add_argument("--renorm", choices=lossy.RENORMS, default="none")
    p.add_argument("--checkpoint", choices=("best", "final"), default="final")
    p.add_argument("--mlp-modes", nargs="+", choices=adapters.MODES, default=list(adapters.MODES))
    p.add_argument("--variants", nargs="+", choices=("best", "stable", "manual"),
                   default=["best", "stable"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--device", default="auto")
    p.add_argument("--bro-weights", default="auto")
    p.add_argument("--cache-bro-weights", action="store_true")
    a = p.parse_args(argv)
    if a.renorm != "none" and a.at != "native":
        p.error("--renorm needs --at native")
    a.ops = [k for k in lossy.OPS if k in a.ops]
    device = env.resolve_device(a.device)
    strategy = env.resolve_bro_strategy(device, a.bro_weights)
    od = TREE.conv_lossy_dir(a.step, a.blocks, a.checkpoint, lossy.place(a.at, a.renorm))
    logger = env.setup_logging(od / "measure.log", name="bronet_mlp.conv_lossy")
    measure(a.step, a.blocks, a.ops, a.at, a.renorm, a.checkpoint, modes=a.mlp_modes,
            variants=a.variants, seeds=a.seeds, device=device, strategy=strategy,
            cache_bro_weights=a.cache_bro_weights, logger=logger)
    return 0


if __name__ == "__main__":
    sys.exit(main())
