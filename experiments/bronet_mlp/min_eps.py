"""Empirical minimal adversarial radius eps*(x) per image, against the certified r(x).

    python -m experiments.bronet_mlp.min_eps --step 2 --n-images 240 --cache-bro-weights
    python -m experiments.bronet_mlp.min_eps --step 1 --n-images 300 --cache-bro-weights
    python -m experiments.bronet_mlp.min_eps --step 2 --n-images 240 --cache-bro-weights --attack apgd-ce

The certificate r(x) = min_j (z_p - z_j) / (L ||w_p - w_j||) (certify.radii, the
radius of every report) is a LOWER bound on the distance from x to the nearest
input with another class: no perturbation with ||delta|| < r(x) flips the
prediction. It is only sufficient, so a larger perturbation may not flip it either.
Here the smallest budget for which the attack DOES flip it is searched per image:

    r(x) <= true minimal adversarial distance <= eps*(x)

eps* is an upper bound (the attack may miss a smaller perturbation); the ratio
eps*/r ("tightness", >= 1) is the gap between what is proved and what the attack
achieves. Per clean-correct image (instance = run x image):

0. check below r (`--no-check-below-r` to skip): one attack at 0.99 r(x); it must
   not succeed (skipped when r < 1e-6, e.g. the vacuous ResNet radii);
1. bracket upward by doubling from lo = r(x) (safe by the theorem): probes at
   max(2 r, eps_floor), then x2 ... up to eps_max; the first success gives hi;
   no success up to eps_max -> censored (eps* > eps_max, kept as such, not dropped);
2. bisect `--bisect-steps` times on [lo, hi] (8 halvings: 0.4 % of eps*).
On every success hi = min(probe eps, ||delta|| actually used), so eps* is the norm
of a perturbation that really flips the image. If an attack succeeds below an eps
where an earlier one failed (attacks are not monotone in eps) lo is lowered to keep
lo <= hi; such instances are counted (`nonmonotone`).

Every probe is one attack of `attacks.attack_deltas` (PGD by default, or any
`--attack` of strong_attacks.py) with its own eps per row and its own seed per
(instance, round), on the gradient network (float16 BRO copies or exact); its
success is decided ONLY by the exact float32 re-evaluation, as in attacks.py.
Probes of all unfinished instances of a chunk run together in batches.

Sanity check (mandatory): a success with ||delta|| < r(x) (official or rigorous L)
is a certificate violation; the command exits with an error after saving.

Outputs, <DERMA_REPRO_OUT>/bronet_mlp/results/attacks/step_<n>[_bs<B>]_<checkpoint>_mineps[_<attack>]/:
min_eps_config.json, progress.jsonl (one line per instance), sanity_check.json,
min_eps_step<n>.csv / .md (per run: eps* quantiles, eps*/r, censored, empirical vs
certified accuracy at 36/72/108/255); per run <run dir>/attacks/min_eps_<attack>_<checkpoint>.csv.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

from . import env

import torch

from experiments.common import layout
from experiments.common.attacks import min_eps as _engine
from . import attacks
from experiments.common.attacks.min_eps import (add_search_args, instance_row, MinEpsConfig,
                                                probe, R_TINY, REL_TOL, search_chunk,
                                                search_settings, summarise, violations,
                                                write_summary, _fmt, _quantile)  # noqa: F401  (re-exported)


def out_dir(step: int, checkpoint: str, batch_size: int = env.DEFAULT_BATCH_SIZE,
            attack: str = "pgd") -> Path:
    return layout.BRONET.mineps_dir(step, batch_size, checkpoint, attack)


# ---------------------------------------------------------------------------
# the search, independent of BRONet (tested on toy models)
# ---------------------------------------------------------------------------


BACKEND = dataclasses.replace(attacks.BACKEND, out_dir=out_dir)


def search(cfg: MinEpsConfig, *, device, strategy: str, batch_size: int = 32,
           eval_chunk: int = 256, super_chunk: int = 256, cache_bro_weights: bool = False,
           restart: bool = False, logger=None, feature_fn=None, targets=None, images=None,
           out: Path | None = None, train_batch_size: int = env.DEFAULT_BATCH_SIZE) -> dict:
    """The full search (`experiments.common.attacks.min_eps.search`) on BRONet;
    injection of feature_fn / targets / images as attacks.run."""
    return _engine.search(cfg, device=device, strategy=strategy, batch_size=batch_size,
                          eval_chunk=eval_chunk, super_chunk=super_chunk,
                          cache_bro_weights=cache_bro_weights, restart=restart, logger=logger,
                          feature_fn=feature_fn, targets=targets, images=images, out=out,
                          train_batch_size=train_batch_size, backend=BACKEND)


def main(argv=None) -> int:
    from repro import console

    console.init()
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--step", type=int, choices=(1, 2), required=True)
    p.add_argument("--checkpoint", choices=("headline", "best", "final"), default="headline")
    p.add_argument("--runs", nargs="*", default=[],
                   help="run names (e.g. butterfly lora_stable); default: all finished")
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--n-images", type=int, default=200)
    add_search_args(p)
    p.add_argument("--grad-precision", choices=("auto", "fp16", "fp32"), default="auto")
    p.add_argument("--attack-batch", "--batch-size", dest="batch_size", type=int, default=32,
                   help="attack instances per batch; lower on OOM "
                        "(--batch-size: alias)")
    p.add_argument("--train-batch-size", type=int, default=env.DEFAULT_BATCH_SIZE,
                   help="training batch of the runs (runs/step_<n>_bs<B>/ unless 256)")
    p.add_argument("--eval-chunk", type=int, default=256)
    p.add_argument("--super-chunk", type=int, default=256,
                   help="instances searched together between progress saves")
    p.add_argument("--device", default="auto")
    p.add_argument("--bro-weights", default="auto")
    p.add_argument("--cache-bro-weights", action="store_true")
    p.add_argument("--restart", action="store_true", help="discard saved progress")
    a = p.parse_args(argv)
    device = env.resolve_device(a.device)
    strategy = env.resolve_bro_strategy(device, a.bro_weights)
    grad = a.grad_precision
    if grad == "auto":
        grad = "fp16" if strategy == "stream" else "fp32"
    if grad == "fp16" and device.type != "cuda":
        p.error("--grad-precision fp16 needs a CUDA device")
    cfg = MinEpsConfig(step=a.step, checkpoint=a.checkpoint, runs=a.runs, seeds=a.seeds,
                       n_images=a.n_images, grad_precision=grad, **search_settings(a))
    od = out_dir(a.step, a.checkpoint, a.train_batch_size, a.attack)
    logger = env.setup_logging(od / "min_eps.log")
    logger.info(f"device {env.device_info(device)}  BRO weights: {strategy}, gradients {grad}")
    try:
        res = search(cfg, device=device, strategy=strategy, batch_size=a.batch_size,
                     eval_chunk=a.eval_chunk, super_chunk=a.super_chunk,
                     cache_bro_weights=a.cache_bro_weights, restart=a.restart, logger=logger,
                     train_batch_size=a.train_batch_size)
    except torch.cuda.OutOfMemoryError as exc:
        raise RuntimeError("CUDA out of memory: lower --batch-size and/or --eval-chunk "
                           "(progress is saved)") from exc
    if not res["sanity"]["ok"]:
        logger.error("CERTIFICATE VIOLATED: a flip inside the certified radius -- see "
                     "sanity_check.json")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
