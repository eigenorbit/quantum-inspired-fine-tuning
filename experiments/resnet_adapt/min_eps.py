"""Empirical minimal adversarial radius eps* on the adapted ResNet152 models, as
`bronet_mlp.min_eps`.

    python -m experiments.resnet_adapt.min_eps --step 2 --batch-size 32
    python -m experiments.resnet_adapt.min_eps --step 1 --batch-size 32 --attack apgd-ce

The search itself is `experiments.common.attacks.min_eps.search` (shared with BRONet): per
clean-correct image, doubling from max(2 r, eps_floor) up to eps_max, then
bisection; eps* = norm of the smallest perturbation found that flips the image.
What is ResNet-specific is as in `resnet_adapt.attacks`: the frozen Front in front
of the classifiers, the runs of the protocol (`--batch-size` = training batch),
seed 0, checkpoint best by default, the same test subset as BRONet (300 / 240
images). The certified radius uses the rigorous bound of spectrum.py (~10^-109), so
the check at 0.99 r is skipped (r < 1e-6) and eps*/r is not reported: eps* itself
is the result, comparable with the BRONet eps* image by image.

Outputs, <DERMA_REPRO_OUT>/resnet_adapt/results/attacks/step_<n>[_bs<B>]_<checkpoint>_mineps[_<attack>]/:
as bronet_mlp.min_eps, plus clean_agreement.json; per run
<run dir>/attacks/min_eps_<attack>_<checkpoint>.csv.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import paths

import torch

from experiments.common.attacks import core as attack_core
from experiments.common.attacks import min_eps as min_search
from repro import seeding
from experiments.common.device import resolve_device, device_info
from experiments.common.io import write_json, setup_logging
from experiments.common import layout

from . import attacks, backbone


def out_dir(step: int, batch_size: int, checkpoint: str, attack: str = "pgd") -> Path:
    return layout.RESNET.mineps_dir(step, batch_size, checkpoint, attack)


def main(argv=None) -> int:
    from repro import console

    console.init()
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--step", type=int, choices=(1, 2), required=True)
    p.add_argument("--train-batch-size", "--batch-size", dest="batch_size", type=int,
                   default=paths.DEFAULT_BATCH_SIZE,
                   help="training batch of the protocol whose runs are attacked (step_<n>_bs<B>; "
                        "--batch-size: alias)")
    p.add_argument("--checkpoint", choices=("best", "final"), default="best")
    p.add_argument("--runs", nargs="*", default=[],
                   help="run names (e.g. butterfly full); default: every finished run")
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--n-images", type=int, default=None,
                   help="test images (default: 300 on step-1, 240 on step-2, as BRONet)")
    min_search.add_search_args(p)
    p.add_argument("--attack-batch", type=int, default=16,
                   help="attack instances per batch; lower on OOM")
    p.add_argument("--eval-chunk", type=int, default=64,
                   help="images per exact re-evaluation pass; lower on OOM")
    p.add_argument("--super-chunk", type=int, default=256,
                   help="instances searched together between progress saves")
    p.add_argument("--device", default="auto")
    p.add_argument("--restart", action="store_true", help="discard saved progress")
    p.add_argument("--out", type=Path, default=None, help="other output folder (smoke tests)")
    a = p.parse_args(argv)

    od = a.out or out_dir(a.step, a.batch_size, a.checkpoint, a.attack)
    logger = setup_logging(od / "min_eps.log", name="resnet_adapt.min_eps")
    device = resolve_device(a.device)
    seeding.configure_determinism(True)
    n_images = a.n_images or attacks.DEFAULT_N_IMAGES[a.step]
    logger.info(f"device {device_info(device)}; step-{a.step}{paths.tag(a.batch_size)}, "
                f"{a.checkpoint} checkpoint, {n_images} images, probes {a.attack}")

    net = backbone.load_resnet152()
    targets = attacks.build_targets(a.step, a.batch_size, a.checkpoint, a.runs, a.seeds, net,
                                    device, logger)
    front = backbone.Front(net).to(device).eval()
    images = attacks.test_images(a.step, n_images, attack_core.AttackConfig(step=a.step).subset_seed)
    agree = attacks.clean_agreement(targets, front, images, device)
    write_json(od / "clean_agreement.json", agree)

    def feature_fn(x, grad):
        return front(x)

    cfg = min_search.MinEpsConfig(step=a.step, checkpoint=a.checkpoint,
                            runs=sorted({t["name"] for t in targets}), seeds=a.seeds,
                            n_images=n_images, grad_precision="fp32", **min_search.search_settings(a))
    try:
        res = min_search.search(cfg, device=device, strategy="resnet", batch_size=a.attack_batch,
                          eval_chunk=a.eval_chunk, super_chunk=a.super_chunk,
                          restart=a.restart, logger=logger, feature_fn=feature_fn,
                          targets=targets, images=images, out=od)
    except torch.cuda.OutOfMemoryError as exc:
        raise RuntimeError("CUDA out of memory: lower --attack-batch and/or --eval-chunk "
                           "(progress is saved)") from exc
    if a.out is None:
        for t in targets:
            mine = [r for r in res["rows"] if (r["run"], r["seed"]) == (t["name"], t["seed"])]
            attack_core._write_csv(Path(t["dir"]) / "attacks"
                                / layout.run_mineps_csv(a.attack, a.checkpoint), mine)
    if not res["sanity"]["ok"]:
        logger.error("a flip lies inside the certified radius -- see sanity_check.json; "
                     "this is a bug")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
