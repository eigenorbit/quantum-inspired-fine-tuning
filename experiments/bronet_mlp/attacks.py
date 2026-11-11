"""Stage 2: end-to-end L2 attacks on the 224x224 input, checked against the certificate.

    python -m experiments.bronet_mlp.attacks --step 2 --cache-bro-weights
    python -m experiments.bronet_mlp.attacks --step 1 --n-images 300 --cache-bro-weights

For every trained run (default: every run name of the tuned protocol, seed 0,
the headline checkpoint) and every eps, on the same fixed random subset of test
images:

* PGD-L2 (as `pgd_l2` in experiments/robustness/experiments.py): start on the
  sphere of radius eps, `--steps` steps of size alpha = 2.5 eps / steps along
  the normalised gradient of the margin loss max_{j != y} z_j - z_y, projection
  onto the ball ||delta|| <= eps, `--restarts` random restarts. Gradients flow
  through the WHOLE frozen BRONet. No clamping to [0, 1], as in the notebook:
  the certificate is about the unconstrained ball, and clamping would only weaken
  the attack.
* random L2 noise of norm exactly eps, `--random-trials` directions.
* `--attack apgd-ce | apgd-dlr | apgd-t | square` replaces PGD by a stronger
  attack of AutoAttack (strong_attacks.py: APGD with adaptive step, momentum and
  best iterate; Square, a gradient-free random search) in the same threat model
  and the same two phases; `--steps` is then the APGD iteration count (default
  100), `--queries` / `--p-init` set Square, `--target-classes` APGD-T.

Only images the model classifies correctly are attacked (a misclassified image is
not robust anyway). Robust accuracy = clean-correct and no restart succeeded.

Two phases, for speed on a GPU that cannot hold the 7.7 GB of BRO matrices:

A. **directions** -- PGD in batches. With `--grad-precision fp16` (default when
   the fp32 matrices do not fit on the GPU) the gradients use float16 copies of
   the BRO matrices resident on the GPU (3.85 GB), so no weights move during
   PGD. With `fp32` (default when they fit, or on the CPU) the exact network.
B. **exact evaluation** -- every PGD result and every random perturbation is run
   again through BRONet in float32 (TF32 off), block-major in chunks of
   `--eval-chunk` images (each block copied once per chunk, as in the feature
   extraction). Success, robustness and the sanity check use ONLY these numbers,
   so a float16 gradient can at worst make the attack weaker, never create a
   false success or hide a certificate violation.

**Sanity check (mandatory).** The certificate says no perturbation with
||delta|| < r(x) changes the class. Every successful attack is therefore checked
against the radius of that model at that image, for the official-style and the
rigorous L: a success with ||delta|| < r(x) is a violation -- a bug in L or in
the formula. Any violation makes the command exit with an error (after saving).

Outputs, under <DERMA_REPRO_OUT>/bronet_mlp/results/attacks/step_<n>_<checkpoint>/
(..._<checkpoint>_<attack>/ for an attack other than PGD):
    attack_config.json   the settings; a resume with other settings is refused (--restart)
    progress.jsonl       one line per (run, seed, image, eps), appended per super-chunk
    sanity_check.json    violations (must be 0), clean-logit agreement with the training
    attacks_step<n>.csv / .md   per (run, seed, eps): clean, certified, random, PGD accuracy
and per run: <run dir>/attacks/<attack>_l2_<checkpoint>.csv (its rows of
progress.jsonl; pgd_l2_<checkpoint>.csv for PGD). The row fields `pgd_success`,
`pgd_robust` hold the result of the attack of the folder, whatever it is.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import env

import torch

from experiments.common import layout
from experiments.common.attacks import core as _core
from experiments.common.attacks.core import (add_attack_args, attack_deltas, attack_settings,
                                             attack_text, AttackConfig, Backend, clean_pass,
                                             EPS_NAMES, hash_seed, margin_loss, multi_logits,
                                             pgd_deltas, random_deltas, run_csv_name,
                                             select_images, STRONG_FIELDS, summarise,
                                             unit_directions, violations, write_summary,
                                             _append_progress, _norms, _read_progress, _row,
                                             _write_csv)  # noqa: F401  (re-exported)


# ---------------------------------------------------------------------------
# the experiment (the attack itself is experiments.common.attacks)
# ---------------------------------------------------------------------------


attack_tag = layout.attack_tag


def out_dir(step: int, checkpoint: str, batch_size: int = env.DEFAULT_BATCH_SIZE,
            attack: str = "pgd") -> Path:
    return layout.BRONET.attack_dir(step, batch_size, checkpoint, attack)


def discover_runs(step: int, names: list[str], seeds: list[int],
                  batch_size: int = env.DEFAULT_BATCH_SIZE) -> list[tuple[str, int, Path]]:
    from .report import RUN_NAMES, stale_reason

    found = []
    for name in (names or RUN_NAMES):
        for s in seeds:
            d = env.runs_dir(step, batch_size) / name / layout.seed_name(s)
            if not (d / "metrics.json").exists():
                continue
            why = stale_reason(step, name, env.read_json(d / "metrics.json"))
            if why:
                print(f"  skipping {d}: {why}")
                continue
            found.append((name, s, d))
    if not found:
        raise FileNotFoundError(f"no finished runs under {env.runs_dir(step, batch_size)} for {names or 'any'}")
    return found


def bronet_functions(cfg: AttackConfig, device, strategy: str, cache_bro_weights: bool,
                     eval_chunk: int, minibatch: int, logger=None):
    """(grad_fn, eval_fn) on the real BRONet. grad_fn: images on `device` -> phi
    with gradients (float16 BRO copies or exact); eval_fn: CPU images -> exact
    float32 phi on the CPU, block-major."""
    from . import bronet

    net, _ = bronet.load_bronet(device, strategy, logger, cache_bro_weights=cache_bro_weights)
    for p in net.parameters():
        p.requires_grad_(False)
    if cfg.grad_precision == "fp16":
        half = bronet.half_bro_weights(net, device)

        def grad_fn(x):
            return bronet.phi_with_grad_half(net, x, half)
    else:
        def grad_fn(x):
            return bronet.phi_with_grad(net, x)

    def eval_fn(X):
        return torch.cat([bronet.phi_layer_major(net, X[i:i + eval_chunk], device, minibatch)
                          for i in range(0, len(X), eval_chunk)])

    return grad_fn, eval_fn


def load_targets(cfg, device, train_batch_size: int = env.DEFAULT_BATCH_SIZE) -> list[dict]:
    """The frozen classifiers of the finished runs (cfg.runs, cfg.seeds, cfg.checkpoint)
    with their Lipschitz constants."""
    from . import train

    targets = []
    for name, s, d in discover_runs(cfg.step, cfg.runs, cfg.seeds, train_batch_size):
        which = (env.read_json(d / "metrics.json").get("headline", "final")
                 if cfg.checkpoint == "headline" else cfg.checkpoint)
        mf = d / layout.metrics_file(which)
        m = env.read_json(mf if mf.exists() else d / layout.HEADLINE_FILE)
        clf = train.load_run(d, device=device, which=which)
        for p in clf.parameters():
            p.requires_grad_(False)
        targets.append({"name": name, "seed": s, "dir": d, "which": which, "clf": clf,
                        "L_official": m["lipschitz"]["L_total_official"],
                        "L_rigorous": m["lipschitz"]["L_total_rigorous"]})
    return targets


def load_test_images(cfg) -> dict:
    """The fixed random subset of the test set (cfg.n_images, cfg.subset_seed)."""
    from . import data as bdata

    _, test, _ = bdata.records(cfg.step)
    idx = select_images(len(test), cfg.n_images, cfg.subset_seed)
    return {"x": bdata.load_images([test[i].path for i in idx]),
            "y": torch.tensor([test[i].label for i in idx]),
            "names": [test[i].path.name for i in idx], "test_index": idx}


BACKEND = Backend(name="BRONet", out_dir=out_dir, load_targets=load_targets,
                  load_test_images=load_test_images, network_functions=bronet_functions)


def network_functions(cfg, device, strategy: str, cache_bro_weights: bool, eval_chunk: int,
                      batch_size: int, timer, logger=None, feature_fn=None):
    """(grad_fn, eval_fn): an injected `feature_fn(x, grad)` (tests, ResNet) or BRONet."""
    return _core.network_functions(cfg, device, strategy, cache_bro_weights, eval_chunk,
                                   batch_size, timer, logger, feature_fn, BACKEND)


def run(cfg: AttackConfig, *, device, strategy: str, batch_size: int = 32,
        eval_chunk: int = 256, super_chunk: int = 512, cache_bro_weights: bool = False,
        restart: bool = False, logger=None, feature_fn=None, targets=None, images=None,
        out: Path | None = None, train_batch_size: int = env.DEFAULT_BATCH_SIZE) -> dict:
    """The full attack (`experiments.common.attacks.core.run`) on BRONet.
    `feature_fn(x, grad)`, `targets`, `images` are injected by the tests; otherwise
    BRONet, the trained runs and the test images are loaded."""
    return _core.run(cfg, device=device, strategy=strategy, batch_size=batch_size,
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
    add_attack_args(p)
    p.add_argument("--random-trials", type=int, default=4)
    p.add_argument("--grad-precision", choices=("auto", "fp16", "fp32"), default="auto",
                   help="BRO matrices for the PGD gradients; auto: fp32 if they sit on the "
                        "GPU (or on the CPU), else fp16 copies on the GPU")
    p.add_argument("--attack-batch", "--batch-size", dest="batch_size", type=int, default=32,
                   help="PGD instances (image x model x eps) per batch; lower on OOM "
                        "(--batch-size: alias)")
    p.add_argument("--train-batch-size", type=int, default=env.DEFAULT_BATCH_SIZE,
                   help="training batch of the runs to attack (runs/step_<n>_bs<B>/ unless 256)")
    p.add_argument("--eval-chunk", type=int, default=256,
                   help="images per block-major exact pass; lower on OOM")
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
    cfg = AttackConfig(step=a.step, checkpoint=a.checkpoint, runs=a.runs, seeds=a.seeds,
                       n_images=a.n_images, random_trials=a.random_trials, grad_precision=grad,
                       **attack_settings(a))
    od = out_dir(a.step, a.checkpoint, a.train_batch_size, a.attack)
    logger = env.setup_logging(od / "attacks.log")
    logger.info(f"device {env.device_info(device)}  BRO weights: {strategy}, gradients {grad}")
    try:
        res = run(cfg, device=device, strategy=strategy, batch_size=a.batch_size,
                  eval_chunk=a.eval_chunk, cache_bro_weights=a.cache_bro_weights,
                  restart=a.restart, logger=logger, train_batch_size=a.train_batch_size)
    except torch.cuda.OutOfMemoryError as exc:
        raise RuntimeError("CUDA out of memory: lower --batch-size and/or --eval-chunk "
                           "(progress is saved)") from exc
    if not res["sanity"]["ok"]:
        logger.error("CERTIFICATE VIOLATED by a successful attack -- see sanity_check.json. "
                     "This is a bug in L or in the radius formula; stop and investigate.")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
