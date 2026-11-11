"""Attacks and the eps* search on the conv_adapt runs (rotations inside the last BRO blocks).

    python -m experiments.bronet_mlp.conv_attacks attack --step 2 --blocks 1 --cache-bro-weights
    python -m experiments.bronet_mlp.conv_attacks attack --step 2 --blocks 1 --attack apgd-t --cache-bro-weights
    python -m experiments.bronet_mlp.conv_attacks mineps --step 2 --blocks 1 --cache-bro-weights

The engine is the protocol's, unchanged (experiments/common/attacks: core.run and
min_eps.search -- PGD, APGD-CE / DLR / T, Square, random noise, eps* search, the
exact float32 re-evaluation and the certificate sanity check). What differs: in a
conv_adapt run the network in front of the MLP is not the frozen BRONet but BRONet
with the run's rotations inside its last blocks, so every run is attacked with its
own feature function

    phi_run(x) = neck( adapted blocks ( frozen BRO blocks 0..d-k-1 ( conv_1(stem(x - 0.5)) ) ) )

on the same fixed subset of test images (`--n-images`, seed 999) as the protocol's
attacks, with the same per-instance seeds for runs of the same name (paired with
the protocol's runs). Gradients flow through the whole network; with
`--grad-precision fp16` the frozen blocks use float16 copies (gradients only, every
reported number is re-evaluated in float32).

Outputs (numbers only, no weights), under results/conv_adapt/:
    attacks_step<n>_last<k>_<checkpoint>[_<attack>]/<run>/seed<s>/   one engine folder
        per run: attack_config.json, progress.jsonl (one line per image and eps),
        sanity_check.json, attacks_step<n>.csv / .md
    attacks_step<n>_last<k>_<checkpoint>[_<attack>]/attacks_step<n>.csv / .md and
        sanity_check.json: every run together
    mineps_step<n>_last<k>_<checkpoint>[_<attack>]/...: the same for the eps* search
and a copy of every run's rows in its run folder (attacks/<attack>_l2_<checkpoint>.csv).
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

from . import env

import torch

from experiments.common import layout
from experiments.common.attacks import core
from experiments.common.attacks import min_eps as mineps_engine
from experiments.common.io import write_json

from . import adapters, conv_adapt, train

TREE = layout.BRONET
KINDS = {"attack": "attacks", "mineps": "mineps"}


# ---------------------------------------------------------------------------
# the trained conv runs
# ---------------------------------------------------------------------------

def discover(step: int, blocks: int, modes, variants, seeds) -> list[tuple[str, int, Path]]:
    """(run name, seed, dir) of the finished conv runs."""
    found = []
    for mode in modes:
        for variant in variants:
            for s in seeds:
                d = TREE.conv_run_dir(step, blocks, mode, s, variant)
                if train.is_complete(d):
                    found.append((layout.run_name(mode, variant), s, d))
    if not found:
        raise FileNotFoundError(f"no finished conv runs under "
                                f"{TREE.root() / 'conv_adapt' / f'step_{step}' / f'last{blocks}'}")
    return found


def load_conv_run(run_dir: Path, which: str, tail, base_weight, base_bias, device):
    """The ConvAdapted of a finished run, from one checkpoint; checks its fingerprint."""
    run_dir = Path(run_dir)
    record = env.read_json(run_dir / "config.json")
    cfg = conv_adapt.ConvConfig(**{k: v for k, v in record["config"].items()
                                   if k in {f.name for f in dataclasses.fields(conv_adapt.ConvConfig)}})
    from repro.config import N_CLASSES

    num_classes = N_CLASSES[cfg.step]
    clf = adapters.build(base_weight, base_bias, cfg.mlp_mode, num_classes, cfg.lora_rank,
                         cfg.seed)
    model = conv_adapt.ConvAdapted(tail, clf, cfg.stages, cfg.perm_seed).to(device)
    ckpt = torch.load(run_dir / layout.CHECKPOINTS_DIR / layout.checkpoint_file(which),
                      map_location="cpu", weights_only=True)
    model.load_trainable_state(ckpt["trainable_state"])
    if adapters.fingerprint(model.trainable_state()) != ckpt["fingerprint"]:
        raise RuntimeError(f"{run_dir}: fingerprint of the restored {which}.pt does not match")
    for p in model.parameters():
        p.requires_grad_(False)
    return model.eval(), cfg


def feature_fn(net, model: conv_adapt.ConvAdapted, half=None):
    """phi_run(x, grad): the frozen prefix of BRONet, then the run's adapted blocks.
    `half`: float16 (real, imag) copies of the prefix blocks, for gradients only."""
    from torch.utils.checkpoint import checkpoint

    from . import bronet

    first = model.tail.blocks[0]

    def prefix(x, grad: bool):
        h = net.conv_1(net.stem(x.sub(0.5)))
        for i in range(first):
            if not grad:
                h = bronet._bro_block(net, h, i)
            elif half is not None:
                h = checkpoint(bronet._bro_block_half, net, h, i, *half[i], use_reentrant=False)
            else:
                h = checkpoint(bronet._bro_block, net, h, i, use_reentrant=False)
        return h

    def fn(x, grad: bool):
        if grad:
            return model.features(prefix(x, True))
        with torch.no_grad():
            return model.features(prefix(x, False))

    return fn


def prefix_half_weights(net, first: int, device):
    with torch.no_grad():
        return [(W.real.to(torch.float16).to(device), W.imag.to(torch.float16).to(device))
                for W in net.conv.eval_weights[:first]]


# ---------------------------------------------------------------------------
# one engine folder per run, then the combined tables
# ---------------------------------------------------------------------------

def out_root(step: int, blocks: int, checkpoint: str, attack: str, kind: str) -> Path:
    return TREE.conv_attack_dir(step, blocks, checkpoint, attack, KINDS[kind])


def run_all(kind: str, cfg, *, step: int, blocks: int, modes, variants, seeds, device,
            strategy: str, cache_bro_weights: bool, batch_size: int, eval_chunk: int,
            super_chunk: int, restart: bool, logger) -> dict:
    """`kind` attack (cfg: core.AttackConfig) or mineps (cfg: MinEpsConfig) on every
    finished conv run; returns {"summary", "sanity"} of all runs together."""
    from . import attacks as battacks
    from . import bronet

    log = logger.info
    root = out_root(step, blocks, cfg.checkpoint, cfg.attack, kind)
    runs = discover(step, blocks, modes, variants, seeds)
    net, _ = bronet.load_bronet(device, strategy, logger, cache_bro_weights=cache_bro_weights)
    for p in net.parameters():
        p.requires_grad_(False)
    tail = conv_adapt.BroTail(net, net.conv.depth - blocks, device)
    base = bronet.load_mlp_base()
    half = (prefix_half_weights(net, tail.blocks[0], device)
            if cfg.grad_precision == "fp16" else None)
    images = battacks.load_test_images(cfg)
    all_rows, sanity_runs = [], {}
    for name, seed, d in runs:
        which = cfg.checkpoint
        if which == "headline":
            which = env.read_json(d / "metrics.json").get("headline", "final")
        m = env.read_json(d / layout.metrics_file(which))
        model, _ = load_conv_run(d, which, tail, base["weight"], base["bias"], device)
        target = {"name": name, "seed": seed, "dir": d, "which": which, "clf": model.clf,
                  "L_official": m["lipschitz"]["L_total_official"],
                  "L_rigorous": m["lipschitz"]["L_total_rigorous"]}
        od = root / name / layout.seed_name(seed)
        log(f"--- {name} seed {seed} ({which}) -> {od}")
        common = dict(device=device, strategy=strategy, batch_size=batch_size,
                      eval_chunk=eval_chunk, super_chunk=super_chunk, restart=restart,
                      logger=logger, feature_fn=feature_fn(net, model, half),
                      targets=[target], images=images, out=od)
        if kind == "attack":
            res = core.run(cfg, **common)
            csv_name = core.run_csv_name(cfg.attack, which)
        else:
            res = mineps_engine.search(cfg, **common)
            csv_name = layout.run_mineps_csv(cfg.attack, which)
        core._write_csv(d / "attacks" / f"conv_{csv_name}", res["rows"])
        all_rows += res["rows"]
        sanity_runs[f"{name}/seed{seed}"] = res["sanity"]
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    ok = all(s["ok"] for s in sanity_runs.values())
    if kind == "attack":
        viol = core.violations(all_rows)
        summary = core.summarise(all_rows)
    else:
        viol = mineps_engine.violations(all_rows)
        summary = mineps_engine.summarise(all_rows, cfg.eps_max)
    sanity = {"ok": ok, "violations": viol, "per_run": sanity_runs, "n_rows": len(all_rows),
              "adapted_blocks": tail.blocks, "grad_precision": cfg.grad_precision}
    write_json(root / "sanity_check.json", sanity)
    write_json(root / "config.json", {"kind": kind, "config": cfg.key(), "blocks": blocks,
                                      "runs": [f"{n}/seed{s}" for n, s, _ in runs],
                                      "device": env.device_info(device), "strategy": strategy})
    if kind == "attack":
        core.write_summary(root, cfg, summary, sanity)
    else:
        mineps_engine.write_summary(root, cfg, summary, sanity)
    log(f"{len(runs)} runs; violations official {viol['official']['count']}, rigorous "
        f"{viol['rigorous']['count']} -> {root}")
    return {"summary": summary, "sanity": sanity}


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    from repro import console

    console.init()
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("command", choices=tuple(KINDS))
    p.add_argument("--step", type=int, choices=(1, 2), required=True)
    p.add_argument("--blocks", type=int, default=1)
    p.add_argument("--checkpoint", choices=("headline", "best", "final"), default="final",
                   help="final (default: comparisons use the last epoch) | best | headline")
    p.add_argument("--mlp-modes", nargs="+", choices=adapters.MODES, default=list(adapters.MODES))
    p.add_argument("--variants", nargs="+", choices=("best", "stable", "manual"),
                   default=["best", "stable"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--n-images", type=int, default=200)
    p.add_argument("--random-trials", type=int, default=4, help="attack: random directions")
    mineps_engine.add_search_args(p)          # --attack and its settings, eps* search flags
    p.add_argument("--grad-precision", choices=("auto", "fp16", "fp32"), default="auto",
                   help="frozen BRO blocks in the gradients; auto: fp32 if the BRO matrices sit "
                        "on the GPU (or on the CPU), else fp16 copies")
    p.add_argument("--attack-batch", dest="batch_size", type=int, default=32)
    p.add_argument("--eval-chunk", type=int, default=256)
    p.add_argument("--super-chunk", type=int, default=256)
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
    if a.command == "attack":
        cfg = core.AttackConfig(step=a.step, checkpoint=a.checkpoint, n_images=a.n_images,
                                random_trials=a.random_trials, grad_precision=grad,
                                **core.attack_settings(a))
    else:
        cfg = mineps_engine.MinEpsConfig(step=a.step, checkpoint=a.checkpoint,
                                         n_images=a.n_images, grad_precision=grad,
                                         **mineps_engine.search_settings(a))
    root = out_root(a.step, a.blocks, a.checkpoint, cfg.attack, a.command)
    logger = env.setup_logging(root / f"{a.command}.log",
                               name=f"bronet_mlp.conv_attacks.{a.command}")
    logger.info(f"device {env.device_info(device)}  BRO weights: {strategy}, gradients {grad}")
    try:
        res = run_all(a.command, cfg, step=a.step, blocks=a.blocks, modes=a.mlp_modes,
                      variants=a.variants, seeds=a.seeds, device=device, strategy=strategy,
                      cache_bro_weights=a.cache_bro_weights, batch_size=a.batch_size,
                      eval_chunk=a.eval_chunk, super_chunk=a.super_chunk, restart=a.restart,
                      logger=logger)
    except torch.cuda.OutOfMemoryError as exc:
        raise RuntimeError("CUDA out of memory: lower --attack-batch and/or --eval-chunk "
                           "(progress is saved)") from exc
    if not res["sanity"]["ok"]:
        logger.error("CERTIFICATE VIOLATED by an attack -- see sanity_check.json")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
