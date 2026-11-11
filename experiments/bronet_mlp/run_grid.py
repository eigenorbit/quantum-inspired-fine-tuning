"""Every mode x seed of one step, resumable; then the report.

    python -m experiments.bronet_mlp.run_grid --step 2 --seeds 0 1 2
    python -m experiments.bronet_mlp.run_grid --step 2 --modes butterfly lora --seeds 0
    python -m experiments.bronet_mlp.run_grid --step 2 --seeds 0 1 2 --from-tuning --force

A finished run (final.pt + metrics.json) is skipped; an interrupted one resumes
from its last.pt. ``--force`` retrains finished runs from scratch. Training
hyper-parameters are the same flags as ``train`` and apply to every run.
Features are loaded once and shared by all runs.
"""

from __future__ import annotations

import argparse
import sys
import time

from . import env
from . import adapters, train


def main(argv=None) -> int:
    from repro import console

    console.init()
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--step", type=int, choices=(1, 2), required=True)
    p.add_argument("--modes", nargs="+", choices=adapters.MODES, default=list(adapters.MODES))
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--force", action="store_true", help="retrain finished runs")
    p.add_argument("--no-report", action="store_true")
    p.add_argument("--variants", nargs="+", choices=("best", "stable"),
                   default=["best", "stable"],
                   help="with --from-tuning: which selections to train; `stable` is skipped "
                        "for a mode whose stable choice equals its best one")
    train.add_args(p)
    a = p.parse_args(argv)

    logger = env.setup_logging(env.runs_dir(a.step, a.batch_size) / "run_grid.log")
    t0 = time.perf_counter()
    inputs = None
    jobs = []
    for mode in a.modes:
        for variant in (a.variants if a.from_tuning else ["best"]):
            if variant == "stable":
                from .tune import stable_equals_best

                if stable_equals_best(a.step, mode, a.batch_size):
                    logger.info(f"{mode}: stable selection = best, no separate runs")
                    continue
                from .tune import selection_path

                sel = env.read_json(selection_path(a.step, a.batch_size))["modes"][mode]
                if sel["variants"]["stable"] is None:
                    logger.warning(f"{mode}: no stable grid point (every lr of its grid loses more "
                                   "than the stability threshold by the last epoch); no stable runs "
                                   "-- widen its lr grid downwards to get one")
                    continue
            jobs += [(mode, variant, seed) for seed in a.seeds]
    for mode, variant, seed in jobs:
        name = env.run_name(mode, variant)
        rd = env.run_dir(a.step, mode, seed, variant, a.batch_size)
        cfg = train.config_from_args(a, a.step, mode, seed, variant)
        if train.is_complete(rd) and not a.force:
            diff = train.completed_config_mismatch(rd, cfg)
            if diff:
                raise RuntimeError(f"{rd} is complete but was trained with other settings "
                                   f"{diff}; pass --force to retrain it")
            logger.info(f"skip {name} seed {seed}: complete (lr {cfg.lr:g}, {cfg.epochs} epochs)")
            continue
        if inputs is None:
            inputs = (train.final_inputs(a.step, logger, a.ignore_mtime) if a.from_tuning
                      else train.prepare(a.step, logger, a.ignore_mtime))
        run_logger = env.setup_logging(rd / "train.log", name=f"bronet_mlp.{name}.{seed}")
        t1 = time.perf_counter()
        logger.info(f"start {name} seed {seed}: lr {cfg.lr:g}, {cfg.epochs} epochs")
        train.fit(cfg, inputs, rd, logger=run_logger, restart=a.force)
        logger.info(f"done {name} seed {seed} in {time.perf_counter() - t1:.0f}s")
    logger.info(f"grid finished in {time.perf_counter() - t0:.0f}s")
    if not a.no_report:
        from . import report

        report.main(["--step", str(a.step), "--batch-size", str(a.batch_size)])
    return 0


if __name__ == "__main__":
    sys.exit(main())
