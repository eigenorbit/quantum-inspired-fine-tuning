"""Every mode x variant x seed of one step, resumable; then the report.

    python -m experiments.resnet_adapt.run_grid --step 2 --seeds 0 1 2 --from-tuning

A finished run is skipped (one with other settings is an error, `--force`
retrains); an interrupted one resumes from its last.pt. The feature memory map
and the pretrained block are loaded once and shared by all runs.
"""

from __future__ import annotations

import argparse
import sys
import time

from . import paths

from experiments.common.io import setup_logging

from . import train


def main(argv=None) -> int:
    from repro import console

    console.init()
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--step", type=int, choices=(1, 2), required=True)
    p.add_argument("--modes", nargs="+", choices=train.MODES, default=list(train.MODES))
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--force", action="store_true", help="retrain finished runs")
    p.add_argument("--no-report", action="store_true")
    p.add_argument("--variants", nargs="+", choices=("best", "stable"),
                   default=["best", "stable"],
                   help="with --from-tuning; `stable` is skipped when it equals `best`")
    train.add_args(p)
    a = p.parse_args(argv)

    logger = setup_logging(paths.runs_dir(a.step, a.batch_size) / "run_grid.log",
                               name="resnet_adapt.run_grid")
    t0 = time.perf_counter()
    jobs = []
    for mode in a.modes:
        for variant in (a.variants if a.from_tuning else ["best"]):
            if variant == "stable":
                from .tune import stable_equals_best

                if stable_equals_best(a.step, mode, a.batch_size):
                    logger.info(f"{mode}: stable selection = best, no separate runs")
                    continue
            jobs += [(mode, variant, seed) for seed in a.seeds]
    inputs = None
    for mode, variant, seed in jobs:
        name = paths.run_name(mode, variant)
        rd = paths.run_dir(a.step, mode, seed, variant, a.batch_size)
        cfg = train.config_from_args(a, a.step, mode, seed, variant)
        if train.is_complete(rd) and not a.force:
            diff = train.completed_config_mismatch(rd, cfg)
            if diff:
                raise RuntimeError(f"{rd} is complete but was trained with other settings "
                                   f"{diff}; pass --force to retrain it")
            logger.info(f"skip {name} seed {seed}: complete (lr {cfg.lr:g})")
            continue
        if inputs is None:
            inputs = (train.final_inputs(a.step, a.feature_dtype, logger, a.ignore_mtime,
                                         a.batch_size)
                      if a.from_tuning
                      else train.prepare(a.step, a.feature_dtype, logger, a.ignore_mtime))
        t1 = time.perf_counter()
        logger.info(f"start {name} seed {seed}: lr {cfg.lr:g}, {cfg.epochs} epochs")
        train.fit(cfg, inputs, rd, restart=a.force,
                  logger=setup_logging(rd / "train.log", name=f"resnet_adapt.{name}.{seed}"))
        logger.info(f"done {name} seed {seed} in {time.perf_counter() - t1:.0f}s")
    logger.info(f"grid finished in {time.perf_counter() - t0:.0f}s")
    if not a.no_report:
        from . import report

        report.main(["--step", str(a.step), "--batch-size", str(a.batch_size)])
    return 0


if __name__ == "__main__":
    sys.exit(main())
