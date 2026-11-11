"""Selection of the learning rate (and, informatively, the epoch) per mode on the
validation split -- the SAME split as the BRONet experiment, never test.

    python -m experiments.resnet_adapt.tune     --step 2
    python -m experiments.resnet_adapt.run_grid --step 2 --seeds 0 1 2 --from-tuning

Protocol and rules are those of `experiments.bronet_mlp.tune` (imported, not
copied): 10 % of the deduplicated train, stratified, grouped by photograph;
every grid point trains up to the epoch ceiling (30) with seed 0; criterion =
highest validation accuracy over all (lr, epoch), ties weighted F1, then the
earlier epoch, then the smaller lr; collapsed epochs only if all collapsed; a
second selection `stable` over the runs that did not diverge (last epoch not
collapsed and within `STABLE_DROP` points of the run's best).

The split is recomputed from the manifest and must equal BRONet's
`<DERMA_REPRO_OUT>/bronet_mlp/tuning/step_<n>/val_split.json` (sha256), so the two
networks are compared image by image. Outputs under
<DERMA_REPRO_OUT>/resnet_adapt/tuning/step_<n>/: val_split.json (copy + source),
<mode>/lr<lr>/seed0/, selection.json, tuning_step<n>.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import paths

import torch

from experiments.bronet_mlp import tune as btune        # the tuned protocol: grids, rules
from experiments.common import data as dataset
from experiments.common.io import read_json, write_json, atomic_write_text, sha256_json, setup_logging
from experiments.common import layout

from . import train

# head-only and butterfly: the grids of bronet_mlp (the same kinds of parameters)
LR_GRID = {"head-only": btune.LR_GRID["head-only"],
           "butterfly": btune.LR_GRID["butterfly"],
           "lora": btune.LR_GRID["lora"],
           "full": (1e-5, 3e-5, 1e-4, 3e-4, 1e-3)}
TUNE_SEED = btune.TUNE_SEED
FIXED_KEYS = ("epochs", "batch_size", "weight_decay", "lr_step_size", "lr_gamma", "tau",
              "feature_dtype", "deterministic")


def split_path(step: int, batch_size: int = paths.DEFAULT_BATCH_SIZE) -> Path:
    return paths.tuning_dir(step, batch_size) / layout.VAL_SPLIT_FILE


def selection_path(step: int, batch_size: int = paths.DEFAULT_BATCH_SIZE) -> Path:
    return paths.tuning_dir(step, batch_size) / layout.SELECTION_FILE


def run_dir(step: int, mode: str, lr: float, seed: int = TUNE_SEED,
            batch_size: int = paths.DEFAULT_BATCH_SIZE) -> Path:
    return (paths.tuning_dir(step, batch_size) / mode / f"lr{lr:g}"
            / layout.seed_name(seed))


def make_split(inputs: train.RunInputs, step: int,
               batch_size: int = paths.DEFAULT_BATCH_SIZE) -> dict:
    """Fit/validation lists, identical to BRONet's saved split (else an error)."""
    rows = inputs.manifest["train"]
    fit_idx, val_idx = dataset.validation_split(rows, btune.VAL_FRACTION, btune.SPLIT_SEED)
    split = {"step": step, "fraction": btune.VAL_FRACTION, "seed": btune.SPLIT_SEED,
             "grouping": "photograph P<patient>_<image id>", "stratified": True,
             "train_manifest_sha256": inputs.manifest["train_sha256"],
             "fit": [rows[i][0] for i in fit_idx], "val": [rows[i][0] for i in val_idx]}
    split["sha256"] = sha256_json({"fit": split["fit"], "val": split["val"]})
    source = btune.split_path(step)
    if not source.exists():
        raise FileNotFoundError(f"{source} not found: the BRONet validation split is required "
                                "(the comparison is image by image)")
    saved = read_json(source)
    if saved["sha256"] != split["sha256"] \
            or saved["train_manifest_sha256"] != split["train_manifest_sha256"]:
        raise RuntimeError(f"{source} differs from the split computed now (data changed?)")
    split["source"] = str(source)
    own = split_path(step, batch_size)
    if own.exists():
        if read_json(own)["sha256"] != split["sha256"]:
            raise RuntimeError(f"{own} differs from BRONet's split; move it away")
    else:
        write_json(own, split)
    split["fit_idx"], split["val_idx"] = fit_idx, val_idx
    return split


def tuning_inputs(inputs: train.RunInputs, split: dict) -> train.RunInputs:
    """Train on the fit part, 'test' on the validation part."""
    fit, val = split["fit_idx"], split["val_idx"]
    rows = inputs.manifest["train"]
    man = {k: v for k, v in inputs.manifest.items() if k not in ("train", "test")}
    man.update(purpose="hyper-parameter tuning: fit / validation (both from train)",
               val_split_sha256=split["sha256"], n_train=len(fit), n_test=len(val),
               train=[rows[i] for i in fit], test=[rows[i] for i in val])
    return train.RunInputs(
        feats=inputs.feats, train_idx=inputs.train_idx[fit],
        train_y=inputs.train_y[torch.tensor(fit)],
        test_idx=inputs.train_idx[val], test_y=inputs.train_y[torch.tensor(val)],
        base_block=inputs.base_block,
        provenance={**inputs.provenance, "data_manifest_sha256": split["sha256"],
                    "val_split": split["source"]},
        test_rows=[{"image": Path(rows[i][0]).name, "patient": dataset.photo_key(rows[i][0])[:5],
                    "source_stem": Path(rows[i][0]).stem} for i in val],
        manifest=man)


def final_inputs(inputs: train.RunInputs, split: dict) -> train.RunInputs:
    """Final protocol: fit part / validation part (early stopping) / real test set."""
    fit, val = split["fit_idx"], split["val_idx"]
    rows = inputs.manifest["train"]
    man = dict(inputs.manifest)
    man.update(purpose="final: fit (train) / validation (early stopping) / test",
               val_split_sha256=split["sha256"], n_train=len(fit), n_val=len(val),
               train=[rows[i] for i in fit], val=[rows[i] for i in val])
    return train.RunInputs(
        feats=inputs.feats, train_idx=inputs.train_idx[fit],
        train_y=inputs.train_y[torch.tensor(fit)],
        test_idx=inputs.test_idx, test_y=inputs.test_y,
        val_idx=inputs.train_idx[val], val_y=inputs.train_y[torch.tensor(val)],
        base_block=inputs.base_block,
        provenance={**inputs.provenance,
                    "data_manifest_sha256": sha256_json(
                        {"manifest": inputs.provenance["data_manifest_sha256"],
                         "val_split": split["sha256"]}),
                    "val_split": split["source"]},
        test_rows=inputs.test_rows, manifest=man)


def epoch_candidates(history: list[dict], lr: float, majority_rate: float) -> list[dict]:
    return [{"lr": lr, "epochs": h["epoch"] + 1, "accuracy": h["test_accuracy"],
             "f1_weighted": h["test_f1_weighted"],
             "collapsed": h["test_accuracy"] <= majority_rate + 1e-6} for h in history]


def collect(step: int, modes, grids, max_epochs: int,
            batch_size: int = paths.DEFAULT_BATCH_SIZE) -> dict:
    out = {}
    for mode in modes:
        per_lr, pool, stable_pool = [], [], []
        for lr in grids[mode]:
            rd = run_dir(step, mode, lr, batch_size=batch_size)
            if not train.is_complete(rd):
                raise RuntimeError(f"tuning run missing: {rd}")
            m = read_json(rd / layout.HEADLINE_FILE)
            if m["config"]["epochs"] != max_epochs:
                raise RuntimeError(f"{rd} ran {m['config']['epochs']} epochs, the ceiling is "
                                   f"{max_epochs}; rerun tune with --force")
            cands = epoch_candidates(read_json(rd / "history.json"), lr,
                                     m["majority_class_rate"])
            pool += cands
            best = btune.select(cands)
            stable = (not m["collapsed"]) and m["accuracy"] >= best["accuracy"] - btune.STABLE_DROP
            per_lr.append({**best, "final_epoch_accuracy": m["accuracy"],
                           "final_epoch_collapsed": m["collapsed"], "stable_run": stable,
                           "seconds_total": m["seconds"]["total"], "run_dir": str(rd)})
            if stable:
                stable_pool += cands
        chosen = btune.select(pool)
        stable_choice = btune.select(stable_pool) if stable_pool else None

        def brief(c):
            return None if c is None else {k: c[k] for k in ("lr", "epochs", "accuracy",
                                                             "f1_weighted")}

        variants = {"best": brief(chosen), "stable": brief(stable_choice)}
        out[mode] = {"lr": chosen["lr"], "epochs": chosen["epochs"], "chosen": chosen,
                     "variants": variants,
                     "stable_equals_best": stable_choice is not None and
                     (stable_choice["lr"], stable_choice["epochs"])
                     == (chosen["lr"], chosen["epochs"]),
                     "all_collapsed": all(c["collapsed"] for c in pool),
                     "candidates": per_lr,
                     "lr_grid_edge": chosen["lr"] in (min(grids[mode]), max(grids[mode])),
                     "epoch_at_ceiling": chosen["epochs"] == max_epochs}
    return out


def write_selection(step: int, chosen: dict, fixed: dict, split: dict) -> dict:
    bs = fixed["batch_size"]
    path = selection_path(step, bs)
    modes = {}
    if path.exists():
        old = read_json(path)
        if old.get("fixed") == fixed and old.get("val_split_sha256") == split["sha256"] \
                and old.get("criterion") == btune.CRITERION \
                and old.get("stable_rule") == btune.STABLE_RULE:
            modes.update(old["modes"])
    modes.update(chosen)
    modes = {m: modes[m] for m in train.MODES if m in modes}
    sel = {"step": step, "criterion": btune.CRITERION, "stable_rule": btune.STABLE_RULE,
           "fixed": fixed, "seed": TUNE_SEED, "val_split_sha256": split["sha256"],
           "val_split_source": split["source"], "n_fit": len(split["fit"]),
           "n_val": len(split["val"]), "modes": modes}
    write_json(path, sel)
    lines = [f"# ResNet152 layer4.2 adaptation: selection of lr, step-{step}", "",
             f"Validation: {len(split['val']):,} of {len(split['fit']) + len(split['val']):,} "
             "train images — the same split as BRONet "
             f"(`{split['source']}`, sha256 {split['sha256'][:12]}). "
             f"Criterion: {btune.CRITERION}. Fixed: "
             + ", ".join(f"{k}={v}" for k, v in fixed.items())
             + " (epochs = ceiling).", "", f"Second selection `stable`: {btune.STABLE_RULE}.", "",
             "| mode | lr | best epoch | val acc, % | val F1, % | chosen | val acc @ last "
             "epoch, % | collapsed @ last | stable run | time, min |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for mode, s in modes.items():
        for c in s["candidates"]:
            tags = [name for name, ch in s["variants"].items()
                    if ch is not None and (c["lr"], c["epochs"]) == (ch["lr"], ch["epochs"])]
            mark = f"**{' = '.join(tags)}**" if tags else ""
            lines.append(f"| {mode} | {c['lr']:g} | {c['epochs']} | {c['accuracy']:.2f} | "
                         f"{c['f1_weighted']:.2f} | {mark} | {c['final_epoch_accuracy']:.2f} | "
                         f"{'yes' if c['final_epoch_collapsed'] else 'no'} | "
                         f"{'yes' if c['stable_run'] else 'no'} | "
                         f"{c['seconds_total'] / 60:.1f} |")
    warn = []
    for m, s in modes.items():
        reasons = [r for r, on in (("lr at the grid edge", s["lr_grid_edge"]),
                                   ("epoch at the ceiling", s["epoch_at_ceiling"]),
                                   ("every epoch collapsed", s["all_collapsed"])) if on]
        if reasons:
            warn.append(f"{m} ({', '.join(reasons)})")
    if warn:
        lines += ["", "Warning: " + "; ".join(warn) + " — the optimum may lie outside the "
                  "searched range."]
    atomic_write_text(paths.tuning_dir(step, bs) / layout.tuning_summary_file(step, bs),
                          "\n".join(lines) + "\n")
    return sel


def tuned_params(step: int, mode: str, cfg: train.RunConfig, variant: str = "best") -> dict:
    path = selection_path(step, cfg.batch_size)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run  python -m experiments.resnet_adapt.tune "
                                f"--step {step} --batch-size {cfg.batch_size}")
    sel = read_json(path)
    now = {k: getattr(cfg, k) for k in FIXED_KEYS if k != "epochs"}
    diff = {k: (sel["fixed"].get(k), v) for k, v in now.items() if sel["fixed"].get(k) != v}
    if diff:
        raise RuntimeError(f"selection {path} was made with other fixed settings: {diff}")
    if mode not in sel["modes"]:
        raise KeyError(f"mode {mode} was not tuned (selection has {list(sel['modes'])})")
    if variant not in btune.VARIANTS:
        raise ValueError(f"variant must be one of {btune.VARIANTS}")
    ch = sel["modes"][mode]["variants"][variant]
    if ch is None:
        raise RuntimeError(f"no stable grid point for {mode}; widen its lr grid")
    return {"lr": float(ch["lr"]), "epochs": int(ch["epochs"]),
            "ceiling": int(sel["fixed"]["epochs"])}


def stable_equals_best(step: int, mode: str, batch_size: int = paths.DEFAULT_BATCH_SIZE) -> bool:
    return bool(read_json(selection_path(step, batch_size))["modes"][mode]["stable_equals_best"])


def main(argv=None) -> int:
    from repro import console

    console.init()
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--step", type=int, choices=(1, 2), required=True)
    p.add_argument("--modes", nargs="+", choices=train.MODES, default=list(train.MODES))
    p.add_argument("--force", action="store_true", help="retrain finished grid points")
    train.add_args(p)
    a = p.parse_args(argv)
    if a.from_tuning:
        p.error("--from-tuning makes no sense for tune itself")

    logger = setup_logging(paths.tuning_dir(a.step, a.batch_size) / "tune.log", name="resnet_adapt.tune")
    inputs = train.prepare(a.step, a.feature_dtype, logger, a.ignore_mtime)
    split = make_split(inputs, a.step, a.batch_size)
    tin = tuning_inputs(inputs, split)
    logger.info(f"tuning step-{a.step}: fit {len(split['fit']):,} / val {len(split['val']):,} "
                f"(BRONet split {split['sha256'][:12]}); epoch ceiling {a.epochs}; grids "
                f"{{{', '.join(f'{m}: {LR_GRID[m]}' for m in a.modes)}}}")
    for mode in a.modes:
        for lr in LR_GRID[mode]:
            rd = run_dir(a.step, mode, lr, batch_size=a.batch_size)
            cfg = train.config_from_args(a, a.step, mode, TUNE_SEED)
            cfg.lr = lr
            if train.is_complete(rd) and not a.force:
                diff = train.completed_config_mismatch(rd, cfg)
                if diff:
                    raise RuntimeError(f"{rd} is complete but was trained with {diff}; "
                                       "pass --force to retrain the grid")
                logger.info(f"skip {mode} lr {lr:g}: complete")
                continue
            train.fit(cfg, tin, rd, restart=a.force,
                      logger=setup_logging(rd / "train.log",
                                               name=f"resnet_adapt.tune.{mode}.{lr:g}"))
    fixed = {k: getattr(train.config_from_args(a, a.step, a.modes[0], TUNE_SEED), k)
             for k in FIXED_KEYS}
    sel = write_selection(a.step, collect(a.step, a.modes, LR_GRID, a.epochs, a.batch_size), fixed, split)
    for mode, s in sel["modes"].items():
        st = s["variants"]["stable"]
        tail = ("stable = best" if s["stable_equals_best"] else "no stable grid point"
                if st is None else f"stable lr {st['lr']:g} (val acc {st['accuracy']:.2f})")
        logger.info(f"{mode}: best lr {s['lr']:g}, epoch {s['epochs']} "
                    f"(val acc {s['chosen']['accuracy']:.2f}); {tail}")
    logger.info(f"-> {selection_path(a.step, a.batch_size)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
