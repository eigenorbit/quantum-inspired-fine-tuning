"""Selection of (learning rate, number of epochs) per mode on a validation split of
train -- never on test.

    python -m experiments.bronet_mlp.tune --step 2            # grid, then selection
    python -m experiments.bronet_mlp.run_grid --step 2 --seeds 0 1 2 --from-tuning --force

Protocol:

* validation = 10 % of the deduplicated train, stratified by class and grouped
  by photograph (`data.validation_split`); the other 90 % is fitted;
* tuned, separately for every mode: the learning rate (own grid per mode --
  the modes train very different parameters) and the number of epochs, i.e.
  early stopping decided on validation: every grid point trains `--epochs`
  (the ceiling, 30) and its per-epoch validation accuracy is in history.json;
* shared and fixed for all modes: tau, eps_train, batch size, schedule, loss,
  the epoch ceiling -- they define the task (the accuracy / radius trade-off);
* one seed (0) per grid point; criterion: highest validation clean accuracy
  over all (lr, epoch) pairs; ties: higher weighted F1, then the earlier epoch,
  then the smaller lr. An epoch whose accuracy is at the majority-class rate
  (collapsed) is only chosen if every epoch of every lr of that mode collapsed;
* two selections per mode: `best` (the criterion) and `stable` (the criterion
  over the lr whose run did not diverge, see STABLE_RULE);
* final protocol (`run_grid --from-tuning`): only the lr is
  taken from here. Every seed trains on the SAME fit part up to the epoch
  ceiling, is scored on the validation part after every epoch and keeps its own
  best-validation epoch (`best.pt`); both that and the last epoch (`final.pt`)
  are evaluated once on test. The epoch recorded here per mode is informative
  (what seed 0 of the tuning grid chose), not used by the final runs.

Re-tuning at another loss (eps warm-up check): `--eps-train` / `--eps-warmup`
other than the protocol's write to tuning/step_<n><loss tag>/ (e.g.
step_1_eps195_wu0.33/), never over the protocol's selection; the val split is the
protocol's. With a warm-up only epochs after the ramp are candidates. `--lrs`
replaces the per-mode grid.

    python -m experiments.bronet_mlp.tune --step 1 --modes butterfly --eps-train 0.7647058823529411 --eps-warmup 0.33

Outputs under <DERMA_REPRO_OUT>/bronet_mlp/tuning/step_<n>/:
    val_split.json            the fit/validation file lists (+ sha256); reused and checked
    <mode>/lr<lr>/seed0/      an ordinary run directory (metrics are on validation)
    selection.json            chosen (lr, epochs) per mode, every candidate, fixed settings
    tuning_step<n>.md         the same as a table
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import env

import torch

from . import adapters
from . import data as bdata
from . import train
from experiments.common import layout

LR_GRID = {
    "head-only": (3e-4, 1e-3, 3e-3, 1e-2),
    "butterfly": (3e-4, 1e-3, 3e-3, 1e-2, 3e-2),
    "lora": (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3),
    "full": (1e-5, 3e-5, 1e-4, 3e-4),
}
VAL_FRACTION = 0.1
SPLIT_SEED = 0
TUNE_SEED = 0
# Settings shared by all modes; a selection is only valid for these values.
# `epochs` here is the ceiling of the tuning runs; the tuned value is per mode.
FIXED_KEYS = ("epochs", "batch_size", "weight_decay", "lr_step_size", "lr_gamma", "loss",
              "tau", "eps_train", "eps_warmup", "eps_start", "lora_rank", "lip_power_iters",
              "deterministic")
CRITERION = ("max validation clean accuracy over (lr, epoch); ties: weighted F1, then the "
             "earlier epoch, then the smaller lr; collapsed epochs only if all collapsed")
# Second selection, "stable": the same criterion restricted to the grid points
# whose run did not diverge -- at its LAST epoch it has not collapsed and its
# validation accuracy is within STABLE_DROP points of its own best epoch
# (2 points ~ 1.7 standard errors of accuracy on the 1.4k validation images).
STABLE_DROP = 2.0
STABLE_RULE = (f"same criterion over the lr whose run is stable: last epoch not collapsed "
               f"and within {STABLE_DROP:g} points of that run's best validation accuracy")
VARIANTS = ("best", "stable")


def split_path(step: int) -> Path:
    return env.tuning_dir(step) / "val_split.json"


def selection_path(step: int, batch_size: int = env.DEFAULT_BATCH_SIZE, tag: str = "") -> Path:
    return env.tuning_dir(step, batch_size, tag) / layout.SELECTION_FILE


def run_dir(step: int, mode: str, lr: float, seed: int = TUNE_SEED,
            batch_size: int = env.DEFAULT_BATCH_SIZE, tag: str = "") -> Path:
    return layout.BRONET.tune_run_dir(step, mode, lr, seed, batch_size, tag)


def config_tag(cfg: train.RunConfig) -> str:
    """Folder tag of the loss a config trains with ('' = the protocol's)."""
    return layout.loss_tag(cfg.eps_train, cfg.eps_warmup, cfg.eps_start)


def make_split(inputs: train.RunInputs, step: int, fraction: float = VAL_FRACTION,
               seed: int = SPLIT_SEED, save: bool = True) -> dict:
    """Fit/validation lists; the first call writes val_split.json, later calls
    must reproduce it exactly."""
    rows = inputs.manifest["train"]
    fit_idx, val_idx = bdata.validation_split(rows, fraction, seed)
    split = {"step": step, "fraction": fraction, "seed": seed,
             "grouping": "photograph P<patient>_<image id>", "stratified": True,
             "train_manifest_sha256": inputs.manifest["train_sha256"],
             "fit": [rows[i][0] for i in fit_idx], "val": [rows[i][0] for i in val_idx]}
    split["sha256"] = env.sha256_json({"fit": split["fit"], "val": split["val"]})
    path = split_path(step)
    if not path.exists() and not save:
        raise FileNotFoundError(f"{path} not found; run tune first")
    if path.exists():
        saved = env.read_json(path)
        if saved["sha256"] != split["sha256"]:
            raise RuntimeError(f"{path} differs from the split computed now "
                               "(data changed?); move it away to redo tuning")
    elif save:
        env.write_json(path, split)
    split["fit_idx"], split["val_idx"] = fit_idx, val_idx
    return split


def tuning_inputs(inputs: train.RunInputs, split: dict) -> train.RunInputs:
    """Train on the fit part, 'test' on the validation part."""
    fit = torch.tensor(split["fit_idx"])
    val = torch.tensor(split["val_idx"])
    rows = inputs.manifest["train"]
    man = {k: v for k, v in inputs.manifest.items() if k not in ("train", "test")}
    man.update(purpose="hyper-parameter tuning: fit / validation (both from train)",
               val_split_sha256=split["sha256"], n_train=len(fit), n_test=len(val),
               train=[rows[i] for i in split["fit_idx"]],
               test=[rows[i] for i in split["val_idx"]])
    prov = {**inputs.provenance, "data_manifest_sha256": split["sha256"],
            "val_split": str(split_path(split["step"]))}
    return train.RunInputs(
        train_x=inputs.train_x[fit], train_y=inputs.train_y[fit],
        test_x=inputs.train_x[val], test_y=inputs.train_y[val],
        base_weight=inputs.base_weight, base_bias=inputs.base_bias,
        L_pre_official=inputs.L_pre_official, L_pre_rigorous=inputs.L_pre_rigorous,
        provenance=prov,
        test_rows=[{"image": Path(rows[i][0]).name, "patient": bdata.photo_key(rows[i][0])[:5],
                    "source_stem": Path(rows[i][0]).stem} for i in split["val_idx"]],
        L_mlp0=inputs.L_mlp0, manifest=man)


def final_inputs(inputs: train.RunInputs, split: dict) -> train.RunInputs:
    """Final protocol: train on the fit part, early-stop on the validation part,
    evaluate on the real test set."""
    fit = torch.tensor(split["fit_idx"])
    val = torch.tensor(split["val_idx"])
    rows = inputs.manifest["train"]
    man = dict(inputs.manifest)
    man.update(purpose="final: fit (train) / validation (early stopping) / test",
               val_split_sha256=split["sha256"], n_train=len(fit), n_val=len(val),
               train=[rows[i] for i in split["fit_idx"]],
               val=[rows[i] for i in split["val_idx"]])
    prov = {**inputs.provenance,
            "data_manifest_sha256": env.sha256_json(
                {"manifest": inputs.provenance["data_manifest_sha256"],
                 "val_split": split["sha256"]}),
            "val_split": str(split_path(split["step"]))}
    return train.RunInputs(
        train_x=inputs.train_x[fit], train_y=inputs.train_y[fit],
        test_x=inputs.test_x, test_y=inputs.test_y,
        val_x=inputs.train_x[val], val_y=inputs.train_y[val],
        base_weight=inputs.base_weight, base_bias=inputs.base_bias,
        L_pre_official=inputs.L_pre_official, L_pre_rigorous=inputs.L_pre_rigorous,
        provenance=prov, test_rows=inputs.test_rows, L_mlp0=inputs.L_mlp0, manifest=man)


def _key(c: dict):
    return (-c["accuracy"], -c["f1_weighted"], c["epochs"], c["lr"])


def select(candidates: list[dict]) -> dict:
    """The chosen candidate under CRITERION. Each: lr, epochs, accuracy, f1_weighted, collapsed."""
    pool = [c for c in candidates if not c["collapsed"]] or candidates
    return sorted(pool, key=_key)[0]


def epoch_candidates(history: list[dict], lr: float, majority_rate: float) -> list[dict]:
    """One candidate per epoch of one tuning run (validation metrics in history.json);
    epochs still inside an eps warm-up are not candidates."""
    return [{"lr": lr, "epochs": h["epoch"] + 1, "accuracy": h["test_accuracy"],
             "f1_weighted": h["test_f1_weighted"],
             "collapsed": h["test_accuracy"] <= majority_rate + 1e-6,
             "L_mlp": h["L_mlp_end_power"],
             "certified_accuracy_36": h["test_certified_accuracy_36"]} for h in history
            if h.get("eps_full", True)]


def collect(step: int, modes, grids, max_epochs: int,
            batch_size: int = env.DEFAULT_BATCH_SIZE, tag: str = "") -> dict:
    out = {}
    for mode in modes:
        per_lr, pool, stable_pool = [], [], []
        for lr in grids[mode]:
            rd = run_dir(step, mode, lr, batch_size=batch_size, tag=tag)
            if not train.is_complete(rd):
                raise RuntimeError(f"tuning run missing: {rd}")
            m = env.read_json(rd / "metrics.json")
            if m["config"]["epochs"] != max_epochs:
                raise RuntimeError(f"{rd} ran {m['config']['epochs']} epochs, the ceiling is "
                                   f"{max_epochs}; rerun tune with --force")
            cands = epoch_candidates(env.read_json(rd / "history.json"), lr,
                                     m["majority_class_rate"])
            pool += cands
            best = select(cands)
            stable = (not m["collapsed"]) and m["accuracy"] >= best["accuracy"] - STABLE_DROP
            per_lr.append({**best, "final_epoch_accuracy": m["accuracy"],
                           "final_epoch_collapsed": m["collapsed"],
                           "final_epoch_L_over_L0": m["lipschitz"]["L_over_L0"],
                           "stable_run": stable, "run_dir": str(rd)})
            if stable:
                stable_pool += cands
        chosen = select(pool)
        stable_choice = select(stable_pool) if stable_pool else None
        variants = {"best": {"lr": chosen["lr"], "epochs": chosen["epochs"],
                             "accuracy": chosen["accuracy"],
                             "f1_weighted": chosen["f1_weighted"]},
                    "stable": (None if stable_choice is None else
                               {"lr": stable_choice["lr"], "epochs": stable_choice["epochs"],
                                "accuracy": stable_choice["accuracy"],
                                "f1_weighted": stable_choice["f1_weighted"]})}
        out[mode] = {"lr": chosen["lr"], "epochs": chosen["epochs"], "chosen": chosen,
                     "variants": variants,
                     "stable_equals_best": variants["stable"] is not None and
                     (variants["stable"]["lr"], variants["stable"]["epochs"])
                     == (chosen["lr"], chosen["epochs"]),
                     "all_collapsed": all(c["collapsed"] for c in pool),
                     "candidates": per_lr,
                     "lr_grid_edge": chosen["lr"] in (min(grids[mode]), max(grids[mode])),
                     "epoch_at_ceiling": chosen["epochs"] == max_epochs}
    return out


def write_selection(step: int, chosen: dict, fixed: dict, split: dict, tag: str = "") -> dict:
    """Write selection.json and the table. Modes tuned earlier with the same
    fixed settings and split are kept (a --modes subset only replaces its own)."""
    bs = fixed["batch_size"]
    path = selection_path(step, bs, tag)
    modes = {}
    if path.exists():
        old = env.read_json(path)
        if old.get("fixed") == fixed and old.get("val_split_sha256") == split["sha256"] \
                and old.get("criterion") == CRITERION and old.get("stable_rule") == STABLE_RULE:
            modes.update(old["modes"])
    modes.update(chosen)
    modes = {m: modes[m] for m in adapters.MODES if m in modes}
    sel = {"step": step, "criterion": CRITERION, "stable_rule": STABLE_RULE, "fixed": fixed,
           "seed": TUNE_SEED,
           "val_split_sha256": split["sha256"], "n_fit": len(split["fit"]),
           "n_val": len(split["val"]), "modes": modes}
    env.write_json(path, sel)
    lines = [f"# Selection of lr and epochs, step-{step}", "",
             f"Validation: {len(split['val']):,} of {len(split['fit']) + len(split['val']):,} "
             "train images (10 %, stratified, grouped by photograph). "
             f"Criterion: {CRITERION}. Fixed for all modes: "
             + ", ".join(f"{k}={v}" for k, v in fixed.items())
             + " (epochs = ceiling of the tuning runs).", "",
             f"Second selection `stable`: {STABLE_RULE}.", "",
             "One row per lr: its best epoch on validation; the last columns describe the "
             "last epoch of the same run and whether the run counts as stable.", "",
             "| mode | lr | best epoch | val acc, % | val F1, % | L_mlp | cert@36/255, % "
             "| chosen | val acc @ last epoch, % | collapsed @ last | stable run |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for mode, s in modes.items():
        v = s.get("variants", {"best": {"lr": s["lr"], "epochs": s["epochs"]}, "stable": None})
        for c in s["candidates"]:
            tags = [name for name, ch in v.items()
                    if ch is not None and (c["lr"], c["epochs"]) == (ch["lr"], ch["epochs"])]
            mark = f"**{' = '.join(tags)}**" if tags else ""
            lines.append(f"| {mode} | {c['lr']:g} | {c['epochs']} | {c['accuracy']:.2f} | "
                         f"{c['f1_weighted']:.2f} | {c['L_mlp']:.4g} | "
                         f"{100 * c['certified_accuracy_36']:.1f} | {mark} | "
                         f"{c['final_epoch_accuracy']:.2f} | "
                         f"{'yes' if c['final_epoch_collapsed'] else 'no'} | "
                         f"{'yes' if c.get('stable_run') else 'no'} |")
    warn = []
    for m, s in modes.items():
        reasons = [r for r, on in (("lr at the grid edge", s["lr_grid_edge"]),
                                   ("epoch at the ceiling", s["epoch_at_ceiling"]),
                                   ("every epoch collapsed", s["all_collapsed"])) if on]
        if reasons:
            warn.append(f"{m} ({', '.join(reasons)})")
    if warn:
        lines += ["", "Warning: " + "; ".join(warn) + " -- the optimum may lie outside the "
                  "searched range."]
    lines += ["", "L_mlp is the float32 power-iteration estimate logged during training "
              "(L_mlp⁰ ≈ 1); the final runs report the exact value."]
    env.atomic_write_text(env.tuning_dir(step, bs, tag) / layout.tuning_summary_file(step, bs),
                          "\n".join(lines) + "\n")
    return sel


def tuned_params(step: int, mode: str, cfg: train.RunConfig, variant: str = "best") -> dict:
    """{"lr", "epochs"} selected for `mode` (`variant` best or stable); errors if
    the fixed settings differ (the epoch ceiling excepted: it is what gets replaced)."""
    path = selection_path(step, cfg.batch_size)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run  python -m experiments.bronet_mlp.tune "
                                f"--step {step} --batch-size {cfg.batch_size}")
    sel = env.read_json(path)
    now = {k: getattr(cfg, k) for k in FIXED_KEYS if k != "epochs"}
    # keys added to FIXED_KEYS later are absent from older selections: they had
    # the RunConfig default
    saved = {**{k: getattr(train.RunConfig(step=step, mode=mode), k) for k in now},
             **sel["fixed"]}
    diff = {k: (saved.get(k), v) for k, v in now.items() if saved.get(k) != v}
    if diff:
        raise RuntimeError(f"selection {path} was made with other fixed settings: {diff}")
    if mode not in sel["modes"]:
        raise KeyError(f"mode {mode} was not tuned (selection has {list(sel['modes'])})")
    s = sel["modes"][mode]
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")
    if "variants" not in s:
        raise RuntimeError(f"{path} predates the best/stable selection; rerun tune (the grid "
                           "is reused, only the selection is recomputed)")
    ch = s["variants"][variant]
    if ch is None:
        raise RuntimeError(f"no stable grid point for {mode}; widen its lr grid")
    return {"lr": float(ch["lr"]), "epochs": int(ch["epochs"]),
            "ceiling": int(sel["fixed"]["epochs"])}


def stable_equals_best(step: int, mode: str, batch_size: int = env.DEFAULT_BATCH_SIZE) -> bool:
    s = env.read_json(selection_path(step, batch_size))["modes"].get(mode)
    if s is None or "variants" not in s:
        raise RuntimeError(f"{selection_path(step, batch_size)} has no best/stable selection for {mode}; "
                           "rerun tune")
    return bool(s["stable_equals_best"])


def main(argv=None) -> int:
    from repro import console

    console.init()
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--step", type=int, choices=(1, 2), required=True)
    p.add_argument("--modes", nargs="+", choices=adapters.MODES, default=list(adapters.MODES))
    p.add_argument("--force", action="store_true", help="retrain finished grid points")
    p.add_argument("--lrs", nargs="+", type=float, default=None,
                   help="lr grid for every mode in --modes, instead of LR_GRID")
    train.add_args(p)
    a = p.parse_args(argv)
    if getattr(a, "from_tuning", False):
        p.error("--from-tuning makes no sense for tune itself")

    tag = config_tag(train.config_from_args(a, a.step, a.modes[0], TUNE_SEED))
    grids = {m: tuple(a.lrs) if a.lrs else LR_GRID[m] for m in a.modes}
    logger = env.setup_logging(env.tuning_dir(a.step, a.batch_size, tag) / "tune.log")
    inputs = train.prepare(a.step, logger, a.ignore_mtime)
    split = make_split(inputs, a.step)
    tin = tuning_inputs(inputs, split)
    logger.info(f"tuning step-{a.step}{tag}: fit {len(split['fit']):,} / val "
                f"{len(split['val']):,}; epoch ceiling {a.epochs}; eps_train "
                f"{255 * a.eps_train:g}/255, warm-up {a.eps_warmup:g}; grids "
                f"{{{', '.join(f'{m}: {grids[m]}' for m in a.modes)}}}")
    for mode in a.modes:
        for lr in grids[mode]:
            rd = run_dir(a.step, mode, lr, batch_size=a.batch_size, tag=tag)
            cfg = train.config_from_args(a, a.step, mode, TUNE_SEED)
            cfg.lr = lr
            if train.is_complete(rd) and not a.force:
                diff = train.completed_config_mismatch(rd, cfg)
                if diff:
                    raise RuntimeError(f"{rd} is complete but was trained with {diff}; "
                                       "pass --force to retrain the grid")
                logger.info(f"skip {mode} lr {lr:g}: complete")
                continue
            train.fit(cfg, tin, rd, logger=env.setup_logging(rd / "train.log",
                                                           name=f"bronet_mlp.tune.{mode}.{lr:g}"),
                      restart=a.force)
    fixed = {k: getattr(train.config_from_args(a, a.step, a.modes[0], TUNE_SEED), k)
             for k in FIXED_KEYS}
    sel = write_selection(a.step, collect(a.step, a.modes, grids, a.epochs, a.batch_size, tag),
                          fixed, split, tag)
    for mode, s in sel["modes"].items():
        st = s["variants"]["stable"]
        if s["stable_equals_best"]:
            tail = "stable = best"
        elif st is None:
            tail = "no stable grid point"
        else:
            tail = f"stable lr {st['lr']:g}, {st['epochs']} epochs (val acc {st['accuracy']:.2f})"
        logger.info(f"{mode}: best lr {s['lr']:g}, {s['epochs']} epochs "
                    f"(val acc {s['chosen']['accuracy']:.2f}); {tail}")
    logger.info(f"-> {selection_path(a.step, a.batch_size, tag)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
