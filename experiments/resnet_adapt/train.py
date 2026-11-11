"""Train one (step, mode, seed) on the cached input of layer4.2, then evaluate.

    python -m experiments.resnet_adapt.train --step 2 --seed 0 --from-tuning

ResNet152 frozen up to layer4.1; layer4.2 adapted in one of `MODES`, plus a new
LLN head (always trained):

* ``full``  all of layer4.2 trains (convolutions, BN affine; BN statistics
  stay ImageNet's);
* ``butterfly``   BN folded into the convolutions, W' = R(theta) W with R a
  butterfly of Givens rotations per convolution (512, 512, 2048 channels;
  15 872 angles): singular values and operator norms kept exactly;
* ``lora``        BN folded, W' = W + B A per convolution, rank 4, B = 0 at start
  (40 960 parameters): the spectrum drifts;
* ``head-only``   BN-folded layer4.2 frozen, only the head trains.

The same protocol as `experiments.bronet_mlp.train`:

* Adam, batch 256, StepLR(2, 0.7), 30 epochs, deterministic, shuffling by
  `randperm` with generator `seed * 1 000 003 + epoch`;
* loss CE(tau * z), tau = 4 -- i.e. `experiments.common.losses.cert_ce` with the
  certificate term L * eps set to 0 (ResNet's global Lipschitz bound is
  astronomical; with it the loss would saturate);
* ``early_stop="val"`` (``--from-tuning``): train on the fit part, keep the best
  validation epoch as ``best.pt``; ``best.pt`` and ``final.pt`` are both
  evaluated on test; ``metrics.json`` = best. Test never picks the epoch.

Checkpoints hold only the trainable tensors, optimizer, scheduler, RNG state,
config and the identities of weights / features / data; atomic writes; a resume
with any of those changed is refused (``--restart`` discards). Outputs:
<DERMA_REPRO_OUT>/resnet_adapt/runs/step_<n>/<mode>[_stable]/seed<s>/.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from . import paths

import numpy as np
import torch

from experiments.bronet_mlp import adapters as badapters
from experiments.bronet_mlp import env                       # git_info, env_snapshot (provenance)
from experiments.bronet_mlp import train as btrain
from experiments.common import layout, losses
from experiments.common.device import resolve_device, device_info
from experiments.common.io import read_json, write_json, atomic_torch_save, setup_logging
from repro import metrics as rmetrics
from repro import seeding
from repro.config import CLASS_NAMES, N_CLASSES

from . import backbone, features

CHECKPOINT_FORMAT = 1
MODES = ("head-only", "butterfly", "lora", "full")
# Old mode name still found in early checkpoints: `last-block` is now `full`, the
# counterpart of bronet_mlp's `full`.
MODE_ALIASES = {"last-block": "full"}
PROVENANCE_KEYS = ("weights_sha256", "features_id", "data_manifest_sha256")
LOSS_TEXT = "CE(tau * z) = bronet_mlp.losses.cert_ce with L * eps = 0"


@dataclass
class RunConfig:
    step: int
    mode: str = "full"
    seed: int = 0
    epochs: int = 30
    batch_size: int = paths.DEFAULT_BATCH_SIZE
    lr: float = 1e-3
    weight_decay: float = 0.0
    lr_step_size: int = 2
    lr_gamma: float = 0.7
    tau: float = 4.0
    feature_dtype: str = "float16"
    deterministic: bool = True
    early_stop: str = "none"      # "none" | "val"
    # --- runtime only: never part of the resume-compatibility check ---
    device: str = "auto"
    keep_every: int = 0

    RUNTIME = ("device", "keep_every")

    def __post_init__(self):
        self.mode = MODE_ALIASES.get(self.mode, self.mode)
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if self.early_stop not in ("none", "val"):
            raise ValueError("early_stop must be 'none' or 'val'")
        if self.feature_dtype not in features.DTYPES:
            raise ValueError(f"feature_dtype must be one of {tuple(features.DTYPES)}")

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def compat(self) -> dict:
        return {k: v for k, v in self.to_dict().items() if k not in self.RUNTIME}

    @classmethod
    def saved_compat(cls, saved: dict) -> dict:
        defaults = {f.name: f.default for f in dataclasses.fields(cls)
                    if f.default is not dataclasses.MISSING}
        merged = {**defaults, **saved}
        merged["mode"] = MODE_ALIASES.get(merged["mode"], merged["mode"])
        return {k: v for k, v in merged.items() if k not in cls.RUNTIME}


@dataclass
class RunInputs:
    """Everything a run consumes: the cached array (memory map or in-memory) and
    row indices into it. Built by `prepare`, or synthetically in the tests."""

    feats: np.ndarray              # (N, 2048, 7, 7)
    train_idx: np.ndarray
    train_y: torch.Tensor
    test_idx: np.ndarray
    test_y: torch.Tensor
    base_block: torch.nn.Module    # pretrained layer4.2
    provenance: dict               # weights_sha256, features_id, data_manifest_sha256, ...
    test_rows: list[dict]          # image, patient, source_stem per test image
    manifest: dict | None = None
    val_idx: np.ndarray | None = None
    val_y: torch.Tensor | None = None


def prepare(step: int, dtype: str = "float16", logger=None, ignore_mtime: bool = False,
            net=None) -> RunInputs:
    from experiments.common import data as dataset

    train, test, stats = dataset.records(step, logger)
    man = dataset.manifest(step, train, test, stats)
    net = backbone.load_resnet152() if net is None else net
    wsha = backbone.state_sha256(net)
    store = features.open_store(step, dtype, wsha, ignore_mtime)
    return RunInputs(
        feats=store.array,
        train_idx=store.rows_of([r[0] for r in man["train"]]),
        train_y=torch.tensor([r.label for r in train]),
        test_idx=store.rows_of([r[0] for r in man["test"]]),
        test_y=torch.tensor([r.label for r in test]),
        base_block=net.layer4[2],
        provenance={"weights": backbone.WEIGHTS, "weights_sha256": wsha,
                    "features_id": store.feature_id,
                    "features_dir": str(features.cache_dir(step, dtype)),
                    "data_manifest_sha256": dataset.manifest_sha256(man)},
        test_rows=[{"image": r.path.name, "patient": r.patient, "source_stem": r.source_stem}
                   for r in test],
        manifest=man)


# ---------------------------------------------------------------------------
# checkpoints
# ---------------------------------------------------------------------------

def _checkpoint(model, opt, sched, epoch, global_step, cfg, inputs, seconds) -> dict:
    state = model.trainable_state()
    return {"format": CHECKPOINT_FORMAT, "trainable_state": state,
            "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
            "epoch": epoch, "global_step": global_step, "rng": seeding.capture(),
            "config": cfg.to_dict(), "seconds": seconds,
            **{k: inputs.provenance.get(k, "") for k in PROVENANCE_KEYS},
            "fingerprint": badapters.fingerprint(state)}


def check_compatible(ckpt: dict, cfg: RunConfig, inputs: RunInputs, path: Path) -> None:
    saved = RunConfig.saved_compat(ckpt["config"])
    problems = [f"config.{k}: checkpoint {saved.get(k)!r} vs now {v!r}"
                for k, v in cfg.compat().items() if saved.get(k) != v]
    problems += [f"config.{k}: only in checkpoint" for k in saved if k not in cfg.compat()]
    for key in PROVENANCE_KEYS:
        if ckpt.get(key, "") != inputs.provenance.get(key, ""):
            problems.append(f"{key}: checkpoint {str(ckpt.get(key))[:16]} vs now "
                            f"{str(inputs.provenance.get(key))[:16]}")
    if problems:
        raise RuntimeError(f"{path} is not compatible with this run:\n  "
                           + "\n  ".join(problems)
                           + "\nPass --restart to discard it and train from scratch.")


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict(model, feats: np.ndarray, rows: np.ndarray, device, batch: int = 256) -> torch.Tensor:
    model.eval()
    return torch.cat([model(features.gather(feats, rows[i:i + batch], device)).cpu()
                      for i in range(0, len(rows), batch)])


def _overall(logits: torch.Tensor, y: torch.Tensor) -> dict:
    q = rmetrics.overall(rmetrics.Predictions(
        y_true=y.numpy(), y_pred=logits.argmax(1).numpy(),
        probabilities=np.zeros((len(y), logits.shape[1]))))
    return {k: float(v) for k, v in q.items()}   # plain types: goes into last.pt


def evaluate(model, cfg: RunConfig, inputs: RunInputs, device) -> dict:
    t0 = time.perf_counter()
    y = inputs.test_y
    logits = predict(model, inputs.feats, inputs.test_idx, device)
    quality = btrain._quality(logits, y, cfg.step, cfg.tau)
    counts = torch.bincount(y, minlength=N_CLASSES[cfg.step]).double()
    majority = float(counts.max() / counts.sum())
    acc = quality["overall"]["accuracy"]
    metrics = {"quality": quality, "accuracy": acc,
               "f1_weighted": quality["overall"]["f1_weighted"],
               "majority_class_rate": 100 * majority,
               "collapsed": acc <= 100 * majority + 1e-6,
               "spectrum": backbone.block_spectrum(model.block, inputs.base_block)}
    if cfg.mode == "butterfly":
        metrics["orthogonality_error_max"] = max(r.orthogonality_error()
                                                 for r in model.block.rotations)
    metrics["seconds_evaluate"] = time.perf_counter() - t0
    arrays = {"logits": logits.numpy(), "y": y.numpy(), "pred": logits.argmax(1).numpy(),
              "head_weight": model.head.get_weight().detach().cpu().numpy()}
    return {"metrics": metrics, "arrays": arrays}


def write_predictions(path: Path, rows: list[dict], arrays: dict, step: int, tau: float) -> None:
    """Same columns as bronet_mlp's predictions_*.csv, without the certificate."""
    names = CLASS_NAMES[step]
    logits = arrays["logits"]
    probs = torch.softmax(tau * torch.from_numpy(logits).double(), 1).numpy()
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as handle:
        w = csv.writer(handle)
        w.writerow(["image", "patient", "source_stem", "true_label", "true_class",
                    "predicted_label", "predicted_class", "correct"]
                   + [f"prob_{n}" for n in names] + [f"logit_{n}" for n in names])
        for i, row in enumerate(rows):
            t, p = int(arrays["y"][i]), int(arrays["pred"][i])
            w.writerow([row["image"], row["patient"], row["source_stem"], t, names[t], p,
                        names[p], int(t == p)] + [f"{v:.6f}" for v in probs[i]]
                       + [f"{v:.8f}" for v in logits[i]])
    tmp.replace(path)


CHECKPOINTS = ("final", "best")


def write_outputs(run_dir: Path, ev: dict, cfg: RunConfig, inputs: RunInputs, extra: dict,
                  which: str) -> dict:
    metrics = {**extra, "checkpoint": which, **ev["metrics"]}
    write_json(run_dir / layout.metrics_file(which), metrics)
    write_predictions(run_dir / layout.predictions_file(which), inputs.test_rows, ev["arrays"],
                      cfg.step, cfg.tau)
    final = run_dir / layout.arrays_file("outputs", which)
    tmp = final.with_name(final.stem + ".tmp.npz")
    np.savez_compressed(tmp, **ev["arrays"])
    tmp.replace(final)
    return metrics


def is_complete(run_dir: Path) -> bool:
    return (run_dir / "checkpoints" / "final.pt").exists() and (run_dir / "metrics.json").exists()


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------

def fit(cfg: RunConfig, inputs: RunInputs, run_dir: Path, *, logger=None, restart=False,
        stop_after: int | None = None) -> dict | None:
    """Train, checkpoint every epoch, evaluate. Returns the headline metrics, or
    None when stopped by `stop_after` (a simulated interruption)."""
    log = logger.info if logger else (lambda *_: None)
    run_dir = Path(run_dir)
    ckdir = run_dir / "checkpoints"
    device = resolve_device(cfg.device)
    if restart and run_dir.exists():
        names = ["checkpoints", "history.json", "metrics.json"] + [
            f"{kind}_{w}.{ext}" for w in CHECKPOINTS
            for kind, ext in (("metrics", "json"), ("predictions", "csv"), ("outputs", "npz"))]
        for name in names:
            target = run_dir / name
            if target.is_dir():
                shutil.rmtree(target)
            elif target.exists():
                target.unlink()
    run_dir.mkdir(parents=True, exist_ok=True)

    seeding.configure_determinism(cfg.deterministic)
    seeding.set_seed(cfg.seed)
    num_classes = N_CLASSES[cfg.step]
    model = backbone.Classifier(inputs.base_block, num_classes, cfg.seed, cfg.mode).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=cfg.lr_step_size, gamma=cfg.lr_gamma)
    history, start_epoch, global_step, best_info = [], 0, 0, None

    last = ckdir / "last.pt"
    if last.exists():
        ckpt = torch.load(last, map_location="cpu", weights_only=True)
        check_compatible(ckpt, cfg, inputs, last)
        model.load_trainable_state(ckpt["trainable_state"])
        opt.load_state_dict(ckpt["optimizer"])
        sched.load_state_dict(ckpt["scheduler"])
        seeding.restore(ckpt["rng"])
        start_epoch, global_step = ckpt["epoch"] + 1, ckpt["global_step"]
        best_info = ckpt.get("best")
        if (run_dir / "history.json").exists():
            history = [h for h in read_json(run_dir / "history.json")
                       if h["epoch"] < start_epoch]
        log(f"resumed from {last} at epoch {start_epoch}")

    write_json(run_dir / "config.json", {
        "config": cfg.to_dict(), "provenance": inputs.provenance, "loss": LOSS_TEXT,
        "trainable_block": backbone.TRAINABLE_BLOCK,
        "trainable_params": model.num_trainable(), "num_classes": num_classes,
        "n_train": len(inputs.train_idx), "n_test": len(inputs.test_idx),
        "n_val": 0 if inputs.val_idx is None else len(inputs.val_idx),
        "env": env.env_snapshot(), "device": device_info(device),
        "git": env.git_info(), "command": sys.argv})
    if inputs.manifest is not None:
        write_json(run_dir / "data_manifest.json", inputs.manifest)

    n = len(inputs.train_idx)
    early = cfg.early_stop == "val"
    if early:
        if inputs.val_idx is None:
            raise ValueError("early_stop='val' needs RunInputs.val_idx / val_y")
        counts = torch.bincount(inputs.val_y, minlength=num_classes).double()
        val_majority = 100 * float(counts.max() / counts.sum())
    log(f"step-{cfg.step} {cfg.mode} seed {cfg.seed}: {model.num_trainable():,} trainable, "
        f"train {n:,}" + (f" / val {len(inputs.val_idx):,}" if early else "")
        + f" / test {len(inputs.test_idx):,}, lr {cfg.lr:g}, device {device}, "
        f"features {cfg.feature_dtype}, early stop {cfg.early_stop}")

    def seconds_so_far():
        return {"train": sum(h["seconds_train"] for h in history),
                "epochs_with_eval": sum(h["seconds_epoch"] for h in history)}

    for epoch in range(start_epoch, cfg.epochs):
        t0 = time.perf_counter()
        model.train()
        g = torch.Generator().manual_seed(cfg.seed * 1_000_003 + epoch)
        perm = torch.randperm(n, generator=g).numpy()
        tot_loss, tot_correct, t_io = 0.0, 0, 0.0
        for i in range(0, n, cfg.batch_size):
            b = np.sort(perm[i:i + cfg.batch_size])     # sorted rows: sequential reads
            ti = time.perf_counter()
            x = features.gather(inputs.feats, inputs.train_idx[b], device)
            y = inputs.train_y[torch.from_numpy(b)].to(device)
            t_io += time.perf_counter() - ti
            logits = model(x)
            loss = losses.cert_ce(logits, y, model.head.get_weight(), 0.0, 0.0, cfg.tau)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            global_step += 1
            tot_loss += float(loss) * len(b)
            tot_correct += int((logits.argmax(1) == y).sum())
        sched.step()
        train_seconds = time.perf_counter() - t0
        q = _overall(predict(model, inputs.feats, inputs.test_idx, device), inputs.test_y)
        val_rec, is_best = {}, False
        if early:
            vq = _overall(predict(model, inputs.feats, inputs.val_idx, device), inputs.val_y)
            collapsed = bool(vq["accuracy"] <= val_majority + 1e-6)
            key = [int(collapsed), -vq["accuracy"], -vq["f1_weighted"]]
            is_best = best_info is None or key < best_info["key"]
            if is_best:
                best_info = {"epoch": epoch, "key": key, "val_accuracy": vq["accuracy"],
                             "val_f1_weighted": vq["f1_weighted"], "val_collapsed": collapsed}
            val_rec = {"val_accuracy": vq["accuracy"], "val_f1_weighted": vq["f1_weighted"],
                       "val_collapsed": collapsed, "is_best_so_far": is_best}
        history.append({"epoch": epoch, "lr": opt.param_groups[0]["lr"], **val_rec,
                        "loss": tot_loss / n, "train_accuracy_running": 100 * tot_correct / n,
                        "test_accuracy": q["accuracy"], "test_f1_weighted": q["f1_weighted"],
                        "seconds_train": train_seconds, "seconds_io": t_io,
                        "seconds_epoch": time.perf_counter() - t0})
        ck = _checkpoint(model, opt, sched, epoch, global_step, cfg, inputs, seconds_so_far())
        if early:
            ck["best"] = best_info
            if is_best:
                atomic_torch_save(ck, ckdir / "best.pt")
        atomic_torch_save(ck, last)
        if cfg.keep_every and (epoch + 1) % cfg.keep_every == 0:
            atomic_torch_save(ck, ckdir / layout.epoch_checkpoint_file(epoch + 1))
        write_json(run_dir / "history.json", history)
        h = history[-1]
        vtxt = f"val {h['val_accuracy']:.2f}{'*' if is_best else ' '} " if early else ""
        log(f"  epoch {epoch + 1:2d}/{cfg.epochs}  loss {h['loss']:.4f}  "
            f"train {h['train_accuracy_running']:.2f}  {vtxt}test {h['test_accuracy']:.2f}  "
            f"F1 {h['test_f1_weighted']:.2f}  ({h['seconds_epoch']:.1f}s, io {t_io:.1f}s)")
        if stop_after is not None and epoch + 1 >= stop_after and epoch + 1 < cfg.epochs:
            log(f"stopping after epoch {epoch + 1} (--stop-after)")
            return None

    final = _checkpoint(model, opt, sched, cfg.epochs - 1, global_step, cfg, inputs,
                        seconds_so_far())
    atomic_torch_save(final, ckdir / "final.pt")
    evs = {"final": (evaluate(model, cfg, inputs, device), final)}
    if early:
        best = torch.load(ckdir / "best.pt", map_location="cpu", weights_only=True)
        model.load_trainable_state(best["trainable_state"])
        evs["best"] = (evaluate(model, cfg, inputs, device), best)
    sec = seconds_so_far()
    sec.update({f"evaluate_{w}": ev["metrics"]["seconds_evaluate"] for w, (ev, _) in evs.items()})
    sec["total"] = sec["epochs_with_eval"] + sum(v for k, v in sec.items()
                                                 if k.startswith("evaluate_"))
    common = {"config": cfg.to_dict(), "provenance": inputs.provenance, "loss": LOSS_TEXT,
              "trainable_block": backbone.TRAINABLE_BLOCK,
              "trainable_params": model.num_trainable(), "epochs_done": cfg.epochs,
              "dedup_stats": (inputs.manifest or {}).get("dedup_stats"),
              "device": device_info(device), "seconds": sec}
    if early:
        common["early_stopping"] = {**best_info, "best_epoch_1based": best_info["epoch"] + 1,
                                    "n_val": len(inputs.val_idx),
                                    "val_majority_rate": val_majority}
    by_ckpt = {w: write_outputs(run_dir, ev, cfg, inputs,
                                {**common, "fingerprint": ck["fingerprint"],
                                 "epoch": ck["epoch"]}, w)
               for w, (ev, ck) in evs.items()}
    metrics = btrain.write_headline(run_dir, cfg, by_ckpt)
    for which, m in by_ckpt.items():
        sp = m["spectrum"]
        log(f"{which} (epoch {m['epoch'] + 1}): test acc {m['accuracy']:.2f}  "
            f"F1 {m['f1_weighted']:.2f}  sigma_max/base {sp['ratio_sigma_max_min']:.4f}.."
            f"{sp['ratio_sigma_max_max']:.4f}  fingerprint {m['fingerprint']}")
    log(f"time: train {sec['train'] / 60:.1f} min, total {sec['total'] / 60:.1f} min")
    return metrics


# ---------------------------------------------------------------------------
# rebuilding a trained run
# ---------------------------------------------------------------------------

def load_run(run_dir: Path, device=None, which: str = "headline", net=None):
    """The trained Classifier of a finished run (h in, logits out)."""
    run_dir = Path(run_dir)
    device = device or resolve_device("auto")
    if which == "headline":
        which = read_json(run_dir / layout.HEADLINE_FILE).get("headline", "final")
    ckpt = torch.load(run_dir / layout.CHECKPOINTS_DIR / layout.checkpoint_file(which),
                      map_location="cpu", weights_only=True)
    cfg = RunConfig(**ckpt["config"])
    net = backbone.load_resnet152() if net is None else net
    if backbone.state_sha256(net) != ckpt["weights_sha256"]:
        raise RuntimeError("these ResNet152 weights are not the ones the run was trained from")
    model = backbone.Classifier(net.layer4[2], N_CLASSES[cfg.step], cfg.seed, cfg.mode)
    model.load_trainable_state(ckpt["trainable_state"])
    if badapters.fingerprint(model.trainable_state()) != ckpt["fingerprint"]:
        raise RuntimeError(f"fingerprint of the restored weights does not match {which}.pt")
    return model.to(device).eval()


# ---------------------------------------------------------------------------

def add_args(p: argparse.ArgumentParser) -> None:
    d = RunConfig(step=2)
    p.add_argument("--mode", choices=MODES, default=d.mode)
    p.add_argument("--epochs", type=int, default=d.epochs)
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--lr", type=float, default=d.lr)
    p.add_argument("--weight-decay", type=float, default=d.weight_decay)
    p.add_argument("--lr-step-size", type=int, default=d.lr_step_size)
    p.add_argument("--lr-gamma", type=float, default=d.lr_gamma)
    p.add_argument("--tau", type=float, default=d.tau)
    p.add_argument("--feature-dtype", choices=tuple(features.DTYPES), default=d.feature_dtype)
    p.add_argument("--no-deterministic", dest="deterministic", action="store_false")
    p.add_argument("--device", default="auto")
    p.add_argument("--keep-every", type=int, default=0, help="also keep epoch_XXX.pt every K")
    p.add_argument("--ignore-mtime", action="store_true")
    p.add_argument("--from-tuning", dest="from_tuning", action="store_true",
                   help="final protocol: lr from tuning/step_<n>/selection.json, train on the "
                        "fit split, early-stop on the validation split, evaluate best and last "
                        "epoch on test")


def config_from_args(a, step: int, mode: str, seed: int, variant: str = "best") -> RunConfig:
    cfg = RunConfig(step=step, mode=mode, seed=seed, epochs=a.epochs, batch_size=a.batch_size,
                    lr=a.lr, weight_decay=a.weight_decay, lr_step_size=a.lr_step_size,
                    lr_gamma=a.lr_gamma, tau=a.tau, feature_dtype=a.feature_dtype,
                    deterministic=a.deterministic, device=a.device, keep_every=a.keep_every)
    if getattr(a, "from_tuning", False):
        from .tune import tuned_params

        tuned = tuned_params(step, mode, cfg, variant)
        cfg.lr, cfg.epochs, cfg.early_stop = tuned["lr"], tuned["ceiling"], "val"
    return cfg


def final_inputs(step: int, dtype: str, logger=None, ignore_mtime: bool = False,
                 batch_size: int = paths.DEFAULT_BATCH_SIZE) -> RunInputs:
    from . import tune

    inputs = prepare(step, dtype, logger, ignore_mtime)
    return tune.final_inputs(inputs, tune.make_split(inputs, step, batch_size))


def completed_config_mismatch(run_dir: Path, cfg: RunConfig) -> dict:
    saved = RunConfig.saved_compat(read_json(Path(run_dir) / "metrics.json")["config"])
    return {k: (saved.get(k), v) for k, v in cfg.compat().items() if saved.get(k) != v}


def main(argv=None) -> int:
    from repro import console

    console.init()
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--step", type=int, choices=(1, 2), required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--restart", action="store_true", help="discard last.pt / results")
    p.add_argument("--variant", choices=("best", "stable"), default="best")
    p.add_argument("--stop-after", type=int, default=None,
                   help="stop after this many epochs (testing resumption)")
    add_args(p)
    a = p.parse_args(argv)
    if a.variant != "best" and not a.from_tuning:
        p.error("--variant needs --from-tuning")
    cfg = config_from_args(a, a.step, a.mode, a.seed, a.variant)
    rd = paths.run_dir(a.step, a.mode, a.seed, a.variant, cfg.batch_size)
    logger = setup_logging(rd / "train.log", name="resnet_adapt.train")
    if is_complete(rd) and not a.restart:
        diff = completed_config_mismatch(rd, cfg)
        if diff:
            raise RuntimeError(f"{rd} is complete but was trained with {diff}; "
                               "--restart to retrain with the new settings")
        logger.info(f"{rd} is complete; --restart to retrain")
        return 0
    inputs = (final_inputs(a.step, a.feature_dtype, logger, a.ignore_mtime, cfg.batch_size)
              if a.from_tuning
              else prepare(a.step, a.feature_dtype, logger, a.ignore_mtime))
    fit(cfg, inputs, rd, logger=logger, restart=a.restart, stop_after=a.stop_after)
    return 0


if __name__ == "__main__":
    sys.exit(main())
