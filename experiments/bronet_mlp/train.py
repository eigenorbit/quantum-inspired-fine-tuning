"""Train one (step, mode, seed) on cached phi features, then evaluate and certify.

    python -m experiments.bronet_mlp.train --step 2 --mode butterfly --seed 0

Resumes from ``checkpoints/last.pt`` when it exists and is compatible (same
config, BRONet checkpoint, features and data manifest); an incompatible one is
an error -- pass ``--restart`` to start over deliberately. A finished run
(``final.pt`` + ``metrics.json``) is not retrained without ``--restart``.

Outputs: <DERMA_REPRO_OUT>/bronet_mlp/runs/step_<n>/<mode>/seed<s>/ -- see README.
Two protocols:

* ``early_stop="none"`` (default): the reported model is the LAST epoch
  (``final.pt``); per-epoch test metrics in ``history.json`` are for analysis only.
* ``early_stop="val"`` (``--from-tuning``): training on the fit part of train, the
  held-out validation part scored after every epoch; the best validation epoch
  is kept as ``best.pt`` (criterion as in `tune`: accuracy, then weighted F1,
  then the earlier epoch; a collapsed epoch only if every epoch collapsed). BOTH
  ``best.pt`` and ``final.pt`` are evaluated on test (``*_best.*``, ``*_final.*``);
  ``metrics.json`` is the headline, i.e. the best-validation model. The test set
  never influences which epoch is kept.
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

from . import env

import numpy as np
import torch

from repro import metrics as rmetrics
from repro import seeding
from repro.config import CLASS_NAMES, N_CLASSES
from repro.published import PUBLISHED_SUPPORT

from . import adapters, certify, losses
from experiments.common import layout

CHECKPOINT_FORMAT = 1


@dataclass
class RunConfig:
    step: int
    mode: str
    seed: int = 0
    epochs: int = 30
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 0.0
    lr_step_size: int = 2
    lr_gamma: float = 0.7
    loss: str = "cert_ce"
    tau: float = 4.0
    eps_train: float = 36 / 255
    # eps warm-up: the loss uses eps_start + (eps_train - eps_start) *
    # min(1, epoch / (eps_warmup * epochs)); 0 = eps_train from the first step
    eps_warmup: float = 0.0
    eps_start: float = 0.0
    lora_rank: int = 4
    lip_power_iters: int = 100
    deterministic: bool = True
    early_stop: str = "none"      # "none" | "val"
    # --- runtime only: never part of the resume-compatibility check ---
    device: str = "auto"
    keep_every: int = 0

    RUNTIME = ("device", "keep_every")

    def __post_init__(self):
        if self.mode not in adapters.MODES:
            raise ValueError(f"mode must be one of {adapters.MODES}")
        if self.loss not in losses.LOSSES:
            raise ValueError(f"loss must be one of {losses.LOSSES}")
        if self.early_stop not in ("none", "val"):
            raise ValueError("early_stop must be 'none' or 'val'")
        if not 0 <= self.eps_warmup < 1:
            raise ValueError("eps_warmup is a fraction of the epochs, in [0, 1)")
        if self.eps_warmup and self.eps_warmup * self.epochs > self.epochs - 1:
            raise ValueError(f"eps_warmup {self.eps_warmup:g} never reaches eps_train within "
                             f"{self.epochs} epochs")
        if not 0 <= self.eps_start <= self.eps_train:
            raise ValueError("eps_start must lie in [0, eps_train]")

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def compat(self) -> dict:
        return {k: v for k, v in self.to_dict().items() if k not in self.RUNTIME}

    @classmethod
    def saved_compat(cls, saved: dict) -> dict:
        """A config read from an older run: fields added since then take their
        default (the behaviour that run actually had), runtime fields dropped."""
        defaults = {f.name: f.default for f in dataclasses.fields(cls)
                    if f.default is not dataclasses.MISSING}
        return {k: v for k, v in {**defaults, **saved}.items() if k not in cls.RUNTIME}

    def eps_at(self, epoch: int) -> float:
        """eps of the loss in `epoch` (0-based): the warm-up ramp, then eps_train."""
        if not self.eps_warmup:
            return self.eps_train
        ramp = min(1.0, epoch / (self.eps_warmup * self.epochs))
        return self.eps_start + (self.eps_train - self.eps_start) * ramp

    def eps_full(self, epoch: int) -> bool:
        """Whether `epoch` trains at the full eps_train (the warm-up is over); only
        such epochs can be the best-validation one."""
        return not self.eps_warmup or epoch >= self.eps_warmup * self.epochs


@dataclass
class RunInputs:
    """Everything a run consumes. Built from the caches by `prepare`, or by hand
    (synthetic features) in the tests."""

    train_x: torch.Tensor
    train_y: torch.Tensor
    test_x: torch.Tensor
    test_y: torch.Tensor
    base_weight: torch.Tensor
    base_bias: torch.Tensor
    L_pre_official: float
    L_pre_rigorous: float
    provenance: dict            # bronet_ckpt_sha256, features_index_sha256, data_manifest_sha256, ...
    test_rows: list[dict]       # image, patient, source_stem per test image
    L_mlp0: dict | None = None  # exact spectrum of the base (computed if None)
    manifest: dict | None = None
    val_x: torch.Tensor | None = None   # held-out part of train, for early stopping
    val_y: torch.Tensor | None = None


# ---------------------------------------------------------------------------
# inputs from the caches
# ---------------------------------------------------------------------------

def base_lmlp0(base_weight: torch.Tensor) -> dict:
    """Exact spectrum of the pretrained MLP, cached per base under backbone/."""
    key = env.sha256_bytes(base_weight.contiguous().numpy().tobytes())
    path = env.backbone_dir() / "L_mlp0.json"
    if path.exists():
        cached = env.read_json(path)
        if cached.get("base_sha256") == key:
            return cached["lmlp"]
    res = certify.lmlp_exact(base_weight)
    env.write_json(path, {"base_sha256": key, "lmlp": res})
    return res


def prepare(step: int, logger=None, ignore_mtime: bool = False) -> RunInputs:
    from . import bronet, features
    from . import data as bdata

    train, test, stats = bdata.records(step, logger)
    man = bdata.manifest(step, train, test, stats)
    tr_paths = [r.path for r in train]
    te_paths = [r.path for r in test]
    train_x = features.load(step, tr_paths, ignore_mtime)
    test_x = features.load(step, te_paths, ignore_mtime)
    rels = [row[0] for row in man["train"]] + [row[0] for row in man["test"]]
    feat_sha = features.features_sha256(rels, torch.cat([train_x, test_x]))
    base = bronet.load_mlp_base()
    lpre = bronet.load_lpre()
    base_file = bronet.mlp_base_path()
    return RunInputs(
        train_x=train_x, train_y=torch.tensor([r.label for r in train]),
        test_x=test_x, test_y=torch.tensor([r.label for r in test]),
        base_weight=base["weight"], base_bias=base["bias"],
        L_pre_official=lpre["L_pre_official"], L_pre_rigorous=lpre["L_pre_rigorous"],
        provenance={"bronet_ckpt_sha256": env.checkpoint_sha256(),
                    "features_index_sha256": feat_sha,
                    "data_manifest_sha256": bdata.manifest_sha256(man),
                    "mlp_base": str(base_file),
                    "mlp_base_sha256": env.sha256_bytes(
                        base["weight"].numpy().tobytes() + base["bias"].numpy().tobytes()),
                    "L_pre_file": str(bronet.lpre_path())},
        test_rows=[{"image": r.path.name, "patient": r.patient, "source_stem": r.source_stem}
                   for r in test],
        L_mlp0=base_lmlp0(base["weight"]), manifest=man)


# ---------------------------------------------------------------------------
# checkpoints
# ---------------------------------------------------------------------------

def _checkpoint(model, opt, sched, epoch, global_step, cfg, inputs) -> dict:
    state = model.trainable_state()
    return {"format": CHECKPOINT_FORMAT, "trainable_state": state,
            "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
            "epoch": epoch, "global_step": global_step, "rng": seeding.capture(),
            "config": cfg.to_dict(),
            "bronet_ckpt_sha256": inputs.provenance["bronet_ckpt_sha256"],
            "features_index_sha256": inputs.provenance["features_index_sha256"],
            "data_manifest_sha256": inputs.provenance["data_manifest_sha256"],
            "mlp_base_sha256": inputs.provenance.get("mlp_base_sha256", ""),
            "fingerprint": adapters.fingerprint(state)}


def check_compatible(ckpt: dict, cfg: RunConfig, inputs: RunInputs, path: Path) -> None:
    saved = RunConfig.saved_compat(ckpt["config"])
    problems = [f"config.{k}: checkpoint {saved.get(k)!r} vs now {v!r}"
                for k, v in cfg.compat().items() if saved.get(k) != v]
    problems += [f"config.{k}: only in checkpoint" for k in saved if k not in cfg.compat()]
    for key in ("bronet_ckpt_sha256", "features_index_sha256", "data_manifest_sha256",
                "mlp_base_sha256"):
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
def predict(model, x: torch.Tensor, batch: int = 1024) -> torch.Tensor:
    model.eval()
    return torch.cat([model(x[i:i + batch]) for i in range(0, len(x), batch)])


def _quality(logits, y, step, tau) -> dict:
    probs = torch.softmax(tau * logits.double(), 1).numpy()
    pred = rmetrics.Predictions(y_true=y.numpy(), y_pred=logits.argmax(1).numpy(),
                                probabilities=probs)
    rep = rmetrics.full_report(pred, step)
    if step == 1:
        support = PUBLISHED_SUPPORT[1]
        rep["accuracy_paper_composition"] = sum(
            rep["per_class"][c]["recall"] * n for c, n in support.items()) / sum(support.values())
    return rep


def evaluate(model, cfg: RunConfig, inputs: RunInputs, device) -> dict:
    """Quality, Lipschitz constants and certificates of the current model."""
    t0 = time.perf_counter()
    x, y = inputs.test_x.to(device), inputs.test_y
    logits = predict(model, x).cpu()
    quality = _quality(logits, y, cfg.step, cfg.tau)
    mats = model.mlp.effective_matrices().detach()
    lm = certify.lmlp_exact(mats)
    l0 = inputs.L_mlp0 or certify.lmlp_exact(inputs.base_weight)
    L_off = inputs.L_pre_official * lm["L_mlp"]
    L_rig = inputs.L_pre_rigorous * lm["L_mlp"]
    head_w = model.head.get_weight().detach().cpu()
    cert_off = certify.summary(logits, y, head_w, L_off)
    cert_rig = certify.summary(logits, y, head_w, L_rig)
    r_off = certify.radii(logits, head_w, L_off)
    r_rig = certify.radii(logits, head_w, L_rig)
    model.eval()
    bound = certify.check_mlp_bound(model.mlp, x[:128], lm["L_mlp"])
    if not bound["respected"]:
        raise RuntimeError(f"empirical MLP bound violated: {bound}")
    counts = torch.bincount(y, minlength=N_CLASSES[cfg.step]).double()
    majority = float(counts.max() / counts.sum())
    acc = quality["overall"]["accuracy"]
    metrics = {
        "quality": quality,
        "accuracy": acc, "f1_weighted": quality["overall"]["f1_weighted"],
        "majority_class_rate": 100 * majority, "collapsed": acc <= 100 * majority + 1e-6,
        "lipschitz": {
            "L_pre_official": inputs.L_pre_official, "L_pre_rigorous": inputs.L_pre_rigorous,
            "L_mlp": lm["L_mlp"], "L_mlp0": l0["L_mlp"],
            "L_total_official": L_off, "L_total_rigorous": L_rig,
            "L0_official": inputs.L_pre_official * l0["L_mlp"],
            "L0_rigorous": inputs.L_pre_rigorous * l0["L_mlp"],
            "L_over_L0": lm["L_mlp"] / l0["L_mlp"],
            "sigma_max_per_layer": lm["sigma_max_per_layer"],
            "sigma_min_per_layer": lm["sigma_min_per_layer"],
            "worst_sigma_max": lm["worst_sigma_max"], "worst_sigma_min": lm["worst_sigma_min"],
            "mlp_bound_check": bound,
        },
        "certificate_official": cert_off, "certificate_rigorous": cert_rig,
        "seconds_evaluate": time.perf_counter() - t0,
    }
    if cfg.mode == "butterfly":
        metrics["lipschitz"]["orthogonality_error_max"] = max(
            r.orthogonality_error() for r in model.mlp.rotations)
    arrays = {"logits": logits.numpy(), "y": y.numpy(), "pred": r_off["pred"].numpy(),
              "margin": r_off["margin"].numpy(),
              "radius_official": r_off["radius"].numpy(),
              "radius_rigorous": r_rig["radius"].numpy(),
              "radius_600_official": r_off["radius"].numpy() / certify.RESIZE_600_TO_224_NORM,
              "radius_600_rigorous": r_rig["radius"].numpy() / certify.RESIZE_600_TO_224_NORM,
              "head_weight": head_w.numpy()}
    return {"metrics": metrics, "arrays": arrays}


def write_predictions(path: Path, rows: list[dict], arrays: dict, step: int, tau: float) -> None:
    names = CLASS_NAMES[step]
    logits = arrays["logits"]
    probs = torch.softmax(tau * torch.from_numpy(logits).double(), 1).numpy()
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as handle:
        w = csv.writer(handle)
        w.writerow(["image", "patient", "source_stem", "true_label", "true_class",
                    "predicted_label", "predicted_class", "correct"]
                   + [f"prob_{n}" for n in names]
                   + ["margin", "radius", "radius_rigorous"] + [f"logit_{n}" for n in names])
        for i, row in enumerate(rows):
            t, p = int(arrays["y"][i]), int(arrays["pred"][i])
            w.writerow([row["image"], row["patient"], row["source_stem"], t, names[t], p,
                        names[p], int(t == p)] + [f"{v:.6f}" for v in probs[i]]
                       + [f"{arrays['margin'][i]:.8f}", f"{arrays['radius_official'][i]:.8f}",
                          f"{arrays['radius_rigorous'][i]:.8f}"]
                       + [f"{v:.8f}" for v in logits[i]])
    tmp.replace(path)


CHECKPOINTS = ("final", "best")


def write_outputs(run_dir: Path, ev: dict, cfg: RunConfig, inputs: RunInputs, extra: dict,
                  which: str = "final") -> dict:
    """metrics_<which>.json, predictions_<which>.csv, certificates_<which>.npz."""
    metrics = {**extra, "checkpoint": which, **ev["metrics"]}
    env.write_json(run_dir / layout.metrics_file(which), metrics)
    write_predictions(run_dir / layout.predictions_file(which), inputs.test_rows, ev["arrays"],
                      cfg.step, cfg.tau)
    final = run_dir / layout.arrays_file("certificates", which)
    tmp = final.with_name(final.stem + ".tmp.npz")
    np.savez_compressed(tmp, **ev["arrays"])
    tmp.replace(final)
    return metrics


def headline(cfg: RunConfig) -> str:
    """Which checkpoint metrics.json describes."""
    return "best" if cfg.early_stop == "val" else "final"


def write_headline(run_dir: Path, cfg: RunConfig, by_ckpt: dict) -> dict:
    main = dict(by_ckpt[headline(cfg)])
    main["headline"] = headline(cfg)
    main["other_checkpoints"] = {k: {"accuracy": v["accuracy"], "f1_weighted": v["f1_weighted"],
                                     "fingerprint": v["fingerprint"], "epoch": v.get("epoch")}
                                 for k, v in by_ckpt.items() if k != headline(cfg)}
    env.write_json(run_dir / "metrics.json", main)
    return main


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------

def is_complete(run_dir: Path) -> bool:
    """final.pt and metrics.json exist (metrics.json is written last)."""
    return (run_dir / "checkpoints" / "final.pt").exists() and (run_dir / "metrics.json").exists()


def _clear_outputs(run_dir: Path) -> None:
    """--restart / --force: delete what a training writes (keeps config.json,
    data_manifest.json, the logs and attacks/)."""
    names = ["checkpoints", "history.json", "metrics.json"] + [
        f"{kind}_{w}.{ext}" for w in CHECKPOINTS
        for kind, ext in (("metrics", "json"), ("predictions", "csv"), ("certificates", "npz"))]
    for name in names:
        target = run_dir / name
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()


def _resume(last: Path, model, opt, sched, cfg: RunConfig, inputs: RunInputs) -> dict:
    """Restore weights, optimizer, scheduler and every RNG from last.pt (after
    checking that it was made with the same settings); returns the checkpoint."""
    ckpt = torch.load(last, map_location="cpu", weights_only=True)
    check_compatible(ckpt, cfg, inputs, last)
    model.load_trainable_state(ckpt["trainable_state"])
    opt.load_state_dict(ckpt["optimizer"])
    sched.load_state_dict(ckpt["scheduler"])
    seeding.restore(ckpt["rng"])
    return ckpt


def _run_record(cfg: RunConfig, inputs: RunInputs, model, num_classes: int, device) -> dict:
    """config.json: the settings, the inputs' provenance and the environment."""
    return {"config": cfg.to_dict(), "provenance": inputs.provenance,
            "trainable_params": model.num_trainable(), "num_classes": num_classes,
            "n_train": len(inputs.train_y), "n_test": len(inputs.test_y),
            "L_pre_official": inputs.L_pre_official,
            "L_pre_rigorous": inputs.L_pre_rigorous,
            "env": env.env_snapshot(), "device": env.device_info(device),
            "git": env.git_info(), "code_sha256": env.code_sha256(),
            "command": sys.argv}


def _train_epoch(model, opt, cfg: RunConfig, x_tr, y_tr, perm, L_train: float, eps: float):
    """One pass over the training set in the order `perm`, the loss at radius
    `eps`: (sum of losses, number correct, optimizer steps)."""
    n = len(y_tr)
    tot_loss, tot_correct, steps = 0.0, 0, 0
    for i in range(0, n, cfg.batch_size):
        idx = perm[i:i + cfg.batch_size]
        logits = model(x_tr[idx])
        loss = losses.compute(cfg.loss, logits, y_tr[idx], model.head.get_weight(),
                              L_train, eps, cfg.tau)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        steps += 1
        tot_loss += float(loss) * len(idx)
        tot_correct += int((logits.argmax(1) == y_tr[idx]).sum())
    return tot_loss, tot_correct, steps


@torch.no_grad()
def head_distances(model) -> dict:
    """min / median / max over class pairs of ||w_i - w_j|| of the LLN head (unit
    rows: 0 = rows coincide, sqrt(2) = orthogonal, 2 = opposite). The loss can
    shrink its penalty L * eps * ||w_y - w_j|| by pulling the rows together."""
    w = model.head.get_weight().detach().double()
    i, j = torch.triu_indices(len(w), len(w), 1)
    d = (w[i] - w[j]).norm(dim=1)
    return {"head_dist_min": float(d.min()), "head_dist_median": float(d.median()),
            "head_dist_max": float(d.max())}


def _validate(model, x_va, y_va, num_classes: int, val_majority: float, epoch: int,
              best_info: dict | None, eligible: bool = True) -> tuple[dict, dict | None]:
    """The history fields of the validation split, and the new best-epoch record
    if this epoch is better than `best_info` (else None). An epoch that is not
    `eligible` (eps warm-up still running) is never the best."""
    vl = predict(model, x_va).cpu()
    vq = rmetrics.overall(rmetrics.Predictions(
        y_true=y_va.numpy(), y_pred=vl.argmax(1).numpy(),
        probabilities=np.zeros((len(y_va), num_classes))))
    # plain python types: this goes into last.pt (weights_only=True on load)
    vq = {k: float(v) for k, v in vq.items()}
    collapsed = bool(vq["accuracy"] <= val_majority + 1e-6)
    # smaller is better; an earlier epoch wins a tie because only a strict
    # improvement replaces the kept one
    key = [int(collapsed), -vq["accuracy"], -vq["f1_weighted"]]
    is_best = eligible and (best_info is None or key < best_info["key"])
    best = ({"epoch": epoch, "key": key, "val_accuracy": vq["accuracy"],
             "val_f1_weighted": vq["f1_weighted"], "val_collapsed": collapsed}
            if is_best else None)
    return ({"val_accuracy": vq["accuracy"], "val_f1_weighted": vq["f1_weighted"],
             "val_collapsed": collapsed, "is_best_so_far": is_best}, best)


def _evaluate_checkpoints(model, cfg: RunConfig, inputs: RunInputs, device, run_dir: Path,
                          final: dict, common: dict, early: bool, timer) -> tuple[dict, dict]:
    """Evaluate final.pt (the model as it is) and, with early stopping, best.pt;
    write their outputs and metrics.json. Returns (headline metrics, per checkpoint)."""
    by_ckpt = {}
    with timer.section("evaluate"):
        ev = evaluate(model, cfg, inputs, device)
    by_ckpt["final"] = write_outputs(
        run_dir, ev, cfg, inputs,
        {**common, "fingerprint": final["fingerprint"], "epoch": cfg.epochs - 1,
         "seconds": dict(timer.seconds)}, "final")
    if early:
        best = torch.load(run_dir / "checkpoints" / "best.pt", map_location="cpu",
                          weights_only=True)
        model.load_trainable_state(best["trainable_state"])
        with timer.section("evaluate best"):
            ev_b = evaluate(model, cfg, inputs, device)
        by_ckpt["best"] = write_outputs(
            run_dir, ev_b, cfg, inputs,
            {**common, "fingerprint": best["fingerprint"], "epoch": best["epoch"],
             "seconds": dict(timer.seconds)}, "best")
    return write_headline(run_dir, cfg, by_ckpt), by_ckpt


def fit(cfg: RunConfig, inputs: RunInputs, run_dir: Path, *, logger=None, restart=False,
        stop_after: int | None = None) -> dict | None:
    """Train, checkpoint every epoch, evaluate `final.pt`. Returns metrics, or None
    when stopped early by `stop_after` (a simulated interruption)."""
    log = logger.info if logger else (lambda *_: None)
    run_dir = Path(run_dir)
    ckdir = run_dir / "checkpoints"
    timer = env.Timer(logger)
    device = env.resolve_device(cfg.device)
    if restart and run_dir.exists():
        _clear_outputs(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    seeding.configure_determinism(cfg.deterministic)
    seeding.set_seed(cfg.seed)
    if inputs.L_mlp0 is None:
        inputs.L_mlp0 = certify.lmlp_exact(inputs.base_weight)
    num_classes = int(max(inputs.train_y.max(), inputs.test_y.max())) + 1
    model = adapters.build(inputs.base_weight, inputs.base_bias, cfg.mode, num_classes,
                           cfg.lora_rank, cfg.seed).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=cfg.lr_step_size, gamma=cfg.lr_gamma)
    history, start_epoch, global_step = [], 0, 0
    if (run_dir / "history.json").exists():
        history = env.read_json(run_dir / "history.json")

    last = ckdir / "last.pt"
    ckpt = _resume(last, model, opt, sched, cfg, inputs) if last.exists() else None
    if ckpt is not None:
        start_epoch, global_step = ckpt["epoch"] + 1, ckpt["global_step"]
        history = [h for h in history if h["epoch"] < start_epoch]
        log(f"resumed from {last} at epoch {start_epoch}")
    else:
        history = []

    env.write_json(run_dir / "config.json", _run_record(cfg, inputs, model, num_classes, device))
    if inputs.manifest is not None:
        env.write_json(run_dir / "data_manifest.json", inputs.manifest)

    x_tr, y_tr = inputs.train_x.to(device), inputs.train_y.to(device)
    x_te, y_te = inputs.test_x.to(device), inputs.test_y
    n = len(y_tr)
    fixed_L = cfg.mode in ("head-only", "butterfly")
    early = cfg.early_stop == "val"
    if early:
        if inputs.val_x is None:
            raise ValueError("early_stop='val' needs RunInputs.val_x / val_y")
        x_va, y_va = inputs.val_x.to(device), inputs.val_y
        counts = torch.bincount(y_va, minlength=num_classes).double()
        val_majority = 100 * float(counts.max() / counts.sum())
    best_info = ckpt.get("best") if ckpt is not None else None
    log(f"step-{cfg.step} {cfg.mode} seed {cfg.seed}: {model.num_trainable():,} trainable, "
        f"train {n:,}" + (f" / val {len(y_va):,}" if early else "")
        + f" / test {len(y_te):,}, device {device}, early stop {cfg.early_stop}")

    t_train = time.perf_counter()
    for epoch in range(start_epoch, cfg.epochs):
        t0 = time.perf_counter()
        model.train()
        L_mlp_train = (inputs.L_mlp0["L_mlp"] if fixed_L
                       else certify.lmlp_power(model.mlp.effective_matrices(),
                                               cfg.lip_power_iters))
        L_train = inputs.L_pre_official * L_mlp_train
        g = torch.Generator().manual_seed(cfg.seed * 1_000_003 + epoch)
        perm = torch.randperm(n, generator=g).to(device)
        eps_now = cfg.eps_at(epoch)
        tot_loss, tot_correct, steps = _train_epoch(model, opt, cfg, x_tr, y_tr, perm, L_train,
                                                    eps_now)
        global_step += steps
        sched.step()
        train_seconds = time.perf_counter() - t0
        test_logits = predict(model, x_te).cpu()
        q = rmetrics.overall(rmetrics.Predictions(
            y_true=y_te.numpy(), y_pred=test_logits.argmax(1).numpy(),
            probabilities=np.zeros((len(y_te), num_classes))))
        L_mlp_now = (inputs.L_mlp0["L_mlp"] if fixed_L
                     else certify.lmlp_power(model.mlp.effective_matrices(),
                                             cfg.lip_power_iters))
        cert = certify.summary(test_logits, y_te, model.head.get_weight().detach().cpu(),
                               inputs.L_pre_official * L_mlp_now)
        val_rec, is_best = {}, False
        if early:
            val_rec, best = _validate(model, x_va, y_va, num_classes, val_majority, epoch,
                                      best_info, cfg.eps_full(epoch))
            is_best = best is not None
            if is_best:
                best_info = best
                env.atomic_torch_save(
                    _checkpoint(model, opt, sched, epoch, global_step, cfg, inputs),
                    ckdir / "best.pt")
        heads = head_distances(model)
        history.append({"epoch": epoch, "lr": opt.param_groups[0]["lr"], **val_rec,
                        "eps": eps_now, "eps_full": cfg.eps_full(epoch), **heads,
                        "loss": tot_loss / n, "train_accuracy_running": 100 * tot_correct / n,
                        "test_accuracy": q["accuracy"], "test_f1_weighted": q["f1_weighted"],
                        "L_train": L_train, "L_mlp_train": L_mlp_train,
                        "L_mlp_end_power": L_mlp_now,
                        "test_certified_accuracy_36": cert["certified_accuracy_named"]["36/255"],
                        "test_median_radius": cert["median_radius"],
                        "seconds_train": train_seconds,
                        "seconds_epoch": time.perf_counter() - t0})
        ck = _checkpoint(model, opt, sched, epoch, global_step, cfg, inputs)
        if early:
            ck["best"] = best_info
        env.atomic_torch_save(ck, last)
        if cfg.keep_every and (epoch + 1) % cfg.keep_every == 0:
            env.atomic_torch_save(ck, ckdir / layout.epoch_checkpoint_file(epoch + 1))
        env.write_json(run_dir / "history.json", history)
        h = history[-1]
        vtxt = (f"val {h['val_accuracy']:.2f}{'*' if is_best else ' '} " if early else "")
        log(f"  epoch {epoch + 1:2d}/{cfg.epochs}  loss {h['loss']:.4f}  "
            f"train {h['train_accuracy_running']:.2f}  {vtxt} test {h['test_accuracy']:.2f}  "
            f"F1 {h['test_f1_weighted']:.2f}  L_mlp {L_mlp_now:.5f}  "
            f"cert@36 {100 * h['test_certified_accuracy_36']:.1f}  "
            f"eps {255 * eps_now:.1f}/255  head |dw| {heads['head_dist_min']:.3f}.."
            f"{heads['head_dist_max']:.3f}  ({h['seconds_epoch']:.1f}s)")
        if stop_after is not None and epoch + 1 >= stop_after and epoch + 1 < cfg.epochs:
            log(f"stopping after epoch {epoch + 1} (--stop-after)")
            return None
    timer.seconds["train"] = time.perf_counter() - t_train

    final = _checkpoint(model, opt, sched, cfg.epochs - 1, global_step, cfg, inputs)
    env.atomic_torch_save(final, ckdir / "final.pt")
    common = {"config": cfg.to_dict(), "provenance": inputs.provenance,
              "trainable_params": model.num_trainable(), "epochs_done": cfg.epochs,
              "dedup_stats": (inputs.manifest or {}).get("dedup_stats"),
              "device": env.device_info(device),
              "history_seconds_train_sum": sum(h["seconds_epoch"] for h in history)}
    if early:
        common["early_stopping"] = {**best_info, "best_epoch_1based": best_info["epoch"] + 1,
                                    "n_val": len(y_va), "val_majority_rate": val_majority}
    metrics, by_ckpt = _evaluate_checkpoints(model, cfg, inputs, device, run_dir, final, common,
                                             early, timer)
    for which, m in by_ckpt.items():
        lip = m["lipschitz"]
        log(f"{which} (epoch {m['epoch'] + 1}): test acc {m['accuracy']:.2f}  "
            f"F1 {m['f1_weighted']:.2f}  L/L0 {lip['L_over_L0']:.6g}  "
            f"L_total {lip['L_total_official']:.4f} (rigorous {lip['L_total_rigorous']:.4f})  "
            f"median r {m['certificate_official']['median_radius']:.4f}  cert@36/255 "
            f"{100 * m['certificate_official']['certified_accuracy_named']['36/255']:.1f}%  "
            f"fingerprint {m['fingerprint']}")
    return metrics


# ---------------------------------------------------------------------------
# rebuilding a trained run
# ---------------------------------------------------------------------------


def load_run(run_dir: Path, device=None, which: str = "headline"):
    """The trained AdaptedClassifier (phi features in) of a finished run, from one
    checkpoint alone. `which`: "final" (last epoch), "best" (best validation epoch,
    early-stopped runs) or "headline" (whatever metrics.json reports)."""
    run_dir = Path(run_dir)
    device = device or env.resolve_device("auto")
    if which == "headline":
        which = env.read_json(run_dir / "metrics.json").get("headline", "final")
    ckpt = torch.load(run_dir / layout.CHECKPOINTS_DIR / layout.checkpoint_file(which),
                      map_location="cpu",
                      weights_only=True)
    conf = env.read_json(run_dir / "config.json")
    cfg = RunConfig(**ckpt["config"])
    base_path = Path(conf["provenance"]["mlp_base"])
    base = torch.load(base_path, map_location="cpu", weights_only=True)
    sha = env.sha256_bytes(base["weight"].numpy().tobytes() + base["bias"].numpy().tobytes())
    if sha != ckpt.get("mlp_base_sha256", sha):
        raise RuntimeError(f"{base_path} is not the MLP base this run was trained from")
    model = adapters.build(base["weight"], base["bias"], cfg.mode, conf["num_classes"],
                           cfg.lora_rank, cfg.seed)
    model.load_trainable_state(ckpt["trainable_state"])
    if adapters.fingerprint(model.trainable_state()) != ckpt["fingerprint"]:
        raise RuntimeError(f"fingerprint of the restored weights does not match {which}.pt")
    return model.to(device).eval()


# ---------------------------------------------------------------------------

def add_args(p: argparse.ArgumentParser) -> None:
    d = RunConfig(step=2, mode="head-only")
    p.add_argument("--epochs", type=int, default=d.epochs)
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--lr", type=float, default=d.lr)
    p.add_argument("--weight-decay", type=float, default=d.weight_decay)
    p.add_argument("--lr-step-size", type=int, default=d.lr_step_size)
    p.add_argument("--lr-gamma", type=float, default=d.lr_gamma)
    p.add_argument("--loss", choices=losses.LOSSES, default=d.loss)
    p.add_argument("--tau", type=float, default=d.tau)
    p.add_argument("--eps-train", type=float, default=d.eps_train)
    p.add_argument("--eps-warmup", type=float, default=d.eps_warmup,
                   help="ramp the loss eps linearly from --eps-start to --eps-train over this "
                        "fraction of the epochs (0 = off); the best-validation epoch is "
                        "chosen only after the ramp")
    p.add_argument("--eps-start", type=float, default=d.eps_start,
                   help="eps at epoch 0 of the warm-up (same units as --eps-train)")
    p.add_argument("--lora-rank", type=int, default=d.lora_rank)
    p.add_argument("--lip-power-iters", type=int, default=d.lip_power_iters)
    p.add_argument("--no-deterministic", dest="deterministic", action="store_false")
    p.add_argument("--device", default="auto")
    p.add_argument("--keep-every", type=int, default=0, help="also keep epoch_XXX.pt every K")
    p.add_argument("--ignore-mtime", action="store_true")
    p.add_argument("--from-tuning", "--lr-from-tuning", dest="from_tuning",
                   action="store_true",
                   help="final protocol: the lr chosen per mode by `tune` "
                        "(tuning/step_<n>/selection.json), train on the fit split up to the "
                        "tuning epoch ceiling, early-stop on the validation split, evaluate "
                        "both the best-validation and the last epoch on test")


def config_from_args(a, step: int, mode: str, seed: int, variant: str = "best") -> RunConfig:
    cfg = RunConfig(step=step, mode=mode, seed=seed, epochs=a.epochs, batch_size=a.batch_size,
                    lr=a.lr, weight_decay=a.weight_decay, lr_step_size=a.lr_step_size,
                    lr_gamma=a.lr_gamma, loss=a.loss, tau=a.tau, eps_train=a.eps_train,
                    eps_warmup=a.eps_warmup, eps_start=a.eps_start, lora_rank=a.lora_rank, lip_power_iters=a.lip_power_iters,
                    deterministic=a.deterministic, device=a.device, keep_every=a.keep_every)
    if getattr(a, "from_tuning", False):
        from .tune import tuned_params

        tuned = tuned_params(step, mode, cfg, variant)
        cfg.lr, cfg.epochs, cfg.early_stop = tuned["lr"], tuned["ceiling"], "val"
    return cfg


def final_inputs(step: int, logger=None, ignore_mtime: bool = False) -> RunInputs:
    """Inputs of the final protocol: fit part of train / validation part / test."""
    from . import tune

    inputs = prepare(step, logger, ignore_mtime)
    return tune.final_inputs(inputs, tune.make_split(inputs, step, save=False))


def completed_config_mismatch(run_dir: Path, cfg: RunConfig) -> dict:
    """{key: (saved, now)} for a finished run whose config differs from `cfg`."""
    saved = RunConfig.saved_compat(env.read_json(Path(run_dir) / "metrics.json")["config"])
    return {k: (saved.get(k), v) for k, v in cfg.compat().items() if saved.get(k) != v}


def main(argv=None) -> int:
    from repro import console

    console.init()
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--step", type=int, choices=(1, 2), required=True)
    p.add_argument("--mode", choices=adapters.MODES, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--restart", action="store_true", help="discard last.pt / results")
    p.add_argument("--variant", choices=("best", "stable"), default="best",
                   help="with --from-tuning: which selection (runs of `stable` go to "
                        "runs/step_<n>/<mode>_stable/)")
    p.add_argument("--stop-after", type=int, default=None,
                   help="stop after this many epochs (testing resumption)")
    add_args(p)
    a = p.parse_args(argv)
    if a.variant != "best" and not a.from_tuning:
        p.error("--variant needs --from-tuning")
    cfg = config_from_args(a, a.step, a.mode, a.seed, a.variant)
    rd = env.run_dir(a.step, a.mode, a.seed, a.variant, cfg.batch_size)
    logger = env.setup_logging(rd / "train.log")
    if is_complete(rd) and not a.restart:
        diff = completed_config_mismatch(rd, cfg)
        if diff:
            raise RuntimeError(f"{rd} is complete but was trained with {diff}; "
                               "--restart to retrain with the new settings")
        logger.info(f"{rd} is complete; --restart to retrain")
        return 0
    with env.Timer(logger).section("prepare inputs"):
        inputs = (final_inputs(a.step, logger, a.ignore_mtime) if a.from_tuning
                  else prepare(a.step, logger, a.ignore_mtime))
    fit(cfg, inputs, rd, logger=logger, restart=a.restart, stop_after=a.stop_after)
    return 0


if __name__ == "__main__":
    sys.exit(main())
