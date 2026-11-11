"""The attack engine: PGD-L2 and random noise, the two-phase runner, the summary.

Independent of the network: the features come from an injected `feature_fn(x, grad)`
(tests, resnet_adapt) or from a `Backend` (bronet_mlp: BRONet, its trained runs and
test images); the stronger attacks are in `strong`, the eps* search in `min_eps`.
How the attacks work, the two phases and the sanity check:
experiments/bronet_mlp/attacks.py.

The row fields `pgd_success` / `pgd_robust` hold the result of whatever attack the
folder ran (the names are shared with the first, PGD, folders).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from .. import certify, layout
from ..device import device_info
from ..display import label
from ..io import Timer, atomic_write_text, read_json, write_csv, write_json


@dataclass
class Backend:
    """What the engine loads when the caller does not inject it."""

    name: str                      # for the timing section "load <name>"
    out_dir: Callable              # (step, checkpoint, train_batch_size, attack) -> Path
    load_targets: Callable         # (cfg, device, train_batch_size) -> list[dict]
    load_test_images: Callable     # (cfg) -> {"x", "y", "names", "test_index"}
    network_functions: Callable    # (cfg, device, strategy, cache_bro_weights, eval_chunk,
                                   #  minibatch, logger) -> (grad_fn, eval_fn)


EPS_NAMES = ("36/255", "72/255", "108/255")



def margin_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """max_{j != y} z_j - z_y per row; positive means the attack has succeeded."""
    true = logits.gather(1, y[:, None]).squeeze(1)
    other = logits.masked_fill(torch.nn.functional.one_hot(y, logits.shape[1]).bool(),
                               float("-inf")).amax(dim=1)
    return other - true



def multi_logits(classifiers, model_idx: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
    """Row i through classifiers[model_idx[i]]."""
    out = None
    for m in model_idx.unique().tolist():
        rows = (model_idx == m).nonzero(as_tuple=True)[0]
        z = classifiers[m](feats[rows])
        if out is None:
            out = feats.new_zeros(len(feats), z.shape[1])
        out = out.index_copy(0, rows, z)
    return out



def unit_directions(shape, seeds: list[int]) -> torch.Tensor:
    """One random unit-L2 direction per row (CPU), each from its own seed, so the
    result does not depend on how instances are batched."""
    rows = []
    for s in seeds:
        g = torch.Generator().manual_seed(int(s) % (2**63))
        d = torch.randn(shape[1:], generator=g)
        rows.append(d / d.norm().clamp_min(1e-12))
    return torch.stack(rows)



def _norms(d: torch.Tensor) -> torch.Tensor:
    return d.flatten(1).norm(dim=1)



def pgd_deltas(grad_fn, classifiers, model_idx, x, y, eps, *, steps: int, restarts: int,
               alpha_factor: float, seeds: list[int]) -> list[torch.Tensor]:
    """Phase A: one PGD-L2 perturbation per restart and row (each row its own eps).
    Returned on the CPU; their effect is measured exactly in phase B."""
    view = (-1,) + (1,) * (x.dim() - 1)
    eps_v = eps.view(view)
    alpha = (alpha_factor * eps / steps).view(view)
    out = []
    for r in range(restarts):
        d = unit_directions(x.shape, [s * 1_000 + r for s in seeds]).to(x.device) * eps_v
        for _ in range(steps):
            d = d.detach().requires_grad_(True)
            loss = margin_loss(multi_logits(classifiers, model_idx, grad_fn(x + d)), y)
            (g,) = torch.autograd.grad(loss.sum(), d)
            g = g / _norms(g).clamp_min(1e-12).view(view)
            d = d.detach() + alpha * g
            d = d * (eps / _norms(d).clamp_min(1e-12)).clamp(max=1.0).view(view)
        out.append(d.detach().cpu())
    return out



def random_deltas(shape, eps: torch.Tensor, trials: int, seeds: list[int]) -> list[torch.Tensor]:
    view = (-1,) + (1,) * (len(shape) - 1)
    return [unit_directions(shape, [s * 1_000 + 500 + t for s in seeds]) * eps.cpu().view(view)
            for t in range(trials)]



def violations(rows: list[dict], rel_tol: float = 1e-3) -> dict:
    """Successful attacks strictly inside the certified radius (must be none)."""
    out = {}
    for lpre in ("official", "rigorous"):
        bad = [r for r in rows if r["pgd_success"] and r["clean_correct"]
               and r[f"radius_{lpre}"] > r["delta_norm"] * (1 + rel_tol)]
        out[lpre] = {"count": len(bad), "examples": bad[:10]}
    return out



@dataclass
class AttackConfig:
    step: int
    checkpoint: str = "headline"       # headline | best | final
    runs: list[str] = field(default_factory=list)   # run names; empty = all found
    seeds: list[int] = field(default_factory=lambda: [0])
    n_images: int = 200
    subset_seed: int = 999
    eps: list[float] = field(default_factory=lambda: list(certify.EPS_REPORT))
    steps: int = 20
    restarts: int = 1
    alpha_factor: float = 2.5
    random_trials: int = 4
    grad_precision: str = "fp32"       # fp32 | fp16 (resolved before the run)
    # stronger attacks (strong.py); `steps` is the APGD iteration count
    attack: str = "pgd"                # pgd | apgd-ce | apgd-dlr | apgd-t | square
    target_classes: int = 0            # apgd-t: highest non-true classes to target, 0 = all
    queries: int = 5000                # square: forward passes per instance and restart
    p_init: float = 0.8                # square: initial window size (fraction of pixels)

    def key(self) -> dict:
        k = asdict(self)
        if self.attack == "pgd":           # the PGD folders were made before these fields
            for f in STRONG_FIELDS:
                k.pop(f)
        return k


STRONG_FIELDS = ("attack", "target_classes", "queries", "p_init")



def attack_deltas(cfg: AttackConfig, grad_fn, classifiers, model_idx, x, y, eps,
                  seeds: list[int]) -> list[torch.Tensor]:
    """Phase A for cfg.attack: one perturbation per restart and row (CPU)."""
    if cfg.attack == "pgd":
        return pgd_deltas(grad_fn, classifiers, model_idx, x, y, eps, steps=cfg.steps,
                          restarts=cfg.restarts, alpha_factor=cfg.alpha_factor, seeds=seeds)
    from . import strong

    return strong.deltas(cfg, grad_fn, classifiers, model_idx, x, y, eps, seeds)



def select_images(n_total: int, n: int, seed: int) -> list[int]:
    g = torch.Generator().manual_seed(seed)
    return sorted(torch.randperm(n_total, generator=g)[:min(n, n_total)].tolist())



def _read_progress(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass                      # a line cut by a crash: its instance reruns
    return rows



def _append_progress(path: Path, rows: list[dict]) -> None:
    # A crash can leave half a line at the end; never glue new rows onto it.
    if path.exists() and path.stat().st_size:
        with open(path, "rb") as h:
            h.seek(-1, 2)
            needs_newline = h.read(1) != b"\n"
        if needs_newline:
            with open(path, "a", encoding="utf-8") as h:
                h.write("\n")
    with open(path, "a", encoding="utf-8") as h:
        for r in rows:
            h.write(json.dumps(r) + "\n")
        h.flush()



def run_csv_name(attack: str, which: str) -> str:
    """Per-run copy of the rows: pgd_l2_<which>.csv for PGD (as before)."""
    return layout.run_attack_csv(attack, which)



def hash_seed(name: str, seed: int, image: str, eps_idx: int) -> int:
    h = hashlib.sha256(f"{name}|{seed}|{image}|{eps_idx}".encode()).hexdigest()
    return int(h[:12], 16)



def network_functions(cfg, device, strategy: str, cache_bro_weights: bool, eval_chunk: int,
                      batch_size: int, timer, logger=None, feature_fn=None, backend=None):
    """(grad_fn, eval_fn): an injected `feature_fn(x, grad)` (tests, ResNet) or the
    network of the backend (BRONet)."""
    if feature_fn is not None:
        def grad_fn(x):
            return feature_fn(x, True)

        def eval_fn(X):
            with torch.no_grad():
                return torch.cat([feature_fn(X[i:i + eval_chunk].to(device), False).cpu()
                                  for i in range(0, len(X), eval_chunk)])
        return grad_fn, eval_fn
    with timer.section(f"load {backend.name}"):
        return backend.network_functions(cfg, device, strategy, cache_bro_weights, eval_chunk,
                                         batch_size, logger)



def clean_pass(targets: list[dict], eval_fn, images: dict, device) -> tuple[list[dict], dict]:
    """Exact end-to-end clean logits: per target {correct, r_off, r_rig} on the images,
    and the max |logit difference| against the ones saved at training time."""
    feats = eval_fn(images["x"])
    clean, agree = [], {}
    for t in targets:
        with torch.no_grad():
            z = t["clf"](feats.to(device)).cpu()
        head_w = t["clf"].head.get_weight().detach().cpu()
        r_off = certify.radii(z, head_w, t["L_official"])
        r_rig = certify.radii(z, head_w, t["L_rigorous"])
        clean.append({"correct": r_off["pred"] == images["y"], "r_off": r_off["radius"],
                      "r_rig": r_rig["radius"]})
        cert_file = (Path(t["dir"]) / layout.arrays_file("certificates", t["which"])
                     if t.get("dir") else None)
        if cert_file is not None and cert_file.exists() and "test_index" in images:
            saved = np.load(cert_file)["logits"][images["test_index"]]
            agree[f"{t['name']}/seed{t['seed']}"] = float(np.abs(saved - z.numpy()).max())
    return clean, agree


def open_output(od: Path, config_name: str, key: dict, restart: bool) -> tuple[Path, Path]:
    """Create the folder of an attack (or search); with `restart` forget its progress;
    refuse to continue progress made with other settings. Returns (progress.jsonl,
    the config file)."""
    od.mkdir(parents=True, exist_ok=True)
    prog_path, cfg_path = od / "progress.jsonl", od / config_name
    if restart:
        for p in (prog_path, cfg_path):
            if p.exists():
                p.unlink()
    if cfg_path.exists() and read_json(cfg_path)["config"] != key:
        raise RuntimeError(f"{cfg_path} holds other settings; pass --restart to discard the "
                           "progress, or use the same flags")
    return prog_path, cfg_path


def _attack_chunk(cfg: AttackConfig, chunk: list[tuple[int, int, int]], targets, clfs,
                  images: dict, clean, grad_fn, eval_fn, device, batch_size: int, timer,
                  phase_a: str) -> list[dict]:
    """Both phases for a chunk of (target, image, eps) instances; their progress rows."""
    x_all, y_all = images["x"], images["y"]
    seeds = [hash_seed(targets[ti]["name"], targets[ti]["seed"], images["names"][ii], ei)
             for ti, ii, ei in chunk]
    eps_c = torch.tensor([cfg.eps[ei] for _, _, ei in chunk], dtype=torch.float32)
    ti_c = torch.tensor([q[0] for q in chunk])
    ii_c = torch.tensor([q[1] for q in chunk])
    # phase A: attack directions (one per restart)
    with timer.section(phase_a):
        pgd = [[] for _ in range(cfg.restarts)]
        for b in range(0, len(chunk), batch_size):
            sl = slice(b, b + batch_size)
            ds = attack_deltas(cfg, grad_fn, clfs, ti_c[sl].to(device),
                               x_all[ii_c[sl]].to(device), y_all[ii_c[sl]].to(device),
                               eps_c[sl].to(device), seeds[sl])
            for r, d in enumerate(ds):
                pgd[r].append(d)
        pgd = [torch.cat(p) for p in pgd]
    rnd = random_deltas((len(chunk),) + tuple(x_all.shape[1:]), eps_c, cfg.random_trials,
                        seeds)
    # phase B: exact evaluation of every perturbed image
    with timer.section("phase B (exact evaluation)"):
        x_c = x_all[ii_c]
        y_c = y_all[ii_c]
        deltas = pgd + rnd
        X = torch.cat([x_c + d for d in deltas])
        f = eval_fn(X)
        with torch.no_grad():
            z = multi_logits(clfs, ti_c.repeat(len(deltas)).to(device), f.to(device)).cpu()
        m = margin_loss(z, y_c.repeat(len(deltas))).view(len(deltas), len(chunk))
    m_pgd, m_rnd = m[:cfg.restarts], m[cfg.restarts:]
    success = (m_pgd > 0).any(0)
    first = torch.where(m_pgd > 0, torch.arange(cfg.restarts)[:, None],
                        cfg.restarts).amin(0)
    best = m_pgd.argmax(0)
    kept = torch.where(success, first.clamp(max=cfg.restarts - 1), best)
    norms = torch.stack([_norms(p) for p in pgd])[kept, torch.arange(len(chunk))]
    margin = m_pgd[kept, torch.arange(len(chunk))]
    survived = (m_rnd <= 0).all(0) if cfg.random_trials else torch.ones(len(chunk), dtype=torch.bool)
    return [_row(targets[ti], images["names"][ii], cfg.eps[ei], clean[ti], ii,
                 bool(success[k]), bool(survived[k]), float(norms[k]), float(margin[k]))
            for k, (ti, ii, ei) in enumerate(chunk)]


def run(cfg: AttackConfig, *, device, strategy: str, batch_size: int = 32,
        eval_chunk: int = 256, super_chunk: int = 512, cache_bro_weights: bool = False,
        restart: bool = False, logger=None, feature_fn=None, targets=None, images=None,
        out: Path | None = None, train_batch_size: int = layout.DEFAULT_BATCH_SIZE,
        backend=None) -> dict:
    """The full attack. `feature_fn(x, grad)`, `targets`, `images` and `out` are
    injected by the tests and resnet_adapt; otherwise the backend (BRONet) gives the
    folder, the trained runs, the test images and the network."""
    log = logger.info if logger else print
    from .strong import ATTACKS

    if cfg.attack not in ATTACKS:
        raise ValueError(f"attack must be one of {ATTACKS}")
    od = Path(out) if out is not None else backend.out_dir(cfg.step, cfg.checkpoint,
                                                           train_batch_size, cfg.attack)
    prog_path, cfg_path = open_output(od, "attack_config.json", cfg.key(), restart)
    timer = Timer(logger)

    # --- targets, images, the network in front of the classifiers ---------------------------
    if targets is None:
        targets = backend.load_targets(cfg, device, train_batch_size)
    names = [f"{t['name']}/seed{t['seed']}" for t in targets]
    clfs = [t["clf"] for t in targets]
    log(f"attack targets: {names}")
    if images is None:
        images = backend.load_test_images(cfg)
    n_img = len(images["y"])
    grad_fn, eval_fn = network_functions(cfg, device, strategy, cache_bro_weights, eval_chunk,
                                         batch_size, timer, logger, feature_fn, backend)

    write_json(cfg_path, {"config": cfg.key(), "targets": names,
                          "images": images["names"], "device": device_info(device),
                          "strategy": strategy, "batch_size": batch_size,
                          "eval_chunk": eval_chunk})

    # --- clean pass, end to end (exact) ------------------------------------------------------
    with timer.section("clean pass"):
        clean, agree = clean_pass(targets, eval_fn, images, device)
    if agree:
        log(f"clean logits end-to-end vs training (cached phi): max |diff| "
            f"{max(agree.values()):.2e}")

    # --- what is left to do ----------------------------------------------------------------
    done = {(r["run"], r["seed"], r["image"], r["eps"]) for r in _read_progress(prog_path)}
    todo = [(ti, ii, ei) for ti, t in enumerate(targets) for ii in range(n_img)
            for ei, e in enumerate(cfg.eps)
            if (t["name"], t["seed"], images["names"][ii], e) not in done]
    wrong = [q for q in todo if not clean[q[0]]["correct"][q[1]]]
    work = [q for q in todo if clean[q[0]]["correct"][q[1]]]
    log(f"{len(done):,} instances done; to do {len(work):,} attacks "
        f"(+{len(wrong):,} misclassified, recorded without attack); "
        f"{cfg.attack} batch {batch_size}, gradients {cfg.grad_precision}, "
        f"exact eval chunk {eval_chunk}")
    _append_progress(prog_path, [_row(targets[ti], images["names"][ii], cfg.eps[ei], clean[ti],
                                      ii, False, False, 0.0, float("nan"))
                                 for ti, ii, ei in wrong])

    phase_a = "phase A (PGD directions)" if cfg.attack == "pgd" else f"phase A ({cfg.attack})"
    t0 = time.perf_counter()
    for c0 in range(0, len(work), super_chunk):
        chunk = work[c0:c0 + super_chunk]
        _append_progress(prog_path, _attack_chunk(cfg, chunk, targets, clfs, images, clean,
                                                  grad_fn, eval_fn, device, batch_size, timer,
                                                  phase_a))
        el = time.perf_counter() - t0
        n_done = c0 + len(chunk)
        log(f"  {n_done:,}/{len(work):,} attacked  {el / n_done:.2f} s/instance  "
            f"ETA {(len(work) - n_done) * el / n_done / 60:.1f} min  "
            f"(A {timer.seconds.get(phase_a, 0):.0f}s, "
            f"B {timer.seconds.get('phase B (exact evaluation)', 0):.0f}s so far)")

    keyset = {(t["name"], t["seed"]) for t in targets}
    all_rows = [r for r in _read_progress(prog_path) if (r["run"], r["seed"]) in keyset]
    viol = violations(all_rows)
    sanity = {"violations": viol, "clean_logits_max_abs_diff_vs_training": agree,
              "n_rows": len(all_rows), "seconds": timer.seconds,
              "ok": viol["official"]["count"] == 0 and viol["rigorous"]["count"] == 0}
    write_json(od / "sanity_check.json", sanity)
    summary = summarise(all_rows)
    write_summary(od, cfg, summary, sanity)
    for t in targets:
        if t.get("dir") and out is None:
            mine = [r for r in all_rows if (r["run"], r["seed"]) == (t["name"], t["seed"])]
            _write_csv(Path(t["dir"]) / "attacks" / run_csv_name(cfg.attack, t["which"]), mine)
    log(f"violations: official {viol['official']['count']}, rigorous "
        f"{viol['rigorous']['count']}  -> {od}")
    return {"summary": summary, "sanity": sanity, "rows": all_rows}



def _row(t, image, eps, clean, ii, success, random_survived, delta_norm, margin) -> dict:
    correct = bool(clean["correct"][ii])
    return {"run": t["name"], "seed": t["seed"], "image": image, "eps": eps,
            "clean_correct": correct,
            "radius_official": float(clean["r_off"][ii]),
            "radius_rigorous": float(clean["r_rig"][ii]),
            "certified_official": correct and float(clean["r_off"][ii]) > eps,
            "certified_rigorous": correct and float(clean["r_rig"][ii]) > eps,
            "random_survived": correct and random_survived,
            "pgd_success": success, "pgd_robust": correct and not success,
            "delta_norm": delta_norm, "adv_margin": margin}



def summarise(rows: list[dict]) -> list[dict]:
    out = []
    for run_name, seed, eps in sorted({(r["run"], r["seed"], r["eps"]) for r in rows}):
        rs = [r for r in rows if (r["run"], r["seed"], r["eps"]) == (run_name, seed, eps)]
        succ = [r["delta_norm"] for r in rs if r["pgd_success"] and r["clean_correct"]]
        name = next((nm for nm, e in zip(EPS_NAMES, certify.EPS_REPORT) if abs(e - eps) < 1e-9),
                    f"{eps:.4f}")
        mean = lambda k: 100 * float(np.mean([r[k] for r in rs]))
        out.append({"run": run_name, "seed": seed, "epsilon": eps, "epsilon_name": name,
                    "n": len(rs), "clean_accuracy": mean("clean_correct"),
                    "certified_official": mean("certified_official"),
                    "certified_rigorous": mean("certified_rigorous"),
                    "random_accuracy": mean("random_survived"),
                    "pgd_robust_accuracy": mean("pgd_robust"),
                    "pgd_success_mean_delta": float(np.mean(succ)) if succ else float("nan"),
                    "gap_pgd_minus_certified": mean("pgd_robust") - mean("certified_official")})
    return out



_write_csv = write_csv          # the name the attack modules have always used



def attack_text(cfg: AttackConfig) -> str:
    """One sentence on the attack of this folder, for the summary."""
    if cfg.attack == "pgd":
        return (f"PGD-L2: {cfg.steps} steps, {cfg.restarts} restart(s), alpha = "
                f"{cfg.alpha_factor} eps / steps, no clamping to [0,1], gradients in "
                f"{cfg.grad_precision}")
    if cfg.attack == "square":
        return (f"Square-L2 (score-based random search, no gradients): {cfg.queries} queries "
                f"per instance, p_init = {cfg.p_init}, {cfg.restarts} restart(s), no clamping "
                f"to [0,1], search on the {cfg.grad_precision} network")
    loss = {"apgd-ce": "cross-entropy", "apgd-dlr": "DLR loss",
            "apgd-t": ("targeted DLR (targeted margin with < 4 classes), "
                       + ("every other class" if cfg.target_classes <= 0
                          else f"{cfg.target_classes} highest non-true classes")
                       + " as targets")}[cfg.attack]
    return (f"{cfg.attack.upper()}-L2 ({loss}): {cfg.steps} iterations, {cfg.restarts} "
            f"restart(s), adaptive step (2 eps at the start), momentum 0.75, best iterate "
            f"kept, no clamping to [0,1], gradients in {cfg.grad_precision}")



def write_summary(od: Path, cfg: AttackConfig, summary: list[dict], sanity: dict) -> None:
    from .strong import LABELS

    name = LABELS[cfg.attack]
    _write_csv(od / layout.attack_summary_file(cfg.step, "csv"), summary)
    lines = [f"# End-to-end L2 attacks, step-{cfg.step} ({cfg.checkpoint} checkpoint)"
             + ("" if cfg.attack == "pgd" else f" — {name}"), "",
             f"Subset of {summary[0]['n'] if summary else 0} test images (seed "
             f"{cfg.subset_seed}), the same for every model. {attack_text(cfg)}; every result "
             f"re-evaluated exactly in float32. Random: {cfg.random_trials} directions of norm "
             f"eps. Accuracies in % of the subset. Certified ⊆ {name}-robust must hold "
             "(checked image by image).", "",
             f"**Sanity check: {'OK' if sanity['ok'] else 'VIOLATED'}** — successful attacks "
             f"inside the certified radius: official {sanity['violations']['official']['count']}, "
             f"rigorous {sanity['violations']['rigorous']['count']}.", "",
             f"| run | seed | ε | clean | certified | certified (rig.) | random | {name}-robust "
             f"| {name} − certified | mean ‖δ‖ of successes |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in summary:
        lines.append(f"| {label(r['run'])} | {r['seed']} | {r['epsilon_name']} | "
                     f"{r['clean_accuracy']:.1f} | {r['certified_official']:.1f} | "
                     f"{r['certified_rigorous']:.1f} | {r['random_accuracy']:.1f} | "
                     f"{r['pgd_robust_accuracy']:.1f} | {r['gap_pgd_minus_certified']:.1f} | "
                     f"{r['pgd_success_mean_delta']:.3f} |")
    lines += ["", f"{name} − certified: empirical robustness that the certificate does not prove "
              "(an upper bound on its looseness against this attack). Random noise probes "
              f"the average case, {name} the worst case."]
    if cfg.attack != "pgd":
        lines += ["", f"The columns `pgd_*` of the CSV files hold the {name} results (the "
                  "names are shared with the PGD folders, so that plots and reports read both)."]
    atomic_write_text(od / layout.attack_summary_file(cfg.step, "md"), "\n".join(lines) + "\n")



def add_attack_args(p: argparse.ArgumentParser) -> None:
    """--attack and its settings (shared with resnet_adapt.attacks)."""
    from .strong import ATTACKS, DEFAULT_STEPS

    p.add_argument("--attack", choices=ATTACKS, default="pgd",
                   help="pgd (default; the original folders) | apgd-ce | apgd-dlr | apgd-t | "
                        "square; every attack but pgd writes to its own folder ..._<attack>")
    p.add_argument("--steps", type=int, default=None,
                   help="PGD steps / APGD iterations (default: "
                        + ", ".join(f"{k} {v}" for k, v in DEFAULT_STEPS.items() if v) + ")")
    p.add_argument("--restarts", type=int, default=1)
    p.add_argument("--target-classes", type=int, default=0,
                   help="apgd-t: target the k highest non-true classes (0 = all other classes)")
    p.add_argument("--queries", type=int, default=5000,
                   help="square: forward passes per instance and restart (AutoAttack: 5000)")
    p.add_argument("--p-init", type=float, default=0.8,
                   help="square: initial window size as a fraction of the pixels (AutoAttack: 0.8)")



def attack_settings(a) -> dict:
    """AttackConfig fields from add_attack_args' flags."""
    from .strong import DEFAULT_STEPS

    steps = a.steps if a.steps is not None else (DEFAULT_STEPS[a.attack] or 20)
    out = {"steps": steps, "restarts": a.restarts}
    if a.attack != "pgd":
        out.update(attack=a.attack, target_classes=a.target_classes, queries=a.queries,
                   p_init=a.p_init)
    return out
