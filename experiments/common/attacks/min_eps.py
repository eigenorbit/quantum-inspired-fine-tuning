"""The eps*(x) search: the smallest perturbation the attack finds that flips an image.

Independent of the network (an injected `feature_fn`, or a `core.Backend`); the
method, the outputs and the sanity check: experiments/bronet_mlp/min_eps.py.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from .. import certify, layout
from ..device import device_info
from ..display import label
from ..io import Timer, atomic_write_text, write_json
from .core import (AttackConfig, EPS_NAMES, _append_progress, _norms, _read_progress, _write_csv,
                   add_attack_args, attack_deltas, attack_settings, attack_text, clean_pass,
                   hash_seed, margin_loss, multi_logits, network_functions, open_output)


R_TINY = 1e-6          # below this r is vacuous: no check at 0.99 r, tightness not reported


REL_TOL = 1e-3         # as attacks.violations



@dataclass
class MinEpsConfig:
    step: int
    checkpoint: str = "headline"
    runs: list[str] = field(default_factory=list)
    seeds: list[int] = field(default_factory=lambda: [0])
    n_images: int = 200
    subset_seed: int = 999
    # the attack of every probe (as AttackConfig)
    attack: str = "pgd"
    steps: int = 20
    restarts: int = 1
    alpha_factor: float = 2.5
    target_classes: int = 0
    queries: int = 5000
    p_init: float = 0.8
    grad_precision: str = "fp32"
    # the search
    eps_floor: float = 1 / 255         # first upward probe: max(2 r, eps_floor)
    eps_max: float = 1020 / 255        # no flip up to here -> censored
    bisect_steps: int = 8
    check_below_r: bool = True
    check_factor: float = 0.99

    def key(self) -> dict:
        return asdict(self)

    def attack_cfg(self) -> AttackConfig:
        return AttackConfig(step=self.step, steps=self.steps, restarts=self.restarts,
                            alpha_factor=self.alpha_factor, grad_precision=self.grad_precision,
                            attack=self.attack, target_classes=self.target_classes,
                            queries=self.queries, p_init=self.p_init)



def probe(acfg: AttackConfig, grad_fn, eval_fn, clfs, ti, x, y, eps, seeds, device,
          batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """One attack per row at its own eps. Returns (success, ||delta|| of the smallest
    successful restart, inf where none), both decided by the exact eval_fn."""
    per_restart = [[] for _ in range(acfg.restarts)]
    for b in range(0, len(y), batch_size):
        sl = slice(b, b + batch_size)
        ds = attack_deltas(acfg, grad_fn, clfs, ti[sl].to(device), x[sl].to(device),
                           y[sl].to(device), eps[sl].to(device), seeds[sl])
        for r, d in enumerate(ds):
            per_restart[r].append(d)
    deltas = [torch.cat(p) for p in per_restart]
    f = eval_fn(torch.cat([x + d for d in deltas]))
    with torch.no_grad():
        z = multi_logits(clfs, ti.repeat(len(deltas)).to(device), f.to(device)).cpu()
    m = margin_loss(z, y.repeat(len(deltas))).view(len(deltas), len(y))
    norms = torch.stack([_norms(d) for d in deltas])
    norms = torch.where(m > 0, norms, torch.full_like(norms, math.inf))
    return (m > 0).any(0), norms.amin(0)



def search_chunk(cfg: MinEpsConfig, grad_fn, eval_fn, clfs, ti, x, y, r_off, seeds_base,
                 device, batch_size: int, log=None) -> list[dict]:
    """eps* for a chunk of clean-correct instances (rows of x, model ti, radius r_off).
    `seeds_base[k]` is the instance seed; probe round j uses seed + j."""
    n = len(y)
    acfg = cfg.attack_cfg()
    st = [{"lo": float(r_off[k]), "hi": math.inf, "next": max(2 * float(r_off[k]), cfg.eps_floor),
           "phase": "bracket", "bisect": 0, "probes": 0, "check_flip": None,
           "censored": False, "nonmonotone": False} for k in range(n)]
    for k, s in enumerate(st):
        if cfg.check_below_r and s["lo"] >= R_TINY:
            s["phase"] = "check"
        elif s["lo"] >= cfg.eps_max:                   # certified beyond the search range
            s["phase"], s["censored"] = "done", True
    rnd = 0
    while True:
        todo = [k for k, s in enumerate(st) if s["phase"] != "done"]
        if not todo:
            break
        eps = []
        for k in todo:
            s = st[k]
            if s["phase"] == "check":
                eps.append(cfg.check_factor * s["lo"])
            elif s["phase"] == "bracket":
                eps.append(min(s["next"], cfg.eps_max))
            else:
                eps.append(0.5 * (s["lo"] + s["hi"]))
        idx = torch.tensor(todo)
        eps_t = torch.tensor(eps, dtype=torch.float32)
        seeds = [seeds_base[k] * 1_000 + rnd for k in todo]
        ok, norm = probe(acfg, grad_fn, eval_fn, clfs, ti[idx], x[idx], y[idx], eps_t, seeds,
                         device, batch_size)
        for j, k in enumerate(todo):
            s, e = st[k], float(eps_t[j])
            s["probes"] += 1
            if ok[j]:
                s["hi"] = min(s["hi"], e, float(norm[j]))
            if s["phase"] == "check":
                s["check_flip"] = bool(ok[j])
                s["phase"] = "bracket" if not ok[j] else ("bisect" if cfg.bisect_steps else "done")
                if s["phase"] == "bracket" and s["lo"] >= cfg.eps_max:
                    s["phase"], s["censored"] = "done", True
                continue
            if s["phase"] == "bracket":
                if ok[j]:
                    s["phase"] = "bisect" if cfg.bisect_steps else "done"
                elif e >= cfg.eps_max:
                    s["lo"], s["phase"], s["censored"] = e, "done", True
                else:
                    s["lo"], s["next"] = e, 2 * e
            else:                                           # bisect
                if not ok[j]:
                    s["lo"] = max(s["lo"], e)
                s["bisect"] += 1
                if s["bisect"] >= cfg.bisect_steps:
                    s["phase"] = "done"
            if s["hi"] < s["lo"]:                          # success below an earlier failure
                s["lo"], s["nonmonotone"] = s["hi"], True
        rnd += 1
        if log:
            log(f"    round {rnd}: {len(todo):,} probes, "
                f"{sum(s['phase'] == 'done' for s in st):,}/{n:,} instances finished")
    return st



def instance_row(t: dict, image: str, clean: dict, ii: int, s: dict | None) -> dict:
    """One progress line; s is None for a misclassified image (eps* = 0)."""
    r_off, r_rig = float(clean["r_off"][ii]), float(clean["r_rig"][ii])
    row = {"run": t["name"], "seed": t["seed"], "image": image,
           "clean_correct": s is not None, "radius_official": r_off, "radius_rigorous": r_rig}
    if s is None:
        return {**row, "eps_star": 0.0, "eps_lo": 0.0, "censored": False, "probes": 0,
                "check_flip": None, "nonmonotone": False, "tightness": None,
                "tightness_rigorous": None}
    star = None if s["censored"] else s["hi"]
    return {**row, "eps_star": star, "eps_lo": s["lo"], "censored": s["censored"],
            "probes": s["probes"], "check_flip": s["check_flip"], "nonmonotone": s["nonmonotone"],
            "tightness": star / r_off if star is not None and r_off >= R_TINY else None,
            "tightness_rigorous": star / r_rig if star is not None and r_rig >= R_TINY else None}



def violations(rows: list[dict]) -> dict:
    """Instances whose flipping perturbation is strictly inside the certified radius."""
    out = {}
    for lpre in ("official", "rigorous"):
        bad = [r for r in rows if r["clean_correct"] and r["eps_star"] is not None
               and r[f"radius_{lpre}"] > r["eps_star"] * (1 + REL_TOL)]
        out[lpre] = {"count": len(bad), "examples": bad[:10]}
    out["check_below_r_flips"] = sum(bool(r.get("check_flip")) for r in rows)
    return out



def _quantile(values: list[float], q: float) -> float:
    """Quantile with censored values as +inf (so a censored median shows as inf)."""
    if not values:
        return float("nan")
    return float(np.quantile(np.array(values, dtype=float), q, method="inverted_cdf"))



def summarise(rows: list[dict], eps_max: float) -> list[dict]:
    out = []
    for run_name, seed in sorted({(r["run"], r["seed"]) for r in rows}):
        rs = [r for r in rows if (r["run"], r["seed"]) == (run_name, seed)]
        att = [r for r in rs if r["clean_correct"]]
        star = [math.inf if r["censored"] else r["eps_star"] for r in att]
        tight = [math.inf if r["censored"] else r["tightness"] for r in att
                 if r["censored"] or r["tightness"] is not None]
        row = {"run": run_name, "seed": seed, "n": len(rs),
               "clean_accuracy": 100 * len(att) / len(rs) if rs else float("nan"),
               "n_attacked": len(att), "censored": sum(r["censored"] for r in att),
               "eps_star_q25": _quantile(star, 0.25), "eps_star_median": _quantile(star, 0.5),
               "eps_star_q75": _quantile(star, 0.75),
               "radius_median": _quantile([r["radius_official"] for r in att], 0.5),
               "tightness_median": _quantile(tight, 0.5),
               "probes_mean": float(np.mean([r["probes"] for r in att])) if att else 0.0,
               "nonmonotone": sum(r["nonmonotone"] for r in att), "eps_max": eps_max}
        for name, e in zip(EPS_NAMES, certify.EPS_REPORT):
            # empirical robust accuracy at e: correct and eps* > e (censored: eps* > eps_max)
            row[f"empirical_{name}"] = 100 * sum(
                r["clean_correct"] and (r["censored"] or r["eps_star"] > e) for r in rs) / len(rs)
            row[f"certified_{name}"] = 100 * sum(
                r["clean_correct"] and r["radius_official"] > e for r in rs) / len(rs)
        out.append(row)
    return out



def _fmt(v: float, digits: int = 3) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return "> max" if math.isinf(v) else f"{v:.{digits}f}"



def write_summary(od: Path, cfg: MinEpsConfig, summary: list[dict], sanity: dict) -> None:
    from .strong import LABELS

    _write_csv(od / layout.attack_summary_file(cfg.step, "csv", "min_eps"), summary)
    v = sanity["violations"]
    lines = [f"# Empirical minimal adversarial radius, step-{cfg.step} ({cfg.checkpoint} checkpoint)",
             "",
             f"Subset of {summary[0]['n'] if summary else 0} test images (seed {cfg.subset_seed}). "
             f"Per clean-correct image: check at {cfg.check_factor} r(x)"
             + ("" if cfg.check_below_r else " (skipped)") + ", then doubling from r(x) "
             f"(first probe max(2 r, {cfg.eps_floor * 255:g}/255)) up to "
             f"{cfg.eps_max * 255:g}/255, then {cfg.bisect_steps} bisections; every probe is "
             f"{attack_text(cfg.attack_cfg())}, judged by the exact float32 re-evaluation. "
             f"eps* = norm of the smallest perturbation found that flips the image (an upper "
             f"bound on the true distance; r(x) is a lower bound). Radii L2 in the 224×224 "
             f"input; divide by 0.37458 for the 600×600 files.", "",
             f"**Sanity check: {'OK' if sanity['ok'] else 'VIOLATED'}** — flips inside the "
             f"certified radius: official {v['official']['count']}, rigorous "
             f"{v['rigorous']['count']}; flips at {cfg.check_factor} r: "
             f"{v['check_below_r_flips']}.", "",
             "| run | seed | clean | attacked | eps* median [q25, q75] | r median | eps*/r median "
             "| censored | nonmonotone | probes | " + " | ".join(
                 f"{n} {LABELS[cfg.attack]} / certified" for n in EPS_NAMES) + " |",
             "|---|---|---|---|---|---|---|---|---|---|" + "---|" * len(EPS_NAMES)]
    for r in summary:
        lines.append(
            f"| {label(r['run'])} | {r['seed']} | {r['clean_accuracy']:.1f} | {r['n_attacked']} | "
            f"{_fmt(r['eps_star_median'])} [{_fmt(r['eps_star_q25'])}, {_fmt(r['eps_star_q75'])}] | "
            f"{_fmt(r['radius_median'], 4)} | {_fmt(r['tightness_median'], 2)} | {r['censored']} | "
            f"{r['nonmonotone']} | {r['probes_mean']:.1f} | " + " | ".join(
                f"{r[f'empirical_{n}']:.1f} / {r[f'certified_{n}']:.1f}" for n in EPS_NAMES) + " |")
    lines += ["", "eps*/r ≥ 1 always (else the certificate is violated): how much larger the "
              "smallest perturbation found is than the proved radius. `> max`: the quantile "
              f"lies among censored images (no flip up to {cfg.eps_max * 255:g}/255). "
              "Empirical accuracy at ε = correct and eps* > ε; certified = correct and r > ε."]
    atomic_write_text(od / layout.attack_summary_file(cfg.step, "md", "min_eps"),
                          "\n".join(lines) + "\n")



def search(cfg: MinEpsConfig, *, device, strategy: str, batch_size: int = 32,
           eval_chunk: int = 256, super_chunk: int = 256, cache_bro_weights: bool = False,
           restart: bool = False, logger=None, feature_fn=None, targets=None, images=None,
           out: Path | None = None, train_batch_size: int = layout.DEFAULT_BATCH_SIZE,
           backend=None) -> dict:
    """The full search; injection of feature_fn / targets / images / out, or the
    backend, as `core.run`."""
    from .strong import ATTACKS

    if cfg.attack not in ATTACKS:
        raise ValueError(f"attack must be one of {ATTACKS}")
    log = logger.info if logger else print
    od = Path(out) if out is not None else backend.out_dir(cfg.step, cfg.checkpoint,
                                                           train_batch_size, cfg.attack)
    prog_path, cfg_path = open_output(od, "min_eps_config.json", cfg.key(), restart)
    timer = Timer(logger)
    if targets is None:
        targets = backend.load_targets(cfg, device, train_batch_size)
    names = [f"{t['name']}/seed{t['seed']}" for t in targets]
    clfs = [t["clf"] for t in targets]
    log(f"targets: {names}")
    if images is None:
        images = backend.load_test_images(cfg)
    x_all, y_all = images["x"], images["y"]
    grad_fn, eval_fn = network_functions(cfg, device, strategy, cache_bro_weights, eval_chunk,
                                         batch_size, timer, logger, feature_fn, backend)
    write_json(cfg_path, {"config": cfg.key(), "targets": names, "images": images["names"],
                              "device": device_info(device), "strategy": strategy,
                              "batch_size": batch_size, "eval_chunk": eval_chunk})
    with timer.section("clean pass"):
        clean, agree = clean_pass(targets, eval_fn, images, device)

    done = {(r["run"], r["seed"], r["image"]) for r in _read_progress(prog_path)}
    todo = [(ti, ii) for ti, t in enumerate(targets) for ii in range(len(y_all))
            if (t["name"], t["seed"], images["names"][ii]) not in done]
    wrong = [q for q in todo if not clean[q[0]]["correct"][q[1]]]
    work = [q for q in todo if clean[q[0]]["correct"][q[1]]]
    log(f"{len(done):,} instances done; to do {len(work):,} searches (+{len(wrong):,} "
        f"misclassified, eps* = 0); probes: {cfg.attack}, batch {batch_size}, gradients "
        f"{cfg.grad_precision}")
    _append_progress(prog_path, [instance_row(targets[ti], images["names"][ii], clean[ti], ii, None)
                                 for ti, ii in wrong])
    t0 = time.perf_counter()
    for c0 in range(0, len(work), super_chunk):
        chunk = work[c0:c0 + super_chunk]
        ti_c = torch.tensor([q[0] for q in chunk])
        ii_c = torch.tensor([q[1] for q in chunk])
        r_off = torch.tensor([float(clean[ti]["r_off"][ii]) for ti, ii in chunk])
        seeds = [hash_seed(targets[ti]["name"], targets[ti]["seed"], images["names"][ii], 777)
                 for ti, ii in chunk]
        with timer.section("search"):
            states = search_chunk(cfg, grad_fn, eval_fn, clfs, ti_c, x_all[ii_c], y_all[ii_c],
                                  r_off, seeds, device, batch_size, log)
        _append_progress(prog_path, [instance_row(targets[ti], images["names"][ii], clean[ti],
                                                  ii, s) for (ti, ii), s in zip(chunk, states)])
        el = time.perf_counter() - t0
        n_done = c0 + len(chunk)
        log(f"  {n_done:,}/{len(work):,} searched  {el / n_done:.2f} s/instance  "
            f"ETA {(len(work) - n_done) * el / n_done / 60:.1f} min")

    keyset = {(t["name"], t["seed"]) for t in targets}
    rows = [r for r in _read_progress(prog_path) if (r["run"], r["seed"]) in keyset]
    viol = violations(rows)
    sanity = {"violations": viol, "clean_logits_max_abs_diff_vs_training": agree,
              "n_rows": len(rows), "seconds": timer.seconds,
              "ok": viol["official"]["count"] == 0 and viol["rigorous"]["count"] == 0}
    write_json(od / "sanity_check.json", sanity)
    summary = summarise(rows, cfg.eps_max)
    write_summary(od, cfg, summary, sanity)
    if out is None:
        for t in targets:
            if t.get("dir"):
                mine = [r for r in rows if (r["run"], r["seed"]) == (t["name"], t["seed"])]
                _write_csv(Path(t["dir"]) / "attacks"
                           / layout.run_mineps_csv(cfg.attack, t["which"]), mine)
    log(f"violations: official {viol['official']['count']}, rigorous "
        f"{viol['rigorous']['count']}  -> {od}")
    return {"summary": summary, "sanity": sanity, "rows": rows}



def add_search_args(p: argparse.ArgumentParser) -> None:
    """The search flags (shared with resnet_adapt.min_eps)."""
    add_attack_args(p)
    p.add_argument("--eps-floor", type=float, default=1.0,
                   help="first upward probe at least this, in units of 1/255 (default 1)")
    p.add_argument("--eps-max", type=float, default=1020.0,
                   help="no flip up to this -> censored, in units of 1/255 (default 1020 = 4.0)")
    p.add_argument("--bisect-steps", type=int, default=8,
                   help="halvings after the bracket (8: eps* to 0.4 %%)")
    p.add_argument("--no-check-below-r", action="store_true",
                   help="skip the attack at 0.99 r(x) (step 0)")



def search_settings(a) -> dict:
    s = attack_settings(a)
    return {"steps": s["steps"], "restarts": s["restarts"], "attack": a.attack,
            "target_classes": a.target_classes, "queries": a.queries, "p_init": a.p_init,
            "eps_floor": a.eps_floor / 255, "eps_max": a.eps_max / 255,
            "bisect_steps": a.bisect_steps, "check_below_r": not a.no_check_below_r}
