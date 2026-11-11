"""End-to-end L2 attacks on the adapted ResNet152 models, as `bronet_mlp.attacks`.

    python -m experiments.resnet_adapt.attacks --step 2 --batch-size 32
    python -m experiments.resnet_adapt.attacks --step 1 --batch-size 32
    python -m experiments.resnet_adapt.attacks --plots-only --batch-size 32
    python -m experiments.resnet_adapt.attacks --step 2 --batch-size 32 --attack apgd-ce

The attack itself is `experiments.common.attacks.core.run` (shared with BRONet): PGD-L2 with
`--steps` steps of size 2.5 eps / steps along the normalised gradient of the
margin loss, `--restarts` random starts on the sphere of radius eps, projection
onto the ball, no clamping to [0, 1]; `--random-trials` random directions of
norm eps; only clean-correct images are attacked; every perturbed image is
re-evaluated exactly in a second pass; resumable (progress.jsonl).

What is ResNet-specific here:

* the network in front of the classifiers is the frozen `backbone.Front`
  (ImageNet normalisation inside the model), so eps is in pixels in [0, 1] of
  the 224x224 input, exactly as for BRONet, and gradients flow through it in
  float32 (TF32 off);
* targets: every run of the protocol (`--batch-size` = the training batch,
  folders step_<n>[_bs<B>]), seed 0, checkpoint `best` (best validation epoch)
  by default, `final` for the last epoch;
* the same random test subset as BRONet (subset seed 999; 300 images on step-1,
  all 240 on step-2 by default), so the two networks are attacked on the same
  images;
* the Lipschitz constant for the mandatory sanity check (no success inside the
  certified radius) is the rigorous upper bound of `spectrum.py` (~10^109), so
  every radius is ~0 and the check cannot fail; it is kept for symmetry;
* clean logits end to end are compared with the ones saved at training time
  (from the float16 feature cache): clean_agreement.json;
* `--attack apgd-ce | apgd-dlr | apgd-t | square` (experiments/common/attacks/strong.py)
  replaces PGD, with the flags of bronet_mlp.attacks (`--steps` = APGD
  iterations, default 100; `--queries`, `--p-init`, `--target-classes`); results
  go to ..._<checkpoint>[_eps...]_<attack>/ and the per-run file
  <attack>_l2_<checkpoint>.csv, figures get the suffix _<attack>.

Outputs, <DERMA_REPRO_OUT>/resnet_adapt/results/attacks/step_<n>[_bs<B>]_<checkpoint>/:
attack_config.json, progress.jsonl, sanity_check.json, clean_agreement.json,
attacks_step<n>.csv / .md; per run <run dir>/attacks/pgd_l2_<checkpoint>.csv;
figures/attacks_step<n>[_bs<B>]_<checkpoint>.png (and both steps together).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import paths

import numpy as np
import torch

from experiments.common import data as dataset
from experiments.common.attacks import core as attack_core
from experiments.common.attacks import plots as attack_plots
from experiments.common.display import STEP_TITLE
from repro import seeding
from experiments.common.device import resolve_device, device_info
from experiments.common.io import read_json, write_json, setup_logging
from experiments.common import layout

from . import backbone, report, spectrum, train

DEFAULT_N_IMAGES = {1: 300, 2: 240}       # as the BRONet attacks
DEFAULT_EPS_255 = (36, 72, 108)            # as the BRONet attacks (in units of 1/255)


def eps_tag(eps_255) -> str:
    """'' for the BRONet epsilons, else '_eps<a>-<b>-...' (units of 1/255)."""
    eps_255 = tuple(int(e) for e in eps_255)
    return "" if eps_255 == DEFAULT_EPS_255 else "_eps" + "-".join(map(str, eps_255))


def out_dir(step: int, batch_size: int, checkpoint: str, eps_255=DEFAULT_EPS_255,
            attack: str = "pgd") -> Path:
    return layout.RESNET.attack_dir(step, batch_size, checkpoint, attack, eps_tag(eps_255))


def build_targets(step: int, batch_size: int, checkpoint: str, names: list[str],
                  seeds: list[int], net, device, logger=None) -> list[dict]:
    """Classifiers of the finished runs (frozen, on `device`) with their L bound."""
    log = logger.info if logger else print
    front = spectrum.front_bound_cached(net, logger)
    runs = report.load_runs(step, checkpoint, batch_size)
    order = list(report.RUN_NAMES)
    targets = []
    for (name, seed), r in sorted(runs.items(), key=lambda kv: (order.index(kv[0][0]), kv[0][1])):
        if (names and name not in names) or seed not in seeds:
            continue
        clf = train.load_run(r["dir"], device=device, which=checkpoint, net=net)
        for p in clf.parameters():
            p.requires_grad_(False)
        log10_L = spectrum.total_bound(front, clf.block)["log10_L"]
        L = 10.0 ** log10_L
        targets.append({"name": name, "seed": seed, "dir": r["dir"], "which": checkpoint,
                        "clf": clf, "L_official": L, "L_rigorous": L, "log10_L": log10_L})
        log(f"target {name}/seed{seed} ({checkpoint}): L <= 10^{log10_L:.3f}")
    if not targets:
        raise FileNotFoundError(f"no finished runs under {paths.runs_dir(step, batch_size)} "
                                f"for {names or 'any run'}, seeds {seeds}")
    return targets


def test_images(step: int, n_images: int, subset_seed: int) -> dict:
    """The fixed random subset of the test set (same rule and seed as BRONet)."""
    _, test, _ = dataset.records(step)
    idx = attack_core.select_images(len(test), n_images, subset_seed)
    return {"x": dataset.load_images([test[i].path for i in idx]),
            "y": torch.tensor([test[i].label for i in idx]),
            "names": [test[i].path.name for i in idx], "test_index": idx}


@torch.no_grad()
def clean_agreement(targets: list[dict], front, images: dict, device, chunk: int = 32) -> dict:
    """End-to-end clean logits (float32 front) vs the logits saved at training time
    (classifier on the float16 feature cache), on the attacked subset."""
    h = torch.cat([front(images["x"][i:i + chunk].to(device)).cpu()
                   for i in range(0, len(images["x"]), chunk)])
    out = {}
    for t in targets:
        z = t["clf"](h.to(device)).cpu().numpy()
        saved = np.load(Path(t["dir"]) / layout.arrays_file("outputs", t["which"]))["logits"][
            images["test_index"]]
        out[f"{t['name']}/seed{t['seed']}"] = {
            "max_abs_diff": float(np.abs(z - saved).max()),
            "max_abs_logit": float(np.abs(saved).max()),
            "prediction_flips": int((z.argmax(1) != saved.argmax(1)).sum())}
    return out


def plot(step: int, batch_size: int, checkpoint: str, eps_255=DEFAULT_EPS_255,
         attack: str = "pgd") -> Path | None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    od = out_dir(step, batch_size, checkpoint, eps_255, attack)
    if not (od / "progress.jsonl").exists():
        return None
    cfg = read_json(od / "attack_config.json")
    wanted = set(cfg["targets"])
    rows = [r for r in attack_core._read_progress(od / "progress.jsonl")
            if f"{r['run']}/seed{r['seed']}" in wanted]
    cv = attack_plots.curves(rows, "official")
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), sharey=True)
    attack_plots._panels(axes, cv, cfg["config"]["eps"], attack)
    h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=9, frameon=False)
    n = next(iter(cv.values()))["n"]
    fig.suptitle(f"ResNet152, layer4.2 adapted — {STEP_TITLE[step]} — {n} test images, seed 0, "
                 f"{checkpoint} checkpoint (certified: L ≤ 10^109, radii ≈ 0)", fontsize=11)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    out = (paths.results_dir() / "figures"
           / layout.attack_figure(step, batch_size, checkpoint, attack, eps_tag(eps_255)))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_both(batch_size: int, checkpoint: str, eps_255=DEFAULT_EPS_255,
              attack: str = "pgd") -> Path | None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    got = {}
    for s in (1, 2):
        od = out_dir(s, batch_size, checkpoint, eps_255, attack)
        if not (od / "progress.jsonl").exists():
            return None
        cfg = read_json(od / "attack_config.json")
        rows = [r for r in attack_core._read_progress(od / "progress.jsonl")
                if f"{r['run']}/seed{r['seed']}" in set(cfg["targets"])]
        got[s] = (rows, cfg)
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.8), sharey=True)
    for row, s in enumerate((1, 2)):
        rows, cfg = got[s]
        cv = attack_plots.curves(rows, "official")
        attack_plots._panels(axes[row], cv, cfg["config"]["eps"], attack)
        n = next(iter(cv.values()))["n"]
        axes[row][0].set_ylabel(f"{STEP_TITLE[s]}\n({n} test images)\naccuracy")
    seen = {}
    for ax in (axes[0][1], axes[1][1]):              # both rows: step-2 has the stable variant
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(ll, hh)
    fig.legend(list(seen.values()), list(seen), loc="lower center", ncol=3, fontsize=9, frameon=False)
    fig.suptitle(f"After adapting a standard backbone (ResNet152, ImageNet) to trichoscopy — "
                 f"seed 0, {checkpoint} checkpoint", fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out = (paths.results_dir() / "figures"
           / layout.attack_figure(None, batch_size, checkpoint, attack, eps_tag(eps_255)))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main(argv=None) -> int:
    from repro import console

    console.init()
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--step", type=int, choices=(1, 2))
    p.add_argument("--train-batch-size", "--batch-size", dest="batch_size", type=int,
                   default=paths.DEFAULT_BATCH_SIZE,
                   help="training batch of the protocol whose runs are attacked (step_<n>_bs<B>; "
                        "--batch-size: alias)")
    p.add_argument("--checkpoint", choices=("best", "final"), default="best")
    p.add_argument("--runs", nargs="*", default=[],
                   help="run names (e.g. butterfly full); default: every finished run")
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--n-images", type=int, default=None,
                   help="test images (default: 300 on step-1, 240 on step-2, as BRONet)")
    p.add_argument("--eps", nargs="+", type=int, default=list(DEFAULT_EPS_255),
                   help="L2 radii in units of 1/255 (default 36 72 108, as BRONet); other "
                        "sets go to their own folder ..._eps<a>-<b>-..., e.g. 2 4 9 18")
    attack_core.add_attack_args(p)
    p.add_argument("--random-trials", type=int, default=4)
    p.add_argument("--attack-batch", type=int, default=16,
                   help="PGD instances (image x model x eps) per batch; lower on OOM")
    p.add_argument("--eval-chunk", type=int, default=64,
                   help="images per exact re-evaluation pass; lower on OOM")
    p.add_argument("--super-chunk", type=int, default=256,
                   help="instances between progress saves")
    p.add_argument("--device", default="auto")
    p.add_argument("--restart", action="store_true", help="discard saved progress")
    p.add_argument("--plots-only", action="store_true")
    p.add_argument("--out", type=Path, default=None, help="other output folder (smoke tests)")
    a = p.parse_args(argv)

    if a.plots_only:
        made = [plot(s, a.batch_size, a.checkpoint, a.eps, a.attack)
                for s in ((a.step,) if a.step else (1, 2))]
        made.append(plot_both(a.batch_size, a.checkpoint, a.eps, a.attack))
        for m in made:
            if m:
                print(f"-> {m}")
        return 0
    if a.step is None:
        p.error("--step is required (except with --plots-only)")

    od = a.out or out_dir(a.step, a.batch_size, a.checkpoint, a.eps, a.attack)
    logger = setup_logging(od / "attacks.log", name="resnet_adapt.attacks")
    device = resolve_device(a.device)
    seeding.configure_determinism(True)
    n_images = a.n_images or DEFAULT_N_IMAGES[a.step]
    logger.info(f"device {device_info(device)}; step-{a.step}{paths.tag(a.batch_size)}, "
                f"{a.checkpoint} checkpoint, {n_images} images, attack {a.attack}")

    net = backbone.load_resnet152()
    targets = build_targets(a.step, a.batch_size, a.checkpoint, a.runs, a.seeds, net, device,
                            logger)
    front = backbone.Front(net).to(device).eval()
    images = test_images(a.step, n_images, attack_core.AttackConfig(step=a.step).subset_seed)
    agree = clean_agreement(targets, front, images, device)
    write_json(od / "clean_agreement.json", agree)
    flips = sum(v["prediction_flips"] for v in agree.values())
    logger.info(f"clean logits end-to-end vs training (float16 cache): max |diff| "
                f"{max(v['max_abs_diff'] for v in agree.values()):.2e}, prediction flips {flips}")

    def feature_fn(x, grad):
        return front(x)

    cfg = attack_core.AttackConfig(step=a.step, checkpoint=a.checkpoint,
                                runs=sorted({t["name"] for t in targets}), seeds=a.seeds,
                                n_images=n_images, eps=[e / 255 for e in a.eps],
                                random_trials=a.random_trials, grad_precision="fp32",
                                **attack_core.attack_settings(a))
    try:
        res = attack_core.run(cfg, device=device, strategy="resnet", batch_size=a.attack_batch,
                           eval_chunk=a.eval_chunk, super_chunk=a.super_chunk,
                           restart=a.restart, logger=logger, feature_fn=feature_fn,
                           targets=targets, images=images, out=od)
    except torch.cuda.OutOfMemoryError as exc:
        raise RuntimeError("CUDA out of memory: lower --attack-batch and/or --eval-chunk "
                           "(progress is saved)") from exc
    if a.out is None:
        for t in targets:
            mine = [r for r in res["rows"] if (r["run"], r["seed"]) == (t["name"], t["seed"])]
            attack_core._write_csv(Path(t["dir"]) / "attacks"
                                / attack_core.run_csv_name(a.attack, a.checkpoint), mine)
        for fig in (plot(a.step, a.batch_size, a.checkpoint, a.eps, a.attack),
                    plot_both(a.batch_size, a.checkpoint, a.eps, a.attack)):
            if fig:
                logger.info(f"-> {fig}")
    if not res["sanity"]["ok"]:
        logger.error("a successful attack lies inside the certified radius -- see "
                     "sanity_check.json; this is a bug")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
