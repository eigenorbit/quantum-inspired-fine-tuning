"""Figures of the attack results, in the style of the notebook's section 6.

    python -m experiments.bronet_mlp.plots            # every step that has attack results
    python -m experiments.bronet_mlp.plots --step 2

Per step, three panels over the perturbation norm eps (L2, 224x224 input, pixels
in [0, 1]), all on the same fixed subset of test images and the same models:

    random direction of norm eps          (average case)
    worst direction of norm eps -- PGD-L2 (empirical worst case, an upper bound)
    certified                              (proved for every direction)

Colours and labels are the notebook's (`CM`, `LM` in analysis.ipynb, cell 24);
the `stable` learning-rate variants are drawn dashed in their mode's colour. The
certified curve is exact on a fine eps grid (the radius of every image is known);
random and PGD are measured at eps = 0 (clean) and at the attacked eps.

Writes results/figures/attacks_step<n>_<checkpoint>.png and, when both steps
exist, attacks_both_steps_<checkpoint>.png (`--attack X` other than pgd: the
results of that attack, files ..._<checkpoint>_X.png).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import env
from experiments.common import layout
from experiments.common.attacks.plots import _panels, curves  # noqa: F401  (re-exported)
from experiments.common.display import CM, LM, ORDER, STEP_TITLE, base_mode, style  # noqa: F401

import numpy as np


def attack_tag(attack: str) -> str:
    from .attacks import attack_tag as tag

    return tag(attack)


def load_rows(step: int, checkpoint: str, batch_size: int = env.DEFAULT_BATCH_SIZE,
              attack: str = "pgd") -> tuple[list[dict], dict] | None:
    from .attacks import _read_progress, out_dir

    od = out_dir(step, checkpoint, batch_size, attack)
    if not (od / "progress.jsonl").exists():
        return None
    cfg = env.read_json(od / "attack_config.json")
    wanted = {t.split("/seed")[0] + "|" + t.split("/seed")[1] for t in cfg["targets"]}
    rows = [r for r in _read_progress(od / "progress.jsonl")
            if f"{r['run']}|{r['seed']}" in wanted]
    return rows, cfg


def plot_step(step: int, checkpoint: str = "headline", lpre: str = "official",
              batch_size: int = env.DEFAULT_BATCH_SIZE, attack: str = "pgd") -> Path | None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    got = load_rows(step, checkpoint, batch_size, attack)
    if got is None:
        return None
    rows, cfg = got
    cv = curves(rows, lpre)
    eps = cfg["config"]["eps"]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), sharey=True)
    _panels(axes, cv, eps, attack)
    h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=9, frameon=False)
    n = next(iter(cv.values()))["n"]
    seeds = sorted({r["seed"] for r in rows})
    fig.suptitle(f"BRONet adapted to trichoscopy — {STEP_TITLE[step]} — {n} test images, "
                 f"seed {', '.join(map(str, seeds))}, {checkpoint} checkpoint, "
                 f"{lpre} L", fontsize=11)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    out = (env.results_dir() / "figures"
           / layout.attack_figure(step, batch_size, checkpoint, attack))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_both(checkpoint: str = "headline", lpre: str = "official",
              batch_size: int = env.DEFAULT_BATCH_SIZE, attack: str = "pgd") -> Path | None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    got = {s: load_rows(s, checkpoint, batch_size, attack) for s in (1, 2)}
    if any(v is None for v in got.values()):
        return None
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.8), sharey=True)
    for row, step in enumerate((1, 2)):
        rows, cfg = got[step]
        cv = curves(rows, lpre)
        _panels(axes[row], cv, cfg["config"]["eps"], attack)
        n = next(iter(cv.values()))["n"]
        axes[row][0].set_ylabel(f"{STEP_TITLE[step]}\n({n} test images)\naccuracy")
    h, l = axes[0][1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=9, frameon=False)
    fig.suptitle(f"After adapting a certified backbone (BRONet, ImageNet) to trichoscopy — "
                 f"seed 0, {checkpoint} checkpoint, {lpre} L", fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out = (env.results_dir() / "figures"
           / layout.attack_figure(None, batch_size, checkpoint, attack))
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _run_metrics(step: int, which: str = "best", batch_size: int = env.DEFAULT_BATCH_SIZE) -> dict:
    """{run name: [metrics of each seed]} of the final runs (report's filtering)."""
    from .report import load_runs

    out: dict[str, list] = {}
    for (name, _seed), r in sorted(load_runs(step, which, batch_size).items()):
        out.setdefault(name, []).append(r["metrics"])
    return {n: out[n] for n in ORDER if n in out}


def plot_certified_seeds(which: str = "best", lpre: str = "official",
                         batch_size: int = env.DEFAULT_BATCH_SIZE) -> Path | None:
    """Certified accuracy vs eps on the whole test set, mean and min-max band over
    seeds, both steps -- the headline result, in the notebook's colours."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mt

    data = {s: _run_metrics(s, which, batch_size) for s in (1, 2)}
    if not any(data.values()):
        return None
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), sharey=True)
    for ax, step in zip(axes, (1, 2)):
        for name, ms in data[step].items():
            st = style(name)
            eps = np.array(ms[0][f"certificate_{lpre}"]["curve"]["epsilon"])
            c = np.array([m[f"certificate_{lpre}"]["curve"]["certified_accuracy"] for m in ms])
            ax.plot(eps, c.mean(0), color=st["color"], ls=st["ls"], lw=1.8, label=st["label"])
            if len(ms) > 1:
                ax.fill_between(eps, c.min(0), c.max(0), color=st["color"], alpha=0.12, lw=0)
        for e in (36 / 255, 72 / 255, 108 / 255):
            ax.axvline(e, color="grey", lw=0.6, ls=":")
        n = data[step][next(iter(data[step]))][0]["quality"]["n_images"] if data[step] else 0
        ax.set_title(f"{STEP_TITLE[step]} — whole test set (n = {n})", fontsize=10)
        ax.set_xlabel(r"$\epsilon$ (L2, 224×224 input, pixels in [0,1]); dotted: 36, 72, 108 /255",
                      fontsize=8)
        ax.grid(alpha=0.3)
        ax.yaxis.set_major_formatter(mt.PercentFormatter(1.0))
    axes[0].set_ylabel("certified accuracy")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=9, frameon=False)
    fig.suptitle(f"Certified accuracy, mean over seeds (band: min–max), "
                 f"{'best-validation' if which == 'best' else 'last'} epoch, {lpre} L",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    out = env.results_dir() / "figures" / layout.certified_curves_figure(batch_size, which, lpre)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_lipschitz(which: str = "best", batch_size: int = env.DEFAULT_BATCH_SIZE) -> Path | None:
    """L_total / L_0 per run and seed, log scale, both steps."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = {s: _run_metrics(s, which, batch_size) for s in (1, 2)}
    if not any(data.values()):
        return None
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2), sharey=True)
    for ax, step in zip(axes, (1, 2)):
        names = list(data[step])
        for k, name in enumerate(names):
            st = style(name)
            vals = [m["lipschitz"]["L_over_L0"] for m in data[step][name]]
            ax.scatter([k] * len(vals), vals, color=st["color"], s=28, zorder=3,
                       marker="o" if st["ls"] == "-" else "s")
            ax.hlines(np.exp(np.mean(np.log(vals))), k - 0.3, k + 0.3, color=st["color"], lw=2)
        ax.set_yscale("log")
        ax.axhline(1.0, color="#d62728", lw=0.8, ls=":")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n.replace("_stable", "\n(stable lr)") for n in names], fontsize=8)
        ax.set_title(STEP_TITLE[step], fontsize=10)
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel(r"$L_{total} / L_0$  (1 = certificate of the pretrained model kept)")
    fig.suptitle("How much each adaptation grows the Lipschitz bound (each dot = one seed; "
                 "bar = geometric mean)", fontsize=11)
    fig.tight_layout()
    out = env.results_dir() / "figures" / layout.lipschitz_growth_figure(batch_size, which)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main(argv=None) -> int:
    from repro import console

    console.init()
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--step", type=int, choices=(1, 2), default=None)
    p.add_argument("--checkpoint", default="headline")
    p.add_argument("--lpre", choices=("official", "rigorous"), default="official")
    p.add_argument("--attack", default="pgd",
                   help="whose attack results to plot (pgd, apgd-ce, apgd-dlr, apgd-t, square)")
    p.add_argument("--batch-size", type=int, default=env.DEFAULT_BATCH_SIZE,
                   help="training batch of the runs (runs/step_<n>_bs<B>/ unless 256)")
    a = p.parse_args(argv)
    bs = a.batch_size
    made = [plot_step(s, a.checkpoint, a.lpre, bs, a.attack)
            for s in ([a.step] if a.step else (1, 2))]
    if a.step is None:
        made.append(plot_both(a.checkpoint, a.lpre, bs, a.attack))
        made.append(plot_certified_seeds("best", a.lpre, bs))
        made.append(plot_lipschitz("best", bs))
    made = [m for m in made if m is not None]
    for m in made:
        print(f"-> {m}")
    return 0 if made else 1


if __name__ == "__main__":
    sys.exit(main())
