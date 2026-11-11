"""Aggregate the runs of one step: per-seed CSV, mean +- SD summary, McNemar, figures.

    python -m experiments.bronet_mlp.report --step 2

Writes to <DERMA_REPRO_OUT>/bronet_mlp/results/:

    adaptation_step<n>.csv   one row per (mode, seed, epsilon); the columns of the
                             notebook's results/adaptation.csv plus L_rigorous,
                             f1_weighted, certified accuracies (accuracies are
                             fractions in [0, 1] there, as in the notebook)
    summary_step<n>.md       HEADLINE: the model kept by early stopping on validation
                             (best.pt): section-6.4-style table, per-class F1, McNemar
    summary_step<n>_last.md  the same for the last epoch (final.pt) of the same runs
    mcnemar_step<n>[_last].csv   every pairwise McNemar test on clean accuracy
    mcnemar_cert_step<n>[_last].csv   the same on "certified at eps" (correct and
                             radius > eps), eps 36/72/108 /255, official and rigorous L
    figures/                 certified accuracy vs epsilon (…_best / …_last)
    adaptation_step<n>.csv   both checkpoints, column `checkpoint`

McNemar: exact two-sided binomial test on the discordant pairs, on the same
test images. Two variants are reported -- seed 0 alone, and the majority vote
over the available seeds (ties broken towards the lowest class index). The
baselines are the `repro` runs of the paper's models, variant ``unique``,
weights ``final`` (their predictions_final.csv, found under
env.baseline_roots()).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from . import env

import numpy as np

from repro.config import CLASS_NAMES

from . import adapters, certify
from experiments.common import layout
from experiments.common.display import CHECKPOINT_TEXT, label  # noqa: F401
from experiments.common.io import write_csv  # noqa: F401
from experiments.common.stats import fmt, holm, majority_vote, mcnemar, mean_sd  # noqa: F401  (re-exported)

# Reference CNNs of `trichoscopy/` (variant unique, final weights, seed 0): accuracy, weighted F1.
BASELINES = {
    1: {"densenet169": (87.59, 87.21), "efficientnet_b0": (87.59, 87.28),
        "resnet152": (88.70, 87.82)},
    2: {"densenet169": (79.58, 79.47), "efficientnet_b0": (81.67, 81.60),
        "resnet152": (79.17, 79.06)},
}
DISPLAY = {"densenet169": "DenseNet169", "efficientnet_b0": "EfficientNetB0",
           "resnet152": "ResNet152"}
EPS_NAMES = ("36/255", "72/255", "108/255")
# Run folders in report order: each mode, then its `stable` tuning variant if trained.
RUN_NAMES = tuple(n for m in adapters.MODES for n in (env.run_name(m), env.run_name(m, "stable")))


def _metrics_file(d: Path, which: str) -> Path | None:
    p = d / layout.metrics_file(which)
    if p.exists():
        return p
    if which == "final" and (d / "metrics.json").exists() \
            and not (d / "metrics_best.json").exists():
        return d / "metrics.json"          # runs written before per-checkpoint files
    return None


def load_runs(step: int, which: str = "best", batch_size: int = env.DEFAULT_BATCH_SIZE) -> dict:
    """{(run name, seed): {"metrics": ..., "dir": ..., "which": ...}} for every
    finished run that has checkpoint `which` ("best" or "final")."""
    out = {}
    root = env.runs_dir(step, batch_size)
    for mode in RUN_NAMES:
        for d in sorted((root / mode).glob("seed*")) if (root / mode).is_dir() else []:
            mf = _metrics_file(d, which)
            if mf is not None and (d / "metrics.json").exists() \
                    and (d / layout.CHECKPOINTS_DIR / layout.checkpoint_file(which)).exists():
                m = env.read_json(mf)
                why = stale_reason(step, mode, m)
                if why:
                    print(f"  skipping {d}: {why}")
                    continue
                out[(mode, int(d.name[4:]))] = {"metrics": m, "dir": d, "which": which}
    return out


def stale_reason(step: int, name: str, metrics: dict) -> str | None:
    """Why a run of the tuned protocol no longer matches tuning/selection.json
    (e.g. the grid was widened and the choice moved), else None. Runs outside
    the tuned protocol (no early stopping) are not checked."""
    if metrics["config"].get("early_stop") != "val":
        return None
    from . import tune

    path = tune.selection_path(step, metrics["config"].get("batch_size", env.DEFAULT_BATCH_SIZE))
    if not path.exists():
        return None
    mode = metrics["config"]["mode"]
    variant = "stable" if name.endswith("_stable") else "best"
    s = env.read_json(path)["modes"].get(mode)
    if s is None or "variants" not in s:
        return None
    if variant == "stable" and s["stable_equals_best"]:
        return "the stable selection now equals the best one (reported under the mode itself)"
    chosen = s["variants"][variant]
    if chosen is None:
        return f"no {variant} selection for {mode} any more"
    if abs(chosen["lr"] - metrics["config"]["lr"]) > 1e-12 * max(1.0, chosen["lr"]):
        return (f"trained with lr {metrics['config']['lr']:g}, selection is now "
                f"{chosen['lr']:g} -- rerun run_grid --from-tuning")
    return None


def read_predictions(path: Path) -> dict[str, tuple[int, int]]:
    """{image: (true, predicted)} from a predictions_final.csv (ours or repro's)."""
    with open(path, newline="", encoding="utf-8") as handle:
        return {r["image"]: (int(r["true_label"]), int(r["predicted_label"]))
                for r in csv.DictReader(handle)}


def find_baseline(step: int, arch: str) -> Path | None:
    for root in env.baseline_roots():
        p = root / "models" / f"step_{step}" / f"{arch}_unique" / "predictions_final.csv"
        if p.exists():
            return p
    return None


# The pre-declared primary comparison: the orthogonal adapter against every other
# system. Holm is applied within this family, and separately over all pairs.
PRIMARY = "butterfly"


def is_primary(row: dict) -> bool:
    return any(x == PRIMARY or x.startswith(PRIMARY + " (") for x in (row["a"], row["b"]))


def add_holm(rows: list[dict], group_keys: tuple[str, ...]) -> list[dict]:
    """Add p_holm_all (all pairs of a group) and p_holm_primary (pairs involving
    PRIMARY, "" otherwise). A group is one family of simultaneous tests."""
    groups: dict[tuple, list[int]] = {}
    for i, r in enumerate(rows):
        groups.setdefault(tuple(r[k] for k in group_keys), []).append(i)
    for idx in groups.values():
        for i, p in zip(idx, holm([rows[i]["p_value"] for i in idx])):
            rows[i]["p_holm_all"] = p
        prim = [i for i in idx if is_primary(rows[i])]
        for i, p in zip(prim, holm([rows[i]["p_value"] for i in prim])):
            rows[i]["p_holm_primary"] = p
        for i in idx:
            rows[i].setdefault("p_holm_primary", "")
    return rows


def _fmt_p(p) -> str:
    return "—" if p == "" else f"{p:.3g}"


def read_certificates(run_dir: Path, which: str) -> dict[str, dict[str, float]]:
    """{image: {"correct", "radius_official", "radius_rigorous"}} of one run.

    Radii come from certificates_<which>.npz (full precision); image names from
    predictions_<which>.csv, whose rows are in the same order."""
    with open(run_dir / layout.predictions_file(which), newline="", encoding="utf-8") as handle:
        images = [r["image"] for r in csv.DictReader(handle)]
    z = np.load(run_dir / layout.arrays_file("certificates", which))
    if len(images) != len(z["y"]):
        raise RuntimeError(f"{run_dir}: predictions and certificates disagree in length")
    correct = z["pred"] == z["y"]
    return {im: {"correct": bool(correct[i]), "radius_official": float(z["radius_official"][i]),
                 "radius_rigorous": float(z["radius_rigorous"][i])}
            for i, im in enumerate(images)}


def certified_mcnemar(runs: dict, which: str) -> list[dict]:
    """Paired McNemar on 'certified at eps' (correct and radius > eps), every pair of
    run names, eps in EPS_REPORT, both Lipschitz constants, seed 0 and majority vote."""
    modes = [m for m in RUN_NAMES if any(k[0] == m for k in runs)]
    seeds = {m: sorted(k[1] for k in runs if k[0] == m) for m in modes}
    certs = {k: read_certificates(r["dir"], which) for k, r in runs.items()}
    images = sorted(next(iter(certs.values())))
    for k, c in certs.items():
        if sorted(c) != images:
            raise RuntimeError(f"run {k}: different test images")
    rows = []
    for lpre in ("official", "rigorous"):
        for name, eps in zip(EPS_NAMES, certify.EPS_REPORT):
            def cert(k):
                c = certs[k]
                return np.array([c[i]["correct"] and c[i][f"radius_{lpre}"] > eps
                                 for i in images])

            systems = {"seed 0": {label(m): cert((m, 0)) for m in modes if 0 in seeds[m]},
                       "majority vote": {
                           label(m): np.stack([cert((m, s)) for s in seeds[m]]).sum(0)
                           > len(seeds[m]) / 2 for m in modes}}
            for variant, sysd in systems.items():
                keys = list(sysd)
                for i, a in enumerate(keys):
                    for b in keys[i + 1:]:
                        res = mcnemar(sysd[a], sysd[b])
                        rows.append({"variant": variant, "L": lpre, "epsilon_name": name,
                                     "epsilon": eps, "a": a, "b": b,
                                     "cert_a": 100 * float(sysd[a].mean()),
                                     "cert_b": 100 * float(sysd[b].mean()),
                                     "only_a_certified": res["only_a_correct"],
                                     "only_b_certified": res["only_b_correct"],
                                     "p_value": res["p_value"]})
    # one family per (variant, L): all eps together
    return add_holm(rows, ("variant", "L"))


def adaptation_rows(runs: dict) -> list[dict]:
    rows = []
    for (mode, seed), r in sorted(runs.items(), key=lambda kv: (RUN_NAMES.index(kv[0][0]),
                                                                  kv[0][1])):
        m = r["metrics"]
        lip, co, cr = m["lipschitz"], m["certificate_official"], m["certificate_rigorous"]
        for name, eps in zip(EPS_NAMES, certify.EPS_REPORT):
            rows.append({
                "mode": m["config"]["mode"], "run": mode, "seed": seed,
                "lr": m["config"]["lr"], "epochs": m["config"]["epochs"],
                "trainable_params": m["trainable_params"],
                "backbone_L": lip["L0_official"], "L_analytic": lip["L_total_official"],
                "worst_sigma_max": lip["worst_sigma_max"],
                "clean_accuracy": m["accuracy"] / 100, "certified_radius": co["median_radius"],
                "epsilon": eps, "random_accuracy": "", "robust_accuracy": "",
                "seconds": m["seconds"].get("train", 0.0) + m["seconds"].get("evaluate", 0.0),
                "L_rigorous": lip["L_total_rigorous"], "f1_weighted": m["f1_weighted"] / 100,
                "L_over_L0": lip["L_over_L0"],
                "certified_accuracy": co["certified_accuracy_named"][name],
                "certified_accuracy_rigorous": cr["certified_accuracy_named"][name],
                "certified_radius_rigorous": cr["median_radius"],
                "fingerprint": m["fingerprint"]})
    return rows


def summary(step: int, runs: dict, which: str = "best") -> tuple[str, list[dict]]:
    names = CLASS_NAMES[step]
    modes = [m for m in RUN_NAMES if any(k[0] == m for k in runs)]
    by_mode = {m: [runs[k]["metrics"] for k in sorted(runs) if k[0] == m] for m in modes}
    seeds = {m: sorted(k[1] for k in runs if k[0] == m) for m in modes}
    best = max(BASELINES[step], key=lambda a: BASELINES[step][a][0])

    def col(fn, digits=2, scale=1.0):
        return [fmt([fn(x) for x in by_mode[m]], digits, scale) for m in modes]

    lines = [f"# BRONet MLP adaptation, step-{step} — "
             f"{'early-stopped (best validation epoch)' if which == 'best' else 'last epoch'}",
             "",
             "Protocol `trichoscopy/`, variant `unique`: delivered train (byte-identical "
             f"copies removed within class) / test originals; {CHECKPOINT_TEXT[which]}. "
             "Mean ± SD over seeds. Certified radius and certified accuracy use the "
             "official-style L_total = L_pre(official) × L_mlp(exact); the rigorous "
             "L_pre gives the `rigorous` rows. Radii are L2 in the 224×224 input "
             "(pixels in [0,1]); divide by 0.37458 for the 600×600 files.", "",
             "Seeds per mode: " + ", ".join(f"{m} {seeds[m]}" for m in modes), ""]
    head = ["", *map(label, modes), f"{DISPLAY[best]} (baseline)"]
    lines += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    bacc, bf1 = BASELINES[step][best]

    def row(label, cells, base="—"):
        lines.append("| " + " | ".join([label, *cells, base]) + " |")

    row("clean accuracy, %", col(lambda x: x["accuracy"]), f"{bacc:.2f}")
    row("epoch kept (1-based)", col(lambda x: x.get("epoch", x["config"]["epochs"] - 1) + 1, 1))
    row("collapsed to one class (seeds)",
        [f"{sum(x['collapsed'] for x in by_mode[m])}/{len(by_mode[m])}" for m in modes])
    if step == 1:
        row("accuracy, paper composition, %",
            col(lambda x: x["quality"]["accuracy_paper_composition"]))
    row("weighted F1, %", col(lambda x: x["f1_weighted"]), f"{bf1:.2f}")
    row("L_total / L₀", col(lambda x: x["lipschitz"]["L_over_L0"], 6))
    row("L_total (official-style)", col(lambda x: x["lipschitz"]["L_total_official"], 4))
    row("L_total (rigorous)", col(lambda x: x["lipschitz"]["L_total_rigorous"], 4))
    row("worst σ_max over MLP layers", col(lambda x: x["lipschitz"]["worst_sigma_max"], 5))
    row("median certified radius", col(lambda x: x["certificate_official"]["median_radius"], 4))
    row("median radius (rigorous)", col(lambda x: x["certificate_rigorous"]["median_radius"], 4))
    for name in EPS_NAMES:
        row(f"certified acc @ {name}, %",
            col(lambda x, n=name: x["certificate_official"]["certified_accuracy_named"][n], 2, 100))
    row("certified acc @ 36/255 (rigorous), %",
        col(lambda x: x["certificate_rigorous"]["certified_accuracy_named"]["36/255"], 2, 100))
    row("(PGD @ ε)", ["stage 2"] * len(modes))
    row("trainable params", [f"{by_mode[m][0]['trainable_params']:,}" for m in modes])
    row("learning rate", [", ".join(sorted({f"{x['config']['lr']:g}" for x in by_mode[m]}))
                          for m in modes])
    row("epochs", [", ".join(sorted({str(x["config"]["epochs"]) for x in by_mode[m]}))
                   for m in modes])
    lines += ["", "A collapsed model (accuracy ≤ majority-class rate) predicts one class "
              "with a large margin; its radii and certified accuracy are trivially high and "
              "say nothing about robustness.",
              "", "Baselines (repro, unique, final): " + ", ".join(
        f"{DISPLAY[a]} {acc:.2f} / F1 {f1:.2f}" for a, (acc, f1) in BASELINES[step].items()),
        ""]

    lines += ["## Per-class F1 (mean over seeds)", "",
              "| class | " + " | ".join(map(label, modes)) + " |", "|" + "---|" * (len(modes) + 1)]
    for c in names:
        lines.append(f"| {c} | " + " | ".join(
            fmt([x["quality"]["per_class"][c]["f1"] for x in by_mode[m]]) for m in modes) + " |")

    # McNemar
    preds = {}
    for (mode, seed), r in runs.items():
        preds[(mode, seed)] = read_predictions(r["dir"] / layout.predictions_file(which))
    images = sorted(next(iter(preds.values())))
    truth = np.array([next(iter(preds.values()))[i][0] for i in images])
    systems: dict[str, dict[str, np.ndarray]] = {"seed 0": {}, "majority vote": {}}
    for m in modes:
        if 0 in seeds[m]:
            systems["seed 0"][label(m)] = np.array([preds[(m, 0)][i][1] for i in images])
        systems["majority vote"][label(m)] = majority_vote(
            [np.array([preds[(m, s)][i][1] for i in images]) for s in seeds[m]], len(names))
    baselines = {}
    for arch in BASELINES[step]:
        p = find_baseline(step, arch)
        if p is None:
            continue
        bp = read_predictions(p)
        if sorted(bp) != images or any(bp[i][0] != t for i, t in zip(images, truth)):
            raise RuntimeError(f"{p}: test images or labels differ from ours")
        baselines[DISPLAY[arch]] = np.array([bp[i][1] for i in images])
    mc_rows = []
    for variant, sysd in systems.items():
        allsys = {**sysd, **baselines}
        keys = list(allsys)
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                if a in baselines and b in baselines:
                    continue
                res = mcnemar(allsys[a] == truth, allsys[b] == truth)
                mc_rows.append({"variant": variant, "a": a, "b": b,
                                "acc_a": 100 * float((allsys[a] == truth).mean()),
                                "acc_b": 100 * float((allsys[b] == truth).mean()), **res})
    add_holm(mc_rows, ("variant",))
    holm_note = (f"`p Holm (primary)`: Holm-Bonferroni over the pre-declared family "
                 f"'{PRIMARY} vs every other system' (— for pairs outside it); "
                 "`p Holm (all)`: over every pair of the table section. ")
    lines += ["", "## McNemar (exact, two-sided)", "",
              f"Same {len(images)} test images for every system. `seed 0` uses one run per "
              "mode; `majority vote` combines the seeds of each mode. Baselines: "
              + (", ".join(f"{k} ({find_baseline(step, a)})" for a, k in
                           ((a, DISPLAY[a]) for a in BASELINES[step]) if k in baselines)
                 or "none found") + ". " + holm_note
              + "Families: one per variant.", "",
              "| variant | A | B | acc A | acc B | only A correct | only B correct | p "
              "| p Holm (primary) | p Holm (all) |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for r in mc_rows:
        lines.append(f"| {r['variant']} | {r['a']} | {r['b']} | {r['acc_a']:.2f} | "
                     f"{r['acc_b']:.2f} | {r['only_a_correct']} | {r['only_b_correct']} | "
                     f"{r['p_value']:.3g} | {_fmt_p(r['p_holm_primary'])} | "
                     f"{_fmt_p(r['p_holm_all'])} |")
    cert_rows = certified_mcnemar(runs, which)
    lines += ["", "## McNemar on certified accuracy (exact, two-sided)", "",
              "Per test image, the outcome is *certified at ε* = classified correctly AND "
              "certified radius > ε (official-style L; the rigorous L is in "
              f"mcnemar_cert_step{step}{'' if which == 'best' else '_last'}.csv). "
              "`seed 0`: one run per mode; `majority vote`: an image counts as certified "
              "when it is certified in more than half of the seeds. The baselines have no "
              "certificate and are not compared. " + holm_note
              + "Families: one per (variant, L), all three ε together.", "",
              "| variant | ε | A | B | cert A, % | cert B, % | only A certified | "
              "only B certified | p | p Holm (primary) | p Holm (all) |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in cert_rows:
        if r["L"] != "official":
            continue
        lines.append(f"| {r['variant']} | {r['epsilon_name']} | {r['a']} | {r['b']} | "
                     f"{r['cert_a']:.2f} | {r['cert_b']:.2f} | {r['only_a_certified']} | "
                     f"{r['only_b_certified']} | {r['p_value']:.3g} | "
                     f"{_fmt_p(r['p_holm_primary'])} | {_fmt_p(r['p_holm_all'])} |")
    lines += ["", "## Limitations", "",
              "- The delivered split is by photograph, not by patient.",
              "- Only the dense MLP after spatial pooling is adapted; the convolutional "
              "part of BRONet is frozen (its width 588 is not a power of two).",
              "- After deduplication the training classes are imbalanced; no class "
              "weighting is used.",
              "- The certificate covers L2 perturbations of the 224×224 input only — not "
              "illumination, device or JPEG changes; JPEG and integer rounding in the "
              "preprocessing are not Lipschitz.",
              ]
    early = any(x["config"].get("early_stop") == "val" for ms in by_mode.values() for x in ms)
    if early:
        lines += [
            "- The lr per mode and the kept epoch were chosen on a 10 % validation split of "
            "train (grouped by photograph, but patients are shared with train, as with test); "
            "the test set did not influence either choice. The adapted models are trained on "
            "the other 90 % of train, the paper's baselines on 100 %.",
            "- Selecting the best of many (lr, epoch) pairs on 1.4k validation images makes "
            "the validation numbers optimistic; the test numbers are not affected."]
    else:
        lines += ["- No validation split: every number is on the test set, final epoch."]
    return "\n".join(lines) + "\n", mc_rows


def figures(step: int, runs: dict, out: Path, which: str = "best",
            batch_size: int = env.DEFAULT_BATCH_SIZE) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for lpre in ("official", "rigorous"):
        fig, ax = plt.subplots(figsize=(6, 4))
        for mode in RUN_NAMES:
            curves = [np.array(r["metrics"][f"certificate_{lpre}"]["curve"]["certified_accuracy"])
                      for (m, _), r in runs.items() if m == mode]
            if not curves:
                continue
            eps = runs[next(k for k in runs if k[0] == mode)]["metrics"][
                f"certificate_{lpre}"]["curve"]["epsilon"]
            c = 100 * np.stack(curves)
            ax.plot(eps, c.mean(0), label=label(mode), ls="--" if "_stable" in mode else "-")
            if len(curves) > 1:
                ax.fill_between(eps, c.min(0), c.max(0), alpha=0.2)
        for e in certify.EPS_REPORT:
            ax.axvline(e, color="grey", lw=0.5, ls=":")
        ax.set_xlabel("ε (L2, 224×224 input, pixels in [0,1])")
        ax.set_ylabel("certified accuracy, %")
        ax.set_title(f"step-{step}, L_pre {lpre}, {'best-val' if which == 'best' else 'last'} epoch")
        ax.legend()
        fig.tight_layout()
        p = out / layout.certified_figure(step, batch_size, lpre, which)
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(p)
    return paths


def main(argv=None) -> int:
    from repro import console

    console.init()
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--step", type=int, choices=(1, 2), required=True)
    p.add_argument("--batch-size", type=int, default=env.DEFAULT_BATCH_SIZE,
                   help="training batch of the runs to report (runs/step_<n>_bs<B>/ unless 256)")
    a = p.parse_args(argv)
    by_which = {w: load_runs(a.step, w, a.batch_size) for w in ("best", "final")}
    if not any(by_which.values()):
        print(f"no finished runs under {env.runs_dir(a.step, a.batch_size)}")
        return 1
    res = env.results_dir()
    rows, figs = [], []
    for which, runs in by_which.items():
        if not runs:
            continue
        rows += [{"checkpoint": which, **r} for r in adaptation_rows(runs)]
        md, mc = summary(a.step, runs, which)
        env.atomic_write_text(res / layout.summary_file(a.step, a.batch_size, which), md)
        write_csv(res / layout.mcnemar_file(a.step, a.batch_size, which), mc)
        write_csv(res / layout.mcnemar_file(a.step, a.batch_size, which, certified=True),
                  certified_mcnemar(runs, which))
        figs += figures(a.step, runs, res / "figures", which, a.batch_size)
        print(md)
    write_csv(res / layout.adaptation_file(a.step, a.batch_size), rows)
    print(f"-> {res}  ({sum(map(len, by_which.values()))} run checkpoints, "
          f"{len(figs)} figures)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
