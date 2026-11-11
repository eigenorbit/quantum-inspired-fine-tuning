"""Aggregate the runs of one step: mean +- SD summary, McNemar against the full
fine-tuning of ResNet152 (reproduction) and against the BRONet runs.

    python -m experiments.resnet_adapt.report --step 2

Writes to <DERMA_REPRO_OUT>/resnet_adapt/results/:

    summary_step<n>.md        HEADLINE: best-validation epoch (best.pt)
    summary_step<n>_last.md   the same runs, last epoch (final.pt)
    runs_step<n>.csv          one row per (run, seed, checkpoint)
    mcnemar_step<n>[_last].csv

McNemar: exact two-sided test on the discordant pairs, on the same test images
(checked name by name and label by label), as `bronet_mlp.report` -- seed 0 and
the majority vote over seeds. Compared with the ResNet152 of the reproduction
(full fine-tuning, variant unique, final weights, seed 0: predictions_final.csv
under bronet_mlp.env.baseline_roots()) and with every BRONet run of the tuned protocol
(best-validation epoch, predictions_best.csv). Holm-Bonferroni over the family
of all comparisons that involve a model of this package, per variant.
"""

from __future__ import annotations

import argparse
import sys

from . import paths

import numpy as np

from experiments.bronet_mlp import report as breport      # the BRONet runs we compare with
from experiments.common import display, stats
from repro.config import CLASS_NAMES
from experiments.common.io import read_json, atomic_write_text, write_csv
from experiments.common import layout

from . import train

RUN_NAMES = tuple(n for m in train.MODES for n in (paths.run_name(m), paths.run_name(m, "stable")))
# ResNet152 of the reproduction: full fine-tuning (58.7 M trainable, paper head
# 2048-256-K), 30 epochs, 100 % of train, last epoch, seed 0; step-1 on the
# RTX 4070 Laptop, step-2 on an AWS T4.
FULL_FT = {1: {"paper_composition": 88.25, "params": 58_669_637, "minutes": "193 (RTX 4070)"},
           2: {"paper_composition": None, "params": 58_669_123, "minutes": "154 (T4)"}}
FULL_FT_NAME = "ResNet152 full FT"
CHECKPOINT_TEXT = display.CHECKPOINT_TEXT


def label(name: str) -> str:
    return display.label(name)


def stale_reason(step: int, name: str, metrics: dict) -> str | None:
    """Why a tuned run no longer matches its selection (lr changed), else None."""
    if metrics["config"].get("early_stop") != "val":
        return None
    from . import tune

    path = tune.selection_path(step, metrics["config"]["batch_size"])
    if not path.exists():
        return None
    s = read_json(path)["modes"].get(metrics["config"]["mode"])
    if s is None:
        return None
    variant = "stable" if name.endswith("_stable") else "best"
    if variant == "stable" and s["stable_equals_best"]:
        return "the stable selection now equals the best one"
    chosen = s["variants"][variant]
    if chosen is None:
        return f"no {variant} selection any more"
    if abs(chosen["lr"] - metrics["config"]["lr"]) > 1e-12 * max(1.0, chosen["lr"]):
        return (f"trained with lr {metrics['config']['lr']:g}, selection is now "
                f"{chosen['lr']:g} -- rerun run_grid --from-tuning")
    return None


def load_runs(step: int, which: str = "best", batch_size: int = paths.DEFAULT_BATCH_SIZE) -> dict:
    """{(run name, seed): {"metrics", "dir"}} of finished runs having checkpoint `which`."""
    out = {}
    root = paths.runs_dir(step, batch_size)
    for name in RUN_NAMES:
        for d in sorted((root / name).glob("seed*")) if (root / name).is_dir() else []:
            mf = d / layout.metrics_file(which)
            if mf.exists() and (d / "metrics.json").exists():
                m = read_json(mf)
                why = stale_reason(step, name, m)
                if why:
                    print(f"  skipping {d}: {why}")
                    continue
                out[(name, int(d.name[4:]))] = {"metrics": m, "dir": d}
    return out


def bronet_runs(step: int, which: str = "best") -> dict:
    """The BRONet runs of the tuned protocol at checkpoint `which` (may be empty)."""
    try:
        return breport.load_runs(step, which)
    except FileNotFoundError:
        return {}


def _systems(preds: dict, names: list[str], images: list[str], k: int) -> dict:
    seeds = {m: sorted(s for n, s in preds if n == m) for m in names}
    out = {"seed 0": {}, "majority vote": {}}
    for m in names:
        if 0 in seeds[m]:
            out["seed 0"][m] = np.array([preds[(m, 0)][i][1] for i in images])
        out["majority vote"][m] = stats.majority_vote(
            [np.array([preds[(m, s)][i][1] for i in images]) for s in seeds[m]], k)
    return out


def check_same_images(ref: dict, other: dict, what: str) -> None:
    if sorted(other) != sorted(ref) or any(other[i][0] != ref[i][0] for i in ref):
        raise RuntimeError(f"{what}: test images or labels differ from ours")


def mcnemar_rows(step: int, runs: dict, which: str, bronet: dict) -> tuple[list[dict], dict]:
    """All pairs involving one of our systems; Holm per variant over those pairs."""
    k = len(CLASS_NAMES[step])
    ours = {key: breport.read_predictions(r["dir"] / layout.predictions_file(which))
            for key, r in runs.items()}
    ref = next(iter(ours.values()))
    images = sorted(ref)
    truth = np.array([ref[i][0] for i in images])
    for key, p in ours.items():
        check_same_images(ref, p, f"run {key}")
    our_names = [label(n) for n in RUN_NAMES if any(key[0] == n for key in ours)]
    systems = _systems({(label(n), s): p for (n, s), p in ours.items()}, our_names, images, k)
    fixed, sources, full_f1 = {}, {}, None
    base = breport.find_baseline(step, "resnet152")
    if base is not None:
        bp = breport.read_predictions(base)
        check_same_images(ref, bp, str(base))
        fixed[FULL_FT_NAME] = np.array([bp[i][1] for i in images])
        sources[FULL_FT_NAME] = str(base)
        from sklearn.metrics import f1_score

        full_f1 = 100 * float(f1_score(truth, fixed[FULL_FT_NAME], average="weighted"))
    bro = {}
    for (name, seed), r in bronet.items():
        p = breport.read_predictions(r["dir"] / layout.predictions_file(which))
        check_same_images(ref, p, f"BRONet run {name}/seed{seed}")
        bro[(f"BRONet {label(name)}", seed)] = p
    bro_names = sorted({n for n, _ in bro}, key=lambda n: [f"BRONet {label(x)}" for x in
                                                           breport.RUN_NAMES].index(n))
    bro_sys = _systems(bro, bro_names, images, k) if bro else {"seed 0": {}, "majority vote": {}}
    rows = []
    for variant in ("seed 0", "majority vote"):
        mine = systems[variant]
        others = {**fixed, **bro_sys[variant]}
        pairs = [(a, b) for i, a in enumerate(mine) for b in list(mine)[i + 1:]]
        pairs += [(a, b) for a in mine for b in others]
        allsys = {**mine, **others}
        block = []
        for a, b in pairs:
            res = stats.mcnemar(allsys[a] == truth, allsys[b] == truth)
            block.append({"variant": variant, "a": a, "b": b,
                          "acc_a": 100 * float((allsys[a] == truth).mean()),
                          "acc_b": 100 * float((allsys[b] == truth).mean()), **res})
        for r, p in zip(block, stats.holm([r["p_value"] for r in block])):
            r["p_holm"] = p
        rows += block
    info = {"n_images": len(images), "sources": sources, "full_f1": full_f1,
            "bronet_seeds": {n: sorted(s for m, s in bro if m == n) for n in bro_names}}
    return rows, info


def _mean_sd(values, digits=2) -> str:
    return stats.fmt(list(values), digits)


def summary(step: int, runs: dict, which: str, bronet: dict) -> tuple[str, list[dict]]:
    names = CLASS_NAMES[step]
    modes = [m for m in RUN_NAMES if any(k[0] == m for k in runs)]
    by_mode = {m: [runs[k]["metrics"] for k in sorted(runs) if k[0] == m] for m in modes}
    seeds = {m: sorted(k[1] for k in runs if k[0] == m) for m in modes}
    mc, info = mcnemar_rows(step, runs, which, bronet)
    full_acc = next((r["acc_b"] for r in mc if r["b"] == FULL_FT_NAME), None)
    bro_modes = [n for n in breport.RUN_NAMES if any(k[0] == n for k in bronet)]
    bro_by = {n: [bronet[k]["metrics"] for k in sorted(bronet) if k[0] == n] for n in bro_modes}

    head = ["", *map(label, modes), f"{FULL_FT_NAME} (repro)",
            *[f"BRONet {label(n)}" for n in bro_modes]]
    lines = [f"# ResNet152, layer4.2 adaptation + LLN head, step-{step} — "
             f"{'early-stopped (best validation epoch)' if which == 'best' else 'last epoch'}",
             "", "ResNet152 (IMAGENET1K_V2) frozen up to layer4.1; layer4.2 adapted: "
             "`full` = all of it trains (convolutions, BN affine; BN statistics frozen); "
             "`butterfly` = BN folded, W' = R(θ)·W per convolution (R orthogonal: spectrum "
             "kept); `lora` = BN folded, W + B·A per convolution, rank 4; `head-only` = BN-folded "
             "block frozen. A new LLN head "
             f"2048→{len(names)} always trains. Loss {train.LOSS_TEXT}. **Batch size "
             + ", ".join(sorted({str(x["config"]["batch_size"]) for ms in by_mode.values()
                                 for x in ms})) + "**. Protocol `trichoscopy/`, variant "
             f"`unique`; {CHECKPOINT_TEXT[which]}; the validation split is BRONet's, image for "
             "image. Mean ± SD over seeds.", "",
             "Seeds: " + ", ".join(f"{label(m)} {seeds[m]}" for m in modes), "",
             f"{FULL_FT_NAME}: the reproduction of the paper — whole network trained (paper head "
             "2048→256→K), 100 % of train, last epoch of 30, seed 0. BRONet columns: the same "
             f"checkpoint ({'best validation' if which == 'best' else 'last'} epoch), mean over "
             "seeds " + str(info["bronet_seeds"]) + ".", "",
             "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]

    def row(title, ours, full="—", bro=None):
        bro = bro if bro is not None else ["—"] * len(bro_modes)
        lines.append("| " + " | ".join([title, *ours, full, *bro]) + " |")

    def col(fn, digits=2):
        return [_mean_sd([fn(x) for x in by_mode[m]], digits) for m in modes]

    def bcol(fn, digits=2):
        return [_mean_sd([fn(x) for x in bro_by[n]], digits) for n in bro_modes]

    row("clean accuracy, %", col(lambda x: x["accuracy"]),
        f"{full_acc:.2f}" if full_acc is not None else "—", bcol(lambda x: x["accuracy"]))
    if step == 1:
        pc = FULL_FT[1]["paper_composition"]
        row("accuracy, paper composition (641), %",
            col(lambda x: x["quality"]["accuracy_paper_composition"]), f"{pc:.2f}",
            bcol(lambda x: x["quality"]["accuracy_paper_composition"]))
    row("weighted F1, %", col(lambda x: x["f1_weighted"]),
        f"{info['full_f1']:.2f}" if info["full_f1"] is not None else "—",
        bcol(lambda x: x["f1_weighted"]))
    row("epoch kept (1-based)", col(lambda x: x["epoch"] + 1, 1), "30",
        bcol(lambda x: x["epoch"] + 1, 1))
    row("collapsed to one class (seeds)",
        [f"{sum(x['collapsed'] for x in by_mode[m])}/{len(by_mode[m])}" for m in modes])
    row("trainable params", [f"{by_mode[m][0]['trainable_params']:,}" for m in modes],
        f"{FULL_FT[step]['params']:,}",
        [f"{bro_by[n][0]['trainable_params']:,}" for n in bro_modes])
    row("learning rate", [", ".join(sorted({f"{x['config']['lr']:g}" for x in by_mode[m]}))
                          for m in modes], "1e-3",
        [", ".join(sorted({f"{x['config']['lr']:g}" for x in bro_by[n]})) for n in bro_modes])
    row("σ_max / σ_max pretrained, layer4.2 convs (min–max)",
        [f"{min(x['spectrum']['ratio_sigma_max_min'] for x in by_mode[m]):.4f}–"
         f"{max(x['spectrum']['ratio_sigma_max_max'] for x in by_mode[m]):.4f}" for m in modes])

    def span(m, key, digits=6):
        xs = [x["spectrum"] for x in by_mode[m]]
        if any(f"{key}_min" not in s for s in xs):
            return "—"                        # runs evaluated before this field existed
        return (f"{min(s[f'{key}_min'] for s in xs):.{digits}f}–"
                f"{max(s[f'{key}_max'] for s in xs):.{digits}f}")

    row("σ_min / σ_min pretrained (min–max)", [span(m, "ratio_sigma_min") for m in modes])
    row("conv operator norm / pretrained (min–max; 3×3 via FFT on 7×7)",
        [span(m, "ratio_operator_norm") for m in modes])
    row("butterfly orthogonality error max |RᵀR − I|",
        [f"{max(x['orthogonality_error_max'] for x in by_mode[m]):.1e}"
         if all("orthogonality_error_max" in x for x in by_mode[m]) else "—" for m in modes])
    row("time: training only, min", col(lambda x: x["seconds"]["train"] / 60, 1),
        FULL_FT[step]["minutes"], bcol(lambda x: x["seconds"].get("train", 0.0) / 60, 1))
    row("time: total (training + per-epoch eval + final eval), min",
        col(lambda x: x["seconds"]["total"] / 60, 1))
    lines += ["", "Times on the machine of each run (see config.json → device); ours exclude "
              "the one-off feature extraction (features/step_<n>/…/index.json → passes). "
              "σ is of the BN-folded convolution as a matrix (out, in·k²).", ""]

    lines += ["## Per-class F1 (mean over seeds)", "",
              "| class | " + " | ".join(map(label, modes)) + " |",
              "|" + "---|" * (len(modes) + 1)]
    for c in names:
        lines.append(f"| {c} | " + " | ".join(
            _mean_sd([x["quality"]["per_class"][c]["f1"] for x in by_mode[m]]) for m in modes)
            + " |")

    lines += ["", "## McNemar (exact, two-sided)", "",
              f"Same {info['n_images']} test images for every system (checked). `seed 0`: one "
              "run per mode; `majority vote`: the seeds of each mode combined (for "
              f"{FULL_FT_NAME} the single run in both). p Holm: Holm-Bonferroni over all pairs "
              "of the variant (every pair involves a model of this experiment). Baseline: "
              + (", ".join(f"{k} ({v})" for k, v in info["sources"].items()) or "not found")
              + ".", "",
              "| variant | A | B | acc A | acc B | only A correct | only B correct | p | "
              "p Holm |", "|---|---|---|---|---|---|---|---|---|"]
    for r in mc:
        lines.append(f"| {r['variant']} | {r['a']} | {r['b']} | {r['acc_a']:.2f} | "
                     f"{r['acc_b']:.2f} | {r['only_a_correct']} | {r['only_b_correct']} | "
                     f"{r['p_value']:.3g} | {r['p_holm']:.3g} |")
    lines += ["", "## Limitations", "",
              "- Split by photograph, not by patient.",
              "- The lr and the kept epoch are chosen on 10 % of train; these models see 90 % "
              f"of train, {FULL_FT_NAME} 100 %; it also keeps the last epoch, without "
              "validation, and has a different head (2048→256→K).",
              "- Deduplicated training classes are imbalanced; no class weighting.",
              "- ResNet152 has no certificate; only clean accuracy is compared here."]
    return "\n".join(lines) + "\n", mc


def run_rows(runs: dict, which: str) -> list[dict]:
    rows = []
    for (name, seed), r in sorted(runs.items()):
        m = r["metrics"]
        rows.append({"checkpoint": which, "run": name, "seed": seed, "lr": m["config"]["lr"],
                     "epoch_1based": m["epoch"] + 1, "accuracy": m["accuracy"],
                     "f1_weighted": m["f1_weighted"], "collapsed": m["collapsed"],
                     "trainable_params": m["trainable_params"],
                     "ratio_sigma_max_min": m["spectrum"]["ratio_sigma_max_min"],
                     "ratio_sigma_max_max": m["spectrum"]["ratio_sigma_max_max"],
                     "seconds_train": m["seconds"]["train"],
                     "seconds_total": m["seconds"]["total"], "fingerprint": m["fingerprint"]})
    return rows


def main(argv=None) -> int:
    from repro import console

    console.init()
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--step", type=int, choices=(1, 2), required=True)
    p.add_argument("--batch-size", type=int, default=paths.DEFAULT_BATCH_SIZE,
                   help="which protocol to report (its runs live in step_<n>_bs<B>/)")
    a = p.parse_args(argv)
    by_which = {w: load_runs(a.step, w, a.batch_size) for w in ("best", "final")}
    if not any(by_which.values()):
        print(f"no finished runs under {paths.runs_dir(a.step, a.batch_size)}")
        return 1
    res = paths.results_dir()
    rows = []
    for which, runs in by_which.items():
        if not runs:
            continue
        md, mc = summary(a.step, runs, which, bronet_runs(a.step, which))
        atomic_write_text(res / layout.summary_file(a.step, a.batch_size, which), md)
        write_csv(res / layout.mcnemar_file(a.step, a.batch_size, which), mc)
        rows += run_rows(runs, which)
        print(md)
    write_csv(res / layout.runs_table_file(a.step, a.batch_size), rows)
    print(f"-> {res}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
