"""Build the comparison tables and figures once models have been trained.

    python -m repro.report                 # every run found, both steps
    python -m repro.report --step 1        # one step
    python -m repro.report --which best    # use the test-selected weights

Reads the ``metrics.json`` each training run writes and produces, under
``<DERMA_REPRO_OUT>/results/``:

    step<n>_comparison.csv    reproduced vs published, per metric, with delta
    step<n>_comparison.md     the same table, readable
    summary.md                overall accuracy / weighted F1 for all six runs
    figures/step<n>_<arch>_confusion.png
    figures/step<n>_<arch>_auc.png
    figures/step<n>_<arch>_auprc.png

`--which final` (the default) reports the weights after the last epoch.
`--which best` reports the epoch with the highest test accuracy -- selected
on the test set and therefore optimistically biased.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import console, paths, published
from .config import ARCHITECTURES, CLASS_NAMES, DISPLAY_NAMES, STEPS

METRIC_ROWS = ("precision", "recall", "f1")


def suffix_for_summary(variant: str) -> str:
    return "" if variant == "full" else f"_{variant}"


def load_run(step: int, architecture: str, variant: str = "full") -> dict | None:
    """The ``metrics.json`` of one run, or ``None`` when it has not been trained."""
    path = paths.run_dir(step, architecture, variant) / "metrics.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _report_of(run: dict, which: str) -> dict:
    reports = run["reports"]
    if which not in reports:
        available = ", ".join(sorted(reports))
        raise KeyError(f"run has no {which!r} report; available: {available}")
    return reports[which]


def comparison_rows(step: int, which: str, variant: str = "full") -> list[dict]:
    """One row per (class, metric), with a column per architecture.

    Each architecture contributes three columns: reproduced, published, delta.
    Missing runs are simply absent.
    """
    names = CLASS_NAMES[step]
    runs = {arch: load_run(step, arch, variant) for arch in ARCHITECTURES}
    runs = {arch: run for arch, run in runs.items() if run is not None}
    if not runs:
        return []

    rows: list[dict] = []

    def add(group: str, metric: str, getter) -> None:
        row = {"group": group, "metric": metric}
        for arch, run in runs.items():
            label = DISPLAY_NAMES[arch]
            reproduced, reference = getter(arch, run)
            row[f"{label} (repro)"] = reproduced
            row[f"{label} (paper)"] = reference
            row[f"{label} (delta)"] = (
                None if reproduced is None or reference is None else reproduced - reference
            )
        rows.append(row)

    add("Overall", "Accuracy", lambda arch, run: (
        _report_of(run, which)["overall"]["accuracy"],
        published.get(step, arch)["accuracy"],
    ))
    add("Overall", "F1 weighted", lambda arch, run: (
        _report_of(run, which)["overall"]["f1_weighted"],
        published.get(step, arch)["f1_weighted"],
    ))

    for name in names:
        for metric in METRIC_ROWS:
            add(
                name,
                metric.capitalize(),
                lambda arch, run, n=name, m=metric: (
                    _report_of(run, which)["per_class"][n][m],
                    published.get(step, arch)["per_class"][n][m],
                ),
            )
    return rows


def rows_to_markdown(rows: list[dict]) -> str:
    """Render comparison rows; deltas keep their sign, values do not."""
    if not rows:
        return "_no trained runs found_\n"
    headers = list(rows[0])
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        cells = []
        for header in headers:
            value = row[header]
            if isinstance(value, str) or value is None:
                cells.append(value if value is not None else "-")
            elif "delta" in header:
                cells.append(f"{value:+.2f}")
            else:
                cells.append(f"{value:.2f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def rows_to_csv(rows: list[dict]) -> str:
    import csv
    import io

    if not rows:
        return ""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    for row in rows:
        writer.writerow({
            k: ("" if v is None else v if isinstance(v, str) else round(v, 4))
            for k, v in row.items()
        })
    return buffer.getvalue()


# --- figures ----------------------------------------------------------------
def _plot_confusion(report: dict, step: int, architecture: str, out: Path) -> None:
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    matrix = np.asarray(report["confusion_matrix"])
    display = ConfusionMatrixDisplay(matrix, display_labels=CLASS_NAMES[step])
    fig, ax = plt.subplots(figsize=(6, 5.5), dpi=140)
    display.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(f"Step-{step} {DISPLAY_NAMES[architecture]} — confusion matrix")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def _plot_area_bars(report: dict, step: int, architecture: str, out: Path,
                    key: str, title: str) -> None:
    """AUC or AUPRC per class plus the micro and weighted averages."""
    import matplotlib.pyplot as plt

    areas = report[key]
    labels = CLASS_NAMES[step] + ["micro", "weighted"]
    values = [areas[label] for label in labels]

    fig, ax = plt.subplots(figsize=(7, 4), dpi=140)
    colors = ["#4C72B0"] * len(CLASS_NAMES[step]) + ["#55A868", "#C44E52"]
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.9, color="grey", linestyle="--", linewidth=0.8)
    ax.set_ylabel(key.upper())
    ax.set_title(f"Step-{step} {DISPLAY_NAMES[architecture]} — {title}")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.015, f"{value:.3f}",
                ha="center", fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def write_figures(step: int, which: str, variant: str = "full") -> list[Path]:
    paths.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for architecture in ARCHITECTURES:
        run = load_run(step, architecture, variant)
        if run is None:
            continue
        report = _report_of(run, which)
        stem = f"step{step}_{architecture}" + ("" if variant == "full" else f"_{variant}")
        confusion_path = paths.FIGURES_DIR / f"{stem}_confusion.png"
        _plot_confusion(report, step, architecture, confusion_path)
        roc_path = paths.FIGURES_DIR / f"{stem}_auc.png"
        _plot_area_bars(report, step, architecture, roc_path, "auc",
                        "ROC area under curve")
        pr_path = paths.FIGURES_DIR / f"{stem}_auprc.png"
        _plot_area_bars(report, step, architecture, pr_path, "auprc",
                        "precision-recall area under curve")
        written += [confusion_path, roc_path, pr_path]
    return written


# --- summary ----------------------------------------------------------------
def summary_markdown(which: str, variant: str = "full") -> str:
    lines = [
        "# Reproduction summary",
        "",
        f"Dataset variant: **{variant}**"
        + ("  (byte-identical duplicates dropped from training)" if variant == "unique" else ""),
        "",
        f"Weights reported: **{which}** "
        + ("(last epoch)" if which == "final" else "(highest test accuracy — test-selected)"),
        "",
        "| Step | Model | Accuracy (repro) | Accuracy (paper) | Δ | F1 weighted (repro) | F1 weighted (paper) | Δ |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    any_run = False
    for step in STEPS:
        for architecture in ARCHITECTURES:
            run = load_run(step, architecture, variant)
            if run is None:
                continue
            any_run = True
            overall = _report_of(run, which)["overall"]
            reference = published.get(step, architecture)
            lines.append(
                f"| {step} | {DISPLAY_NAMES[architecture]} "
                f"| {overall['accuracy']:.2f} | {reference['accuracy']:.2f} "
                f"| {overall['accuracy'] - reference['accuracy']:+.2f} "
                f"| {overall['f1_weighted']:.2f} | {reference['f1_weighted']:.2f} "
                f"| {overall['f1_weighted'] - reference['f1_weighted']:+.2f} |"
            )
    if not any_run:
        return "# Reproduction summary\n\n_No trained runs found._\n"

    lines += [
        "",
        "## Run provenance",
        "",
        "| Step | Model | Epoch | Weight fingerprint | Seed | Deterministic |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for step in STEPS:
        for architecture in ARCHITECTURES:
            run = load_run(step, architecture, variant)
            if run is None:
                continue
            checkpoint = run.get("checkpoints", {}).get(which, {})
            config = run.get("config", {})
            lines.append(
                f"| {step} | {DISPLAY_NAMES[architecture]} "
                f"| {checkpoint.get('epoch', '-')} "
                f"| `{checkpoint.get('fingerprint', '-')}` "
                f"| {config.get('seed', '-')} "
                f"| {config.get('deterministic', '-')} |"
            )
    lines += [
        "",
        "Two runs of the same command with the same seed must show the same",
        "fingerprint. `python -m repro.selftest` checks that end to end.",
        "",
        "## Reading the deltas",
        "",
        "Step-2 is directly comparable: the evaluation set is the same 240 original",
        "images the paper used (78 active / 48 inactive / 114 regrowth).",
        "",
        "Step-1 is **not** directly comparable. The delivered test split holds 717",
        "original images against the published 641. The two agree on AA (241),",
        "FFA-LPP (121), DF (60) and AGA (14) and differ only in CHP/HV: 281 here",
        "against 205 in the paper. Any Step-1 delta therefore mixes a genuine",
        "training difference with a different denominator on the largest class.",
        "",
    ]
    return "\n".join(lines)


def variant_comparison_markdown(which: str) -> str | None:
    """Every trained variant against the "full" baseline, per model and class.

    **unique** shares its test set with full -- evaluation is on original
    photographs, and those carry no duplicates -- so the delta isolates the
    effect of training on deduplicated data, on identical images.
    """
    from .paths import VARIANTS

    baseline = "full"
    others = [v for v in VARIANTS if v != baseline]

    rows = []
    for step in STEPS:
        for architecture in ARCHITECTURES:
            base = load_run(step, architecture, baseline)
            if base is None:
                continue
            base_overall = _report_of(base, which)["overall"]
            for variant in others:
                run = load_run(step, architecture, variant)
                if run is None:
                    continue
                overall = _report_of(run, which)["overall"]
                rows.append({
                    "step": step,
                    "arch": architecture,
                    "variant": variant,
                    "n_test": _report_of(run, which)["n_images"],
                    "n_test_base": _report_of(base, which)["n_images"],
                    "accuracy": overall["accuracy"],
                    "accuracy_base": base_overall["accuracy"],
                    "f1": overall["f1_weighted"],
                    "f1_base": base_overall["f1_weighted"],
                })
    if not rows:
        return None

    lines = [
        "# Dataset variants against the baseline",
        "",
        f"Weights reported: **{which}**. Baseline is **full** - the delivered data,",
        "duplicates included.",
        "",
        "| Step | Model | Variant | Test images | Accuracy | vs full | F1 weighted | vs full |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        same_test = row["n_test"] == row["n_test_base"]
        note = "" if same_test else f" ({row['n_test_base']} in full)"
        lines.append(
            f"| {row['step']} | {DISPLAY_NAMES[row['arch']]} | {row['variant']} "
            f"| {row['n_test']}{note} "
            f"| {row['accuracy']:.2f} | {row['accuracy'] - row['accuracy_base']:+.2f} "
            f"| {row['f1']:.2f} | {row['f1'] - row['f1_base']:+.2f} |"
        )

    lines += [
        "",
        "## Per-class F1",
        "",
    ]
    for step in STEPS:
        for architecture in ARCHITECTURES:
            base = load_run(step, architecture, baseline)
            if base is None:
                continue
            present = [(v, load_run(step, architecture, v)) for v in others]
            present = [(v, r) for v, r in present if r is not None]
            if not present:
                continue
            lines += [
                f"### Step-{step}, {DISPLAY_NAMES[architecture]}",
                "",
                "| Class | full | " + " | ".join(f"{v} | Δ" for v, _ in present) + " |",
                "| --- | --- | " + " | ".join("--- | ---" for _ in present) + " |",
            ]
            for name in CLASS_NAMES[step]:
                base_f1 = _report_of(base, which)["per_class"][name]["f1"]
                cells = [f"{base_f1:.2f}"]
                for _, run in present:
                    value = _report_of(run, which)["per_class"][name]["f1"]
                    cells.append(f"{value:.2f} | {value - base_f1:+.2f}")
                lines.append(f"| {name} | " + " | ".join(cells) + " |")
            lines.append("")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    console.init()
    parser = argparse.ArgumentParser(
        description="Compare trained runs with the published Tables 5 and 6.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--step", type=int, choices=(1, 2), default=None,
                        help="default: both steps")
    parser.add_argument("--which", choices=("final", "best"), default="final",
                        help="which saved weights to report")
    parser.add_argument("--variant", default="full",
                        choices=("full", "unique"),
                        help="which dataset variant's runs to report")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)

    paths.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    steps = [args.step] if args.step else list(STEPS)

    produced = []
    for step in steps:
        rows = comparison_rows(step, args.which, args.variant)
        if not rows:
            print(f"step {step}: no trained runs found, skipping", flush=True)
            continue
        suffix = "" if args.variant == "full" else f"_{args.variant}"
        csv_path = paths.RESULTS_DIR / f"step{step}_comparison{suffix}.csv"
        md_path = paths.RESULTS_DIR / f"step{step}_comparison{suffix}.md"
        csv_path.write_text(rows_to_csv(rows), encoding="utf-8")
        md_path.write_text(
            f"# Step-{step}: reproduced vs published ({args.which} weights)\n\n"
            + rows_to_markdown(rows),
            encoding="utf-8",
        )
        produced += [csv_path, md_path]
        if not args.no_figures:
            produced += write_figures(step, args.which, args.variant)

    summary_path = paths.RESULTS_DIR / f"summary{suffix_for_summary(args.variant)}.md"
    summary_path.write_text(summary_markdown(args.which, args.variant), encoding="utf-8")
    produced.append(summary_path)

    variant_table = variant_comparison_markdown(args.which)
    if variant_table is not None:
        variant_path = paths.RESULTS_DIR / "variants_vs_full.md"
        variant_path.write_text(variant_table, encoding="utf-8")
        produced.append(variant_path)

    print(summary_markdown(args.which, args.variant))
    for path in produced:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
