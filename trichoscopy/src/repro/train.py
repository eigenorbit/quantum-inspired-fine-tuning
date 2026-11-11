"""Train one (step, architecture) pair, reproducing the paper's recipe.

    python -m repro.train --step 2 --arch densenet169

Defaults come from supplementary Table S2; every one of them can be overridden
on the command line, and whatever was actually used is written to
``config.json`` next to the checkpoints.

Outputs, under ``<DERMA_REPRO_OUT>/models/step_<n>/<architecture>[_<variant>]/``:

    final.pth               state_dict after the last epoch  <- the reported model
    best.pth                state_dict of the best test-accuracy epoch
    last_full.pth           optimiser/scheduler/RNG state, for --resume
    config.json             the resolved configuration
    history.json            per-epoch metrics, in full
    metrics.json            run summary: final and best reports
    metrics_final.json      metrics of final.pth alone
    metrics_best.json       metrics of best.pth alone
    predictions_final.csv   per-image prediction of final.pth
    predictions_best.csv    per-image prediction of best.pth

Every one of those is rewritten at the end of each epoch, so a run interrupted
at epoch 29 still leaves the metrics of the checkpoints it wrote -- weights and
the numbers describing them are never out of step.

`best.pth` is selected on the test set, because the delivered data has no
validation split. That is a test-set leak and the figure it produces is
optimistically biased; it is kept as the usual best-epoch checkpoint.
`final.pth` is the honest number. Both are reported.

Runs are reproducible: the same command with the same seed produces the same
weights, bit for bit, including across an interrupted-and-resumed run. See
`repro.seeding`. `metrics.json` records a fingerprint of the weights, so two
runs can be compared without diffing checkpoints.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from . import console, paths, seeding
from .config import ARCHITECTURES, TrainConfig
from .data import build_loaders, class_counts
from .engine import build_criterion, build_optimizer, build_scheduler, evaluate, train_one_epoch
from .metrics import full_report, write_predictions_csv
from .models import build_model, count_parameters, get_device


def default_workers() -> int:
    """One dataloader process per core, capped at 8.

    Oversubscribing hurts: on a 4-vCPU box, 8 workers contend for the same
    cores and slow decoding down. This is a throughput knob only -- the
    shuffling order is seeded independently of worker count, so no value here
    changes the trained weights.
    """
    import os

    return max(1, min(8, os.cpu_count() or 4))


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one model of the two-step trichoscopy framework.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--step", type=int, required=True, choices=(1, 2),
                        help="1 = diagnosis (5 classes), 2 = AA activity (3 classes)")
    parser.add_argument("--arch", required=True, choices=ARCHITECTURES)
    parser.add_argument("--epochs", type=int, default=30, help="Table S2: 30")
    parser.add_argument("--batch-size", type=int, default=32, help="Table S2: 32")
    parser.add_argument("--lr", type=float, default=0.001, help="Table S2: 0.001")
    parser.add_argument("--weight-decay", type=float, default=None,
                        help="default: Table S2 value for this step/architecture")
    parser.add_argument("--lr-step-size", type=int, default=2, help="Table S2: 2")
    parser.add_argument("--lr-gamma", type=float, default=0.7, help="Table S2: 0.7")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=default_workers(),
                        help="dataloader processes. Not a training setting: batch "
                             "composition is seeded, so this does not affect results "
                             "(verified byte-identical across worker counts). Defaults "
                             "to one per CPU core, capped at 8")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--amp", action="store_true",
                        help="mixed precision; ~2x faster and halves GPU memory")
    parser.add_argument("--no-deterministic", dest="deterministic", action="store_false",
                        help="let cuDNN pick algorithms by benchmarking: ~5-10%% faster, "
                             "but runs stop being bit-for-bit reproducible")
    parser.add_argument("--unique", dest="unique_only", action="store_true",
                        help="train on unique images only: drop byte-identical "
                             "duplicates left by the augmentation (see repro.dedup). "
                             "Writes to <arch>_unique/, so the baseline run is kept")
    parser.add_argument("--no-pretrained", action="store_true",
                        help="skip the ImageNet weights (ablation only)")
    parser.add_argument("--limit-train", type=int, default=None,
                        help="truncate the training set; smoke tests only")
    parser.add_argument("--output", type=Path, default=None,
                        help="default: <DERMA_REPRO_OUT>/models/step_<n>/<arch>[_<variant>]/")
    parser.add_argument("--resume", action="store_true",
                        help="continue from last_full.pth in the output folder")
    parser.add_argument("--log-every", type=int, default=50,
                        help="batches between progress lines; 0 disables")
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> TrainConfig:
    return TrainConfig(
        step=args.step,
        architecture=args.arch,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        lr_step_size=args.lr_step_size,
        lr_gamma=args.lr_gamma,
        seed=args.seed,
        num_workers=args.num_workers,
        amp=args.amp,
        device=args.device,
        deterministic=args.deterministic,
        unique_only=args.unique_only,
        pretrained=not args.no_pretrained,
    )


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _persist_metrics(output_dir: Path, config: TrainConfig, *, reports: dict,
                     elapsed_minutes: float, epochs_done: int,
                     dedup_stats: dict | None = None) -> dict:
    """Write the run summary and the per-checkpoint metric files.

    Called after every epoch, so the numbers on disk always describe the
    weights on disk.
    """
    payload = {
        "step": config.step,
        "architecture": config.architecture,
        "display_name": config.display_name,
        "config": config.to_dict(),
        "variant": config.variant,
        "epochs_completed": epochs_done,
        "training_minutes": elapsed_minutes,
        # What deduplication actually removed, recorded with the run so the
        # experiment can be read back without re-scanning the data.
        "dedup": dedup_stats,
        "reports": {name: block["report"] for name, block in reports.items()},
        "checkpoints": {
            name: {
                "epoch": block["epoch"],
                "fingerprint": block["fingerprint"],
                "accuracy": block["report"]["overall"]["accuracy"],
                "f1_weighted": block["report"]["overall"]["f1_weighted"],
            }
            for name, block in reports.items()
        },
    }
    _write_json(output_dir / "metrics.json", payload)
    for name, block in reports.items():
        _write_json(output_dir / f"metrics_{name}.json", {
            "step": config.step,
            "architecture": config.architecture,
            "checkpoint": f"{name}.pth",
            "epoch": block["epoch"],
            "fingerprint": block["fingerprint"],
            "report": block["report"],
        })
    return payload


def run(config: TrainConfig, output_dir: Path, *, resume: bool = False,
        limit_train: int | None = None, log_every: int = 50) -> dict:
    """Train one model. Returns the run summary that `metrics.json` holds."""
    paths.check_data_root()
    output_dir.mkdir(parents=True, exist_ok=True)

    seeding.configure_determinism(config.deterministic)
    seeding.set_seed(config.seed)

    device = get_device(config.device)
    if config.amp and device.type != "cuda":
        print("warning: --amp requires CUDA; continuing in full precision", flush=True)
        config.amp = False

    train_loader, test_loader, train_records, test_records = build_loaders(
        config, limit_train=limit_train
    )
    dedup_stats = getattr(build_loaders, "last_dedup_stats", None)
    if dedup_stats is not None:
        from .dedup import print_stats

        print_stats(dedup_stats)
    print(
        f"Step-{config.step} / {config.display_name} on {device}\n"
        f"  train {len(train_records):,} images "
        + ("(unique only, duplicates dropped)\n" if config.unique_only
           else "(augmented copies included)\n")
        + f"        {class_counts(train_records, config.step)}\n"
        f"  test  {len(test_records):,} images (originals only, as the paper reports)\n"
        f"        {class_counts(test_records, config.step)}\n"
        f"  epochs {config.epochs}  batch {config.batch_size}  lr {config.learning_rate}  "
        f"weight_decay {config.weight_decay}  StepLR({config.lr_step_size}, {config.lr_gamma})\n"
        f"  seed {config.seed}  deterministic {config.deterministic}  amp {config.amp}",
        flush=True,
    )

    # Re-seed so the freshly initialised head does not depend on how much
    # randomness building the loaders above happened to consume.
    seeding.set_seed(config.seed)
    model = build_model(config.architecture, config.num_classes, pretrained=config.pretrained)
    model.to(device)
    print(f"  trainable parameters: {count_parameters(model):,}", flush=True)

    criterion = build_criterion(config)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    scaler = torch.amp.GradScaler("cuda") if config.amp else None

    start_epoch = 1
    best_accuracy = -1.0
    best_epoch = None
    history: list[dict] = []
    elapsed_before = 0.0

    resume_path = output_dir / "last_full.pth"
    if resume and resume_path.exists():
        state = torch.load(resume_path, map_location=device, weights_only=True)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        if scaler is not None and state.get("scaler") is not None:
            scaler.load_state_dict(state["scaler"])
        start_epoch = state["epoch"] + 1
        best_accuracy = state["best_accuracy"]
        best_epoch = state.get("best_epoch")
        elapsed_before = state.get("elapsed_minutes", 0.0)

        history_path = output_dir / "history.json"
        if history_path.exists():
            history = json.loads(history_path.read_text(encoding="utf-8"))

        if not seeding.restore(state.get("rng")):
            print(
                "warning: this checkpoint carries no RNG state, so the resumed run "
                "will not match an uninterrupted one bit for bit",
                flush=True,
            )
        print(f"  resumed from epoch {state['epoch']}", flush=True)

    _write_json(output_dir / "config.json", config.to_dict())

    reports: dict[str, dict] = {}
    started = time.perf_counter()

    for epoch in range(start_epoch, config.epochs + 1):
        # Shuffling order is a function of (seed, epoch) alone, so a resumed
        # epoch sees exactly the batches an uninterrupted one would.
        seeding.seed_epoch(train_loader, epoch)
        learning_rate = optimizer.param_groups[0]["lr"]
        train_stats = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            scaler=scaler, log_every=log_every, epoch=epoch,
        )
        scheduler.step()

        predictions, _ = evaluate(model, test_loader, device, criterion)
        report = full_report(predictions, config.step)
        block = {
            "epoch": epoch,
            "fingerprint": seeding.fingerprint(model),
            "report": report,
        }
        test_accuracy = report["overall"]["accuracy"] / 100.0

        entry = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "train_loss": train_stats["loss"],
            "train_accuracy": train_stats["accuracy"],
            "train_seconds": train_stats["seconds"],
            "test_accuracy": test_accuracy,
            "fingerprint": block["fingerprint"],
            # The full per-class table for this epoch, not just the headline.
            "test_metrics": report,
        }

        is_best = test_accuracy > best_accuracy
        if is_best:
            best_accuracy = test_accuracy
            best_epoch = epoch
            entry["is_best"] = True
            torch.save(model.state_dict(), output_dir / "best.pth")
            reports["best"] = block
            write_predictions_csv(
                output_dir / "predictions_best.csv", test_records, predictions, config.step
            )

        history.append(entry)
        reports["final"] = block
        elapsed = elapsed_before + (time.perf_counter() - started) / 60

        # Weights first, then everything that describes them.
        torch.save(model.state_dict(), output_dir / "final.pth")
        write_predictions_csv(
            output_dir / "predictions_final.csv", test_records, predictions, config.step
        )
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict() if scaler is not None else None,
                "best_accuracy": best_accuracy,
                "best_epoch": best_epoch,
                "elapsed_minutes": elapsed,
                "rng": seeding.capture(),
            },
            output_dir / "last_full.pth",
        )
        _write_json(output_dir / "history.json", history)
        _persist_metrics(output_dir, config, reports=reports,
                         elapsed_minutes=elapsed, epochs_done=epoch,
                         dedup_stats=dedup_stats)

        print(
            f"epoch {epoch}/{config.epochs}  lr {learning_rate:.2e}  "
            f"train loss {entry['train_loss']:.4f} acc {entry['train_accuracy']:.4f}  "
            f"test acc {test_accuracy:.4f}{'  <- best' if is_best else ''}  "
            f"[{train_stats['seconds'] / 60:.1f} min]",
            flush=True,
        )

    total_minutes = elapsed_before + (time.perf_counter() - started) / 60

    # Nothing was trained (a completed run re-entered with --resume): read the
    # reports back from what is already on disk.
    for name in ("final", "best"):
        path = output_dir / f"metrics_{name}.json"
        if name not in reports and path.exists():
            stored = json.loads(path.read_text(encoding="utf-8"))
            reports[name] = {
                "epoch": stored["epoch"],
                "fingerprint": stored["fingerprint"],
                "report": stored["report"],
            }

    # Integrity check: best.pth must reproduce the metrics recorded for it.
    # Catches a checkpoint that failed to round-trip.
    best_path = output_dir / "best.pth"
    if "best" in reports and best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
        reloaded, _ = evaluate(model, test_loader, device)
        reloaded_accuracy = full_report(reloaded, config.step)["overall"]["accuracy"]
        recorded = reports["best"]["report"]["overall"]["accuracy"]
        if abs(reloaded_accuracy - recorded) > 1e-9:
            print(
                f"warning: best.pth re-evaluates to {reloaded_accuracy:.4f} but "
                f"{recorded:.4f} was recorded for it",
                flush=True,
            )

    payload = _persist_metrics(output_dir, config, reports=reports,
                               elapsed_minutes=total_minutes,
                               epochs_done=max(config.epochs, start_epoch - 1),
                               dedup_stats=dedup_stats)

    print(f"\ntraining finished in {total_minutes:.1f} min", flush=True)
    if "final" in reports:
        overall = reports["final"]["report"]["overall"]
        print(
            f"final-epoch weights (epoch {reports['final']['epoch']}, "
            f"fingerprint {reports['final']['fingerprint']}): "
            f"accuracy {overall['accuracy']:.2f}  weighted F1 {overall['f1_weighted']:.2f}",
            flush=True,
        )
    if "best" in reports:
        overall = reports["best"]["report"]["overall"]
        print(
            f"best-on-test weights (epoch {reports['best']['epoch']}, "
            f"fingerprint {reports['best']['fingerprint']}): "
            f"accuracy {overall['accuracy']:.2f}  weighted F1 {overall['f1_weighted']:.2f}"
            "  (test-selected, biased)",
            flush=True,
        )
    print(f"written to {output_dir}", flush=True)
    return payload


def main(argv=None) -> int:
    console.init()
    args = parse_args(argv)
    config = config_from_args(args)
    output_dir = args.output or paths.run_dir(
        config.step, config.architecture, config.variant
    )
    run(
        config,
        output_dir,
        resume=args.resume,
        limit_train=args.limit_train,
        log_every=args.log_every,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
