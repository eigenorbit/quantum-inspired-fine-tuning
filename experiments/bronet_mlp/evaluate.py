"""Recompute every metric of a finished run from its checkpoints, without training.

    python -m experiments.bronet_mlp.evaluate --run-dir <.../runs/step_2/butterfly/seed0>

Evaluates ``final.pt`` and, for an early-stopped run, ``best.pt``, and rewrites
metrics_<which>.json, predictions_<which>.csv, certificates_<which>.npz and
metrics.json (history.json and the checkpoints are left alone). Refuses when the
data manifest, the features or the BRONet checkpoint differ from the ones the run
was trained on.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from . import env

import torch

from . import train
from experiments.common import layout


def evaluate_run(run_dir: Path, logger=None, ignore_mtime: bool = False) -> dict:
    run_dir = Path(run_dir)
    final = torch.load(run_dir / "checkpoints" / "final.pt", map_location="cpu",
                       weights_only=True)
    cfg = train.RunConfig(**final["config"])
    t0 = time.perf_counter()
    inputs = (train.final_inputs(cfg.step, logger, ignore_mtime) if cfg.early_stop == "val"
              else train.prepare(cfg.step, logger, ignore_mtime))
    for key in ("bronet_ckpt_sha256", "features_index_sha256", "data_manifest_sha256",
                "mlp_base_sha256"):
        if final.get(key) != inputs.provenance.get(key):
            raise RuntimeError(f"{key} of {run_dir} differs from the current inputs; "
                               "the run was trained on different data/features")
    device = env.resolve_device(cfg.device)
    old = env.read_json(run_dir / "metrics.json") if (run_dir / "metrics.json").exists() else {}
    by_ckpt = {}
    for which in train.CHECKPOINTS:
        path = run_dir / layout.CHECKPOINTS_DIR / layout.checkpoint_file(which)
        if not path.exists():
            continue
        ck = torch.load(path, map_location="cpu", weights_only=True)
        model = train.load_run(run_dir, device=device, which=which)
        ev = train.evaluate(model, cfg, inputs, device)
        extra = {k: old[k] for k in ("early_stopping", "history_seconds_train_sum") if k in old}
        extra.update(config=cfg.to_dict(), provenance=inputs.provenance,
                     trainable_params=model.num_trainable(), fingerprint=ck["fingerprint"],
                     epoch=ck["epoch"], epochs_done=cfg.epochs,
                     dedup_stats=inputs.manifest["dedup_stats"],
                     device=env.device_info(device),
                     reevaluated_seconds=time.perf_counter() - t0)
        by_ckpt[which] = train.write_outputs(run_dir, ev, cfg, inputs, extra, which)
    return train.write_headline(run_dir, cfg, by_ckpt)


def main(argv=None) -> int:
    from repro import console

    console.init()
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--ignore-mtime", action="store_true")
    a = p.parse_args(argv)
    logger = env.setup_logging(a.run_dir / "evaluate.log")
    m = evaluate_run(a.run_dir, logger, a.ignore_mtime)
    cert = m["certificate_official"]
    logger.info(f"{m['headline']}: acc {m['accuracy']:.2f}  F1 {m['f1_weighted']:.2f}  "
                f"L/L0 {m['lipschitz']['L_over_L0']:.6g}  median r {cert['median_radius']:.4f}  "
                f"cert@36/255 {100 * cert['certified_accuracy_named']['36/255']:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
