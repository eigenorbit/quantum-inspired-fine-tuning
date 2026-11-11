"""results/pipeline.csv -- does the certificate cover ordinary image handling?
JPEG re-encoding, bit-depth reduction, resize round-trips, sensor noise (Fig. 10).

Usage:  python reproduce/pipeline.py

For each operation T this records the median ||T(x) - x||_2, the fraction of
correctly classified images with ||T(x) - x||_2 < r(x) -- which the certificate
PROVES unchanged -- and the fraction whose prediction actually changes.
Both transfers, all four adaptation modes, seed 0.
"""
import csv, os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
from core import build_loaders
from experiments import (MODES, PIPELINE_CSV, adapt, build_and_train,
                         class_weights_for, freeze_all, pipeline_coverage,
                         resolve_device, set_seed, train_adapted)

CFG = [dict(source="mnist35", target="mnist08", epochs=10, weights=False),
       dict(source="bloodmnist", target="dermamnist", epochs=30, weights=True)]
N_IMAGES, DEPTH, SEED = 200, 32, 0
device = resolve_device()

rows = []
for cfg in CFG:
    for mode in MODES:
        backbone, _, _ = build_and_train(cfg["source"], "butterfly", DEPTH, SEED,
                                         epochs=cfg["epochs"],
                                         class_weights=cfg["weights"])
        loaders = build_loaders(cfg["target"], seed=SEED)
        set_seed(SEED + 1)
        m = adapt(backbone, mode, loaders["num_classes"], 4,
                  in_channels=loaders["in_channels"])
        w = (class_weights_for(loaders["train"], loaders["num_classes"], device)
             if cfg["weights"] else None)
        m = train_adapted(m, loaders, device, cfg["epochs"], weight=w)
        freeze_all(m)
        L = m.analytic_lipschitz()
        xs, ys = [], []
        for xb, yb in loaders["eval"]:
            xs.append(xb); ys.append(yb)
            if sum(t.shape[0] for t in xs) >= N_IMAGES: break
        x = torch.cat(xs)[:N_IMAGES].to(device); y = torch.cat(ys)[:N_IMAGES].to(device)
        g = torch.Generator(device=device).manual_seed(3)   # sensor-noise draws
        r = pipeline_coverage(m, x, y, L,
                              {"dataset": cfg["target"], "mode": mode, "L_analytic": L}, g)
        rows += r
        for row in r:
            print(f"  {cfg['target']:11s} {mode:10s} {row['operation']:20s} "
                  f"|d| {row['median_norm']:.4f}  certified-unchanged "
                  f"{row['certified_unchanged']:.1%}  changed "
                  f"{row['changed_prediction']:.1%}", flush=True)
        with open(PIPELINE_CSV, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
print("saved", PIPELINE_CSV)
