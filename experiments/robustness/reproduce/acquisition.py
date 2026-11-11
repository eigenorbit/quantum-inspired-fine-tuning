"""results/acquisition_seed{0..5}.npz -- per-image survival under the acquisition
family (exposure, contrast, gamma, focus, tilt, white balance), random draw and
worst case. NOT used in the article; kept because section 8.4 reports it.

Usage:  python reproduce/acquisition.py SEED        (run for SEED = 0..5)
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np, torch
from core import build_loaders
from experiments import (MODES, adapt, build_and_train, class_weights_for,
                         freeze_all, resolve_device, scale_ranges,
                         semantic_attack, semantic_random, set_seed, train_adapted)

SEED = int(sys.argv[1])
SCALES = (0.0, 0.1, 0.2, 0.3, 0.4, 0.6)
N_IMAGES, EPOCHS, DEPTH = 256, 30, 32
device = resolve_device()

out = {}
for mode in MODES:
    backbone, _, _ = build_and_train("bloodmnist", "butterfly", DEPTH, SEED,
                                     epochs=EPOCHS, class_weights=True)
    loaders = build_loaders("dermamnist", seed=SEED)
    set_seed(SEED + 1)
    m = adapt(backbone, mode, loaders["num_classes"], 4, in_channels=loaders["in_channels"])
    w = class_weights_for(loaders["train"], loaders["num_classes"], device)
    m = train_adapted(m, loaders, device, EPOCHS, weight=w)
    freeze_all(m)
    xs, ys = [], []
    for xb, yb in loaders["eval"]:
        xs.append(xb); ys.append(yb)
        if sum(t.shape[0] for t in xs) >= N_IMAGES: break
    x = torch.cat(xs)[:N_IMAGES].to(device); y = torch.cat(ys)[:N_IMAGES].to(device)
    g = torch.Generator(device=device).manual_seed(11)
    for s in SCALES:
        if s == 0.0:
            surv = (m(x).argmax(1) == y); rnd = surv
        else:
            r = scale_ranges(s)
            _, rnd = semantic_random(m, x, y, r, g)
            _, surv = semantic_attack(m, x, y, steps=60, restarts=3, ranges=r, generator=g)
        out[f"{mode}|{s}|worst"] = surv.cpu().numpy()
        out[f"{mode}|{s}|random"] = rnd.cpu().numpy()
        print(f"  s{SEED} {mode:10s} sev {s:.1f}  worst {float(surv.float().mean()):.4f}"
              f"  random {float(rnd.float().mean()):.4f}", flush=True)
np.savez_compressed(f"results/acquisition_seed{SEED}.npz", y=y.cpu().numpy(), **out)
print("saved results/acquisition_seed%d.npz" % SEED)
