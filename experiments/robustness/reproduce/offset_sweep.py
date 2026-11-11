"""results/offset_sweep.csv -- certified radius and accuracy against the margin
the loss demands (Fig. 10).

Usage:  python reproduce/offset_sweep.py [SEED]      (default 0; article uses 0,1,2)

`train_adapted(..., offset=k)` computes cross-entropy on z - k*onehot(y), so the
model only scores well once its margin exceeds k. L is untouched by this, so the
sweep moves the numerator of r = margin / (sqrt(2) L) only.

Append the three seeds into one file afterwards:
    python -c "import glob,pandas as pd; \
      pd.concat([pd.read_csv(f) for f in sorted(glob.glob('results/offset_sweep_seed*.csv'))]) \
        .to_csv('results/offset_sweep.csv', index=False)"
"""
import copy, csv, math, os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
from core import build_loaders
from experiments import (_accuracy_detail, _batch_margins, adapt, build_and_train,
                         class_weights_for, freeze_all, pipeline_coverage,
                         resolve_device, set_seed, train_adapted)

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
OFFSETS = (0.5, 1.0, 2.0, 4.0, 8.0)
MODES_SWEPT = ("butterfly", "head-only")
N_IMAGES, EPOCHS, DEPTH = 200, 30, 32
OUT = f"results/offset_sweep_seed{SEED}.csv"
device = resolve_device()

backbone, _, _ = build_and_train("bloodmnist", "butterfly", DEPTH, SEED,
                                 epochs=EPOCHS, class_weights=True)
loaders = build_loaders("dermamnist", seed=SEED)
w = class_weights_for(loaders["train"], loaders["num_classes"], device)
xs, ys = [], []
for xb, yb in loaders["eval"]:
    xs.append(xb); ys.append(yb)
    if sum(t.shape[0] for t in xs) >= N_IMAGES: break
x = torch.cat(xs)[:N_IMAGES].to(device); y = torch.cat(ys)[:N_IMAGES].to(device)

rows = []
for mode in MODES_SWEPT:
    for off in OFFSETS:
        set_seed(SEED + 1)
        m = adapt(copy.deepcopy(backbone), mode, loaders["num_classes"], 4,
                  in_channels=loaders["in_channels"])
        m = train_adapted(m, loaders, device, EPOCHS, weight=w, offset=off)
        freeze_all(m)
        L = m.analytic_lipschitz()
        acc, bal, _ = _accuracy_detail(m, loaders["test"], device)
        with torch.no_grad():
            correct = m(x).argmax(1) == y
            radius = _batch_margins(m, x, y).clamp_min(0) / (math.sqrt(2) * L)
        cov = {r["operation"]: r["certified_unchanged"]
               for r in pipeline_coverage(m, x, y, L)}
        rows.append({"seed": SEED, "mode": mode, "offset": off,
                     "L_analytic": round(L, 6),
                     "clean_accuracy": round(acc, 4), "balanced_accuracy": round(bal, 4),
                     "median_radius": round(float(radius[correct].median()), 4),
                     "jpeg90": round(cov["jpeg q=90"], 3),
                     "jpeg80": round(cov["jpeg q=80"], 3),
                     "bit6": round(cov["6-bit depth"], 3),
                     "resize20": round(cov["resize 28->20->28"], 3)})
        print(rows[-1], flush=True)
        with open(OUT, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
print("saved", OUT)
