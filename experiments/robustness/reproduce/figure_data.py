"""results/figure_data_{mnist,derma}.npz -- one image, one shared budget, four
adapters (Fig. 4). Writes the arrays only; the notebook does the drawing.

Usage:  python reproduce/figure_data.py mnist
        python reproduce/figure_data.py derma

HOW THE EXAMPLE IMAGE IS CHOSEN (Fig. 4): from the first 64 evaluation images,
take those that (i) all four adapters classify correctly when clean, (ii) the
butterfly adapter still classifies correctly at the shared budget, and (iii) all
three other adapters get wrong at that same budget; then take the lowest index.
If no image satisfies (iii) for all three, the rule relaxes to at least two.
It is therefore a deliberately favourable example and the caption must say so --
`counts` in the npz carries the honest survivor count out of 64 for each arm.

The per-arm budgets in `eps_star` below are the median breaking points measured
in section 8.2; they are inputs to this figure, not results of it.
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np, torch
from core import MNIST_SUBSETS, build_loaders
from experiments import (MODES, adapt, adversarial_delta, build_and_train,
                         class_weights_for, freeze_all, resolve_device,
                         semantic_attack, set_seed, train_adapted)

WHICH = sys.argv[1]
CFG = {
    "mnist": dict(source="mnist35", target="mnist08", depth=32, epochs=10,
                  weights=False, shared_eps=1.3,
                  eps_star={"head-only": 1.10, "butterfly": 1.77, "lora": 0.99, "full": 1.13}),
    "derma": dict(source="bloodmnist", target="dermamnist", depth=32, epochs=30,
                  weights=True, shared_eps=0.35,
                  eps_star={"head-only": 0.35, "butterfly": 0.50, "lora": 0.15, "full": 0.15}),
}[WHICH]
SEED = 0
device = resolve_device()

models = {}
for mode in MODES:
    backbone, _, _ = build_and_train(CFG["source"], "butterfly", CFG["depth"], SEED,
                                     epochs=CFG["epochs"], class_weights=CFG["weights"])
    loaders = build_loaders(CFG["target"], seed=SEED)
    set_seed(SEED + 1)
    m = adapt(backbone, mode, loaders["num_classes"], 4, in_channels=loaders["in_channels"])
    w = (class_weights_for(loaders["train"], loaders["num_classes"], device)
         if CFG["weights"] else None)
    m = train_adapted(m, loaders, device, CFG["epochs"], weight=w)
    freeze_all(m)
    models[mode] = m
    print("trained", mode, flush=True)

x, y = next(iter(loaders["eval"]))
x, y = x[:64].to(device), y[:64].to(device)
digits = MNIST_SUBSETS.get(CFG["target"])
xn = float(x.flatten(1).norm(dim=1).median())

eps = CFG["shared_eps"]
deltas, flips = {}, {}
for mode, m in models.items():
    d, surv = adversarial_delta(m, x, y, eps, steps=100, restarts=5)
    deltas[mode], flips[mode] = d.cpu(), (~surv).cpu()
counts = {m: int(flips[m].sum()) for m in MODES}
print("flipped at eps", eps, counts, flush=True)

others = sum(flips[m].int() for m in MODES if m != "butterfly")
clean_ok = torch.stack([(models[m](x).argmax(1) == y).cpu() for m in MODES]).all(0)
cand = ((~flips["butterfly"]) & (others == 3) & clean_ok).nonzero().flatten()
if not len(cand):
    cand = ((~flips["butterfly"]) & (others >= 2) & clean_ok).nonzero().flatten()
i = int(cand[0]); print("chosen image", i, "label", int(y[i]))

sem = {}
for mode, m in models.items():
    d, _ = semantic_attack(m, x, y, steps=60, restarts=3)
    sem[mode] = d.cpu()

np.savez_compressed(
    f"results/figure_data_{WHICH}.npz",
    x=x.cpu().numpy(), y=y.cpu().numpy(), chosen=np.array([i]),
    shared_eps=np.array([eps]), x_norm=np.array([xn]),
    modes=np.array(list(MODES)),
    counts=np.array([counts[m] for m in MODES]),
    eps_star=np.array([CFG["eps_star"][m] for m in MODES]),
    labels=np.array(digits if digits else []),
    **{f"delta_{m}": deltas[m].numpy() for m in MODES},
    **{f"semantic_{m}": sem[m].numpy() for m in MODES},
    **{f"own_{m}": adversarial_delta(models[m], x[i:i+1], y[i:i+1],
                                     CFG["eps_star"][m], steps=100,
                                     restarts=5)[0].cpu().numpy() for m in MODES},
    preds=np.array([[int(models[m]((x[i] + deltas[m][i].to(device))[None]).argmax(1))
                     for m in MODES]]),
)
print(f"saved results/figure_data_{WHICH}.npz")
