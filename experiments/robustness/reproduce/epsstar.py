"""results/epsstar_seed{0..5}.npz -- per-image eps*, butterfly vs LoRA (Fig. 5b).

Usage:  python reproduce/epsstar.py SEED        (run for SEED = 0..5)

One backbone per seed, deep-copied into both arms so the only difference is the
adapter. eps* is found by doubling the budget until the attack succeeds, then
bisecting; images the model already gets wrong are 0, images no budget up to
eps_max breaks are +inf (censored, not clipped).
"""
import copy, math, os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np, torch
from core import build_loaders
from experiments import (_batch_margins, adapt, build_and_train, class_weights_for,
                         freeze_all, min_adversarial_distance, pgd_l2,
                         resolve_device, set_seed, train_adapted)

SEED = int(sys.argv[1])
N_IMAGES, EPOCHS, DEPTH = 200, 30, 32
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

out = {"y": y.cpu().numpy()}
for mode in ("butterfly", "lora"):
    set_seed(SEED + 1)
    m = adapt(copy.deepcopy(backbone), mode, loaders["num_classes"], 4,
              in_channels=loaders["in_channels"])
    m = train_adapted(m, loaders, device, EPOCHS, weight=w)
    freeze_all(m)
    L = m.analytic_lipschitz()
    g = torch.Generator(device=device).manual_seed(SEED + 777)
    eps_star, correct = min_adversarial_distance(
        m, x, y, attack=lambda M, a, b, e, **_: pgd_l2(M, a, b, e, 50, 2, g, decay=True),
        eps_start=0.25, eps_max=8.0, bisect=8)
    radius = _batch_margins(m, x, y).clamp_min(0) / (math.sqrt(2) * L)
    live = correct & torch.isfinite(eps_star)
    out[f"{mode}|eps_star"] = eps_star.cpu().numpy()
    out[f"{mode}|correct"] = correct.cpu().numpy()
    out[f"{mode}|radius"] = radius.cpu().numpy()
    out[f"{mode}|L"] = np.array([L])
    print(f"  s{SEED} {mode:10s} L {L:.6g}  median eps* "
          f"{float(eps_star[live].median()):.4f}  correct {int(correct.sum())}/{N_IMAGES}  "
          f"censored {int((correct & ~torch.isfinite(eps_star)).sum())}", flush=True)

np.savez_compressed(f"results/epsstar_seed{SEED}.npz", **out)
print("saved results/epsstar_seed%d.npz" % SEED)
