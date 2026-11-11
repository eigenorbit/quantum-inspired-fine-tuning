"""The two experiments, and the measurement they share.

    §5  depth study        does a Lipschitz certificate survive depth?
                           three mixers x six depths -> depth.csv, perturbation.csv

    §6  adaptation study   does a certificate survive finetuning?
                           four adapters on one certified backbone -> adaptation.csv

    §7.6 baseline           what does an ordinary CNN get on the same data?
                           the reference point for how much the constraints cost
                           -> baseline.csv

Run either from the notebook (`RUN = True`) or from the command line:

    python experiments.py --experiment depth
    python experiments.py --experiment adaptation
    python experiments.py --experiment baseline --dataset dermamnist

Why both perturbations, at matched norm
---------------------------------------
A Lipschitz certificate bounds the WORST direction. A random direction in d
dimensions has expected alignment 1/sqrt(d) with any fixed direction, so random
noise probes average-case behaviour and says nothing about the certificate.
Holding ||d|| fixed and changing only the direction is what isolates the two.

Perturbations are never clamped to [0, 1]: the certificate is a statement about
unconstrained L2 balls, so clamping would test a weaker threat model than the
one being certified.
"""

import argparse
import csv
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from eigenkit.orthogonal.torch import Orthogonal

from core import (ARMS, ButterflyOrthogonal, DepthNet, build_loaders,
                  channels_for, freeze_all)
from init_draws import draw_butterfly_angles

__all__ = ["set_seed", "resolve_device", "append_csv", "build_and_train",
           "measure_model", "random_l2", "pgd_l2", "perturbation_curve",
           "run_one", "run_grid", "AdaptedMixer", "adapt", "run_adaptation",
           "run_adaptation_grid", "DEPTHS", "EPSILONS", "MODES",
           "DEPTH_CSV", "PERTURBATION_CSV", "ADAPT_CSV", "DERMA_DEPTHS",
           "DERMA_DEPTH_CSV", "DERMA_PERTURBATION_CSV", "DERMA_ADAPT_CSV",
           "BASELINE_CSV", "BASELINE_PERTURBATION_CSV", "SimpleCNN",
           "dihedral", "run_baseline", "run_baseline_grid",
           "TIGHTNESS_CSV", "TIGHTNESS_IMAGES_CSV", "ATTACK_STRENGTH_CSV",
           "min_adversarial_distance", "certificate_tightness", "blackbox_l2",
           "strong_pgd_l2", "pgd_l2_targeted", "run_tightness_depth",
           "run_tightness_adaptation", "run_attack_strength", "SEMANTIC_CSV",
           "SEMANTIC_RANGES", "adversarial_delta", "semantic_attack",
           "semantic_random", "scale_ranges", "augment_batch",
           "DERMA_TIGHTNESS_CSV", "DERMA_TIGHTNESS_IMAGES_CSV",
           "DERMA_ATTACK_STRENGTH_CSV", "PIPELINE_CSV", "pipeline_ops",
           "pipeline_coverage"]

DEPTHS = (4, 8, 16, 32, 64, 128)
EPSILONS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0)
MODES = ("head-only", "butterfly", "lora", "full")

DEPTH_CSV = "results/depth.csv"
PERTURBATION_CSV = "results/perturbation.csv"
ADAPT_CSV = "results/adaptation.csv"

# The DermaMNIST replication writes to its own files so the MNIST results, which
# are bit-for-bit reproducible from this code, are never touched.
DERMA_DEPTH_CSV = "results/derma_depth.csv"
DERMA_PERTURBATION_CSV = "results/derma_perturbation.csv"
DERMA_ADAPT_CSV = "results/derma_adaptation.csv"
DERMA_DEPTHS = (4, 16, 32, 64)
BASELINE_CSV = "results/baseline.csv"
BASELINE_PERTURBATION_CSV = "results/baseline_perturbation.csv"


# ---------------------------------------------------------------------------
# plumbing
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def resolve_device(spec: str = "auto") -> torch.device:
    if spec != "auto":
        return torch.device(spec)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def append_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    exists = os.path.exists(path)
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        if not exists:
            w.writeheader()
        w.writerows(rows)


def class_weights_for(loader, num_classes, device):
    """Inverse-frequency class weights.

    DermaMNIST is 67% melanocytic nevi and its rarest class has 80 training
    images. Without reweighting the loss is dominated by the majority class and
    even the *unconstrained* model barely beats a constant predictor -- measured
    at 0.24 balanced accuracy against 0.14 chance. With it, the same model
    reaches 0.54. The comparison between arms is only meaningful once the task
    is actually being learned.
    """
    counts = torch.bincount(torch.cat([y for _, y in loader]),
                            minlength=num_classes).float()
    return (counts.sum() / (num_classes * counts.clamp_min(1))).to(device)


def margin_loss(logits, y, scale: float = 8.0, offset: float = 1.0, weight=None):
    """Cross-entropy on ``scale * (z - offset * onehot(y))``.

    Plain cross-entropy is a poor fit for a 1-Lipschitz model, whose logits are
    inherently O(1): it is nearly linear there and barely separates examples.
    The scale and offset are applied ONLY inside the loss, so the deployed
    forward pass -- and therefore L_analytic -- is untouched. Every arm and every
    adapter uses this same loss, so the comparisons stay fair.
    """
    z = logits - offset * F.one_hot(y, logits.shape[1]).to(logits.dtype)
    return F.cross_entropy(z * scale, y, weight=weight)


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------

def dihedral(x: torch.Tensor) -> torch.Tensor:
    """A random element of the dihedral group: flips and quarter turns.

    Exact -- no interpolation, so no blurring and no boundary artefacts -- and
    label-preserving here, because a dermatoscope image has no canonical
    orientation. Applied per batch rather than per image, which is cheaper and,
    over enough batches, equivalent.
    """
    if torch.rand(()) < 0.5:
        x = x.flip(-1)
    if torch.rand(()) < 0.5:
        x = x.flip(-2)
    k = int(torch.randint(0, 4, ()))
    return torch.rot90(x, k, (-2, -1)) if k else x


def augment_batch(x, kind="dihedral", generator=None):
    """Training-time augmentation. ``dihedral`` is the free one (flips and quarter
    turns, exact); ``acquisition`` draws a random member of the same family of
    exposure / contrast / gamma / focus / tilt / white-balance changes that §8.5
    attacks, which is how you make a model invariant to it rather than merely
    certified against a ball that does not contain it.
    """
    if kind in ("dihedral", "both"):
        x = dihedral(x)
    if kind in ("acquisition", "both"):
        b, c = x.shape[:2]
        p = {}
        for k, (lo, hi) in SEMANTIC_RANGES.items():
            shape = (b, c) if k == "channel_gain" else (b,)
            u = torch.rand(shape, device=x.device, generator=generator)
            p[k] = lo + (hi - lo) * u
        x = _apply_semantic(x, p)
    return x


def _augment_kind(augment):
    """``True`` keeps meaning dihedral, so nothing committed changes."""
    if not augment:
        return None
    return "dihedral" if augment is True else str(augment)


def build_and_train(dataset: str, arm: str, depth: int, seed: int, epochs: int = 10,
                    lr: float = 1e-3, batch_size: int = 128, eval_subset: int = 2000,
                    scale: float = 8.0, offset: float = 1.0, device=None,
                    channels: int | None = None, class_weights: bool = False,
                    augment: bool = False):
    """Train one (arm, depth) from scratch and return (model, loaders, device).

    ``channels`` defaults to the smallest width the input allows: 64 for
    greyscale, 256 for RGB, since the network needs `lift >= in_channels` and the
    butterfly circuit needs a power of two.
    """
    device = device or resolve_device()
    loaders = build_loaders(dataset, batch_size=batch_size, seed=seed,
                            eval_subset=eval_subset)
    in_ch = loaders["in_channels"]
    channels = channels or channels_for(in_ch)
    set_seed(seed)
    model = DepthNet(loaders["num_classes"], arm, depth, channels=channels,
                     in_channels=in_ch).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    w = class_weights_for(loaders["train"], loaders["num_classes"], device) \
        if class_weights else None
    for _ in range(epochs):
        model.train()
        kind = _augment_kind(augment)
        for x, y in loaders["train"]:
            x, y = x.to(device), y.to(device)
            if kind:
                x = augment_batch(x, kind)
            opt.zero_grad(set_to_none=True)
            margin_loss(model(x), y, scale, offset, w).backward()
            opt.step()
    model.eval()
    return model, loaders, device


# ---------------------------------------------------------------------------
# measurement
# ---------------------------------------------------------------------------

@torch.no_grad()
def _accuracy(model, loader, device) -> float:
    correct = torch.zeros((), device=device, dtype=torch.long)
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum()
        total += y.numel()
    return float(correct) / max(total, 1)


@torch.no_grad()
def _margins(model, loader, device):
    """Clean logit margin M(x) = z_top1 - z_top2, and whether each is correct."""
    margins, correct = [], []
    for x, y in loader:
        z = model(x.to(device))
        top2 = z.topk(2, dim=1).values
        margins.append((top2[:, 0] - top2[:, 1]).cpu())
        correct.append((z.argmax(1).cpu() == y))
    return torch.cat(margins), torch.cat(correct)


@torch.no_grad()
def _accuracy_detail(model, loader, device) -> tuple[float, float, int]:
    """(accuracy, balanced accuracy, number of distinct predicted classes).

    Balanced accuracy -- the mean per-class recall -- is the honest metric when
    the classes are as skewed as DermaMNIST's 67/1 split, where a constant
    predictor scores 0.669 on plain accuracy.

    The distinct-class count is how ``collapsed`` is decided: a model that emits
    one class for every input is trivially unflippable, so its robustness
    numbers are meaningless whatever the metric says.
    """
    preds, ys = [], []
    for x, y in loader:
        preds.append(model(x.to(device)).argmax(1).cpu())
        ys.append(y)
    p, y = torch.cat(preds), torch.cat(ys)
    # Integer counting, not a float32 mean: averaging thousands of 0/1 floats
    # loses precision in the eighth decimal and would break bit-for-bit
    # reproducibility of the committed CSVs.
    accuracy = int((p == y).sum()) / max(y.numel(), 1)
    per_class = [int((p[y == c] == c).sum()) / int((y == c).sum())
                 for c in y.unique().tolist()]
    return accuracy, float(np.mean(per_class)), int(p.unique().numel())


@torch.no_grad()
def _majority_rate(loader) -> float:
    """Accuracy of always predicting the commonest class.

    A model that has collapsed to a constant predictor is trivially unflippable,
    so its robustness numbers are meaningless. Flag it rather than plot it.
    """
    counts: dict[int, int] = {}
    total = 0
    for _, y in loader:
        for c in y.tolist():
            counts[c] = counts.get(c, 0) + 1
        total += y.numel()
    return max(counts.values()) / max(total, 1)


@torch.no_grad()
def _check_bound(model, x, bound, scales=(0.01, 0.1, 1.0, 3.0)) -> tuple[bool, float]:
    """||f(x+d) - f(x)|| <= L ||d|| must never be violated.

    A certificate that is not empirically respected is worse than no certificate,
    so this runs on every trained model.
    """
    clean = model(x)
    worst = 0.0
    for s in scales:
        d = torch.randn_like(x) * s
        ratio = ((model(x + d) - clean).norm(dim=1)
                 / d.flatten(1).norm(dim=1).clamp_min(1e-12)).max().item()
        worst = max(worst, ratio)
    return worst <= bound * 1.0001, worst


def measure_model(model, loaders, device) -> dict:
    """Everything that does not need an attack."""
    bound = model.analytic_lipschitz()
    margins, _ = _margins(model, loaders["eval"], device)
    xb = next(iter(loaders["eval"]))[0][:200].to(device)
    ok, worst = _check_bound(model, xb, bound)
    smax, smin = model.worst_singular_values()
    majority = _majority_rate(loaders["eval"])
    clean, balanced, n_pred = _accuracy_detail(model, loaders["test"], device)
    return {
        "clean_accuracy": clean,
        "balanced_accuracy": balanced,
        "n_predicted_classes": n_pred,
        "norm_retention": model.norm_retention(xb),
        "L_analytic": bound,
        "median_margin": float(margins.median()),
        "certified_radius": float((margins / (math.sqrt(2) * bound)).median()),
        "mixer_params": model.mixer_parameters(),
        "mixer_sigma_max": smax, "mixer_sigma_min": smin,
        "bound_respected": ok, "worst_observed_ratio": worst,
        "majority_class_rate": majority,
        # A constant predictor, tested directly rather than inferred from a
        # metric: with class weights a well-trained model can score *below* the
        # majority rate and still be learning, which a plain accuracy-vs-majority
        # test would mislabel.
        "collapsed": n_pred == 1,
    }


# ---------------------------------------------------------------------------
# the two perturbations, at matched L2 norm
# ---------------------------------------------------------------------------

@torch.no_grad()
def random_l2(model, x, y, eps, trials: int = 4, generator=None):
    """Survival under random directions of L2 norm exactly eps."""
    survived = model(x).argmax(1) == y
    if eps == 0:
        return survived
    for _ in range(trials):
        d = torch.randn(x.shape, device=x.device, generator=generator)
        d = d / d.flatten(1).norm(dim=1).clamp_min(1e-12).view(-1, 1, 1, 1) * eps
        survived &= model(x + d).argmax(1) == y
    return survived


def _attack_loss(logits, y):
    """max_{j != y} z_j - z_y. Positive means the attack has succeeded."""
    true = logits.gather(1, y[:, None]).squeeze(1)
    other = logits.masked_fill(F.one_hot(y, logits.shape[1]).bool(),
                               float("-inf")).amax(dim=1)
    return other - true


def _as_eps(eps, x) -> torch.Tensor:
    """A per-image (B, 1, 1, 1) budget, from a scalar or a per-image tensor."""
    if torch.is_tensor(eps):
        return eps.reshape(-1, 1, 1, 1).to(x.dtype)
    return torch.full((x.shape[0], 1, 1, 1), float(eps), device=x.device,
                      dtype=x.dtype)


def _pgd_step_size(eps, steps, t, decay):
    """Constant 2.5*eps/steps, or a cosine anneal from eps/2 down to eps/200.

    The constant schedule is what every committed number was measured with, so
    it stays the default. The anneal takes large exploratory steps first and
    then refines, and is the single cheapest way to make the attack stronger.
    """
    if not decay:
        return 2.5 * eps / steps
    a0, a1 = 0.5 * eps, 0.005 * eps
    return a1 + 0.5 * (a0 - a1) * (1 + math.cos(math.pi * t / steps))


def pgd_l2(model, x, y, eps, steps: int = 50, restarts: int = 2, generator=None,
           decay: bool = False, target=None, momentum: float = 0.0):
    """Survival under a projected-gradient L2 attack of budget eps.

    ``eps`` is a scalar or a per-image tensor -- the per-image form is what the
    bisection in ``min_adversarial_distance`` needs. ``target`` makes the attack
    aim at one class instead of following the nearest boundary. The defaults
    reproduce the attack every committed CSV was measured with.
    """
    if not torch.is_tensor(eps) and eps == 0:
        with torch.no_grad():
            return model(x).argmax(1) == y
    e = _as_eps(eps, x)
    with torch.no_grad():
        survived = model(x).argmax(1) == y
    for _ in range(restarts):
        d = torch.randn(x.shape, device=x.device, generator=generator)
        d = d / d.flatten(1).norm(dim=1).clamp_min(1e-12).view(-1, 1, 1, 1) * e
        buf = torch.zeros_like(d)
        for t in range(steps):
            d = d.detach().requires_grad_(True)
            z = model(x + d)
            loss = (_attack_loss(z, y) if target is None
                    else z.gather(1, target[:, None]).squeeze(1)
                    - z.gather(1, y[:, None]).squeeze(1))
            (g,) = torch.autograd.grad(loss.sum(), d)
            if momentum:
                buf = momentum * buf + g / g.flatten(1).abs().sum(1).clamp_min(1e-12).view(-1, 1, 1, 1)
                g = buf
            gn = g.flatten(1).norm(dim=1).clamp_min(1e-12).view(-1, 1, 1, 1)
            d = d.detach() + _pgd_step_size(e, steps, t, decay) * g / gn
            dn = d.flatten(1).norm(dim=1).view(-1, 1, 1, 1).clamp_min(1e-12)
            d = d * (e / dn).clamp(max=1.0)
        with torch.no_grad():
            survived &= model(x + d).argmax(1) == y
    return survived


@torch.no_grad()
def _eval_batches(loader, device, limit):
    seen, out = 0, []
    for x, y in loader:
        if seen >= limit:
            break
        take = min(limit - seen, x.shape[0])
        out.append((x[:take].to(device), y[:take].to(device)))
        seen += take
    return out


def perturbation_curve(model, loaders, device, bound, n_images=500, steps=50,
                       restarts=2, seed=0) -> list[dict]:
    """Accuracy vs perturbation norm, for random and worst-case directions."""
    freeze_all(model)                       # attack the fixed function
    batches = _eval_batches(loaders["eval"], device, n_images)
    try:
        gen = torch.Generator(device=device).manual_seed(seed + 555)
    except (RuntimeError, TypeError):
        gen = None
    margins, correct = _margins(model, loaders["eval"], device)
    radii = margins / (math.sqrt(2) * bound)
    n = min(n_images, radii.numel())

    rows = []
    for eps in EPSILONS:
        rand = torch.cat([random_l2(model, x, y, eps, 4, gen).cpu() for x, y in batches])
        surv = torch.cat([pgd_l2(model, x, y, eps, steps, restarts, gen).cpu()
                          for x, y in batches])
        certified = float(((radii[:n] >= eps) & correct[:n]).float().mean()) if eps > 0 \
            else float(correct[:n].float().mean())
        rows.append({"epsilon": float(eps),
                     "random_accuracy": float(rand.float().mean()),
                     "robust_accuracy": float(surv.float().mean()),
                     "certified_accuracy": certified,
                     "n_images": int(surv.numel())})
    return rows


# ---------------------------------------------------------------------------
# experiment 1 (§5): does a certificate survive depth?
# ---------------------------------------------------------------------------

def run_one(dataset, arm, depth, seed, epochs=10, n_images=500, steps=50,
            restarts=2, attack=True, verbose=True, class_weights=False,
            augment=False, depth_csv=None, pert_csv=None) -> dict:
    depth_csv = depth_csv or DEPTH_CSV
    pert_csv = pert_csv or PERTURBATION_CSV
    t0 = time.time()
    model, loaders, device = build_and_train(dataset, arm, depth, seed, epochs,
                                             class_weights=class_weights,
                                             augment=augment)
    tag = {"dataset": dataset, "arm": arm, "depth": depth, "seed": seed,
           "epochs": epochs}
    stats = measure_model(model, loaders, device)
    if not stats["bound_respected"]:
        raise RuntimeError(f"{tag}: bound {stats['L_analytic']} violated "
                           f"(ratio {stats['worst_observed_ratio']})")
    append_csv(depth_csv, [{**tag, **stats, "seconds": time.time() - t0}])

    if attack:
        rows = perturbation_curve(model, loaders, device, stats["L_analytic"],
                                  n_images, steps, restarts, seed)
        append_csv(pert_csv,
                   [{**tag, **r, "collapsed": stats["collapsed"]} for r in rows])
    if verbose:
        flag = "   *** COLLAPSED to a constant predictor ***" if stats["collapsed"] else ""
        print(f"  d={depth:<4} {arm:10s} s{seed}  acc {stats['clean_accuracy']:.4f}  "
              f"bal {stats['balanced_accuracy']:.4f}  "
              f"L {stats['L_analytic']:.4g}  norm {stats['norm_retention']:.4f}  "
              f"cert_r {stats['certified_radius']:.3f}  "
              f"({time.time() - t0:.0f}s){flag}", flush=True)
    return stats


def run_grid(dataset="mnist35", arms=ARMS, depths=DEPTHS, seeds=2, epochs=10,
             attack=True, depth_csv=None, pert_csv=None, **kw):
    """Every (arm, depth, seed). Appends to the CSVs, so clear them for a fresh run."""
    depth_csv = depth_csv or DEPTH_CSV
    pert_csv = pert_csv or PERTURBATION_CSV
    print(f"device {resolve_device()}  ->  {depth_csv}, {pert_csv}", flush=True)
    for depth in depths:
        for arm in arms:
            for seed in range(seeds):
                run_one(dataset, arm, depth, seed, epochs, attack=attack,
                        depth_csv=depth_csv, pert_csv=pert_csv, **kw)


# ---------------------------------------------------------------------------
# experiment 2 (§6): does a certificate survive finetuning?
# ---------------------------------------------------------------------------

class AdaptedMixer(nn.Module):
    """One pretrained mixing layer, adapted in one of four ways.

    ``base`` is the pretrained C x C matrix, materialised from the backbone. In
    every mode except ``full`` it is frozen; what varies is what is composed with
    or added to it. Every mode is the identity at initialisation, so all four
    start from exactly the pretrained function -- see §6.2.
    """

    def __init__(self, base: torch.Tensor, bias: torch.Tensor, mode: str, rank: int = 4):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}, expected one of {MODES}")
        self.mode, self.channels = mode, base.shape[0]
        self._frozen_W = None

        if mode == "full":
            # Ordinary finetuning: the orthogonality of the pretrained weights was
            # a property of how they were *trained*; train them freely and they
            # leave the manifold immediately.
            self.weight = nn.Parameter(base.clone())
            self.bias = nn.Parameter(bias.clone())
        else:
            self.register_buffer("weight_frozen", base.clone())
            self.register_buffer("bias_frozen", bias.clone())

        if mode == "butterfly":
            # theta = 0 gives the identity, so adaptation starts exactly at the
            # pretrained function and the certificate holds from step zero. The
            # angles are drawn from the global generator and then set to zero
            # (see init_draws.py).
            draw_butterfly_angles([self.channels])
            self.adapter = Orthogonal(self.channels)
        elif mode == "lora":
            self.lora_A = nn.Parameter(torch.randn(rank, self.channels) * 0.01)
            self.lora_B = nn.Parameter(torch.zeros(self.channels, rank))  # zero at init

    def effective_matrix(self) -> torch.Tensor:
        """The single C x C matrix this layer applies, whatever the mode."""
        if self.mode == "full":
            return self.weight
        w = self.weight_frozen
        if self.mode == "butterfly":
            return self.adapter.matrix() @ w          # orthogonal o orthogonal
        if self.mode == "lora":
            return w + self.lora_B @ self.lora_A      # the standard low-rank update
        return w

    def _bias(self):
        return self.bias if self.mode == "full" else self.bias_frozen

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self._frozen_W if self._frozen_W is not None else self.effective_matrix()
        return F.conv2d(x, w.reshape(self.channels, self.channels, 1, 1), self._bias())

    def freeze_matrix(self) -> None:
        with torch.no_grad():
            self._frozen_W = self.effective_matrix().detach()

    def unfreeze_matrix(self) -> None:
        self._frozen_W = None

    def singular_values(self) -> tuple[float, float]:
        with torch.no_grad():
            s = torch.linalg.svdvals(self.effective_matrix().cpu().double())
            return float(s.max()), float(s.min())


def adapt(backbone: DepthNet, mode: str, num_classes: int, rank: int = 4,
          in_channels: int = 1) -> DepthNet:
    """Replace every mixer of a trained backbone with an AdaptedMixer, fresh head."""
    model = DepthNet(num_classes, backbone.arm, backbone.depth,
                     channels=backbone.channels, spatial=backbone.spatial,
                     in_channels=in_channels)
    new_layers = []
    for layer in backbone.mixers:
        if isinstance(layer, ButterflyOrthogonal):
            with torch.no_grad():
                base = layer.matrix().detach().clone()
                bias = (layer.bias.detach().clone() if layer.bias is not None
                        else torch.zeros(layer.channels))
            new_layers.append(AdaptedMixer(base, bias, mode, rank))
        else:
            new_layers.append(layer)                  # GroupSort / Roll: no parameters
    model.mixers = nn.Sequential(*new_layers)
    for p in model.parameters():                      # head stays trainable throughout
        p.requires_grad_(True)
    for m in model.modules():
        if isinstance(m, AdaptedMixer) and m.mode != "full":
            for name, p in m.named_parameters():
                if name.startswith(("weight", "bias")):
                    p.requires_grad_(False)
    return model


def train_adapted(model, loaders, device, epochs, lr=1e-3, scale=8.0, offset=1.0,
                  weight=None, augment=None):
    model.to(device)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    kind = _augment_kind(augment)
    for _ in range(epochs):
        model.train()
        for x, y in loaders["train"]:
            x, y = x.to(device), y.to(device)
            if kind:
                x = augment_batch(x, kind)
            opt.zero_grad(set_to_none=True)
            margin_loss(model(x), y, scale, offset, weight).backward()
            opt.step()
    model.eval()
    return model


def run_adaptation(source, target, mode, seed, epochs=10, depth=32, channels=None,
                   rank=4, n_images=500, steps=50, restarts=2,
                   class_weights=False, adapt_csv=None) -> list[dict]:
    """Certify a backbone on `source`, adapt it to `target`, measure what survives."""
    t0 = time.time()
    device = resolve_device()

    # 1. the certified backbone, trained on the source task
    adapt_csv = adapt_csv or ADAPT_CSV
    backbone, _, _ = build_and_train(source, "butterfly", depth, seed,
                                     epochs=epochs, channels=channels,
                                     class_weights=class_weights)
    base_L = backbone.analytic_lipschitz()

    # 2. adapt it to the target task
    loaders = build_loaders(target, seed=seed)
    set_seed(seed + 1)
    model = adapt(backbone, mode, loaders["num_classes"], rank,
                  in_channels=loaders["in_channels"])
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    w = (class_weights_for(loaders["train"], loaders["num_classes"], device)
         if class_weights else None)
    model = train_adapted(model, loaders, device, epochs, weight=w)

    # 3. measure
    L = model.analytic_lipschitz()
    acc, balanced, n_pred = _accuracy_detail(model, loaders["test"], device)
    margins, _ = _margins(model, loaders["eval"], device)
    radius = float((margins / (math.sqrt(2) * L)).median())
    smax = max(m.singular_values()[0] for m in model.modules()
               if isinstance(m, AdaptedMixer))

    freeze_all(model)
    x, y = next(iter(loaders["eval"]))
    x, y = x[:n_images].to(device), y[:n_images].to(device)
    rows = [{
        "source": source, "target": target, "mode": mode, "seed": seed,
        "depth": depth, "channels": backbone.channels, "epochs": epochs,
        "trainable_params": n_train, "backbone_L": base_L,
        "L_analytic": L, "worst_sigma_max": smax,
        "clean_accuracy": acc, "balanced_accuracy": balanced,
        "n_predicted_classes": n_pred, "collapsed": n_pred == 1,
        "certified_radius": radius,
        "epsilon": float(eps),
        "random_accuracy": float(random_l2(model, x, y, eps, 4).float().mean()),
        "robust_accuracy": float(pgd_l2(model, x, y, eps, steps, restarts).float().mean()),
        "seconds": time.time() - t0,
    } for eps in EPSILONS]
    append_csv(adapt_csv, rows)

    at = {r["epsilon"]: r["robust_accuracy"] for r in rows}
    print(f"  {mode:10s} s{seed}  acc {acc:.4f}  bal {balanced:.4f}  L {L:11.4g}  cert_r {radius:6.3f}  "
          f"PGD@1.0 {at[1.0]:.3f}  PGD@2.0 {at[2.0]:.3f}  "
          f"trainable {n_train:>7d}  ({time.time()-t0:.0f}s)", flush=True)
    return rows


def run_adaptation_grid(source="mnist35", target="mnist08", modes=MODES, seeds=2,
                        epochs=10, adapt_csv=None, **kw):
    """Every (mode, seed). Appends to the CSV, so clear it for a fresh run."""
    adapt_csv = adapt_csv or ADAPT_CSV
    print(f"device {resolve_device()}  |  {source} -> {target}  ->  {adapt_csv}",
          flush=True)
    for mode in modes:
        for seed in range(seeds):
            run_adaptation(source, target, mode, seed, epochs=epochs,
                           adapt_csv=adapt_csv, **kw)


# ---------------------------------------------------------------------------
# how tight is the certificate? (§5.6)
#
# Attacking AT the certified radius is a check on the implementation, not a
# measurement: below r(x) nothing can flip the label, by construction. The
# information is just above it. So for each image we search for eps*(x), the
# smallest budget at which the attack succeeds, and report
#
#     tightness(x) = eps*(x) / r(x)   >= 1 always, or there is a bug.
#
# A tightness near 1 means the proof captures essentially all of the robustness
# that is really there; a large one means the bound is vacuous even though L is
# finite. Because eps* comes from an attack it is an UPPER bound on the true
# minimum distance, so the measured tightness overstates the gap: a weak attack
# can only make a certificate look looser than it is, never tighter.
# ---------------------------------------------------------------------------

TIGHTNESS_CSV = "results/tightness.csv"
TIGHTNESS_IMAGES_CSV = "results/tightness_images.csv"
ATTACK_STRENGTH_CSV = "results/attack_strength.csv"

# The DermaMNIST half of the same two studies, kept in its own files.
DERMA_TIGHTNESS_CSV = "results/derma_tightness.csv"
DERMA_TIGHTNESS_IMAGES_CSV = "results/derma_tightness_images.csv"
DERMA_ATTACK_STRENGTH_CSV = "results/derma_attack_strength.csv"


@torch.no_grad()
def _batch_margins(model, x, y) -> torch.Tensor:
    """z_y - max_{j != y}, per image. Positive exactly when the image is right."""
    z = model(x)
    true = z.gather(1, y[:, None]).squeeze(1)
    other = z.masked_fill(F.one_hot(y, z.shape[1]).bool(), float("-inf")).amax(dim=1)
    return true - other


def pgd_l2_targeted(model, x, y, eps, num_classes, **kw):
    """PGD aimed at each wrong class in turn, surviving only if all of them fail.

    The untargeted margin loss follows whichever class is closest, which on a
    7-class problem leaves most of the decision boundary unexplored. Running one
    attack per target is the standard fix and is what makes the DermaMNIST
    numbers trustworthy.
    """
    with torch.no_grad():
        survived = model(x).argmax(1) == y
    for k in range(num_classes):
        mask = y != k
        if not mask.any():
            continue
        e = eps[mask] if torch.is_tensor(eps) else eps
        s = pgd_l2(model, x[mask], y[mask], e,
                   target=torch.full_like(y[mask], k), **kw)
        survived[mask] &= s
    return survived


def blackbox_l2(model, x, y, eps, queries: int = 2000, generator=None):
    """Gradient-free random search inside the eps-ball.

    Not AutoAttack's Square -- it is a plain accept-if-better random walk -- but
    it uses no gradients, which is the point: GroupSort is piecewise linear and
    a kinked loss surface can hide a vulnerability from a gradient attack. If
    this finds images PGD could not, the gradients are masked.
    """
    e = _as_eps(eps, x)
    with torch.no_grad():
        survived = model(x).argmax(1) == y
        if float(e.max()) == 0 or not survived.any():
            return survived
        d = torch.zeros_like(x)
        best = _attack_loss(model(x), y)
        for q in range(queries):
            step = 0.5 * e * (1.0 - q / queries) + 0.05 * e
            n = torch.randn(x.shape, device=x.device, generator=generator)
            n = n / n.flatten(1).norm(dim=1).clamp_min(1e-12).view(-1, 1, 1, 1)
            cand = d + step * n
            cn = cand.flatten(1).norm(dim=1).view(-1, 1, 1, 1).clamp_min(1e-12)
            cand = cand * (e / cn).clamp(max=1.0)
            loss = _attack_loss(model(x + cand), y)
            take = (loss > best).view(-1, 1, 1, 1)
            d = torch.where(take, cand, d)
            best = torch.maximum(best, loss)
        survived &= model(x + d).argmax(1) == y
    return survived


def strong_pgd_l2(model, x, y, eps, num_classes=None, steps=100, restarts=10,
                  generator=None, blackbox_queries=0):
    """Everything at once: decayed-step untargeted PGD, every target, random search."""
    survived = pgd_l2(model, x, y, eps, steps, restarts, generator, decay=True)
    if num_classes and num_classes > 2:
        survived &= pgd_l2_targeted(model, x, y, eps, num_classes, steps=steps,
                                    restarts=max(2, restarts // 3),
                                    generator=generator, decay=True)
    if blackbox_queries:
        survived &= blackbox_l2(model, x, y, eps, blackbox_queries, generator)
    return survived


def min_adversarial_distance(model, x, y, attack=None, eps_start=0.25,
                             eps_max=8.0, grow=6, bisect=8, **kw):
    """eps*(x): the smallest budget at which the attack flips each image.

    Returns (eps_star, clean_correct). Images the model already gets wrong are
    given 0; images no budget up to ``eps_max`` could break are given inf and
    counted as censored rather than silently clipped.
    """
    attack = attack or pgd_l2
    with torch.no_grad():
        correct = model(x).argmax(1) == y
    n = y.numel()
    lo = torch.zeros(n, device=x.device)
    hi = torch.full((n,), float("inf"), device=x.device)
    e = torch.full((n,), float(eps_start), device=x.device)

    for _ in range(grow):                       # bracket by doubling
        if not (torch.isinf(hi) & correct).any():
            break
        survived = attack(model, x, y, e, **kw)
        broke = torch.isinf(hi) & correct & ~survived
        hi = torch.where(broke, e, hi)
        lo = torch.where(torch.isinf(hi) & correct & survived, e, lo)
        e = torch.where(torch.isinf(hi), (e * 2).clamp(max=eps_max), e)

    for _ in range(bisect):                     # refine
        active = torch.isfinite(hi) & correct & (hi - lo > 1e-4)
        if not active.any():
            break
        mid = torch.where(active, 0.5 * (lo + hi), lo)
        survived = attack(model, x, y, mid, **kw)
        hi = torch.where(active & ~survived, mid, hi)
        lo = torch.where(active & survived, mid, lo)

    return torch.where(correct, hi, torch.zeros_like(hi)), correct


def certificate_tightness(model, loaders, device, bound, n_images=200, seed=0,
                          num_classes=None, steps=50, restarts=2, targeted=False,
                          eps_start=0.25, eps_max=8.0, bisect=8):
    """eps*(x) and r(x) for a sample of test images, and their ratio."""
    freeze_all(model)
    try:
        gen = torch.Generator(device=device).manual_seed(seed + 777)
    except (RuntimeError, TypeError):
        gen = None

    def attack(m, a, b, e, **_):
        if targeted and num_classes and num_classes > 2:
            return pgd_l2_targeted(m, a, b, e, num_classes, steps=steps,
                                   restarts=restarts, generator=gen, decay=True)
        return pgd_l2(m, a, b, e, steps, restarts, gen, decay=True)

    eps_star, margins, correct = [], [], []
    for x, y in _eval_batches(loaders["eval"], device, n_images):
        es, ok = min_adversarial_distance(model, x, y, attack=attack,
                                          eps_start=eps_start, eps_max=eps_max,
                                          bisect=bisect)
        eps_star.append(es.cpu()); correct.append(ok.cpu())
        margins.append(_batch_margins(model, x, y).cpu())

    eps_star = torch.cat(eps_star).double()
    correct = torch.cat(correct)
    radius = torch.cat(margins).double().clamp_min(0) / (math.sqrt(2) * bound)
    censored = torch.isinf(eps_star)
    usable = correct & ~censored & (radius > 0)
    ratio = (eps_star[usable] / radius[usable])

    per_image = [{"image": i, "correct": bool(c), "margin_radius": float(r),
                  "eps_star": float(e), "censored": bool(z)}
                 for i, (c, r, e, z) in enumerate(zip(correct, radius, eps_star, censored))]
    summary = {
        "n_images": int(correct.numel()),
        "n_correct": int(correct.sum()),
        "n_censored": int((censored & correct).sum()),
        "n_violations": int((correct & (eps_star < radius - 1e-6)).sum()),
        "median_radius": float(radius[correct].median()) if correct.any() else float("nan"),
        "median_eps_star": float(eps_star[usable].median()) if usable.any() else float("nan"),
        "median_tightness": float(ratio.median()) if usable.any() else float("nan"),
        "p90_tightness": float(ratio.quantile(0.9)) if usable.any() else float("nan"),
        "attack_steps": steps, "attack_restarts": restarts,
        "attack_targeted": targeted, "bisect": bisect,
    }
    return summary, per_image


def _log_tightness(tag, s):
    print(f"  {tag}  median r {s['median_radius']:.4f}  median eps* "
          f"{s['median_eps_star']:.4f}  tightness {s['median_tightness']:.2f}x  "
          f"(p90 {s['p90_tightness']:.2f}x)  censored {s['n_censored']}  "
          f"violations {s['n_violations']}", flush=True)


def run_tightness_depth(dataset="mnist35", arms=ARMS, depth=128, seeds=2, epochs=10,
                        n_images=200, class_weights=False, eps_max=8.0,
                        steps=50, restarts=2, targeted=False, csv=None, images_csv=None):
    """Train each arm at one depth, then measure how tight its certificate is."""
    csv = csv or TIGHTNESS_CSV
    images_csv = images_csv or TIGHTNESS_IMAGES_CSV
    print(f"device {resolve_device()}  ->  {csv}", flush=True)
    for arm in arms:
        for seed in range(seeds):
            t0 = time.time()
            model, loaders, device = build_and_train(dataset, arm, depth, seed, epochs,
                                                     class_weights=class_weights)
            bound = model.analytic_lipschitz()
            summary, per_image = certificate_tightness(
                model, loaders, device, bound, n_images, seed,
                loaders["num_classes"], steps, restarts, targeted, eps_max=eps_max)
            tag = {"dataset": dataset, "study": "depth", "model": arm, "depth": depth,
                   "seed": seed, "L_analytic": bound}
            append_csv(csv, [{**tag, **summary, "seconds": time.time() - t0}])
            append_csv(images_csv, [{**tag, **r} for r in per_image])
            _log_tightness(f"{arm:10s} d{depth} s{seed}", summary)


def run_tightness_adaptation(source="mnist35", target="mnist08", modes=MODES, seeds=2,
                             epochs=10, depth=32, channels=None, rank=4, n_images=200,
                             class_weights=False, eps_max=8.0, steps=50, restarts=2,
                             targeted=False, csv=None, images_csv=None):
    """The same measurement for the four adapters of §6/§7."""
    csv = csv or TIGHTNESS_CSV
    images_csv = images_csv or TIGHTNESS_IMAGES_CSV
    print(f"device {resolve_device()}  |  {source} -> {target}  ->  {csv}", flush=True)
    device = resolve_device()
    for mode in modes:
        for seed in range(seeds):
            t0 = time.time()
            backbone, _, _ = build_and_train(source, "butterfly", depth, seed,
                                             epochs=epochs, channels=channels,
                                             class_weights=class_weights)
            loaders = build_loaders(target, seed=seed)
            set_seed(seed + 1)
            model = adapt(backbone, mode, loaders["num_classes"], rank,
                          in_channels=loaders["in_channels"])
            w = (class_weights_for(loaders["train"], loaders["num_classes"], device)
                 if class_weights else None)
            model = train_adapted(model, loaders, device, epochs, weight=w)
            bound = model.analytic_lipschitz()
            summary, per_image = certificate_tightness(
                model, loaders, device, bound, n_images, seed,
                loaders["num_classes"], steps, restarts, targeted, eps_max=eps_max)
            tag = {"dataset": f"{source}->{target}", "study": "adaptation",
                   "model": mode, "depth": depth, "seed": seed, "L_analytic": bound}
            append_csv(csv, [{**tag, **summary, "seconds": time.time() - t0}])
            append_csv(images_csv, [{**tag, **r} for r in per_image])
            _log_tightness(f"{mode:10s} s{seed}", summary)


def run_attack_strength(dataset="mnist35", arm="butterfly", depth=128, seed=0,
                        epochs=10, epsilons=(1.0, 1.5), n_images=200,
                        class_weights=False, csv=None, model_spec=None):
    """Is our attack strong enough? Four attacks of increasing strength, same images.

    If the numbers barely move from `pgd-50x2` to `strong+blackbox`, the attack
    was already saturated and the robust accuracies can be trusted. If they drop
    a lot, every empirical number in the notebook is an overestimate.
    """
    csv = csv or ATTACK_STRENGTH_CSV
    model, loaders, device = build_and_train(dataset, arm, depth, seed, epochs,
                                             class_weights=class_weights)
    k = loaders["num_classes"]
    freeze_all(model)
    try:
        gen = torch.Generator(device=device).manual_seed(seed + 777)
    except (RuntimeError, TypeError):
        gen = None
    recipes = {
        "pgd-50x2 (committed)": lambda x, y, e: pgd_l2(model, x, y, e, 50, 2, gen),
        "pgd-100x10 + decay": lambda x, y, e: pgd_l2(model, x, y, e, 100, 10, gen, decay=True),
        "+ targeted": lambda x, y, e: strong_pgd_l2(model, x, y, e, k, 100, 10, gen),
        "+ black-box": lambda x, y, e: strong_pgd_l2(model, x, y, e, k, 100, 10, gen,
                                                     blackbox_queries=1500),
    }
    rows = []
    for eps in epsilons:
        for name, fn in recipes.items():
            t0 = time.time()
            survived = torch.cat([fn(x, y, eps).cpu()
                                  for x, y in _eval_batches(loaders["eval"], device, n_images)])
            rows.append({"dataset": dataset, "model": arm, "depth": depth, "seed": seed,
                         "epsilon": float(eps), "attack": name,
                         "robust_accuracy": float(survived.float().mean()),
                         "n_images": int(survived.numel()),
                         "seconds": time.time() - t0})
            print(f"  {arm:10s} eps {eps:<5.2f} {name:22s} "
                  f"{rows[-1]['robust_accuracy']:.4f}  ({rows[-1]['seconds']:.0f}s)",
                  flush=True)
    append_csv(csv, rows)
    return rows


# ---------------------------------------------------------------------------
# what does a perturbation of size eps actually look like? (§5.7)
#
# An L2 ball is not a picture. These two functions turn the numbers above into
# something you can look at: the adversarial perturbation itself, and the worst
# case inside a family of changes a real dermatoscope could produce.
# ---------------------------------------------------------------------------

SEMANTIC_CSV = "results/semantic.csv"

# Plausible acquisition variation, not an adversary with pixel-level freedom:
# a different light, a slightly misfocused lens, a tilted probe, a different
# white balance. The ranges are deliberately conservative.
SEMANTIC_RANGES = {
    "brightness": (-0.15, 0.15),
    "contrast": (0.80, 1.20),
    "gamma": (0.80, 1.25),
    "blur": (0.0, 1.5),            # pixels
    "rotation": (-10.0, 10.0),     # degrees
    "channel_gain": (0.90, 1.10),  # per colour channel; a white-balance shift
}


def adversarial_delta(model, x, y, eps, steps=100, restarts=5, generator=None,
                      decay=True):
    """The perturbation itself, not just whether it worked.

    Keeps, per image, the single worst perturbation found across restarts, so
    the figures show the attack that actually did the damage.
    """
    e = _as_eps(eps, x)
    best = torch.zeros_like(x)
    with torch.no_grad():
        best_loss = _attack_loss(model(x), y)
    for _ in range(restarts):
        d = torch.randn(x.shape, device=x.device, generator=generator)
        d = d / d.flatten(1).norm(dim=1).clamp_min(1e-12).view(-1, 1, 1, 1) * e
        for t in range(steps):
            d = d.detach().requires_grad_(True)
            (g,) = torch.autograd.grad(_attack_loss(model(x + d), y).sum(), d)
            gn = g.flatten(1).norm(dim=1).clamp_min(1e-12).view(-1, 1, 1, 1)
            d = d.detach() + _pgd_step_size(e, steps, t, decay) * g / gn
            dn = d.flatten(1).norm(dim=1).view(-1, 1, 1, 1).clamp_min(1e-12)
            d = d * (e / dn).clamp(max=1.0)
        with torch.no_grad():
            loss = _attack_loss(model(x + d), y)
            take = (loss > best_loss).view(-1, 1, 1, 1)
            best = torch.where(take, d, best)
            best_loss = torch.maximum(best_loss, loss)
    with torch.no_grad():
        survived = model(x + best).argmax(1) == y
    return best, survived


def _to_range(u, lo, hi):
    """An unconstrained parameter mapped into [lo, hi], so the search is free."""
    return lo + (hi - lo) * 0.5 * (1 + torch.tanh(u))


def _gaussian_blur(x, sigma, radius: int = 3):
    """Separable Gaussian blur with a per-image sigma, differentiable in sigma."""
    b, c, h, w = x.shape
    k = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    g = torch.exp(-(k[None, :] ** 2) / (2 * sigma[:, None].clamp_min(1e-2) ** 2))
    g = g / g.sum(1, keepdim=True)                       # (b, K)
    g = g.repeat_interleave(c, 0).view(b * c, 1, 1, -1)  # one kernel per image+channel
    z = x.reshape(1, b * c, h, w)
    z = F.conv2d(F.pad(z, (radius, radius, 0, 0), mode="replicate"), g, groups=b * c)
    z = F.conv2d(F.pad(z, (0, 0, radius, radius), mode="replicate"),
                 g.transpose(2, 3), groups=b * c)
    return z.reshape(b, c, h, w)


def _apply_semantic(x, p):
    """Rotate, defocus, gamma, contrast, white balance, exposure -- in that order."""
    th = p["rotation"] * math.pi / 180.0
    cos, sin, zero = torch.cos(th), torch.sin(th), torch.zeros_like(th)
    mat = torch.stack([torch.stack([cos, -sin, zero], 1),
                       torch.stack([sin, cos, zero], 1)], 1)
    grid = F.affine_grid(mat, list(x.shape), align_corners=False)
    h = F.grid_sample(x, grid, align_corners=False, padding_mode="border")
    h = _gaussian_blur(h, p["blur"])
    h = h.clamp_min(1e-6) ** p["gamma"].view(-1, 1, 1, 1)
    m = h.mean(dim=(1, 2, 3), keepdim=True)
    h = (h - m) * p["contrast"].view(-1, 1, 1, 1) + m
    h = h * p["channel_gain"].view(x.shape[0], x.shape[1], 1, 1)
    h = h + p["brightness"].view(-1, 1, 1, 1)
    return h.clamp(0.0, 1.0)      # a camera cannot produce a pixel outside [0, 1]


SEMANTIC_IDENTITY = {"brightness": 0.0, "contrast": 1.0, "gamma": 1.0,
                     "blur": 0.0, "rotation": 0.0, "channel_gain": 1.0}


def scale_ranges(scale: float, ranges=None) -> dict:
    """Shrink or widen the family around the identity transform."""
    ranges = ranges or SEMANTIC_RANGES
    return {k: (SEMANTIC_IDENTITY[k] + scale * (lo - SEMANTIC_IDENTITY[k]),
                SEMANTIC_IDENTITY[k] + scale * (hi - SEMANTIC_IDENTITY[k]))
            for k, (lo, hi) in ranges.items()}


def semantic_random(model, x, y, ranges=None, generator=None):
    """A RANDOM draw from the acquisition family, not its worst corner.

    The control for ``semantic_attack``. Section 5.4's lesson applies here too:
    a random member of a family and the worst member of the same family are very
    different things, and only one of them is what a clinic would actually hand
    you.
    """
    ranges = ranges or SEMANTIC_RANGES
    b, c = x.shape[:2]
    with torch.no_grad():
        p = {}
        for k, (lo, hi) in ranges.items():
            shape = (b, c) if k == "channel_gain" else (b,)
            u = torch.rand(shape, device=x.device, generator=generator)
            p[k] = lo + (hi - lo) * u
        adv = _apply_semantic(x, p)
        return adv - x, model(adv).argmax(1) == y


def semantic_attack(model, x, y, steps: int = 60, lr: float = 0.15,
                    restarts: int = 3, ranges=None, generator=None):
    """The worst image a plausible acquisition change could have produced.

    Six parameters -- exposure, contrast, gamma, focus, probe angle, white
    balance -- optimised by gradient ascent on the margin, inside ranges a real
    dermatoscope stays within. Unlike a random corruption this is adversarial,
    and unlike PGD it cannot leave the manifold of images a camera can produce.
    Returns (delta, survived); ``delta.flatten(1).norm(dim=1)`` is the L2 size
    that places it against the certified radius.
    """
    ranges = ranges or SEMANTIC_RANGES
    b, c = x.shape[:2]
    best = torch.zeros_like(x)
    with torch.no_grad():
        best_loss = _attack_loss(model(x), y)
    for _ in range(restarts):
        us = {k: (0.4 * torch.randn((b, c) if k == "channel_gain" else (b,),
                                    device=x.device, generator=generator)
                  ).requires_grad_(True) for k in ranges}
        opt = torch.optim.Adam(list(us.values()), lr=lr)
        for _ in range(steps):
            p = {k: _to_range(us[k], *ranges[k]) for k in ranges}
            loss = _attack_loss(model(_apply_semantic(x, p)), y).sum()
            opt.zero_grad(set_to_none=True)
            (-loss).backward()
            opt.step()
        with torch.no_grad():
            p = {k: _to_range(us[k], *ranges[k]) for k in ranges}
            adv = _apply_semantic(x, p)
            loss = _attack_loss(model(adv), y)
            take = (loss > best_loss).view(-1, 1, 1, 1)
            best = torch.where(take, adv - x, best)
            best_loss = torch.maximum(best_loss, loss)
    with torch.no_grad():
        survived = model(x + best).argmax(1) == y
    return best, survived


# ---------------------------------------------------------------------------
# does the certificate cover anything that actually happens? (§8.9)
#
# §8.5 showed the certified radius is far too small for a change of lighting or
# white balance. But a clinical image is also altered by things nobody thinks of
# as perturbations at all: a PACS re-encodes it to JPEG, a viewer resizes it, a
# sensor adds read noise. Those are small. If their L2 norm falls below r(x), the
# certificate PROVES the diagnosis is unchanged -- which is a claim about
# reproducibility rather than about adversaries, and one a hospital can use.
# ---------------------------------------------------------------------------

PIPELINE_CSV = "results/pipeline.csv"


def _jpeg_roundtrip(x: torch.Tensor, quality: int) -> torch.Tensor:
    """Encode to JPEG at the given quality and decode again, as a PACS would."""
    import io
    from PIL import Image
    out = []
    for img in x.detach().cpu():
        a = (img.permute(1, 2, 0).numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
        pil = Image.fromarray(a[:, :, 0] if a.shape[2] == 1 else a)
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        b = np.asarray(Image.open(buf)).astype(np.float32) / 255.0
        if b.ndim == 2:
            b = b[:, :, None]
        out.append(torch.from_numpy(b).permute(2, 0, 1))
    return torch.stack(out).to(x.device)


def _requantise(x: torch.Tensor, bits: int) -> torch.Tensor:
    levels = 2 ** bits - 1
    return (x * levels).round() / levels


def _resize_roundtrip(x: torch.Tensor, size: int) -> torch.Tensor:
    h = F.interpolate(x, size=(size, size), mode="bicubic", align_corners=False)
    return F.interpolate(h, size=x.shape[-2:], mode="bicubic",
                         align_corners=False).clamp(0, 1)


def pipeline_ops(generator=None) -> dict:
    """Things that happen to a clinical image without anyone calling it an attack."""
    return {
        "jpeg q=100": lambda x: _jpeg_roundtrip(x, 100),
        "jpeg q=95": lambda x: _jpeg_roundtrip(x, 95),
        "jpeg q=90": lambda x: _jpeg_roundtrip(x, 90),
        "jpeg q=80": lambda x: _jpeg_roundtrip(x, 80),
        "7-bit depth": lambda x: _requantise(x, 7),
        "6-bit depth": lambda x: _requantise(x, 6),
        "resize 28->24->28": lambda x: _resize_roundtrip(x, 24),
        "resize 28->20->28": lambda x: _resize_roundtrip(x, 20),
        "resize 28->14->28": lambda x: _resize_roundtrip(x, 14),
        "sensor noise 0.002": lambda x: (x + 0.002 * torch.randn(
            x.shape, device=x.device, generator=generator)).clamp(0, 1),
        "sensor noise 0.005": lambda x: (x + 0.005 * torch.randn(
            x.shape, device=x.device, generator=generator)).clamp(0, 1),
    }


@torch.no_grad()
def pipeline_coverage(model, x, y, bound, tag=None, generator=None) -> list[dict]:
    """For each operation: how big is it, and does the certificate already cover it?

    ``certified_unchanged`` is the fraction of images for which the perturbation
    is provably harmless -- no experiment needed, the theorem settles it.
    ``changed`` is how often the prediction actually moved.
    """
    radius = (_batch_margins(model, x, y).clamp_min(0) / (math.sqrt(2) * bound))
    correct = model(x).argmax(1) == y
    base = model(x).argmax(1)
    rows = []
    for name, op in pipeline_ops(generator).items():
        xa = op(x)
        d = (xa - x).flatten(1).norm(dim=1)
        pred = model(xa).argmax(1)
        inside = correct & (d < radius)
        rows.append({
            **(tag or {}), "operation": name,
            "median_norm": float(d.median()), "max_norm": float(d.max()),
            "median_radius": float(radius[correct].median()),
            "certified_unchanged": float(inside.float().sum() / correct.sum()),
            "accuracy_before": float(correct.float().mean()),
            "accuracy_after": float((pred == y).float().mean()),
            "changed_prediction": float((pred != base).float().mean()),
            "n_images": int(y.numel()),
        })
    return rows


# ---------------------------------------------------------------------------
# the reference point (§7.6): what does an ordinary CNN get on the same data?
# ---------------------------------------------------------------------------

class SimpleCNN(nn.Module):
    """The convnet anyone would write for 28x28: conv-BN-ReLU, pooling, dropout.

    Not a fourth arm. It has no Lipschitz constraint, no certificate and no
    isometric stem, so it answers a different question: how much of the low
    accuracy on DermaMNIST is the constrained architecture, and how much is the
    dataset and the metric? See §7.6.
    """

    def __init__(self, num_classes: int, in_channels: int = 3, width: int = 32):
        super().__init__()

        def block(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                nn.MaxPool2d(2))

        w = width
        self.features = nn.Sequential(block(in_channels, w), block(w, 2 * w),
                                      block(2 * w, 4 * w))
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.3), nn.Linear(4 * w * 3 * 3, 128),
            nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, num_classes))

    def forward(self, x):
        return self.head(self.features(x))


def _baseline_model(name: str, loaders):
    """``simple-cnn``, or one of our arms as ``<arm>-d<depth>``."""
    if name == "simple-cnn":
        return SimpleCNN(loaders["num_classes"], loaders["in_channels"]), "ce"
    arm, depth = name.rsplit("-d", 1)
    return DepthNet(loaders["num_classes"], arm, int(depth),
                    channels=channels_for(loaders["in_channels"]),
                    in_channels=loaders["in_channels"]), "margin"


def run_baseline(model_name="simple-cnn", dataset="dermamnist", seed=0, epochs=30,
                 class_weights=True, augment=False, schedule=False, loss=None,
                 lr=1e-3, scale=8.0, offset=1.0, verbose=True, attack=False,
                 n_images=300, steps=50, restarts=2, baseline_csv=None,
                 pert_csv=None) -> dict:
    """Train one model under one training recipe; report BOTH accuracies.

    The point of this grid is §7.6: how much of the low DermaMNIST accuracy is
    the Lipschitz constraint, and how much is the dataset, the metric and the
    training recipe? So the same knobs -- class weights, augmentation, schedule
    -- are applied to an unconstrained CNN and to our arms alike.
    """
    t0 = time.time()
    baseline_csv = baseline_csv or BASELINE_CSV
    device = resolve_device()
    loaders = build_loaders(dataset, seed=seed)
    set_seed(seed)
    model, default_loss = _baseline_model(model_name, loaders)
    model = model.to(device)
    loss = loss or default_loss
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
             if schedule else None)
    w = (class_weights_for(loaders["train"], loaders["num_classes"], device)
         if class_weights else None)
    for _ in range(epochs):
        model.train()
        for x, y in loaders["train"]:
            x, y = x.to(device), y.to(device)
            if augment:
                x = dihedral(x)
            opt.zero_grad(set_to_none=True)
            out = model(x)
            (F.cross_entropy(out, y, weight=w) if loss == "ce"
             else margin_loss(out, y, scale, offset, w)).backward()
            opt.step()
        if sched:
            sched.step()
    model.eval()
    acc, balanced, n_pred = _accuracy_detail(model, loaders["test"], device)
    row = {"dataset": dataset, "model": model_name, "seed": seed,
           "epochs": epochs, "loss": loss, "class_weights": class_weights,
           "augment": augment, "schedule": schedule, "margin_offset": offset,
           "clean_accuracy": acc,
           "balanced_accuracy": balanced, "n_predicted_classes": n_pred,
           "params": sum(p.numel() for p in model.parameters()),
           "seconds": time.time() - t0}
    append_csv(baseline_csv, [row])

    if attack:
        # The CNN has no analytic bound, so certified accuracy is 0 by
        # construction -- an infinite bound certifies nothing. The random and
        # PGD columns are what make it a reference line in the §5.5/§7.3 plots.
        tag = {k: row[k] for k in ("dataset", "model", "seed", "epochs",
                                   "class_weights", "augment", "schedule")}
        rows = perturbation_curve(model, loaders, device, float("inf"),
                                  n_images, steps, restarts, seed)
        append_csv(pert_csv or BASELINE_PERTURBATION_CSV,
                   [{**tag, **r, "clean_accuracy": acc,
                     "balanced_accuracy": balanced} for r in rows])

    if verbose:
        print(f"  {model_name:14s} {loss:6s} w{int(class_weights)} "
              f"a{int(augment)} s{int(schedule)} o{offset:g} e{epochs:<3d} "
              f"acc {acc:.4f}  bal {balanced:.4f}  "
              f"classes {n_pred}/{loaders['num_classes']}  "
              f"({row['seconds']:.0f}s)", flush=True)
    return row


BASELINE_MODELS = ("simple-cnn", "dense-d4", "butterfly-d4", "butterfly-d32")


def run_baseline_grid(dataset="dermamnist", seeds=2, epochs=30,
                      models=BASELINE_MODELS, attack_models=("simple-cnn",),
                      baseline_csv=None, pert_csv=None):
    """Every model under three recipes, plus a longer run and a loss ablation.

    ``attack_models`` also get a perturbation curve, so the unconstrained CNN can
    be drawn as a reference line in the robustness figures. Our own arms are
    attacked by the depth study already, so attacking them again here would only
    duplicate work.
    """
    baseline_csv = baseline_csv or BASELINE_CSV
    print(f"device {resolve_device()}  ->  {baseline_csv}", flush=True)
    for seed in range(seeds):
        for name in models:
            hit = name in attack_models
            for class_weights, augment in ((False, False), (True, False),
                                           (True, True)):
                run_baseline(name, dataset, seed, epochs, class_weights,
                             augment, attack=hit, baseline_csv=baseline_csv,
                             pert_csv=pert_csv)
            # does simply training longer, with a schedule, close the gap?
            run_baseline(name, dataset, seed, 2 * epochs, True, True,
                         schedule=True, attack=hit, baseline_csv=baseline_csv,
                         pert_csv=pert_csv)
        # the margin loss's offset was tuned on two-class MNIST; is it wrong here?
        if "butterfly-d4" in models:
            run_baseline("butterfly-d4", dataset, seed, epochs, True, False,
                         offset=0.5, baseline_csv=baseline_csv)


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment",
                   choices=("depth", "adaptation", "baseline", "tightness",
                            "tightness-adapt", "attack-strength"),
                   default="depth")
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--n-images", type=int, default=500)
    # depth study
    p.add_argument("--dataset", default="mnist35")
    p.add_argument("--arms", nargs="+", default=list(ARMS))
    p.add_argument("--depths", type=int, nargs="+", default=list(DEPTHS))
    p.add_argument("--no-attack", action="store_true")
    # adaptation study
    p.add_argument("--source", default="mnist35", help="task the backbone is certified on")
    p.add_argument("--target", default="mnist08", help="task to adapt to")
    p.add_argument("--modes", nargs="+", default=list(MODES))
    p.add_argument("--depth", type=int, default=32)

    # needed by the DermaMNIST replication of section 7: an inverse-frequency
    # weighted loss, and its own CSVs so the MNIST results are left alone.
    p.add_argument("--class-weights", action="store_true",
                   help="weight the loss by inverse class frequency (imbalanced data)")
    p.add_argument("--out-prefix", default="",
                   help="prefix for the result filenames, e.g. 'derma_'")
    p.add_argument("--augment", action="store_true",
                   help="random flips and quarter turns during training")
    p.add_argument("--models", nargs="+", default=list(BASELINE_MODELS),
                   help="which models the baseline grid trains")
    p.add_argument("--n-attack-images", type=int, default=200,
                   help="images used by the tightness and attack-strength studies")
    p.add_argument("--eps-max", type=float, default=8.0,
                   help="largest budget the minimum-distance search will try")
    p.add_argument("--targeted", action="store_true",
                   help="also attack every wrong class (slow, stronger)")
    p.add_argument("--attack-epsilons", type=float, nargs="+", default=[1.0, 1.5],
                   help="budgets for the attack-strength ladder")
    a = p.parse_args()

    pre = a.out_prefix
    if a.experiment == "tightness":
        run_tightness_depth(a.dataset, a.arms, a.depth, a.seeds, a.epochs,
                            a.n_attack_images, a.class_weights, a.eps_max,
                            targeted=a.targeted,
                            csv=f"results/{pre}tightness.csv",
                            images_csv=f"results/{pre}tightness_images.csv")
    elif a.experiment == "tightness-adapt":
        run_tightness_adaptation(a.source, a.target, a.modes, a.seeds, a.epochs,
                                 a.depth, n_images=a.n_attack_images,
                                 class_weights=a.class_weights, eps_max=a.eps_max,
                                 targeted=a.targeted,
                                 csv=f"results/{pre}tightness.csv",
                                 images_csv=f"results/{pre}tightness_images.csv")
    elif a.experiment == "attack-strength":
        for arm in a.arms:
            run_attack_strength(a.dataset, arm, a.depth, 0, a.epochs,
                                epsilons=tuple(a.attack_epsilons),
                                n_images=a.n_attack_images,
                                class_weights=a.class_weights,
                                csv=f"results/{pre}attack_strength.csv")
    elif a.experiment == "baseline":
        run_baseline_grid(a.dataset, a.seeds, a.epochs, models=a.models,
                          baseline_csv=f"results/{pre}baseline.csv",
                          pert_csv=f"results/{pre}baseline_perturbation.csv")
    elif a.experiment == "depth":
        run_grid(a.dataset, a.arms, a.depths, a.seeds, a.epochs,
                 attack=not a.no_attack, n_images=a.n_images,
                 class_weights=a.class_weights, augment=a.augment,
                 depth_csv=f"results/{pre}depth.csv",
                 pert_csv=f"results/{pre}perturbation.csv")
    else:
        run_adaptation_grid(a.source, a.target, a.modes, a.seeds, a.epochs,
                            depth=a.depth, n_images=a.n_images,
                            class_weights=a.class_weights,
                            adapt_csv=f"results/{pre}adaptation.csv")


if __name__ == "__main__":
    main()
