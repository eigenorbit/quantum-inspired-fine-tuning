"""Lipschitz constant of the MLP, certified radii, certified accuracy.

Certificate (BRONet's `margin_layer.get_predictions`): with prediction p, class
j cannot overtake p under any ||delta||_2 <= eps if

    z_p - z_j > L * eps * ||w_p - w_j||,

w the unit-norm rows of the LLN head and L the Lipschitz constant of everything
before the head, here L_total = L_pre * L_mlp. Hence the radius

    r(x) = min_{j != p} (z_p - z_j) / (L * ||w_p - w_j||),

and an image counts as certified at eps when it is classified correctly AND
r(x) > eps. The radius is in the 224x224 input space of BRONet (pixels in
[0, 1]); in the space of the 600x600 files it is r / 0.37458, the operator norm
of the 600 -> 224 bilinear resize (experiments/bronet_mlp/reference/preprocessing_lipschitz.py).
"""

from __future__ import annotations

import copy
import math

import numpy as np
import torch
from eigenkit.orthogonal.torch import certified_accuracy, certified_radius, head_distances

EPS_REPORT = (36 / 255, 72 / 255, 108 / 255)
EPS_CURVE = tuple(float(e) for e in np.linspace(0.0, 0.5, 51))
RESIZE_600_TO_224_NORM = 0.37458


@torch.no_grad()
def layer_sigmas(mats: torch.Tensor) -> list[tuple[float, float]]:
    """Exact (sigma_max, sigma_min) of every layer: float64 SVD on the CPU."""
    out = []
    for w in mats.detach():
        s = torch.linalg.svdvals(w.cpu().double())
        out.append((float(s.max()), float(s.min())))
    return out


def lmlp_exact(mats: torch.Tensor) -> dict:
    sig = layer_sigmas(mats)
    return {"L_mlp": math.prod(s for s, _ in sig),
            "sigma_max_per_layer": [s for s, _ in sig],
            "sigma_min_per_layer": [s for _, s in sig],
            "worst_sigma_max": max(s for s, _ in sig),
            "worst_sigma_min": min(s for _, s in sig)}


@torch.no_grad()
def lmlp_power(mats: torch.Tensor, iters: int = 100) -> float:
    """Fast float32 estimate (power iteration on W^T W, per layer, on the
    matrices' device). Slightly below the true value; used only inside training."""
    mats = mats.detach().float()
    g = torch.Generator(device="cpu").manual_seed(0)
    v = torch.randn(mats.shape[0], mats.shape[2], 1, generator=g).to(mats.device)
    v = v / v.norm(dim=1, keepdim=True)
    for _ in range(iters):
        v = mats.transpose(1, 2) @ (mats @ v)
        v = v / v.norm(dim=1, keepdim=True).clamp_min(1e-30)
    return float((mats @ v).norm(dim=1).prod())


# `head_distances` ((K, K) matrix of ||w_i - w_j||, float64) and
# `certified_accuracy` (correct and radius > eps; plain accuracy at eps = 0) are
# eigenkit's, imported above.


@torch.no_grad()
def radii(logits: torch.Tensor, head_weight: torch.Tensor, L: float) -> dict:
    """pred, top-1 margin, and the certified radius of every row of `logits`
    (float64, on the CPU), as a dict: eigenkit's `certified_radius`. A class
    whose head row equals that of the prediction can never overtake it, and
    does not limit the radius."""
    return certified_radius(logits, head_weight, L)._asdict()


def summary(logits: torch.Tensor, y: torch.Tensor, head_weight: torch.Tensor,
            L: float) -> dict:
    """Radii statistics and certified accuracy for one Lipschitz constant."""
    res = radii(logits, head_weight, L)
    correct = res["pred"] == y.cpu()
    rad = res["radius"]
    rad_correct = torch.where(correct, rad, torch.zeros_like(rad))
    return {
        "L": L,
        "median_radius": float(rad.median()),
        "mean_radius": float(rad.mean()),
        # radius counted as 0 for misclassified images: the usual "certified radius"
        "median_radius_correct_or_zero": float(rad_correct.median()),
        "certified_accuracy": {f"{e:.6f}": certified_accuracy(correct, rad, e)
                               for e in EPS_REPORT},
        "certified_accuracy_named": {name: certified_accuracy(correct, rad, e) for name, e in
                                     zip(("36/255", "72/255", "108/255"), EPS_REPORT)},
        "curve": {"epsilon": list(EPS_CURVE),
                  "certified_accuracy": [certified_accuracy(correct, rad, e)
                                         for e in EPS_CURVE]},
        "median_radius_600": float(rad.median()) / RESIZE_600_TO_224_NORM,
    }


@torch.no_grad()
def check_mlp_bound(mlp, phi: torch.Tensor, L_mlp: float,
                    scales=(1e-3, 1e-2, 1e-1, 1.0)) -> dict:
    """||MLP(phi + d) - MLP(phi)|| <= L_mlp ||d|| on random d: must never fail.

    Runs on a float64 copy of the MLP: for a diverged run (L_mlp ~ 1e20) the
    float32 output norms overflow to inf and the check fails spuriously.
    """
    g = torch.Generator(device="cpu").manual_seed(123)
    mlp = copy.deepcopy(mlp).double()
    phi = phi.double()
    clean = mlp(phi)
    worst = 0.0
    for s in scales:
        d = (torch.randn(phi.shape, generator=g) * s).to(phi.device)
        ratio = ((mlp(phi + d) - clean).norm(dim=1) / d.norm(dim=1).clamp_min(1e-12)).max()
        worst = max(worst, float(ratio))
    return {"worst_ratio": worst, "L_mlp": L_mlp, "respected": worst <= L_mlp * (1 + 1e-4)}
