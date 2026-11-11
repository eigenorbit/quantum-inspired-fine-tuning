"""Certificate-aware losses, identical for every adaptation mode.

``cert_ce`` (default), in the style of BRONet's `margin_layer.margin_loss`:

    z_j <- z_j + L * eps * ||w_y - w_j||   for j != y   (0 for j = y)
    loss = CE(tau * z, y)

i.e. cross-entropy on the logits of the worst case the certificate allows at
radius eps. L is the current L_total (detached). The notebook's
`margin_loss(offset=1)` assumed L = 1; with L_0 ~ 8.1 it would train for a
radius of ~0.09 < 36/255, so it is not used.

``logit_annealing`` is BRONet's own training loss (`logit_annealing_loss` with
EMMA-style per-class eps), ported without the DDP wrapper; hyper-parameters
default to the ImageNet config (offset 2, temperature 0.75, gamma 5).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from eigenkit.orthogonal.torch import cert_ce

LOSSES = ("cert_ce", "logit_annealing")


def _pair_dist(head_weight: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """(B, K): ||w_y - w_j|| for each row's label y."""
    return (head_weight[y].unsqueeze(1) - head_weight.unsqueeze(0)).norm(dim=-1)


# `cert_ce(logits, y, head_weight, L, eps, tau=4.0)` is eigenkit's, imported above:
# CE(tau * (z + L * eps * ||w_y - w_j||), y).


def logit_annealing(logits, y, head_weight, L: float, eps: float, offset: float = 2.0,
                    temperature: float = 0.75, gamma: float = 5.0):
    num_classes = logits.shape[1]
    z = logits - logits.gather(1, y[:, None])
    margin = L * _pair_dist(head_weight, y)
    eps_ji = (-z / (margin * eps).clip(1e-8)).clip(0.01, 1)
    z_ = z + eps_ji.detach() * eps * margin
    out = (z_ - offset * F.one_hot(y, num_classes)) / temperature
    ce = F.cross_entropy(out, y, reduction="none")
    pt = torch.exp(-ce)
    return ((1 - pt) ** gamma * ce * temperature).mean()


def compute(name: str, logits, y, head_weight, L: float, eps: float, tau: float):
    if name == "cert_ce":
        return cert_ce(logits, y, head_weight, L, eps, tau)
    if name == "logit_annealing":
        return logit_annealing(logits, y, head_weight, L, eps)
    raise ValueError(f"unknown loss {name!r}, expected one of {LOSSES}")
