"""The adapted MLP and the new LLN head.

    f(x) = head( MLP_adapted( phi(x) ) ),  MLP_adapted: h <- MinMax(W_i' h + b_i'), i = 1..8

    mode        W_i'                         trained                L_mlp
    head-only   W_i                          head                   = L_mlp^0
    butterfly   R(theta_i) W_i, theta_i = 0  theta, head            = L_mlp^0 by construction
    lora        W_i + B_i A_i, B_i = 0       A, B, head             drifts
    full        W_i (free, from W_i)         W, b, head             drifts

W_i, b_i are BRONet's `linear.eval_weight` / `linear.bias`. The butterfly is
eigenkit's `Orthogonal` on the butterfly layout, applied with `rotate_weight`
(R @ W), as in the notebook's `AdaptedMixer.effective_matrix`: for orthogonal R,
(RW)^T(RW) = W^T W, so RW has exactly the singular values of W -- for any W,
orthogonal or not.

Every mode computes exactly the pretrained MLP at initialisation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from eigenkit.orthogonal.torch import LLNHead, MaxMin, Orthogonal, rotate_weight

from init_draws import draw_butterfly_angles  # experiments/robustness/init_draws.py

MODES = ("head-only", "butterfly", "lora", "full")


_MINMAX = MaxMin(pairing="halves")


def minmax(x: torch.Tensor) -> torch.Tensor:
    """BRONet's MinMax (models/layers/activation.py), dim 1: split channels in two
    halves a, b; output [max(a, b), min(a, b)]. 1-Lipschitz and norm preserving.
    eigenkit's `MaxMin(pairing="halves")`; the robustness network's GroupSort is
    the other pairing, `"adjacent"`."""
    return _MINMAX(x)


# `LLNHead` (imported above) is eigenkit's: a linear layer with unit-norm rows
# (BRONet's LLN head), `self.linear` with its bias, `get_weight()` the normalised
# rows. The bias cancels in every logit difference, so the certificate does not
# see it.


class AdaptedMLP(nn.Module):
    """BRONet's 8-layer dense MLP, adapted in one of `MODES`."""

    def __init__(self, base_weight: torch.Tensor, base_bias: torch.Tensor, mode: str,
                 rank: int = 4):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}, expected one of {MODES}")
        depth, width, width2 = base_weight.shape
        if width != width2 or base_bias.shape != (depth, width):
            raise ValueError(f"bad base shapes {tuple(base_weight.shape)}, {tuple(base_bias.shape)}")
        self.mode, self.depth, self.width, self.rank = mode, depth, width, rank
        base_weight = base_weight.detach().float().clone()
        base_bias = base_bias.detach().float().clone()

        if mode == "full":
            self.weight = nn.Parameter(base_weight)
            self.bias = nn.Parameter(base_bias)
        else:
            self.register_buffer("weight_frozen", base_weight)
            self.register_buffer("bias_frozen", base_bias)

        if mode == "butterfly":
            # R = I: start at the pretrained function. The angles are drawn from
            # the generator and then zeroed, so the adapter starts at the identity
            # and the following draws (head, LoRA factors, batch order) are fixed
            # by the seed.
            draw_butterfly_angles([width] * depth)
            self.rotations = nn.ModuleList(Orthogonal(width) for _ in range(depth))
        elif mode == "lora":
            self.lora_A = nn.Parameter(torch.randn(depth, rank, width) * 0.01)
            self.lora_B = nn.Parameter(torch.zeros(depth, width, rank))

    def effective_matrices(self) -> torch.Tensor:
        """(depth, width, width): the matrices this MLP applies, whatever the mode."""
        if self.mode == "full":
            return self.weight
        w = self.weight_frozen
        if self.mode == "butterfly":
            return torch.stack([rotate_weight(r.matrix(), w[i])
                                for i, r in enumerate(self.rotations)])
        if self.mode == "lora":
            return w + self.lora_B @ self.lora_A
        return w

    def biases(self) -> torch.Tensor:
        return self.bias if self.mode == "full" else self.bias_frozen

    def forward(self, h: torch.Tensor, matrices: torch.Tensor | None = None) -> torch.Tensor:
        mats = self.effective_matrices() if matrices is None else matrices
        for w, b in zip(mats, self.biases()):
            h = minmax(F.linear(h, w, b))
        return h


class AdaptedClassifier(nn.Module):
    """MLP_adapted + LLN head, on cached phi features."""

    def __init__(self, base_weight, base_bias, mode: str, num_classes: int, rank: int = 4):
        super().__init__()
        self.mlp = AdaptedMLP(base_weight, base_bias, mode, rank)
        self.head = LLNHead(self.mlp.width, num_classes)

    @property
    def mode(self) -> str:
        return self.mlp.mode

    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        return self.head(self.mlp(phi))

    def trainable_state(self) -> dict[str, torch.Tensor]:
        """Only what training changes; the frozen base is rebuilt from BRONet."""
        return {n: p.detach().cpu().clone() for n, p in self.named_parameters()
                if p.requires_grad}

    def load_trainable_state(self, state: dict[str, torch.Tensor]) -> None:
        own = dict(self.named_parameters())
        want = {n for n, p in own.items() if p.requires_grad}
        if set(state) != want:
            raise KeyError(f"trainable state mismatch: missing {sorted(want - set(state))}, "
                           f"unexpected {sorted(set(state) - want)}")
        with torch.no_grad():
            for n, t in state.items():
                own[n].copy_(t.to(own[n].device, own[n].dtype))

    def num_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build(base_weight, base_bias, mode: str, num_classes: int, rank: int = 4,
          seed: int = 0) -> AdaptedClassifier:
    """The model for one run. The head is initialised from `seed` (standard
    nn.Linear init) and is therefore identical across modes at a given seed;
    LoRA's A comes from a separate stream."""
    torch.manual_seed(seed + 7919)
    model = AdaptedClassifier(base_weight, base_bias, mode, num_classes, rank)
    torch.manual_seed(seed)
    model.head = LLNHead(model.mlp.width, num_classes)
    return model


def fingerprint(state: dict[str, torch.Tensor]) -> str:
    """Short sha256 over a trainable state (as repro.seeding.fingerprint)."""
    import hashlib

    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode())
        digest.update(state[name].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()[:16]
