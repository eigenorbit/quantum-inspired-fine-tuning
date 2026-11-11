"""Orthogonal channel rotation for a width that is not a power of two (option 1b).

The butterfly of experiments/robustness/core.py needs a power-of-two width; the
convolutional part of BRONet has 588 channels. Split the width into powers of two
(its binary digits, 588 = 512 + 64 + 8 + 4), rotate every part with its own
butterfly, and mix the parts by a fixed permutation between two such stages:

    R = (P_S^T B_S P_S) ... (P_2^T B_2 P_2) B_1,   B_s = blockdiag(butterfly_512, butterfly_64, ...)

i.e. stage s >= 2 permutes the channels, rotates the parts and permutes back. Each
factor is orthogonal, so R is orthogonal for every value of the angles; R = I when
all angles are zero (training starts at the pretrained function). As
for the MLP butterfly, a rotation in front of an orthogonal (or any) layer keeps
its singular values: the Lipschitz constant of the network does not change.

A single power-of-two width is one butterfly; a part of size 1 (odd width) is left
as it is.
"""

from __future__ import annotations

import torch
from eigenkit.orthogonal.torch import PartitionedOrthogonal, binary_parts, mixing_permutation

from init_draws import draw_butterfly_angles  # experiments/robustness/init_draws.py

__all__ = ["PartitionedButterfly", "binary_parts", "mixing_permutation"]


class PartitionedButterfly(PartitionedOrthogonal):
    """C x C orthogonal matrix from `stages` block-diagonal butterflies over the
    binary parts of C, with a fixed channel permutation between consecutive stages.

    Trainable: the angles only (no bias). Acts on dim 1 of its input, so it is a
    1x1 convolution on (B, C, H, W) and a linear map on (B, C).

    The module is eigenkit's `PartitionedOrthogonal` on butterflies; this class
    keeps the paper's names and the paper's use of the random number generator.
    eigenkit pins the operations, so the numbers are fixed bit for bit: stages
    outer, parts inner; the permutation of stage s >= 1 is
    `mixing_permutation(width, perm_seed * 1000 + s - 1)`; the
    stage matrix is `block_diag` of the parts (a part of size 1 stays as it is);
    `R = B_s[inv][:, inv] @ R` with `inv = argsort(perm)`, i.e. P^T B P with
    P = I[perm]. The angles start at zero (R = I): every butterfly's angles are
    drawn from the global generator and then zeroed
    (`init_draws.draw_butterfly_angles`), so the draws that follow are fixed by
    the seed.
    """

    def __init__(self, width: int, stages: int = 2, perm_seed: int = 0):
        if stages < 1:
            raise ValueError("stages must be at least 1")
        rotating = [p for p in binary_parts(width) if p >= 2]
        draw_butterfly_angles(rotating * stages)        # stages outer, parts inner
        super().__init__(width, stages, perm_seed, "butterfly")

    @property
    def width(self) -> int:
        return self.n

    @property
    def num_angles(self) -> int:
        return self.n_angles

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return super().forward(z, dim=1)
