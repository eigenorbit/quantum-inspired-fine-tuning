"""The starting draws of the butterfly adapters.

A butterfly adapter starts at the identity: its angles are drawn from the global
generator, `torch.randn(n_angles)`, and then set to zero. The starting matrix
therefore does not depend on the seed, while every draw that follows it (a head,
LoRA factors, the order of the batches) does. eigenkit's `init="identity"` sets
the angles to zero without drawing, so for layers built that way
`draw_butterfly_angles`, called right before the layers are built, makes those
draws.
"""

import torch
from eigenkit.orthogonal import butterfly

__all__ = ["draw_butterfly_angles"]


def draw_butterfly_angles(widths) -> None:
    """Draw the starting angles of butterfly layers on `widths` from the global
    generator, in that order: one `torch.randn(n_angles)` each, whose values are
    discarded."""
    for width in widths:
        torch.randn(butterfly(width).n_angles)
