"""The model and the data.

Contents, in order:

    ButterflyOrthogonal   the quantum orthogonal layer
    GroupSort, Roll       the two norm-preserving fixed operations
    Dense1x1              the unconstrained / spectrally normalised mixer
    DepthNet              the network: stem, `depth` mixers, spectrally normalised head
    build_loaders         MNIST subsets -- the only place to edit for your own data

The architecture and the reasoning behind it are explained in `analysis.ipynb` §1;
this file is the implementation.

The one idea worth repeating here, because the code is meaningless without it:
everything in the network except the mixing layer is an exact isometry or is
1-Lipschitz, so **the mixer is the only thing that can contract**, and depth is
the only other variable. That is what makes the comparison in §5 attributable.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from eigenkit.orthogonal.torch import MaxMin, Orthogonal, OrthogonalConv1x1, sigma_max
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms

__all__ = ["ARMS", "ButterflyOrthogonal", "Dense1x1", "DepthNet", "GroupSort", "Roll",
           "build_butterfly_matrix", "butterfly_pairs", "num_butterfly_angles",
           "sigma_max", "freeze_all", "make_mixer", "DATASETS", "MNIST_SUBSETS",
           "MEDMNIST_SETS", "build_loaders", "channels_for"]


def channels_for(in_channels: int, width_multiple: int = 1) -> int:
    """Smallest valid channel count for an input with `in_channels` channels.

    The network needs `channels = lift * 4**3` with `lift >= in_channels`, and the
    butterfly circuit needs a power of two. For RGB that gives lift = 4 and
    C = 256; for greyscale, lift = 1 and C = 64. `width_multiple` doubles it.
    """
    lift = 1
    while lift < in_channels:
        lift *= 2
    return lift * 64 * width_multiple

ARMS = ("dense", "dense-sn", "butterfly")


# ---------------------------------------------------------------------------
# the quantum orthogonal layer
# ---------------------------------------------------------------------------

def _check_width(channels: int) -> None:
    if not isinstance(channels, int) or channels < 2 or channels & (channels - 1):
        raise ValueError(
            f"butterfly width must be a power of two and at least 2, got {channels!r}. "
            "Widths are never padded silently: pick a power-of-two channel count."
        )


def num_butterfly_angles(channels: int) -> int:
    """(C / 2) * log2(C) -- the gate count of the butterfly circuit.

    Table 1 of Cherrat et al., "Quantum Vision Transformers" (Quantum 2024).
    For C = 64 that is 192 angles, against 4096 for a dense 64x64 matrix.
    """
    _check_width(channels)
    return (channels // 2) * int(round(math.log2(channels)))


def butterfly_pairs(channels: int, level: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Index pairs for one level: (i, j) with j = i XOR 2**level and i < j.

    The Cooley-Tukey FFT pattern. Every index appears in exactly one pair, so
    the rotations within a level act on disjoint pairs and commute.
    """
    _check_width(channels)
    m = int(round(math.log2(channels)))
    if not 0 <= level < m:
        raise ValueError(f"level must be in [0, {m}), got {level}")
    bit = 1 << level
    idx = torch.arange(channels, dtype=torch.long)
    i_idx = idx[(idx & bit) == 0]
    return i_idx, i_idx | bit


def build_butterfly_matrix(theta: torch.Tensor, channels: int, levels) -> torch.Tensor:
    """Materialise W(theta) by left-multiplying the identity by each level.

    Uses out-of-place ``index_copy`` so autograd sees a clean graph. The only
    Python loop is over the log2(C) levels, never over the C/2 gates in a level.
    """
    expected = sum(int(i.numel()) for i, _ in levels)
    if theta.numel() != expected:
        raise ValueError(f"circuit needs {expected} angles, got {theta.numel()}")
    W = torch.eye(channels, device=theta.device, dtype=theta.dtype)
    at = 0
    for i_idx, j_idx in levels:
        k = i_idx.numel()
        c = torch.cos(theta[at:at + k]).unsqueeze(1)
        s = torch.sin(theta[at:at + k]).unsqueeze(1)
        at += k
        Wi, Wj = W.index_select(0, i_idx), W.index_select(0, j_idx)
        W = W.index_copy(0, i_idx, c * Wi + s * Wj) \
             .index_copy(0, j_idx, -s * Wi + c * Wj)
    return W


class ButterflyOrthogonal(OrthogonalConv1x1):
    """Orthogonal 1x1 convolution parameterised by the angles of a Givens circuit.

    A Givens rotation is orthogonal for *every* value of its angle, and a product
    of orthogonal matrices is orthogonal -- so there is no constraint to enforce,
    no projection step and no penalty term. The angles train with plain Adam and
    the matrix is orthogonal at every step by construction.

    On quantum hardware each rotation is one RBS gate; acting on unary states an
    RBS gate *is* a Givens rotation, so this is the classical simulation of a
    quantum orthogonal layer and the angle count is the gate count. See
    `analysis.ipynb` §1 for the references.

    The layer is eigenkit's `OrthogonalConv1x1` on the butterfly layout; this
    class only fixes the paper's settings and gives it the paper's attribute
    names. The settings, and the operations eigenkit pins for them:

    - the angles start as `torch.randn(num_angles) * 0.1` (small angles: W starts
      near the identity, which trains more stably than a uniformly random
      orthogonal matrix), drawn before the bias is created;
    - "materialise, then GEMM": W is built level by level with `index_copy` and
      applied as an ordinary 1x1 convolution, `F.conv2d(x, W[..., None, None],
      bias)`. It is rebuilt every training step -- caching it would starve the
      angles of gradients -- and cached only when autograd is off;
    - `freeze_matrix()` snapshots W, detached, for input-space attacks;
    - `singular_values()` and `orthogonality_error()` check W in float64.
    """

    def __init__(self, channels: int, bias: bool = True):
        _check_width(channels)
        super().__init__(channels, "butterfly", init="random", std=0.1, bias=bias)

    @property
    def channels(self) -> int:
        return self.n

    @property
    def num_angles(self) -> int:
        return self.n_angles

    @property
    def num_levels(self) -> int:
        return self.topology.depth

    def extra_repr(self) -> str:
        return (f"channels={self.channels}, angles={self.num_angles}, "
                f"levels={self.num_levels}")


# ---------------------------------------------------------------------------
# the fixed, norm-preserving operations
# ---------------------------------------------------------------------------

# `sigma_max` (imported above) is eigenkit's: the largest singular value of a
# matrix, via its Gram matrix, sqrt(max eigvalsh(W W^T)). Exact and
# differentiable. The Gram matrix is tiny for the readout (num_classes x
# num_classes), and for a square mixer this is the same cost as an SVD but stays
# on the accelerator.


class GroupSort(MaxMin):
    """MaxMin: sort channels within disjoint pairs (eigenkit's `MaxMin` with
    adjacent pairs: channels 2k and 2k+1 become their min, then their max).

    1-Lipschitz, and norm preserving because sorting only permutes values. Not
    distance preserving, though: it maps (1,0) and (0,1) to the same point, so
    the network is 1-Lipschitz but not bi-Lipschitz. (Anil, Lucas & Grosse,
    ICML 2019.)
    """

    def __init__(self):
        super().__init__(pairing="adjacent")


class Roll(nn.Module):
    """Circular shift of half the channels: a permutation, hence an isometry.

    Present because pixel_unshuffle builds a strict quadtree -- without it,
    adjacent 2x2 blocks would never exchange information.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.shape[1] // 2
        return torch.cat([x[:, :h], torch.roll(x[:, h:], (1, 1), (2, 3))], dim=1)


class Dense1x1(nn.Module):
    """Ordinary 1x1 convolution, optionally divided by its largest singular value.

    With ``normalise=True`` this is the `dense-sn` arm: sigma_max is pinned to 1,
    but nothing pins the rest, so it is 1-Lipschitz and *contractive*. That
    distinction is the subject of §2 and §5.2.

    Initialised orthogonal so all three arms start from the same place; only
    `butterfly` is constrained to *stay* there.
    """

    def __init__(self, channels: int, normalise: bool):
        super().__init__()
        self.channels = channels
        self.normalise = normalise
        self.weight = nn.Parameter(torch.empty(channels, channels))
        nn.init.orthogonal_(self.weight)
        self.bias = nn.Parameter(torch.zeros(channels))
        self._frozen_W = None

    def effective_weight(self) -> torch.Tensor:
        if self._frozen_W is not None:
            return self._frozen_W
        w = self.weight
        return w / sigma_max(w).clamp_min(1e-8) if self.normalise else w

    def freeze_matrix(self) -> None:
        with torch.no_grad():
            self._frozen_W = self.effective_weight().detach()

    def unfreeze_matrix(self) -> None:
        self._frozen_W = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.effective_weight().reshape(self.channels, self.channels, 1, 1)
        return F.conv2d(x, w, self.bias)

    def singular_values(self) -> tuple[float, float]:
        with torch.no_grad():
            s = torch.linalg.svdvals(self.effective_weight().cpu().double())
            return float(s.max()), float(s.min())


def make_mixer(arm: str, channels: int) -> nn.Module:
    if arm == "butterfly":
        return ButterflyOrthogonal(channels, bias=True)
    if arm in ("dense", "dense-sn"):
        return Dense1x1(channels, normalise=(arm == "dense-sn"))
    raise ValueError(f"unknown arm {arm!r}, expected one of {ARMS}")


# ---------------------------------------------------------------------------
# the network
# ---------------------------------------------------------------------------

class DepthNet(nn.Module):
    """The network. ``arm`` selects the mixer, ``depth`` the number of them.

        input 28x28 -> zero-pad to 32x32           (an injection: isometry)
                    -> pixel_unshuffle x3           (a permutation: isometry)
                       1 -> 4 -> 16 -> 64 channels, 32 -> 16 -> 8 -> 4 spatial
                    -> depth x [ mixer -> GroupSort -> Roll ]
                    -> flatten -> Linear / sigma_max    (spectral norm exactly 1)
    """

    def __init__(self, num_classes: int, arm: str, depth: int,
                 channels: int = 64, spatial: int = 32, in_channels: int = 1):
        super().__init__()
        self.arm, self.depth, self.channels, self.spatial = arm, depth, channels, spatial
        self.n_unshuffle = 3                       # each one multiplies channels by 4

        # Width is decoupled from the downsampling cascade by an orthonormal
        # lift: in_channels -> lift, with orthonormal columns, so it is an
        # isometry and contributes exactly 1 to the bound. The cascade then
        # gives channels = lift * 4**n_unshuffle, letting C be set independently
        # of the input. lift = 1 reproduces 1 -> 4 -> 16 -> 64.
        lift, rem = divmod(channels, 4 ** self.n_unshuffle)
        if rem or lift < in_channels:
            raise ValueError(
                f"channels must be in_channels * k * {4 ** self.n_unshuffle} "
                f"for integer k >= 1, got {channels}")
        self.lift = lift
        self.register_buffer("lift_weight",
                             torch.eye(lift, in_channels).reshape(lift, in_channels, 1, 1),
                             persistent=False)

        layers = []
        for i in range(depth):
            layers += [make_mixer(arm, channels), GroupSort()]
            if i < depth - 1:
                layers.append(Roll())
        self.mixers = nn.Sequential(*layers)

        side = spatial // 2 ** self.n_unshuffle
        self.head = nn.Linear(channels * side * side, num_classes)
        with torch.no_grad():
            self.head.weight.mul_(0.1)

    def stem(self, x: torch.Tensor) -> torch.Tensor:
        """Everything before the mixers. Every step here is an exact isometry."""
        pad = (self.spatial - x.shape[-1]) // 2
        h = F.pad(x, (pad,) * 4)
        if self.lift > 1:
            h = F.conv2d(h, self.lift_weight)      # orthonormal columns: isometry
        for _ in range(self.n_unshuffle):
            h = F.pixel_unshuffle(h, 2)
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.mixers(self.stem(x))
        w = self.head.weight
        return F.linear(torch.flatten(h, 1), w / sigma_max(w).clamp_min(1e-8),
                        self.head.bias)

    # ---- Lipschitz accounting ------------------------------------------------

    def analytic_lipschitz(self) -> float:
        """Global L2 Lipschitz upper bound.

        Constants multiply along a composition. Padding, pixel_unshuffle, rolls
        and flatten are isometries (1); GroupSort is 1-Lipschitz (1); the head is
        spectrally normalised (1). So the bound is the product over the mixers.

        An orthogonal layer (a ButterflyOrthogonal mixer, or the eigenkit
        `Orthogonal` inside a butterfly ``AdaptedMixer``) contributes exactly 1
        by construction and is not measured here -- ``measure_model`` verifies
        that numerically after training. Anything else exposing
        ``singular_values`` is measured, which covers `dense`, `dense-sn`, and
        the ``AdaptedMixer`` of §6.
        """
        total = 1.0
        for m in self.modules():
            if isinstance(m, Orthogonal):
                continue                            # exactly 1, verified elsewhere
            if m is not self and hasattr(m, "singular_values"):
                total *= m.singular_values()[0]
        return total

    @torch.no_grad()
    def norm_retention(self, x: torch.Tensor) -> float:
        """||h_out|| / ||h_in|| across the mixer stack: the contraction, measured."""
        h0 = self.stem(x)
        h1 = self.mixers(h0)
        return float((h1.flatten(1).norm(dim=1) / h0.flatten(1).norm(dim=1)).median())

    def mixer_parameters(self) -> int:
        return sum(p.numel() for m in self.modules()
                   if isinstance(m, (Dense1x1, Orthogonal))
                   for p in m.parameters())

    def worst_singular_values(self) -> tuple[float, float]:
        """(largest sigma_max, smallest sigma_min) over all mixers."""
        pairs = [m.singular_values() for m in self.modules()
                 if isinstance(m, (Dense1x1, Orthogonal))]
        return max(p[0] for p in pairs), min(p[1] for p in pairs)


def freeze_all(model: nn.Module) -> nn.Module:
    """Freeze every mixer matrix. Required before an input-space attack."""
    for m in model.modules():
        if hasattr(m, "freeze_matrix"):
            m.freeze_matrix()
    return model


# ---------------------------------------------------------------------------
# the data -- the only section to edit for your own dataset
# ---------------------------------------------------------------------------

# MNIST subsets: name -> the digits it keeps. mnist35 (3 vs 5) is genuinely
# harder than 0 vs 1, so accuracy and the robustness curves move.
MNIST_SUBSETS = {
    "mnist01": (0, 1),
    "mnist35": (3, 5),
    "mnist08": (0, 8),
    "mnist10": tuple(range(10)),
}

# MedMNIST v2 datasets (Yang et al., Scientific Data 2023), all 28x28 RGB.
#
#   dermamnist   HAM10000 skin lesions, 7 classes, 7,007 train / 2,005 test.
#                Severely imbalanced: melanocytic nevi are 67% of the data and
#                the rarest class has 80 training images. A constant predictor
#                therefore scores 66.9%, which is why `collapsed` matters here
#                far more than it did on MNIST.
#   bloodmnist   peripheral blood cells, 8 classes, 11,959 train / 3,421 test,
#                roughly balanced. Used as the pretraining source in §6: larger
#                and balanced, adapted to the smaller imbalanced clinical target.
MEDMNIST_SETS = ("dermamnist", "bloodmnist", "pathmnist", "octmnist",
                 "organamnist", "pneumoniamnist", "retinamnist", "breastmnist")

DATASETS = {**{k: v for k, v in MNIST_SUBSETS.items()},
            **{k: None for k in MEDMNIST_SETS}}


def _subset(ds, digits) -> TensorDataset:
    """Keep the requested digits, remap labels to 0..k-1, scale to [0, 1]."""
    targets = ds.targets if isinstance(ds.targets, torch.Tensor) else torch.tensor(ds.targets)
    mask = torch.zeros_like(targets, dtype=torch.bool)
    for d in digits:
        mask |= targets == d
    idx = mask.nonzero(as_tuple=True)[0]
    x = ds.data[idx].to(torch.float32).div_(255.0).unsqueeze(1)
    remap = {int(d): i for i, d in enumerate(digits)}
    y = torch.tensor([remap[int(t)] for t in targets[idx]], dtype=torch.long)
    return TensorDataset(x, y)


def _medmnist_tensors(name: str, split: str, root: str):
    """One MedMNIST split as (x, y) with x float in [0, 1], shaped (N, C, H, W)."""
    import medmnist
    from medmnist import INFO
    ds = getattr(medmnist, INFO[name]["python_class"])(
        split=split, download=True, root=root)
    x = torch.from_numpy(ds.imgs)
    if x.dim() == 3:                       # single-channel sets arrive as (N, H, W)
        x = x.unsqueeze(-1)
    x = x.permute(0, 3, 1, 2).to(torch.float32).div_(255.0)
    return x, torch.from_numpy(ds.labels).flatten().long()


def build_loaders(name: str = "mnist35", batch_size: int = 128, seed: int = 0,
                  root: str = "./data", eval_subset: int | None = 2000) -> dict:
    """Train / test / eval loaders.

    Pixels are scaled to [0, 1] and nothing else -- no augmentation, no mean/std
    normalisation -- so a perturbation norm is directly interpretable in input
    space and the certified radius means what it says.

    ``eval`` is a fixed subset of the test set used for the expensive
    measurements (margins, attacks); ``test`` is the full test set and is what
    clean accuracy is reported on. The subset is drawn with a fixed generator, so
    every arm, depth and seed sees exactly the same images.

    **To use your own data**, return this same dict, with each loader yielding
    (x, y) where x is a float tensor in [0, 1] shaped (B, C, H, W). Keep the
    [0, 1] scaling: otherwise epsilon is no longer a norm in input space.
    """
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}, expected one of {sorted(DATASETS)}")

    if name in MEDMNIST_SETS:
        xtr, ytr = _medmnist_tensors(name, "train", root)
        xte, yte = _medmnist_tensors(name, "test", root)
        train_ds, test_ds = TensorDataset(xtr, ytr), TensorDataset(xte, yte)
        num_classes, in_channels = int(ytr.max()) + 1, xtr.shape[1]
    else:
        digits = MNIST_SUBSETS[name]
        tf = transforms.ToTensor()
        train_ds = _subset(datasets.MNIST(root, train=True, download=True, transform=tf), digits)
        test_ds = _subset(datasets.MNIST(root, train=False, download=True, transform=tf), digits)
        num_classes, in_channels = len(digits), 1

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(seed))
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)

    if eval_subset is not None and eval_subset < len(test_ds):
        keep = torch.randperm(len(test_ds),
                              generator=torch.Generator().manual_seed(999))[:eval_subset]
        eval_loader = DataLoader(Subset(test_ds, keep.tolist()), batch_size=512,
                                 shuffle=False)
    else:
        eval_loader = test_loader

    return {"train": train_loader, "test": test_loader, "eval": eval_loader,
            "num_classes": num_classes, "in_channels": in_channels,
            "sizes": {"train": len(train_ds), "test": len(test_ds),
                      "eval": len(eval_loader.dataset)}}
