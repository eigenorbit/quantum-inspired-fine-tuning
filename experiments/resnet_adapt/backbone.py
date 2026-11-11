"""ResNet152 (IMAGENET1K_V2) split at the input of ``layer4.2``.

    images in [0, 1] -> Front (frozen, cached) -> h (2048 x 7 x 7) -> Classifier -> logits

    Front       Normalize(ImageNet mean/std) -> conv1, bn1, relu, maxpool
                -> layer1, layer2, layer3 -> layer4[0], layer4[1]
    Classifier  layer4[2] (trainable) -> global average pool -> LLN head 2048 -> K

The ImageNet normalisation is part of the model, not of the dataset: the
pixels are the same [0, 1] tensors BRONet sees (`bronet_mlp.data`), so every
epsilon and every attack lives in the same space for both networks.

BatchNorm is ALWAYS in eval mode, in the front and in the trainable block: the
running statistics stay those of ImageNet, only the affine parameters (gamma,
beta) of the block's BN layers train with its convolutions. With eval-mode BN
the block is a fixed function of its weights, so the cached front and
end-to-end evaluation agree, and nothing depends on the composition of a batch.

The head is BRONet's LLN (`bronet_mlp.adapters.LLNHead`): one linear layer with
unit-norm rows and a bias, standard nn.Linear init from the run's seed.
"""

from __future__ import annotations

import copy
import hashlib

import torch
import torch.nn as nn
import torch.nn.functional as F

from eigenkit.orthogonal.torch import Orthogonal, fold_bn, rotate_weight

from experiments.bronet_mlp.adapters import LLNHead
from experiments.bronet_mlp.bronet import conv_fft_bound
from repro.config import NORM_MEAN, NORM_STD

from init_draws import draw_butterfly_angles  # experiments/robustness/init_draws.py (via bronet_mlp)

WEIGHTS = "IMAGENET1K_V2"
FEATURE_SHAPE = (2048, 7, 7)
FRONT_DEFINITION = ("normalize(ImageNet mean/std) -> conv1,bn1,relu,maxpool -> layer1..3 -> "
                    "layer4[0:2]; eval mode; float32 with TF32 disabled")
TRAINABLE_BLOCK = "layer4.2"
LORA_RANK = 4          # as bronet_mlp (RunConfig.lora_rank)


def load_resnet152(pretrained: bool = True) -> nn.Module:
    """torchvision ResNet152, IMAGENET1K_V2 weights (ImageNet top-1 82.284 %), eval mode."""
    from torchvision.models import ResNet152_Weights, resnet152

    net = resnet152(weights=ResNet152_Weights.IMAGENET1K_V2 if pretrained else None)
    return net.eval()


def state_sha256(module: nn.Module) -> str:
    """sha256 over every tensor of a state dict: identity of the weights actually used."""
    digest = hashlib.sha256()
    for name, t in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(t.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class Normalize(nn.Module):
    def __init__(self, mean=NORM_MEAN, std=NORM_STD):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


def _bn_eval(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()


class Front(nn.Module):
    """Everything before `layer4.2`, frozen and permanently in eval mode."""

    def __init__(self, net: nn.Module):
        super().__init__()
        self.normalize = Normalize()
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1, self.layer2, self.layer3 = net.layer1, net.layer2, net.layer3
        self.layer4_01 = nn.Sequential(net.layer4[0], net.layer4[1])
        for p in self.parameters():
            p.requires_grad_(False)
        super().train(False)

    def train(self, mode: bool = True):
        return super().train(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(self.normalize(x))
        return self.layer4_01(self.layer3(self.layer2(self.layer1(h))))


# `fold_bn` (imported above, used here and by spectrum.py) is eigenkit's: (W_eff,
# b_eff), float64 on the CPU, the eval-mode conv + BN as one convolution,
# W_eff = W * gamma / sqrt(var + eps), b_eff = beta - gamma * mean / sqrt(var + eps).
# Every convolution folded here has no bias of its own.


class AdaptedBottleneck(nn.Module):
    """A torchvision Bottleneck without downsample (e.g. layer4.2), BN folded into
    its three convolutions, adapted in one of three modes:

        butterfly   W_i' = R(theta_i) W_eff_i, R = eigenkit Orthogonal(out_i), theta = 0 at start
        lora        W_i' = W_eff_i + B_i A_i, rank LORA_RANK, B_i = 0 at start,
                    A_i (rank, in*k*k) ~ 0.01 N(0, 1), B_i (out, rank)
        head-only   W_i' = W_eff_i (frozen)

    W_eff (out, in, k, k) is taken as the matrix (out, in*k*k); R mixes output
    channels only, so (RW)^T (RW) = W^T W: every singular value of the matrix and
    the operator norm of the convolution (R acts pixel-wise) are kept exactly.
    LoRA changes the spectrum freely. The folded biases are frozen, as the biases
    of BRONet's MLP in bronet_mlp. Every mode computes exactly the pretrained block
    at initialisation.
    """

    MODES = ("butterfly", "lora", "head-only")

    def __init__(self, block: nn.Module, mode: str):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}")
        if getattr(block, "downsample", None) is not None:
            raise NotImplementedError("blocks with a downsample branch are not supported")
        self.mode = mode
        self.geometry = []
        for i in (1, 2, 3):
            conv, bn = getattr(block, f"conv{i}"), getattr(block, f"bn{i}")
            if conv.groups != 1 or conv.dilation != (1, 1) or conv.bias is not None:
                raise NotImplementedError(f"conv{i}: only plain convolutions are supported")
            w, b = fold_bn(conv, bn)
            self.register_buffer(f"weight{i}", w.float())
            self.register_buffer(f"bias{i}", b.float())
            self.geometry.append((conv.stride, conv.padding))
        if mode == "butterfly":
            # R = I: start at the pretrained block. The angles are drawn from the
            # generator and then zeroed, so the adapter starts at the identity and
            # the following draws (head, LoRA factors, batch order) are fixed by
            # the seed.
            widths = [getattr(self, f"weight{i}").shape[0] for i in (1, 2, 3)]
            draw_butterfly_angles(widths)
            self.rotations = nn.ModuleList(Orthogonal(w) for w in widths)
        elif mode == "lora":
            for i in (1, 2, 3):
                w = getattr(self, f"weight{i}")
                fan_in = w[0].numel()
                setattr(self, f"lora_A{i}", nn.Parameter(torch.randn(LORA_RANK, fan_in) * 0.01))
                setattr(self, f"lora_B{i}", nn.Parameter(torch.zeros(w.shape[0], LORA_RANK)))

    def weights(self) -> list[torch.Tensor]:
        """The three effective kernels (out, in, k, k)."""
        ws = [getattr(self, f"weight{i}") for i in (1, 2, 3)]
        if self.mode == "head-only":
            return ws
        if self.mode == "lora":
            return [w + (getattr(self, f"lora_B{i}") @ getattr(self, f"lora_A{i}"))
                    .reshape(w.shape) for i, w in zip((1, 2, 3), ws)]
        return [rotate_weight(r.matrix(), w) for r, w in zip(self.rotations, ws)]

    def effective_convs(self) -> list[tuple[str, torch.Tensor]]:
        return [(f"conv{i}", w.detach().double().cpu())
                for i, w in zip((1, 2, 3), self.weights())]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1, w2, w3 = self.weights()
        (s1, p1), (s2, p2), (s3, p3) = self.geometry
        h = F.relu(F.conv2d(x, w1, self.bias1, s1, p1))
        h = F.relu(F.conv2d(h, w2, self.bias2, s2, p2))
        h = F.conv2d(h, w3, self.bias3, s3, p3)
        return F.relu(h + x)


class Classifier(nn.Module):
    """The adapted `layer4.2` + global average pooling + a new LLN head.

    full  a copy of the pretrained block, all of it trainable (BN in eval)
    butterfly   AdaptedBottleneck: BN folded, R(theta) W per conv; theta trainable
    lora        AdaptedBottleneck: BN folded, W + B A per conv (rank 4); A, B trainable
    head-only   AdaptedBottleneck: BN folded, frozen; only the head trains

    The head is always drawn from `seed` (identical across modes at one seed);
    LoRA's A comes from the separate stream `seed + 7919`, as in bronet_mlp.
    """

    def __init__(self, block: nn.Module, num_classes: int, seed: int = 0,
                 mode: str = "full"):
        super().__init__()
        self.mode = mode
        if mode == "full":
            self.block = copy.deepcopy(block)
            for p in self.block.parameters():
                p.requires_grad_(True)
        else:
            torch.manual_seed(seed + 7919)
            self.block = AdaptedBottleneck(block, mode)
        torch.manual_seed(seed)
        self.head = LLNHead(FEATURE_SHAPE[0], num_classes)
        self.train(False)

    def train(self, mode: bool = True):
        super().train(mode)
        _bn_eval(self)
        return self

    def features(self, h: torch.Tensor) -> torch.Tensor:
        """2048-d pooled features (what torchvision feeds to `fc`)."""
        return self.block(h).mean(dim=(2, 3))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(h))

    def trainable_state(self) -> dict[str, torch.Tensor]:
        """Only what training changes; BN running statistics are ImageNet's."""
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


class FullModel(nn.Module):
    """Images in [0, 1] (224 x 224) -> logits: Front + Classifier."""

    def __init__(self, front: Front, classifier: Classifier):
        super().__init__()
        self.front = front
        self.classifier = classifier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.front(x))


# ---------------------------------------------------------------------------
# spectrum of the trainable block
# ---------------------------------------------------------------------------

def folded_convs(block: nn.Module) -> list[tuple[str, torch.Tensor]]:
    """(name, W_eff) of the block's three convolutions, float64 on the CPU: the
    linear maps the block actually applies (plus a bias). For a torchvision
    Bottleneck the eval-mode BN is folded in (`fold_bn`); an AdaptedBottleneck
    gives its effective (rotated / frozen) kernels."""
    if hasattr(block, "effective_convs"):
        return block.effective_convs()
    return [(f"conv{i}", fold_bn(getattr(block, f"conv{i}"), getattr(block, f"bn{i}"))[0])
            for i in (1, 2, 3)]


def operator_norm(w: torch.Tensor, n: int = FEATURE_SHAPE[1]) -> float:
    """||conv2d(., w)|| on n x n inputs: sigma_max for 1x1 kernels (exact); for
    3x3 kernels (stride 1, padding 1) the rigorous FFT upper bound of
    `bronet_mlp.bronet.conv_fft_bound`."""
    if w.shape[-2:] == (1, 1):
        return float(torch.linalg.svdvals(w.reshape(w.shape[0], -1).double()).max())
    if w.shape[-2:] == (3, 3):
        return conv_fft_bound(w, n)
    raise NotImplementedError(f"kernel {tuple(w.shape[-2:])}")


@torch.no_grad()
def block_spectrum(block: nn.Module, base: nn.Module | None = None) -> dict:
    """Per BN-folded conv: sigma_max / sigma_min of the kernel as a matrix
    (out, in*k*k), float64 SVD, and the operator norm of the convolution on the
    7x7 input (`operator_norm`); with `base` (the pretrained block) the ratios.
    For 1x1 convolutions the two norms coincide."""
    now = folded_convs(block)
    ref = folded_convs(base) if base is not None else None
    layers = []
    for k, (name, w) in enumerate(now):
        m = w.reshape(w.shape[0], -1)
        s = torch.linalg.svdvals(m)
        row = {"name": name, "matrix_shape": list(m.shape), "sigma_max": float(s.max()),
               "sigma_min": float(s.min()), "operator_norm": operator_norm(w)}
        if ref is not None:
            w0 = ref[k][1]
            s0 = torch.linalg.svdvals(w0.reshape(w0.shape[0], -1))
            op0 = operator_norm(w0)
            row.update(sigma_max_base=float(s0.max()), sigma_min_base=float(s0.min()),
                       operator_norm_base=op0,
                       ratio_sigma_max=float(s.max() / s0.max()),
                       ratio_sigma_min=float(s.min() / s0.min()),
                       ratio_operator_norm=row["operator_norm"] / op0)
        layers.append(row)
    res = {"definition": "BN-folded conv: kernel as matrix (out, in*k*k), float64 SVD; "
                         "operator norm on 7x7 (1x1: sigma_max, 3x3: FFT bound)",
           "layers": layers}
    if ref is not None:
        for key in ("ratio_sigma_max", "ratio_sigma_min", "ratio_operator_norm"):
            vals = [r[key] for r in layers]
            res.update({f"{key}_min": min(vals), f"{key}_max": max(vals)})
    return res
