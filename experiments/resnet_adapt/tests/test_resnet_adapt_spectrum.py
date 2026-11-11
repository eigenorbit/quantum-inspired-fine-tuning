"""The Lipschitz upper bound of ResNet152 (spectrum.py)."""

import math

import pytest
import torch
import torch.nn.functional as F

from experiments.bronet_mlp.bronet import conv_fft_bound
from experiments.resnet_adapt import backbone, spectrum


def _power(apply, adjoint, shape, iters=60, seed=0):
    """Lower bound on ||A|| by power iteration on A^T A."""
    x = torch.randn(shape, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)
    for _ in range(iters):
        x = adjoint(apply(x))
        x = x / x.norm()
    return float(apply(x).norm())


def test_conv_norm_equals_bronet_bound_for_3x3():
    w = torch.randn(16, 8, 3, 3, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
    assert abs(spectrum.conv_norm(w, 7) - conv_fft_bound(w, 7)) < 1e-9 * conv_fft_bound(w, 7)


@pytest.mark.parametrize("k,stride,n", [(3, 1, 9), (3, 2, 10), (7, 2, 16), (1, 2, 8)])
def test_conv_norm_is_an_upper_bound(k, stride, n):
    w = torch.randn(6, 4, k, k, generator=torch.Generator().manual_seed(k), dtype=torch.float64)
    pad = k // 2
    out = F.conv2d(torch.zeros(1, 4, n, n, dtype=torch.float64), w, stride=stride, padding=pad)

    def apply(x):
        return F.conv2d(x, w, stride=stride, padding=pad)

    def adjoint(y):
        return F.conv_transpose2d(y, w, stride=stride, padding=pad,
                                  output_padding=n - ((out.shape[-1] - 1) * stride - 2 * pad + k))

    lower = _power(apply, adjoint, (1, 4, n, n))
    bound = spectrum.conv_norm(w, n)
    assert lower <= bound * (1 + 1e-9)
    if stride == 1:
        assert lower >= 0.7 * bound           # the FFT bound is close for stride 1


def test_maxpool_factor_two_is_tight():
    pool = torch.nn.MaxPool2d(3, 2, 1)
    x = torch.zeros(1, 1, 9, 9)
    d = torch.zeros_like(x)
    d[0, 0, 3, 3] = 1.0                        # odd row and column: in 2 x 2 windows
    ratio = float((pool(x + d) - pool(x)).norm() / d.norm())
    assert ratio == pytest.approx(spectrum.MAXPOOL_FACTOR)
    g = torch.Generator().manual_seed(0)
    for _ in range(20):
        a, b = torch.randn(1, 2, 12, 12, generator=g), torch.randn(1, 2, 12, 12, generator=g)
        assert (pool(a) - pool(b)).norm() <= spectrum.MAXPOOL_FACTOR * (a - b).norm() + 1e-6


@torch.no_grad()
def test_stem_kernel_folds_normalisation_and_bn(net):
    """conv1(bn) on normalised input and the folded kernel on raw pixels differ by a
    constant only -- so their differences agree."""
    x1, x2 = (torch.rand(1, 3, 32, 32, generator=torch.Generator().manual_seed(s))
              for s in (1, 2))
    norm = backbone.Normalize()
    ref = net.bn1(net.conv1(norm(x1))) - net.bn1(net.conv1(norm(x2)))
    w = spectrum.stem_kernel(net).float()
    ours = F.conv2d(x1, w, stride=2, padding=3) - F.conv2d(x2, w, stride=2, padding=3)
    assert torch.allclose(ours, ref, rtol=1e-4, atol=1e-4 * ref.abs().max())


def test_layer42_term_kept_by_butterfly_changed_by_full(net):
    from test_resnet_adapt_butterfly import _train_steps

    base = spectrum.layer42_term(net.layer4[2])["term"]
    bf = _train_steps(backbone.Classifier(net.layer4[2], 3, seed=0, mode="butterfly"))
    assert abs(spectrum.layer42_term(bf.block)["term"] / base - 1) < 1e-5
    full = _train_steps(backbone.Classifier(net.layer4[2], 3, seed=0, mode="full"), lr=1e-2)
    assert abs(spectrum.layer42_term(full.block)["term"] / base - 1) > 1e-4


@pytest.mark.slow
@torch.no_grad()
def test_total_bound_dominates_the_network(net):
    front = spectrum.front_bound(net)
    tb = spectrum.total_bound(front, net.layer4[2])
    assert len(front["blocks"]) == 49 and math.isfinite(tb["log10_L"])
    f = backbone.FullModel(backbone.Front(net), backbone.Classifier(net.layer4[2], 3, 0)).eval()
    g = torch.Generator().manual_seed(0)
    x = torch.rand(2, 3, 224, 224, generator=g)
    for scale in (1e-3, 1e-1):
        d = torch.randn(x.shape, generator=g) * scale
        feat = f.classifier.features
        ratio = ((feat(f.front(x + d)) - feat(f.front(x))).norm(dim=1)
                 / d.flatten(1).norm(dim=1)).max()
        assert math.log10(float(ratio)) <= tb["log10_L"]
