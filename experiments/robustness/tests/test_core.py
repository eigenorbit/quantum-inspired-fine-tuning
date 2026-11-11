"""Tests for the three things that have to be true.

  1. the butterfly circuit is orthogonal -- at init, and after training
  2. the analytic bound is correct -- it is never violated, by noise or by PGD
  3. spectral normalisation contracts while orthogonality does not -- the
     mechanism the whole study rests on

Run with:  .venv/bin/python -m pytest tests/ -q
"""

import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import (ARMS, ButterflyOrthogonal, Dense1x1, DepthNet, GroupSort,  # noqa: E402
                  build_butterfly_matrix, butterfly_pairs, channels_for,
                  freeze_all, num_butterfly_angles, sigma_max)
from experiments import (AdaptedMixer, SimpleCNN, _accuracy_detail,  # noqa: E402
                         _batch_margins, adapt, blackbox_l2, build_and_train,
                         class_weights_for, dihedral, measure_model,
                         min_adversarial_distance, pgd_l2, pgd_l2_targeted,
                         random_l2)

WIDTHS = [8, 16, 32, 64, 128]
TOL = 1e-5


# --- 1. the butterfly circuit is orthogonal --------------------------------

@pytest.mark.parametrize("c", WIDTHS)
def test_angle_count_matches_the_paper(c):
    """(C/2) * log2(C) -- Table 1 of the Quantum Vision Transformers paper."""
    expected = (c // 2) * int(round(math.log2(c)))
    assert num_butterfly_angles(c) == expected
    assert ButterflyOrthogonal(c, bias=False).theta.numel() == expected


@pytest.mark.parametrize("c", WIDTHS)
@pytest.mark.parametrize("draw", range(3))
def test_orthogonal_at_init(c, draw):
    torch.manual_seed(draw)
    layer = ButterflyOrthogonal(c)
    with torch.no_grad():
        layer.theta.normal_(0.0, 2.0)
    assert layer.orthogonality_error() < TOL
    smax, smin = layer.singular_values()
    assert abs(smax - 1.0) < TOL and abs(smin - 1.0) < TOL


def test_orthogonal_after_training():
    """100 Adam steps must not degrade orthogonality -- it is structural."""
    torch.manual_seed(0)
    layer = ButterflyOrthogonal(64)
    x, target = torch.randn(8, 64, 4, 4), torch.randn(8, 64, 4, 4)
    opt = torch.optim.Adam(layer.parameters(), lr=1e-2)
    for _ in range(100):
        opt.zero_grad()
        (layer(x) - target).pow(2).mean().backward()
        opt.step()
    assert layer.theta.grad.abs().max() > 0          # the angles really moved
    assert layer.orthogonality_error() < TOL


@pytest.mark.parametrize("c", WIDTHS)
def test_preserves_norm(c):
    torch.manual_seed(3)
    layer = ButterflyOrthogonal(c)
    with torch.no_grad():
        layer.theta.normal_(0.0, 2.0)
        w, v = layer.matrix(), torch.randn(c, 64)
    assert torch.allclose((w @ v).norm(dim=0), v.norm(dim=0), atol=TOL)


@pytest.mark.parametrize("c", WIDTHS)
def test_levels_are_disjoint_pairs(c):
    for level in range(int(round(math.log2(c)))):
        i_idx, j_idx = butterfly_pairs(c, level)
        assert i_idx.numel() == c // 2
        assert torch.all(j_idx == i_idx ^ (1 << level))
        assert sorted(torch.cat([i_idx, j_idx]).tolist()) == list(range(c))


@pytest.mark.parametrize("c", [3, 6, 12, 20, 63, 100])
def test_non_power_of_two_raises(c):
    with pytest.raises(ValueError, match="power of two"):
        ButterflyOrthogonal(c)


def test_wrong_angle_count_raises():
    levels = [butterfly_pairs(8, l) for l in range(3)]
    with pytest.raises(ValueError, match="angles"):
        build_butterfly_matrix(torch.zeros(5), 8, levels)


def test_freeze_is_transparent():
    """Freezing must not change the function, and input gradients must survive."""
    layer = ButterflyOrthogonal(16).eval()
    x = torch.randn(2, 16, 3, 3, requires_grad=True)
    reference = layer(x)
    layer.freeze_matrix()
    frozen = layer(x)
    assert torch.allclose(reference, frozen, atol=1e-6)
    (g,) = torch.autograd.grad(frozen.sum(), x)
    assert g.abs().max() > 0


# --- 2. the bound is correct ------------------------------------------------

@pytest.mark.parametrize("arm", ARMS)
def test_bound_is_never_violated(arm):
    """||f(x+d) - f(x)|| <= L ||d|| over 200 random pairs at four scales."""
    torch.manual_seed(0)
    model = DepthNet(2, arm, 4).eval()
    bound = model.analytic_lipschitz()
    x, clean = torch.rand(200, 1, 28, 28), None
    with torch.no_grad():
        clean = model(x)
        for scale in (0.01, 0.1, 1.0, 3.0):
            d = torch.randn_like(x) * scale
            ratio = ((model(x + d) - clean).norm(dim=1)
                     / d.flatten(1).norm(dim=1)).max().item()
            assert ratio <= bound * 1.0001, f"{arm} scale {scale}: {ratio} > {bound}"


@pytest.mark.parametrize("arm", ["dense-sn", "butterfly"])
def test_constrained_arms_have_bound_one(arm):
    torch.manual_seed(0)
    assert abs(DepthNet(2, arm, 6).eval().analytic_lipschitz() - 1.0) < 1e-3


def test_unconstrained_bound_grows_with_depth():
    torch.manual_seed(0)
    with torch.no_grad():
        for m in (shallow := DepthNet(2, "dense", 2).eval()).modules():
            if isinstance(m, Dense1x1):
                m.weight.mul_(1.5)
        for m in (deep := DepthNet(2, "dense", 8).eval()).modules():
            if isinstance(m, Dense1x1):
                m.weight.mul_(1.5)
    assert deep.analytic_lipschitz() > shallow.analytic_lipschitz() > 1.0


@pytest.mark.slow
def test_pgd_cannot_break_a_certified_point():
    """The certificate against a real adversary, not against random noise.

    For an image whose certified radius exceeds epsilon, no perturbation of that
    size can flip it -- so PGD must fail on every such point.
    """
    model, loaders, dev = build_and_train("mnist01", "butterfly", 4, seed=0, epochs=2)
    stats = measure_model(model, loaders, dev)
    margins, correct = [], []
    for x, y in loaders["eval"]:
        z = model(x.to(dev))
        top2 = z.topk(2, 1).values
        margins.append((top2[:, 0] - top2[:, 1]).cpu())
        correct.append(z.argmax(1).cpu() == y)
    radii = torch.cat(margins) / (math.sqrt(2) * stats["L_analytic"])
    correct = torch.cat(correct)
    freeze_all(model)

    x, y = next(iter(loaders["eval"]))
    x, y = x[:256].to(dev), y[:256].to(dev)
    eps = 0.5
    inside = (correct[:256] & (radii[:256] >= eps)).to(dev)
    assert int(inside.sum()) > 20, "too few certified points for the test to mean anything"
    survived = pgd_l2(model, x, y, eps, steps=50, restarts=2)
    assert int((inside & ~survived).sum()) == 0


# --- 3. the mechanism: contraction vs isometry ------------------------------

def test_spectral_normalisation_pins_only_the_largest_singular_value():
    """The crux. sigma_max = 1, but the rest are free to shrink -- and do."""
    torch.manual_seed(0)
    layer = Dense1x1(64, normalise=True)
    with torch.no_grad():
        layer.weight.normal_(0, 3.0)
    smax, smin = layer.singular_values()
    assert abs(smax - 1.0) < 1e-5
    assert smin < 0.5, "a spectrally normalised matrix is not an isometry"


def test_butterfly_pins_every_singular_value():
    torch.manual_seed(0)
    layer = ButterflyOrthogonal(64)
    with torch.no_grad():
        layer.theta.normal_(0, 3.0)
    smax, smin = layer.singular_values()
    assert abs(smax - 1.0) < TOL and abs(smin - 1.0) < TOL


def test_orthogonal_stack_keeps_its_norm():
    torch.manual_seed(0)
    model = DepthNet(2, "butterfly", 8).eval()
    assert abs(model.norm_retention(torch.rand(16, 1, 28, 28)) - 1.0) < 1e-3


def test_spectrally_normalised_stack_contracts():
    torch.manual_seed(0)
    model = DepthNet(2, "dense-sn", 8).eval()
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, Dense1x1):
                m.weight.normal_(0, 1.0)          # a generic, non-orthogonal matrix
    assert model.norm_retention(torch.rand(16, 1, 28, 28)) < 0.9


def test_sigma_max_matches_svd():
    """The Gram/eigvalsh route used in training agrees with a plain SVD."""
    torch.manual_seed(0)
    for _ in range(5):
        w = torch.randn(64, 64)
        assert abs(float(sigma_max(w)) - float(torch.linalg.svdvals(w).max())) < 1e-4


# --- the activation ---------------------------------------------------------

def test_groupsort_is_1_lipschitz_and_norm_preserving():
    torch.manual_seed(0)
    gs = GroupSort()
    for _ in range(20):
        a, b = torch.randn(8, 16, 5, 5), torch.randn(8, 16, 5, 5)
        num = (gs(a) - gs(b)).flatten(1).norm(dim=1)
        den = (a - b).flatten(1).norm(dim=1)
        assert float((num / den).max()) <= 1.0 + 1e-6
        assert torch.allclose(gs(a).flatten(1).norm(dim=1),
                              a.flatten(1).norm(dim=1), atol=1e-5)


def test_groupsort_is_not_distance_preserving():
    """Why the network is 1-Lipschitz but not bi-Lipschitz."""
    gs = GroupSort()
    a = torch.tensor([[[[1.0]], [[0.0]]]])
    b = torch.tensor([[[[0.0]], [[1.0]]]])
    assert torch.equal(gs(a), gs(b)) and (a - b).norm() > 1.0


# --- the two perturbations --------------------------------------------------

def test_random_l2_uses_the_exact_budget():
    torch.manual_seed(0)
    model = freeze_all(DepthNet(2, "butterfly", 2).eval())
    x, y = torch.rand(16, 1, 28, 28), torch.randint(0, 2, (16,))
    assert random_l2(model, x, y, 0.0).shape == (16,)
    assert bool(random_l2(model, x, y, 0.0).eq(model(x).argmax(1) == y).all())


def test_pgd_is_at_least_as_strong_as_random():
    """The worst direction cannot do worse than an arbitrary one."""
    torch.manual_seed(0)
    model, loaders, dev = build_and_train("mnist01", "butterfly", 2, seed=0, epochs=1)
    freeze_all(model)
    x, y = next(iter(loaders["eval"]))
    x, y = x[:128].to(dev), y[:128].to(dev)
    for eps in (0.5, 2.0):
        assert int(pgd_l2(model, x, y, eps, steps=20, restarts=1).sum()) <= \
               int(random_l2(model, x, y, eps, trials=4).sum())






@pytest.mark.parametrize("channels", [64, 256, 1024])
def test_orthonormal_lift_is_an_isometry(channels):
    """Width is decoupled from the downsampling cascade without costing the bound."""
    torch.manual_seed(0)
    model = DepthNet(2, "butterfly", 2, channels=channels).eval()
    assert model.channels == channels
    assert abs(model.analytic_lipschitz() - 1.0) < 1e-4
    assert abs(model.norm_retention(torch.rand(8, 1, 28, 28)) - 1.0) < 1e-3


def test_invalid_width_raises():
    with pytest.raises(ValueError, match="channels must be"):
        DepthNet(2, "butterfly", 2, channels=100)


# --- adaptation: does a certificate survive finetuning? ---------------------

def _tiny_backbone():
    torch.manual_seed(0)
    return DepthNet(2, "butterfly", 4, channels=64).eval()


@pytest.mark.parametrize("mode", ["head-only", "butterfly", "lora", "full"])
def test_adapter_starts_at_the_pretrained_function(mode):
    """Every adapter must be the identity at initialisation.

    Otherwise the comparison starts from four different models rather than four
    ways of moving away from one.
    """
    torch.manual_seed(0)
    base = torch.linalg.qr(torch.randn(64, 64))[0]
    m = AdaptedMixer(base, torch.zeros(64), mode).eval()
    assert torch.allclose(m.effective_matrix(), base, atol=1e-6)


def test_orthogonal_adapter_preserves_the_certificate():
    """The claim the whole pipeline rests on: orthogonal o orthogonal = orthogonal."""
    torch.manual_seed(0)
    base = torch.linalg.qr(torch.randn(64, 64))[0]
    m = AdaptedMixer(base, torch.zeros(64), "butterfly")
    with torch.no_grad():
        m.adapter.theta.normal_(0.0, 2.0)          # move the adapter far from identity
    smax, smin = m.singular_values()
    assert abs(smax - 1.0) < 1e-5 and abs(smin - 1.0) < 1e-5
    assert not torch.allclose(m.effective_matrix(), base, atol=1e-2), "it did not move"


def test_lora_adapter_breaks_the_certificate():
    """The contrast: an additive low-rank update moves the singular values."""
    torch.manual_seed(0)
    base = torch.linalg.qr(torch.randn(64, 64))[0]
    m = AdaptedMixer(base, torch.zeros(64), "lora", rank=4)
    with torch.no_grad():
        m.lora_B.normal_(0.0, 0.5)
    assert m.singular_values()[0] > 1.1


def test_adapt_freezes_the_backbone_except_in_full_mode():
    backbone = _tiny_backbone()
    for mode, expect_frozen in (("butterfly", True), ("lora", True), ("full", False)):
        model = adapt(backbone, mode, num_classes=2)
        mixers = [m for m in model.modules() if isinstance(m, AdaptedMixer)]
        assert len(mixers) == backbone.depth
        has_trainable_base = any(
            p.requires_grad for m in mixers for n, p in m.named_parameters()
            if n.startswith(("weight", "bias")))
        assert has_trainable_base is not expect_frozen


def test_adapted_model_lipschitz_accounting():
    """analytic_lipschitz must read the *effective* matrix, adapter included."""
    backbone = _tiny_backbone()
    model = adapt(backbone, "butterfly", num_classes=2).eval()
    assert abs(model.analytic_lipschitz() - 1.0) < 1e-4
    lora = adapt(backbone, "lora", num_classes=2).eval()
    with torch.no_grad():
        for m in lora.modules():
            if isinstance(m, AdaptedMixer):
                m.lora_B.normal_(0.0, 0.5)
    assert lora.analytic_lipschitz() > 2.0


# ---------------------------------------------------------------------------
# multichannel, imbalanced data (section 7: DermaMNIST)
# ---------------------------------------------------------------------------

def test_channels_for_picks_a_power_of_two_wide_enough_for_the_input():
    """RGB forces a 4-channel lift, hence C = 256 rather than 64."""
    assert channels_for(1) == 64
    assert channels_for(3) == 256          # lift 4, the next power of two >= 3
    assert channels_for(4) == 256
    assert channels_for(3, width_multiple=2) == 512
    for in_ch in (1, 2, 3, 4, 8):
        c = channels_for(in_ch)
        lift = c // 64
        assert lift & (lift - 1) == 0      # the butterfly needs a power of two
        assert lift >= in_ch


def test_rgb_input_lift_is_an_isometry_and_keeps_the_bound_at_one():
    """The 3 -> 4 lift has orthonormal columns, so it contributes exactly 1."""
    model = DepthNet(num_classes=7, arm="butterfly", depth=4,
                     channels=channels_for(3), spatial=32, in_channels=3).eval()
    assert abs(model.analytic_lipschitz() - 1.0) < 1e-5

    w = model.lift_weight.flatten(1)                       # (lift, in_channels)
    assert torch.allclose(w.T @ w, torch.eye(3), atol=1e-6)

    x = torch.randn(4, 3, 32, 32)
    lifted = torch.nn.functional.conv2d(x, model.lift_weight)
    assert torch.allclose(lifted.flatten(1).norm(dim=1),
                          x.flatten(1).norm(dim=1), atol=1e-5)


def test_class_weights_are_inverse_frequency_and_average_to_one():
    y = torch.tensor([0] * 90 + [1] * 9 + [2] * 1)
    loader = [(torch.zeros(len(y), 1), y)]
    w = class_weights_for(loader, 3, torch.device("cpu"))
    assert torch.allclose(w * torch.tensor([90.0, 9.0, 1.0]) * 3 / 100,
                          torch.ones(3), atol=1e-5)
    assert w[2] > w[1] > w[0]              # the rarest class is weighted most


def test_balanced_accuracy_and_collapse_detection_on_imbalanced_labels():
    """A constant predictor scores well on accuracy and at chance on balance.

    This is why section 7 reports balanced accuracy and defines collapse as
    "predicts one class", not as "scores below the majority rate": under a
    class-weighted loss a *good* model scores below the majority rate.
    """
    y = torch.tensor([0] * 90 + [1] * 5 + [2] * 5)
    loader = [(torch.zeros(len(y), 1), y)]

    class Constant(torch.nn.Module):
        def forward(self, x):
            return torch.tensor([[1.0, 0.0, 0.0]]).repeat(len(x), 1)

    acc, bal, n_pred = _accuracy_detail(Constant(), loader, torch.device("cpu"))
    assert acc == 0.9 and abs(bal - 1 / 3) < 1e-9 and n_pred == 1

    class Perfect(torch.nn.Module):
        def forward(self, x):
            return torch.nn.functional.one_hot(y, 3).float()

    acc, bal, n_pred = _accuracy_detail(Perfect(), loader, torch.device("cpu"))
    assert acc == 1.0 and bal == 1.0 and n_pred == 3


def test_dihedral_augmentation_only_permutes_pixels():
    """Flips and quarter turns are exact: same pixels, same norm, no blur."""
    x = torch.randn(8, 3, 28, 28)
    torch.manual_seed(0)
    seen = {torch.sort(dihedral(x).flatten())[0].sum().item() for _ in range(20)}
    assert len(seen) == 1                                   # no value ever changes
    for _ in range(10):
        a = dihedral(x)
        assert torch.allclose(a.flatten(1).norm(dim=1), x.flatten(1).norm(dim=1))
        assert torch.allclose(torch.sort(a.flatten())[0], torch.sort(x.flatten())[0])


def test_simple_cnn_is_a_reference_point_not_a_certified_arm():
    """It must run on the same inputs, and must NOT be 1-Lipschitz."""
    model = SimpleCNN(num_classes=7, in_channels=3).eval()
    x = torch.randn(4, 3, 28, 28)
    assert model(x).shape == (4, 7)
    assert not hasattr(model, "analytic_lipschitz")

    # No certificate machinery at all -- there is nothing to bound, and nothing
    # stops the map from expanding: scaling the head is enough to show it.
    assert not any(hasattr(model, a) for a in
                   ("analytic_lipschitz", "worst_singular_values", "norm_retention"))
    d = torch.randn_like(x)
    d = d / d.flatten(1).norm(dim=1).view(-1, 1, 1, 1)       # unit L2 perturbation
    with torch.no_grad():
        model.head[-1].weight.mul_(1e4)
    assert ((model(x + d) - model(x)).flatten(1).norm(dim=1) > 1.0).any()


# ---------------------------------------------------------------------------
# the minimum-distance search (§5.6)
# ---------------------------------------------------------------------------

def test_per_image_budget_matches_the_scalar_attack():
    """pgd_l2 must accept a per-image eps -- the bisection depends on it."""
    model, loaders, device = build_and_train("mnist01", "butterfly", 2, 0, epochs=1)
    freeze_all(model)
    x, y = next(iter(loaders["eval"]))
    x, y = x[:24].to(device), y[:24].to(device)
    g = lambda: torch.Generator(device=device).manual_seed(7)
    a = pgd_l2(model, x, y, 1.0, steps=10, restarts=1, generator=g())
    b = pgd_l2(model, x, y, torch.full((24,), 1.0, device=device),
               steps=10, restarts=1, generator=g())
    assert torch.equal(a, b)


def test_no_image_breaks_inside_its_certified_radius():
    """The invariant the whole study rests on: eps*(x) >= r(x), never below.

    This is the per-image version of the bound check. A violation means either
    the Lipschitz accounting or the radius formula is wrong.
    """
    model, loaders, device = build_and_train("mnist35", "butterfly", 2, 0, epochs=2)
    freeze_all(model)
    bound = model.analytic_lipschitz()
    x, y = next(iter(loaders["eval"]))
    x, y = x[:24].to(device), y[:24].to(device)

    eps_star, correct = min_adversarial_distance(
        model, x, y, eps_start=0.5, eps_max=8.0, grow=5, bisect=5,
        steps=15, restarts=1)
    radius = _batch_margins(model, x, y).clamp_min(0) / (math.sqrt(2) * bound)

    live = correct & torch.isfinite(eps_star)
    assert live.any(), "no image was broken at all; widen eps_max"
    assert (eps_star[live] >= radius[live] - 1e-5).all()
    # and the search is informative: the certificate is not absurdly loose here
    assert (eps_star[live] / radius[live]).median() < 10


def test_blackbox_attack_uses_no_gradients_and_is_weaker_than_pgd():
    model, loaders, device = build_and_train("mnist35", "butterfly", 2, 0, epochs=2)
    freeze_all(model)
    for p in model.parameters():
        p.requires_grad_(False)                 # would break any gradient attack
    x, y = next(iter(loaders["eval"]))
    x, y = x[:24].to(device), y[:24].to(device)
    survived = blackbox_l2(model, x, y, 1.0, queries=100)
    assert survived.dtype == torch.bool and survived.numel() == 24
    clean = model(x).argmax(1) == y
    assert (survived <= clean).all()             # it can only ever break images


def test_targeted_attack_only_ever_breaks_more():
    model, loaders, device = build_and_train("mnist10", "butterfly", 2, 0, epochs=1)
    freeze_all(model)
    x, y = next(iter(loaders["eval"]))
    x, y = x[:16].to(device), y[:16].to(device)
    clean = model(x).argmax(1) == y
    survived = pgd_l2_targeted(model, x, y, 1.0, 10, steps=10, restarts=1)
    assert (survived <= clean).all()
