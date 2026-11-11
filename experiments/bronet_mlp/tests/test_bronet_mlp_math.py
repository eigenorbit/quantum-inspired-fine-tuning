"""Identity at init, orthogonality, L preservation,
the certificate formula, and the radius checked empirically."""

import math

import pytest
import torch

from experiments.bronet_mlp import adapters, certify, losses
from experiments.bronet_mlp.adapters import MODES, AdaptedMLP, minmax
from core import ButterflyOrthogonal

from conftest import needs_bronet_src, random_base


def reference_mlp(w, b, h):
    for wi, bi in zip(w, b):
        h = minmax(torch.nn.functional.linear(h, wi, bi))
    return h


# --- 1. every mode is the pretrained MLP at initialisation -------------------

@pytest.mark.parametrize("mode", MODES)
def test_identity_at_init(mode):
    w, b = random_base(depth=3, width=64)
    mlp = AdaptedMLP(w, b, mode)
    h = torch.randn(32, 64)
    assert torch.allclose(mlp(h), reference_mlp(w, b, h), atol=1e-5)


@needs_bronet_src
def test_minmax_matches_bronet():
    from experiments.bronet_mlp import bronet

    models = bronet.import_models()
    from models.layers import build_activation

    x = torch.randn(16, 2048)
    assert torch.equal(build_activation("MinMax", dim=1)(x), minmax(x))


@needs_bronet_src
@pytest.mark.parametrize("mode", MODES)
def test_identity_vs_bronet_liresmlp(mode):
    """Against BRONet's own LiResMLP forward (random weights, real class)."""
    from experiments.bronet_mlp import bronet

    bronet.import_models()
    from models.dense import LiResMLP
    from models.layers import build_activation

    torch.manual_seed(0)
    ref = LiResMLP(num_features=256, depth=3, activation=build_activation("MinMax", dim=1))
    with torch.no_grad():
        ref.bias.normal_(0, 0.1)
    ref.eval()
    mlp = AdaptedMLP(ref.eval_weight, ref.bias, mode)
    h = torch.randn(8, 256)
    assert torch.allclose(mlp(h), ref(h), atol=1e-5)


# --- 2. the butterfly is orthogonal and keeps singular values ----------------

def test_butterfly_orthogonal_2048():
    r = ButterflyOrthogonal(2048, bias=False)       # random theta ~ N(0, 0.1^2)
    with torch.no_grad():
        r.theta.normal_(0, 1.0)
    R = r.matrix().double()
    assert float((R.T @ R - torch.eye(2048, dtype=R.dtype)).abs().max()) < 1e-5


def test_butterfly_keeps_singular_values():
    torch.manual_seed(0)
    r = ButterflyOrthogonal(256, bias=False)
    with torch.no_grad():
        r.theta.normal_(0, 1.0)
    W = torch.randn(256, 256)                        # arbitrary, not orthogonal
    s0 = torch.linalg.svdvals(W.double())
    s1 = torch.linalg.svdvals((r.matrix() @ W).double())
    assert torch.allclose(s0, s1, rtol=1e-5, atol=1e-6)


# --- 3. L_mlp after optimisation: kept by head-only/butterfly, not by lora/full

@pytest.mark.parametrize("mode", MODES)
def test_lipschitz_after_training(mode):
    w, b = random_base(depth=2, width=64)
    model = adapters.build(w, b, mode, num_classes=3, seed=0)
    L0 = certify.lmlp_exact(w)["L_mlp"]
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=3e-2)
    g = torch.Generator().manual_seed(0)
    for _ in range(40):
        x = torch.randn(64, 64, generator=g)
        y = torch.randint(0, 3, (64,), generator=g)
        loss = losses.cert_ce(model(x), y, model.head.get_weight(), 8.0, 36 / 255)
        opt.zero_grad()
        loss.backward()
        opt.step()
    L = certify.lmlp_exact(model.mlp.effective_matrices().detach())["L_mlp"]
    if mode in ("head-only", "butterfly"):
        assert abs(L / L0 - 1) < 1e-5
    else:
        assert abs(L / L0 - 1) > 1e-3


def test_power_iteration_close_to_exact():
    w, _ = random_base(depth=3, width=128)
    w = w * torch.tensor([1.3, 0.7, 2.0]).view(3, 1, 1)
    exact = certify.lmlp_exact(w)["L_mlp"]
    est = certify.lmlp_power(w, 300)
    # a lower estimate (clustered top singular values converge slowly)
    assert est <= exact * (1 + 1e-5) and est / exact > 0.99


# --- 4. radius formula agrees with BRONet's get_predictions -------------------

@needs_bronet_src
@pytest.mark.parametrize("K", [3, 5])
def test_radius_matches_bronet_get_predictions(K):
    from experiments.bronet_mlp import bronet

    models = bronet.import_models()
    torch.manual_seed(K)
    head = adapters.LLNHead(64, K)

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.head = head

        def forward(self, x):
            return self.head(x)

    x = torch.randn(500, 64)
    L = 8.13
    eps_list = [0.02, 36 / 255, 72 / 255, 0.5]
    with torch.no_grad():
        clean, perturbed = models.get_predictions(M(), x, eps_list, L)
        r = certify.radii(clean, head.get_weight(), L)["radius"]
    pred = clean.argmax(1)
    for eps, y_ in zip(eps_list, perturbed):
        bronet_cert = y_.argmax(1) == pred          # the clean logit of pred beats every
        ours = r > eps                              # worst-case competitor
        assert torch.equal(bronet_cert, ours), eps


def test_radius_hand_example():
    w = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    z = torch.tensor([[3.0, 1.0]])
    r = certify.radii(z, w, L=2.0)["radius"]
    assert math.isclose(float(r), 2.0 / (2.0 * math.sqrt(2)), rel_tol=1e-12)


# --- 5. the radius is respected by random perturbations ----------------------

@pytest.mark.parametrize("mode", ["butterfly", "lora"])
def test_radius_empirically(mode):
    w, b = random_base(depth=2, width=64, seed=3)
    model = adapters.build(w, b, mode, num_classes=3, seed=1)
    with torch.no_grad():
        if mode == "lora":
            model.mlp.lora_B.normal_(0, 0.3)
        else:
            for r in model.mlp.rotations:
                r.theta.normal_(0, 1.0)
    model.eval()
    L = certify.lmlp_exact(model.mlp.effective_matrices().detach())["L_mlp"]
    g = torch.Generator().manual_seed(0)
    phi = torch.randn(200, 64, generator=g) * 2
    with torch.no_grad():
        z = model(phi)
        res = certify.radii(z, model.head.get_weight(), L)
        for _ in range(20):
            d = torch.randn(phi.shape, generator=g)
            d = d / d.norm(dim=1, keepdim=True) * (0.999 * res["radius"]).clamp(max=1e3).float()[:, None]
            assert torch.equal(model(phi + d).argmax(1), res["pred"])
