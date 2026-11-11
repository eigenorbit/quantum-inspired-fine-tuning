"""Modes butterfly, lora and head-only: BN folding, identity at start, orthogonality,
spectrum and operator norm kept exactly, training / resumption / round trip."""

import numpy as np
import pytest
import torch

from resnet_adapt_helpers import synthetic_inputs
from experiments.bronet_mlp import env
from experiments.resnet_adapt import backbone, train

ANGLES = 2 * (256 * 9) + 1024 * 11          # butterflies of 512, 512, 2048 channels
LORA = 4 * (2048 + 512) + 4 * (4608 + 512) + 4 * (512 + 2048)   # rank 4: A (4, in*k*k), B (out, 4)


def _h(n=4, seed=3):
    return torch.rand(n, 2048, 7, 7, generator=torch.Generator().manual_seed(seed))


@torch.no_grad()
@pytest.mark.parametrize("mode", ["head-only", "butterfly", "lora"])
def test_folded_block_equals_pretrained_at_start(net, mode):
    h = _h()
    ref = net.layer4[2](h)
    ours = backbone.AdaptedBottleneck(net.layer4[2], mode)(h)
    assert torch.allclose(ours, ref, rtol=1e-5, atol=1e-5 * ref.abs().max())


def test_trainable_parameters(net):
    b = backbone.Classifier(net.layer4[2], 5, seed=0, mode="butterfly")
    h = backbone.Classifier(net.layer4[2], 5, seed=0, mode="head-only")
    assert b.num_trainable() == ANGLES + 2048 * 5 + 5
    assert h.num_trainable() == 2048 * 5 + 5
    lo = backbone.Classifier(net.layer4[2], 5, seed=0, mode="lora")
    assert lo.num_trainable() == LORA + 2048 * 5 + 5 == 40_960 + 10_245
    assert torch.equal(lo.head.linear.weight, h.head.linear.weight)
    assert {n.split(".")[0] for n, p in b.named_parameters() if p.requires_grad} == {"block", "head"}
    assert all(n.endswith("theta") for n, p in b.block.named_parameters())
    # the same seed gives the same head in every mode
    last = backbone.Classifier(net.layer4[2], 5, seed=0, mode="full")
    assert torch.equal(b.head.linear.weight, last.head.linear.weight)


def _train_steps(clf, steps=5, lr=3e-2):
    clf.train()
    opt = torch.optim.Adam([p for p in clf.parameters() if p.requires_grad], lr=lr)
    h, y = _h(8, 5), torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])
    for _ in range(steps):
        opt.zero_grad()
        torch.nn.functional.cross_entropy(4 * clf(h), y).backward()
        opt.step()
    return clf.eval()


def test_butterfly_keeps_spectrum_and_operator_norms_exactly(net):
    clf = _train_steps(backbone.Classifier(net.layer4[2], 3, seed=0, mode="butterfly"))
    thetas = torch.cat([r.theta.detach() for r in clf.block.rotations])
    assert thetas.abs().max() > 1e-3                        # it did train
    for r in clf.block.rotations:
        R = r.matrix().detach().double()
        assert torch.allclose(R.T @ R, torch.eye(len(R), dtype=R.dtype), atol=1e-5)
    s = backbone.block_spectrum(clf.block, net.layer4[2])
    for key in ("ratio_sigma_max", "ratio_sigma_min", "ratio_operator_norm"):
        assert abs(s[f"{key}_min"] - 1) < 1e-5 and abs(s[f"{key}_max"] - 1) < 1e-5, key
    with torch.no_grad():                                    # ...but the function changed
        assert not torch.allclose(clf.block(_h()), net.layer4[2](_h()), atol=1e-4)


def test_head_only_block_frozen_and_last_block_spectrum_moves(net):
    ho = _train_steps(backbone.Classifier(net.layer4[2], 3, seed=0, mode="head-only"))
    with torch.no_grad():
        assert torch.allclose(ho.block(_h()), net.layer4[2](_h()), rtol=1e-5,
                              atol=1e-5 * net.layer4[2](_h()).abs().max())
    s = backbone.block_spectrum(ho.block, net.layer4[2])
    assert abs(s["ratio_operator_norm_max"] - 1) < 1e-7          # float32 buffers
    for mode, lr in (("full", 1e-2), ("lora", 3e-2)):
        m = _train_steps(backbone.Classifier(net.layer4[2], 3, seed=0, mode=mode), lr=lr)
        s = backbone.block_spectrum(m.block, net.layer4[2])
        drift = max(abs(s["ratio_operator_norm_min"] - 1), abs(s["ratio_operator_norm_max"] - 1))
        assert drift > 1e-4, mode


def test_fft_operator_norm_is_an_upper_bound(net):
    w = backbone.folded_convs(net.layer4[2])[1][1]          # the 3x3 conv
    bound = backbone.operator_norm(w)
    # power iteration on the real 7x7 convolution: a lower bound
    x = torch.randn(1, w.shape[1], 7, 7, generator=torch.Generator().manual_seed(0),
                    dtype=torch.float64)
    for _ in range(50):
        y = torch.nn.functional.conv2d(x, w, padding=1)
        x = torch.nn.functional.conv_transpose2d(y, w, padding=1)
        x = x / x.norm()
    lower = float(torch.nn.functional.conv2d(x, w, padding=1).norm())
    assert lower <= bound * (1 + 1e-9)
    assert lower >= 0.5 * bound                              # and not absurdly loose


@pytest.fixture(scope="module")
def inputs(net):
    inp = synthetic_inputs(net.layer4[2], n_train=64, n_val=32, n_test=32)
    inp.provenance["weights_sha256"] = backbone.state_sha256(net)
    return inp


@pytest.mark.parametrize("mode", ["butterfly", "lora"])
def test_fit_deterministic_resumable_and_round_trip(tmp_path, inputs, net, mode):
    cfg = train.RunConfig(step=2, mode=mode, epochs=2, batch_size=32, lr=1e-2,
                          early_stop="val", device="cpu")
    train.fit(cfg, inputs, tmp_path / "a")
    train.fit(cfg, inputs, tmp_path / "b", stop_after=1)
    train.fit(cfg, inputs, tmp_path / "b")
    ma, mb = (env.read_json(tmp_path / d / "metrics.json") for d in ("a", "b"))
    assert ma["fingerprint"] == mb["fingerprint"]
    assert ma["other_checkpoints"]["final"]["fingerprint"] == \
        mb["other_checkpoints"]["final"]["fingerprint"]
    if mode == "butterfly":
        assert ma["trainable_params"] == ANGLES + 2048 * 3 + 3
        assert ma["orthogonality_error_max"] < 1e-5
        assert abs(ma["spectrum"]["ratio_operator_norm_max"] - 1) < 1e-5
    else:
        assert ma["trainable_params"] == LORA + 2048 * 3 + 3
        assert "orthogonality_error_max" not in ma
    model = train.load_run(tmp_path / "a", device="cpu", which="best", net=net)
    z = train.predict(model, inputs.feats, inputs.test_idx, "cpu").numpy()
    assert np.allclose(z, np.load(tmp_path / "a" / "outputs_best.npz")["logits"], atol=1e-5)
