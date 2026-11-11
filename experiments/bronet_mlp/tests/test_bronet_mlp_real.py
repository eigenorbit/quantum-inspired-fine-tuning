"""Tests that need the dataset and/or the BRONet
checkpoint and skip themselves otherwise. 6 and 9 load the full model (minutes)."""

import math

import pytest
import torch

from experiments.bronet_mlp import env

from conftest import needs_bronet, needs_data


# --- 7. records identical to repro.data.build_loaders (variant unique)

@needs_data
@pytest.mark.data
@pytest.mark.parametrize("step", [2, 1])
def test_records_match_repro(step):
    from repro.config import TrainConfig
    from repro.data import build_loaders

    from experiments.bronet_mlp import data as bdata

    train, test, stats = bdata.records(step)
    cfg = TrainConfig(step=step, architecture="resnet152", unique_only=True, num_workers=0)
    _, _, ref_train, ref_test = build_loaders(cfg)
    assert [r.path for r in train] == [r.path for r in ref_train]
    assert [r.path for r in test] == [r.path for r in ref_test]
    exp = bdata.EXPECTED_SIZES[step]
    assert (len(train), len(test), stats["before"]) == (exp["train"], exp["test"],
                                                       exp["train_before"])
    if step == 2:
        assert [sum(r.label == k for r in test) for k in range(3)] == [78, 48, 114]


@needs_data
@pytest.mark.data
def test_transform_has_no_normalize():
    from experiments.bronet_mlp import data as bdata

    train, _, _ = bdata.records(2)
    x = bdata.load_images([train[0].path])
    assert x.shape == (1, 3, 224, 224) and 0.0 <= float(x.min()) and float(x.max()) <= 1.0


# --- 9. official L_pre x official L_mlp of the released model ------------------
#
# BRONet's own number is 8.1305 (test.py setting, 500 power iterations) -- but that
# is with TF32 convolutions, which this package disables. In full float32 the same
# 500 iterations give 8.13205 on an RTX 4070 (and 3000 iterations give
# 8.1323 in the reference audit, i.e. float32 is closer to the converged value).
# The package uses float32, so the expected value is the float32 one.
OFFICIAL_SUB_LIPSCHITZ_FP32 = 8.1321

@needs_bronet
@pytest.mark.slow
def test_official_lipschitz_matches_bronet():
    from experiments.bronet_mlp import bronet

    device = env.resolve_device("auto")
    model, info = bronet.load_bronet(device, env.resolve_bro_strategy(device),
                                     cache_bro_weights=True)
    assert info["missing"] == [] and info["unexpected"] == []
    assert "Lip: 8.14" in info["training_log_last"]
    res = bronet.compute_lpre(model, rigorous=False)       # fresh model: warm starts untouched
    assert not torch.backends.cudnn.allow_tf32          # the package must have disabled it
    assert math.isclose(res["official_sub_lipschitz"], OFFICIAL_SUB_LIPSCHITZ_FP32,
                        abs_tol=1e-3)
    if bronet.lpre_path().exists():
        saved = bronet.load_lpre()
        assert math.isclose(saved["L_pre_official"], res["L_pre_official"], rel_tol=1e-4)
        assert 8.39 < saved["L_pre_rigorous"] < 8.42


# --- 6. the feature cache agrees with an end-to-end forward -------------------

@needs_bronet
@needs_data
@pytest.mark.slow
def test_feature_cache_matches_end_to_end():
    from experiments.bronet_mlp import features

    step = 2
    if features.read_index(step) is None:
        pytest.skip("no feature cache for step 2 yet")
    device = env.resolve_device("auto")
    res = features.verify(step, device=device, strategy=env.resolve_bro_strategy(device),
                          n=4, cache_bro_weights=True)
    assert res["ok"], res


@needs_bronet
@needs_data
def test_feature_cache_rejects_wrong_metadata(tmp_path, monkeypatch):
    from experiments.bronet_mlp import features

    step = 2
    index = features.read_index(step)
    if index is None:
        pytest.skip("no feature cache for step 2 yet")
    meta = features.expected_meta()
    bad = {**meta, "transform_version": "something-else"}
    with pytest.raises(RuntimeError, match="different settings"):
        features.check_meta(index, bad)
    # a changed file on disk (size/mtime) is an error, not a silent recompute
    rel = next(iter(index["files"]))
    tampered = {**index, "files": {rel: {**index["files"][rel], "size": -1}}}
    with pytest.raises(RuntimeError, match="changed on disk"):
        features.check_files(tampered, [env.data_root() / rel])


# --- stage 2: PGD end to end through the real BRONet (smoke test) ----------------

@needs_bronet
@needs_data
@pytest.mark.slow
def test_pgd_end_to_end_smoke(tmp_path):
    from experiments.bronet_mlp import attacks

    step = 2
    runs = [n for n in ("head-only", "butterfly")
            if (env.runs_dir(step) / n / "seed0" / "metrics.json").exists()]
    if not runs:
        pytest.skip("no finished step-2 run")
    device = env.resolve_device("auto")
    strategy = env.resolve_bro_strategy(device)
    grad = "fp16" if strategy == "stream" else "fp32"
    cfg = attacks.AttackConfig(step=step, runs=runs[:1], n_images=2, eps=[108 / 255],
                               steps=2, restarts=1, random_trials=1, grad_precision=grad)
    res = attacks.run(cfg, device=device, strategy=strategy, batch_size=2, eval_chunk=4,
                      cache_bro_weights=True, out=tmp_path / "attack")
    assert res["sanity"]["ok"]
    agree = res["sanity"]["clean_logits_max_abs_diff_vs_training"]
    assert agree and max(agree.values()) < 1e-3          # end-to-end == cached-phi logits
    assert len(res["rows"]) == 2



@needs_bronet
@needs_data
@pytest.mark.slow
def test_fp16_gradient_path_matches_fp32():
    """The float16 BRO copy used for attack gradients computes nearly the same phi
    and nearly the same input gradient as the exact float32 network."""
    from experiments.bronet_mlp import bronet
    from experiments.bronet_mlp import data as bdata

    device = env.resolve_device("auto")
    if device.type != "cuda":
        pytest.skip("fp16 path needs CUDA")
    net, _ = bronet.load_bronet(device, env.resolve_bro_strategy(device), cache_bro_weights=True)
    for p in net.parameters():
        p.requires_grad_(False)
    _, test, _ = bdata.records(2)
    x = bdata.load_images([r.path for r in test[:2]]).to(device)
    half = bronet.half_bro_weights(net, device)
    xs = [x.clone().requires_grad_(True) for _ in range(2)]
    f32 = bronet.phi_with_grad(net, xs[0])
    f16 = bronet.phi_with_grad_half(net, xs[1], half)
    rel = float((f16 - f32).norm() / f32.norm())
    assert rel < 1e-2, rel
    g = [torch.autograd.grad(f.square().sum(), xi)[0] for f, xi in ((f32, xs[0]), (f16, xs[1]))]
    cos = float(torch.nn.functional.cosine_similarity(g[0].flatten(), g[1].flatten(), dim=0))
    assert cos > 0.99, cos
