"""PGD through the frozen ResNet front (bronet_mlp.attacks.run with an injected network)."""

import numpy as np
import pytest
import torch

from experiments.bronet_mlp import attacks as battacks
from experiments.resnet_adapt import attacks, backbone, paths


def test_out_dir_names(monkeypatch, tmp_path):
    monkeypatch.setenv("DERMA_REPRO_OUT", str(tmp_path))
    assert attacks.out_dir(2, 32, "best").name == "step_2_bs32_best"
    assert attacks.out_dir(1, paths.DEFAULT_BATCH_SIZE, "final").name == "step_1_final"
    assert attacks.out_dir(2, 32, "best", (2, 4, 9)).name == "step_2_bs32_best_eps2-4-9"
    assert attacks.out_dir(2, 32, "best", attack="pgd").name == "step_2_bs32_best"
    assert attacks.out_dir(1, 32, "best", attack="apgd-t").name == "step_1_bs32_best_apgd-t"
    assert attacks.out_dir(2, 32, "best", (2, 4), "square").name == "step_2_bs32_best_eps2-4_square"
    from experiments.resnet_adapt import min_eps
    assert min_eps.out_dir(2, 32, "best").name == "step_2_bs32_best_mineps"
    assert min_eps.out_dir(1, 32, "best", "apgd-ce").name == "step_1_bs32_best_mineps_apgd-ce"


@pytest.mark.slow
def test_attack_through_resnet_front(net, tmp_path):
    front = backbone.Front(net).eval()
    clf = backbone.Classifier(net.layer4[2], 3, seed=0, mode="butterfly").eval()
    for p in clf.parameters():
        p.requires_grad_(False)
    x = torch.rand(2, 3, 224, 224, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        y = clf(front(x)).argmax(1)                         # clean-correct by construction
    eps = 2.0
    cfg = battacks.AttackConfig(step=2, n_images=2, eps=[eps], steps=3, restarts=1,
                                random_trials=1, runs=["butterfly"])
    targets = [{"name": "butterfly", "seed": 0, "which": "best", "clf": clf,
                "L_official": 1e109, "L_rigorous": 1e109}]
    res = battacks.run(cfg, device=torch.device("cpu"), strategy="resnet", batch_size=2,
                       eval_chunk=2, super_chunk=8, feature_fn=lambda v, g: front(v),
                       targets=targets, images={"x": x, "y": y, "names": ["a.jpg", "b.jpg"]},
                       out=tmp_path)
    assert res["sanity"]["ok"]
    rows = res["rows"]
    assert len(rows) == 2 and all(r["clean_correct"] for r in rows)
    assert all(np.isfinite(r["adv_margin"]) for r in rows)
    assert all(r["delta_norm"] <= eps * (1 + 1e-4) for r in rows)
    assert all(r["radius_official"] < 1e-100 for r in rows)  # vacuous certificate


@pytest.mark.slow
@pytest.mark.parametrize("attack", ["apgd-ce", "apgd-t", "square"])
def test_strong_attack_through_resnet_front(net, tmp_path, attack):
    front = backbone.Front(net).eval()
    clf = backbone.Classifier(net.layer4[2], 3, seed=0, mode="butterfly").eval()
    for p in clf.parameters():
        p.requires_grad_(False)
    x = torch.rand(2, 3, 224, 224, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        y = clf(front(x)).argmax(1)
    eps = 2.0
    cfg = battacks.AttackConfig(step=2, n_images=2, eps=[eps], steps=3, restarts=1,
                                random_trials=1, runs=["butterfly"], attack=attack, queries=4)
    targets = [{"name": "butterfly", "seed": 0, "which": "best", "clf": clf,
                "L_official": 1e109, "L_rigorous": 1e109}]
    res = battacks.run(cfg, device=torch.device("cpu"), strategy="resnet", batch_size=2,
                       eval_chunk=2, super_chunk=8, feature_fn=lambda v, g: front(v),
                       targets=targets, images={"x": x, "y": y, "names": ["a.jpg", "b.jpg"]},
                       out=tmp_path)
    assert res["sanity"]["ok"]
    rows = res["rows"]
    assert len(rows) == 2 and all(np.isfinite(r["adv_margin"]) for r in rows)
    assert all(r["delta_norm"] <= eps * (1 + 1e-4) for r in rows)
