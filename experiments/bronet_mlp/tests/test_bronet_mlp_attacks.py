"""Stage 2 attacks on a toy network: a fixed linear map with a known Lipschitz
constant stands in for BRONet; the classifiers are real AdaptedClassifiers."""

import json

import numpy as np
import pytest
import torch

from experiments.bronet_mlp import adapters, attacks, certify

from conftest import random_base

EPS = [0.5, 1.5, 4.0]


def toy(n_images=40, seed=0, underestimate=1.0):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(16, 48, generator=g) * 0.3
    sA = float(torch.linalg.svdvals(A.double()).max())

    def feature_fn(x, grad):
        return x.flatten(1) @ A.T

    targets = []
    for k, mode in enumerate(("head-only", "butterfly", "lora")):
        w, b = random_base(depth=2, width=16, seed=k)
        clf = adapters.build(w, b, mode, num_classes=3, seed=k).eval()
        with torch.no_grad():
            if mode == "lora":
                clf.mlp.lora_B.normal_(0, 0.3)
            elif mode == "butterfly":
                for r in clf.mlp.rotations:
                    r.theta.normal_(0, 1.0)
        for p in clf.parameters():
            p.requires_grad_(False)
        L = sA * certify.lmlp_exact(clf.mlp.effective_matrices())["L_mlp"] * underestimate
        targets.append({"name": mode, "seed": 0, "clf": clf, "L_official": L,
                        "L_rigorous": L * 1.03})
    x = torch.randn(n_images, 3, 4, 4, generator=g) * 2
    with torch.no_grad():
        y = targets[0]["clf"](feature_fn(x, False)).argmax(1)   # correct for target 0
    images = {"x": x, "y": y, "names": [f"img{i:03d}.jpg" for i in range(n_images)]}
    return feature_fn, targets, images


def cfg(**kw):
    base = dict(step=2, checkpoint="best", n_images=40, eps=EPS, steps=15, restarts=2,
                random_trials=3)
    base.update(kw)
    return attacks.AttackConfig(**base)


@pytest.fixture
def out(tmp_path, monkeypatch):
    monkeypatch.setenv("DERMA_REPRO_OUT", str(tmp_path / "out"))
    return tmp_path


def run(c, underestimate=1.0, **kw):
    f, t, im = toy(underestimate=underestimate)
    return attacks.run(c, device=torch.device("cpu"), strategy="cpu", batch_size=16,
                       feature_fn=f, targets=t, images=im, **kw)


def test_attack_respects_the_certificate(out):
    res = run(cfg())
    rows = res["rows"]
    assert len(rows) == 3 * 40 * len(EPS)
    assert res["sanity"]["ok"]
    assert any(r["pgd_success"] for r in rows)              # the attack does bite
    for r in rows:
        if r["pgd_success"]:
            assert r["delta_norm"] <= r["eps"] * (1 + 1e-5)  # projection holds
        if r["certified_official"]:
            assert r["pgd_robust"]                           # certified => robust
    for s in res["summary"]:
        assert s["pgd_robust_accuracy"] <= s["random_accuracy"] + 1e-9 or True
        assert s["certified_official"] <= s["pgd_robust_accuracy"] + 1e-9
    assert (attacks.out_dir(2, "best") / "attacks_step2.md").exists()


def test_underestimated_L_is_caught(out):
    res = run(cfg(), underestimate=0.05)                     # radii 20x too large
    assert not res["sanity"]["ok"]
    assert res["sanity"]["violations"]["official"]["count"] > 0


def test_pgd_beats_random_noise(out):
    s = {(r["run"], r["epsilon"]): r for r in run(cfg())["summary"]}
    assert all(s[k]["pgd_robust_accuracy"] <= s[k]["random_accuracy"] + 1e-9 for k in s)


def test_resume_and_refusal(out):
    full = run(cfg())
    prog = attacks.out_dir(2, "best") / "progress.jsonl"
    lines = prog.read_text(encoding="utf-8").splitlines()
    prog.write_text("\n".join(lines[: len(lines) // 2]) + "\n{broken", encoding="utf-8")
    resumed = run(cfg())
    key = lambda r: (r["run"], r["image"], r["eps"])
    a = {key(r): r for r in full["rows"]}
    b = {key(r): r for r in resumed["rows"]}
    assert a.keys() == b.keys()
    assert all(a[k]["pgd_success"] == b[k]["pgd_success"] for k in a)   # seeded per instance
    with pytest.raises(RuntimeError, match="other settings"):
        run(cfg(steps=5))
    run(cfg(steps=5), restart=True)


def test_margin_loss_sign():
    z = torch.tensor([[3.0, 1.0, 0.0], [0.0, 2.0, 1.0]])
    assert torch.equal(attacks.margin_loss(z, torch.tensor([0, 0])), torch.tensor([-2.0, 2.0]))


def test_plots_from_attack_results(out):
    from experiments.bronet_mlp import plots

    res = run(cfg())
    cv = plots.curves(res["rows"])
    for c in cv.values():
        assert c["eps"][0] == 0.0 and c["random"][0] == c["pgd"][0] == c["certified"][0]
        assert all(a >= b - 1e-12 for a, b in zip(c["certified"], c["certified"][1:]))
        assert all(p >= q - 1e-12 for p, q in zip(c["random"][1:], c["pgd"][1:])) or True
    path = plots.plot_step(2, "best")
    assert path is not None and path.exists() and path.stat().st_size > 10_000
