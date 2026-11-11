"""Search of the empirical minimal adversarial radius eps* (min_eps.py): exact on a
linear model, censoring, violations, and the full pipeline on the toy network."""

import math

import pytest
import torch

from experiments.bronet_mlp import min_eps

from test_bronet_mlp_attacks import toy


def linear(n=30, classes=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(classes, 48, generator=g)
    x = torch.randn(n, 3, 4, 4, generator=g)
    z = x.flatten(1) @ W.T
    y = z.argmax(1)
    gap = z.gather(1, y[:, None]) - z                                  # z_y - z_j >= 0
    dist = (W[y][:, None, :] - W[None, :, :]).norm(dim=2)
    d_true = (gap / dist).masked_fill(torch.nn.functional.one_hot(y, classes).bool(),
                                      math.inf).amin(1)                # exact distance
    clfs = [lambda f: f @ W.T]
    return clfs, x, y, d_true


def run_chunk(cfg, clfs, x, y, r):
    flat = lambda v: v.flatten(1)
    return min_eps.search_chunk(cfg, flat, flat, clfs, torch.zeros(len(y), dtype=torch.long),
                                x, y, r, list(range(len(y))), torch.device("cpu"), 16)


def test_eps_star_is_exact_on_a_linear_model():
    clfs, x, y, d = linear()
    cfg = min_eps.MinEpsConfig(step=1, attack="apgd-t", steps=40, eps_floor=0.01, eps_max=50.0)
    st = run_chunk(cfg, clfs, x, y, 0.5 * d)                         # valid, loose radius
    star = torch.tensor([s["hi"] for s in st])
    assert not any(s["censored"] for s in st)
    assert (star >= d * (1 - 1e-4)).all()                           # never below the truth
    assert (star <= d * 1.01).all()                                  # bisection: 0.4 %
    assert all(s["check_flip"] is False for s in st)                 # nothing below 0.99 r
    assert all(s["lo"] <= s["hi"] for s in st)
    # a tight certificate (r = true distance): eps*/r -> 1
    st = run_chunk(cfg, clfs, x, y, d)
    assert all(abs(s["hi"] / float(r) - 1) < 0.01 for s, r in zip(st, d))


def test_censoring_and_floor():
    clfs, x, y, d = linear(seed=1)
    cfg = min_eps.MinEpsConfig(step=1, attack="apgd-t", steps=30, eps_floor=0.01,
                               eps_max=float(d.median()))
    st = run_chunk(cfg, clfs, x, y, torch.zeros(len(y)))             # vacuous radius (ResNet)
    cens = torch.tensor([s["censored"] for s in st])
    assert cens.any() and (~cens).any()
    assert (d[cens] > cfg.eps_max * (1 - 1e-3)).all()                # truly out of range
    assert (d[~cens] <= cfg.eps_max * (1 + 1e-4)).all()
    assert all(s["hi"] <= cfg.eps_max for s in st if not s["censored"])
    assert all(s["check_flip"] is None for s in st)                   # r = 0: no check
    assert all(s["lo"] == cfg.eps_max for s in st if s["censored"])


def test_overestimated_radius_is_caught():
    clfs, x, y, d = linear(seed=2)
    cfg = min_eps.MinEpsConfig(step=1, attack="apgd-t", steps=30, eps_floor=0.01, eps_max=50.0)
    st = run_chunk(cfg, clfs, x, y, 3.0 * d)                         # radius claims too much
    assert all(s["check_flip"] for s in st)
    t = {"name": "lin", "seed": 0}
    clean = {"r_off": 3.0 * d, "r_rig": 3.0 * d}
    rows = [min_eps.instance_row(t, f"i{k}", clean, k, s) for k, s in enumerate(st)]
    v = min_eps.violations(rows)
    assert v["official"]["count"] == len(y) and v["check_below_r_flips"] == len(y)


# ---------------------------------------------------------------------------
# the full pipeline on the toy network of test_bronet_mlp_attacks
# ---------------------------------------------------------------------------

@pytest.fixture
def out(tmp_path, monkeypatch):
    monkeypatch.setenv("DERMA_REPRO_OUT", str(tmp_path / "out"))
    return tmp_path


def cfg(**kw):
    base = dict(step=2, checkpoint="best", n_images=40, steps=15, restarts=1, eps_floor=0.05,
                eps_max=6.0, bisect_steps=5)
    base.update(kw)
    return min_eps.MinEpsConfig(**base)


def search(c, underestimate=1.0, **kw):
    f, t, im = toy(underestimate=underestimate)
    return min_eps.search(c, device=torch.device("cpu"), strategy="cpu", batch_size=16,
                          feature_fn=f, targets=t, images=im, **kw)


@pytest.mark.parametrize("attack", ["pgd", "apgd-dlr"])
def test_pipeline(out, attack):
    res = search(cfg(attack=attack))
    rows = res["rows"]
    assert len(rows) == 3 * 40 and res["sanity"]["ok"]
    for r in rows:
        if not r["clean_correct"]:
            assert r["eps_star"] == 0.0
        elif not r["censored"]:
            assert r["eps_star"] >= r["radius_official"] * (1 - 1e-3)
            assert r["eps_lo"] <= r["eps_star"] and r["tightness"] >= 1 - 1e-3
    for s in res["summary"]:
        for name in ("36/255", "72/255", "108/255"):
            assert s[f"empirical_{name}"] >= s[f"certified_{name}"] - 1e-9
    od = min_eps.out_dir(2, "best", attack=attack)
    assert od.name == "step_2_best_mineps" + ("" if attack == "pgd" else "_apgd-dlr")
    md = (od / "min_eps_step2.md").read_text(encoding="utf-8")
    assert "Sanity check: OK" in md and "eps*/r" in md


def test_underestimated_L_is_caught(out):
    res = search(cfg(), underestimate=0.05)
    assert not res["sanity"]["ok"]
    assert res["sanity"]["violations"]["check_below_r_flips"] > 0


def test_resume_and_refusal(out):
    full = search(cfg())
    prog = min_eps.out_dir(2, "best") / "progress.jsonl"
    lines = prog.read_text(encoding="utf-8").splitlines()
    prog.write_text("\n".join(lines[: len(lines) // 2]) + "\n{broken", encoding="utf-8")
    resumed = search(cfg())
    key = lambda r: (r["run"], r["image"])
    a = {key(r): r["eps_star"] for r in full["rows"]}
    b = {key(r): r["eps_star"] for r in resumed["rows"]}
    assert a == b                                                    # seeded per instance
    with pytest.raises(RuntimeError, match="other settings"):
        search(cfg(bisect_steps=3))
    search(cfg(bisect_steps=3), restart=True)


def test_quantiles_with_censored_values():
    assert min_eps._quantile([1.0, 2.0, math.inf], 0.5) == 2.0
    assert math.isinf(min_eps._quantile([1.0, math.inf, math.inf], 0.5))
    assert min_eps._fmt(math.inf) == "> max"
