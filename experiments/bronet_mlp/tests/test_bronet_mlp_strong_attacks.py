"""APGD and Square (strong_attacks.py): losses, invariants, exactness on a linear
model where the worst case is known, and the full `attacks.run` pipeline."""

import argparse

import pytest
import torch

from experiments.bronet_mlp import attacks
from experiments.bronet_mlp import strong_attacks as sa

from test_bronet_mlp_attacks import EPS, cfg, run


# ---------------------------------------------------------------------------
# losses
# ---------------------------------------------------------------------------

def test_dlr_loss_values_and_scale_invariance():
    z = torch.tensor([[3.0, 1.0, 0.0, -1.0], [0.0, 2.0, 1.0, -1.0]])
    y = torch.tensor([0, 0])
    # row 0: -(3 - 1) / (3 - 0); row 1: -(0 - 2) / (2 - 0)
    assert torch.allclose(sa.dlr_loss(z, y), torch.tensor([-2 / 3, 1.0]))
    assert torch.allclose(sa.dlr_loss(7.5 * z + 3.0, y), sa.dlr_loss(z, y))
    t = torch.tensor([3, 1])
    # -(z_y - z_t) / (z_pi1 - (z_pi3 + z_pi4) / 2)
    want = torch.tensor([-(3 - -1) / (3 - (0 + -1) / 2), -(0 - 2) / (2 - (0 + -1) / 2)])
    assert torch.allclose(sa.dlr_targeted_loss(z, y, t), want)
    with pytest.raises(ValueError):
        sa.dlr_targeted_loss(z[:, :3], y, t.clamp(max=2))
    assert torch.equal(sa.attack_loss("margin-t", z, y, t), torch.tensor([-4.0, 2.0]))


def test_target_order_excludes_the_true_class():
    z = torch.tensor([[0.1, 3.0, 2.0, 5.0], [9.0, 1.0, 2.0, 3.0]])
    order = sa.target_order(z, torch.tensor([3, 0]))
    assert order.tolist() == [[1, 2, 0], [3, 2, 1]]


# ---------------------------------------------------------------------------
# Square building blocks
# ---------------------------------------------------------------------------

def test_square_schedule_and_bump():
    ps = [sa.p_selection(0.8, it, 5000) for it in range(5000)]
    assert ps[0] == 0.8 and ps[-1] == 0.8 / 512
    assert all(a >= b for a, b in zip(ps, ps[1:]))
    g = torch.Generator().manual_seed(0)
    for s in (1, 3, 5, 44):
        e = sa._eta(s, g)
        assert e.shape == (s, s) and torch.isclose(e.norm(), torch.tensor(1.0))


def test_square_init_has_norm_eps():
    eps = torch.tensor([0.5, 2.0, 7.0])
    gens = [torch.Generator().manual_seed(i) for i in range(3)]
    for shape in ((3, 3, 4, 4), (3, 3, 224, 224)):
        d = sa.square_init(shape, eps, gens)
        assert torch.allclose(attacks._norms(d), eps, rtol=1e-4)


# ---------------------------------------------------------------------------
# exactness on a linear model: the worst case is known in closed form
# ---------------------------------------------------------------------------

def linear(n=60, classes=5, seed=0, hw=4):
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(classes, 3 * hw * hw, generator=g)
    x = torch.randn(n, 3, hw, hw, generator=g)
    z = x.flatten(1) @ W.T
    y = z.argmax(1)                                  # every image clean-correct

    def fwd(v):
        return v.flatten(1) @ W.T

    # the attack can succeed iff  max_j (z_j - z_y) + eps ||w_j - w_y|| > 0
    def vulnerable(eps):
        dist = (W[None, :, :] - W[y][:, None, :]).norm(dim=2)
        gap = z - z.gather(1, y[:, None])
        val = (gap + eps[:, None] * dist).masked_fill(
            torch.nn.functional.one_hot(y, classes).bool(), float("-inf"))
        return val.amax(1)
    return fwd, x, y, vulnerable


def test_apgd_targeted_is_exact_on_a_linear_model():
    fwd, x, y, vulnerable = linear()
    eps = torch.full((len(y),), 1.5)
    v = vulnerable(eps)
    clear = v.abs() > 0.05                           # away from the decision boundary
    c = argparse.Namespace(attack="apgd-t", restarts=1, steps=50, target_classes=0)
    (d,) = sa.deltas(c, lambda u: u, [fwd], torch.zeros(len(y), dtype=torch.long), x, y, eps,
                     list(range(len(y))))
    assert (attacks._norms(d) <= eps * (1 + 1e-5)).all()
    success = attacks.margin_loss(fwd(x + d), y) > 0
    assert success.any() and (~success).any()        # the test is not trivial
    assert torch.equal(success[clear], (v > 0)[clear])


@pytest.mark.parametrize("loss", ["ce", "dlr"])
def test_apgd_untargeted_is_sound_and_finds_most(loss):
    fwd, x, y, vulnerable = linear(seed=1)
    eps = torch.full((len(y),), 1.5)
    d = sa.apgd_run(fwd, x, y, eps, n_iter=50, loss=loss, seeds=list(range(len(y))))
    assert (attacks._norms(d) <= eps * (1 + 1e-5)).all()
    success = attacks.margin_loss(fwd(x + d), y) > 0
    vul = vulnerable(eps) > 0
    assert not (success & ~vul).any()                # cannot beat the closed form
    assert success.sum() >= 0.9 * vul.sum()


def test_square_is_sound_finds_most_and_keeps_the_norm():
    fwd, x, y, vulnerable = linear(seed=2, hw=8)
    eps = torch.full((len(y),), 3.0)
    d, q = sa.square_run(lambda v, rows: fwd(v), x, y, eps, n_queries=300, p_init=0.8, seeds=list(range(len(y))))
    assert torch.allclose(attacks._norms(d), eps, rtol=1e-4)
    success = attacks.margin_loss(fwd(x + d), y) > 0
    vul = vulnerable(eps) > 0
    assert not (success & ~vul).any()
    assert success.sum() >= 0.6 * vul.sum()
    assert (q[success] < 300).all() and (q[~success] == 300).all()   # stops on success


def test_square_and_apgd_do_not_depend_on_batching():
    fwd, x, y, _ = linear(n=6, seed=3, hw=8)
    eps = torch.full((6,), 3.0)
    seeds = [11, 22, 33, 44, 55, 66]
    d_all, _ = sa.square_run(lambda v, rows: fwd(v), x, y, eps, n_queries=60, p_init=0.8, seeds=seeds)
    d_one = torch.cat([sa.square_run(lambda v, rows: fwd(v), x[i:i + 1], y[i:i + 1], eps[i:i + 1], n_queries=60,
                                     p_init=0.8, seeds=seeds[i:i + 1])[0] for i in range(6)])
    assert torch.allclose(d_all, d_one, atol=1e-6)
    a_all = sa.apgd_run(fwd, x, y, eps, n_iter=20, loss="dlr", seeds=seeds)
    a_one = torch.cat([sa.apgd_run(fwd, x[i:i + 1], y[i:i + 1], eps[i:i + 1], n_iter=20,
                                   loss="dlr", seeds=seeds[i:i + 1]) for i in range(6)])
    assert torch.allclose(a_all, a_one, atol=1e-5)


# ---------------------------------------------------------------------------
# the full pipeline (toy network of test_bronet_mlp_attacks)
# ---------------------------------------------------------------------------

@pytest.fixture
def out(tmp_path, monkeypatch):
    monkeypatch.setenv("DERMA_REPRO_OUT", str(tmp_path / "out"))
    return tmp_path


def test_pgd_config_and_folders_unchanged(out):
    keys = set(attacks.AttackConfig(step=1).key())
    assert not keys & set(attacks.STRONG_FIELDS)     # old attack_config.json still match
    assert set(attacks.AttackConfig(step=1, attack="square").key()) >= set(attacks.STRONG_FIELDS)
    assert attacks.out_dir(2, "best").name == "step_2_best"
    assert attacks.out_dir(2, "best", 32, "apgd-t").name == "step_2_bs32_best_apgd-t"
    assert attacks.run_csv_name("pgd", "best") == "pgd_l2_best.csv"
    assert attacks.run_csv_name("apgd-ce", "final") == "apgd_ce_l2_final.csv"


def test_cli_settings():
    p = argparse.ArgumentParser()
    attacks.add_attack_args(p)
    assert attacks.attack_settings(p.parse_args([])) == {"steps": 20, "restarts": 1}
    s = attacks.attack_settings(p.parse_args(["--attack", "apgd-dlr"]))
    assert s["steps"] == 100 and s["attack"] == "apgd-dlr"
    s = attacks.attack_settings(p.parse_args(["--attack", "square", "--queries", "50"]))
    assert s["queries"] == 50 and s["attack"] == "square"


@pytest.mark.parametrize("attack", ["apgd-ce", "apgd-dlr", "apgd-t", "square"])
def test_strong_attack_through_the_pipeline(out, attack):
    c = cfg(attack=attack, steps=15, restarts=1, queries=150)
    res = run(c)
    rows = res["rows"]
    assert len(rows) == 3 * 40 * len(EPS)
    assert res["sanity"]["ok"]
    assert any(r["pgd_success"] for r in rows)
    for r in rows:
        if r["pgd_success"]:
            assert r["delta_norm"] <= r["eps"] * (1 + 1e-5)
        if r["certified_official"]:
            assert r["pgd_robust"]
    od = attacks.out_dir(2, "best", attack=attack)
    md = (od / "attacks_step2.md").read_text(encoding="utf-8")
    assert sa.LABELS[attack] + "-robust" in md and "Sanity check: OK" in md
    assert not (attacks.out_dir(2, "best") / "progress.jsonl").exists()   # PGD folder untouched


def test_strong_attack_resume_is_seeded(out):
    c = cfg(attack="apgd-dlr", steps=10, restarts=1)
    full = run(c)
    prog = attacks.out_dir(2, "best", attack="apgd-dlr") / "progress.jsonl"
    lines = prog.read_text(encoding="utf-8").splitlines()
    prog.write_text("\n".join(lines[: len(lines) // 2]) + "\n", encoding="utf-8")
    resumed = run(c)
    key = lambda r: (r["run"], r["image"], r["eps"])
    a = {key(r): r["pgd_success"] for r in full["rows"]}
    b = {key(r): r["pgd_success"] for r in resumed["rows"]}
    assert a == b
    with pytest.raises(RuntimeError, match="other settings"):
        run(cfg(attack="apgd-dlr", steps=5, restarts=1))


def test_apgd_is_at_least_as_strong_as_pgd_on_the_toy(out):
    pgd = {(s["run"], s["epsilon"]): s["pgd_robust_accuracy"]
           for s in run(cfg(steps=15, restarts=1))["summary"]}
    apgd = {(s["run"], s["epsilon"]): s["pgd_robust_accuracy"]
            for s in run(cfg(attack="apgd-t", steps=15, restarts=1))["summary"]}
    # per instance neither dominates (random starts; a borderline image can go either
    # way), but in total APGD-T finds more and never trails by more than one image
    assert sum(apgd.values()) < sum(pgd.values())
    assert all(apgd[k] <= pgd[k] + 100 / 40 + 1e-9 for k in pgd)


def test_plots_of_a_strong_attack(out):
    from experiments.bronet_mlp import plots

    run(cfg(attack="square", restarts=1, queries=50))
    path = plots.plot_step(2, "best", attack="square")
    assert path is not None and path.name == "attacks_step2_best_square.png"
    assert plots.plot_step(2, "best") is None        # no PGD results in this folder


def test_full_size_images_through_a_conv_net():
    """224x224 inputs: Square's windows (up to 201 px) and APGD on a real image shape."""
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 8, stride=8), torch.nn.ReLU(),
                              torch.nn.AdaptiveAvgPool2d(2), torch.nn.Flatten(),
                              torch.nn.Linear(16, 3)).eval()
    for p in net.parameters():
        p.requires_grad_(False)
    x = torch.rand(3, 3, 224, 224)
    y = net(x).argmax(1)
    eps = torch.tensor([0.5, 1.0, 2.0])
    d, q = sa.square_run(lambda v, rows: net(v), x, y, eps, n_queries=30, p_init=0.8,
                         seeds=[1, 2, 3])
    assert torch.allclose(attacks._norms(d), eps, rtol=1e-4) and (q <= 30).all()
    d = sa.apgd_run(net, x, y, eps, n_iter=5, loss="ce", seeds=[1, 2, 3])
    assert (attacks._norms(d) <= eps * (1 + 1e-5)).all()
