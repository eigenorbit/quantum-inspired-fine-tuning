"""Rotations inside the last BRO blocks (conv_adapt.py, partitioned.py), on a small
synthetic stand-in for the frozen BRONet tail: no BRONet, no dataset."""

import dataclasses

import numpy as np
import pytest
import torch

from experiments.bronet_mlp import adapters, conv_adapt, env, train
from experiments.bronet_mlp.partitioned import PartitionedButterfly, binary_parts

from conftest import synthetic_inputs

WIDTH, SIDE, FEAT = 12, 3, 64          # 12 = 8 + 4 channels; neck -> 64 = MLP width


# ---------------------------------------------------------------------------
# PartitionedButterfly
# ---------------------------------------------------------------------------

def test_binary_parts():
    assert binary_parts(588) == [512, 64, 8, 4]
    assert binary_parts(2048) == [2048]
    assert binary_parts(13) == [8, 4, 1]
    with pytest.raises(ValueError):
        binary_parts(0)


@pytest.mark.parametrize("width,stages", [(588, 2), (12, 2), (13, 3), (64, 1)])
def test_partitioned_is_orthogonal_and_starts_at_identity(width, stages):
    r = PartitionedButterfly(width, stages, perm_seed=3)
    assert torch.equal(r.matrix(), torch.eye(width))
    with torch.no_grad():
        for stage in r.blocks:
            for b in stage:
                b.theta.normal_(0, 1)
    assert r.orthogonality_error() < 1e-5
    x = torch.randn(5, width, 4, 4)
    assert torch.allclose(r(x).norm(dim=1), x.norm(dim=1), atol=1e-4)
    v = torch.randn(7, width)
    assert torch.allclose(r(v), v @ r.matrix().t(), atol=1e-6)


def test_partitioned_mixes_all_parts_and_has_gradients():
    r = PartitionedButterfly(588)
    assert r.num_angles == 2 * (2304 + 192 + 12 + 4)
    with torch.no_grad():
        for stage in r.blocks:
            for b in stage:
                b.theta.normal_(0, 1)
    R = r.matrix()
    assert (R.abs() > 1e-9).float().mean() > 0.9            # the parts are mixed
    r.matrix().sum().backward()
    assert all(b.theta.grad is not None for stage in r.blocks for b in stage)
    single = PartitionedButterfly(588, stages=1)
    with torch.no_grad():
        for b in single.blocks[0]:
            b.theta.normal_(0, 1)
    assert (single.matrix()[:512, 512:] == 0).all()          # one stage: block diagonal


# ---------------------------------------------------------------------------
# a synthetic frozen tail with the interface of conv_adapt.BroTail
# ---------------------------------------------------------------------------

class FakeTail:
    """Two orthogonal 1x1 'convolutions', MinMax, and a fixed linear neck."""

    def __init__(self, n_blocks=2, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.blocks = list(range(8 - n_blocks, 8))
        self.width = WIDTH
        self.q = [torch.linalg.qr(torch.randn(WIDTH, WIDTH, generator=g))[0] for _ in self.blocks]
        self.b = [0.05 * torch.randn(WIDTH, 1, 1, generator=g) for _ in self.blocks]
        self.neck_w = torch.randn(FEAT, WIDTH * SIDE * SIDE, generator=g) / (WIDTH * SIDE)

    def to(self, device):
        self.q = [t.to(device) for t in self.q]
        self.b = [t.to(device) for t in self.b]
        self.neck_w = self.neck_w.to(device)
        return self

    def conv(self, h, j):
        return torch.einsum("oc,bchw->bohw", self.q[j], h)

    def post(self, z, j):
        return adapters.minmax(z + self.b[j])

    def neck(self, h):
        return adapters.minmax(h.flatten(1) @ self.neck_w.t())

    def phi(self, h):
        for j in range(len(self.blocks)):
            h = self.post(self.conv(h, j), j)
        return self.neck(h)


def fake_data(tmp_path, tail, n_train=320, n_test=64, seed=0):
    g = torch.Generator().manual_seed(seed + 5)
    centers = torch.randn(3, WIDTH, SIDE, SIDE, generator=g)

    def draw(n):
        y = torch.arange(n) % 3
        return (centers[y] + 0.7 * torch.randn(n, WIDTH, SIDE, SIDE, generator=g)).half(), y

    htr, ytr = draw(n_train)
    hte, yte = draw(n_test)
    base = synthetic_inputs(tmp_path, width=FEAT, n_train=n_train, n_test=n_test)
    with torch.no_grad():
        phi_te = tail.phi(hte.float().to(tail.neck_w.device)).cpu()
    inputs = dataclasses.replace(base, train_y=ytr, test_x=phi_te, test_y=yte)
    idx = np.arange(n_train)
    return conv_adapt.ConvData(h_train=htr.numpy(), h_test=hte, y_train=ytr,
                               fit_idx=idx[idx % 8 != 0], val_idx=idx[idx % 8 == 0],
                               inputs=inputs, cache_sha256="cache")


def cfg_for(device, **kw):
    base = dict(step=2, blocks=2, mlp_mode="butterfly", seed=0, epochs=3, batch_size=64,
                micro_batch=32, lr=3e-2, lr_conv=3e-2, device=device)
    base.update(kw)
    return conv_adapt.ConvConfig(**base)


def test_model_starts_at_the_frozen_function(tmp_path, device):
    tail = FakeTail().to(device)
    data = fake_data(tmp_path, tail)
    model = conv_adapt.build_model(cfg_for(device), tail, data.inputs, 3, device)
    h = data.h_test[:16].float().to(device)
    with torch.no_grad():
        assert torch.allclose(model.features(h), tail.phi(h), atol=1e-6)
        ref = adapters.build(data.inputs.base_weight, data.inputs.base_bias, "butterfly", 3,
                             seed=0).to(device)
        assert torch.allclose(model(h), ref(tail.phi(h)), atol=1e-5)
    names = set(model.trainable_state())
    assert any(n.startswith("rotations.") for n in names)
    assert any(n.startswith("clf.mlp.") for n in names) and any(n.startswith("clf.head.") for n in names)
    assert model.num_trainable() == sum(r.num_angles for r in model.rotations) + \
        model.clf.num_trainable()


def test_fit_writes_protocol_outputs_and_keeps_rotations_orthogonal(tmp_path, device):
    tail = FakeTail().to(device)
    data = fake_data(tmp_path, tail)
    rd = tmp_path / "run"
    m = conv_adapt.fit(cfg_for(device), data, tail, rd)
    assert m["headline"] == "best" and train.is_complete(rd)
    for which in ("best", "final"):
        mw = env.read_json(rd / f"metrics_{which}.json")
        assert mw["conv_rotations"]["orthogonality_error_max"] < 1e-5
        assert mw["lipschitz"]["L_over_L0"] == pytest.approx(1.0, abs=1e-5)   # butterfly MLP
        assert (rd / f"predictions_{which}.csv").exists()
        assert (rd / f"certificates_{which}.npz").exists()
    hist = env.read_json(rd / "history.json")
    assert len(hist) == 3 and all("val_accuracy" in h for h in hist)
    # the angles moved: the rotation is not the identity any more
    state = torch.load(rd / "checkpoints" / "final.pt", weights_only=True)["trainable_state"]
    assert any(v.abs().max() > 0 for k, v in state.items() if k.startswith("rotations."))


@pytest.mark.parametrize("mlp_mode", ["head-only", "lora", "full"])
def test_fit_with_every_mlp_mode(tmp_path, device, mlp_mode):
    tail = FakeTail().to(device)
    data = fake_data(tmp_path, tail)
    rd = tmp_path / mlp_mode
    m = conv_adapt.fit(cfg_for(device, mlp_mode=mlp_mode, epochs=2, lr=1e-3), data, tail, rd)
    assert m["config"]["mode"] == mlp_mode and train.is_complete(rd)
    assert m["conv_rotations"]["orthogonality_error_max"] < 1e-5
    state = torch.load(rd / "checkpoints" / "final.pt", weights_only=True)["trainable_state"]
    assert any(k.startswith("rotations.") for k in state)
    if mlp_mode == "lora":
        assert any("lora_A" in k for k in state) and any("lora_B" in k for k in state)
    hist = env.read_json(rd / "history.json")
    if mlp_mode in ("lora", "full"):        # L of the MLP is re-estimated every epoch
        assert all(h["L_train"] > 0 for h in hist)


def test_resume_equals_straight_run(tmp_path, device):
    tail = FakeTail().to(device)
    data = fake_data(tmp_path, tail)
    straight = conv_adapt.fit(cfg_for(device), data, tail, tmp_path / "a")
    rd = tmp_path / "b"
    assert conv_adapt.fit(cfg_for(device), data, tail, rd, stop_after=2) is None
    resumed = conv_adapt.fit(cfg_for(device), data, tail, rd)
    assert resumed["fingerprint"] == straight["fingerprint"]
    assert resumed["other_checkpoints"]["final"]["fingerprint"] == \
        straight["other_checkpoints"]["final"]["fingerprint"]


def test_micro_batches_give_the_same_update(tmp_path, device):
    tail = FakeTail().to(device)
    data = fake_data(tmp_path, tail)
    s = {}
    for micro in (64, 16):
        rd = tmp_path / f"m{micro}"
        conv_adapt.fit(cfg_for(device, epochs=1, micro_batch=micro), data, tail, rd)
        s[micro] = torch.load(rd / "checkpoints" / "final.pt", weights_only=True)["trainable_state"]
    for k in s[64]:
        assert torch.allclose(s[64][k], s[16][k], atol=1e-4), k


def test_resume_refuses_other_settings(tmp_path, device):
    tail = FakeTail().to(device)
    data = fake_data(tmp_path, tail)
    rd = tmp_path / "run"
    conv_adapt.fit(cfg_for(device), data, tail, rd, stop_after=1)
    with pytest.raises(RuntimeError, match="not compatible"):
        conv_adapt.fit(cfg_for(device, lr_conv=1e-3), data, tail, rd)


# ---------------------------------------------------------------------------
# learning-rate tuning
# ---------------------------------------------------------------------------

@pytest.fixture
def out(tmp_path, monkeypatch):
    monkeypatch.setenv("DERMA_REPRO_OUT", str(tmp_path / "out"))
    return tmp_path


def write_protocol_selection(variants: dict):
    """tuning/step_2/selection.json of the protocol: {mode: (best lr, stable lr | None)}."""
    import json

    from experiments.bronet_mlp import tune

    cfg = train.RunConfig(step=2, mode="butterfly")
    modes = {}
    for mode, (b, s) in variants.items():
        modes[mode] = {"variants": {"best": {"lr": b, "epochs": 3},
                                    "stable": None if s is None else {"lr": s, "epochs": 3}},
                       "stable_equals_best": s == b}
    path = tune.selection_path(2)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"fixed": {k: getattr(cfg, k) for k in tune.FIXED_KEYS},
                                "modes": modes}), encoding="utf-8")


def fake_run(accs, lr_conv, lr_mlp, collapsed_last=False):
    cands = [{"lr_conv": lr_conv, "lr_mlp": lr_mlp, "epochs": i + 1, "accuracy": a,
              "f1_weighted": a, "collapsed": collapsed_last and i == len(accs) - 1}
             for i, a in enumerate(accs)]
    best = conv_adapt._select(cands)
    last = cands[-1]
    stable = not last["collapsed"] and last["accuracy"] >= best["accuracy"] - 2.0
    return cands, {**best, "final_epoch_accuracy": last["accuracy"], "stable_run": stable}


def test_choose_best_and_stable():
    diverging = fake_run([60, 75, 50], 0.03, 1e-3)          # best epoch high, then breaks
    steady = fake_run([60, 70, 70], 0.003, 1e-3)
    assert conv_adapt.choose([diverging, steady], "best")["lr_conv"] == 0.03
    assert conv_adapt.choose([diverging, steady], "stable")["lr_conv"] == 0.003
    assert conv_adapt.choose([diverging], "stable") is None
    # ties: the earlier epoch, then the smaller lr
    a, b = fake_run([70, 70], 0.01, 1e-3), fake_run([70, 70], 0.001, 1e-3)
    c = conv_adapt.choose([a, b], "best")
    assert (c["epochs"], c["lr_conv"]) == (1, 0.001)


def test_tuning_end_to_end_and_train_from_it(out, device, monkeypatch):
    write_protocol_selection({"butterfly": (3e-2, 3e-2), "lora": (3e-3, 1e-3)})
    assert conv_adapt.variants_of(2, "butterfly") == ["best"]
    assert conv_adapt.variants_of(2, "lora") == ["best", "stable"]
    tail = FakeTail().to(device)
    data = fake_data(out, tail)
    base = cfg_for(device, epochs=2)
    grid = (1e-3, 3e-2)
    r = conv_adapt.tune_mode(2, 2, "lora", "stable", data, tail, base=base, conv_grid=grid)
    assert len(r["stage_a"]) == 2 and len(r["stage_b"]) == 2
    assert r["protocol_lr_mlp"] == 1e-3
    assert {s["lr_mlp"] for s in r["stage_b"]} == {conv_adapt._lr(1e-3 / 3), 3e-3}
    if not r["mlp_lr_changed"]:
        assert r["lr_mlp"] == 1e-3
    run_dirs = [env.repro_out() / "bronet_mlp" / "conv_adapt" / "tuning" / "step_2" / "last2"
                / "lora_stable" / f"conv{lc:g}_mlp0.001" / "seed0" for lc in grid]
    assert all(conv_adapt.tune_complete(d) for d in run_dirs)
    assert not any((d / "metrics_final.json").exists() for d in run_dirs)   # test never used
    # a second call reuses the finished runs
    again = conv_adapt.tune_mode(2, 2, "lora", "stable", data, tail, base=base, conv_grid=grid)
    assert (again["lr_conv"], again["lr_mlp"]) == (r["lr_conv"], r["lr_mlp"])
    sel = conv_adapt.write_selection(2, 2, {"lora": {"stable": r}}, conv_adapt.tuning_fixed(base),
                                     grid)
    assert sel["modes"]["lora"]["stable"]["lr_conv"] == r["lr_conv"]
    assert (conv_adapt.selection_path(2, 2).with_name("tuning_step2_last2.md")).exists()
    lrs = conv_adapt.tuned_lrs(2, 2, "lora", "stable", base)
    assert lrs == {"lr_mlp": r["lr_mlp"], "lr_conv": r["lr_conv"]}
    assert conv_adapt.tuned_lrs(2, 2, "lora", "best", base) is None
    with pytest.raises(RuntimeError, match="other settings"):
        conv_adapt.tuned_lrs(2, 2, "lora", "stable", dataclasses.replace(base, epochs=5))
    # micro-batch is not a tuning setting
    assert conv_adapt.tuned_lrs(2, 2, "lora", "stable",
                                dataclasses.replace(base, micro_batch=16)) == lrs


def test_stage_b_switch_needs_the_margin(out, device, monkeypatch):
    """Stage B changes the MLP lr only when it wins by MLP_SWITCH_MARGIN points."""
    write_protocol_selection({"butterfly": (1e-3, 1e-3)})
    tail = FakeTail().to(device)
    data = fake_data(out, tail)
    accs = {(1e-3, 1e-3): [70, 71], (3e-3, 1e-3): [70, 72],
            (3e-3, conv_adapt._lr(1e-3 / 3)): [70, 72.5], (3e-3, 3e-3): [70, 72.9]}

    def fake_candidates(run_dir, lr_conv, lr_mlp):
        return fake_run(accs[(lr_conv, lr_mlp)], lr_conv, lr_mlp)

    monkeypatch.setattr(conv_adapt, "fit", lambda *a, **k: None)
    monkeypatch.setattr(conv_adapt, "tune_complete", lambda d: False)
    monkeypatch.setattr(conv_adapt, "run_candidates", fake_candidates)
    r = conv_adapt.tune_mode(2, 2, "butterfly", "best", data, tail, base=cfg_for(device),
                             conv_grid=(1e-3, 3e-3))
    assert r["lr_conv"] == 3e-3 and r["lr_mlp"] == 1e-3 and not r["mlp_lr_changed"]   # +0.9 < 1
    accs[(3e-3, 3e-3)] = [70, 73.5]
    r = conv_adapt.tune_mode(2, 2, "butterfly", "best", data, tail, base=cfg_for(device),
                             conv_grid=(1e-3, 3e-3))
    assert r["lr_mlp"] == 3e-3 and r["mlp_lr_changed"]                                 # +1.5


# ---------------------------------------------------------------------------
# attacks on conv runs, export
# ---------------------------------------------------------------------------

def test_attacks_and_mineps_on_a_conv_run(out, device):
    from experiments.bronet_mlp import conv_attacks
    from experiments.common.attacks import core
    from experiments.common.attacks import min_eps as mineps_engine

    tail = FakeTail().to(device)
    data = fake_data(out, tail)
    rd = conv_adapt.TREE.conv_run_dir(2, 2, "butterfly", 0)
    conv_adapt.fit(cfg_for(device), data, tail, rd)
    model, cfg = conv_attacks.load_conv_run(rd, "final", tail, data.inputs.base_weight,
                                            data.inputs.base_bias, device)
    assert cfg.mlp_mode == "butterfly" and not any(p.requires_grad for p in model.parameters())
    m = env.read_json(rd / "metrics_final.json")
    # the restored model reproduces the saved test logits
    saved = np.load(rd / "certificates_final.npz")["logits"]
    with torch.no_grad():
        z = model(data.h_test.float().to(device)).cpu().numpy()
    assert np.abs(z - saved).max() < 1e-4
    target = {"name": "butterfly", "seed": 0, "dir": rd, "which": "final", "clf": model.clf,
              "L_official": m["lipschitz"]["L_total_official"],
              "L_rigorous": m["lipschitz"]["L_total_rigorous"]}
    images = {"x": data.h_test[:12].float(), "y": data.inputs.test_y[:12],
              "names": [f"img{i}.jpg" for i in range(12)], "test_index": list(range(12))}
    fn = lambda x, grad: model.features(x)                     # the tail is the whole network
    acfg = core.AttackConfig(step=2, checkpoint="final", eps=[0.05, 0.5], steps=5)
    res = core.run(acfg, device=torch.device(device), strategy="cpu", targets=[target],
                   images=images, feature_fn=fn, out=out / "att")
    assert res["sanity"]["ok"] and len(res["rows"]) == 24
    assert (out / "att" / "attacks_step2.csv").exists()
    mcfg = mineps_engine.MinEpsConfig(step=2, checkpoint="final", steps=5, bisect_steps=2,
                                      eps_max=2.0)
    res = mineps_engine.search(mcfg, device=torch.device(device), strategy="cpu",
                               targets=[target], images=images, feature_fn=fn, out=out / "me")
    assert res["sanity"]["ok"] and len(res["rows"]) == 12


def test_lossy_coverage_of_conv_runs(out, device):
    from experiments.bronet_mlp import conv_attacks, conv_lossy, lossy

    tail = FakeTail().to(device)
    data = fake_data(out, tail)
    rd = conv_adapt.TREE.conv_run_dir(2, 2, "butterfly", 0)
    conv_adapt.fit(cfg_for(device), data, tail, rd)
    model, _ = conv_attacks.load_conv_run(rd, "final", tail, data.inputs.base_weight,
                                          data.inputs.base_bias, device)
    arrays = dict(np.load(rd / "certificates_final.npz"))
    h = data.h_test.float()
    g = torch.Generator().manual_seed(3)
    noise = torch.randn(h.shape, generator=g)
    sizes = {"jpeg90": 0.02, "jpeg80": 0.5}

    class FakeInputs:
        images = [f"img{i}.jpg" for i in range(len(h))]

        def clean(self):
            return h

        def transformed(self, key):
            s = sizes.get(key, 0.0)
            d = s * noise / noise.flatten(1).norm(dim=1).view(-1, 1, 1, 1)
            blob = {"h": h + d, "dist": d.flatten(1).double().norm(dim=1), "label": key}
            blob["dist_attack"] = blob["dist"] + 1.0
            return blob

    keys = lossy.with_identity(["jpeg90", "jpeg80"], "ref")
    per_seed, per_image, mism, blobs = conv_lossy.evaluate(
        FakeInputs(), [("butterfly", 0, model, arrays)], keys, device)
    assert mism == {"butterfly/seed0": 0}
    assert {r["operation"] for r in per_seed} == {"none", "jpeg90", "jpeg80"}
    for r in per_image:
        assert r["covered_official"] == (r["correct"] and r["dist"] < r["radius_official"])
        assert r["covered_attack_official"] == (r["correct"] and r["dist_attack"] < r["radius_official"])
    none = next(r for r in per_seed if r["operation"] == "none")
    assert none["changed"] == 0 and none["covered_official"] == 100.0
    assert "h" not in blobs["jpeg90"]
    od = conv_adapt.TREE.conv_lossy_dir(2, 2, "final", lossy.place("native", "ref"))
    lossy.write_measure(od, 2, "final", "native", blobs, per_seed, "ref")
    assert "covered without renormalisation" in (od / "coverage.md").read_text(encoding="utf-8")


def test_export_has_numbers_only(out, device):
    tail = FakeTail().to(device)
    data = fake_data(out, tail)
    rd = conv_adapt.TREE.conv_run_dir(2, 2, "butterfly", 0)
    conv_adapt.fit(cfg_for(device, epochs=1), data, tail, rd)
    cache = conv_adapt.TREE.conv_features_dir(2, 2, "float16")
    cache.mkdir(parents=True)
    (cache / "train.npy").write_bytes(b"x")
    (cache / "index.json").write_text("{}")
    other = conv_adapt.TREE.conv_attack_dir(1, 2, "final")    # another step: not exported
    other.mkdir(parents=True)
    (other / "attacks_step1.csv").write_text("a")
    mine = conv_adapt.TREE.conv_attack_dir(2, 2, "final") / "butterfly" / "seed0"
    mine.mkdir(parents=True)
    (mine / "progress.jsonl").write_text("{}")
    z = conv_adapt.export(2, 2)
    import zipfile

    names = zipfile.ZipFile(z).namelist()
    assert not any(n.endswith((".pt", ".npy")) for n in names)
    assert any(n.endswith("seed0/metrics_final.json") for n in names)
    assert any(n.endswith("certificates_final.npz") for n in names)
    assert any(n.endswith("history.json") for n in names)
    assert any(n.endswith("conv_features/step_2/last2_float16/index.json") for n in names)
    assert any(n.endswith("attacks_step2_last2_final/butterfly/seed0/progress.jsonl") for n in names)
    assert not any("attacks_step1_last2" in n for n in names)


def test_config_checks():
    with pytest.raises(ValueError):
        conv_adapt.ConvConfig(step=2, batch_size=256, micro_batch=48)
    with pytest.raises(ValueError):
        conv_adapt.ConvConfig(step=2, mlp_mode="nope")
    c = conv_adapt.ConvConfig(step=2, mlp_mode="head-only")
    rc = c.run_config()
    assert (rc.mode, rc.early_stop, rc.batch_size, rc.eps_train) == ("head-only", "val", 256, 36 / 255)
    assert "device" not in c.compat()
