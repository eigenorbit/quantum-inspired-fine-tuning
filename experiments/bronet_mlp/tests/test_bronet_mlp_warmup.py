"""eps warm-up of the certificate-aware loss and the head-row logging."""

import dataclasses
import json

import pytest
import torch

from experiments.bronet_mlp import env, lossy, train, tune
from experiments.common import layout

from conftest import synthetic_inputs


@pytest.fixture
def out(tmp_path, monkeypatch):
    monkeypatch.setenv("DERMA_REPRO_OUT", str(tmp_path / "out"))
    return tmp_path


def es_inputs(tmp_path):
    inp = synthetic_inputs(tmp_path, n_train=448)
    idx = torch.arange(len(inp.train_y))
    val, fit = idx[idx % 8 == 0], idx[idx % 8 != 0]
    return dataclasses.replace(inp, train_x=inp.train_x[fit], train_y=inp.train_y[fit],
                               val_x=inp.train_x[val], val_y=inp.train_y[val])


def test_schedule_is_the_linear_ramp():
    eps = 195 / 255
    cfg = train.RunConfig(step=1, mode="butterfly", epochs=30, eps_train=eps, eps_warmup=0.33)
    for epoch in range(30):
        want = eps * min(1.0, epoch / (0.33 * 30))
        assert cfg.eps_at(epoch) == pytest.approx(want)
        assert cfg.eps_full(epoch) == (epoch >= 10)       # 0.33 * 30 = 9.9
    off = train.RunConfig(step=1, mode="butterfly", eps_train=eps)
    assert all(off.eps_at(e) == eps and off.eps_full(e) for e in range(30))
    start = train.RunConfig(step=1, mode="butterfly", epochs=30, eps_train=eps,
                            eps_warmup=0.33, eps_start=36 / 255)
    assert start.eps_at(0) == pytest.approx(36 / 255) and start.eps_at(10) == pytest.approx(eps)


@pytest.mark.parametrize("kw", [dict(eps_warmup=1.0), dict(eps_warmup=0.99),
                                dict(eps_warmup=-0.1), dict(eps_warmup=0.3, eps_start=1.0)])
def test_schedule_checks(kw):
    with pytest.raises(ValueError):
        train.RunConfig(step=1, mode="butterfly", epochs=30, **kw)


def test_older_runs_had_no_warmup():
    saved = train.RunConfig(step=2, mode="lora").to_dict()
    del saved["eps_warmup"], saved["eps_start"]
    compat = train.RunConfig.saved_compat(saved)
    assert compat["eps_warmup"] == 0.0 and compat["eps_start"] == 0.0
    assert compat == train.RunConfig(step=2, mode="lora").compat()


def test_head_distances():
    class M:
        class head:
            @staticmethod
            def get_weight():
                return torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    d = train.head_distances(M)
    assert d["head_dist_min"] == pytest.approx(2 ** 0.5)
    assert d["head_dist_max"] == pytest.approx(2.0)


def test_fit_ramps_eps_and_picks_best_after_the_ramp(tmp_path, device):
    cfg = train.RunConfig(step=2, mode="butterfly", seed=0, epochs=6, batch_size=64, lr=3e-2,
                          device=device, early_stop="val", eps_train=108 / 255,
                          eps_warmup=0.5)
    rd = tmp_path / "run"
    m = train.fit(cfg, es_inputs(tmp_path), rd)
    hist = env.read_json(rd / "history.json")
    assert [h["eps"] for h in hist] == pytest.approx([cfg.eps_at(e) for e in range(6)])
    assert [h["eps_full"] for h in hist] == [False] * 3 + [True] * 3
    assert all(0 <= h["head_dist_min"] <= h["head_dist_median"] <= h["head_dist_max"] <= 2
               for h in hist)
    assert m["epoch"] >= 3 and m["early_stopping"]["epoch"] >= 3
    assert not any(h["is_best_so_far"] for h in hist[:3])


def test_no_warmup_trains_as_before(tmp_path, device):
    """Without a warm-up the schedule is the constant eps_train: the run equals one
    trained with the warm-up fields absent from the config (older code)."""
    base = dict(step=2, mode="head-only", seed=0, epochs=3, batch_size=64, lr=3e-2,
                device=device)
    a = train.fit(train.RunConfig(**base), synthetic_inputs(tmp_path), tmp_path / "a")
    hist = env.read_json(tmp_path / "a" / "history.json")
    assert all(h["eps"] == 36 / 255 and h["eps_full"] for h in hist)
    b = train.fit(train.RunConfig(**base, eps_warmup=0.0, eps_start=0.0),
                  synthetic_inputs(tmp_path), tmp_path / "b")
    assert a["fingerprint"] == b["fingerprint"]


def test_tuning_candidates_skip_the_ramp():
    hist = [{"epoch": e, "test_accuracy": 90.0 - e, "test_f1_weighted": 50.0,
             "L_mlp_end_power": 1.0, "test_certified_accuracy_36": 0.5, "eps_full": e >= 2}
            for e in range(4)]
    cands = tune.epoch_candidates(hist, 1e-3, majority_rate=40.0)
    assert [c["epochs"] for c in cands] == [3, 4]
    old = [{k: v for k, v in h.items() if k != "eps_full"} for h in hist]
    assert len(tune.epoch_candidates(old, 1e-3, majority_rate=40.0)) == 4


def test_retuning_goes_to_its_own_folder(out):
    cfg = train.RunConfig(step=1, mode="butterfly", eps_train=195 / 255, eps_warmup=0.33)
    tag = tune.config_tag(cfg)
    assert tag == "_eps195_wu0.33"
    assert tune.config_tag(train.RunConfig(step=1, mode="butterfly")) == ""
    root = env.repro_out()
    assert tune.selection_path(1, 256, tag).relative_to(root).as_posix() == \
        "bronet_mlp/tuning/step_1_eps195_wu0.33/selection.json"
    assert tune.run_dir(1, "butterfly", 0.01, tag=tag).relative_to(root).as_posix() == \
        "bronet_mlp/tuning/step_1_eps195_wu0.33/butterfly/lr0.01/seed0"
    assert tune.selection_path(1) != tune.selection_path(1, 256, tag)


def write_selection(lrs: dict, drop=()):
    """A protocol selection; `drop`: FIXED_KEYS absent, as in selections made
    before they were added."""
    cfg = train.RunConfig(step=2, mode="butterfly")
    sel = {"fixed": {k: getattr(cfg, k) for k in tune.FIXED_KEYS if k not in drop},
           "modes": {m: {"variants": {"best": {"lr": lr, "epochs": 4}, "stable": None},
                         "stable_equals_best": False} for m, lr in lrs.items()}}
    path = tune.selection_path(2)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sel), encoding="utf-8")


def test_old_selection_still_serves_the_protocol(out):
    write_selection({"butterfly": 3e-3}, drop=("eps_warmup", "eps_start"))
    cfg = train.RunConfig(step=2, mode="butterfly")
    assert tune.tuned_params(2, "butterfly", cfg)["lr"] == 3e-3
    with pytest.raises(RuntimeError, match="eps_warmup"):
        tune.tuned_params(2, "butterfly", dataclasses.replace(cfg, eps_warmup=0.33))


def test_sweep_with_warmup_paths_and_config(out):
    write_selection({"butterfly": 3e-3}, drop=("eps_warmup", "eps_start"))
    root = env.repro_out()
    assert lossy.sweep_run_dir(2, 36, "butterfly", 1) == env.run_dir(2, "butterfly", 1)
    assert lossy.sweep_run_dir(2, 36, "butterfly", 1, warmup=0.33).relative_to(root).as_posix() \
        == "bronet_mlp/eps_sweep/step_2/eps36_wu0.33/butterfly/seed1"
    assert lossy.sweep_run_dir(2, 195, "butterfly", 1, "best", 0.33, 9).relative_to(root) \
        .as_posix() == "bronet_mlp/eps_sweep/step_2/eps195_wu0.33from9/butterfly/seed1"
    plain = lossy.sweep_config(2, "butterfly", 1, 195)
    warm = lossy.sweep_config(2, "butterfly", 1, 195, warmup=0.33, start_255=9)
    assert (warm.eps_warmup, warm.eps_start) == (0.33, pytest.approx(9 / 255))
    assert (plain.eps_warmup, plain.eps_start) == (0.0, 0.0)
    assert {k: v for k, v in warm.compat().items() if k not in ("eps_warmup", "eps_start")} == \
        {k: v for k, v in plain.compat().items() if k not in ("eps_warmup", "eps_start")}
    with pytest.raises(ValueError):
        lossy.sweep_config(2, "butterfly", 1, 195, warmup=1.0)
    assert layout.warmup_tag(warm.eps_warmup, warm.eps_start) == "_wu0.33from9"
