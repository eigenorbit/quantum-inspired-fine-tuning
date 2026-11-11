"""Learning-rate tuning: the validation split, the selection rule, the plumbing.
Synthetic and short (a couple of 1-epoch fits on 64-d features)."""

import pytest
import torch

from experiments.bronet_mlp import data as bdata
from experiments.bronet_mlp import env, train, tune

from conftest import synthetic_inputs


def fake_rows(n_photos=300, copies=5, n_classes=3):
    """Manifest-style train rows: several augmented copies per photograph."""
    rows = []
    for p in range(n_photos):
        label = p % n_classes
        stem = f"P{p % 40:04d}_{10000 + p}_aa_act"
        rows.append([f"model_data_step2_anon/train/{label}/{stem}.jpg", label])
        for k in range(copies - 1):
            rows.append([f"model_data_step2_anon/train/{label}/{stem}-aug{k}.jpg", label])
    return rows


def test_photo_key_groups_relabelled_copies():
    assert bdata.photo_key("x/0/P0001_00001_aa_act-aug3.jpg") == "P0001_00001"
    assert bdata.photo_key("x/1/P0001_00001_aa_ina.jpg") == "P0001_00001"


def test_validation_split_grouped_stratified_deterministic():
    rows = fake_rows()
    fit, val = bdata.validation_split(rows, 0.1, seed=0)
    assert sorted(fit + val) == list(range(len(rows))) and not set(fit) & set(val)
    assert {bdata.photo_key(rows[i][0]) for i in fit}.isdisjoint(
        {bdata.photo_key(rows[i][0]) for i in val})
    assert 0.07 < len(val) / len(rows) < 0.13
    for c in range(3):
        share = sum(rows[i][1] == c for i in val) / len(val)
        assert abs(share - 1 / 3) < 0.05
    assert bdata.validation_split(rows, 0.1, seed=0) == (fit, val)
    assert bdata.validation_split(rows, 0.1, seed=1) != (fit, val)


def test_select_rule():
    def c(lr, acc, f1=50.0, col=False, ep=10):
        return {"lr": lr, "epochs": ep, "accuracy": acc, "f1_weighted": f1, "collapsed": col}

    assert tune.select([c(1e-3, 70), c(3e-3, 72)])["lr"] == 3e-3
    assert tune.select([c(1e-3, 70, 60), c(3e-4, 70, 61)])["lr"] == 3e-4     # F1 breaks ties
    assert tune.select([c(1e-3, 70, 60, ep=5), c(3e-4, 70, 60, ep=9)])["epochs"] == 5
    assert tune.select([c(1e-3, 70, 60), c(3e-4, 70, 60)])["lr"] == 3e-4     # then smaller lr
    assert tune.select([c(1e-3, 60), c(3e-3, 75, col=True)])["lr"] == 1e-3   # collapsed skipped
    assert tune.select([c(1e-3, 40, col=True), c(3e-3, 47, col=True)])["lr"] == 3e-3


def test_epoch_candidates_mark_collapse():
    hist = [{"epoch": e, "test_accuracy": a, "test_f1_weighted": a, "L_mlp_end_power": 1.0,
             "test_certified_accuracy_36": 0.3} for e, a in enumerate([60.0, 70.0, 38.7])]
    cands = tune.epoch_candidates(hist, 1e-3, majority_rate=38.7)
    assert [c["epochs"] for c in cands] == [1, 2, 3]
    assert [c["collapsed"] for c in cands] == [False, False, True]
    assert tune.select(cands)["epochs"] == 2


def _with_manifest(inputs):
    n = len(inputs.train_y)
    rows = [[f"model_data_step2_anon/train/{int(y)}/P{i % 30:04d}_{i // 2}_aa_act"
             + ("" if i % 2 == 0 else "-aug1") + ".jpg", int(y)]
            for i, y in enumerate(inputs.train_y.tolist())]
    inputs.manifest = {"train": rows, "test": [], "train_sha256": env.sha256_json(rows),
                       "dedup_stats": None}
    assert len(rows) == n
    return inputs


def test_tuning_end_to_end(tmp_path, monkeypatch, device):
    monkeypatch.setenv("DERMA_REPRO_OUT", str(tmp_path / "out"))
    monkeypatch.setitem(tune.LR_GRID, "butterfly", (1e-3, 3e-2))
    inputs = _with_manifest(synthetic_inputs(tmp_path))
    split = tune.make_split(inputs, step=2)
    assert tune.split_path(2).exists()
    again = tune.make_split(inputs, step=2)                    # reproducible, checked
    assert again["sha256"] == split["sha256"]
    tin = tune.tuning_inputs(inputs, split)
    assert len(tin.train_y) + len(tin.test_y) == len(inputs.train_y)
    assert torch.equal(tin.test_x, inputs.train_x[torch.tensor(split["val_idx"])])

    base = train.RunConfig(step=2, mode="butterfly", seed=0, epochs=3, batch_size=64,
                           device=device)
    for lr in tune.LR_GRID["butterfly"]:
        cfg = train.RunConfig(**{**base.to_dict(), "lr": lr})
        train.fit(cfg, tin, tune.run_dir(2, "butterfly", lr, batch_size=64))
    fixed = {k: getattr(base, k) for k in tune.FIXED_KEYS}
    collected = tune.collect(2, ["butterfly"], tune.LR_GRID, max_epochs=3, batch_size=64)
    sel = tune.write_selection(2, collected, fixed, split)
    s_bf = sel["modes"]["butterfly"]
    assert s_bf["lr"] in tune.LR_GRID["butterfly"] and 1 <= s_bf["epochs"] <= 3
    # the chosen (lr, epochs) is the validation optimum over every epoch of every lr
    best = max(((h["test_accuracy"], lr, h["epoch"] + 1)
                for lr in tune.LR_GRID["butterfly"]
                for h in env.read_json(tune.run_dir(2, "butterfly", lr, batch_size=64) / "history.json")),
               key=lambda t: t[0])
    assert s_bf["chosen"]["accuracy"] == best[0]
    # batch 64 is not the default 256: its tuning lives in its own folder, the default one is untouched
    assert (env.tuning_dir(2, 64) / "tuning_step2_bs64.md").exists()
    assert tune.selection_path(2, 64).exists() and not tune.selection_path(2).exists()
    assert tune.tuned_params(2, "butterfly", base) == {"lr": s_bf["lr"], "epochs": s_bf["epochs"], "ceiling": 3}
    # a different epoch ceiling is fine (it is what gets replaced), other settings are not
    assert tune.tuned_params(2, "butterfly", train.RunConfig(**{**base.to_dict(), "epochs": 30}))
    with pytest.raises(RuntimeError, match="other fixed settings"):
        tune.tuned_params(2, "butterfly", train.RunConfig(**{**base.to_dict(), "tau": 2.0}))
    with pytest.raises(KeyError):
        tune.tuned_params(2, "lora", base)
    with pytest.raises(RuntimeError, match="ceiling"):
        tune.collect(2, ["butterfly"], tune.LR_GRID, max_epochs=30, batch_size=64)

    # tuning another mode later keeps the butterfly selection
    monkeypatch.setitem(tune.LR_GRID, "head-only", (1e-3,))
    train.fit(train.RunConfig(**{**base.to_dict(), "mode": "head-only", "lr": 1e-3}), tin,
              tune.run_dir(2, "head-only", 1e-3, batch_size=64))
    sel2 = tune.write_selection(2, tune.collect(2, ["head-only"], tune.LR_GRID, 3, batch_size=64),
                                fixed, split)
    assert set(sel2["modes"]) == {"head-only", "butterfly"}
    assert sel2["modes"]["butterfly"] == s_bf


@pytest.mark.parametrize("mode", ["butterfly", "full"])
def test_short_run_equals_prefix_of_long_run(tmp_path, device, mode):
    """Training E epochs == the first E epochs of a longer run (what makes
    choosing the epoch count on validation and retraining for E epochs exact)."""
    long_cfg = train.RunConfig(step=2, mode=mode, epochs=4, batch_size=64, device=device,
                               keep_every=1)
    train.fit(long_cfg, synthetic_inputs(tmp_path), tmp_path / "long")
    ck = torch.load(tmp_path / "long" / "checkpoints" / "epoch_002.pt", map_location="cpu",
                    weights_only=True)
    short = train.fit(train.RunConfig(**{**long_cfg.to_dict(), "epochs": 2, "keep_every": 0}),
                      synthetic_inputs(tmp_path), tmp_path / "short")
    assert short["fingerprint"] == ck["fingerprint"]


def test_completed_run_with_other_config_is_detected(tmp_path, device):
    rd = tmp_path / "run"
    cfg = train.RunConfig(step=2, mode="head-only", epochs=1, batch_size=64, device=device)
    train.fit(cfg, synthetic_inputs(tmp_path), rd)
    assert train.completed_config_mismatch(rd, cfg) == {}
    other = train.RunConfig(**{**cfg.to_dict(), "lr": 3e-4, "device": "cpu"})
    assert set(train.completed_config_mismatch(rd, other)) == {"lr"}
    # a run written before a config field existed counts as having its default
    m = env.read_json(rd / "metrics.json")
    del m["config"]["early_stop"]
    env.write_json(rd / "metrics.json", m)
    assert train.completed_config_mismatch(rd, cfg) == {}
    assert set(train.completed_config_mismatch(
        rd, train.RunConfig(**{**cfg.to_dict(), "early_stop": "val"}))) == {"early_stop"}
