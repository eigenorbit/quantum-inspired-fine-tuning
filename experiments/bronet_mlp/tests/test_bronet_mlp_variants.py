"""The two tuning selections per mode (best / stable) and their runs in the report."""

import pytest

from experiments.bronet_mlp import env, report, train, tune

from conftest import synthetic_inputs


def fake_tuning_run(step, mode, lr, accs, majority=38.7, ceiling=None):
    """A finished tuning run with a given per-epoch validation accuracy curve."""
    rd = tune.run_dir(step, mode, lr)
    (rd / "checkpoints").mkdir(parents=True, exist_ok=True)
    (rd / "checkpoints" / "final.pt").write_bytes(b"")
    hist = [{"epoch": e, "test_accuracy": a, "test_f1_weighted": a, "L_mlp_end_power": 1.0 + e,
             "test_certified_accuracy_36": 0.2} for e, a in enumerate(accs)]
    env.write_json(rd / "history.json", hist)
    env.write_json(rd / "metrics.json", {
        "config": {"epochs": ceiling or len(accs)}, "majority_class_rate": majority,
        "accuracy": accs[-1], "f1_weighted": accs[-1], "collapsed": accs[-1] <= majority,
        "lipschitz": {"L_over_L0": 1.0}})


@pytest.fixture
def out(tmp_path, monkeypatch):
    monkeypatch.setenv("DERMA_REPRO_OUT", str(tmp_path / "out"))
    return tmp_path


def test_best_and_stable_selection(out, monkeypatch):
    monkeypatch.setitem(tune.LR_GRID, "lora", (1e-4, 3e-4))
    monkeypatch.setitem(tune.LR_GRID, "butterfly", (1e-3,))
    fake_tuning_run(2, "lora", 1e-4, [50, 60, 69.9, 70.0])        # stable, peak 70.0 @4
    fake_tuning_run(2, "lora", 3e-4, [60, 70.8, 55, 38.7])        # peak 70.8 @2, collapses
    fake_tuning_run(2, "butterfly", 1e-3, [60, 72, 73.5, 73.4])   # stable
    got = tune.collect(2, ["lora", "butterfly"], tune.LR_GRID, max_epochs=4)
    lora = got["lora"]
    assert lora["variants"]["best"]["lr"] == 3e-4 and lora["variants"]["best"]["epochs"] == 2
    assert lora["variants"]["stable"]["lr"] == 1e-4 and lora["variants"]["stable"]["epochs"] == 4
    assert not lora["stable_equals_best"]
    assert [c["stable_run"] for c in lora["candidates"]] == [True, False]
    assert got["butterfly"]["stable_equals_best"]

    split = {"sha256": "s", "fit": ["a"] * 9, "val": ["b"]}
    fixed = {k: getattr(train.RunConfig(step=2, mode="lora", epochs=4), k)
             for k in tune.FIXED_KEYS}
    tune.write_selection(2, got, fixed, split)
    cfg = train.RunConfig(step=2, mode="lora", epochs=30)
    assert tune.tuned_params(2, "lora", cfg) == {"lr": 3e-4, "epochs": 2, "ceiling": 4}
    assert tune.tuned_params(2, "lora", cfg, "stable") == {"lr": 1e-4, "epochs": 4, "ceiling": 4}
    assert tune.stable_equals_best(2, "butterfly") and not tune.stable_equals_best(2, "lora")
    table = (env.tuning_dir(2) / "tuning_step2.md").read_text(encoding="utf-8")
    assert "**best**" in table and "**stable**" in table and "**best = stable**" in table


def test_drop_just_over_threshold_is_unstable(out, monkeypatch):
    monkeypatch.setitem(tune.LR_GRID, "full", (1e-5,))
    fake_tuning_run(2, "full", 1e-5, [70, 74.0, 71.9])            # 2.1 below its peak
    got = tune.collect(2, ["full"], tune.LR_GRID, max_epochs=3)
    assert got["full"]["variants"]["stable"] is None


def test_run_names(out):
    assert env.run_name("lora") == "lora" and env.run_name("lora", "stable") == "lora_stable"
    assert env.run_dir(2, "full", 1, "stable").parent.name == "full_stable"


def test_report_shows_both_variants(out, device, monkeypatch):
    monkeypatch.setenv("DERMA_BASELINE_ROOTS", str(out / "no_baselines"))
    for variant, lr in (("best", 3e-3), ("stable", 1e-3)):
        cfg = train.RunConfig(step=2, mode="lora", lr=lr, epochs=1, batch_size=64, device=device)
        train.fit(cfg, synthetic_inputs(out), env.run_dir(2, "lora", 0, variant))
    runs = report.load_runs(2, "final")
    assert set(runs) == {("lora", 0), ("lora_stable", 0)}
    assert report.load_runs(2, "best") == {}        # not early-stopped: no best.pt
    md, mc = report.summary(2, runs, "final")
    assert "| lora | lora (stable) |" in md
    assert any({r["a"], r["b"]} == {"lora", "lora (stable)"} for r in mc)
    rows = report.adaptation_rows(runs)
    assert {(r["mode"], r["run"]) for r in rows} == {("lora", "lora"), ("lora", "lora_stable")}


def test_report_skips_runs_that_no_longer_match_the_selection(out, monkeypatch):
    monkeypatch.setitem(tune.LR_GRID, "butterfly", (1e-3, 3e-3))
    fake_tuning_run(2, "butterfly", 1e-3, [60, 70, 70.5])
    fake_tuning_run(2, "butterfly", 3e-3, [60, 72, 72.0])          # best and stable
    got = tune.collect(2, ["butterfly"], tune.LR_GRID, max_epochs=3)
    fixed = {k: getattr(train.RunConfig(step=2, mode="butterfly", epochs=3), k)
             for k in tune.FIXED_KEYS}
    tune.write_selection(2, got, fixed, {"sha256": "s", "fit": ["a"], "val": ["b"]})

    def run_metrics(lr):
        return {"config": {"mode": "butterfly", "lr": lr, "early_stop": "val"}}

    assert report.stale_reason(2, "butterfly", run_metrics(3e-3)) is None
    assert "selection is now 0.003" in report.stale_reason(2, "butterfly", run_metrics(1e-3))
    assert "equals the best" in report.stale_reason(2, "butterfly_stable", run_metrics(1e-3))
    # runs outside the tuned protocol are never second-guessed
    assert report.stale_reason(2, "butterfly", {"config": {"mode": "butterfly", "lr": 1.0}}) is None
