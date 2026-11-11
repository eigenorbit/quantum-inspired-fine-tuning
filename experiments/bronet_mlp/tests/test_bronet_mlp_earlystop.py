"""Final protocol B: train on fit, early-stop on validation, evaluate best AND last on test."""

import dataclasses

import numpy as np
import pytest
import torch

from experiments.bronet_mlp import env, report, train

from conftest import synthetic_inputs


def es_inputs(tmp_path):
    """Synthetic inputs with a held-out validation part (every 8th train row)."""
    inp = synthetic_inputs(tmp_path, n_train=448)
    idx = torch.arange(len(inp.train_y))
    val, fit = idx[idx % 8 == 0], idx[idx % 8 != 0]
    return dataclasses.replace(inp, train_x=inp.train_x[fit], train_y=inp.train_y[fit],
                               val_x=inp.train_x[val], val_y=inp.train_y[val])


def es_cfg(device, mode="lora", **kw):
    base = dict(step=2, mode=mode, seed=0, epochs=6, batch_size=64, lr=3e-2, device=device,
                early_stop="val")
    base.update(kw)
    return train.RunConfig(**base)


def test_best_epoch_is_the_validation_optimum(tmp_path, device):
    rd = tmp_path / "run"
    m = train.fit(es_cfg(device), es_inputs(tmp_path), rd)
    hist = env.read_json(rd / "history.json")
    keys = [(int(h["val_collapsed"]), -h["val_accuracy"], -h["val_f1_weighted"], h["epoch"])
            for h in hist]
    best_epoch = min(keys)[3]
    assert m["headline"] == "best" and m["epoch"] == best_epoch
    assert m["early_stopping"]["epoch"] == best_epoch
    for which in ("best", "final"):
        for name in (f"metrics_{which}.json", f"predictions_{which}.csv",
                     f"certificates_{which}.npz"):
            assert (rd / name).exists(), name
        model = train.load_run(rd, device=torch.device(device), which=which)
        with torch.no_grad():
            logits = model(es_inputs(tmp_path).test_x.to(device)).cpu().numpy()
        np.testing.assert_allclose(logits, np.load(rd / f"certificates_{which}.npz")["logits"],
                                   atol=1e-5)
    final = env.read_json(rd / "metrics_final.json")
    assert final["epoch"] == 5
    assert m["other_checkpoints"]["final"]["fingerprint"] == final["fingerprint"]
    # the test set never picks the epoch: best test epoch may differ, kept epoch may not
    assert all("val_accuracy" in h for h in hist)


def test_resume_keeps_best_tracking(tmp_path, device):
    straight = train.fit(es_cfg(device), es_inputs(tmp_path), tmp_path / "a")
    rd = tmp_path / "b"
    assert train.fit(es_cfg(device), es_inputs(tmp_path), rd, stop_after=3) is None
    resumed = train.fit(es_cfg(device), es_inputs(tmp_path), rd)
    assert resumed["fingerprint"] == straight["fingerprint"]          # best.pt
    assert resumed["other_checkpoints"]["final"]["fingerprint"] == \
        straight["other_checkpoints"]["final"]["fingerprint"]
    assert resumed["epoch"] == straight["epoch"]


def test_early_stop_needs_validation(tmp_path, device):
    with pytest.raises(ValueError, match="val_x"):
        train.fit(es_cfg(device), synthetic_inputs(tmp_path), tmp_path / "x")


def test_report_writes_best_and_last(tmp_path, device, monkeypatch):
    monkeypatch.setenv("DERMA_REPRO_OUT", str(tmp_path / "out"))
    monkeypatch.setenv("DERMA_BASELINE_ROOTS", str(tmp_path / "none"))
    for mode in ("head-only", "lora"):
        train.fit(es_cfg(device, mode=mode), es_inputs(tmp_path), env.run_dir(2, mode, 0))
    assert report.main(["--step", "2"]) == 0
    res = env.results_dir()
    best_md = (res / "summary_step2.md").read_text(encoding="utf-8")
    last_md = (res / "summary_step2_last.md").read_text(encoding="utf-8")
    assert "best validation epoch" in best_md and "last epoch" in last_md
    assert "epoch kept" in best_md
    csv_text = (res / "adaptation_step2.csv").read_text(encoding="utf-8")
    assert "\nbest," in csv_text and "\nfinal," in csv_text
