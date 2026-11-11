"""Training (determinism, resumption, early stopping), the feature cache, the
validation split shared with BRONet, and the report's McNemar."""

import numpy as np
import pytest
import torch
from PIL import Image

from conftest import needs_data
from resnet_adapt_helpers import synthetic_inputs
from experiments.bronet_mlp import env
from experiments.resnet_adapt import backbone, features, report, train, tune


def _cfg(**kw):
    base = dict(step=2, epochs=2, batch_size=32, lr=1e-3, early_stop="val", device="cpu")
    base.update(kw)
    return train.RunConfig(**base)


@pytest.fixture(scope="module")
def inputs(net):
    inp = synthetic_inputs(net.layer4[2], n_train=64, n_val=32, n_test=32)
    inp.provenance["weights_sha256"] = backbone.state_sha256(net)
    return inp


def _fps(run_dir):
    m = env.read_json(run_dir / "metrics.json")
    return m["fingerprint"], m["other_checkpoints"]["final"]["fingerprint"]


def test_same_seed_same_fingerprint_and_resume_equals_straight(tmp_path, inputs):
    cfg = _cfg()
    train.fit(cfg, inputs, tmp_path / "a")
    train.fit(cfg, inputs, tmp_path / "b")
    assert _fps(tmp_path / "a") == _fps(tmp_path / "b")
    assert train.fit(cfg, inputs, tmp_path / "c", stop_after=1) is None
    assert not train.is_complete(tmp_path / "c")
    train.fit(cfg, inputs, tmp_path / "c")                      # resumes at epoch 2
    assert _fps(tmp_path / "c") == _fps(tmp_path / "a")
    hist = env.read_json(tmp_path / "c" / "history.json")
    assert [h["epoch"] for h in hist] == [0, 1]
    m = env.read_json(tmp_path / "a" / "metrics.json")
    assert m["headline"] == "best" and m["seconds"]["total"] >= m["seconds"]["train"] > 0
    assert (tmp_path / "a" / "predictions_best.csv").exists()
    assert (tmp_path / "a" / "predictions_final.csv").exists()


def test_incompatible_resume_refused_and_load_run_round_trip(tmp_path, inputs, net):
    cfg = _cfg()
    train.fit(cfg, inputs, tmp_path / "r", stop_after=1)
    with pytest.raises(RuntimeError, match="not compatible"):
        train.fit(_cfg(lr=3e-4), inputs, tmp_path / "r")
    train.fit(cfg, inputs, tmp_path / "r")
    model = train.load_run(tmp_path / "r", device="cpu", which="best", net=net)
    z = train.predict(model, inputs.feats, inputs.test_idx, "cpu").numpy()
    saved = np.load(tmp_path / "r" / "outputs_best.npz")["logits"]
    assert np.allclose(z, saved, atol=1e-5)


def _images(root, n=5):
    d = root / "model_data_step2_anon" / "train" / "0"
    d.mkdir(parents=True)
    rng = np.random.default_rng(0)
    out = []
    for i in range(n):
        p = d / f"P0001_{i:05d}_aa_act.jpg"
        Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)).save(p)
        out.append(p)
    return out


@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_cache_equals_end_to_end_and_resumes(tmp_path, monkeypatch, net, dtype):
    monkeypatch.setenv("DERMA_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("DERMA_REPRO_OUT", str(tmp_path / "out"))
    imgs = _images(tmp_path / "data")
    kw = dict(device=torch.device("cpu"), net=net, dtype=dtype, batch_size=2, num_workers=0,
              flush_rows=1)
    features.extract(2, imgs, limit=3, **kw)                   # interrupted after 3 rows
    with pytest.raises(FileNotFoundError):
        features.open_store(2, dtype, backbone.state_sha256(net))
    features.extract(2, imgs, **kw)                            # resumes at row 3 (of 4..5)
    store = features.open_store(2, dtype, backbone.state_sha256(net))
    from experiments.bronet_mlp import data as bdata
    with torch.no_grad():
        fresh = backbone.Front(net)(bdata.load_images(imgs)).numpy()
    cached = np.asarray(store.array).astype(np.float32)
    tol = features.TOLERANCE[dtype] * max(1.0, np.abs(fresh).max())
    assert np.abs(cached - fresh).max() <= tol
    assert store.rows_of([bdata.relative_path(imgs[4])]).tolist() == [4]
    with pytest.raises(RuntimeError, match="different settings"):
        features.open_store(2, dtype, "other-weights")
    Image.fromarray(np.zeros((80, 80, 3), dtype=np.uint8)).save(imgs[1])   # file changed
    with pytest.raises(RuntimeError, match="changed on disk"):
        features.open_store(2, dtype, backbone.state_sha256(net))


@needs_data
def test_validation_split_is_bronets(tmp_path, monkeypatch):
    from experiments.bronet_mlp import data as bdata
    from experiments.bronet_mlp import tune as btune

    if not btune.split_path(2).exists():
        pytest.skip("BRONet val_split.json not found")
    train_r, test_r, stats = bdata.records(2)
    assert (len(train_r), len(test_r)) == (13_707, 240)
    man = bdata.manifest(2, train_r, test_r, stats)
    monkeypatch.setattr(tune.paths, "tuning_dir", lambda step, batch_size=256: tmp_path)
    fake = train.RunInputs(feats=None, train_idx=None, train_y=None, test_idx=None,
                           test_y=None, base_block=None, provenance={}, test_rows=[],
                           manifest=man)
    split = tune.make_split(fake, 2)
    assert split["sha256"] == env.read_json(btune.split_path(2))["sha256"]
    assert env.read_json(tmp_path / "val_split.json")["sha256"] == split["sha256"]
    assert len(split["val_idx"]) + len(split["fit_idx"]) == 13_707


def _write_preds(d, y, pred, names):
    d.mkdir(parents=True)
    logits = np.eye(3)[pred].astype(np.float64)
    train.write_predictions(d / "predictions_best.csv",
                            [{"image": n, "patient": "P0", "source_stem": n} for n in names],
                            {"logits": logits, "y": y, "pred": pred}, 2, 4.0)


def test_mcnemar_same_images_and_holm(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    names = [f"img{i}.jpg" for i in range(60)]
    y = rng.integers(0, 3, 60)
    runs = {}
    for s in (0, 1, 2):
        pred = np.where(rng.random(60) < 0.8, y, (y + 1) % 3)
        _write_preds(tmp_path / f"s{s}", y, pred, names)
        runs[("full", s)] = {"dir": tmp_path / f"s{s}", "metrics": {}}
    base_pred = np.where(rng.random(60) < 0.6, y, (y + 2) % 3)
    _write_preds(tmp_path / "base", y, base_pred, names)
    base = tmp_path / "base" / "predictions_best.csv"
    monkeypatch.setattr(report.breport, "find_baseline", lambda step, arch: base)
    rows, info = report.mcnemar_rows(2, runs, "best", {})
    assert {(r["variant"], r["a"], r["b"]) for r in rows} == {
        ("seed 0", "full", report.FULL_FT_NAME),
        ("majority vote", "full", report.FULL_FT_NAME)}
    assert all(r["p_holm"] >= r["p_value"] for r in rows)
    # a baseline on other images is refused
    _write_preds(tmp_path / "bad", y, base_pred, [f"x{i}.jpg" for i in range(60)])
    monkeypatch.setattr(report.breport, "find_baseline",
                        lambda step, arch: tmp_path / "bad" / "predictions_best.csv")
    with pytest.raises(RuntimeError, match="differ"):
        report.mcnemar_rows(2, runs, "best", {})


def test_batch_sizes_have_separate_folders(tmp_path, monkeypatch):
    from experiments.resnet_adapt import paths

    monkeypatch.setenv("DERMA_REPRO_OUT", str(tmp_path))
    assert paths.tuning_dir(2).name == "step_2" and paths.runs_dir(2).name == "step_2"
    assert paths.tuning_dir(2, 32).name == "step_2_bs32"
    assert paths.run_dir(1, "full", 0, "stable", 32) == \
        tmp_path / "resnet_adapt" / "runs" / "step_1_bs32" / "full_stable" / "seed0"
    assert tune.selection_path(2, 32).parent == paths.tuning_dir(2, 32)
    with pytest.raises(FileNotFoundError, match="--batch-size 32"):
        tune.tuned_params(2, "full", train.RunConfig(step=2, batch_size=32))
