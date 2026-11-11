"""Training on synthetic features: determinism,
resumption, checkpoint round trip, atomic writes."""

import numpy as np
import pytest
import torch

from experiments.bronet_mlp import env, train
from experiments.bronet_mlp.adapters import MODES

from conftest import synthetic_inputs


def cfg(mode, device, **kw):
    base = dict(step=2, mode=mode, seed=0, epochs=4, batch_size=64, device=device)
    base.update(kw)
    return train.RunConfig(**base)


def final_fp(run_dir):
    ck = torch.load(run_dir / "checkpoints" / "final.pt", map_location="cpu", weights_only=True)
    return ck["fingerprint"]


# --- 8. two runs, same seed -> same weights ----------------------------------

@pytest.mark.parametrize("mode", MODES)
def test_deterministic(tmp_path, device, mode):
    fps = []
    for k in range(2):
        inputs = synthetic_inputs(tmp_path)
        m = train.fit(cfg(mode, device), inputs, tmp_path / f"run{k}")
        fps.append(m["fingerprint"])
    assert fps[0] == fps[1]
    if mode in ("head-only", "butterfly"):
        assert abs(m["lipschitz"]["L_over_L0"] - 1) < 1e-5


def test_seed_changes_weights(tmp_path, device):
    a = train.fit(cfg("lora", device, seed=0), synthetic_inputs(tmp_path), tmp_path / "a")
    b = train.fit(cfg("lora", device, seed=1), synthetic_inputs(tmp_path), tmp_path / "b")
    assert a["fingerprint"] != b["fingerprint"]


# --- 10. interrupted + resumed == uninterrupted ------------------------------

@pytest.mark.parametrize("mode", ["butterfly", "full"])
def test_resume_same_fingerprint(tmp_path, device, mode):
    straight = train.fit(cfg(mode, device), synthetic_inputs(tmp_path), tmp_path / "straight")
    rd = tmp_path / "resumed"
    assert train.fit(cfg(mode, device), synthetic_inputs(tmp_path), rd, stop_after=2) is None
    assert (rd / "checkpoints" / "last.pt").exists()
    assert not (rd / "checkpoints" / "final.pt").exists()
    resumed = train.fit(cfg(mode, device), synthetic_inputs(tmp_path), rd)
    assert resumed["fingerprint"] == straight["fingerprint"]
    assert len(env.read_json(rd / "history.json")) == 4


# --- 11. checkpoint round trip; incompatible last.pt is refused ---------------

def test_load_run_reproduces_logits(tmp_path, device):
    rd = tmp_path / "run"
    inputs = synthetic_inputs(tmp_path)
    train.fit(cfg("butterfly", device), inputs, rd)
    ck = torch.load(rd / "checkpoints" / "final.pt", map_location="cpu", weights_only=True)
    assert set(ck) >= {"trainable_state", "optimizer", "scheduler", "epoch", "global_step",
                       "rng", "config", "bronet_ckpt_sha256", "features_index_sha256",
                       "data_manifest_sha256", "fingerprint"}
    model = train.load_run(rd, device=torch.device(device))
    with torch.no_grad():
        logits = model(inputs.test_x.to(device)).cpu().numpy()
    saved = np.load(rd / "certificates_final.npz")["logits"]
    np.testing.assert_allclose(logits, saved, atol=1e-5)
    for name in ("config.json", "history.json", "metrics.json", "predictions_final.csv"):
        assert (rd / name).exists()


def test_incompatible_last_is_refused(tmp_path, device):
    rd = tmp_path / "run"
    train.fit(cfg("lora", device), synthetic_inputs(tmp_path), rd, stop_after=1)
    with pytest.raises(RuntimeError, match="not compatible"):
        train.fit(cfg("lora", device, lr=3e-3), synthetic_inputs(tmp_path), rd)
    inputs = synthetic_inputs(tmp_path)
    inputs.provenance["features_index_sha256"] = "other"
    with pytest.raises(RuntimeError, match="features_index_sha256"):
        train.fit(cfg("lora", device), inputs, rd)
    # runtime-only settings do not block a resume, and --restart discards
    train.fit(cfg("lora", device, keep_every=1), synthetic_inputs(tmp_path), rd)
    train.fit(cfg("lora", device, lr=3e-3), synthetic_inputs(tmp_path), rd, restart=True)


# --- 12. a crash while writing does not corrupt the previous checkpoint ------

def test_atomic_save(tmp_path, monkeypatch):
    path = tmp_path / "last.pt"
    env.atomic_torch_save({"epoch": 1, "w": torch.ones(3)}, path)

    real_save = torch.save

    def crashing_save(obj, f, *a, **k):
        with open(f, "wb") as handle:
            handle.write(b"partial garbage")
        raise OSError("disk full (simulated)")

    monkeypatch.setattr(torch, "save", crashing_save)
    with pytest.raises(OSError):
        env.atomic_torch_save({"epoch": 2, "w": torch.zeros(3)}, path)
    monkeypatch.setattr(torch, "save", real_save)
    ck = torch.load(path, weights_only=True)
    assert ck["epoch"] == 1 and torch.equal(ck["w"], torch.ones(3))
    assert not path.with_name(path.name + ".tmp").exists()
