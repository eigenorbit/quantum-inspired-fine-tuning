"""JPEG coverage of the certificate and the eps_train sweep (lossy.py), on synthetic
images and features: no BRONet, no dataset."""

import dataclasses
import json

import numpy as np
import pytest
import torch

from experiments.bronet_mlp import env, lossy, train, tune
from experiments.common import layout
from experiments.common.data import load_images

from conftest import synthetic_inputs


@pytest.fixture
def out(tmp_path, monkeypatch):
    monkeypatch.setenv("DERMA_REPRO_OUT", str(tmp_path / "out"))
    return tmp_path


def jpeg_files(tmp_path, n=3):
    """600x600 JPEG files like the delivered ones (smooth content + texture)."""
    from PIL import Image

    g = np.random.default_rng(0)
    yy, xx = np.mgrid[0:600, 0:600] / 600.0
    paths = []
    for i in range(n):
        img = np.stack([np.sin(6 * xx + i), np.cos(5 * yy - i), xx * yy], -1) * 90 + 128
        img = np.clip(img + g.normal(0, 12, img.shape), 0, 255).astype(np.uint8)
        p = tmp_path / f"P000{i}_1000{i}_aa_act.jpg"
        Image.fromarray(img).save(p, quality=75)
        paths.append(p)
    return paths


def test_inputs_follow_the_network_preprocessing(tmp_path):
    paths = jpeg_files(tmp_path)
    clean = lossy.load_inputs(paths)
    assert clean.shape == (3, 3, 224, 224)
    assert torch.equal(clean, load_images(paths))            # same pixels as the model sees


def _dist(paths, key, at):
    clean = lossy.load_inputs(paths)
    return float((lossy.load_inputs(paths, lossy.OPS[key], at) - clean).flatten(1).norm(dim=1).median())


@pytest.mark.parametrize("at", lossy.PLACES)
def test_every_operation_and_its_severity_order(tmp_path, at):
    paths = jpeg_files(tmp_path)
    d = {k: _dist(paths, k, at) for k in lossy.OPS}
    assert all(v > 0 for v in d.values())
    assert d["jpeg100"] < d["jpeg90"] < d["jpeg80"]
    if at == "input":       # on the file, 8-bit re-storage and the resize can swap them
        assert d["bits7"] < d["bits6"]
    assert d["resize86"] < d["resize50"] and d["resize71"] < d["resize50"]
    assert d["noise0.002"] < d["noise0.005"]
    for k in lossy.OPS:                                       # seeded: identical twice
        x1 = lossy.load_inputs(paths, lossy.OPS[k], at)
        assert torch.equal(x1, lossy.load_inputs(paths, lossy.OPS[k], at))


def test_input_operations_match_the_notebook(tmp_path):
    """--at input reproduces the notebook's pipeline_ops on the network input."""
    import torch.nn.functional as F

    x = lossy.load_inputs(jpeg_files(tmp_path))
    gen = torch.Generator().manual_seed(0)
    levels = 2 ** 6 - 1
    assert torch.equal(lossy.apply_input(x, lossy.OPS["bits6"], gen), (x * levels).round() / levels)
    h = F.interpolate(x, size=(112, 112), mode="bicubic", align_corners=False)
    want = F.interpolate(h, size=(224, 224), mode="bicubic", align_corners=False).clamp(0, 1)
    assert torch.equal(lossy.apply_input(x, lossy.OPS["resize50"], gen), want)
    noisy = lossy.apply_input(x, lossy.OPS["noise0.005"], torch.Generator().manual_seed(1))
    ref = (x + 0.005 * torch.randn(x.shape, generator=torch.Generator().manual_seed(1))).clamp(0, 1)
    assert torch.equal(noisy, ref)
    assert lossy.OPS["resize86"].label(28) == "resize 28->24->28"       # the notebook's names
    assert lossy.OPS["resize71"].label(28) == "resize 28->20->28"
    assert lossy.OPS["jpeg90"].label(224) == "jpeg q=90"


def test_coverage_arithmetic():
    c = lossy.coverage(dist=[0.1, 0.3, 0.1, 0.5], r_off=[0.2, 0.2, 0.2, 0.0],
                       r_rig=[0.05, 0.2, 0.2, 0.0], y=[0, 1, 2, 0], pred=[0, 1, 1, 0],
                       pred_t=[0, 2, 1, 0])
    assert c["correct"].tolist() == [True, True, False, True]
    assert c["covered_official"].tolist() == [True, False, False, False]
    assert c["covered_rigorous"].tolist() == [False, False, False, False]
    assert np.allclose(c["factor"][:2], [0.5, 1.5]) and np.isnan(c["factor"][2])
    assert np.isinf(c["factor"][3])
    assert c["changed"].tolist() == [False, True, False, False]
    assert c["correct_after"].tolist() == [True, False, False, True]


def test_summary_factors():
    rows = [{"dist": d, "radius_official": r, "radius_rigorous": r, "correct": True,
             "covered_official": d < r, "covered_rigorous": d < r, "factor": d / r,
             "changed": False, "correct_after": True}
            for d, r in [(0.1, 0.2), (0.2, 0.2), (0.3, 0.1), (0.4, 0.1)]]
    s = lossy.summarise(rows)
    assert s["covered_official"] == 25.0 and s["changed"] == 0.0
    assert s["factor_0.5"] == 1.0 and s["factor_0.9"] == pytest.approx(4.0)


def write_selection(lrs: dict):
    cfg = train.RunConfig(step=2, mode="butterfly")
    sel = {"fixed": {k: getattr(cfg, k) for k in tune.FIXED_KEYS},
           "modes": {m: {"variants": {"best": {"lr": lr, "epochs": 4}, "stable": None},
                         "stable_equals_best": False} for m, lr in lrs.items()}}
    path = tune.selection_path(2)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sel), encoding="utf-8")


def test_sweep_paths_and_config(out):
    write_selection({"butterfly": 3e-3})
    assert lossy.sweep_run_dir(2, 36, "butterfly", 1) == env.run_dir(2, "butterfly", 1)
    rd = lossy.sweep_run_dir(2, 108, "butterfly", 1)
    assert rd.relative_to(env.repro_out()).as_posix() == "bronet_mlp/eps_sweep/step_2/eps108/butterfly/seed1"
    base = lossy.sweep_config(2, "butterfly", 1, 36)
    big = lossy.sweep_config(2, "butterfly", 1, 108)
    assert base.eps_train == train.RunConfig(step=2, mode="butterfly").eps_train   # the protocol
    assert big.eps_train == pytest.approx(108 / 255)
    assert (big.lr, big.early_stop) == (3e-3, "val")
    assert {k: v for k, v in big.compat().items() if k != "eps_train"} == \
        {k: v for k, v in base.compat().items() if k != "eps_train"}


def es_inputs(tmp_path):
    inp = synthetic_inputs(tmp_path, n_train=448)
    idx = torch.arange(len(inp.train_y))
    val, fit = idx[idx % 8 == 0], idx[idx % 8 != 0]
    return dataclasses.replace(inp, train_x=inp.train_x[fit], train_y=inp.train_y[fit],
                               val_x=inp.train_x[val], val_y=inp.train_y[val])


def test_sweep_runs_coverage_and_report(out, device):
    inputs = es_inputs(out)
    names = [r["image"] for r in inputs.test_rows]
    g = torch.Generator().manual_seed(5)
    noise = torch.randn(inputs.test_x.shape, generator=g)
    jpeg = {k: {"images": names, "label": lossy.OPS[k].label(600),
                "dist": torch.full((len(names),), s, dtype=torch.float64),
                "features": inputs.test_x + s * noise / noise.norm(dim=1, keepdim=True)}
            for k, s in (("jpeg90", 0.05), ("bits6", 0.2))}
    radii = {}
    for eps in (36, 108):
        cfg = train.RunConfig(step=2, mode="butterfly", epochs=4, batch_size=64, lr=3e-2,
                              early_stop="val", device=device)
        if eps != lossy.DEFAULT_EPS_255:
            cfg.eps_train = eps / 255
        rd = lossy.sweep_run_dir(2, eps, "butterfly", 0)
        train.fit(cfg, inputs, rd)
        rows = lossy.run_coverage(rd, "best", jpeg, torch.device(device))
        assert len(rows) == 2 * len(names)
        for r in rows:
            assert r["covered_official"] == (r["correct"] and r["dist"] < r["radius_official"])
        radii[eps] = env.read_json(rd / layout.metrics_file("best"))["certificate_official"]["median_radius"]
    assert radii[108] > radii[36]                       # asking for a larger radius gets one
    rows = lossy.sweep_rows(2, "best", [36, 108], ["butterfly"], ["best"], [0], jpeg,
                            torch.device(device))
    assert [r["eps_255"] for r in rows] == [36.0, 108.0]
    assert all("covered_jpeg90" in r and "changed_bits6" in r for r in rows)
    od = layout.BRONET.lossy_dir(2, "best", "native")
    lossy.write_sweep(od, 2, "best", "native", rows, jpeg)
    md = (od / "eps_sweep.md").read_text(encoding="utf-8")
    assert "| butterfly | 108/255 | 1 |" in md and "| jpeg q=90 | 6-bit depth |" in md


# ---------------------------------------------------------------------------
# renormalisation after the operation (--renorm)
# ---------------------------------------------------------------------------

REF = {"sha256": "test", "crop": 420, "rgb_int": [167, 152, 131], "global_int": 150,
       "rgb": [167.1, 152.9, 131.2]}


@pytest.fixture
def ref(monkeypatch):
    monkeypatch.setattr(lossy, "_REFERENCE", dict(REF))
    return REF


def authors_normalize(bgr, ref_bgr, ref_gl, penalty=(0.8, 0.8, 0.8)):
    """The dataset's colour normalisation, reference implementation (BGR)."""
    mean_rgb = np.mean(bgr, axis=(0, 1)).astype(np.int32)
    gl_mean = np.mean(bgr, axis=(0, 1, 2)).astype(np.int32)
    delta_rgb = np.subtract(ref_bgr, mean_rgb)
    delta_rgb = (delta_rgb * np.array(penalty)).astype(np.int32)
    delta_global = np.subtract(ref_gl, gl_mean)
    out = bgr.astype(np.int32)
    for dim in range(3):
        out[:, :, dim] += delta_rgb[dim] + delta_global
    return np.clip(out, 0, 255).astype(np.uint8)


def test_renorm_authors_is_the_datasets_normalisation(ref):
    from PIL import Image

    g = np.random.default_rng(1)
    for shift in (-40, 0, 25):
        rgb = np.clip(g.normal(140 + shift, 30, (60, 60, 3)), 0, 255).astype(np.uint8)
        got = np.asarray(lossy.renormalise(Image.fromarray(rgb), "authors"))
        want = authors_normalize(rgb[:, :, ::-1], REF["rgb_int"][::-1], REF["global_int"])[:, :, ::-1]
        np.testing.assert_array_equal(got, want)


def test_renorm_ref_and_oracle(ref):
    from PIL import Image

    g = np.random.default_rng(2)
    clean = np.clip(g.normal(140, 20, (50, 50, 3)), 0, 255).astype(np.uint8)
    shifted = np.clip(clean.astype(int) + [9, -4, 6], 0, 255).astype(np.uint8)
    # oracle undoes a uniform colour shift exactly (no clipping here)
    back = lossy.renormalise(Image.fromarray(shifted), "oracle", clean=Image.fromarray(clean))
    np.testing.assert_array_equal(np.asarray(back), clean)
    # ref: channel means at the reference, and a second pass changes nothing
    once = lossy.renormalise(Image.fromarray(shifted), "ref")
    np.testing.assert_allclose(np.asarray(once).mean(axis=(0, 1)), REF["rgb"], atol=0.6)
    twice = lossy.renormalise(once, "ref")
    assert np.abs(np.asarray(twice).astype(int) - np.asarray(once)).max() <= 1
    # the dataset's own normalisation is not idempotent
    a1 = lossy.renormalise(Image.fromarray(shifted), "authors")
    a2 = lossy.renormalise(a1, "authors")
    assert np.abs(np.asarray(a2).astype(int) - np.asarray(a1)).mean() > 1
    with pytest.raises(ValueError):
        lossy.renormalise(Image.fromarray(shifted), "oracle")


def test_renorm_inputs_places_and_cache_meta(tmp_path, ref):
    paths = jpeg_files(tmp_path)
    x = lossy.load_inputs(paths)
    # renormalisation alone, oracle: the image itself
    assert torch.equal(lossy.load_inputs(paths, lossy.IDENTITY, "native", "oracle"), x)
    # without --renorm nothing changes
    assert torch.equal(lossy.load_inputs(paths, lossy.IDENTITY, "native"), x)
    j = lossy.load_inputs(paths, lossy.OPS["jpeg90"], "native")
    assert torch.equal(lossy.load_inputs(paths, lossy.OPS["jpeg90"], "native", "none"), j)
    jo = lossy.load_inputs(paths, lossy.OPS["jpeg90"], "native", "oracle")
    assert (jo - x).flatten(1).norm(dim=1).le((j - x).flatten(1).norm(dim=1) + 1e-3).all()
    with pytest.raises(ValueError):
        lossy.load_inputs(paths, lossy.OPS["jpeg90"], "input", "ref")
    assert lossy.place("native") == "native"
    assert lossy.place("native", "authors") == "native_renorm-authors"
    assert lossy.with_identity(["jpeg90"], "none") == ["jpeg90"]
    assert lossy.with_identity(["jpeg90"], "ref") == ["none", "jpeg90"]


def test_cache_meta_without_renorm_is_unchanged(tmp_path, monkeypatch, ref):
    """Caches made before --renorm existed must stay valid."""
    monkeypatch.setattr(lossy.env, "checkpoint_sha256", lambda: "x")
    monkeypatch.setattr(lossy.bdata, "relative_path", lambda p: str(p))
    paths = jpeg_files(tmp_path, 1)
    m = lossy._cache_meta(2, paths, lossy.OPS["jpeg90"], "native")
    assert set(m) == {"step", "op", "kind", "value", "at", "noise_seed", "images_sha256",
                      "transform", "pillow", "bronet_ckpt_sha256"}
    r = lossy._cache_meta(2, paths, lossy.OPS["jpeg90"], "native", "authors")
    assert r["renorm"] == "authors" and r["reference"] == REF and r["penalty"] == 0.8


def test_renorm_coverage_rows_and_report(out, device, ref):
    inputs = es_inputs(out)
    names = [r["image"] for r in inputs.test_rows]
    cfg = train.RunConfig(step=2, mode="head-only", epochs=2, batch_size=64, lr=3e-2,
                          early_stop="val", device=device)
    rd = env.run_dir(2, "head-only", 0)
    train.fit(cfg, inputs, rd)
    n = len(names)
    blob = {"images": names, "label": "jpeg q=90 + renorm ref",
            "dist": torch.full((n,), 0.01, dtype=torch.float64),
            "dist_attack": torch.full((n,), 1e3, dtype=torch.float64),
            "features": inputs.test_x}
    rows = lossy.run_coverage(rd, "best", {"jpeg90": blob}, torch.device(device))
    assert all("dist_attack" in r and not r["covered_attack_official"] for r in rows)
    s = lossy.summarise(rows)
    assert s["covered_attack_official"] == 0 and s["median_dist_attack"] == 1e3
    od = layout.BRONET.lossy_dir(2, "best", lossy.place("native", "ref"))
    lossy.write_measure(od, 2, "best", "native", {"jpeg90": blob},
                        [{"run": "head-only", "seed": 0, "operation": "jpeg90", **s}], "ref")
    md = (od / "coverage.md").read_text(encoding="utf-8")
    assert "colour-normalised again" in md and "covered without renormalisation" in md
