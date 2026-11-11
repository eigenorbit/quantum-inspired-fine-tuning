"""Paired McNemar on certified accuracy: counts and majority vote on hand-made runs."""

import csv

import numpy as np
from scipy.stats import binomtest

from experiments.bronet_mlp import report

EPS36 = 36 / 255


def fake_run(d, correct, radius):
    d.mkdir(parents=True)
    n = len(correct)
    with open(d / "predictions_best.csv", "w", newline="", encoding="utf-8") as h:
        w = csv.writer(h)
        w.writerow(["image"])
        w.writerows([[f"img{i:03d}.jpg"] for i in range(n)])
    y = np.zeros(n, dtype=int)
    pred = np.where(np.asarray(correct), 0, 1)
    r = np.asarray(radius, dtype=float)
    np.savez(d / "certificates_best.npz", y=y, pred=pred, radius_official=r,
             radius_rigorous=r * 0.96)
    return {"dir": d}


def test_certified_mcnemar_counts(tmp_path):
    n = 40
    # A: correct everywhere, radius 0.3 on the first 30 images, 0.1 on the rest
    a_rad = [0.3] * 30 + [0.1] * 10
    # B: correct on the first 35, radius 0.3 on images 0..9 and 30..34 only
    b_correct = [True] * 35 + [False] * 5
    b_rad = [0.3] * 10 + [0.1] * 20 + [0.3] * 5 + [0.3] * 5    # last 5 wrong: not certified
    runs = {("butterfly", 0): fake_run(tmp_path / "a", [True] * n, a_rad),
            ("lora", 0): fake_run(tmp_path / "b", b_correct, b_rad)}
    rows = report.certified_mcnemar(runs, "best")
    r = next(x for x in rows if x["variant"] == "seed 0" and x["L"] == "official"
             and x["epsilon_name"] == "36/255")
    # A certified: 0..29 (30). B certified: 0..9 and 30..34 (15).
    # only A: 10..29 (20); only B: 30..34 (5)
    assert (r["only_a_certified"], r["only_b_certified"]) == (20, 5)
    assert r["cert_a"] == 75.0 and r["cert_b"] == 37.5
    assert np.isclose(r["p_value"], binomtest(5, 25, 0.5).pvalue)
    # rigorous L shrinks every radius; at 108/255 = 0.4235 nobody is certified
    r108 = next(x for x in rows if x["variant"] == "seed 0" and x["L"] == "rigorous"
                and x["epsilon_name"] == "108/255")
    assert (r108["only_a_certified"], r108["only_b_certified"], r108["p_value"]) == (0, 0, 1.0)


def test_majority_vote_needs_more_than_half_of_the_seeds(tmp_path):
    n = 4
    # image i certified in exactly i of the 3 seeds (i = 0..3)
    radii = [[0.3 if i > s else 0.0 for i in range(n)] for s in range(3)]
    runs = {("butterfly", s): fake_run(tmp_path / f"b{s}", [True] * n, radii[s])
            for s in range(3)}
    runs[("head-only", 0)] = fake_run(tmp_path / "h", [True] * n, [0.0] * n)
    rows = report.certified_mcnemar(runs, "best")
    r = next(x for x in rows if x["variant"] == "majority vote" and x["L"] == "official"
             and x["epsilon_name"] == "36/255")
    assert r["a"] == "head-only" or r["b"] == "head-only"
    butterfly_cert = r["cert_a"] if r["a"] == "butterfly" else r["cert_b"]
    assert butterfly_cert == 50.0            # images certified in 2 and 3 seeds


def test_holm_known_values():
    # sorted: .005*4=.02, .01*3=.03, .03*2=.06, .04*1=.04 -> monotone .06
    assert np.allclose(report.holm([0.01, 0.04, 0.03, 0.005]), [0.03, 0.06, 0.06, 0.02])
    assert report.holm([0.9, 0.8]) == [1.0, 1.0]            # capped at 1
    assert report.holm([]) == []


def test_add_holm_families_and_primary():
    rows = [{"variant": v, "a": a, "b": b, "p_value": p}
            for v in ("seed 0", "majority vote")
            for a, b, p in (("butterfly", "lora", 0.01), ("butterfly", "full", 0.02),
                            ("lora", "full", 0.03))]
    report.add_holm(rows, ("variant",))
    for v in ("seed 0", "majority vote"):
        r = {(x["a"], x["b"]): x for x in rows if x["variant"] == v}
        # primary family = the two butterfly pairs: .01*2=.02, .02*1=.02
        assert np.isclose(r[("butterfly", "lora")]["p_holm_primary"], 0.02)
        assert np.isclose(r[("butterfly", "full")]["p_holm_primary"], 0.02)
        assert r[("lora", "full")]["p_holm_primary"] == ""
        # all pairs: .01*3=.03, .02*2=.04, .03*1 -> .04
        assert np.allclose([r[k]["p_holm_all"] for k in
                            (("butterfly", "lora"), ("butterfly", "full"), ("lora", "full"))],
                           [0.03, 0.04, 0.04])


def test_certified_rows_carry_holm(tmp_path):
    runs = {("butterfly", 0): fake_run(tmp_path / "a", [True] * 20, [0.3] * 20),
            ("lora", 0): fake_run(tmp_path / "b", [True] * 20, [0.0] * 20)}
    rows = report.certified_mcnemar(runs, "best")
    fam = [r for r in rows if r["variant"] == "seed 0" and r["L"] == "official"]
    assert len(fam) == 3                                     # three eps, one family
    raw = [r["p_value"] for r in fam]
    assert np.allclose([r["p_holm_primary"] for r in fam], report.holm(raw))
