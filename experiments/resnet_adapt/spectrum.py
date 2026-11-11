"""Rigorous upper bound on the Lipschitz constant of ResNet152, pixels -> features.

    python -m experiments.resnet_adapt.spectrum --step 2 --batch-size 32

The map is x in [0, 1]^(3x224x224) -> the 2048-d pooled features fed to the LLN
head (L2 norms on both sides, as BRONet's L). With eval-mode BN folded into
every convolution and ReLU 1-Lipschitz:

    L <= ||stem|| * 2 * prod_blocks (||shortcut|| + ||conv1|| ||conv2|| ||conv3||) * 1/7

* stem: conv1 with the ImageNet normalisation (x - mean) / std folded into its
  input channels and bn1 into its output channels; the mean shift is a constant
  and drops out;
* 2: max-pool 3x3 / stride 2 / padding 1 -- windows overlap, each input pixel is
  in up to 4 windows, so ||maxpool(x) - maxpool(y)|| <= 2 ||x - y|| (tight);
* a bottleneck: ReLU(shortcut(x) + conv3(ReLU(conv2(ReLU(conv1(x)))))); the
  shortcut is the identity (1) or the folded 1x1 downsample conv;
* 1/7: global average pooling over 7x7 = 49 positions, norm 1/sqrt(49).

Convolution norms (float64, CPU): 1x1 kernels -- sigma_max of the (out, in)
matrix, exact on any input size; k x k kernels -- max over frequencies of the
spectral norm of the kernel's DFT on an (n + k - 1)^2 grid (zero padding = embed
into a circular convolution, then crop; both have norm <= 1), which reduces to
`bronet_mlp.bronet.conv_fft_bound` for 3x3. A stride-s convolution is the
stride-1 one followed by subsampling (norm 1), so its stride-1 bound applies.

Only layer4.2 differs between the adaptation modes; the rest is computed once.
Butterfly keeps every term of layer4.2, hence L, exactly; full / lora change it.

The front part takes ~4-5 min on the CPU (mostly the 36 blocks of layer3) and is
cached in <DERMA_REPRO_OUT>/resnet_adapt/backbone/lipschitz_front.json.

Outputs, <DERMA_REPRO_OUT>/resnet_adapt/results/:
    lipschitz_step<n>[_bs<B>].json / .md   the bound per run (+ the pretrained
    network), per-stage factors, and the certified accuracy it implies
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

from . import paths

import numpy as np
import torch

from experiments.common import certify
from repro.config import NORM_STD
from experiments.common.io import read_json, write_json, atomic_write_text, setup_logging
from experiments.common import layout

from . import backbone

INPUT_SIZE = 224
MAXPOOL_FACTOR = 2.0
AVGPOOL_FACTOR = 1 / 7


@torch.no_grad()
def conv_norm(w: torch.Tensor, n: int) -> float:
    """Upper bound on ||conv2d(., w, stride 1, 'same' zero padding)|| on n x n
    inputs (exact for 1x1)."""
    w = w.detach().double().cpu()
    k = w.shape[-1]
    if w.shape[-2:] == (1, 1):
        return float(torch.linalg.matrix_norm(w[:, :, 0, 0], ord=2))
    if w.shape[-2] != k or k % 2 == 0:
        raise NotImplementedError(f"kernel {tuple(w.shape[-2:])}")
    m = n + k - 1
    sym = torch.fft.rfft2(w, s=(m, m)).permute(2, 3, 0, 1).reshape(-1, *w.shape[:2])
    return max(float(torch.linalg.matrix_norm(c, ord=2).max()) for c in sym.split(512))


def stem_kernel(net) -> torch.Tensor:
    """conv1 with bn1 folded (output channels) and 1/std folded (input channels)."""
    w, _ = backbone.fold_bn(net.conv1, net.bn1)
    return w / torch.tensor(NORM_STD, dtype=w.dtype).view(1, 3, 1, 1)


def _block_sizes():
    """(layer name, block index, input size n, stride) of the 50 bottlenecks."""
    out, n = [], INPUT_SIZE // 4                       # 56 after stem + max-pool
    for name, blocks in (("layer1", 3), ("layer2", 8), ("layer3", 36), ("layer4", 3)):
        for i in range(blocks):
            stride = 2 if (i == 0 and name != "layer1") else 1
            out.append((name, i, n, stride))
            n //= stride
    return out


def block_term(convs: list[torch.Tensor], n: int, shortcut: float) -> dict:
    """||shortcut|| + ||conv1|| ||conv2|| ||conv3|| for one bottleneck whose input is n x n
    (conv2 carries the stride; its stride-1 bound on n x n is used)."""
    norms = [conv_norm(convs[0], n), conv_norm(convs[1], n), conv_norm(convs[2], n)]
    return {"conv_norms": norms, "shortcut": shortcut,
            "branch": math.prod(norms), "term": shortcut + math.prod(norms)}


def front_bound(net, logger=None) -> dict:
    """Everything except layer4.2 (identical for every mode): stem, max-pool,
    the first 49 bottlenecks, average pooling. log10 factors per stage."""
    log = logger.info if logger else print
    t0 = time.perf_counter()
    stem = conv_norm(stem_kernel(net), INPUT_SIZE)
    blocks, stages = [], {}
    for name, i, n, stride in _block_sizes():
        if (name, i) == ("layer4", 2):
            continue
        blk = getattr(net, name)[i]
        convs = [w for _, w in backbone.folded_convs(blk)]
        sc = 1.0
        if blk.downsample is not None:
            sc = conv_norm(backbone.fold_bn(blk.downsample[0], blk.downsample[1])[0], n)
        b = block_term(convs, n, sc)
        b.update(block=f"{name}.{i}", input_size=n, stride=stride)
        blocks.append(b)
        stages[name] = stages.get(name, 0.0) + math.log10(b["term"])
    log10_front = (math.log10(stem) + math.log10(MAXPOOL_FACTOR) + sum(stages.values())
                   + math.log10(AVGPOOL_FACTOR))
    log(f"front bound computed in {time.perf_counter() - t0:.0f}s: stem {stem:.4g}, "
        + ", ".join(f"{k} 10^{v:.2f}" for k, v in stages.items()))
    return {"stem": stem, "maxpool": MAXPOOL_FACTOR, "avgpool": AVGPOOL_FACTOR,
            "log10_stages_without_layer4_2": stages, "blocks": blocks,
            "log10_front": log10_front, "seconds": time.perf_counter() - t0}


FRONT_VERSION = "front-bound/v1"   # bump when front_bound changes


def front_bound_cached(net, logger=None) -> dict:
    """front_bound, cached under <out>/backbone/ by the sha256 of the weights
    (it is the same for every mode, step and seed)."""
    path = paths.out_root() / "backbone" / "lipschitz_front.json"
    sha = backbone.state_sha256(net)
    if path.exists():
        cached = read_json(path)
        if cached.get("weights_sha256") == sha and cached.get("version") == FRONT_VERSION:
            return cached["front"]
    front = front_bound(net, logger)
    write_json(path, {"weights_sha256": sha, "version": FRONT_VERSION, "front": front})
    return front


def layer42_term(block) -> dict:
    """The bound's factor for the (adapted) layer4.2: identity shortcut, 7x7 input."""
    convs = [w for _, w in backbone.folded_convs(block)]
    return block_term(convs, 7, 1.0)


def total_bound(front: dict, block) -> dict:
    t = layer42_term(block)
    log10_L = front["log10_front"] + math.log10(t["term"])
    return {"log10_L": log10_L, "L": 10 ** log10_L if log10_L < 300 else math.inf,
            "layer4_2": t}


def certificate(outputs_npz: Path, log10_L: float) -> dict:
    """Radii and certified accuracy implied by L for one evaluated checkpoint
    (logits and LLN head weights from outputs_<which>.npz)."""
    z = np.load(outputs_npz)
    logits = torch.from_numpy(z["logits"])
    head = torch.from_numpy(z["head_weight"])
    y = torch.from_numpy(z["y"])
    L = 10 ** log10_L
    s = certify.summary(logits, y, head, L)
    return {"median_radius": s["median_radius"], "max_radius": float(
        certify.radii(logits, head, L)["radius"].max()),
        "certified_accuracy_named": s["certified_accuracy_named"]}


def main(argv=None) -> int:
    from repro import console

    from . import report, train

    console.init()
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--step", type=int, choices=(1, 2), required=True)
    p.add_argument("--batch-size", type=int, default=paths.DEFAULT_BATCH_SIZE,
                   help="which protocol's runs (step_<n>_bs<B>/)")
    a = p.parse_args(argv)
    logger = setup_logging(paths.results_dir() / "lipschitz.log", name="resnet_adapt.lip")
    net = backbone.load_resnet152()
    front = front_bound_cached(net, logger)
    base = total_bound(front, net.layer4[2])
    logger.info(f"pretrained ResNet152: L <= 10^{base['log10_L']:.3f}")
    rows = []
    runs = report.load_runs(a.step, "best", a.batch_size)
    for (name, seed), r in sorted(runs.items()):
        model = train.load_run(r["dir"], device=torch.device("cpu"), which="best", net=net)
        tb = total_bound(front, model.block)
        cert = certificate(r["dir"] / "outputs_best.npz", tb["log10_L"])
        rows.append({"run": name, "seed": seed, "log10_L": tb["log10_L"],
                     "L_over_L0": 10 ** (tb["log10_L"] - base["log10_L"]),
                     "layer4_2": tb["layer4_2"], "certificate": cert,
                     "accuracy": r["metrics"]["accuracy"]})
        logger.info(f"{name} seed {seed}: L <= 10^{tb['log10_L']:.3f} "
                    f"(L/L0 = {rows[-1]['L_over_L0']:.6g}), max radius {cert['max_radius']:.3g}, "
                    f"certified @36/255 {100 * cert['certified_accuracy_named']['36/255']:.1f} %")
    tag = paths.tag(a.batch_size)
    write_json(paths.results_dir() / layout.lipschitz_file(a.step, a.batch_size, "json"),
                   {"definition": __doc__.split("\n\n")[1], "front": front,
                    "pretrained": base, "runs": rows})
    lines = [f"# Upper bound on the Lipschitz constant of ResNet152, step-{a.step}{tag}", "",
             "Pixels in [0, 1] (224×224) → 2048-d features before the LLN head, L2. Rigorous "
             "product bound (BN folded, residual blocks as ‖shortcut‖ + ∏‖conv‖, FFT bounds "
             "for k×k convolutions); see spectrum.py.", "",
             f"**Pretrained ResNet152: L ≤ 10^{base['log10_L']:.2f}** "
             f"(stem {front['stem']:.3g}, max-pool ×2, average pool ×1/7; log10 per stage: "
             + ", ".join(f"{k} {v:.2f}" for k, v in front["log10_stages_without_layer4_2"].items())
             + f", layer4.2 {math.log10(base['layer4_2']['term']):.2f}).", "",
             "| run | seed | test acc, % | log10 L | L / L₀ | max radius | certified @ 36/255, % |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        c = r["certificate"]
        lines.append(f"| {report.label(r['run'])} | {r['seed']} | {r['accuracy']:.2f} | "
                     f"{r['log10_L']:.3f} | {r['L_over_L0']:.6g} | {c['max_radius']:.3g} | "
                     f"{100 * c['certified_accuracy_named']['36/255']:.1f} |")
    lines += ["", "Radius r(x) = min_j (z_p − z_j) / (L ‖w_p − w_j‖), as for BRONet; 36/255 ≈ "
              f"{36 / 255:.3f}. For comparison BRONet: L₀ ≈ 8.1 (official) / 8.4 (rigorous)."]
    atomic_write_text(paths.results_dir() / layout.lipschitz_file(a.step, a.batch_size, "md"),
                          "\n".join(lines) + "\n")
    logger.info(f"-> {paths.results_dir() / f'lipschitz_step{a.step}{tag}.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
