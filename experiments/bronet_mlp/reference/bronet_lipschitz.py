"""Lipschitz constant of the released BRONet ImageNet checkpoint.

Computes the bound two ways and cross-checks them:

  official   BRONet's own ``model.sub_lipschitz()`` with 500 power iterations,
             exactly as ``bronet/test.py`` does before certifying.
  audit      an independent, block-by-block recomputation:
               stem     exact SVD (14x14 stride-14 patches = a 588x588 matrix)
               conv_1   6 LiResConv blocks: power iteration (lower estimate) AND a
                        rigorous FFT upper bound (see ``conv_fft_bound``)
               conv     8 BRO blocks: all singular values of every frequency
                        block, plus power iteration on the real operator
               neck     exact SVD of the 9408->2048 linear map (L2 pooling is 1-Lip)
               linear   exact spectral norm of the 8 dense 2048x2048 layers
  empirical  a LOWER bound on L: power iteration on the input Jacobian of the
             sub-network at a few inputs. Must come out <= the bound.

The head uses LLN (unit-norm rows) and is handled by the certificate formula,
not by L:  class j is certified at radius eps iff
    y_pred - y_j > L * eps * ||w_pred - w_j||.

Run:  python pretrained/bronet_lipschitz.py [--quick]
      (--quick = random weights, few iterations: a smoke test of the code path)
"""
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "BRONet" / "bronet"))
# The dense MLP asks torch.distributed for the world size at construction time;
# on a single GPU the answer is 1 (this only disables its multi-GPU code path).
torch.distributed.get_world_size = lambda *a, **k: 1
import models  # noqa: E402

CKPT_DIR = HERE / "bronet_imagenet"
CKPT = CKPT_DIR / "imagenet-14x588_399.pth"
OUT = CKPT_DIR / "lipschitz.json"
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def build():
    cfg = yaml.safe_load(open(CKPT_DIR / "config.yaml"))
    return models.BRONet(**cfg["model"], **cfg["dataset"])


def load():
    model = build()
    # weights_only: never execute pickled code from a downloaded file.
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=True)
    sd = ckpt["backbone"]
    own = model.state_dict()
    info = {"ckpt_keys": sorted(ckpt.keys()),
            "unexpected": [k for k in sd if k not in own],
            "missing": [k for k in own if k not in sd]}
    model.load_state_dict({k: v for k, v in sd.items() if k in own})
    return to_device(model), info


def load_random():
    return to_device(build()), {"random_init": True}


def to_device(model):
    """eval() BEFORE moving: OrthoConvV2.train(False) materialises every BRO
    block's per-frequency matrices (2784 complex 588x588 = ~7.7 GB), which does
    not fit an 8 GB GPU. They are a plain list attribute, so .to() leaves them on
    the CPU; the code below moves one layer at a time."""
    model.eval()
    model.to(dev)
    return model


@torch.no_grad()
def sigma_range(mats, chunk=64):
    """(min, max) singular value over a batch of matrices, in float64 on the CPU
    via the eigenvalues of W^H W. (float32 SVD is off by ~1e-4 here, which is too
    coarse to tell an orthogonal matrix from a nearly orthogonal one, and consumer
    GPUs run float64 ~64x slower than the CPU's LAPACK.)"""
    lo, hi = math.inf, 0.0
    for c in mats.split(chunk):
        c = c.cpu().to(torch.complex128 if c.is_complex() else torch.float64)
        ev = torch.linalg.eigvalsh(c.mH @ c)
        lo, hi = min(lo, float(ev.min())), max(hi, float(ev.max()))
    return math.sqrt(max(lo, 0.0)), math.sqrt(hi)


@torch.no_grad()
def conv_fft_bound(weight, n):
    """Rigorous upper bound on ||conv2d(., weight, padding=1)|| on n x n inputs.

    A zero-padded 3x3 convolution on n x n equals: embed into (n+2)x(n+2) with a
    zero border, circular convolution on (n+2)x(n+2), crop the centre. Embedding
    and cropping have norm <= 1 and no wrap-around ever reaches the kept outputs,
    so the norm is at most that of the circular convolution, which is exactly the
    largest singular value of its symbol over the (n+2)^2 DFT grid.
    """
    m = n + 2
    # rfft2 suffices: the symbol at -w is the conjugate of the symbol at w.
    sym = torch.fft.rfft2(weight.double().cpu(), s=(m, m))  # (out, in, m, m//2+1)
    sym = sym.permute(2, 3, 0, 1).reshape(-1, *weight.shape[:2])
    return sigma_range(sym)[1]


@torch.no_grad()
def power_iter(op, opT, shape, iters, seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    x = torch.randn(shape, generator=g, device=dev)
    x /= x.norm()
    for _ in range(iters):
        x = opT(op(x))
        x /= x.norm()
    return float(op(x).norm())


def audit(model, iters):
    r = {}

    # stem: kernel 14, stride 14, no padding on 224 -> non-overlapping patches,
    # i.e. one 588 x (3*14*14 = 588) matrix applied to every patch.
    conv = model.stem[0]
    w = conv.weight.detach()
    lo, hi = sigma_range(w.reshape(1, w.shape[0], -1))
    r["stem"] = {"exact": hi, "sigma_min": lo,
                 "kernel": list(w.shape), "stride": list(conv.stride),
                 "padding": list(conv.padding)}
    log(f"stem exact {r['stem']['exact']:.6f}  (sigma_min {r['stem']['sigma_min']:.4f})")

    # conv_1: 6 LiResConv blocks, each x -> MinMax(conv3x3(x) + b) on 16x16.
    c1 = model.conv_1
    n = c1.init_x.shape[-1]
    per = []
    for k, W in enumerate(c1.get_weight().detach()):
        op = lambda x, W=W: F.conv2d(x, W, padding=1)
        opT = lambda y, W=W: F.conv_transpose2d(y, W, padding=1)
        pi = power_iter(op, opT, (1, c1.width, n, n), iters)
        ub = conv_fft_bound(W, n)
        per.append({"power_iter": pi, "fft_upper": ub})
        log(f"conv_1[{k}] power_iter {pi:.6f}  fft_upper {ub:.6f}")
    r["conv_1"] = {"per_block": per, "input_size": n,
                   "power_iter": math.prod(p["power_iter"] for p in per),
                   "fft_upper": math.prod(p["fft_upper"] for p in per)}

    # conv: 8 BRO blocks. eval_weights are the per-frequency complex matrices;
    # for an orthogonal operator every singular value of every block is 1.
    c = model.conv
    blocks = []
    for i, Wf in enumerate(c.eval_weights):
        lo, hi = sigma_range(Wf)
        n_in = c.padding_input_sizes[i] - 2
        Wd = Wf.to(dev)
        op = lambda x, i=i, Wd=Wd: c.frequency_convolution(x, Wd, i)

        def opT(y, op=op, n_in=n_in):
            x = torch.zeros(y.shape[0], c.width, n_in, n_in, device=dev, requires_grad=True)
            with torch.enable_grad():
                (g,) = torch.autograd.grad(op(x), x, y)
            return g

        pi = power_iter(op, opT, (1, c.width, n_in, n_in), min(iters, 300))
        del Wd
        blocks.append({"sigma_max": hi, "sigma_min": lo,
                       "power_iter": pi, "input_size": n_in, "n_freq": Wf.shape[0]})
        log(f"conv[{i}] BRO sigma in [{lo:.9f}, {hi:.9f}]  power_iter {pi:.6f}")
    r["conv"] = {"per_block": blocks,
                 "product": math.prod(b["sigma_max"] for b in blocks),
                 "sigma_max": max(b["sigma_max"] for b in blocks),
                 "sigma_min": min(b["sigma_min"] for b in blocks),
                 "claimed": float(c.lipschitz())}

    # neck: LPPool2d(p=2, 8x8, stride 8) is 1-Lipschitz; then one linear map.
    kernel, kernel_2, W = model.neck.get_weight()
    assert kernel is None and kernel_2 is None  # neck_conv_type = l2
    lo, hi = sigma_range(W.detach().T.unsqueeze(0))       # 9408x2048: small Gram side
    r["neck"] = {"exact": hi, "sigma_min": lo, "shape": list(W.shape)}
    log(f"neck exact {r['neck']['exact']:.6f}  (sigma_min {r['neck']['sigma_min']:.4f})")

    # linear: 8 dense layers, each x -> MinMax(Wx + b).
    rng = [sigma_range(W.unsqueeze(0)) for W in model.linear.get_weight().detach()]
    r["linear"] = {"per_layer": [h for _, h in rng],
                   "sigma_min": min(l for l, _ in rng),
                   "exact": math.prod(h for _, h in rng)}
    log(f"linear exact {r['linear']['exact']:.6f}  (sigma_min {r['linear']['sigma_min']:.4f})")

    # head (LLN): rows normalised to unit norm; enters the certificate as
    # ||w_i - w_j|| <= 2, not through L.
    hw = model.head.get_weight().detach().double()
    r["head"] = {"row_norm_min": float(hw.norm(dim=1).min()),
                 "row_norm_max": float(hw.norm(dim=1).max()),
                 "max_pair_dist": float(torch.cdist(hw, hw).max())}
    return r


def features(model, x):
    """The sub-network (everything but the head), re-implemented so gradients
    flow: OrthoConvV2's eval branch runs under torch.no_grad(). Each BRO layer is
    checkpointed and pulls its ~1 GB of frequency matrices to the GPU only while
    it runs, so the whole pass fits in 8 GB."""
    from torch.utils.checkpoint import checkpoint
    c = model.conv
    biases = c.bias - c.running_mean

    def bro(x, i):
        z = c.frequency_convolution(x, c.eval_weights[i].to(dev, non_blocking=True), i)
        return c.act(z + biases[i].view(-1, 1, 1))

    x = model.conv_1(model.stem(x.sub(0.5)))
    for i in range(c.depth):
        x = checkpoint(bro, x, i, use_reentrant=False)
    return model.linear(model.neck(x))


def empirical_lower(model, n_points, iters, h=1e-2):
    """A LOWER bound on L that needs no assumption at all: for any x, d,
        L >= ||f(x + d) - f(x)|| / ||d||.
    d is steered towards the worst direction by power iteration on J^T J, with
    J v from a central finite difference and J^T u from one backward pass."""
    out = []
    g = torch.Generator(device=dev).manual_seed(1)
    f = lambda x: features(model, x)
    for k in range(n_points):
        x0 = torch.rand(1, 3, 224, 224, generator=g, device=dev)
        v = torch.randn(x0.shape, generator=g, device=dev)
        v /= v.norm()
        with torch.no_grad():
            y0 = f(x0)
        best = 0.0
        for _ in range(iters):
            with torch.no_grad():
                jv = (f(x0 + h * v) - f(x0 - h * v)) / (2 * h)
                ratio = float((f(x0 + h * v) - y0).norm() / h)
            best = max(best, ratio)
            x = x0.clone().requires_grad_(True)
            (v,) = torch.autograd.grad(f(x), x, jv)
            v = v / v.norm()
        out.append(best)
        log(f"empirical lower bound at random input {k}: {best:.6f}")
    return out


def main(quick=False):
    model, info = load_random() if quick else load()
    log("loaded:", info)
    it = 20 if quick else 3000

    with torch.no_grad():
        model.set_num_lc_iter(20 if quick else 500)        # what test.py does
        official = float(model.sub_lipschitz())
        log(f"official sub_lipschitz (test.py setting): {official:.6f}")
        model.set_num_lc_iter(it)
        official_long = float(model.sub_lipschitz())
        log(f"official sub_lipschitz ({it} it): {official_long:.6f}")

    a = audit(model, iters=it)
    common = a["stem"]["exact"] * a["conv"]["product"] * a["neck"]["exact"] * a["linear"]["exact"]
    rigorous = common * a["conv_1"]["fft_upper"]
    estimate = common * a["conv_1"]["power_iter"]
    emp = empirical_lower(model, n_points=1 if quick else 3, iters=3 if quick else 10)

    res = {"checkpoint": CKPT.name, "load": info,
           "official_500": official, f"official_{it}": official_long,
           "audit": a, "L_estimate": estimate, "L_rigorous_upper": rigorous,
           "empirical_lower": emp, "seconds": time.time() - T0,
           "gpu": torch.cuda.get_device_name() if dev.type == "cuda" else "cpu",
           "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30 if dev.type == "cuda" else 0}
    if not quick:
        OUT.write_text(json.dumps(res, indent=2))

    log(f"L official (test.py)    {official:.6f}")
    log(f"L estimate (audit)      {estimate:.6f}")
    log(f"L rigorous upper bound  {rigorous:.6f}")
    log(f"empirical lower bounds  {[round(e, 4) for e in emp]}")
    log(f"peak GPU {res['peak_gpu_gib']:.2f} GiB" + ("" if quick else f"  -> {OUT}"))


if __name__ == "__main__":
    main(quick="--quick" in sys.argv)
