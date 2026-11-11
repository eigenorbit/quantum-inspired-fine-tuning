"""BRONet: loading, the frozen feature map phi, the MLP base, and L_pre.

    python -m experiments.bronet_mlp.bronet lpre     # L_pre.json (~15 min, once)
    python -m experiments.bronet_mlp.bronet info     # load + sanity check only

The released ImageNet checkpoint (Table 1 of the BRONet paper) is

    x in [0,1]^{3x224x224} -> x - 0.5 -> stem (14x14/14 patch conv + MinMax)
      -> conv_1 (6 LiResConv) -> conv (8 BRO blocks, frequency domain)
      -> neck (L2 pool + 9408->2048 linear + MinMax)          = phi(x), cached
      -> linear (LiResMLP: 8 x MinMax(W_i h + b_i), 2048)     = what we adapt
      -> head (LLN)

Traps handled here:

* `model.eval()` materialises every BRO block's frequency matrices (~7.7 GB,
  a plain list attribute). It is called ONCE, on the CPU, before `.to(cuda)`;
  never again afterwards. `place_bro_weights` then decides where they live.
* The eval branch of `OrthoConvV2.forward` runs under `torch.no_grad()`, and
  expects its weights on the input's device. `phi` re-implements that loop
  (identical arithmetic), copying one block at a time when streaming;
  `phi_with_grad` additionally checkpoints each block so gradients flow.
* `lipschitz()` of the power-iteration parts mutates a warm-start buffer, so
  the official constants are taken once, on a freshly loaded model.
* The clone of BRONet is never modified: `torch.distributed.get_world_size`
  is monkeypatched here, not there.
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
import time
from pathlib import Path

from . import env  # noqa: F401  (package __init__ sets paths first)

import torch
import torch.nn.functional as F

LPRE_FILE = "L_pre.json"
MLP_BASE_FILE = "mlp_base.pt"

_models = None


# ---------------------------------------------------------------------------
# importing and building
# ---------------------------------------------------------------------------

def bronet_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=env.bronet_root(),
                             capture_output=True, check=True)
        return out.stdout.decode().strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def import_models():
    """BRONet's `models` package, imported from BRONET_ROOT/bronet."""
    global _models
    if _models is not None:
        return _models
    src = env.bronet_root() / "bronet"
    if not (src / "models" / "__init__.py").is_file():
        raise FileNotFoundError(
            f"BRONet source not found under {src}.\n"
            "Set BRONET_ROOT to a clone of https://github.com/ntuaislab/BRONet "
            f"(commit {env.BRONET_COMMIT}) or run scripts/setup_bronet.(ps1|sh).")
    commit = bronet_commit()
    if commit is not None and not commit.startswith(env.BRONET_COMMIT):
        raise RuntimeError(f"BRONet clone at {env.bronet_root()} is at {commit[:7]}, "
                           f"expected {env.BRONET_COMMIT}")
    # LiResMLP.__init__ asks torch.distributed for the world size; on one GPU
    # the answer is 1 (this only disables its multi-GPU code path).
    torch.distributed.get_world_size = lambda *a, **k: 1
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    import models  # noqa: E402

    _models = models
    return models


def load_config() -> dict:
    import yaml

    with open(env.bronet_ckpt_dir() / "config.yaml", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_state_dict() -> dict:
    """The raw checkpoint dict. weights_only: never unpickle code from a download."""
    path = env.bronet_ckpt_dir() / env.CKPT_NAME
    return torch.load(path, map_location="cpu", weights_only=True)


def load_bronet(device, strategy: str, logger=None, verify_sha: bool = True,
                cache_bro_weights: bool = False):
    """``(model, info)``: the released model, in eval mode, placed per `strategy`.

    eval() happens on the CPU, then parameters and buffers move to `device`;
    the BRO frequency matrices follow only for strategy "gpu".

    With `cache_bro_weights` the BRO matrices are computed one block at a time
    (same CPU arithmetic as `OrthoConvV2.train(False)`), written to
    backbone/eval_weights/ once, and memory-mapped from there: peak RAM ~1.5 GB
    instead of ~9 GB, and no recomputation on later loads.
    """
    log = logger.info if logger else (lambda *_: None)
    t0 = time.perf_counter()
    sha = env.checkpoint_sha256(verify=verify_sha)
    models = import_models()
    cfg = load_config()
    model = models.BRONet(**cfg["model"], **cfg["dataset"])
    ckpt = load_state_dict()
    missing, unexpected = model.load_state_dict(ckpt["backbone"], strict=True)
    last_log = ckpt.get("training_logs", [""])[-1] if ckpt.get("training_logs") else ""
    del ckpt
    if cache_bro_weights:
        log(f"BRONet loaded ({time.perf_counter() - t0:.0f}s); BRO weights from disk cache")
        _eval_except_bro(model)
        model.conv.eval_weights = cached_bro_weights(model, logger)
    else:
        log(f"BRONet loaded ({time.perf_counter() - t0:.0f}s); computing eval weights on CPU")
        model.eval()                       # CPU: materialises the ~7.7 GB BRO matrices
    model.to(device)                       # the list stays on the CPU
    place_bro_weights(model, device, strategy)
    info = {"checkpoint": env.CKPT_NAME, "checkpoint_sha256": sha,
            "bronet_commit": bronet_commit(), "missing": list(missing),
            "unexpected": list(unexpected), "training_log_last": last_log,
            "strategy": strategy, "bro_weights_cached_on_disk": cache_bro_weights,
            "load_seconds": time.perf_counter() - t0}
    log(f"BRONet ready on {device} (BRO weights: {strategy}, "
        f"{time.perf_counter() - t0:.0f}s)")
    return model, info


def _eval_except_bro(model) -> None:
    """model.eval() for everything but the BRO blocks, whose train(False) would
    materialise all 8 layers of frequency matrices at once."""
    model.training = False
    for child in model.children():
        if child is model.conv:
            for m in child.modules():
                m.training = False
        else:
            child.eval()


def bro_weights_dir():
    return env.backbone_dir() / "eval_weights"


@torch.no_grad()
def _bro_layer(c, i: int) -> torch.Tensor:
    """Frequency matrices of BRO block i, exactly as `OrthoConvV2.get_wfft_ortho`."""
    V_i = (c.identity + c.weights[i] * c.gamma[i] * c.scale) if c.quasi_residual \
        else c.gamma[i] * c.weights[i]
    return c.process_vfft(V_i, c.padding_input_sizes[i], c.BRO)


def cached_bro_weights(model, logger=None) -> list:
    """Memory-mapped BRO eval weights; computed (CPU, one block at a time) and
    written on first use. meta.json records sizes and sha256 per file."""
    log = logger.info if logger else (lambda *_: None)
    c = model.conv
    d = bro_weights_dir()
    meta_path = d / "meta.json"
    meta = env.read_json(meta_path) if meta_path.exists() else None
    if meta is not None and meta.get("checkpoint_sha256") != env.CKPT_SHA256:
        raise RuntimeError(f"{d} was built from another checkpoint; move it away")
    if meta is None:
        meta = {"checkpoint_sha256": env.CKPT_SHA256, "bronet_commit": bronet_commit(),
                "computed_on": "cpu", "layers": {}}
    out = []
    for i in range(c.depth):
        name = f"layer_{i}.pt"
        path = d / name
        entry = meta["layers"].get(name)
        if entry is None or not path.exists() or path.stat().st_size != entry["size"]:
            t0 = time.perf_counter()
            W = _bro_layer(c, i).contiguous()
            env.atomic_torch_save(W, path)
            del W
            meta["layers"][name] = {"size": path.stat().st_size, "sha256": env.sha256_file(path),
                                    "seconds": time.perf_counter() - t0}
            env.write_json(meta_path, meta)
            log(f"  BRO block {i}: computed and cached ({meta['layers'][name]['seconds']:.0f}s)")
        out.append(torch.load(path, map_location="cpu", weights_only=True, mmap=True))
    return out


def place_bro_weights(model, device, strategy: str) -> None:
    c = model.conv
    if strategy == "gpu":
        c.eval_weights = [w.to(device) for w in c.eval_weights]
    elif strategy in ("stream", "cpu"):
        c.eval_weights = [w.cpu() for w in c.eval_weights]
    else:
        raise ValueError(f"unknown BRO strategy {strategy!r}")


# ---------------------------------------------------------------------------
# the frozen feature map phi = neck(conv(conv_1(stem(x - 0.5))))
# ---------------------------------------------------------------------------

def _bro_block(model, h, i):
    c = model.conv
    weight = c.eval_weights[i]
    if weight.device != h.device:
        weight = weight.to(h.device, non_blocking=True)
    z = c.frequency_convolution(h, weight, i)
    return c.act(z + (c.bias[i] - c.running_mean[i]).view(-1, 1, 1))


@torch.no_grad()
def phi(model, x: torch.Tensor) -> torch.Tensor:
    """The 2048-d neck output; same arithmetic as BRONet's own eval forward."""
    h = model.conv_1(model.stem(x.sub(0.5)))
    for i in range(model.conv.depth):
        h = _bro_block(model, h, i)
    return model.neck(h)


@torch.no_grad()
def phi_layer_major(model, x: torch.Tensor, device, minibatch: int = 128) -> torch.Tensor:
    """phi for a large chunk of images, one BRO block at a time.

    `phi` pulls all 8 blocks (~7.7 GB) through the GPU once per batch, which is
    I/O-bound when the matrices live on the CPU or on disk. Here the whole chunk
    goes through block i before block i+1 is loaded, so each block is copied
    once per chunk. Same per-image arithmetic as `phi`; the chunk's activations
    (up to 588x32x32 floats = 2.4 MB per image) stay on `device`.
    """
    hs = [model.conv_1(model.stem(x[i:i + minibatch].to(device, non_blocking=True).sub(0.5)))
          for i in range(0, len(x), minibatch)]
    c = model.conv
    for i in range(c.depth):
        weight = c.eval_weights[i]
        if weight.device != device:
            weight = weight.to(device, non_blocking=True)
        bias = (c.bias[i] - c.running_mean[i]).view(-1, 1, 1)
        for k in range(len(hs)):
            hs[k] = c.act(c.frequency_convolution(hs[k], weight, i) + bias)
        del weight
    return torch.cat([model.neck(h).float().cpu() for h in hs])


@torch.no_grad()
def half_bro_weights(model, device) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """(real, imag) float16 copies of every BRO block's frequency matrices on
    `device`: ~3.85 GB instead of 7.7, so they stay resident on an 8 GB GPU.
    Used ONLY for attack gradients; every reported number is recomputed in float32."""
    out = []
    for W in model.conv.eval_weights:
        out.append((W.real.to(torch.float16).to(device), W.imag.to(torch.float16).to(device)))
    return out


def _bro_block_half(model, h, i, Wr, Wi):
    """`_bro_block` with float16 weights: the complex product as 4 real half matmuls."""
    c = model.conv
    x = F.pad(h, (c.kernel_size // 2,) * 4) if c.zero_padding else h
    batches, _, n, _ = x.shape
    if n != c.padding_input_sizes[i]:
        raise ValueError(f"input size {n} != {c.padding_input_sizes[i]}")
    xfft = torch.fft.rfft2(x, norm="ortho").permute(2, 3, 1, 0).reshape(
        n * (n // 2 + 1), c.width, batches)
    xr, xi = xfft.real.to(torch.float16), xfft.imag.to(torch.float16)
    zr = (Wr @ xr - Wi @ xi).float()
    zi = (Wr @ xi + Wi @ xr).float()
    zfft = torch.complex(zr, zi).reshape(n, n // 2 + 1, c.width, batches).permute(3, 2, 0, 1)
    z = torch.fft.irfft2(zfft, norm="ortho")
    return c.act(z + (c.bias[i] - c.running_mean[i]).view(-1, 1, 1))


def phi_with_grad_half(model, x: torch.Tensor, half_weights) -> torch.Tensor:
    """phi with gradients, BRO blocks in float16 (see `half_bro_weights`)."""
    from torch.utils.checkpoint import checkpoint

    h = model.conv_1(model.stem(x.sub(0.5)))
    for i in range(model.conv.depth):
        Wr, Wi = half_weights[i]
        h = checkpoint(_bro_block_half, model, h, i, Wr, Wi, use_reentrant=False)
    return model.neck(h)


def phi_with_grad(model, x: torch.Tensor) -> torch.Tensor:
    """phi with gradients w.r.t. x, for attacks. Each BRO block is checkpointed
    and pulls its weights to the GPU only while it runs (peak ~3.5 GB)."""
    from torch.utils.checkpoint import checkpoint

    h = model.conv_1(model.stem(x.sub(0.5)))
    for i in range(model.conv.depth):
        h = checkpoint(_bro_block, model, h, i, use_reentrant=False)
    return model.neck(h)


# ---------------------------------------------------------------------------
# the MLP base: the frozen W_i, b_i every adapter starts from
# ---------------------------------------------------------------------------

def compute_mlp_base() -> dict:
    """W_i = CholeskyOrth(I + V_i gamma_i s) exactly as `LiResMLP.train(False)`
    computes it, plus b_i and the ImageNet head. Needs only the checkpoint and
    the LiResMLP class -- not the 7.7 GB of BRO matrices."""
    models = import_models()
    from models.dense import LiResMLP
    from models.layers import build_activation

    cfg = load_config()["model"]
    ckpt = load_state_dict()
    sd = ckpt["backbone"]
    mlp = LiResMLP(num_features=cfg["dense_width"], depth=cfg["linear_num"],
                   activation=build_activation(cfg["act_name"], dim=1))
    prefix = "linear."
    mlp.load_state_dict({k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)})
    mlp.eval()
    base = {"weight": mlp.eval_weight.detach().clone().contiguous(),
            "bias": mlp.bias.detach().clone().contiguous(),
            "head_weight": sd["head.weight"].clone(),
            "head_bias": sd["head.bias"].clone(),
            "checkpoint_sha256": env.checkpoint_sha256()}
    del ckpt, sd
    assert models is not None
    return base


def mlp_base_path() -> Path:
    return env.backbone_dir() / MLP_BASE_FILE


def load_mlp_base(path: Path | None = None) -> dict:
    """The cached MLP base, created on first use; checked against the checkpoint."""
    path = Path(path) if path is not None else mlp_base_path()
    if not path.exists():
        if path != mlp_base_path():
            raise FileNotFoundError(path)
        env.atomic_torch_save(compute_mlp_base(), path)
    base = torch.load(path, map_location="cpu", weights_only=True)
    if path == mlp_base_path() and base["checkpoint_sha256"] != env.CKPT_SHA256:
        raise RuntimeError(f"{path} was built from checkpoint "
                           f"{base['checkpoint_sha256'][:12]}, expected {env.CKPT_SHA256[:12]}")
    return base


# ---------------------------------------------------------------------------
# Lipschitz constant of everything before the MLP
# ---------------------------------------------------------------------------

@torch.no_grad()
def sigma_range(mats: torch.Tensor, chunk: int = 64) -> tuple[float, float]:
    """(min, max) singular value over a batch of matrices, float64 on the CPU,
    via the eigenvalues of W^H W (from reference/bronet_lipschitz.py)."""
    lo, hi = math.inf, 0.0
    for c in mats.split(chunk):
        c = c.cpu().to(torch.complex128 if c.is_complex() else torch.float64)
        ev = torch.linalg.eigvalsh(c.mH @ c)
        lo, hi = min(lo, float(ev.min())), max(hi, float(ev.max()))
    return math.sqrt(max(lo, 0.0)), math.sqrt(hi)


@torch.no_grad()
def conv_fft_bound(weight: torch.Tensor, n: int) -> float:
    """Rigorous upper bound on ||conv2d(., weight, padding=1)|| on n x n inputs
    (zero padding = embed, circular conv on (n+2)^2, crop; see the reference)."""
    m = n + 2
    sym = torch.fft.rfft2(weight.double().cpu(), s=(m, m))
    sym = sym.permute(2, 3, 0, 1).reshape(-1, *weight.shape[:2])
    return sigma_range(sym)[1]


def compute_lpre(model, logger=None, num_lc_iter: int = 500, rigorous: bool = True) -> dict:
    """L_pre two ways, on a FRESHLY loaded model (the official power iterations
    warm-start from a buffer they overwrite).

    official   BRONet's own per-part `lipschitz()` with 500 iterations, as its
               test.py; the MLP part is recorded too, for the sanity check
               official L_pre x official L_mlp = sub_lipschitz = 8.1305.
    rigorous   stem exact SVD x conv_1 FFT upper bound x prod sigma_max of every
               BRO frequency block x neck exact SVD (float64, CPU).
    """
    log = logger.info if logger else print
    t0 = time.perf_counter()
    with torch.no_grad():
        model.set_num_lc_iter(num_lc_iter)
        official = {"stem": float(model.stem.lipschitz()),
                    "conv_1": float(model.conv_1.lipschitz()),
                    "conv": float(model.conv.lipschitz()),
                    "neck": float(model.neck.lipschitz()),
                    "linear": float(model.linear.lipschitz())}
    official_pre = official["stem"] * official["conv_1"] * official["conv"] * official["neck"]
    log(f"official parts {official}  L_pre {official_pre:.6f}  "
        f"sub_lipschitz {official_pre * official['linear']:.6f}")
    result = {"L_pre_official": official_pre, "official_parts": official,
              "official_num_lc_iter": num_lc_iter,
              "official_sub_lipschitz": official_pre * official["linear"]}
    if not rigorous:
        return result

    rig: dict = {}
    w = model.stem[0].weight.detach()
    lo, hi = sigma_range(w.reshape(1, w.shape[0], -1))
    rig["stem"] = {"exact": hi, "sigma_min": lo}
    c1 = model.conv_1
    n = c1.init_x.shape[-1]
    per = [conv_fft_bound(W, n) for W in c1.get_weight().detach()]
    rig["conv_1"] = {"fft_upper_per_block": per, "fft_upper": math.prod(per)}
    log(f"rigorous stem {hi:.6f}  conv_1 {rig['conv_1']['fft_upper']:.6f}")
    blocks = []
    for i, Wf in enumerate(model.conv.eval_weights):
        lo, hi = sigma_range(Wf)
        blocks.append({"sigma_min": lo, "sigma_max": hi})
        log(f"rigorous BRO[{i}] sigma in [{lo:.9f}, {hi:.9f}]  "
            f"({time.perf_counter() - t0:.0f}s)")
    rig["conv"] = {"per_block": blocks, "product": math.prod(b["sigma_max"] for b in blocks)}
    _, _, W = model.neck.get_weight()
    lo, hi = sigma_range(W.detach().T.unsqueeze(0))
    rig["neck"] = {"exact": hi, "sigma_min": lo}
    rigorous_pre = (rig["stem"]["exact"] * rig["conv_1"]["fft_upper"]
                    * rig["conv"]["product"] * rig["neck"]["exact"])
    log(f"L_pre official {official_pre:.6f}  rigorous {rigorous_pre:.6f}")
    return {**result, "L_pre_rigorous": rigorous_pre, "rigorous_parts": rig,
            "seconds": time.perf_counter() - t0}


def lpre_path() -> Path:
    return env.backbone_dir() / LPRE_FILE


def load_lpre() -> dict:
    path = lpre_path()
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; run  python -m experiments.bronet_mlp.bronet lpre")
    data = env.read_json(path)
    if data.get("checkpoint_sha256") != env.CKPT_SHA256:
        raise RuntimeError(f"{path} does not belong to checkpoint {env.CKPT_SHA256[:12]}")
    if data.get("tf32") is not False:
        raise RuntimeError(
            f"{path} was computed with TF32 convolutions (or predates that flag); the "
            "package runs in full float32. Recompute:\n"
            "  python -m experiments.bronet_mlp.bronet lpre --cache-bro-weights --force")
    return data


# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    from repro import console

    console.init()
    p = argparse.ArgumentParser(description="BRONet loading and L_pre")
    p.add_argument("command", choices=("lpre", "info"))
    p.add_argument("--device", default="auto")
    p.add_argument("--bro-weights", default="auto")
    p.add_argument("--force", action="store_true", help="recompute L_pre.json")
    p.add_argument("--cache-bro-weights", action="store_true",
                   help="compute BRO eval weights once to backbone/eval_weights/ (~7.7 GB "
                        "on disk) and memory-map them: low RAM, fast later loads")
    a = p.parse_args(argv)

    logger = env.setup_logging(env.backbone_dir() / f"{a.command}.log")
    device = env.resolve_device(a.device)
    strategy = env.resolve_bro_strategy(device, a.bro_weights)
    if a.command == "lpre" and lpre_path().exists() and not a.force:
        logger.info(f"{lpre_path()} exists; --force to recompute")
        return 0
    model, info = load_bronet(device, strategy, logger,
                              cache_bro_weights=a.cache_bro_weights)
    logger.info(f"load info: {info}")
    base = load_mlp_base()
    logger.info(f"MLP base: weight {tuple(base['weight'].shape)}, bias {tuple(base['bias'].shape)}")
    if a.command == "lpre":
        res = compute_lpre(model, logger)
        res.update(tf32=bool(torch.backends.cudnn.allow_tf32 or torch.backends.cuda.matmul.allow_tf32),
                   checkpoint_sha256=info["checkpoint_sha256"],
                   bronet_commit=info["bronet_commit"],
                   code_sha256=env.code_sha256(), git=env.git_info(),
                   device=env.device_info(device), strategy=strategy)
        env.write_json(lpre_path(), res)
        audit = Path(__file__).resolve().parent / "reference" / "lipschitz.json"
        if audit.exists():
            env.atomic_write_bytes(env.backbone_dir() / "bronet_audit.json", audit.read_bytes())
        logger.info(f"-> {lpre_path()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
