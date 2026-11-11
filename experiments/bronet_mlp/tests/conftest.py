"""Shared fixtures. Synthetic tests run anywhere (CPU is enough); tests that need
the dataset or the BRONet checkpoint skip themselves when those are absent.

Run from the repository root:

    python -m pytest experiments/bronet_mlp/tests            # everything available
    python -m pytest experiments/bronet_mlp/tests -m "not slow"
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import experiments.bronet_mlp  # noqa: E402,F401  (paths, env defaults)
from experiments.bronet_mlp import env  # noqa: E402
from repro.settings import LocationError  # noqa: E402

import torch  # noqa: E402


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: needs the BRONet checkpoint (minutes)")
    config.addinivalue_line("markers", "data: needs the trichoscopy dataset")


def _exists(location, *parts, is_dir=False) -> bool:
    """Whether `location()/parts` exists; False when the location is not set."""
    try:
        path = location().joinpath(*parts)
    except LocationError:
        return False
    return path.is_dir() if is_dir else path.is_file()


def have_data() -> bool:
    return _exists(env.data_root, "model_data_step2_anon", "train", is_dir=True)


def have_bronet_src() -> bool:
    return _exists(env.bronet_root, "bronet", "models", "__init__.py")


def have_bronet() -> bool:
    return have_bronet_src() and _exists(env.bronet_ckpt_dir, env.CKPT_NAME)


needs_data = pytest.mark.skipif(not have_data(), reason="dataset not found (DERMA_DATA_ROOT)")
needs_bronet = pytest.mark.skipif(not have_bronet(), reason="BRONet clone/checkpoint not found")
needs_bronet_src = pytest.mark.skipif(not have_bronet_src(),
                                      reason="BRONet source not found (BRONET_ROOT)")


def random_base(depth=2, width=64, seed=0):
    """Pretrained-looking MLP base: near-orthogonal matrices, small biases."""
    g = torch.Generator().manual_seed(seed)
    q = torch.linalg.qr(torch.randn(depth, width, width, generator=g))[0]
    w = q * (1 + 0.05 * torch.randn(depth, width, 1, generator=g))
    b = 0.05 * torch.randn(depth, width, generator=g)
    return w.float(), b.float()


def synthetic_inputs(tmp_path, depth=2, width=64, n_train=384, n_test=96, seed=0):
    """A RunInputs on random, learnable features (3 classes, step-2 names)."""
    from experiments.bronet_mlp.train import RunInputs

    g = torch.Generator().manual_seed(seed + 1)
    w, b = random_base(depth, width, seed)
    centers = torch.randn(3, width, generator=g) * 1.5

    def draw(n):
        y = torch.arange(n) % 3
        return centers[y] + torch.randn(n, width, generator=g), y

    xtr, ytr = draw(n_train)
    xte, yte = draw(n_test)
    base_path = Path(tmp_path) / "mlp_base.pt"
    torch.save({"weight": w, "bias": b, "checkpoint_sha256": "synthetic"}, base_path)
    return RunInputs(
        train_x=xtr, train_y=ytr, test_x=xte, test_y=yte, base_weight=w, base_bias=b,
        L_pre_official=8.0, L_pre_rigorous=8.4,
        provenance={"bronet_ckpt_sha256": "synthetic", "features_index_sha256": "feat",
                    "data_manifest_sha256": "manifest", "mlp_base": str(base_path),
                    "mlp_base_sha256": env.sha256_bytes(w.numpy().tobytes() + b.numpy().tobytes())},
        test_rows=[{"image": f"img{i}.jpg", "patient": "P0", "source_stem": f"img{i}"}
                   for i in range(n_test)])


@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"
