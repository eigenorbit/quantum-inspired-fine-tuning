"""Shared fixtures. Everything runs on the CPU with a randomly initialised
ResNet152 (random BN statistics, so BN matters); tests needing the dataset or
BRONet's outputs skip themselves.

    python -m pytest experiments/resnet_adapt/tests
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import experiments.resnet_adapt  # noqa: E402,F401
from experiments.bronet_mlp import env  # noqa: E402
from repro.settings import LocationError  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402


def pytest_configure(config):
    config.addinivalue_line("markers", "data: needs the trichoscopy dataset")
    config.addinivalue_line("markers", "slow: minutes on the CPU (full Lipschitz bound)")


def have_data() -> bool:
    try:
        return (env.data_root() / "model_data_step2_anon" / "train").is_dir()
    except LocationError:           # DERMA_DATA_ROOT not set
        return False


needs_data = pytest.mark.skipif(not have_data(), reason="dataset not found (DERMA_DATA_ROOT)")


@pytest.fixture(scope="session")
def net():
    """ResNet152 with random weights and random BN running statistics."""
    from experiments.resnet_adapt import backbone

    torch.manual_seed(0)
    m = backbone.load_resnet152(pretrained=False)
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for mod in m.modules():
            if isinstance(mod, torch.nn.BatchNorm2d):
                mod.running_mean.copy_(0.1 * torch.randn(mod.num_features, generator=g))
                mod.running_var.copy_(0.5 + torch.rand(mod.num_features, generator=g))
                mod.weight.copy_(0.5 + torch.rand(mod.num_features, generator=g))
                mod.bias.copy_(0.1 * torch.randn(mod.num_features, generator=g))
        # small residual branches, as in a trained ResNet: activations stay O(1)
        # through 50 blocks (float16 must not overflow)
        for mod in m.modules():
            if hasattr(mod, "bn3"):
                mod.bn3.weight.mul_(0.1)
    return m.eval()
