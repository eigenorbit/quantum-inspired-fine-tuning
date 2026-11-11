"""Helpers shared by the tests (a uniquely named module: bronet_mlp's tests
import their own `conftest`)."""

import numpy as np
import torch


def synthetic_inputs(block, n_train=96, n_val=48, n_test=48, seed=0):
    """RunInputs on random, learnable 2048x7x7 'features' (3 classes = step-2)."""
    from experiments.resnet_adapt.train import RunInputs

    g = torch.Generator().manual_seed(seed + 1)
    centers = torch.randn(3, 2048, 1, 1, generator=g).abs()
    n = n_train + n_val + n_test
    y = torch.arange(n) % 3
    x = (centers[y] + 0.5 * torch.rand(n, 2048, 7, 7, generator=g)).numpy().astype(np.float32)
    tr = np.arange(n_train)
    va = np.arange(n_train, n_train + n_val)
    te = np.arange(n_train + n_val, n)
    return RunInputs(
        feats=x, train_idx=tr, train_y=y[:n_train], test_idx=te, test_y=y[n_train + n_val:],
        val_idx=va, val_y=y[n_train:n_train + n_val], base_block=block,
        provenance={"weights_sha256": "synthetic", "features_id": "feat",
                    "data_manifest_sha256": "manifest"},
        test_rows=[{"image": f"img{i}.jpg", "patient": "P0", "source_stem": f"img{i}"}
                   for i in range(n_test)])
