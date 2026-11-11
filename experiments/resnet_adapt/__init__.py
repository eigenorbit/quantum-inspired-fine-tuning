"""ResNet152 control experiments for the BRONet adaptation study.

First experiment (``full``): ResNet152 (IMAGENET1K_V2) frozen up to and
including ``layer4.1``; trained are the last bottleneck ``layer4.2`` and a new
LLN head (as in BRONet), with the tuned protocol of ``experiments/bronet_mlp``
(validation split shared with BRONet, lr per mode, early stopping, 3 seeds).
How to run: ``README.md`` in this folder.

Importing ``experiments.bronet_mlp`` first gives this package the same
environment: path defaults (``DERMA_DATA_ROOT``, ``DERMA_REPRO_OUT``), ``repro``
on ``sys.path`` with deterministic cuBLAS, and TF32 disabled.
"""

import experiments.bronet_mlp  # noqa: F401  (env defaults, repro, TF32 off)
