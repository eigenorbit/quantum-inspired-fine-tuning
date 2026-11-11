"""Code shared by the experiment packages (bronet_mlp, resnet_adapt).

Modules here import nothing heavy at import time; the environment (locations,
deterministic cuBLAS, TF32 off) is set up by importing ``experiments.bronet_mlp``.
"""
