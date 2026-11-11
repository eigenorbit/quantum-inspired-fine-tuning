"""Device selection and a description of it for the provenance records."""

from __future__ import annotations

import platform


def resolve_device(spec: str = "auto"):
    import torch

    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(spec)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available")
    return device


def _cuda_index(device) -> int:
    import torch

    return device.index if device.index is not None else torch.cuda.current_device()


def device_info(device) -> dict:
    import torch

    info = {"device": str(device), "torch": torch.__version__,
            "cuda": torch.version.cuda, "python": platform.python_version(),
            "platform": platform.platform()}
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info(_cuda_index(device))
        info.update(gpu=torch.cuda.get_device_name(_cuda_index(device)),
                    gpu_total_gib=round(total / 2**30, 2),
                    gpu_free_gib_at_start=round(free / 2**30, 2))
    return info
