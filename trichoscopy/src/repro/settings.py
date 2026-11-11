"""Where the data, the outputs and the pretrained weights live -- the one place.

Every location is an environment variable. A variable that is not set is taken
from ``paths.toml`` in the repository root (not in git; copy
``paths.example.toml``), or from the file named by ``DERMA_PATHS_FILE``::

    DERMA_DATA_ROOT       model_data_step{1,2}_anon/{train,test}/<label>/*.jpg   (required)
    DERMA_REPRO_OUT       root of every output        default <DERMA_DATA_ROOT>/repro_out
    BRONET_ROOT           clone of ntuaislab/BRONet   default <DERMA_DATA_ROOT>/pretrained/BRONet
    BRONET_CKPT           BRONet ImageNet checkpoint  default <DERMA_DATA_ROOT>/pretrained/bronet_imagenet
    DERMA_BASELINE_ROOTS  where `repro` runs live, for the McNemar tests of the
                          experiments (os.pathsep-separated)   default <DERMA_REPRO_OUT>
    DERMA_COLOR_REFERENCE reference image of the colour normalisation          (no default)

Order: the environment, then the file, then the default. `apply` writes the
result back into ``os.environ`` so that child processes (DataLoader workers,
subprocesses) see the same values. ``DERMA_BASELINE_ROOTS`` keeps no default in
the environment: it follows ``DERMA_REPRO_OUT`` at the moment it is read.

No third-party imports (TOML via `tomllib`, or `tomli` before Python 3.11):
`repro/__init__` calls `apply` before anything imports torch.
Command line: `python -m repro.where`.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FILE = REPO_ROOT / "paths.toml"

KEYS = ("DERMA_DATA_ROOT", "DERMA_REPRO_OUT", "BRONET_ROOT", "BRONET_CKPT",
        "DERMA_BASELINE_ROOTS", "DERMA_COLOR_REFERENCE")

# Defaults relative to the data root (written to the environment by `apply`).
_DERIVED = {
    "DERMA_REPRO_OUT": ("repro_out",),
    "BRONET_ROOT": ("pretrained", "BRONet"),
    "BRONET_CKPT": ("pretrained", "bronet_imagenet"),
}


class LocationError(RuntimeError):
    pass


def config_file() -> Path:
    return Path(os.environ.get("DERMA_PATHS_FILE", DEFAULT_FILE))


def read_file(path: Path | None = None) -> dict[str, str]:
    """The settings in the file, as environment-variable strings ({} if no file)."""
    path = config_file() if path is None else Path(path)
    if not path.is_file():
        return {}
    try:
        import tomllib                     # Python >= 3.11
    except ModuleNotFoundError:            # Python 3.10: the same parser, as package tomli
        try:
            import tomli as tomllib
        except ModuleNotFoundError:
            raise LocationError(f"reading {path} needs Python >= 3.11 or `pip install tomli`; "
                                "or set the locations as environment variables instead "
                                "(e.g. DERMA_DATA_ROOT) and remove the file") from None
    with open(path, "rb") as handle:
        raw = tomllib.load(handle)
    unknown = sorted(set(raw) - set(KEYS))
    if unknown:
        raise LocationError(f"{path}: unknown setting(s) {unknown}; known: {list(KEYS)}")
    out = {}
    for key, value in raw.items():
        values = value if isinstance(value, list) else [value]
        resolved = [_resolve(str(v), path.parent) for v in values]
        out[key] = os.pathsep.join(resolved)
    return out


def _resolve(value: str, base: Path) -> str:
    p = Path(value).expanduser()
    return str(p if p.is_absolute() else (base / p).resolve())


def apply() -> None:
    """Fill in the unset variables from the file, then the derived defaults.
    Explicit environment settings always win. Never raises for a missing data
    root: that is reported when a location is actually needed."""
    for key, value in read_file().items():
        os.environ.setdefault(key, value)
    data_root = os.environ.get("DERMA_DATA_ROOT")
    if data_root:
        for key, parts in _DERIVED.items():
            os.environ.setdefault(key, str(Path(data_root, *parts)))


def require(key: str) -> str:
    """The value of a location variable, or a clear error saying how to set it."""
    value = os.environ.get(key)
    if value:
        return value
    raise LocationError(
        f"{key} is not set. Set the environment variable, or put it into "
        f"{config_file()} (copy {REPO_ROOT / 'paths.example.toml'}); "
        "see `python -m repro.where`.")
