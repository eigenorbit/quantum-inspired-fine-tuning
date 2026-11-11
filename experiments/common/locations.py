"""Where the data and the outputs are, read when asked (`repro.settings` rules).

Reading at call time keeps a changed environment (tests) effective at once.
"""

from __future__ import annotations

from pathlib import Path


def _location(key: str) -> Path:
    from repro import settings

    return Path(settings.require(key))


def data_root() -> Path:
    return _location("DERMA_DATA_ROOT")


def repro_out() -> Path:
    return _location("DERMA_REPRO_OUT")


def bronet_root() -> Path:
    return _location("BRONET_ROOT")


def bronet_ckpt_dir() -> Path:
    return _location("BRONET_CKPT")


def color_reference() -> Path:
    """The dataset's colour-normalisation reference photograph (no default)."""
    return _location("DERMA_COLOR_REFERENCE")
