"""Console encoding helper.

The report tables use a few non-ASCII characters (Greek delta, em dash). On a
Windows console whose code page is not UTF-8 (e.g. cp1251), printing them raises `UnicodeEncodeError` even though the files
on disk are written as UTF-8. Both CLI entry points call `init()` first.
"""

from __future__ import annotations

import sys


def init() -> None:
    """Make stdout/stderr tolerate non-ASCII, whatever the console code page."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):  # pragma: no cover - exotic consoles
                pass
